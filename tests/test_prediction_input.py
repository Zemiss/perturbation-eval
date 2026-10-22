"""Prediction-only model interface: real references, metadata and native metric parity."""

from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import anndata as ad
from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from evaluate.data import canonical_label, load_gene_map
from evaluate.metrics import METRICS
from evaluate.run import evaluate


@pytest.fixture
def inputs(tmp_path):
    rng = np.random.default_rng(47)
    genes = [f"G{i}" for i in range(40)]
    effects = rng.integers(0, 12, (3, 40))
    pred = rng.poisson(5, (96, 40)) + np.repeat(effects, 32, axis=0)
    truth = rng.poisson(5, (144, 40)) + np.repeat(effects, 48, axis=0)
    control = rng.poisson(5, (36, 40))
    pred_labels = np.repeat(["G0", "G1", "G2"], 32)
    truth_labels = np.repeat(["G0", "G1", "G2"], 48)

    def save(name, matrix, labels):
        data = ad.AnnData(sparse.csr_matrix(matrix, dtype=np.int32),
                          obs=pd.DataFrame({"target_gene": labels, "cell_type": "H1"},
                                           index=[f"{name}_{i}" for i in range(len(labels))]),
                          var=pd.DataFrame(index=genes))
        path = tmp_path / f"{name}.h5ad"
        data.write_h5ad(path)
        return path

    prediction = save("prediction", pred, pred_labels)
    real = save("real", np.vstack([truth, control]), np.append(truth_labels, ["non-targeting"] * 36))
    controls = save("controls", control, ["non-targeting"] * 36)
    return prediction, real, controls, pred, truth, control, genes, pred_labels, truth_labels


def expected_native(inputs):
    _, _, _, pred, truth, control, genes, pred_labels, truth_labels = inputs

    def matrix(values, labels):
        return ad.AnnData(sparse.csr_matrix(np.vstack([values, control])),
                          obs=pd.DataFrame({"condition": np.append(labels, ["ctrl"] * 36)},
                                           index=[str(i) for i in range(len(labels) + 36)]),
                          var=pd.DataFrame(index=genes))

    preset = EvalConfig.from_preset("vcc2026")
    cfg = replace(preset, pert_col="condition", control="ctrl", device="cpu", num_threads=1,
                  de=replace(preset.de, backend="pdex"),
                  target_gene_map={f"G{i}": f"G{i}" for i in range(3)})
    tidy = compute_metrics(matrix(pred, pred_labels), matrix(truth, truth_labels), config=cfg)
    return dict(aggregate_metrics(tidy, metrics=list(METRICS)).select("metric", "mean").iter_rows())


def test_counts_cli_reads_raw_lingshu_output_and_separate_real_controls(inputs, tmp_path):
    prediction, truth, control, *_ = inputs
    # The native result needs no evaluation metadata and can contain unused control rows.
    pred = ad.read_h5ad(prediction)
    unused = pred[:2].copy()
    unused.obs["target_gene"] = "non-targeting"
    unused.X = sparse.csr_matrix(np.full(unused.shape, 999, dtype=np.int32))
    ad.concat([pred, unused], index_unique="-").write_h5ad(prediction)
    # Gene order in the real file is independent of the model's output.
    real = ad.read_h5ad(truth)
    real[:, ::-1].copy().write_h5ad(truth)
    output = tmp_path / "results"
    result = subprocess.run([sys.executable, "-m", "evaluate", "--pred-path", str(prediction),
                             "--truth-path", str(truth), "--control-path", str(control), "--output-dir", str(output),
                             "--num-threads", "1", "--protocol", "vcc2026"], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert summary["protocol"] == "h5ad-vcc2026-counts-v1"
    assert summary["control_source"] == "separate_h5ad"
    assert summary["control_data"] == str(control)
    assert summary["n_control_cells"] == 36
    assert summary["n_pred_cells"] == 96
    assert summary["n_real_test_cells"] == 144
    assert summary["ignored_prediction_controls"] == 2
    assert summary["ignored_truth_controls"] == 36
    assert summary["config"]["input_type"] == "counts"
    assert summary["config"]["filter"]["filter_gene_min_cpm_cell"] == 5.0
    assert summary["config"]["num_threads"] == 1
    assert set(summary["metrics"]) == set(METRICS)
    expected = expected_native(inputs)
    for name in METRICS:
        assert summary["metrics"][name] == pytest.approx(expected[name], rel=1e-9, abs=1e-10)


def test_separate_control_overrides_embedded_real_controls(inputs, tmp_path, monkeypatch):
    import evaluate.metrics as metric_module

    prediction, truth, control, *_ = inputs
    compute = metric_module.compute_counts_six

    def check_control_rows(pred, real, targets, **kwargs):
        # Inspect the actual metric inputs: the same independently loaded real cells appear on both sides.
        np.testing.assert_array_equal(pred.X[-36:], inputs[5])
        np.testing.assert_array_equal(real.X[-36:], inputs[5])
        assert pred.obs.condition.tolist()[-36:] == real.obs.condition.tolist()[-36:] == ["ctrl"] * 36
        return compute(pred, real, targets, **kwargs)

    monkeypatch.setattr(metric_module, "compute_counts_six", check_control_rows)
    direct = evaluate(prediction, truth, control, tmp_path / "direct", num_threads=1, protocol="vcc2026")
    real = ad.read_h5ad(truth)
    real.X = real.X.toarray()
    real.X[-36:] = 999
    real.write_h5ad(truth)
    explicit = evaluate(prediction, truth, control, tmp_path / "explicit", num_threads=1, protocol="vcc2026")
    assert explicit["control_source"] == "separate_h5ad"
    assert explicit["metrics"] == direct["metrics"]
    assert explicit["expression_sha256"] == direct["expression_sha256"]


def test_custom_labels_and_independent_expression_layers(inputs, tmp_path):
    prediction, truth, control, *_ = inputs
    expected = evaluate(prediction, truth, control, tmp_path / "default", num_threads=1)
    for path in [prediction, truth, control]:
        data = ad.read_h5ad(path)
        labels = data.obs.target_gene.astype(str).replace({"non-targeting": "PBS"})
        data.obs = pd.DataFrame({"perturbation": labels}, index=data.obs_names)
        data.layers["raw_counts"] = data.X.copy()
        data.X = data.X.astype(np.float32)
        data.X.data = np.log1p(data.X.data)
        data.write_h5ad(path)
    actual = evaluate(prediction, truth, control, tmp_path / "custom", num_threads=1,
                      pert_col="perturbation", control_label="PBS",
                      pred_layer="raw_counts", truth_layer="raw_counts", control_layer="raw_counts")
    assert actual["metrics"] == expected["metrics"]
    assert actual["input_layers"] == {"pred": "raw_counts", "truth": "raw_counts", "control": "raw_counts"}


def test_gene_map_and_explicit_panel_avoid_a_model_adapter(inputs, tmp_path):
    prediction, truth, control, _, _, _, genes, *_ = inputs
    mapping = tmp_path / "gene_id_name.tsv"
    mapping.write_text("".join(f"ENSG{i}\t{gene}\n" for i, gene in enumerate(genes)))
    real = ad.read_h5ad(truth)
    real.var_names = [f"ENSG{i}" for i in range(40)]
    real.obs["target_gene"] = real.obs.target_gene.astype(str).replace(
        {f"G{i}": f"ENSG{i}" for i in range(3)})
    real.write_h5ad(truth)
    pred = ad.read_h5ad(prediction)
    pred.obs["target_gene"] = pred.obs.target_gene.astype(str) + "+ctrl"
    pred[:, :38].copy().write_h5ad(prediction)
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(genes[:38]) + "\n")
    with pytest.raises(ValueError, match="missing 2 reference genes"):
        evaluate(prediction, truth, control, tmp_path / "invalid", gene_map=mapping)
    summary = evaluate(prediction, truth, control, tmp_path / "mapped", gene_map=mapping,
                       gene_panel=panel, num_threads=1)
    assert summary["gene_names"] == genes[:38]
    assert summary["perturbations"] == ["G0", "G1", "G2"]
    assert all(value is not None for value in summary["metrics"].values())


def test_lognorm_override_requires_explicit_protocol(inputs, tmp_path):
    prediction, truth, control, *_ = inputs
    pred = ad.read_h5ad(prediction)
    values = pred.X.toarray().astype(np.float64)
    pred.X = np.log1p(values / values.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
    pred.write_h5ad(prediction)
    with pytest.raises(ValueError, match="--protocol lognorm"):
        evaluate(prediction, truth, control, tmp_path / "bad", pred_input_type="lognorm", protocol="vcc2026")
    summary = evaluate(prediction, truth, control, tmp_path / "lognorm", protocol="lognorm",
                       pred_input_type="lognorm", num_threads=1)
    assert summary["protocol"] == "h5ad-reference-panel-lognorm-v1"
    assert summary["input_types"] == {"pred": "lognorm", "truth": "counts", "control": "counts"}
    assert summary["n_control_cells"] == 36
    assert set(summary["metrics"]) == set(METRICS)
    assert all(value is not None for value in summary["metrics"].values())


def test_dataset_training_controls_never_fall_back_to_embedded_controls(inputs, tmp_path):
    prediction, truth, control, *_ = inputs
    pred, controls = ad.read_h5ad(prediction), ad.read_h5ad(control)
    ad.concat([pred, controls], index_unique="-").write_h5ad(prediction)
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    # Both prediction and truth contain controls, but the selected training source has none.
    pred.write_h5ad(root / "train/adata_Training.h5ad")
    with pytest.raises(ValueError, match="--control-path"):
        evaluate(prediction, truth, output_dir=tmp_path / "no_controls", data_root=root)
    assert not (tmp_path / "no_controls/summary.json").exists()


def test_counts_preset_rejects_multi_gene_instead_of_ignoring_a_target(inputs, tmp_path):
    prediction, truth, control, *_ = inputs
    for path in [prediction, truth]:
        data = ad.read_h5ad(path)
        data.obs["target_gene"] = data.obs.target_gene.astype(str).replace({"G0": "G0+G3"})
        data.write_h5ad(path)
    with pytest.raises(ValueError, match="single-gene"):
        evaluate(prediction, truth, control, tmp_path / "multi", protocol="vcc2026")


def test_dataset_defaults_accept_both_native_output_formats(inputs, tmp_path, monkeypatch):
    import evaluate.metrics as metric_module
    import evaluate.run as runner

    prediction, truth, control, _, _, control_values, genes, *_ = inputs
    root = tmp_path / "dataset"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir()
    mapping = tmp_path / "gene_id_name.csv"
    mapping.write_text("".join(f"ENSG{i:011d}\t{gene}\n" for i, gene in enumerate(genes)))
    panel = tmp_path / "vcc_vocab.json"
    panel.write_text(json.dumps({'target_gene_to_token_id_info': {'gene_names_ordered': genes}}))
    monkeypatch.setattr(runner, "DEFAULT_DATA_ROOT", root)
    monkeypatch.setattr(runner, "DEFAULT_GENE_MAP", mapping)
    monkeypatch.setattr(runner, "DEFAULT_GENE_PANEL", panel)

    real = ad.read_h5ad(truth)
    real.X = real.X.toarray()
    real.X[-36:] = 999  # Test controls must not replace the chosen training control pool.
    train = ad.concat([ad.read_h5ad(control), real[:2]], index_unique="-")
    for data, path in [(real, root / "test/adata_Test.h5ad"),
                       (train, root / "train/adata_Training.h5ad")]:
        data.var_names = [f"ENSG{i:011d}.2" for i in range(40)]
        data.obs["target_gene"] = data.obs.target_gene.astype(str).replace(
            {gene: f"ENSG{i:011d}.2" for i, gene in enumerate(genes)})
        data.write_h5ad(path)

    normalized_control = np.log1p(control_values / control_values.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
    compute = metric_module.compute_six

    def check_controls(pred, real, targets, **kwargs):
        np.testing.assert_array_equal(pred.X[-36:], normalized_control)
        np.testing.assert_array_equal(real.X[-36:], normalized_control)
        return compute(pred, real, targets, **kwargs)

    monkeypatch.setattr(metric_module, "compute_six", check_controls)
    summaries = []
    for model in ["lingshucell", "gears"]:
        path = tmp_path / "outputs" / model / "prediction.h5ad"
        path.parent.mkdir(parents=True)
        native = ad.read_h5ad(prediction)
        if model == "gears":
            values = native.X.toarray().astype(np.float64)
            native.X = np.log1p(values / values.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
            native.var['gene_name'] = genes
            native.uns['evaluation'] = {'expression_scale': 'lognorm', 'producer': 'GEARS VCC',
                                        'negative_values_clipped': 0}
        native.write_h5ad(path)
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        output = tmp_path / "results" / model
        # The complete user interface: prediction and output paths, with no prepared references.
        summary = runner.evaluate(path, output_dir=output, num_threads=1)
        summaries.append(summary)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        assert summary['control_source'] == 'dataset_train'
        assert summary['control_data'] == str(root / 'train/adata_Training.h5ad')
        assert summary['truth_data'] == str(root / 'test/adata_Test.h5ad')
        assert summary['n_control_cells'] == 36
        assert summary['n_real_test_cells'] == 144
        assert summary['gene_names'] == genes
        assert summary['protocol'] == 'h5ad-reference-panel-lognorm-v1'
        assert summary['input_types']['pred'] == ('counts' if model == 'lingshucell' else 'lognorm')
        assert set(summary['metrics']) == set(METRICS)
        assert all(value is not None for value in summary['metrics'].values())
        assert sorted(p.name for p in output.iterdir()) == ['summary.csv', 'summary.json']
        assert sorted(p.name for p in path.parent.iterdir()) == ['prediction.h5ad']
    for role in ['truth', 'control']:
        assert summaries[0]['expression_sha256'][role] == summaries[1]['expression_sha256'][role]


def test_full_dataset_without_labels_is_not_assumed_to_be_all_control(inputs, tmp_path):
    prediction, truth, control, *_ = inputs
    root = tmp_path / 'dataset'
    (root / 'train').mkdir(parents=True)
    data = ad.read_h5ad(control)
    del data.obs['target_gene']
    data.write_h5ad(root / 'train/adata_Training.h5ad')
    with pytest.raises(ValueError, match='obs.target_gene'):
        evaluate(prediction, truth, output_dir=tmp_path / 'invalid', data_root=root)


def test_gene_map_rejects_ambiguous_mapping(tmp_path):
    path = tmp_path / "ambiguous.tsv"
    path.write_text("ENSG0\tA\nENSG0\tB\n")
    with pytest.raises(ValueError, match="Conflicting"):
        load_gene_map(path)


@pytest.mark.parametrize("label, expected", [("ctrl+NT", "ctrl"), ("G0+ctrl", "G0"),
                                             ("G2+G0", "G0+G2")])
def test_equivalent_perturbation_labels(label, expected):
    assert canonical_label(label) == expected
