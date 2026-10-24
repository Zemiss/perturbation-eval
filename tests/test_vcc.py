"""Small end-to-end checks of fixed references across VCC model outputs."""

import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from baseline.prepare_reference import prepare_reference
from evaluate.data import fill_normalized, load_gene_map, open_source, reference_genes
from evaluate.metrics import METRICS
from evaluate.run import evaluate


@pytest.fixture
def vcc_case(tmp_path):
    rng = np.random.default_rng(20)
    genes = [f"G{i}" for i in range(40)]
    labels = np.repeat(["G0", "G1", "G2"], 48)
    ctrl = rng.poisson(6, (48, 40)).astype(np.int32)
    effects = rng.integers(0, 18, (3, 40))
    truth = (rng.poisson(6, (144, 40)) + np.repeat(effects, 48, axis=0)).astype(np.int32)
    mapping = tmp_path / "genes.tsv"
    mapping.write_text("".join(f"ENSG{i}\t{gene}\n" for i, gene in enumerate(genes)))

    def save(name, matrix, labels, gene_names, scale=None):
        obs = pd.DataFrame({"cell_type": "H1", "target_gene": labels},
                           index=[f"{name}_{i}" for i in range(len(labels))])
        data = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=gene_names))
        if scale:
            data.uns["vcc"] = {"expression_scale": scale}
        path = tmp_path / f"{name}.h5ad"
        data.write_h5ad(path)
        return path

    truth_path = save("truth", truth, [f"ENSG{i}" for i in np.repeat([0, 1, 2], 48)],
                      [f"ENSG{i}" for i in range(40)])
    control_path = save("control", ctrl, ["non-targeting"] * 48, genes)
    lingshu = save("lingshu", truth, labels, genes, "counts")
    # Chreode uses a different order, a smaller gene panel, and native lognorm.
    native_lognorm = np.log1p(truth / truth.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
    order = np.arange(37, -1, -1)
    chreode = save("chreode", native_lognorm[:, order], labels, [genes[i] for i in order], "lognorm")
    return lingshu, chreode, truth_path, control_path, mapping, truth, genes


def test_four_path_cli_compares_models_with_one_prepared_reference(vcc_case, tmp_path):
    lingshu, chreode, raw_truth, raw_control, mapping, *_ = vcc_case
    prepared = tmp_path / "reference"
    result = subprocess.run([sys.executable, "-m", "baseline.prepare_reference",
                             "--pred-path", str(chreode), "--truth-path", str(raw_truth),
                             "--control-path", str(raw_control), "--output-dir", str(prepared),
                             "--gene-map", str(mapping)], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summaries = []
    for name, prediction in [("lingshucell", lingshu), ("chreode", chreode)]:
        output = tmp_path / "results" / name
        command = [sys.executable, "-m", "evaluate", "--pred-path", str(prediction),
                   "--truth-path", str(prepared / "truth.h5ad"),
                   "--control-path", str(prepared / "control.h5ad"), "--output-dir", str(output),
                   "--protocol", "lognorm"]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((output / "summary.json").read_text())
        summaries.append(summary)
        assert set(summary["metrics"]) == set(METRICS)
        assert all(value is not None for value in summary["metrics"].values())
        assert sorted(path.name for path in output.iterdir()) == ["summary.csv", "summary.json"]
        assert summary["config"]["input_type"] == "lognorm"
        assert summary["metrics"]["pds_cosine"] == pytest.approx(1.)
        assert summary["metrics"]["de_wilcoxon_lfc_nmae"] < 1e-5
        assert summary["gene_names"] == [f"G{i}" for i in range(38)]
        assert summary["n_control_cells"] == 48
    assert summaries[0]["gene_panel_sha256"] == summaries[1]["gene_panel_sha256"]
    for role in ["truth", "control"]:
        assert summaries[0]["normalized_expression_sha256"][role] == summaries[1]["normalized_expression_sha256"][role]
    assert summaries[0]["input_types"]["pred"] == "counts"
    assert summaries[1]["input_types"]["pred"] == "lognorm"


def test_native_lognorm_is_normalized_on_reference_genes(vcc_case):
    _, prediction, _, _, _, counts, genes = vcc_case
    source = open_source(prediction, "pred")
    try:
        normalized = np.empty((144, 38), dtype=np.float32)
        fill_normalized(source, genes[:38], normalized)
    finally:
        source.adata.file.close()
    expected = np.log1p(counts[:, :38] / counts[:, :38].sum(axis=1, keepdims=True) * 10000)
    np.testing.assert_allclose(normalized, expected, rtol=2e-6, atol=2e-6)


def test_declared_counts_layer_and_unpaired_populations(vcc_case, tmp_path):
    lingshu, chreode, raw_truth, raw_control, mapping, *_ = vcc_case
    for path in [raw_truth, raw_control]:
        data = ad.read_h5ad(path)
        counts = np.asarray(data.X).copy()
        data.layers["counts"] = counts
        data.X = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
        data.uns["evaluation"] = {"expression_scale": "counts", "layer": "counts"}
        data.write_h5ad(path)
    truth, control = prepare_reference(chreode, raw_truth, raw_control, tmp_path / "reference", mapping)
    # Direct evaluator layer declarations are stored in H5AD, requiring no fifth flag.
    data = ad.read_h5ad(truth)
    counts = data.X.copy()
    data.layers["counts"] = counts
    data.X = np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
    data.uns["evaluation"].update({"expression_scale": "counts", "layer": "counts"})
    data.write_h5ad(truth)
    # Population metrics support different numbers of predicted and real cells.
    prediction = ad.read_h5ad(lingshu)
    prediction = prediction[:-1].copy()
    prediction.write_h5ad(lingshu)
    summary = evaluate(lingshu, truth, control, tmp_path / "results", protocol="lognorm")
    assert summary["n_pred_cells"] == 143 and summary["n_real_test_cells"] == 144
    assert summary["input_layers"]["truth"] == "counts"
    assert summary["metrics"]["pds_cosine"] == pytest.approx(1.)


def test_fixed_reference_never_silently_loses_genes(vcc_case, tmp_path):
    lingshu, chreode, raw_truth, raw_control, mapping, *_ = vcc_case
    truth, control = prepare_reference(lingshu, raw_truth, raw_control, tmp_path / "reference", mapping)
    with pytest.raises(ValueError, match="missing 2 reference genes"):
        evaluate(chreode, truth, control, tmp_path / "results")
    assert not (tmp_path / "results/summary.json").exists()


def test_native_chreode_h5ad_defines_one_common_panel_for_all_models(vcc_case, tmp_path):
    lingshu, chreode, raw_truth, raw_control, mapping, *_ = vcc_case
    summaries = []
    for name, prediction in [('lingshucell', lingshu), ('chreode', chreode)]:
        output = tmp_path / 'common_results' / name
        result = subprocess.run([sys.executable, '-m', 'evaluate', '--pred-path', str(prediction),
                                 '--truth-path', str(raw_truth), '--control-path', str(raw_control),
                                 '--gene-map', str(mapping), '--gene-panel', str(chreode),
                                 '--output-dir', str(output), '--num-threads', '1'],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        summary = json.loads((output / 'summary.json').read_text())
        summaries.append(summary)
        assert summary['gene_panel'] == str(chreode)
        assert summary['gene_names'] == [f'G{i}' for i in range(37, -1, -1)]
        assert summary['n_genes'] == 38
        assert summary['n_real_test_cells'] == 144
        assert summary['n_control_cells'] == 48
        assert set(summary['metrics']) == set(METRICS)
        assert all(value is not None for value in summary['metrics'].values())
        assert summary['metrics']['pds_cosine'] == pytest.approx(1.)
        assert summary['metrics']['de_wilcoxon_lfc_nmae'] < 1e-5
        assert sorted(path.name for path in output.iterdir()) == ['summary.csv', 'summary.json']
    assert summaries[0]['gene_panel_sha256'] == summaries[1]['gene_panel_sha256']
    for role in ['truth', 'control']:
        assert summaries[0]['expression_sha256'][role] == summaries[1]['expression_sha256'][role]
    # The files continue to hold native outputs on their original gene axes.
    assert ad.read_h5ad(lingshu).n_vars == 40
    assert ad.read_h5ad(chreode).n_vars == 38


def test_h5ad_gene_panel_uses_only_gene_annotations(vcc_case, tmp_path):
    _, _, raw_truth, _, mapping, *_ = vcc_case
    path = tmp_path / 'gene_only.h5ad'
    # A panel H5AD does not need X, cells, perturbation labels or expression metadata.
    ad.AnnData(var=pd.DataFrame({'gene_name': ['G2', 'G0', 'G1']},
                                index=['ENSG2', 'ENSG0', 'ENSG1'])).write_h5ad(path)
    source = open_source(raw_truth, 'truth', gene_map=load_gene_map(mapping))
    try:
        assert reference_genes(source, path) == ['G2', 'G0', 'G1']
        missing = tmp_path / 'missing_gene.h5ad'
        ad.AnnData(var=pd.DataFrame(index=['G0', 'ABSENT'])).write_h5ad(missing)
        with pytest.raises(ValueError, match='Truth H5AD is missing 1 panel genes'):
            reference_genes(source, missing)
        duplicate = tmp_path / 'duplicate_gene.h5ad'
        ad.AnnData(var=pd.DataFrame(index=['G0', 'g0'])).write_h5ad(duplicate)
        with pytest.raises(ValueError, match='unique'):
            reference_genes(source, duplicate)
    finally:
        source.adata.file.close()


def test_control_only_h5ad_needs_no_label_column(vcc_case, tmp_path):
    _, chreode, raw_truth, raw_control, mapping, *_ = vcc_case
    truth, control = prepare_reference(chreode, raw_truth, raw_control, tmp_path / "reference", mapping)
    data = ad.read_h5ad(control)
    del data.obs["target_gene"]
    data.write_h5ad(control)
    summary = evaluate(chreode, truth, control, tmp_path / "results", protocol="lognorm")
    assert summary["n_control_cells"] == 48
