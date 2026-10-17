"""Four-path evaluation of generated counts against independent real controls."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from evaluate.metrics import METRICS


def counts_adata(matrix, labels, genes, *, key="target_gene"):
    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(matrix.shape[0])])
    if labels is not None:
        obs[key] = labels
    obs["cell_type"] = "H1"
    return ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))


@pytest.fixture
def h5ad_case(tmp_path):
    rng = np.random.default_rng(23)
    genes = [f"G{i}" for i in range(40)]
    pred_labels = np.repeat(["G0", "G1", "G2"], 48)
    real_labels = np.repeat(["G0", "G1", "G2"], 64)
    effects = rng.integers(0, 24, (3, 40))
    pred = (rng.poisson(6, (144, 40)) + np.repeat(effects, 48, axis=0)).astype(np.int32)
    truth = (rng.poisson(6, (192, 40)) + np.repeat(effects, 64, axis=0)).astype(np.int32)
    ctrl = rng.poisson(6, (56, 40)).astype(np.int32)
    paths = {name: tmp_path / f"{name}.h5ad" for name in ("pred", "truth", "control")}
    counts_adata(sparse.csr_matrix(pred), pred_labels, genes).write_h5ad(paths["pred"])
    # Include an unused true perturbation and a different gene order.
    truth_file = counts_adata(np.vstack([truth, np.ones((2, 40), dtype=np.int32)]),
                              np.append(real_labels, ["G3", "G3"]), genes)
    truth_file[:, ::-1].copy().write_h5ad(paths["truth"])
    # A full experimental file can supply controls; its perturbed rows are ignored.
    counts_adata(sparse.csr_matrix(np.vstack([ctrl, np.ones((1, 40), dtype=np.int32)])),
                 ["non-targeting"] * 56 + ["G4"], genes).write_h5ad(paths["control"])
    return paths, genes, pred_labels, real_labels, pred, truth, ctrl


def test_h5ad_aligns_genes_selects_truth_and_uses_independent_control(h5ad_case):
    from evaluate.h5ad import load_h5ad

    paths, genes, pred_labels, real_labels, pred, truth, ctrl = h5ad_case
    predicted, real, targets, meta = load_h5ad(paths["pred"], paths["truth"], paths["control"])
    np.testing.assert_array_equal(predicted.X.toarray(), np.vstack([pred, ctrl]))
    np.testing.assert_array_equal(real.X.toarray(), np.vstack([truth, ctrl]))
    assert list(predicted.var_names) == list(real.var_names) == genes
    assert predicted.obs.condition.tolist() == [*pred_labels, *(["ctrl"] * len(ctrl))]
    assert real.obs.condition.tolist() == [*real_labels, *(["ctrl"] * len(ctrl))]
    assert targets == {"G0": ["G0"], "G1": ["G1"], "G2": ["G2"]}
    assert meta["n_predicted_cells"] == 144
    assert meta["n_truth_cells"] == 192
    assert meta["n_control_cells"] == 56
    assert meta["control_data"] == str(paths["control"])
    assert meta["skipped_truth_perturbations"] == ["G3"]


@pytest.mark.parametrize("dense", [False, True])
def test_counts_six_match_upstream_on_independent_expected_matrices(h5ad_case, dense):
    from evaluate.h5ad import load_h5ad
    from evaluate.metrics import compute_counts_six

    paths, genes, pred_labels, real_labels, pred, truth, ctrl = h5ad_case
    if dense:
        for path in paths.values():
            matrix = ad.read_h5ad(path)
            if sparse.issparse(matrix.X):
                matrix.X = matrix.X.toarray()
            matrix.write_h5ad(path)
    predicted, real, targets, _ = load_h5ad(paths["pred"], paths["truth"], paths["control"])
    actual, cfg = compute_counts_six(predicted, real, targets)
    expected_pred = counts_adata(sparse.csr_matrix(np.vstack([pred, ctrl])),
                                 [*pred_labels, *(["ctrl"] * len(ctrl))], genes, key="condition")
    expected_real = counts_adata(sparse.csr_matrix(np.vstack([truth, ctrl])),
                                 [*real_labels, *(["ctrl"] * len(ctrl))], genes, key="condition")
    preset = EvalConfig.from_preset("vcc2026")
    expected_cfg = replace(preset, pert_col="condition", control="ctrl", device="cpu",
                           de=replace(preset.de, backend="pdex"),
                           target_gene_map={f"G{i}": f"G{i}" for i in range(3)})
    expected = dict(aggregate_metrics(compute_metrics(expected_pred, expected_real,
                                                     config=expected_cfg),
                                      metrics=list(METRICS)).select("metric", "mean").iter_rows())
    assert cfg == expected_cfg
    assert set(actual) == set(METRICS)
    for name in METRICS:
        assert actual[name] is not None
        assert actual[name] == pytest.approx(expected[name], rel=1e-11, abs=1e-12)


@pytest.mark.parametrize("embedded_controls", [False, True])
def test_only_the_explicit_control_file_supplies_controls(h5ad_case, embedded_controls):
    from evaluate.h5ad import load_h5ad

    paths, genes, *_rest, ctrl = h5ad_case
    for name in ("pred", "truth"):
        matrix = ad.read_h5ad(paths[name])
        if embedded_controls:
            wrong_ctrl = counts_adata(np.full((3, 40), 999, dtype=np.int32),
                                      ["non-targeting"] * 3, genes)
            matrix = ad.concat([matrix, wrong_ctrl], index_unique="-")
        matrix.write_h5ad(paths[name])
    # A label-free control-only file is also a supported four-path input.
    counts_adata(ctrl, None, genes).write_h5ad(paths["control"])
    pred, real, _, meta = load_h5ad(paths["pred"], paths["truth"], paths["control"])
    for matrix in (pred, real):
        values = matrix.X.toarray() if sparse.issparse(matrix.X) else matrix.X
        np.testing.assert_array_equal(values[-len(ctrl):], ctrl)
        assert sum(matrix.obs.condition == "ctrl") == len(ctrl)
    assert meta["ignored_prediction_controls"] == (3 if embedded_controls else 0)


def test_h5ad_gene_annotations_and_condition_alias(h5ad_case):
    from evaluate.h5ad import load_h5ad

    paths, genes, *_ = h5ad_case
    for name in ("truth", "control"):
        matrix = ad.read_h5ad(paths[name])
        matrix.var["gene_name"] = matrix.var_names.to_numpy()
        matrix.var_names = [f"ENSG_{i}" for i in range(matrix.n_vars)]
        matrix.obs.rename(columns={"target_gene": "condition"}, inplace=True)
        matrix.obs["condition"] = matrix.obs.condition.astype(str).replace({"non-targeting": "ctrl"})
        matrix.write_h5ad(paths[name])
    predicted, real, _, meta = load_h5ad(paths["pred"], paths["truth"], paths["control"])
    assert list(predicted.var_names) == list(real.var_names) == genes
    assert meta["gene_order"]["truth"] == "gene_name"


@pytest.mark.parametrize("problem, match", [
    ("missing_gene", "gene"), ("duplicate_gene", "unique"),
    ("missing_perturbation", "perturbation"), ("missing_labels", "obs"),
    ("conflicting_labels", "conflict"), ("missing_control", "control"),
    ("nan_label", "label"), ("different_context", "cell_type"),
    ("multiple_contexts", "cell_type"), ("negative", "negative"),
    ("fractional", "counts"), ("nan", "NaN"), ("large_counts", "1000000"),
])
def test_invalid_h5ad_is_rejected(h5ad_case, problem, match):
    from evaluate.h5ad import load_h5ad

    paths, *_ = h5ad_case
    name = "control" if problem == "missing_control" else "truth"
    matrix = ad.read_h5ad(paths[name])
    if problem == "missing_gene":
        matrix = matrix[:, 1:].copy()
    elif problem == "duplicate_gene":
        matrix.var_names = ["duplicate"] * matrix.n_vars
    elif problem == "missing_perturbation":
        matrix.obs["target_gene"] = matrix.obs.target_gene.astype(str).replace({"G1": "G9"})
    elif problem == "missing_labels":
        del matrix.obs["target_gene"]
    elif problem == "conflicting_labels":
        matrix.obs["condition"] = "G9"
    elif problem == "missing_control":
        matrix.obs["target_gene"] = "G9"
    elif problem == "nan_label":
        matrix.obs.loc[matrix.obs_names[0], "target_gene"] = None
    elif problem == "different_context":
        matrix.obs["cell_type"] = "K562"
    elif problem == "multiple_contexts":
        matrix.obs["cell_type"] = matrix.obs.cell_type.astype(str)
        matrix.obs.loc[matrix.obs_names[0], "cell_type"] = "K562"
    else:
        matrix.X = np.asarray(matrix.X, dtype=float)
        matrix.X[0, 4] = {"negative": -1, "fractional": 1.25, "nan": np.nan,
                           "large_counts": 1000001}[problem]
    matrix.write_h5ad(paths[name])
    if problem == "duplicate_gene":
        with pytest.warns(UserWarning, match="not unique"), pytest.raises(ValueError, match=match):
            load_h5ad(paths["pred"], paths["truth"], paths["control"])
    else:
        with pytest.raises(ValueError, match=match):
            load_h5ad(paths["pred"], paths["truth"], paths["control"])


def test_unmeasured_targets_cannot_bypass_resolution_with_alias_labels(h5ad_case):
    from evaluate.h5ad import load_h5ad

    paths, *_ = h5ad_case
    for name in ("pred", "truth"):
        matrix = ad.read_h5ad(paths[name])
        matrix.obs["target_gene"] = matrix.obs.target_gene.astype(str).replace(
            {"G0": "U", "G1": "U+ctrl", "G2": "U+non-targeting"})
        matrix.write_h5ad(paths[name])
    with pytest.raises(ValueError, match="No perturbation target"):
        load_h5ad(paths["pred"], paths["truth"], paths["control"])


def test_four_path_cli(h5ad_case, tmp_path):
    paths, *_ = h5ad_case
    output = tmp_path / "scores"
    script = Path(__file__).resolve().parents[1] / "run.py"
    result = subprocess.run([sys.executable, str(script), "--pred-path", str(paths["pred"]),
                             "--truth-path", str(paths["truth"]), "--control-path",
                             str(paths["control"]), "--output-dir", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["metrics"]) == set(METRICS)
    assert all(value is not None for value in summary["metrics"].values())
    assert summary["protocol"] == "h5ad-counts-vcc2026-metrics"
    assert summary["config"]["input_type"] == "counts"
    assert summary["config"]["control_source"] == "real"
    assert summary["config"]["de"]["backend"] == "pdex"
    assert summary["control_data"] == str(paths["control"])
    assert summary["n_truth_cells"] == 192
    assert summary["n_predicted_cells"] == 144
    assert sorted(p.name for p in output.iterdir()) == ["summary.csv", "summary.json"]
    assert (output / "summary.csv").read_text().splitlines()[0] == ",".join(METRICS)


def test_h5ad_cli_requires_independent_control(h5ad_case, tmp_path):
    paths, *_ = h5ad_case
    script = Path(__file__).resolve().parents[1] / "run.py"
    result = subprocess.run([sys.executable, str(script), "--pred-path", str(paths["pred"]),
                             "--truth-path", str(paths["truth"]),
                             "--output-dir", str(tmp_path / "missing_control")],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "--control-path" in result.stderr
    assert not (tmp_path / "missing_control").exists()


def test_gears_rejects_an_unused_control_argument(tmp_path, capsys):
    from evaluate.run import main

    with pytest.raises(SystemExit) as result:
        main(["--pred-path", str(tmp_path / "test_res.pkl"),
              "--truth-path", str(tmp_path / "truth.h5ad"),
              "--control-path", str(tmp_path / "control.h5ad"),
              "--output-dir", str(tmp_path / "results")])
    assert result.value.code == 2
    assert "--control-path is supported only with H5AD" in capsys.readouterr().err
