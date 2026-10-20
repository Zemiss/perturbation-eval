"""Four-path H5AD evaluation and upstream six-metric parity."""

import csv
import json
import pickle
import subprocess
import sys

import anndata as ad
from cell_eval2 import aggregate_metrics, compute_metrics
from cell_eval2.moments import GroupMoments
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from baseline.gears.export import export_results
from evaluate.data import attach, fill_normalized, open_source
from evaluate.metrics import METRICS, _expression_rows, compute_six, metric_config
from evaluate.run import evaluate


@pytest.fixture
def case(tmp_path):
    rng = np.random.default_rng(20)
    genes = [f"G{i}" for i in range(40)]
    labels = np.repeat(["G0+ctrl", "G1+ctrl", "G2+ctrl"], 48)
    def lognorm(counts):
        return np.log1p(counts / counts.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
    ctrl = lognorm(rng.poisson(6, (48, 40)))
    effects = rng.integers(0, 18, (3, 40))
    truth = lognorm(rng.poisson(6, (144, 40)) + np.repeat(effects, 48, axis=0))
    pred = np.maximum(truth + rng.normal(0, 0.08, truth.shape), 0).astype(np.float32)
    pred[0, 5] = -0.1
    payload = {"pred": np.vstack([pred, np.zeros((1, 40), dtype=np.float32)]),
               "truth": np.vstack([truth, np.zeros((1, 40), dtype=np.float32)]),
               "pert_cat": np.append(labels, "ctrl")}
    # These unrelated full-data perturbations must not replace test truth.
    source = ad.AnnData(np.vstack([ctrl, np.full((3, 40), 8, dtype=np.float32)]),
                        obs=pd.DataFrame({"condition": ["ctrl"] * 48 + list(np.unique(labels))},
                                         index=[str(i) for i in range(51)]),
                        var=pd.DataFrame({"gene_name": genes}, index=[f"ENSG{i}" for i in range(40)]))
    return payload, source, genes, labels, pred, truth, ctrl


def load_pair(paths):
    sources = {role: open_source(path, "pred" if role == "prediction" else role)
               for role, path in paths.items()}
    try:
        genes = sources["truth"].genes
        arrays = {}
        for role, source in sources.items():
            arrays[role] = np.empty((len(source.rows), len(genes)), dtype=np.float32)
            fill_normalized(source, genes, arrays[role])
        ctrl = arrays["control"]
        pred = attach(np.vstack([arrays["prediction"], ctrl]),
                      np.append(sources["prediction"].labels, ["ctrl"] * len(ctrl)), genes)
        truth = attach(np.vstack([arrays["truth"], ctrl]),
                       np.append(sources["truth"].labels, ["ctrl"] * len(ctrl)), genes)
        return pred, truth
    finally:
        for source in sources.values():
            source.adata.file.close()


@pytest.mark.parametrize("sparse_control", [False, True])
def test_h5ad_export_and_six_match_upstream(case, tmp_path, sparse_control):
    payload, source, genes, labels, pred, truth, ctrl = case
    if sparse_control:
        source.X = sparse.csr_matrix(source.X)
    paths = export_results(payload, source, tmp_path / "output", tmp_path / "reference")
    exported = ad.read_h5ad(paths["prediction"])
    assert exported.shape == (144, 40)
    assert list(exported.var_names) == genes
    assert exported.obs.target_gene.tolist() == labels.tolist()
    assert exported.uns["evaluation"]["negative_values_clipped"] == 1
    np.testing.assert_array_equal(exported.X, np.maximum(pred, 0))
    np.testing.assert_array_equal(ad.read_h5ad(paths["truth"]).X, truth)
    controls = ad.read_h5ad(paths["control"]).X
    np.testing.assert_array_equal(controls.toarray() if sparse.issparse(controls) else controls, ctrl)
    assert sorted(p.name for p in (tmp_path / "output").iterdir()) == ["prediction.h5ad"]
    pred_ad, real_ad = load_pair(paths)
    # Independent normalization verifies scale conversion and use of real controls.
    def normalize(x):
        linear = np.expm1(np.asarray(x, dtype=np.float64))
        return np.log1p(linear / linear.sum(axis=1, keepdims=True) * 10000).astype(np.float32)
    np.testing.assert_array_equal(pred_ad.X, np.vstack([normalize(np.maximum(pred, 0)), normalize(ctrl)]))
    np.testing.assert_array_equal(real_ad.X, np.vstack([normalize(truth), normalize(ctrl)]))
    if sparse_control:
        pred_ad.X, real_ad.X = sparse.csr_matrix(pred_ad.X), sparse.csr_matrix(real_ad.X)
    targets = {label: [label.split("+")[0]] for label in np.unique(labels)}
    reference = aggregate_metrics(compute_metrics(pred_ad, real_ad, config=metric_config(targets)))
    expected = dict(reference.select("metric", "mean").iter_rows())
    actual, _ = compute_six(pred_ad, real_ad, targets)
    assert set(actual) == set(METRICS)
    for name in METRICS:
        assert actual[name] is not None
        assert actual[name] == pytest.approx(expected[name], rel=1e-11, abs=1e-12)


def test_four_path_cli_with_combination(case, tmp_path):
    payload, source, *_ = case
    payload["pert_cat"][payload["pert_cat"] == "G1+ctrl"] = "G1+G2"
    paths = export_results(payload, source, tmp_path / "output", tmp_path / "reference")
    output = tmp_path / "results"
    command = [sys.executable, "-m", "evaluate", "--pred-path", paths["prediction"],
               "--truth-path", paths["truth"], "--control-path", paths["control"],
               "--output-dir", str(output)]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["metrics"]) == set(METRICS)
    assert all(value is not None for value in summary["metrics"].values())
    assert summary["target_genes"]["G1+G2"] == ["G1", "G2"]
    assert summary["config"]["input_type"] == "lognorm"
    assert summary["config"]["de"]["backend"] == "pdex"
    assert "avg_score" not in summary
    assert sorted(p.name for p in output.iterdir()) == ["summary.csv", "summary.json"]
    with (output / "summary.csv").open() as handle:
        assert next(csv.reader(handle)) == list(METRICS)
    pred_ad, real_ad = load_pair(paths)
    own_rows = np.flatnonzero(pred_ad.obs.condition.to_numpy() == "G1+G2")
    pred_ad.X[np.ix_(own_rows, [1, 2])] += 1.0
    changed, _ = compute_six(pred_ad, real_ad, summary["target_genes"])
    for name in ("pds_cosine", "de_wilcoxon_lfc_nmae"):
        assert changed[name] == pytest.approx(summary["metrics"][name], abs=1e-12)


def test_legacy_gears_conversion_reorders_declared_genes(case, tmp_path):
    payload, source, genes, _, pred, *_ = case
    order = np.arange(39, -1, -1)
    payload["pred"], payload["truth"] = payload["pred"][:, order], payload["truth"][:, order]
    payload["gene_names"] = np.asarray(genes)[order]
    pkl, processed = tmp_path / "old.pkl", tmp_path / "processed.h5ad"
    with pkl.open("wb") as handle:
        pickle.dump(payload, handle)
    source.write_h5ad(processed)
    result = subprocess.run([sys.executable, "-m", "baseline.gears.export", "--pred-path", str(pkl),
                             "--data-path", str(processed), "--output-dir", str(tmp_path / "output"),
                             "--reference-dir", str(tmp_path / "reference")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    np.testing.assert_array_equal(ad.read_h5ad(tmp_path / "output/prediction.h5ad").X, np.maximum(pred, 0))


@pytest.mark.parametrize("problem", ["nan", "missing_gene", "missing_ctrl", "wrong_scale", "missing_perturbation", "context"])
def test_invalid_h5ad_is_rejected(case, tmp_path, problem):
    payload, source, *_ = case
    paths = export_results(payload, source, tmp_path / "output", tmp_path / "reference")
    role = "control" if problem == "missing_ctrl" else "prediction"
    data = ad.read_h5ad(paths[role])
    if problem == "nan":
        data.X[0, 0] = np.nan
    elif problem == "missing_gene":
        data = data[:, :-1].copy()
    elif problem == "missing_ctrl":
        data.obs["target_gene"] = "other"
    elif problem == "wrong_scale":
        data.uns["evaluation"]["expression_scale"] = "counts"
    elif problem == "context":
        data.obs["cell_type"] = ["H1"] * (data.n_obs - 1) + ["other"]
    else:
        data = data[data.obs.target_gene != "G2+ctrl"].copy()
    data.write_h5ad(paths[role])
    with pytest.raises(ValueError):
        evaluate(paths["prediction"], paths["truth"], paths["control"], tmp_path / "results")
    assert not (tmp_path / "results/summary.json").exists()


def test_control_path_required_and_no_pkl_in_evaluator(case, tmp_path):
    help_result = subprocess.run([sys.executable, "-m", "evaluate", "--help"], capture_output=True, text=True)
    assert help_result.returncode == 0
    assert all(flag in help_result.stdout for flag in ["--pred-path", "--truth-path", "--control-path", "--output-dir"])
    with pytest.raises(ValueError, match="H5AD"):
        open_source(tmp_path / "test_res.pkl", "pred")
    missing = subprocess.run([sys.executable, "-m", "evaluate", "--pred-path", "a.h5ad",
                              "--truth-path", "b.h5ad", "--output-dir", str(tmp_path)], capture_output=True, text=True)
    assert missing.returncode == 2 and "--control-path" in missing.stderr


def test_expression_excludes_every_component():
    # Zero sampling noise isolates the multi-target mask and per-row gene divisor.
    perts = np.asarray(["G0+G1", "G2+ctrl", "ctrl"])
    pm = np.asarray([[7., 8., 3., 2.], [2., 4., 9., 3.], [1., 1., 1., 1.]])
    rm = np.asarray([[2., 3., 5., 4.], [3., 2., 4., 6.], [1., 1., 1., 1.]])
    def bulk(means):
        return perts, means, GroupMoments(perts, np.full(3, 32),
                                          32 * (means ** 2).sum(axis=1))
    rows = _expression_rows(bulk(pm), bulk(rm),
                            {"G0+G1": ["G0", "G1"], "G2+ctrl": ["G2"]},
                            ["G0", "G1", "G2", "G3"])
    values = {(row["metric"], row["perturbation"]): row["value"] for row in rows}
    assert values["expr_mse_unbiased_capped", "G0+G1"] == pytest.approx((4 + 4) / 2)
    assert values["expr_distance_unbiased", "G0+G1"] == pytest.approx((16 + 9) / 2)
    assert values["expr_mse_unbiased_capped", "G2+ctrl"] == pytest.approx((1 + 4 + 9) / 3)
