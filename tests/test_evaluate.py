"""Small regression suite for the supported three-path GEARS workflow."""

import csv
import json
from pathlib import Path
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

from evaluate.data import load_gears
from evaluate.metrics import METRICS, _expression_rows, compute_six, gears_config


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
    prediction = tmp_path / "outputs" / "GEARS" / "test_res.pkl"
    prediction.parent.mkdir(parents=True)
    with prediction.open("wb") as handle:
        # PKL ctrl rows must not replace the H5AD's real control pool.
        pickle.dump({"pred": np.vstack([pred, np.zeros((1, 40), dtype=np.float32)]),
                     "truth": np.vstack([truth, np.zeros((1, 40), dtype=np.float32)]),
                     "pert_cat": np.append(labels, "ctrl")}, handle)
    root = tmp_path / "data"
    source = root / "norman" / "perturb_processed.h5ad"
    source.parent.mkdir(parents=True)
    # Full-data perturbed rows deliberately differ from the paired test truth.
    ad.AnnData(np.vstack([ctrl, np.full((3, 40), 8, dtype=np.float32)]),
               obs=pd.DataFrame({"condition": ["ctrl"] * 48 + list(np.unique(labels))},
                                index=[str(i) for i in range(51)]),
               var=pd.DataFrame({"gene_name": genes}, index=[f"ENSG{i}" for i in range(40)])
               ).write_h5ad(source)
    return prediction, root, source, genes, labels, pred, truth, ctrl


@pytest.mark.parametrize("sparse_control", [False, True])
def test_six_match_upstream_lognorm(case, sparse_control):
    prediction, root, source, genes, labels, pred, truth, ctrl = case
    if sparse_control:
        ref = ad.read_h5ad(source)
        ref.X = sparse.csr_matrix(ref.X)
        ref.write_h5ad(source)
    pred_ad, real_ad, targets, meta = load_gears(prediction, root)
    assert meta["negative_predictions_clipped_to_zero"] == 1
    assert meta["n_control_cells"] == 48
    # Independent expected matrices verify use of PKL truth, real controls and gene_name.
    obs = pd.DataFrame({"condition": np.append(labels, ["ctrl"] * 48)},
                       index=[str(i) for i in range(192)])
    expected_pred = ad.AnnData(np.vstack([np.maximum(pred, 0), ctrl]), obs=obs.copy(),
                              var=pd.DataFrame(index=genes))
    expected_real = ad.AnnData(np.vstack([truth, ctrl]), obs=obs.copy(),
                              var=pd.DataFrame(index=genes))
    if sparse_control:
        expected_pred.X = sparse.csr_matrix(expected_pred.X)
        expected_real.X = sparse.csr_matrix(expected_real.X)
    cfg = gears_config(targets)
    reference = aggregate_metrics(compute_metrics(expected_pred, expected_real, config=cfg))
    expected = dict(reference.select("metric", "mean").iter_rows())
    actual, _ = compute_six(pred_ad, real_ad, targets)
    assert set(actual) == set(METRICS)
    for name in METRICS:
        assert actual[name] is not None
        assert actual[name] == pytest.approx(expected[name], rel=1e-11, abs=1e-12)


def test_three_path_cli_with_combination(case, tmp_path):
    prediction, root, source, *_ = case
    with prediction.open("rb") as handle:
        payload = pickle.load(handle)
    payload["pert_cat"][payload["pert_cat"] == "G1+ctrl"] = "G1+G2"
    with prediction.open("wb") as handle:
        pickle.dump(payload, handle)
    ref = ad.read_h5ad(source)
    ref.obs["condition"] = ref.obs.condition.astype(str).replace({"G1+ctrl": "G1+G2"})
    ref.write_h5ad(source)
    output = tmp_path / "results"
    command = Path(sys.executable).parent / "gears-eval"
    result = subprocess.run([str(command), "--pred-path", str(prediction),
                             "--truth-path", str(root), "--output-dir", str(output)],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text())
    assert set(summary["metrics"]) == set(METRICS)
    assert all(value is not None for value in summary["metrics"].values())
    assert summary["target_genes"]["G1+G2"] == ["G1", "G2"]
    assert summary["config"]["input_type"] == "lognorm"
    assert summary["config"]["de"]["backend"] == "pdex"
    assert summary["negative_predictions_clipped_to_zero"] == 1
    assert "avg_score" not in summary
    assert sorted(p.name for p in output.iterdir()) == ["summary.csv", "summary.json"]
    with (output / "summary.csv").open() as handle:
        assert next(csv.reader(handle)) == list(METRICS)
    pred_ad, real_ad, targets, _ = load_gears(prediction, root)
    own_rows = np.flatnonzero(pred_ad.obs.condition.to_numpy() == "G1+G2")
    pred_ad.X[np.ix_(own_rows, [1, 2])] += 1.0
    changed, _ = compute_six(pred_ad, real_ad, targets)
    # Both targeted genes must be absent from PDS and the DE fold-change error.
    for name in ("pds_cosine", "de_wilcoxon_lfc_nmae"):
        assert changed[name] == pytest.approx(summary["metrics"][name], abs=1e-12)


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


@pytest.mark.parametrize("problem", ["nan", "shape", "missing_ctrl", "counts_ctrl"])
def test_invalid_input_is_rejected(case, problem):
    prediction, root, source, *_ = case
    if problem in {"missing_ctrl", "counts_ctrl"}:
        ref = ad.read_h5ad(source)
        if problem == "missing_ctrl":
            ref.obs["condition"] = ref.obs.condition.astype(str).replace({"ctrl": "other"})
        else:
            ref.X[:48] = np.rint(np.expm1(ref.X[:48]))
        ref.write_h5ad(source)
    else:
        with prediction.open("rb") as handle:
            payload = pickle.load(handle)
        if problem == "nan":
            payload["pred"][0, 0] = np.nan
        else:
            payload["pred"] = payload["pred"][:, :-1]
        with prediction.open("wb") as handle:
            pickle.dump(payload, handle)
    with pytest.raises(ValueError):
        load_gears(prediction, root)
