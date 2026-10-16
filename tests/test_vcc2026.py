"""Strict counts inputs and parity with the original cell-eval2 dispatcher."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from evaluate.vcc2026 import run_vcc2026_evaluation


SIX = {
    "pds_cosine", "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
}


def counts_pair(root, matrix_kind="dense"):
    """Different per-gene responses and within-condition variance on both sides."""
    root.mkdir(parents=True, exist_ok=True)
    names = ["G1", "G2", "G3"] + [f"g{i}" for i in range(37)]
    labels = np.repeat(["non-targeting", "G1", "G2", "G3"], 32)
    effects = np.random.default_rng(10).integers(0, 13, (3, 40))
    paths = []
    for name, seed in [("pred", 12), ("truth", 13)]:
        x = np.random.default_rng(seed).poisson(6, (128, 40)).astype(np.float32)
        for i, effect in enumerate(effects, 1):
            x[i * 32:(i + 1) * 32] += effect
        obj = ad.AnnData(sparse.csr_matrix(x) if matrix_kind == "sparse" else x,
                         obs=pd.DataFrame({"target": labels}, index=[f"c{i}" for i in range(128)]),
                         var=pd.DataFrame(index=names))
        path = root / f"{name}.h5ad"
        obj.write_h5ad(path)
        paths.append(path)
    return paths


@pytest.mark.parametrize("matrix_kind", ["dense", "sparse"])
@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_six_metrics_match_original_dispatch_and_aggregation(tmp_path, matrix_kind, device):
    from cell_eval2 import EvalConfig, aggregate_metrics, compute_metrics

    pred, truth = counts_pair(tmp_path / "inputs", matrix_kind)
    cfg = EvalConfig.from_preset("vcc2026")
    if device == "cpu":
        cfg = replace(cfg, device="cpu", de=replace(cfg.de, backend="pdex"))
    expected = aggregate_metrics(compute_metrics(pred, truth, config=cfg))
    reference = {row["metric"]: row["mean"] for row in expected.to_dicts() if row["metric"] in SIX}
    result = run_vcc2026_evaluation(pred, truth, tmp_path / "scores", device=device)
    assert set(result["metrics"]) == SIX
    for name in SIX:
        assert result["metrics"][name] == pytest.approx(reference[name], abs=1e-12, rel=1e-12)
    if device == "cpu":
        golden = json.loads((Path(__file__).parent / "fixtures/vcc2026_reference.json").read_text())
        for name in SIX:
            assert result["metrics"][name] == pytest.approx(golden["cases"][matrix_kind][name], abs=1e-12, rel=1e-12)
    assert result["config"]["input_type"] == "counts"
    assert result["config"] == asdict(cfg)
    assert result["config"]["autodetect_input_type"] is False
    assert result["config"]["allow_fractional_counts"] is False
    assert result["config"]["discrimination"]["exclusion_scope"] == "panel"
    assert result["config"]["de"]["fdr_scope"] == "per_pert"
    assert result["score_type"] == "raw_metrics"
    assert "avg_score" not in result
    saved = json.loads((tmp_path / "scores" / "summary.json").read_text())
    assert saved["metrics"] == result["metrics"]
    assert set(pd.read_csv(tmp_path / "scores" / "summary.csv").columns) == SIX
    assert {"results.csv", "agg_results.csv", "run_meta.json", "run_params.yaml"}.issubset(
        {path.name for path in (tmp_path / "scores").iterdir()})
    # The upstream derived expression metric has no per-perturbation value.
    assert "expr_mse_unbiased_capped_norm" not in set(pd.read_csv(tmp_path / "scores" / "results.csv")["metric"])


@pytest.mark.parametrize("bad,reason", [
    (-1.0, "non-negative"), (0.5, "integer"),
    (np.nan, "finite"), (np.inf, "finite"), (1_000_001, "1000000"),
])
@pytest.mark.parametrize("side", [0, 1])
def test_invalid_counts_are_rejected_before_reports(tmp_path, bad, reason, side):
    paths = counts_pair(tmp_path / "inputs")
    obj = ad.read_h5ad(paths[side])
    obj.X[0, 0] = bad
    obj.write_h5ad(paths[side])
    with pytest.raises(ValueError, match=reason):
        run_vcc2026_evaluation(*paths, tmp_path / "scores")
    assert not (tmp_path / "scores" / "summary.json").exists()


@pytest.mark.parametrize("fault,reason", [
    ("gene_order", "gene names/order"), ("duplicate_gene", "unique"),
    ("missing_control", "control"), ("label_mismatch", "perturbation sets"),
    ("missing_label", "labels"),
])
@pytest.mark.filterwarnings("ignore:Variable names are not unique:UserWarning")
def test_structural_mismatch_is_rejected(tmp_path, fault, reason):
    pred, truth = counts_pair(tmp_path / "inputs")
    obj = ad.read_h5ad(pred)
    if fault == "gene_order":
        obj = obj[:, ::-1].copy()
    elif fault == "duplicate_gene":
        obj.var_names = ["G1"] * obj.n_vars
    elif fault == "missing_control":
        obj = obj[obj.obs.target != "non-targeting"].copy()
    elif fault == "label_mismatch":
        obj.obs["target"] = obj.obs["target"].astype(str)
        obj.obs.loc[obj.obs.target == "G1", "target"] = "different"
    else:
        obj.obs.iloc[0, 0] = None
    obj.write_h5ad(pred)
    with pytest.raises(ValueError, match=reason):
        run_vcc2026_evaluation(pred, truth, tmp_path / "scores")


def test_normalized_gears_pickle_is_not_a_counts_input(tmp_path):
    prediction = tmp_path / "test_res.pkl"
    prediction.write_bytes(b"not a counts h5ad")
    with pytest.raises(ValueError, match="counts.*h5ad|H5AD"):
        run_vcc2026_evaluation(prediction, tmp_path, tmp_path / "out")


def test_directory_requires_one_unambiguous_h5ad(tmp_path):
    counts_pair(tmp_path / "multiple")
    with pytest.raises(ValueError, match="exactly one|ambiguous"):
        run_vcc2026_evaluation(tmp_path / "multiple", tmp_path / "multiple", tmp_path / "out")


@pytest.mark.parametrize("module", ["evaluate", "evaluate.run", "evaluate.evaluate_gears"])
def test_default_entrypoints_advertise_counts_and_require_three_paths(tmp_path, module):
    process = subprocess.run([sys.executable, "-m", module, "--help"], cwd=tmp_path,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "VCC2026" in process.stdout
    assert "--pred-path" in process.stdout and "--truth-path" in process.stdout
    assert "--output-dir" in process.stdout
    missing = subprocess.run([sys.executable, "-m", module, "--pred-path", "p", "--truth-path", "t"],
                             cwd=tmp_path, capture_output=True, text=True)
    assert missing.returncode == 2 and "--output-dir" in missing.stderr


def test_counts_cli_with_custom_labels_and_target_map(tmp_path):
    pred, truth = counts_pair(tmp_path / "inputs", "sparse")
    for path in (pred, truth):
        obj = ad.read_h5ad(path)
        obj.obs["condition"] = obj.obs.pop("target").astype(str).replace(
            {"non-targeting": "ctrl", "G1": "guide1", "G2": "guide2", "G3": "guide3"})
        obj.write_h5ad(path)
    mapping = tmp_path / "genes.json"
    mapping.write_text(json.dumps({"guide1": "G1", "guide2": "G2", "guide3": "G3"}))
    process = subprocess.run([
        sys.executable, "-m", "evaluate", "--pred-path", str(pred), "--truth-path", str(truth),
        "--output-dir", str(tmp_path / "scores"), "--pert-col", "condition", "--control-label", "ctrl",
        "--target-gene-map", str(mapping),
    ], cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    summary = json.loads((tmp_path / "scores" / "summary.json").read_text())
    assert set(summary["metrics"]) == SIX
    assert summary["config"]["target_gene_map"] == json.loads(mapping.read_text())
    assert summary["config"]["pert_col"] == "condition"
    assert summary["config"]["control"] == "ctrl"
    assert "no calibrated competition total" in process.stdout


def test_no_measurable_reference_effect_fails_without_fabricating_mse(tmp_path):
    pred, truth = counts_pair(tmp_path / "inputs")
    obj = ad.read_h5ad(truth)
    # No effect AND no variance: the normalized error denominator is exactly zero.
    obj.X[:] = 6
    obj.write_h5ad(truth)
    with pytest.raises(ValueError, match="not positive|no measurable"):
        run_vcc2026_evaluation(pred, truth, tmp_path / "scores")
    assert not (tmp_path / "scores" / "summary.json").exists()


def test_backend_runtime_error_is_a_clean_cli_failure(monkeypatch, capsys):
    from evaluate import vcc2026

    def unavailable(*args, **kwargs):
        raise RuntimeError("GPU DE backend is unavailable")

    monkeypatch.setattr(vcc2026, "run_vcc2026_evaluation", unavailable)
    with pytest.raises(SystemExit) as error:
        vcc2026.main(["--pred-path", "p", "--truth-path", "t", "--output-dir", "out", "--device", "cuda"])
    assert error.value.code == 2
    assert "GPU DE backend is unavailable" in capsys.readouterr().err
