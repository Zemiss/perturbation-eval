import json
import subprocess
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from evaluate.data import discover_pairs
from evaluate.run import evaluate_pair


def write_h5ad(path, x, conditions, genes=("g0", "g1", "g2"), **obs):
    path.parent.mkdir(parents=True, exist_ok=True)
    ad.AnnData(
        X=x if sparse.issparse(x) else np.asarray(x, dtype=float),
        obs=pd.DataFrame({"condition": conditions, **obs}, index=[str(i) for i in range(len(conditions))]),
        var=pd.DataFrame(index=genes),
    ).write_h5ad(path)
    return path


def pair(tmp_path, sparse_truth=False):
    x = np.array([[10, 20, 30], [11, 22, 34], [16, 24, 32]], dtype=float)
    truth = write_h5ad(tmp_path / "truth.h5ad", sparse.csr_matrix(x) if sparse_truth else x, ["ctrl", "a", "b"])
    pred = write_h5ad(tmp_path / "pred.h5ad", x[1:, ::-1], ["a", "b"], genes=("g2", "g1", "g0"))
    return pred, truth


@pytest.mark.parametrize("sparse_truth", [False, True])
def test_gene_alignment_and_chunked_pseudobulks(tmp_path, sparse_truth):
    pred, truth = pair(tmp_path, sparse_truth)
    result = evaluate_pair(pred, truth, chunk_size=1)
    assert result["metrics"] == pytest.approx({"de20_mse": 0, "de20_delta_r": 1, "pearson_delta": 1, "pds": 1})
    assert result["n_shared_genes"] == 3
    assert result["control_label"] == "ctrl"


def test_npz_predictions_ignore_embedded_truth(tmp_path):
    _, truth = pair(tmp_path)
    pred = tmp_path / "pred.npz"
    np.savez(pred, pred=[[11, 22, 34], [16, 24, 32]], truth=np.zeros((2, 3)), conditions=["a", "b"], gene_names=["g0", "g1", "g2"])
    assert evaluate_pair(pred, truth)["metrics"]["de20_mse"] == 0


def test_pseudobulk_weights_cells_within_each_perturbation(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3], [3, 6, 9]], ["ctrl", "a", "a"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[2, 4, 6]], ["a"])
    result = evaluate_pair(pred, truth, chunk_size=1)
    assert result["metrics"]["de20_mse"] == 0
    assert result["per_condition"][0]["n_truth_cells"] == 2


def test_background_specific_controls_and_auto_groups(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3], [100, 100, 100], [101, 102, 103]], ["ctrl", "a", "ctrl", "a"], cell_type=["x", "x", "y", "y"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[1, 2, 3], [101, 102, 103]], ["a", "a"], cell_type=["x", "y"])
    result = evaluate_pair(pred, truth)
    assert result["group_cols"] == ["cell_type"]
    assert result["metrics"]["de20_delta_r"] == pytest.approx(1)
    assert [r["pds_candidates"] for r in result["per_condition"]] == [1, 1]


def test_dose_is_part_of_perturbation_identity_not_control_background(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3], [2, 4, 6]], ["ctrl", "drug_1uM", "drug_2uM"], dose=[0, 1, 2])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[1, 2, 3], [2, 4, 6]], ["drug_1uM", "drug_2uM"], dose=[1, 2])
    result = evaluate_pair(pred, truth)
    assert result["group_cols"] == []
    assert result["metrics"]["pds"] == 1


def test_background_column_aliases_are_aligned(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3], [10, 20, 30], [11, 22, 33]], ["ctrl", "a", "ctrl", "a"], donor_id=["x", "x", "y", "y"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[1, 2, 3], [11, 22, 33]], ["a", "a"], donor=["x", "y"])
    assert evaluate_pair(pred, truth)["metrics"]["de20_delta_r"] == pytest.approx(1)


def test_missing_background_metadata_is_rejected(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3]], ["ctrl", "a"], donor=["d1", "d2"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[1, 2, 3]], ["a"])
    with pytest.raises(ValueError, match="donor"):
        evaluate_pair(pred, truth)


def test_missing_control_is_rejected(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[1, 2, 3]], ["a"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[1, 2, 3]], ["a"])
    with pytest.raises(ValueError, match="control"):
        evaluate_pair(pred, truth)


def test_separate_real_control_file(tmp_path):
    pred, full_truth = pair(tmp_path)
    truth = tmp_path / "no_control.h5ad"
    real = ad.read_h5ad(full_truth)
    real[1:].write_h5ad(truth)
    control = tmp_path / "control.h5ad"
    real[:1].write_h5ad(control)
    assert evaluate_pair(pred, truth, control_path=control)["metrics"]["pearson_delta"] == pytest.approx(1)


def test_prediction_condition_missing_from_truth_is_rejected(tmp_path):
    _, truth = pair(tmp_path)
    pred = write_h5ad(tmp_path / "pred.h5ad", [[1, 2, 3]], ["unknown"])
    with pytest.raises(ValueError, match="unknown"):
        evaluate_pair(pred, truth)


def test_unpredicted_truth_conditions_are_reported(tmp_path):
    pred, truth = pair(tmp_path)
    real = ad.read_h5ad(pred)
    real[:1].write_h5ad(pred)
    result = evaluate_pair(pred, truth)
    assert result["n_conditions"] == 1
    assert result["unpredicted_truth_conditions"] == [{"condition": "b", "group": []}]


def test_duplicate_gene_names_are_rejected(tmp_path):
    pred, truth = pair(tmp_path)
    obj = ad.read_h5ad(pred)
    obj.var["gene_name"] = ["g0", "g0", "g1"]
    obj.write_h5ad(pred)
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_pair(pred, truth, gene_col="gene_name")


def test_conflicting_perturbation_columns_require_explicit_selection(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3], [3, 6, 9]], ["ctrl", "a_dose1", "a_dose2"], target_gene=["ctrl", "a", "a"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[3, 6, 9], [1, 2, 3]], ["a_dose1", "a_dose2"], target_gene=["a", "a"])
    with pytest.raises(ValueError, match="pert-col"):
        evaluate_pair(pred, truth)
    explicit = evaluate_pair(pred, truth, pert_col="condition")
    assert explicit["n_conditions"] == 2
    assert explicit["metrics"]["de20_mse"] == pytest.approx(56 / 3)


def test_conflicting_gene_alignment_ties_require_explicit_columns(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[0, 0, 0], [1, 2, 3]], ["ctrl", "a"], genes=("0", "1", "2"))
    pred = write_h5ad(tmp_path / "pred.h5ad", [[3, 2, 1]], ["a"], genes=("0", "1", "2"))
    for path, genes in [(truth, ["g0", "g1", "g2"]), (pred, ["g2", "g1", "g0"])]:
        obj = ad.read_h5ad(path)
        obj.var["gene_name"] = genes
        obj.write_h5ad(path)
    with pytest.raises(ValueError, match="gene-col"):
        evaluate_pair(pred, truth)
    assert evaluate_pair(pred, truth, gene_col="gene_name", truth_gene_col="gene_name")["metrics"]["de20_mse"] == 0


@pytest.mark.parametrize("annotations", [["a", "a", "b"], ["a", None, "b"]])
def test_unused_invalid_gene_annotations_do_not_block_unique_ids(tmp_path, annotations):
    pred, truth = pair(tmp_path)
    for path in [pred, truth]:
        obj = ad.read_h5ad(path)
        obj.var["gene_name"] = annotations
        obj.write_h5ad(path)
    assert evaluate_pair(pred, truth)["metrics"]["de20_mse"] == 0


def test_log1p_normalizes_each_cell_before_pseudobulk(tmp_path):
    truth = write_h5ad(tmp_path / "truth.h5ad", [[1, 2, 3], [2, 4, 6]], ["ctrl", "a"])
    pred = write_h5ad(tmp_path / "pred.h5ad", [[20, 40, 60]], ["a"])
    assert evaluate_pair(pred, truth, preprocess="log1p", chunk_size=1)["metrics"]["de20_mse"] == pytest.approx(0, abs=1e-20)
    assert evaluate_pair(pred, truth)["metrics"]["de20_mse"] > 0


def test_directory_pairs_use_relative_path(tmp_path):
    p = write_h5ad(tmp_path / "pred" / "vcc" / "test.h5ad", [[1, 2, 3]], ["a"])
    t = write_h5ad(tmp_path / "truth" / "vcc" / "test.h5ad", [[1, 2, 3]], ["a"])
    write_h5ad(tmp_path / "truth" / "other" / "test.h5ad", [[1, 2, 3]], ["a"])
    assert discover_pairs(tmp_path / "pred", tmp_path / "truth") == [(p, t)]


def test_ambiguous_directory_pairing_fails(tmp_path):
    write_h5ad(tmp_path / "pred" / "merged.h5ad", [[1, 2, 3]], ["a"])
    for name in ["vcc", "parse"]:
        write_h5ad(tmp_path / "truth" / name / "test.h5ad", [[1, 2, 3]], ["a"])
    with pytest.raises(ValueError, match="manifest"):
        discover_pairs(tmp_path / "pred", tmp_path / "truth")


@pytest.mark.parametrize("use_manifest", [False, True])
def test_cli_writes_four_scores_and_details(tmp_path, use_manifest):
    pred, truth = pair(tmp_path)
    output = tmp_path / "scores"
    args = ["--pred-path", str(pred), "--truth-path", str(truth), "--output-dir", str(output)]
    if use_manifest:
        manifest = tmp_path / "manifest.json"
        manifest.write_text(json.dumps({"evaluations": [{"name": "norman", "prediction": pred.name, "truth": truth.name}]}))
        args = ["--pred-path", str(tmp_path), "--truth-path", str(tmp_path), "--manifest", str(manifest), "--output-dir", str(output)]
    proc = subprocess.run([sys.executable, "-m", "evaluate.legacy", *args], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = json.loads((output / "summary.json").read_text())
    assert len(result["evaluations"]) == 1
    metrics = result["evaluations"][0]["metrics"]
    assert metrics["pds"] == 1
    assert set(metrics) == {"de20_mse", "de20_delta_r", "pearson_delta", "pds"}
    assert (output / "summary.csv").exists()
    assert len(pd.read_csv(output / result["evaluations"][0]["details_csv"])) == 2
