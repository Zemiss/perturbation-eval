import json
import pickle
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from evaluate.evaluate_gears import run_gears_evaluation


def gears_files(tmp_path, prediction_format):
    data_dir = tmp_path / "data"
    norman_dir = data_dir / "norman"
    norman_dir.mkdir(parents=True)
    output_dir = tmp_path / "outputs" / "GEARS"
    output_dir.mkdir(parents=True)
    x = np.array([[10, 20, 30], [11, 22, 34], [16, 24, 32]], dtype=float)
    ad.AnnData(
        X=sparse.csr_matrix(x),
        obs=pd.DataFrame({"condition": ["ctrl", "a+ctrl", "a+b"], "cell_type": ["K562"] * 3}, index=["c0", "c1", "c2"]),
        var=pd.DataFrame({"gene_name": ["a", "b", "c"]}, index=["ENSG1", "ENSG2", "ENSG3"]),
    ).write_h5ad(norman_dir / "perturb_processed.h5ad")
    if prediction_format == "npz":
        np.savez_compressed(output_dir / "predictions.npz", pred=x[1:, ::-1],
                            conditions=["a+ctrl", "a+b"], gene_names=["c", "b", "a"],
                            truth=x[1:, ::-1])
    elif prediction_format == "pkl":
        with (output_dir / "test_res.pkl").open("wb") as handle:
            pickle.dump({"pred": x[1:], "pert_cat": np.array(["a+ctrl", "a+b"], dtype=object), "truth": x[1:]}, handle)
    (output_dir / "shared_vocab_genes.txt").write_text("a\nb\nc\n")
    return data_dir, output_dir


@pytest.mark.parametrize("prediction_format", ["npz", "pkl"])
def test_gears_three_paths_score_existing_results(tmp_path, prediction_format):
    data_dir, output_dir = gears_files(tmp_path, prediction_format)
    scores = tmp_path / "scores"
    result = run_gears_evaluation(output_dir, data_dir, scores)
    assert result["metrics"] == pytest.approx({"de20_mse": 0, "de20_delta_r": 1, "pearson_delta": 1, "pds": 1})
    assert result["truth"] == str((data_dir / "norman" / "perturb_processed.h5ad").resolve())
    source_name = "predictions.npz" if prediction_format == "npz" else "test_res.pkl"
    assert result["prediction"] == str((output_dir / source_name).resolve())
    if prediction_format == "pkl":
        assert not list(output_dir.glob("*.npz"))
    report = json.loads((scores / "gears_evaluation_summary.json").read_text())
    assert report["evaluations"][0]["metrics"] == result["metrics"]
    assert (scores / "gears_evaluation_summary.csv").exists()
    assert not (output_dir / "gears_evaluation_summary.json").exists()


def test_gears_accepts_norman_subdirectory(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "npz")
    assert run_gears_evaluation(output_dir, data_dir / "norman", tmp_path / "scores")["metrics"]["de20_mse"] == 0


def test_gears_preserves_existing_training_summary_and_predictions(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "pkl")
    summary = output_dir / "summary.json"
    summary.write_text('{"seed": 1, "epochs": 20}')
    before = (output_dir / "test_res.pkl").read_bytes()
    run_gears_evaluation(output_dir, data_dir, tmp_path / "scores")
    assert summary.read_text() == '{"seed": 1, "epochs": 20}'
    assert (output_dir / "test_res.pkl").read_bytes() == before


def test_gears_missing_predictions_does_not_generate_them(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "none")
    with pytest.raises(FileNotFoundError, match="predictions.npz.*test_res.pkl"):
        run_gears_evaluation(output_dir, data_dir, tmp_path / "scores")
    assert [p.name for p in output_dir.iterdir()] == ["shared_vocab_genes.txt"]


def test_gears_missing_norman_does_not_choose_other_dataset(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "npz")
    (data_dir / "norman").rename(data_dir / "adamson")
    with pytest.raises(FileNotFoundError, match="perturb_processed.h5ad"):
        run_gears_evaluation(output_dir, data_dir, tmp_path / "scores")


def test_gears_npz_has_priority_over_pickle(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "npz")
    (output_dir / "test_res.pkl").write_bytes(b"do not read this fallback")
    assert run_gears_evaluation(output_dir, data_dir, tmp_path / "scores")["prediction"].endswith("predictions.npz")


def test_gears_pickle_matrix_must_match_native_gene_order(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "pkl")
    with (output_dir / "test_res.pkl").open("wb") as handle:
        pickle.dump({"pred": [[1, 2]], "truth": [[1, 2]], "pert_cat": ["a+ctrl"]}, handle)
    with pytest.raises(ValueError, match="gene"):
        run_gears_evaluation(output_dir, data_dir, tmp_path / "scores")


def test_legacy_gears_module_runs_with_three_required_paths(tmp_path):
    data_dir, output_dir = gears_files(tmp_path, "pkl")
    scores = tmp_path / "scores"
    process = subprocess.run([sys.executable, "-m", "evaluate.chreode_gears", "--pred-path", str(output_dir), "--truth-path", str(data_dir), "--output-dir", str(scores)],
                             cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert "de20_mse" in process.stdout and "pds" in process.stdout
    assert (scores / "gears_evaluation_summary.json").exists()


def test_gears_uses_paired_test_truth_like_chreode(tmp_path):
    data_dir, prediction_dir = gears_files(tmp_path, "npz")
    path = data_dir / "norman" / "perturb_processed.h5ad"
    obj = ad.read_h5ad(path)
    x = obj.X.toarray()
    x[1:] += 100  # the dataset contains additional/different cells from the saved test rows
    obj.X = sparse.csr_matrix(x)
    obj.write_h5ad(path)
    result = run_gears_evaluation(prediction_dir, path, tmp_path / "scores")
    assert result["metrics"]["de20_mse"] == 0
    assert result["truth_source"] == "paired_test_truth"


def test_gears_filters_to_chreode_allowed_gene_vocabulary(tmp_path):
    data_dir, prediction_dir = gears_files(tmp_path, "npz")
    (prediction_dir / "shared_vocab_genes.txt").write_text("a\nb\n")
    with np.load(prediction_dir / "predictions.npz") as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    payload["pred"][:, 0] += 1000  # c lies outside Chreode's shared vocabulary
    np.savez(prediction_dir / "predictions.npz", **payload)
    result = run_gears_evaluation(prediction_dir, data_dir, tmp_path / "scores")
    assert result["shared_genes"] == ["a", "b"]
    assert result["metrics"]["de20_mse"] == 0


def test_gears_missing_vocabulary_does_not_silently_score_all_genes(tmp_path):
    data_dir, prediction_dir = gears_files(tmp_path, "npz")
    (prediction_dir / "shared_vocab_genes.txt").unlink()
    with pytest.raises(FileNotFoundError, match="gene-vocab"):
        run_gears_evaluation(prediction_dir, data_dir, tmp_path / "scores")


def test_gears_missing_paired_truth_is_rejected(tmp_path):
    data_dir, prediction_dir = gears_files(tmp_path, "pkl")
    with (prediction_dir / "test_res.pkl").open("wb") as handle:
        pickle.dump({"pred": [[11, 22, 34]], "pert_cat": ["a+ctrl"]}, handle)
    with pytest.raises(ValueError, match="truth"):
        run_gears_evaluation(prediction_dir, data_dir, tmp_path / "scores")


def test_gears_auto_finds_chreode_artifact_vocabulary(tmp_path):
    data_dir, prediction_dir = gears_files(tmp_path, "npz")
    root = tmp_path / "chreode"
    root.mkdir()
    moved_data = root / "data"
    data_dir.rename(moved_data)
    (prediction_dir / "shared_vocab_genes.txt").unlink()
    (root / "artifacts").mkdir()
    pd.DataFrame({"human_symbol": ["a", "b", "c"]}).to_parquet(root / "artifacts" / "gene_vocab.parquet")
    result = run_gears_evaluation(prediction_dir, moved_data, tmp_path / "scores")
    assert result["gene_vocab"].endswith("artifacts/gene_vocab.parquet") or result["gene_vocab"].endswith("artifacts\\gene_vocab.parquet")
    assert result["metrics"]["de20_mse"] == 0
