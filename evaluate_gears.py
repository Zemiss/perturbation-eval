"""Score existing official GEARS Norman predictions with two directory arguments."""

import argparse
from contextlib import closing
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data import ExpressionSource
from evaluate.run import _save_results, evaluate_pair


def _norman_adata(data_dir):
    candidates = [data_dir / "norman" / "perturb_processed.h5ad", data_dir / "perturb_processed.h5ad"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Norman data not found: expected {candidates[0]} or {candidates[1]}")


def _convert_test_res(path, adata_path, destination):
    # Chreode's run_norman_gears.py saves the official GEARS dictionary here.
    # These local pickle artifacts must come from a trusted GEARS run.
    with path.open("rb") as handle:
        test_res = pickle.load(handle)
    if not isinstance(test_res, dict) or not {"pred", "pert_cat"}.issubset(test_res):
        raise ValueError(f"{path}: GEARS test_res.pkl requires pred and pert_cat")
    with closing(ExpressionSource(adata_path)) as source:
        column = "gene_name" if "gene_name" in source.var else "var_names"
        gene_names = np.asarray(source.gene_options(column)[column], dtype=str)
    pred = np.asarray(test_res["pred"])
    conditions = np.asarray(test_res["pert_cat"], dtype=str)
    if pred.ndim != 2 or pred.shape[1] != len(gene_names):
        raise ValueError(f"{path}: prediction gene dimension must match Norman's native gene order ({len(gene_names)} genes)")
    if conditions.ndim != 1 or len(conditions) != pred.shape[0]:
        raise ValueError(f"{path}: pert_cat must have one condition per prediction row")
    # Embedded truth is deliberately unused; the supplied Norman H5AD is truth.
    np.savez_compressed(destination, pred=pred, conditions=conditions, gene_names=gene_names)


def run_gears_evaluation(data_dir, output_dir):
    """Read Norman truth plus existing GEARS output, save and return four scores."""
    data_dir, output_dir = Path(data_dir).resolve(), Path(output_dir).resolve()
    adata_path = _norman_adata(data_dir)
    npz_path, pickle_path = output_dir / "predictions.npz", output_dir / "test_res.pkl"
    if npz_path.is_file():
        prediction = npz_path
    elif pickle_path.is_file():
        prediction = pickle_path
    else:
        raise FileNotFoundError(f"No existing GEARS predictions: expected {npz_path} or {pickle_path}")
    with closing(ExpressionSource(adata_path)) as source:
        truth_gene_col = "gene_name" if "gene_name" in source.var else "var_names"
    with TemporaryDirectory(prefix="gears_eval_") as temporary:
        if prediction == pickle_path:
            eval_prediction = Path(temporary) / "predictions.npz"
            _convert_test_res(prediction, adata_path, eval_prediction)
        else:
            eval_prediction = prediction
        result = {
            "name": "GEARS",
            **evaluate_pair(eval_prediction, adata_path, pert_col="condition", control_label="ctrl",
                            group_cols=[], gene_col="var_names", truth_gene_col=truth_gene_col,
                            preprocess="none"),
        }
    result.update({"prediction": str(prediction), "data_dir": str(data_dir),
                   "prediction_format": prediction.suffix.lstrip(".")})
    # Preserve Chreode's existing training summary.json in this same directory.
    _save_results([dict(result)], output_dir, summary_stem="gears_evaluation_summary")
    print("GEARS Norman evaluation", flush=True)
    print(f"Truth: {adata_path}", flush=True)
    print(f"Prediction: {prediction}", flush=True)
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.6f}", flush=True)
    print(f"Saved: {output_dir / 'gears_evaluation_summary.json'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score existing GEARS Norman results: DE20 MSE, DE20 delta-r, Pearson-delta and PDS.")
    parser.add_argument("--data-dir", required=True, type=Path, help="GEARS data root containing norman/perturb_processed.h5ad (or the norman directory)")
    parser.add_argument("--output-dir", required=True, type=Path, help="Existing GEARS predictions directory; also receives evaluation outputs")
    args = parser.parse_args(argv)
    try:
        run_gears_evaluation(args.data_dir, args.output_dir)
    except (ValueError, FileNotFoundError, KeyError, OSError, pickle.UnpicklingError, EOFError) as exc:
        parser.exit(2, f"GEARS evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
