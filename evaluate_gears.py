"""Score GEARS predictions with Chreode's shared-vocabulary DE20 protocol."""

import argparse
from contextlib import closing
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate.data import ExpressionSource
from evaluate.gears_protocol import load_gene_vocab, resolve_adata, resolve_gene_vocab, resolve_prediction, shared_indexes
from evaluate.metrics import compute_metrics
from evaluate.run import _save_results


def _load_predictions(path, reference_genes):
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as handle:
            payload = pickle.load(handle)  # trusted official GEARS artifact
        if not isinstance(payload, dict) or not {"pred", "truth", "pert_cat"}.issubset(payload):
            raise ValueError(f"{path}: GEARS test_res.pkl requires pred, truth and pert_cat")
        pred, truth = (np.asarray(payload[key], dtype=np.float32) for key in ["pred", "truth"])
        conditions = np.asarray(payload["pert_cat"]).astype(str)
        genes = np.asarray(reference_genes, dtype=str)
    else:
        with np.load(path, allow_pickle=False) as payload:
            if not {"pred", "truth", "conditions", "gene_names"}.issubset(payload.files):
                raise ValueError(f"{path}: NPZ requires pred, truth, conditions and gene_names for Chreode reproduction")
            pred, truth = (np.asarray(payload[key]) for key in ["pred", "truth"])
            conditions = payload["conditions"].astype(str)
            genes = payload["gene_names"].astype(str)
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape[1] != len(genes):
        raise ValueError(f"{path}: prediction/truth shapes and gene dimensions must match")
    if conditions.ndim != 1 or len(conditions) != pred.shape[0]:
        raise ValueError(f"{path}: one condition is required per prediction/truth row")
    if genes.ndim != 1 or not np.isfinite(pred).all() or not np.isfinite(truth).all():
        raise ValueError(f"{path}: gene names must be one-dimensional and expression must be finite")
    return pred.astype(np.float64), truth.astype(np.float64), conditions, genes


def run_gears_evaluation(pred_path, truth_path, output_dir, *, gene_vocab=None):
    """Three inputs: saved predictions, real GEARS data, and evaluation reports."""
    prediction = resolve_prediction(Path(pred_path).resolve())
    adata_path = resolve_adata(Path(truth_path).resolve())
    output_dir = Path(output_dir).resolve()
    vocab_path = resolve_gene_vocab(prediction, adata_path, gene_vocab)
    allowed_genes = load_gene_vocab(vocab_path)
    with closing(ExpressionSource(adata_path)) as source:
        reference_genes = source.var["gene_name"].astype(str).tolist() if "gene_name" in source.var else source.var.index.astype(str).tolist()
        pred, truth, labels, pred_genes = _load_predictions(prediction, reference_genes)
        genes, pred_indexes, ref_indexes = shared_indexes(reference_genes, pred_genes, allowed_genes)
        if "condition" not in source.obs or source.obs["condition"].isna().any():
            raise ValueError("Real GEARS data needs complete obs.condition labels")
        real_labels = source.obs["condition"].astype(str).to_numpy()
        ctrl_rows = np.flatnonzero(real_labels == "ctrl")
        if len(ctrl_rows) == 0:
            raise ValueError("No real ctrl cells found in the supplied truth data")
        # Preserve the reference's sparse/dense mean precision and operation order.
        ctrl = source.matrix[ctrl_rows][:, ref_indexes]
        ctrl_mean = np.asarray(ctrl.mean(axis=0)).reshape(-1).astype(np.float64)
        conditions = sorted(condition for condition in np.unique(labels) if condition != "ctrl")
        if not conditions:
            raise ValueError("No non-control test predictions to evaluate")
        missing = set(conditions) - set(real_labels)
        if missing:
            raise ValueError(f"Predicted conditions missing from real GEARS data: {sorted(missing)}")
        pred_shared, truth_shared = pred[:, pred_indexes], truth[:, pred_indexes]
        pred_bulk = np.stack([pred_shared[labels == condition].mean(axis=0) for condition in conditions])
        truth_bulk = np.stack([truth_shared[labels == condition].mean(axis=0) for condition in conditions])
        scores = compute_metrics(pred_bulk, truth_bulk, np.broadcast_to(ctrl_mean, pred_bulk.shape),
                                 genes, conditions, de_protocol="chreode")
        for condition, row in zip(conditions, scores["per_condition"]):
            n = int(np.count_nonzero(labels == condition))
            row.update({"n_prediction_cells": n, "n_truth_cells": n, "n_control_cells": len(ctrl_rows)})
    result = {
        "name": "GEARS", "protocol": "chreode-gears-v1", "prediction": str(prediction), "truth": str(adata_path),
        "output_dir": str(output_dir), "prediction_format": prediction.suffix.lstrip("."),
        "truth_source": "paired_test_truth", "control_source": str(adata_path),
        "gene_vocab": str(vocab_path), "gene_vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "n_prediction_rows": len(labels), "n_prediction_genes": len(pred_genes), "n_truth_genes": len(reference_genes),
        "n_ours_genes_raw": len(allowed_genes), "n_shared_genes": len(genes), "shared_genes": genes,
        "n_conditions": len(conditions), "n_groups": 1, "group_cols": [], "top_k": 20,
        "control_label": "ctrl", "pert_col": "condition", "preprocess": "none", **scores,
        "chreode_metrics": {"shared_de_mse": scores["metrics"]["de20_mse"],
                            "shared_de_delta_pearson": scores["metrics"]["de20_delta_r"]},
    }
    _save_results([dict(result)], output_dir, summary_stem="gears_evaluation_summary", protocol=result["protocol"])
    print(f"Prediction: {prediction}\nTruth/control data: {adata_path}\nChreode gene vocabulary: {vocab_path}", flush=True)
    print("Truth source: paired test truth saved with predictions", flush=True)
    for name, value in result["metrics"].items():
        print(f"{name}: {value:.6f}", flush=True)
    print(f"Saved: {output_dir / 'gears_evaluation_summary.json'}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="Score GEARS using Chreode's shared-vocabulary DE20 protocol and the two downstream metrics.")
    parser.add_argument("--pred-path", "--prediction", required=True, type=Path, help="GEARS predictions.npz/test_res.pkl or their directory")
    parser.add_argument("--truth-path", "--truth", required=True, type=Path, help="Real perturb_processed.h5ad, Norman directory, or GEARS data root")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory receiving evaluation reports")
    parser.add_argument("--gene-vocab", type=Path, help="Original Chreode --ours-genes vocabulary; otherwise detect saved shared genes or Chreode artifacts")
    args = parser.parse_args(argv)
    try:
        run_gears_evaluation(args.pred_path, args.truth_path, args.output_dir, gene_vocab=args.gene_vocab)
    except (ValueError, FileNotFoundError, KeyError, OSError, ImportError, pickle.UnpicklingError, EOFError) as exc:
        parser.exit(2, f"GEARS evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
