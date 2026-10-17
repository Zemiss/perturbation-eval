"""Read a GEARS test result and the matching processed data without loading training cells."""

from pathlib import Path
import pickle

import anndata as ad
from cell_eval2.norm import validate_input_type
import numpy as np
import pandas as pd
from scipy import sparse


CONTROL = "ctrl"
PERT_COL = "condition"


def _labels(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or pd.isna(values).any():
        raise ValueError(f"{name}: expected a complete one-dimensional label array")
    labels = values.astype(str)
    if any(not label.strip() for label in labels):
        raise ValueError(f"{name}: empty perturbation label")
    return labels


def _check_matrix(matrix, name, *, nonnegative=True):
    if len(matrix.shape) != 2 or not all(matrix.shape):
        raise ValueError(f"{name}: expected a nonempty cells-by-genes matrix")
    for start in range(0, matrix.shape[0], 512):
        block = matrix[start:start + 512]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all():
            raise ValueError(f"{name}: expression contains NaN or infinity")
        if nonnegative and np.any(values < 0):
            raise ValueError(f"{name}: real log-normalized expression must be non-negative")


def load_gears(pred_path, truth_path):
    prediction, source = Path(pred_path).resolve(), Path(truth_path).resolve()
    if prediction.is_dir():
        prediction /= "test_res.pkl"
    if prediction.suffix.lower() != ".pkl":
        raise ValueError("--pred-path must be GEARS test_res.pkl or its directory")
    if source.is_dir():
        candidates = [source / "norman" / "perturb_processed.h5ad",
                      source / "perturb_processed.h5ad"]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise ValueError(f"Specify one processed H5AD; found {len(existing)} among {candidates}")
        source = existing[0]
    if source.suffix.lower() != ".h5ad":
        raise ValueError("--truth-path must be a GEARS data directory or processed H5AD")
    with prediction.open("rb") as handle:
        payload = pickle.load(handle)  # User's trusted GEARS prediction artifact.
    if not isinstance(payload, dict) or not {"pred", "truth", "pert_cat"} <= payload.keys():
        raise ValueError("GEARS PKL requires pred, truth and pert_cat")
    pred, truth = np.asarray(payload["pred"]), np.asarray(payload["truth"])
    _check_matrix(pred, "PKL pred", nonnegative=False)
    _check_matrix(truth, "PKL truth")
    labels = _labels(payload["pert_cat"], "PKL pert_cat")
    if pred.shape != truth.shape or len(labels) != pred.shape[0]:
        raise ValueError("PKL pred/truth shapes must match, with one pert_cat per cell")
    keep = labels != CONTROL
    pred, truth, labels = pred[keep], truth[keep], labels[keep]
    if not len(labels):
        raise ValueError("PKL contains no non-control test cells")

    reference = ad.read_h5ad(source, backed="r")
    try:
        if PERT_COL not in reference.obs:
            raise ValueError("Processed H5AD needs obs.condition, including ctrl cells")
        real_labels = _labels(reference.obs[PERT_COL], "H5AD condition")
        missing = sorted(set(labels) - set(real_labels))
        if missing:
            raise ValueError(f"PKL perturbations absent from processed H5AD: {missing[:10]}")
        gene_values = reference.var["gene_name"] if "gene_name" in reference.var else reference.var_names
        if pd.isna(gene_values).any():
            raise ValueError("H5AD gene names must not be missing")
        genes = pd.Index(np.asarray(gene_values, dtype=str))
        if not genes.is_unique or any(not gene.strip() for gene in genes):
            raise ValueError("H5AD gene names must be nonempty and unique")
        if pred.shape[1] != len(genes):
            raise ValueError("PKL gene dimension must match processed H5AD gene order")
        if "gene_names" in payload:
            supplied = pd.Index(_labels(payload["gene_names"], "PKL gene_names"))
            if not supplied.is_unique or len(supplied) != len(genes) or set(supplied) != set(genes):
                raise ValueError("PKL gene_names must match the processed H5AD gene set")
            order = supplied.get_indexer(genes)
            pred, truth = pred[:, order], truth[:, order]
        ctrl_rows = np.flatnonzero(real_labels == CONTROL)
        if not len(ctrl_rows):
            raise ValueError("No real ctrl cells found in processed H5AD")
        ctrl = reference.X[ctrl_rows]
    finally:
        reference.file.close()
    _check_matrix(ctrl, "H5AD ctrl")
    # Validate the two real sources separately: concatenating counts controls
    # with fractional test truth would otherwise hide a mixed-scale input.
    validate_input_type(ad.AnnData(truth), "lognorm")
    validate_input_type(ad.AnnData(ctrl), "lognorm")

    # Expression is already in log1p space. Never round it or reconstruct raw counts.
    n_negative = int(np.count_nonzero(pred < 0))
    pred = np.maximum(pred, 0)
    obs = pd.DataFrame({PERT_COL: np.concatenate([labels, np.repeat(CONTROL, len(ctrl_rows))])})
    obs.index = [f"cell_{i}" for i in range(len(obs))]
    var = pd.DataFrame(index=genes)

    def attach_control(matrix):
        if sparse.issparse(ctrl):
            return sparse.vstack([sparse.csr_matrix(matrix), ctrl], format="csr")
        return np.concatenate([matrix, np.asarray(ctrl)])

    pred_ad = ad.AnnData(attach_control(pred), obs=obs.copy(), var=var.copy())
    truth_ad = ad.AnnData(attach_control(truth), obs=obs.copy(), var=var.copy())

    gene_lookup = {gene.upper(): gene for gene in genes}
    if len(gene_lookup) != len(genes):
        raise ValueError("Case-insensitive gene names are ambiguous")
    targets = {}
    for label in sorted(set(labels)):
        parts = [part.strip() for part in label.split("+")]
        if any(not part for part in parts):
            raise ValueError(f"Invalid GEARS perturbation label: {label!r}")
        components = sorted(set(part for part in parts if part != CONTROL))
        if not components:
            raise ValueError(f"Invalid GEARS target: {label!r}")
        targets[label] = [gene_lookup.get(part.upper(), part) for part in components]
    metadata = {
        "prediction": str(prediction), "truth_data": str(source),
        "truth_source": "paired test truth from PKL; controls from processed H5AD X",
        "n_test_cells": len(labels), "n_control_cells": len(ctrl_rows),
        "n_genes": len(genes), "n_perturbations": len(targets),
        "negative_predictions_clipped_to_zero": n_negative,
        "target_genes": targets,
        "unmeasured_target_genes": sorted({g for ts in targets.values() for g in ts if g not in genes}),
        "gene_order": "aligned by PKL gene_names" if "gene_names" in payload else "assumed identical to processed H5AD",
    }
    return pred_ad, truth_ad, targets, metadata
