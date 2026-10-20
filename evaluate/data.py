"""Read three self-contained H5AD files on one reference-defined gene panel."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

CONTROL = "ctrl"
PERT_COL = "condition"
CONTROL_LABELS = {"ctrl", "control", "non-targeting", "non-perturbation", "nt"}
LABEL_COLUMNS = ("target_gene", "gene_target", "condition", "target")


def is_control(value):
    return value.casefold() in CONTROL_LABELS


def _labels(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or pd.isna(values).any():
        raise ValueError(f"{name}: expected complete one-dimensional labels")
    labels = values.astype(str)
    if any(not value.strip() or value != value.strip() for value in labels):
        raise ValueError(f"{name}: labels must be nonempty without surrounding whitespace")
    return labels


def _check_matrix(matrix, name, *, nonnegative=True):
    if len(matrix.shape) != 2 or not all(matrix.shape):
        raise ValueError(f"{name}: expected a nonempty cells-by-genes matrix")
    for start in range(0, matrix.shape[0], 256):
        block = matrix[start:start + 256]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all() or (nonnegative and (values < 0).any()):
            raise ValueError(f"{name}: expression must be finite (no NaN/Inf) and nonnegative")


def perturbation_labels(source, name, *, control_only=False):
    columns = [key for key in LABEL_COLUMNS if key in source.obs]
    if not columns:
        if control_only:
            return np.repeat(CONTROL, source.n_obs)
        raise ValueError(f"{name}: obs.target_gene (or gene_target, condition, target) is required")
    candidates = []
    for key in columns:
        labels = _labels(source.obs[key], f"{name} obs.{key} labels")
        candidates.append(np.asarray([CONTROL if is_control(value) else value for value in labels]))
    if any(not np.array_equal(candidates[0], other) for other in candidates[1:]):
        raise ValueError(f"{name}: conflicting perturbation columns {columns}")
    return candidates[0]


def gene_names(source):
    column = next((key for key in ["gene_name", "gene_symbols", "symbol", "feature_name"]
                   if key in source.var), None)
    genes = _labels(source.var[column] if column else source.var_names, "Gene axis").tolist()
    if not genes or len({gene.upper() for gene in genes}) != len(genes):
        raise ValueError("Gene names must be unique, including case-insensitive matching")
    return genes


@dataclass
class Source:
    path: Path
    adata: object
    genes: list
    rows: np.ndarray
    labels: np.ndarray
    scale: str
    layer: object
    context: object


def open_source(path, role):
    path = Path(path).resolve()
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"{role}: expected an H5AD file, not PKL or a directory")
    source = ad.read_h5ad(path, backed="r")
    try:
        genes = gene_names(source)
        metadata = source.uns.get("evaluation", {})
        if not isinstance(metadata, dict):
            raise ValueError("uns['evaluation'] must be a metadata dictionary")
        scale = metadata.get("expression_scale", source.uns.get("vcc", {}).get("expression_scale", "counts"))
        layer = metadata.get("layer")
        if scale not in {"counts", "lognorm"}:
            raise ValueError("Expression scale must be counts or lognorm")
        if layer is not None and layer not in source.layers:
            raise ValueError(f"Declared expression layer is missing: {layer}")
        labels = perturbation_labels(source, role, control_only=role == "control")
        rows = np.flatnonzero(labels == CONTROL if role == "control" else labels != CONTROL)
        if not len(rows):
            raise ValueError(f"{role}: no selected cells")
        context = None
        if "cell_type" in source.obs:
            contexts = set(_labels(source.obs.iloc[rows].cell_type, f"{role} cell_type"))
            if len(contexts) != 1:
                raise ValueError(f"{role}: prepare a single cell context before evaluation")
            context = next(iter(contexts))
        return Source(path, source, genes, rows, labels[rows], scale, layer, context)
    except Exception:
        source.file.close()
        raise


def fill_normalized(source, genes, destination):
    """Normalize declared linear abundance on the fixed reference gene axis."""
    lookup = {gene.upper(): i for i, gene in enumerate(source.genes)}
    missing = [gene for gene in genes if gene.upper() not in lookup]
    if missing:
        raise ValueError(f"{source.path.name}: missing {len(missing)} reference genes; "
                         "prepare a fixed common reference panel first")
    columns = [lookup[gene.upper()] for gene in genes]
    matrix = source.adata.X if source.layer is None else source.adata.layers[source.layer]
    if matrix is None:
        raise ValueError(f"{source.path.name}: expression matrix is missing")
    digest = hashlib.sha256()
    for start in range(0, len(source.rows), 256):
        block = matrix[source.rows[start:start + 256]]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{source.path.name}: expression must be finite and nonnegative")
        if source.scale == "counts" and (values != np.floor(values)).any():
            raise ValueError("Fractional counts: annotate lognorm in uns['evaluation']['expression_scale'] "
                             "or prepare raw counts in X")
        block = block[:, columns]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        block = block.astype(np.float64, copy=False)
        if source.scale == "lognorm":
            with np.errstate(over="raise", invalid="raise"):
                block = np.expm1(block)
        totals = block.sum(axis=1, keepdims=True)
        if not np.isfinite(totals).all() or (totals <= 0).any():
            raise ValueError("A cell has zero or invalid abundance on the reference gene panel")
        normalized = np.log1p(block / totals * 10000).astype(np.float32)
        destination[start:start + len(block)] = normalized
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def attach(matrix, labels, genes):
    obs = pd.DataFrame({PERT_COL: labels}, index=[f"cell_{i}" for i in range(len(labels))])
    return ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))
