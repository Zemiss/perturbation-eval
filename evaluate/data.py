"""Read model-independent H5AD inputs on a fixed reference gene panel."""

from dataclasses import dataclass
import hashlib
import json
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


def load_gene_map(path):
    """Read an optional two-column identifier-to-symbol TSV, without a header."""
    if path is None:
        return {}
    table = pd.read_csv(Path(path).expanduser(), sep="\t", header=None, dtype=str, keep_default_na=False)
    if table.shape[1] != 2 or table.empty:
        raise ValueError("--gene-map needs a nonempty two-column TSV: gene_id<TAB>gene_symbol")
    mapping = {}
    for identifier, symbol in table.itertuples(index=False, name=None):
        identifier, symbol = identifier.strip(), symbol.strip()
        if not identifier or not symbol:
            raise ValueError("--gene-map contains an empty identifier or symbol")
        if identifier in mapping and mapping[identifier] != symbol:
            raise ValueError(f"Conflicting gene mapping for {identifier}")
        mapping[identifier] = symbol
    return mapping


def mapped_gene(value, mapping):
    key = value.split('.')[0] if value.startswith('ENSG') else value
    return mapping.get(value, mapping.get(key, value))


def canonical_label(value, *, control_label=None, gene_map=None):
    mapping = gene_map or {}
    parts = value.split("+")
    if any(not part.strip() for part in parts):
        raise ValueError(f"Invalid perturbation label: {value!r}")
    genes = {mapped_gene(part.strip(), mapping) for part in parts
             if not is_control(part.strip()) and
             (control_label is None or part.strip().casefold() != control_label.casefold())}
    return "+".join(sorted(genes)) if genes else CONTROL


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


def perturbation_labels(source, name, *, control_only=False, pert_col=None,
                        control_label=None, gene_map=None):
    if pert_col is not None and pert_col not in source.obs:
        raise ValueError(f"{name}: obs.{pert_col} is missing")
    columns = [pert_col] if pert_col else [key for key in LABEL_COLUMNS if key in source.obs]
    if not columns:
        if control_only:
            return np.repeat(CONTROL, source.n_obs)
        raise ValueError(f"{name}: obs.target_gene (or gene_target, condition, target) is required")
    candidates = []
    for key in columns:
        labels = _labels(source.obs[key], f"{name} obs.{key} labels")
        candidates.append(np.asarray([canonical_label(value, control_label=control_label,
                                                       gene_map=gene_map) for value in labels]))
    if any(not np.array_equal(candidates[0], other) for other in candidates[1:]):
        raise ValueError(f"{name}: conflicting perturbation columns {columns}")
    return candidates[0]


def gene_names(source, gene_map=None):
    column = next((key for key in ["gene_name", "gene_symbols", "symbol", "feature_name"]
                   if key in source.var), None)
    genes = _labels(source.var[column] if column else source.var_names, "Gene axis").tolist()
    genes = [mapped_gene(gene, gene_map or {}) for gene in genes]
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


def open_source(path, role, *, pert_col=None, control_label=None, gene_map=None,
                input_type=None, layer=None, allow_unlabeled_controls=True):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"{role}: expected an H5AD file, not PKL or a directory")
    source = ad.read_h5ad(path, backed="r")
    try:
        genes = gene_names(source, gene_map)
        metadata = source.uns.get("evaluation", {})
        if not isinstance(metadata, dict):
            raise ValueError("uns['evaluation'] must be a metadata dictionary")
        legacy = source.uns.get("vcc", {})
        if not isinstance(legacy, dict):
            legacy = {}
        scale = input_type or metadata.get("expression_scale", legacy.get("expression_scale", "counts"))
        layer = layer if layer is not None else metadata.get("layer")
        if scale not in {"counts", "lognorm"}:
            raise ValueError("Expression scale must be counts or lognorm")
        if layer is not None and layer not in source.layers:
            raise ValueError(f"Declared expression layer is missing: {layer}")
        labels = perturbation_labels(source, role, control_only=role == "control" and allow_unlabeled_controls,
                                     pert_col=pert_col, control_label=control_label, gene_map=gene_map)
        rows = np.flatnonzero(labels == CONTROL if role == "control" else labels != CONTROL)
        if not len(rows):
            hint = "; provide --control-path with real control cells or set --control-label" if role == "control" else ""
            raise ValueError(f"{role}: no selected cells{hint}")
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


def fill_expression(source, genes, destination, *, normalize):
    """Copy counts or normalize linear abundance on the fixed reference gene axis."""
    if not normalize and source.scale != "counts":
        raise ValueError(f"{source.path.name}: vcc2026 requires raw counts; use --protocol lognorm "
                         "for already log-normalized predictions and all models compared with them")
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
            raise ValueError("Fractional counts: declare the correct --pred-input-type/--truth-input-type/"
                             "--control-input-type lognorm (with --protocol lognorm), or provide raw counts")
        block = block[:, columns]
        block = block.toarray() if sparse.issparse(block) else np.asarray(block)
        block = block.astype(np.float64, copy=False)
        if normalize and source.scale == "lognorm":
            with np.errstate(over="raise", invalid="raise"):
                block = np.expm1(block)
        totals = block.sum(axis=1, keepdims=True)
        if not np.isfinite(totals).all() or (totals <= 0).any():
            raise ValueError("A cell has zero or invalid abundance on the reference gene panel")
        prepared = (np.log1p(block / totals * 10000) if normalize else block).astype(np.float32)
        destination[start:start + len(block)] = prepared
        digest.update(prepared.tobytes())
    return digest.hexdigest()


def fill_normalized(source, genes, destination):
    """Compatibility helper for the explicitly selected lognorm protocol."""
    return fill_expression(source, genes, destination, normalize=True)


def reference_genes(truth, panel_path=None, gene_map=None):
    if panel_path is None:
        return truth.genes
    path = Path(panel_path).expanduser()
    if path.suffix.lower() == '.json':
        panel = json.loads(path.read_text(encoding='utf-8'))
        genes = _labels(panel['target_gene_to_token_id_info']['gene_names_ordered'], 'VCC gene panel').tolist()
    else:
        table = pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",", header=None,
                            dtype=str, keep_default_na=False, comment="#")
        if table.shape[1] != 1 or table.empty:
            raise ValueError("--gene-panel needs one gene per row in TXT/CSV/TSV or a Lingshu vocabulary JSON")
        genes = table.iloc[:, 0].str.strip().tolist()
        if genes and genes[0].casefold() in {"gene", "gene_name", "gene_symbol", "symbol", "genes"}:
            genes = genes[1:]
    genes = [mapped_gene(gene, gene_map or {}) for gene in genes]
    if not genes or not all(genes) or len({gene.upper() for gene in genes}) != len(genes):
        raise ValueError("--gene-panel must contain nonempty, unique gene names")
    lookup = {gene.upper(): gene for gene in truth.genes}
    missing = [gene for gene in genes if gene.upper() not in lookup]
    if missing:
        raise ValueError(f"Truth H5AD is missing {len(missing)} panel genes: {missing[:10]}")
    return [lookup[gene.upper()] for gene in genes]


def attach(matrix, labels, genes):
    obs = pd.DataFrame({PERT_COL: labels}, index=[f"cell_{i}" for i in range(len(labels))])
    return ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))
