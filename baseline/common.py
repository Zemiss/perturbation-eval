"""Shared VCC conditions, gene mapping and count input validation."""

import hashlib
import json
from pathlib import Path
import subprocess

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DATA_ROOT = Path("/data/dataset/lingshucell/vcc")
LINGSHU_REPO = Path("/data/xiebo/Lingshu-Cell")
CHREODE_REPO = Path("/data/xiebo/Chreode")
LINGSHU_ASSETS = Path("/data/xiebo/models/lingshucell")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(path):
    result = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")


def load_conditions(path):
    table = pd.read_csv(path, keep_default_na=False)
    if not {"cell_type", "target_gene", "n_cells"} <= set(table):
        raise ValueError("Conditions need cell_type,target_gene,n_cells")
    table = table[["cell_type", "target_gene", "n_cells"]].copy()
    if table.empty or not table.cell_type.eq("H1").all():
        raise ValueError("Conditions must contain H1 cells only")
    if table.target_gene.str.strip().ne(table.target_gene).any() or table.target_gene.eq("").any():
        raise ValueError("Condition target genes must be nonempty without surrounding whitespace")
    if table.target_gene.duplicated().any() or table.target_gene.eq("non-targeting").any():
        raise ValueError("Conditions must contain unique non-control perturbations")
    sizes = pd.to_numeric(table.n_cells, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(sizes).all() or (sizes <= 0).any() or (sizes != np.floor(sizes)).any():
        raise ValueError("n_cells must be positive integers")
    table["n_cells"] = sizes.astype(np.int64)
    return table


def lingshu_genes(repository):
    path = Path(repository) / "data/vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json"
    genes = json.loads(path.read_text())["target_gene_to_token_id_info"]["gene_names_ordered"]
    if len({g.upper() for g in genes}) != len(genes):
        raise ValueError("Lingshu gene vocabulary is ambiguous")
    return list(genes)


def native_genes(repository):
    vocab = pd.read_parquet(Path(repository) / "artifacts/gene_vocab.parquet")
    vocab = vocab.sort_values("canonical_index")
    if not np.array_equal(vocab.canonical_index.to_numpy(), np.arange(len(vocab))):
        raise ValueError("Chreode canonical_index must be contiguous")
    genes = vocab.human_symbol.fillna("").astype(str).tolist()
    present = [name.upper() for name in genes if name]
    if len(present) != len(set(present)):
        raise ValueError("Ambiguous human symbols in Chreode vocabulary")
    return genes


def load_gene_priors(path, action_genes, mapping_file):
    """Use fixed ESM priors on the full VCC action axis, independent of output genes."""
    mapping = pd.read_csv(mapping_file, sep="\t", header=None, names=["id", "symbol"])
    ids = dict(zip(mapping.symbol, mapping.id))
    priors, found = None, []
    with np.load(path, allow_pickle=False) as archive:
        for i, gene in enumerate(action_genes):
            key = ids.get(gene, gene)
            if key not in archive:
                continue
            value = np.asarray(archive[key], dtype=np.float32).reshape(-1)
            if not len(value) or not np.isfinite(value).all():
                raise ValueError(f"Invalid ESM prior for {gene}")
            if priors is None:
                priors = np.zeros((len(action_genes) + 1, len(value)), dtype=np.float32)
            if value.shape != priors.shape[1:]:
                raise ValueError("ESM prior dimensions differ between genes")
            priors[i] = value
            found.append(i)
    if not found:
        raise ValueError("No VCC gene priors could be loaded")
    measured = priors[found]
    priors[found] = (measured - measured.mean(axis=0)) / (measured.std(axis=0) + 1e-5)
    found = set(found)
    missing = [gene for i, gene in enumerate(action_genes) if i not in found]
    return priors, missing


def gene_symbols(adata, mapping_file):
    column = next((key for key in ["gene_name", "gene_symbols", "symbol", "feature_name"]
                   if key in adata.var), None)
    values = adata.var[column] if column else adata.var_names
    if pd.isna(values).any():
        raise ValueError("Gene names must not be missing")
    mapping = pd.read_csv(mapping_file, sep="\t", header=None, names=["id", "symbol"])
    lookup = dict(zip(mapping.id, mapping.symbol))
    names = [str(lookup.get(str(value), value)) for value in values]
    if any(not name.strip() for name in names) or len({g.upper() for g in names}) != len(names):
        raise ValueError("Gene names must map to unique nonempty symbols")
    return names


def mapped_counts(adata, rows, source_genes, native_genes, *, layer=None):
    """Map raw counts before normalize_total/log1p; never infer expression scale."""
    matrix = adata.X if layer is None else adata.layers[layer]
    block = matrix[rows]
    values = block.data if sparse.issparse(block) else np.asarray(block)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Chreode VCC inputs must be raw integer counts; select --layer explicitly")
    source = {name.upper(): i for i, name in enumerate(source_genes)}
    destinations = [i for i, name in enumerate(native_genes) if name.upper() in source]
    if not destinations:
        raise ValueError("No measured genes map to the Chreode vocabulary")
    columns = [source[native_genes[i].upper()] for i in destinations]
    selected = block[:, columns]
    if sparse.issparse(selected):
        selected = selected.toarray()
    out = np.zeros((len(rows), len(native_genes)), dtype=np.float32)
    out[:, destinations] = selected
    if (out.sum(axis=1) <= 0).any():
        raise ValueError("A cell has no positive counts after Chreode vocabulary mapping")
    return out


def encode_h1(path, backbone, native_genes, mapping_file, *, condition_col="target_gene",
              cell_type_col="cell_type", layer=None, batch_size=256, controls_only=False):
    """Stream H1 counts through the fixed released encoder into small latent arrays."""
    source = ad.read_h5ad(path, backed="r")
    try:
        if condition_col not in source.obs:
            raise ValueError(f"Missing obs.{condition_col} in {path}")
        if source.obs[condition_col].isna().any():
            raise ValueError("Perturbation labels must not be missing")
        labels = source.obs[condition_col].astype(str).to_numpy()
        mapping = pd.read_csv(mapping_file, sep="\t", header=None, names=["id", "symbol"])
        names = dict(zip(mapping.id, mapping.symbol))
        labels = np.asarray([names.get(label, label) for label in labels])
        labels = np.asarray(["non-targeting" if label in {"ctrl", "control", "NT", "non-targeting"}
                             else label for label in labels])
        if any(not label.strip() or label.strip() != label for label in labels):
            raise ValueError("Perturbation labels must be nonempty without surrounding whitespace")
        keep = np.ones(len(labels), dtype=bool)
        if cell_type_col in source.obs:
            keep &= source.obs[cell_type_col].astype(str).eq("H1").to_numpy()
        if controls_only:
            keep &= labels == "non-targeting"
        rows = np.flatnonzero(keep)
        if not len(rows):
            raise ValueError(f"No selected H1 cells in {path}")
        genes = gene_symbols(source, mapping_file)
        chunks = []
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start + batch_size]
            x = mapped_counts(source, selected, genes, native_genes, layer=layer)
            chunks.append(backbone.encode(x, batch_size=batch_size).detach().cpu().numpy())
        return np.concatenate(chunks), labels[rows], source.obs_names[rows].astype(str).tolist()
    finally:
        source.file.close()


def check_training_labels(train_labels, val_labels, conditions, action_genes):
    test = set(conditions.target_gene)
    allowed = set(action_genes) | {"non-targeting"}
    for name, labels in [("training", train_labels), ("validation", val_labels)]:
        labels = set(labels)
        if "non-targeting" not in labels or len(labels) < 2:
            raise ValueError(f"{name} data need real controls and perturbations")
        if labels - allowed:
            raise ValueError(f"Unknown VCC targets in {name}: {sorted(labels - allowed)[:10]}")
        if labels & test:
            raise ValueError(f"Test perturbations occur in {name} data: {sorted(labels & test)[:10]}")


def validate_lingshu_prediction(path, conditions, genes):
    source = ad.read_h5ad(path, backed="r")
    try:
        if list(source.var_names) != genes or source.X.dtype.kind not in "iu":
            raise ValueError("Lingshu output must preserve its full ordered vocabulary and integer counts")
        if not {"cell_type", "target_gene"} <= set(source.obs) or not source.obs.cell_type.eq("H1").all():
            raise ValueError("Lingshu output is missing the expected H1 perturbation labels")
        sizes = source.obs.target_gene.astype(str).value_counts().to_dict()
        if sizes != conditions.set_index("target_gene").n_cells.to_dict():
            raise ValueError("Lingshu output cell counts differ from the shared condition table")
    finally:
        source.file.close()


def write_prediction(path, expression, labels, genes, metadata):
    expression = np.asarray(expression, dtype=np.float32)
    if expression.shape != (len(labels), len(genes)):
        raise ValueError("Prediction must have cells-by-genes shape")
    negative = 0
    for start in range(0, len(labels), 512):
        block = expression[start:start + 512]
        if not np.isfinite(block).all():
            raise ValueError("Predicted expression contains NaN or infinity")
        negative += int(np.count_nonzero(block < 0))
        np.maximum(block, 0, out=block)
    obs = pd.DataFrame({"cell_type": "H1", "target_gene": labels},
                       index=[f"pred_{i}" for i in range(len(labels))])
    output = ad.AnnData(expression, obs=obs, var=pd.DataFrame(index=genes))
    output.uns["vcc"] = {**metadata, "negative_values_clipped": negative}
    path = Path(path)
    temporary = path.with_suffix(".tmp.h5ad")
    output.write_h5ad(temporary, compression="gzip")
    temporary.replace(path)
    return negative
