"""Input resolution and vocabulary alignment for Chreode's GEARS reproduction."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def resolve_adata(path):
    if path.is_file() and path.suffix.lower() == ".h5ad":
        return path
    candidates = [path / "norman" / "perturb_processed.h5ad", path / "perturb_processed.h5ad"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Norman truth not found: expected {candidates[0]} or {candidates[1]}")


def resolve_prediction(path):
    if path.is_file():
        if path.suffix.lower() not in {".npz", ".pkl"}:
            raise ValueError("GEARS prediction must be a .npz or trusted .pkl file")
        return path
    candidates = [path / "predictions.npz", path / "test_res.pkl"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"GEARS predictions not found: expected {candidates[0]} or {candidates[1]}")


def resolve_gene_vocab(prediction, adata, requested):
    if requested is not None:
        path = Path(requested).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Gene vocabulary not found: {path}")
        return path
    for path in [prediction.parent / "shared_vocab_genes.txt",
                 prediction.parent / "shared_eval" / "shared_vocab_genes.txt"]:
        if path.is_file():
            return path
    for root in adata.parents:
        if root.name.lower() != "chreode" and not (root / "src" / "cellworldmodel").is_dir():
            continue
        for relative in ["artifacts/gene_vocab.parquet", "data/phase0/gene_vocab.parquet",
                         "output/phase0/gene_vocab.parquet"]:
            path = root / relative
            if path.is_file():
                return path
    raise FileNotFoundError("Chreode gene vocabulary not found; supply --gene-vocab with the original --ours-genes file. Scoring all genes would change the reproduction protocol.")


def load_gene_vocab(path):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        if "human_symbol" in frame:
            fallback = frame["canonical_gene"] if "canonical_gene" in frame else pd.Series([""] * len(frame))
            return [str(human) if str(human) and str(human).upper() != "NAN" else str(canonical)
                    for human, canonical in zip(frame["human_symbol"], fallback)]
        for column in ["gene_name", "symbol", "canonical_gene"]:
            if column in frame:
                return frame[column].astype(str).tolist()
        raise ValueError(f"No gene-name column in {path}")
    if suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            obj = next((obj[key] for key in ["gene_names", "genes", "ours_gene_names"] if key in obj), None)
        if not isinstance(obj, list):
            raise ValueError(f"No gene-name list in {path}")
        return [str(value) for value in obj]
    if suffix == ".npy":
        return np.load(path, allow_pickle=False).astype(str).tolist()
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            for key in ["gene_names", "genes", "ours_gene_names"]:
                if key in arrays:
                    return arrays[key].astype(str).tolist()
        raise ValueError(f"No gene-name array in {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def shared_indexes(reference_genes, prediction_genes, allowed_genes):
    """Match Chreode's case-folding, reference order and duplicate policy."""
    prediction_lookup = {str(gene).upper(): i for i, gene in enumerate(prediction_genes)}
    allowed = {str(gene).upper() for gene in allowed_genes if str(gene) and str(gene).upper() != "NAN"}
    genes, pred_indexes, ref_indexes, seen = [], [], [], set()
    for i, gene in enumerate(reference_genes):
        key = str(gene).upper()
        if key not in seen and key in allowed and key in prediction_lookup:
            seen.add(key)
            genes.append(str(gene))
            pred_indexes.append(prediction_lookup[key])
            ref_indexes.append(i)
    if len(genes) < 2:
        raise ValueError("Need at least two common genes inside Chreode's allowed vocabulary")
    return genes, np.asarray(pred_indexes, dtype=np.int64), np.asarray(ref_indexes, dtype=np.int64)
