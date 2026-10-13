"""Deterministic file pairing and bounded-memory expression pseudobulks."""

from contextlib import ExitStack
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata.io import read_elem
except ImportError:  # AnnData 0.10
    from anndata._io.specs import read_elem

PERT_COLUMNS = ("target_gene", "condition", "cytokine", "drug__dose", "perturbation", "perturbation_name")
GROUP_COLUMNS = (
    ("cell_type", "celltype", "cell_type_name", "celltype_name"),
    ("cell_line_id", "cell_line", "cell_line_name"),
    ("donor", "donor_id", "donor_name"),
    ("timepoint", "time", "time_point"),
    ("dose", "dose_value", "dose_values", "concentration"),
)
CONTROL_LABELS = {"ctrl", "control", "non-targeting", "non_targeting", "non-perturbation", "pbs", "dmso", "dmso_tf__0.0 um", "vehicle", "ntc"}
GENE_COLUMNS = ("gene_name", "gene_names", "gene_symbol", "feature_name", "symbol")


def discover_pairs(pred_path, truth_path, *, exclude=None):
    """Match relative paths, then unique stems, then a single real file.

    An ambiguous match is an error rather than a guess based on filename order.
    """
    pred_path, truth_path = Path(pred_path).resolve(), Path(truth_path).resolve()
    excluded = Path(exclude).resolve() if exclude else None

    def files(path, extensions):
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path] if path.is_file() else sorted(path.rglob("*"))
        return [p for p in candidates if p.is_file() and p.suffix.lower() in extensions
                and not any(part.startswith(".") for part in p.relative_to(path.parent if path.is_file() else path).parts)
                and not (excluded and p.is_relative_to(excluded))]

    predictions, truths = files(pred_path, {".h5ad", ".npz"}), files(truth_path, {".h5ad"})
    if not predictions or not truths:
        raise ValueError("No prediction (.h5ad/.npz) or truth (.h5ad) files found")
    pairs = []
    for prediction in predictions:
        matches = []
        if pred_path.is_dir() and truth_path.is_dir():
            relative = prediction.relative_to(pred_path).with_suffix(".h5ad")
            matches = [p for p in truths if p.relative_to(truth_path) == relative]
        if not matches:
            matches = [p for p in truths if p.stem == prediction.stem]
        if not matches and len(truths) == 1:
            matches = truths
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely pair {prediction}; use --manifest with explicit prediction/truth paths")
        if prediction == matches[0]:
            raise ValueError(f"Prediction and truth resolve to the same file: {prediction}")
        pairs.append((prediction, matches[0]))
    return pairs


class ExpressionSource:
    def __init__(self, path, *, prediction=False):
        self.path = Path(path)
        self.adata = None
        self.is_npz = self.path.suffix.lower() == ".npz"
        if self.is_npz:
            if not prediction:
                raise ValueError("Truth/control inputs must be .h5ad files")
            with np.load(self.path, allow_pickle=False) as data:
                required = {"pred", "conditions", "gene_names"}
                if not required.issubset(data.files):
                    raise ValueError(f"{self.path}: NPZ requires {sorted(required)}")
                self.matrix = np.asarray(data["pred"])
                self.obs = pd.DataFrame({"condition": data["conditions"].astype(str)})
                self.var = pd.DataFrame(index=data["gene_names"].astype(str))
                for aliases in GROUP_COLUMNS:
                    for name in aliases:
                        if name in data.files:
                            self.obs[name] = data[name]
            if self.matrix.ndim != 2 or self.matrix.shape != (len(self.obs), len(self.var)):
                raise ValueError(f"{self.path}: NPZ expression/label shape mismatch")
        elif self.path.suffix.lower() == ".h5ad":
            # Reading a full backed AnnData still loads layers/obsm/uns. Read
            # only metadata, then open X on disk (as in Lingshu's eval loader).
            with h5py.File(self.path, "r") as handle:
                obs, var = read_elem(handle["obs"]), read_elem(handle["var"])
            self.adata = ad.AnnData(obs=obs, var=var, shape=(len(obs), len(var)),
                                    filename=str(self.path), filemode="r")
            self.obs, self.var = self.adata.obs.copy(), self.adata.var
            # Do not retain a sparse on-disk X object beyond an open file handle.
            self.matrix = self.adata.X
            if self.matrix is None:
                self.close()
                raise ValueError(f"{self.path}: missing expression matrix X")
        else:
            raise ValueError(f"Unsupported input: {self.path}; expected .h5ad/.npz")
        if len(self.obs) == 0 or len(self.var) == 0:
            self.close()
            raise ValueError(f"{self.path}: empty expression data")

    def close(self):
        if self.adata is not None:
            self.adata.file.close()

    def gene_options(self, column):
        if column and column != "var_names":
            if column not in self.var:
                raise ValueError(f"{self.path}: missing gene column {column!r}")
            return {column: self.var[column]}
        result = {"var_names": self.var.index}
        if not column:
            result.update({name: self.var[name] for name in GENE_COLUMNS if name in self.var})
        return result

    def bulk(self, pert_col, group_cols, indexes, *, chunk_size, preprocess, target_sum, wanted=None):
        labels = self.obs[[*group_cols, pert_col]]
        if labels.isna().any().any():
            raise ValueError(f"{self.path}: missing perturbation/background labels")
        keys = list(labels.astype(str).itertuples(index=False, name=None))
        sums, counts = {}, {}
        for start in range(0, len(keys), chunk_size):
            end = min(start + chunk_size, len(keys))
            block_keys = keys[start:end]
            positions = {}
            for i, key in enumerate(block_keys):
                if wanted is None or key in wanted:
                    positions.setdefault(key, []).append(i)
            if not positions:
                continue
            x = self.matrix[start:end].astype(np.float64)
            values = x.data if sparse.issparse(x) else np.asarray(x)
            if not np.isfinite(values).all():
                raise ValueError(f"{self.path}: expression values must be finite")
            if preprocess == "log1p":
                if np.any(values < 0):
                    raise ValueError(f"{self.path}: log1p preprocessing requires nonnegative counts")
                totals = np.asarray(x.sum(axis=1)).ravel()
                scales = np.divide(target_sum, totals, out=np.zeros_like(totals), where=totals > 0)
                if sparse.issparse(x):
                    x = x.multiply(scales[:, None]).tocsr()
                    x.data = np.log1p(x.data)
                else:
                    x = np.log1p(x * scales[:, None])
            x = x[:, indexes]
            for key, rows in positions.items():
                subtotal = np.asarray(x[rows].sum(axis=0)).ravel()
                sums[key] = sums.get(key, np.zeros(len(indexes))) + subtotal
                counts[key] = counts.get(key, 0) + len(rows)
        return {key: value / counts[key] for key, value in sums.items()}, counts


def _pert_column(source, requested=None):
    if source.is_npz:
        return "condition"
    if requested:
        if requested not in source.obs:
            raise ValueError(f"{source.path}: missing perturbation column {requested!r}")
        return requested
    candidates = [column for column in PERT_COLUMNS if column in source.obs]
    if candidates:
        first = candidates[0]
        for other in candidates[1:]:
            pairs = source.obs[[first, other]].drop_duplicates()
            if (pairs.groupby(first, dropna=False, observed=True)[other].nunique(dropna=False).max() > 1
                    or pairs.groupby(other, dropna=False, observed=True)[first].nunique(dropna=False).max() > 1):
                raise ValueError(f"{source.path}: conflicting perturbation columns {first!r}/{other!r}; specify --pert-col with the complete identity")
        return first
    raise ValueError(f"{source.path}: no perturbation labels; specify --pert-col")


def _align_genes(sources, columns):
    choices = []
    for source, column in zip(sources, columns):
        valid = {}
        for name, values in source.gene_options(column).items():
            missing = pd.isna(values).any()
            index = pd.Index(values).astype(str)
            if missing or index.has_duplicates:
                if column:
                    issue = "Missing" if missing else "Duplicate"
                    raise ValueError(f"{issue} gene names in {source.path} column {name!r}; resolve annotations before evaluation")
                continue
            valid[name] = index
        if not valid:
            raise ValueError(f"{source.path}: no complete unique gene names available")
        choices.append(valid)
    # Try available name vocabularies; choose the largest intersection.
    from itertools import product
    best, best_alignment, ambiguous = None, None, False
    for options in product(*(choice.items() for choice in choices)):
        names = [index for _, index in options]
        common = set(names[0]).intersection(*(set(x) for x in names[1:]))
        alignment = tuple(sorted(zip(*(index.get_indexer(sorted(common)) for index in names))))
        if best is None or len(common) > len(best[0]):
            best = (common, names, [key for key, _ in options])
            best_alignment, ambiguous = alignment, False
        elif len(common) == len(best[0]) and alignment != best_alignment:
            ambiguous = True
    if ambiguous:
        raise ValueError("Tied gene vocabularies imply different alignments; specify --gene-col and --truth-gene-col")
    common, names, selected = best
    genes = [name for name in names[1] if name in common]  # truth order
    if len(genes) < 2:
        raise ValueError("Need at least two common genes; align gene IDs or specify gene columns")
    return genes, [index.get_indexer(genes) for index in names], selected


def prepare_pair(pred_path, truth_path, *, pert_col=None, control_label=None, group_cols=None,
                 control_path=None, gene_col=None, truth_gene_col=None, preprocess="none",
                 target_sum=10000.0, chunk_size=2048):
    if chunk_size < 1 or not np.isfinite(target_sum) or target_sum <= 0:
        raise ValueError("chunk_size and target_sum must be positive")
    if preprocess not in {"none", "log1p"}:
        raise ValueError("preprocess must be none or log1p")
    with ExitStack() as stack:
        sources = []
        for path, prediction in [(pred_path, True), (truth_path, False), *(([(control_path, False)]) if control_path else [])]:
            source = ExpressionSource(path, prediction=prediction)
            stack.callback(source.close)
            sources.append(source)
        pred, truth = sources[:2]
        control = sources[2] if control_path else truth
        true_col = _pert_column(truth, pert_col)
        pred_col = _pert_column(pred, pert_col)
        if pert_col is None and true_col in pred.obs:
            pred_col = true_col
        ctrl_col = _pert_column(control, true_col)
        ctrl_values = control.obs[ctrl_col].dropna().astype(str).unique()
        if control_label is None:
            matches = [value for value in ctrl_values if value.lower() in CONTROL_LABELS]
            if len(matches) != 1:
                raise ValueError(f"Cannot infer one real control label from {control.path}; specify --control-label (found {matches})")
            control_label = matches[0]
        if str(control_label) not in ctrl_values:
            raise ValueError(f"No real control cells with label {control_label!r}")
        control_label = str(control_label)

        auto_groups = group_cols is None
        if auto_groups:
            group_cols = [next(name for name in aliases if any(name in s.obs for s in sources))
                          for aliases in GROUP_COLUMNS[:-1] if any(name in s.obs for name in aliases for s in sources)]
        group_cols = list(group_cols)
        if len(set(group_cols)) != len(group_cols) or any(col in {pred_col, true_col, ctrl_col} for col in group_cols):
            raise ValueError("group_cols must be unique background columns, separate from perturbation")
        for column in group_cols:
            if auto_groups:
                aliases = next(aliases for aliases in GROUP_COLUMNS if column in aliases)
                for source in sources:
                    if column not in source.obs:
                        alternate = next((name for name in aliases if name in source.obs), None)
                        if alternate is not None:
                            source.obs[column] = source.obs[alternate]
            known = [s.obs[column] for s in sources if column in s.obs]
            if not known:
                raise ValueError(f"Missing group column {column!r}")
            unique = pd.concat(known).dropna().astype(str).unique()
            for source in sources:
                if column not in source.obs:
                    if len(unique) != 1:
                        raise ValueError(f"{source.path}: missing heterogeneous background column {column!r}")
                    source.obs[column] = unique[0]

        genes, indexes, selected_columns = _align_genes(sources, [gene_col, truth_gene_col, *([truth_gene_col] if control_path else [])])
        kwargs = dict(chunk_size=chunk_size, preprocess=preprocess, target_sum=target_sum)
        pred_bulk, pred_counts = pred.bulk(pred_col, group_cols, indexes[0], **kwargs)
        keys = sorted(key for key in pred_bulk if key[-1] != control_label)
        if not keys:
            raise ValueError("No predicted non-control perturbations to evaluate")
        wanted = set(keys) | {(*key[:-1], control_label) for key in keys}
        true_bulk, true_counts = truth.bulk(true_col, group_cols, indexes[1], wanted=wanted, **kwargs)
        if control_path:
            ctrl_bulk, ctrl_counts = control.bulk(ctrl_col, group_cols, indexes[2], wanted=wanted, **kwargs)
        else:
            ctrl_bulk, ctrl_counts = true_bulk, true_counts
        for key in keys:
            if key not in true_bulk:
                raise ValueError(f"No truth for predicted condition/group {key}")
            if (*key[:-1], control_label) not in ctrl_bulk:
                raise ValueError(f"No matching real control for condition/group {key}; supply --control-path")
        truth_labels = truth.obs[[*group_cols, true_col]]
        if truth_labels.isna().any().any():
            raise ValueError("Truth contains missing perturbation/background labels")
        truth_keys = set(truth_labels.astype(str).itertuples(index=False, name=None))
        missing = sorted(key for key in truth_keys if key[-1] != control_label and key not in keys)
        metadata = {
            "prediction": str(Path(pred_path).resolve()), "truth": str(Path(truth_path).resolve()),
            "control_path": str(Path(control_path).resolve()) if control_path else None,
            "pert_col": true_col, "pred_pert_col": pred_col, "control_label": control_label,
            "group_cols": group_cols, "preprocess": preprocess, "target_sum": target_sum,
            "gene_columns": selected_columns, "shared_genes": genes, "n_shared_genes": len(genes),
            "n_prediction_genes": len(pred.var), "n_truth_genes": len(truth.var),
            "n_conditions": len(keys), "n_groups": len({key[:-1] for key in keys}),
            "unpredicted_truth_conditions": [{"condition": key[-1], "group": list(key[:-1])} for key in missing],
        }
        row_metadata = [{"n_prediction_cells": pred_counts[key], "n_truth_cells": true_counts[key],
                         "n_control_cells": ctrl_counts[(*key[:-1], control_label)]} for key in keys]
        return (np.stack([pred_bulk[key] for key in keys]), np.stack([true_bulk[key] for key in keys]),
                np.stack([ctrl_bulk[(*key[:-1], control_label)] for key in keys]),
                [key[-1] for key in keys], [key[:-1] for key in keys], metadata, row_metadata)
