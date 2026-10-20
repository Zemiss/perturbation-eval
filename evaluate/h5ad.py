"""Legacy counts Python API; the four-path CLI uses the fixed reference panel in data.py."""

from contextlib import ExitStack
from pathlib import Path

import anndata as ad
from cell_eval2 import EvalConfig
from cell_eval2.norm import check_scale_limit, validate_input_type
import numpy as np
import pandas as pd

from .data import CONTROL, PERT_COL, _check_matrix, _labels


CONTROL_LABELS = {"ctrl", "non-targeting", "non-perturbation"}
LABEL_COLUMNS = ("target_gene", "gene_target", "condition", "target")


def _perturbations(source, name, *, control_only=False):
    columns = [key for key in LABEL_COLUMNS if key in source.obs]
    if not columns:
        if control_only:
            return np.repeat(CONTROL, source.n_obs)
        raise ValueError(f"{name} needs obs.target_gene (or gene_target, condition, target)")
    labels = []
    for key in columns:
        values = _labels(source.obs[key], f"{name} obs.{key} labels")
        labels.append(np.asarray([CONTROL if v.lower() in CONTROL_LABELS else v for v in values]))
    if any(not np.array_equal(labels[0], other) for other in labels[1:]):
        raise ValueError(f"{name}: conflicting perturbation columns {columns}")
    return labels[0]


def _gene_axes(source, name):
    names = pd.Index(_labels(source.var_names, f"{name} gene names"))
    if not names.is_unique:
        raise ValueError(f"{name} gene names must be unique")
    axes = [("var_names", names)]
    if "gene_name" in source.var:
        values = source.var["gene_name"]
        if not pd.isna(values).any():
            symbols = pd.Index(np.asarray(values, dtype=str))
            if symbols.is_unique and all(g.strip() for g in symbols):
                axes.append(("gene_name", symbols))
    return axes


def _align_axes(sources):
    axes = {name: _gene_axes(source, name) for name, source in sources.items()}
    for pred_key, genes in axes["prediction"]:
        selected = {"prediction": (pred_key, np.arange(len(genes)))}
        for name in ("truth", "control"):
            matches = [(key, values.get_indexer(genes)) for key, values in axes[name]
                       if genes.isin(values).all()]
            if not matches:
                break
            if any(not np.array_equal(matches[0][1], order) for _, order in matches[1:]):
                raise ValueError(f"{name}: conflicting gene_name and var_names alignment")
            selected[name] = matches[0]
        if len(selected) == 3:
            return genes, selected, axes["prediction"]
    raise ValueError("All prediction genes must match unique genes in truth and control; "
                     "use matching var_names or var.gene_name annotations")


def load_h5ad(pred_path, truth_path, control_path):
    paths = {"prediction": Path(pred_path).resolve(), "truth": Path(truth_path).resolve(),
             "control": Path(control_path).resolve()}
    for name, path in paths.items():
        if path.suffix.lower() != ".h5ad":
            raise ValueError(f"{name} path must be an H5AD file")
    cfg = EvalConfig.from_preset("vcc2026")
    with ExitStack() as stack:
        sources = {}
        for name, path in paths.items():
            sources[name] = ad.read_h5ad(path, backed="r")
            stack.callback(sources[name].file.close)
        labels = {name: _perturbations(source, name, control_only=name == "control")
                  for name, source in sources.items()}
        perts = sorted(set(labels["prediction"]) - {CONTROL})
        if not perts:
            raise ValueError("Prediction H5AD contains no non-control perturbations")
        missing = sorted(set(perts) - set(labels["truth"]))
        if missing:
            raise ValueError(f"Prediction perturbations absent from truth H5AD: {missing[:10]}")
        rows = {"prediction": np.flatnonzero(labels["prediction"] != CONTROL),
                "truth": np.flatnonzero(np.isin(labels["truth"], perts)),
                "control": np.flatnonzero(labels["control"] == CONTROL)}
        if not len(rows["control"]):
            raise ValueError("No real control cells found in --control-path")
        contexts = set()
        for name, source in sources.items():
            if "cell_type" in source.obs:
                contexts.update(_labels(source.obs.iloc[rows[name]]["cell_type"],
                                        f"{name} cell_type"))
        if len(contexts) > 1:
            raise ValueError(f"cell_type must be a single matching context across inputs: {sorted(contexts)}")

        genes, orders, pred_axes = _align_axes(sources)
        matrices = {}
        for name, source in sources.items():
            if source.X is None:
                raise ValueError(f"{name} H5AD needs expression counts in X")
            matrix = source.X[rows[name]]
            # Validate each source before combining; never hide fractional/invalid real data.
            _check_matrix(matrix, f"{name} X", nonnegative=False)
            validate_input_type(ad.AnnData(matrix), "counts")
            check_scale_limit(ad.AnnData(matrix), "counts", cfg.max_counts_per_cell)
            matrices[name] = ad.AnnData(
                matrix[:, orders[name][1]],
                obs=pd.DataFrame({PERT_COL: labels[name][rows[name]]},
                                 index=[f"{name}_{i}" for i in range(len(rows[name]))]),
                var=pd.DataFrame(index=genes),
            )
        predicted = ad.concat([matrices["prediction"], matrices["control"]], merge="same")
        real = ad.concat([matrices["truth"], matrices["control"]], merge="same")

        lookup = {}
        for _, values in pred_axes:
            for alias, gene in zip(values, genes):
                key = alias.upper()
                if key in lookup and lookup[key] != gene:
                    raise ValueError(f"Ambiguous gene alias: {alias}")
                lookup[key] = gene
        targets = {}
        for label in perts:
            components = [g.strip() for g in label.split("+") if g.lower() not in CONTROL_LABELS]
            if len(components) != 1 or not components[0]:
                raise ValueError(f"H5AD counts requires single-gene perturbations: {label!r}")
            targets[label] = [lookup.get(components[0].upper(), components[0])]
        unmeasured = sorted({g for ts in targets.values() for g in ts if g not in genes})
        if not any(g in genes for ts in targets.values() for g in ts):
            raise ValueError("No perturbation target matches a measured gene; check gene identifiers")
        metadata = {
            "prediction": str(paths["prediction"]), "truth_data": str(paths["truth"]),
            "control_data": str(paths["control"]),
            "truth_source": "perturbed cells from truth H5AD X; real controls from --control-path X",
            "n_predicted_cells": len(rows["prediction"]), "n_truth_cells": len(rows["truth"]),
            "n_control_cells": len(rows["control"]), "n_genes": len(genes),
            "n_perturbations": len(targets), "target_genes": targets,
            "unmeasured_target_genes": unmeasured,
            "gene_order": {name: key for name, (key, _) in orders.items()},
            "discarded_genes": {name: source.n_vars - len(genes) for name, source in sources.items()},
            "skipped_truth_perturbations": sorted(set(labels["truth"]) - set(perts) - {CONTROL}),
            "ignored_prediction_controls": int(np.count_nonzero(labels["prediction"] == CONTROL)),
            "ignored_truth_controls": int(np.count_nonzero(labels["truth"] == CONTROL)),
        }
    return predicted, real, targets, metadata
