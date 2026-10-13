"""The four pseudobulk metrics documented in README.md."""

from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cdist


def _pearson(x, y):
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    x = x - x.mean()
    y = y - y.mean()
    scale_x, scale_y = np.max(np.abs(x)), np.max(np.abs(y))
    if scale_x == 0 or scale_y == 0:
        return 0.0, True
    x, y = x / scale_x, y / scale_y
    return float(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1)), False


def compute_metrics(pred, truth, control, gene_names, conditions, *, top_k=20, groups=None):
    """Score aligned pseudobulks; control is the SAME real control for both deltas.

    Rows represent perturbations within background groups. PDS candidates are
    restricted to each group. Other metrics combine all evaluated rows.
    """
    pred, truth, control = (np.asarray(x, dtype=np.float64) for x in (pred, truth, control))
    if pred.ndim != 2 or pred.shape != truth.shape or pred.shape != control.shape:
        raise ValueError("pred, truth and control must have equal 2D shapes")
    n, g = pred.shape
    if n == 0 or g < 2 or top_k < 2:
        raise ValueError("Need at least one perturbation, two common genes and top_k >= 2")
    if len(gene_names) != g or len(conditions) != n:
        raise ValueError("Gene/condition labels do not match matrix dimensions")
    if not all(np.isfinite(x).all() for x in (pred, truth, control)):
        raise ValueError("Expression values must be finite")
    groups = [()] * n if groups is None else [tuple(x) for x in groups]
    if len(groups) != n:
        raise ValueError("groups length does not match pseudobulk rows")
    if len(set(zip(groups, conditions))) != n:
        raise ValueError("Duplicate perturbations within the same group")

    true_delta, pred_delta = truth - control, pred - control
    rows, undefined_de = [], 0
    k = min(top_k, g)
    for i, condition in enumerate(conditions):
        # Stable ties are resolved by the shared gene order, using ONLY truth.
        de = np.argsort(-np.abs(true_delta[i]), kind="stable")[:k]
        correlation, undefined = _pearson(pred_delta[i, de], true_delta[i, de])
        undefined_de += int(undefined)
        rows.append({
            "condition": str(condition), "group": list(groups[i]), "de_k": k,
            "de_genes": [str(gene_names[j]) for j in de],
            "de20_mse": float(np.mean((pred_delta[i, de] - true_delta[i, de]) ** 2)),
            "de20_delta_r": correlation, "de20_delta_r_undefined": undefined,
        })

    by_group = defaultdict(list)
    for i, group in enumerate(groups):
        by_group[group].append(i)
    for indexes in by_group.values():
        references = truth[indexes]
        t = len(indexes)
        for local, i in enumerate(indexes):
            distances = cdist(pred[i:i + 1], references, metric="cityblock")[0]
            closer = int(np.count_nonzero(distances < distances[local]))
            rows[i].update({
                "pds": 1.0 - 2.0 * closer / t, "pds_rank": closer,
                "pds_candidates": t, "pds_correct_distance": float(distances[local]),
            })
    pearson, undefined_global = _pearson(np.abs(pred_delta), np.abs(true_delta))
    return {
        "metrics": {
            "de20_mse": float(np.mean([r["de20_mse"] for r in rows])),
            "de20_delta_r": float(np.mean([r["de20_delta_r"] for r in rows])),
            "pearson_delta": pearson,
            "pds": float(np.mean([r["pds"] for r in rows])),
        },
        "undefined_correlations": {"de20_delta_r": undefined_de, "pearson_delta": int(undefined_global)},
        "per_condition": rows,
    }
