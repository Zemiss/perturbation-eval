"""VCC2026 metric functions on GEARS lognorm, extending exclusion to multi-gene targets."""

from dataclasses import replace

from cell_eval2 import EvalConfig, aggregate_metrics
from cell_eval2.de import TargetResolution, prepare_de
from cell_eval2.de_compute import compute_de
from cell_eval2.metrics.delta import PRED_TRACE_CAP_K, _across_pert_budget
from cell_eval2.metrics.discrimination import discrimination_score
from cell_eval2.moments import correction_for, unbiased_sq_dist
from cell_eval2.norm import check_scale_limit, validate_input_type
from cell_eval2.prep import pseudobulk_with_moments
from cell_eval2.run import dispatch_de_metrics
import numpy as np
import polars as pl

from .data import CONTROL, PERT_COL


METRICS = (
    "pds_cosine", "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)


def gears_config(targets):
    cfg = EvalConfig.from_preset("vcc2026")
    return replace(
        cfg, pert_col=PERT_COL, control=CONTROL, input_type="lognorm", device="cpu",
        # Upstream CPU lognorm skips its counts-only CPM filter; make that explicit.
        filter=replace(cfg.filter, filter_gene_min_cpm_cell=None),
        de=replace(cfg.de, backend="pdex"),
        target_gene_map={label: genes[0] for label, genes in targets.items()},
    )


def _expression_rows(pred_bulk, real_bulk, targets, genes):
    """Upstream lognorm correction and cap, excluding ALL component genes of each target.

    Match cell-eval2 0.18.0: full-panel variance corrections stay intact after
    removing on-target coordinates; the final metric is a ratio of sums.
    """
    perts, pm, p_moments = pred_bulk
    real_perts, rm, r_moments = real_bulk
    if not np.array_equal(perts, real_perts):
        raise ValueError("Prediction/truth perturbations do not align")
    pc = correction_for(p_moments, perts, pm, comparator="lognorm")
    rc = correction_for(r_moments, perts, rm, comparator="lognorm")
    ci = list(perts).index(CONTROL)
    rows = np.flatnonzero(perts != CONTROL)
    positions = {gene: i for i, gene in enumerate(genes)}
    excluded = {
        k: np.asarray([positions[g] for g in targets[str(perts[i])] if g in positions], dtype=int)
        for k, i in enumerate(rows)
    }
    widths = np.asarray([len(genes) - len(excluded[k]) for k in range(len(rows))], dtype=float)
    if np.any(widths <= 0):
        raise ValueError("A perturbation has no genes left after target-gene exclusion")
    weights = 1.0 / widths
    pred_term = np.minimum(pc[rows], PRED_TRACE_CAP_K * rc[rows])
    claim = float((pred_term * weights).sum())
    if claim > 0:
        budget = _across_pert_budget(pm[rows], excluded, weights)
        pred_term = pred_term * min(1.0, budget / claim)
    numerator = unbiased_sq_dist(pm[rows], rm[rows], pred_term, rc[rows])
    denominator = unbiased_sq_dist(rm[rows], np.broadcast_to(rm[ci], rm[rows].shape),
                                   rc[rows], np.full(len(rows), rc[ci]))
    for k, i in enumerate(rows):
        for col in excluded[k]:
            numerator[k] -= (pm[i, col] - rm[i, col]) ** 2
            denominator[k] -= (rm[i, col] - rm[ci, col]) ** 2
    out = []
    for k, i in enumerate(rows):
        for name, value in [("expr_mse_unbiased_capped", numerator[k] / widths[k]),
                            ("expr_distance_unbiased", denominator[k] / widths[k])]:
            out.append({"metric": name, "perturbation": str(perts[i]), "value": float(value)})
    return out


def compute_six(pred, truth, targets):
    cfg = gears_config(targets)
    for matrix in (pred, truth):
        validate_input_type(matrix, "lognorm")
        check_scale_limit(matrix, "lognorm", cfg.max_counts_per_cell)
    genes = list(truth.var_names)
    pb = pseudobulk_with_moments(pred, PERT_COL)
    rb = pseudobulk_with_moments(truth, PERT_COL)
    rows = _expression_rows(pb, rb, targets, genes)

    panel_targets = {gene for ts in targets.values() for gene in ts}
    keep = np.asarray([gene not in panel_targets for gene in genes])
    if not keep.any():
        raise ValueError("No genes left for PDS after excluding all panel target genes")
    pds = discrimination_score(
        pred_bulk=(pb[0], pb[1][:, keep]), real_bulk=(rb[0], rb[1][:, keep]),
        control=CONTROL, control_source="real", distance="cosine",
        rank_denominator="n-1", tie_policy="midrank", exclude_target_gene=False,
    )
    rows.extend({"metric": "pds_cosine", "perturbation": label, "value": value}
                for label, value in pds.items())

    # Compute Wilcoxon and BH-FDR over the full gene panel BEFORE on-target exclusion.
    # The upstream target map is single-gene; pre-excluding every component extends
    # the same rule to GEARS A+B without dropping or splitting its perturbation identity.
    on_target = pl.DataFrame(
        [(label, gene) for label, ts in targets.items() for gene in ts],
        schema={"target": pl.String, "feature": pl.String}, orient="row",
    )
    de_tables = []
    for matrix in (pred, truth):
        table = compute_de(
            matrix, backend="pdex", groupby=PERT_COL, reference=CONTROL,
            mean_calc=cfg.de.mean_calc, epsilon=cfg.de.epsilon, input_type="lognorm",
            target_sum=cfg.target_sum, filter_gene_min_cpm_cell=None,
            fdr_scope=cfg.de.fdr_scope, threads=cfg.num_threads,
        )
        de_tables.append(table.join(on_target, on=["target", "feature"], how="anti"))
    prepared = prepare_de(
        *de_tables, control=CONTROL, sort_by=cfg.de.sort_by,
        p_adj_threshold=cfg.de.p_adj_threshold, nan_lfc_policy=cfg.de.nan_lfc_policy,
        target_resolution=TargetResolution(cfg.target_gene_map, len(targets)),
    )
    rows.extend(dispatch_de_metrics(list(METRICS[2:]), prepared, cfg))
    tidy = pl.DataFrame(rows, schema={"metric": pl.String, "perturbation": pl.String,
                                      "value": pl.Float64})
    aggregated = aggregate_metrics(tidy, metrics=list(METRICS))
    values = dict(aggregated.select("metric", "mean").iter_rows())
    result = {name: float(values[name]) if name in values and np.isfinite(values[name]) else None
              for name in METRICS}
    return result, cfg
