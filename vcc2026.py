"""Three-path evaluation using cell-eval2's unmodified VCC2026 counts preset."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
from scipy import sparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


SCORED_METRICS = (
    "pds_cosine", "expr_mse_unbiased_capped_norm",
    "de_wilcoxon_direction_fidelity_yield_raw",
    "de_wilcoxon_direction_reach_raw", "de_wilcoxon_sig_jaccard",
    "de_wilcoxon_lfc_nmae",
)


def _h5ad_path(value):
    path = Path(value).resolve()
    if path.is_dir():
        candidates = sorted(path.rglob("*.h5ad"))
        if len(candidates) != 1:
            raise ValueError(f"{path}: expected exactly one counts .h5ad; found {len(candidates)}. Specify the file explicitly.")
        path = candidates[0]
    if path.suffix.lower() != ".h5ad":
        raise ValueError(f"{path}: VCC2026 requires single-cell counts in H5AD; normalized GEARS PKL/NPZ is not a counts input")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _validate_counts(source, path, pert_col, control):
    if not source.n_obs or not source.n_vars:
        raise ValueError(f"{path}: empty cell/gene axis")
    if not source.var_names.is_unique or source.var_names.isna().any() or any(not x.strip() for x in source.var_names):
        raise ValueError(f"{path}: gene names must be nonempty and unique")
    if pert_col not in source.obs:
        raise ValueError(f"{path}: missing perturbation column {pert_col!r}")
    labels = source.obs[pert_col]
    if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
        raise ValueError(f"{path}: perturbation labels must be complete and nonempty")
    if control not in set(labels.astype(str)):
        raise ValueError(f"{path}: control label {control!r} is missing")
    if set(labels.astype(str)) == {control}:
        raise ValueError(f"{path}: at least one non-control perturbation is required")
    for start in range(0, source.n_obs, 512):
        block = source.X[start:start + 512]
        values = block.data if sparse.issparse(block) else np.asarray(block)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: counts must be finite")
        if np.any(values < 0):
            raise ValueError(f"{path}: counts must be non-negative")
        if np.any(values != np.rint(values)):
            raise ValueError(f"{path}: counts must be integer values; normalized expression is not accepted")
        if np.any(np.asarray(block.astype(np.float64).sum(axis=1)).ravel() > 1_000_000):
            raise ValueError(f"{path}: counts per cell must not exceed 1000000")


def run_vcc2026_evaluation(pred_path, truth_path, output_dir, *, pert_col="target",
                           control_label="non-targeting", target_gene_map=None, device="auto"):
    """Compute the six raw metrics; no baseline/replicate-calibrated total is inferred."""
    from cell_eval2 import __version__, EvalConfig, aggregate_metrics, aggregate_metrics_wide, compute_metrics
    from cell_eval2.baseline import build_run_meta
    from cell_eval2.io import validate_pair
    from cell_eval2.run import metric_cohorts, metric_output_names

    prediction, truth = _h5ad_path(pred_path), _h5ad_path(truth_path)
    output = Path(output_dir).resolve()
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if target_gene_map is not None and (not isinstance(target_gene_map, dict) or
            not all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in target_gene_map.items())):
        raise ValueError("target_gene_map must map perturbation labels to gene-name strings")
    sources = []
    try:
        for path in (prediction, truth):
            source = ad.read_h5ad(path, backed="r")
            sources.append(source)
            _validate_counts(source, path, pert_col, control_label)
        validate_pair(*sources, pert_col=pert_col, control=control_label)
    finally:
        for source in sources:
            source.file.close()

    cfg = EvalConfig.from_preset("vcc2026")
    cfg = replace(cfg, pert_col=pert_col, control=control_label, device=device, target_gene_map=target_gene_map)
    if device != "auto":
        cfg = replace(cfg, de=replace(cfg.de, backend="pdex" if device == "cpu" else "gpudge"))
    print(f"VCC2026 counts: {prediction} -> {truth} ({device}, {cfg.de.backend})", flush=True)
    run_meta = build_run_meta(cfg, truth, prediction)
    tidy = compute_metrics(prediction, truth, config=cfg)
    names = metric_output_names(cfg)
    aggregate = aggregate_metrics(tidy, metrics=names)
    wide = aggregate_metrics_wide(tidy, metrics=names)
    values = {row["metric"]: row["mean"] for row in aggregate.to_dicts()}
    metrics = {name: float(values[name]) if values.get(name) is not None and np.isfinite(values[name]) else None
               for name in SCORED_METRICS}
    summary = {
        "protocol": "cell-eval2-vcc2026-counts", "cell_eval2_version": __version__,
        "score_type": "raw_metrics", "prediction": str(prediction), "truth": str(truth),
        "output_dir": str(output), "config": asdict(cfg), "metrics": metrics,
        "metric_status": {name: "ok" if value is not None else "undefined" for name, value in metrics.items()},
        "metric_cohorts": metric_cohorts(tidy, metrics=names).to_dicts(),
    }
    # Complete computation and aggregation before publishing this run's reports.
    output.mkdir(parents=True, exist_ok=True)
    tidy.write_csv(output / "results.csv")
    wide.write_csv(output / "agg_results.csv")
    import polars as pl
    pl.DataFrame([metrics], schema={name: pl.Float64 for name in metrics}).write_csv(output / "summary.csv")
    cfg.to_yaml(str(output / "run_params.yaml"))
    (output / "run_meta.json").write_text(json.dumps(run_meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    for name, value in metrics.items():
        print(f"{name}: {value:.8g}" if value is not None else f"{name}: undefined", flush=True)
    print(f"Saved: {output / 'summary.json'} (six raw metrics; no calibrated competition total)", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute the six VCC2026 metrics from original single-cell counts using cell-eval2.")
    parser.add_argument("--pred-path", "--prediction", required=True, type=Path, help="Predicted counts H5AD, or directory containing exactly one H5AD")
    parser.add_argument("--truth-path", "--truth", required=True, type=Path, help="Real counts H5AD, or directory containing exactly one H5AD")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory receiving VCC2026 reports")
    parser.add_argument("--pert-col", default="target", help="Perturbation column (default: target)")
    parser.add_argument("--control-label", default="non-targeting", help="Control label (default: non-targeting)")
    parser.add_argument("--target-gene-map", type=Path, help="JSON object mapping construct IDs to their target gene names")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Default: upstream auto selection; cpu pins pdex, cuda requires GPU/gpudge extras")
    args = parser.parse_args(argv)
    try:
        mapping = json.loads(args.target_gene_map.read_text(encoding="utf-8")) if args.target_gene_map else None
        run_vcc2026_evaluation(args.pred_path, args.truth_path, args.output_dir, pert_col=args.pert_col,
                               control_label=args.control_label, target_gene_map=mapping, device=args.device)
    except (ValueError, FileNotFoundError, KeyError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f"VCC2026 evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
