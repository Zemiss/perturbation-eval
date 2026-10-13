"""CLI for the four downstream perturbation prediction metrics."""

import argparse
import json
import re
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from evaluate.data import discover_pairs, prepare_pair
from evaluate.metrics import compute_metrics

OPTION_NAMES = {
    "pert_col", "control_label", "group_cols", "control_path", "gene_col",
    "truth_gene_col", "preprocess", "target_sum", "chunk_size", "top_k",
}


def evaluate_pair(pred_path, truth_path, *, top_k=20, **options):
    """Return scores and provenance for one prediction/truth pair, without writing."""
    pred, truth, control, conditions, groups, metadata, counts = prepare_pair(pred_path, truth_path, **options)
    result = compute_metrics(pred, truth, control, metadata["shared_genes"], conditions, top_k=top_k, groups=groups)
    for row, count in zip(result["per_condition"], counts):
        row.update(count)
    return {**metadata, "top_k": top_k, "protocol": "downstream-design-v1", **result}


def _manifest_jobs(path, pred_root, truth_root, defaults):
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    entries = manifest.get("evaluations") if isinstance(manifest, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("manifest must contain a nonempty evaluations list")
    jobs, names = [], set()
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not {"prediction", "truth"}.issubset(entry):
            raise ValueError(f"manifest entry {i}: requires prediction and truth")
        unknown = set(entry) - OPTION_NAMES - {"name", "prediction", "truth"}
        if unknown:
            raise ValueError(f"manifest entry {i}: unknown options {sorted(unknown)}")
        name = str(entry.get("name", f"evaluation_{i + 1}"))
        if name in names:
            raise ValueError(f"Duplicate manifest name {name!r}")
        names.add(name)
        options = {**defaults, **{key: value for key, value in entry.items() if key in OPTION_NAMES}}
        if options.get("control_path"):
            options["control_path"] = truth_root / options["control_path"]
        if options.get("group_cols") is not None and not isinstance(options["group_cols"], list):
            raise ValueError("manifest group_cols must be a JSON list, or null for auto")
        jobs.append((name, pred_root / entry["prediction"], truth_root / entry["truth"], options))
    return jobs


def _save_results(results, output, *, summary_stem="summary"):
    output.mkdir(parents=True, exist_ok=True)
    summaries, csv_rows = [], []
    for i, result in enumerate(results, 1):
        name = result["name"]
        safe_name = re.sub(r"[^\w.-]+", "_", name).strip(".") or "evaluation"
        stem = f"{i:03d}_{safe_name}"
        rows = result.pop("per_condition")
        details = []
        for row in rows:
            row = dict(row)
            group = row.pop("group")
            row.update(dict(zip(result["group_cols"], group)))
            row["de_genes"] = json.dumps(row["de_genes"], ensure_ascii=False)
            details.append(row)
        details_name = f"{stem}_conditions.csv"
        pd.DataFrame(details).to_csv(output / details_name, index=False)
        genes_name = f"{stem}_shared_genes.txt"
        (output / genes_name).write_text("\n".join(result.pop("shared_genes")) + "\n", encoding="utf-8")
        result.update({"details_csv": details_name, "shared_genes_file": genes_name})
        summaries.append(result)
        csv_rows.append({"name": name, **result["metrics"], "n_conditions": result["n_conditions"],
                         "n_groups": result["n_groups"], "n_shared_genes": result["n_shared_genes"],
                         "prediction": result["prediction"], "truth": result["truth"]})
    pd.DataFrame(csv_rows).to_csv(output / f"{summary_stem}.csv", index=False)
    (output / f"{summary_stem}.json").write_text(
        json.dumps({"protocol": "downstream-design-v1", "evaluations": summaries}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Compute DE20 MSE, DE20 delta-r, Pearson-delta and PDS from perturbation expression predictions.")
    parser.add_argument("--pred-path", "--prediction", required=True, type=Path, help="Prediction .h5ad/.npz file or root directory")
    parser.add_argument("--truth-path", "--truth", required=True, type=Path, help="Real .h5ad file or root directory")
    parser.add_argument("--output-dir", type=Path, help="Default: <prediction directory>/evaluation")
    parser.add_argument("--manifest", type=Path, help="JSON list of explicit pairs and dataset-specific options")
    parser.add_argument("--pert-col", help="Perturbation identity column (auto-detected if omitted)")
    parser.add_argument("--control-label", help="Real unperturbed label (auto-detected if omitted)")
    parser.add_argument("--control-path", type=Path, help="Separate real control .h5ad, if truth file omits controls")
    parser.add_argument("--group-cols", help="Comma-separated background columns; auto if omitted; empty string disables grouping")
    parser.add_argument("--gene-col", help="Prediction var column for gene names; default: best shared vocabulary")
    parser.add_argument("--truth-gene-col", help="Truth/control var column for gene names; use var_names for index")
    parser.add_argument("--preprocess", choices=["none", "log1p"], default="none", help="none preserves expression; log1p normalizes each count cell then logs")
    parser.add_argument("--target-sum", type=float, default=10000.0, help="Per-cell total for log1p preprocessing")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Maximum rows per expression read")
    parser.add_argument("--top-k", type=int, default=20, help="True-effect DE genes, default 20")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        pred_root, truth_root = args.pred_path.resolve(), args.truth_path.resolve()
        output = (args.output_dir or ((pred_root if pred_root.is_dir() else pred_root.parent) / "evaluation")).resolve()
        options = {key: getattr(args, key) for key in OPTION_NAMES}
        options["group_cols"] = None if args.group_cols is None else [x.strip() for x in args.group_cols.split(",") if x.strip()]
        if args.manifest:
            if not pred_root.is_dir() or not truth_root.is_dir():
                raise ValueError("With --manifest, --pred-path and --truth-path must be root directories")
            jobs = _manifest_jobs(args.manifest, pred_root, truth_root, options)
        else:
            pairs = discover_pairs(pred_root, truth_root, exclude=output)
            jobs = [(str(p.relative_to(pred_root).with_suffix("")) if pred_root.is_dir() else p.stem, p, t, options) for p, t in pairs]
        results = []
        for name, pred, truth, job_options in jobs:
            print(f"Evaluating {name}: {pred} -> {truth}", flush=True)
            result = {"name": name, **evaluate_pair(pred, truth, **job_options)}
            results.append(result)
            print("  " + "  ".join(f"{key}={value:.6f}" for key, value in result["metrics"].items()), flush=True)
            if result["unpredicted_truth_conditions"]:
                print(f"  Note: {len(result['unpredicted_truth_conditions'])} truth condition/group pairs were not predicted; see summary.json", flush=True)
            if result["undefined_correlations"]["de20_delta_r"] or result["undefined_correlations"]["pearson_delta"]:
                print("  Note: constant-vector correlations were set to 0; see undefined_correlations", flush=True)
            if any(row["pds_candidates"] == 1 for row in result["per_condition"]):
                print("  Note: a group has only one perturbation; its PDS=1 has no discrimination information", flush=True)
        _save_results(results, output)
        print(f"Results: {output / 'summary.json'}", flush=True)
        return 0
    except (ValueError, FileNotFoundError, KeyError, OSError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
