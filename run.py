"""Evaluate H5AD counts with independent controls, or existing GEARS artifacts."""

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import pickle
import sys

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def evaluate(pred_path, truth_path, output_dir, control_path=None):
    from cell_eval2 import __version__ as cell_eval_version
    from evaluate.data import load_gears
    from evaluate.metrics import compute_counts_six, compute_six

    if Path(pred_path).suffix.lower() == ".h5ad":
        if control_path is None:
            raise ValueError("H5AD predictions require --control-path with real experimental controls")
        from evaluate.h5ad import load_h5ad

        pred, truth, targets, metadata = load_h5ad(pred_path, truth_path, control_path)
        protocol = "h5ad-counts-vcc2026-metrics"
        print(f"H5AD counts: {metadata['n_predicted_cells']} predicted cells, "
              f"{metadata['n_truth_cells']} true cells, {metadata['n_control_cells']} real controls; "
              f"{len(targets)} perturbations, {metadata['n_genes']} genes; CPU/pdex", flush=True)
        print(f"Real controls: {metadata['control_data']}", flush=True)
        metrics, cfg = compute_counts_six(pred, truth, targets)
    else:
        if control_path is not None:
            raise ValueError("--control-path is supported only with H5AD predictions; "
                             "GEARS PKL uses controls from --truth-path")
        pred, truth, targets, metadata = load_gears(pred_path, truth_path)
        protocol = "gears-lognorm-vcc2026-metrics"
        print(f"GEARS lognorm: {metadata['n_test_cells']} test cells, "
              f"{len(targets)} perturbations, {metadata['n_genes']} genes; CPU/pdex", flush=True)
        if metadata["negative_predictions_clipped_to_zero"]:
            print(f"Clipped {metadata['negative_predictions_clipped_to_zero']} negative "
                  "predicted values to zero in log1p space.", flush=True)
        metrics, cfg = compute_six(pred, truth, targets)
    summary = {
        "protocol": protocol, "score_type": "raw_metrics",
        "version": "0.5.0", "cell_eval2_version": cell_eval_version,
        **metadata, "config": asdict(cfg), "metrics": metrics,
        "target_exclusion": "PDS: all panel targets; expression and DE: each perturbation's component genes",
        "metric_status": {key: "ok" if value is not None else "undefined"
                          for key, value in metrics.items()},
    }
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow(metrics)
    for name, value in metrics.items():
        print(f"{name}: {value:.8g}" if value is not None else f"{name}: undefined", flush=True)
    print(f"Saved: {output / 'summary.json'}", flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compute six VCC2026 raw metrics from H5AD counts with independent real "
                    "controls, or GEARS lognorm predictions (CPU).")
    parser.add_argument("--pred-path", required=True, type=Path,
                        help="Predicted counts H5AD; also accepts trusted GEARS test_res.pkl or its directory")
    parser.add_argument("--truth-path", required=True, type=Path,
                        help="True perturbation counts H5AD; for GEARS, processed H5AD or data directory")
    parser.add_argument("--control-path", type=Path,
                        help="Real control counts H5AD (required for H5AD predictions)")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory for summary.csv and summary.json")
    args = parser.parse_args(argv)
    try:
        evaluate(args.pred_path, args.truth_path, args.output_dir, control_path=args.control_path)
    except (ValueError, FileNotFoundError, KeyError, OSError, ImportError,
            pickle.UnpicklingError, EOFError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
