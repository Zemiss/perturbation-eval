"""Score several predictions on one shared truth/control panel and compare them."""

import argparse
import csv
import json
from pathlib import Path

from evaluate.metrics import METRICS

LOWER_IS_BETTER = {"expr_mse_unbiased_capped_norm", "de_wilcoxon_lfc_nmae"}


def _parse_prediction(value):
    if "=" not in value:
        raise ValueError("Each --pred must use NAME=PATH")
    name, path = value.split("=", 1)
    name, path = name.strip(), Path(path).expanduser()
    if not name or any(character in name for character in "/\\ "):
        raise ValueError(f"Invalid model name: {name!r}")
    if not path.is_file():
        raise ValueError(f"Prediction does not exist: {path}")
    return name, path.resolve()


def compare(predictions, truth_path, control_path, output_dir):
    from evaluate.run import evaluate

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries, panels = {}, set()
    for name, path in predictions.items():
        summary = evaluate(path, truth_path, control_path, output / name)
        summaries[name] = summary
        panels.add(summary["gene_panel_sha256"])
    if len(panels) != 1:
        raise ValueError("Models were scored on different gene panels")

    rows = []
    for name, summary in summaries.items():
        rows.append({"model": name, **{metric: summary["metrics"][metric] for metric in METRICS}})

    ranks = {name: {} for name in summaries}
    for metric in METRICS:
        scored = [(name, summary["metrics"][metric]) for name, summary in summaries.items()
                  if summary["metrics"][metric] is not None]
        reverse = metric not in LOWER_IS_BETTER
        scored.sort(key=lambda item: item[1], reverse=reverse)
        for rank, (name, _) in enumerate(scored, start=1):
            ranks[name][metric] = rank
    for row in rows:
        values = [ranks[row["model"]][metric] for metric in METRICS
                  if ranks[row["model"]].get(metric) is not None]
        row["average_rank"] = sum(values) / len(values) if values else None
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", *METRICS, "average_rank"])
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "models": list(predictions),
        "metrics": list(METRICS),
        "lower_is_better": sorted(LOWER_IS_BETTER),
        "truth_path": str(Path(truth_path).resolve()),
        "control_path": str(Path(control_path).resolve()),
        "gene_panel_sha256": next(iter(panels)),
        "n_genes": next(iter(summaries.values()))["n_genes"],
        "n_real_test_cells": next(iter(summaries.values()))["n_real_test_cells"],
        "n_control_cells": next(iter(summaries.values()))["n_control_cells"],
        "summaries": {name: {"metrics": summary["metrics"], "n_pred_cells": summary["n_pred_cells"],
                             "path": str(output / name / "summary.json")}
                      for name, summary in summaries.items()},
        "ranks": ranks,
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    lines = ["| model | " + " | ".join(METRICS) + " | average rank |",
             "| --- | " + " | ".join(["---"] * len(METRICS)) + " | --- |"]
    for row in rows:
        values = ["null" if row[metric] is None else f"{row[metric]:.8g}" for metric in METRICS]
        average = "null" if row["average_rank"] is None else f"{row['average_rank']:.3f}"
        lines.append(f"| {row['model']} | " + " | ".join(values) + f" | {average} |")
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--truth-path", type=Path, required=True)
    parser.add_argument("--control-path", type=Path, required=True)
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=PATH",
                        help="Prediction H5AD; repeat for each model")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    predictions = {}
    for value in args.pred:
        name, path = _parse_prediction(value)
        if name in predictions:
            parser.error(f"duplicate model name: {name}")
        predictions[name] = path
    compare(predictions, args.truth_path, args.control_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
