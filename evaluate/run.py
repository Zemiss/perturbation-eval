"""Evaluate predicted, real perturbed and real control H5AD files."""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile


def evaluate(pred_path, truth_path, control_path, output_dir):
    from cell_eval2 import __version__ as cell_eval_version
    import numpy as np
    from .data import CONTROL, attach, fill_normalized, is_control, open_source
    from .metrics import compute_six

    with ExitStack() as stack:
        sources = {}
        for role, path in [('pred', pred_path), ('truth', truth_path), ('control', control_path)]:
            sources[role] = open_source(path, role)
            stack.callback(sources[role].adata.file.close)
        pred, truth, control = [sources[role] for role in ['pred', 'truth', 'control']]
        if set(pred.labels) != set(truth.labels):
            raise ValueError('Prediction and real perturbation panels must match exactly')
        contexts = {source.context for source in sources.values() if source.context is not None}
        if len(contexts) > 1:
            raise ValueError('Prediction, truth and controls belong to different cell contexts')
        genes = truth.genes
        for source in [pred, control]:
            missing = set(g.upper() for g in genes) - set(g.upper() for g in source.genes)
            if missing:
                raise ValueError(f'{source.path.name}: missing {len(missing)} reference genes; '
                                 'prepare a fixed common reference panel first')
        lookup = {gene.upper(): gene for gene in genes}
        targets = {}
        for label in sorted(set(truth.labels)):
            parts = label.split('+')
            if any(not part.strip() for part in parts):
                raise ValueError(f'Invalid perturbation label: {label}')
            components = sorted({part.strip() for part in parts if not is_control(part.strip())})
            if not components:
                raise ValueError(f'Invalid perturbation label: {label}')
            targets[label] = [lookup.get(gene.upper(), gene) for gene in components]
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        counts = {role: len(source.rows) for role, source in sources.items()}
        hashes = {}
        with tempfile.TemporaryDirectory(prefix='.evaluation-', dir=output) as work:
            matrices = {}
            for role in ['truth', 'pred']:
                matrices[role] = np.memmap(Path(work) / f'{role}.float32', mode='w+', dtype=np.float32,
                                           shape=(counts[role] + counts['control'], len(genes)))
                hashes[role] = fill_normalized(sources[role], genes, matrices[role][:counts[role]])
            hashes['control'] = fill_normalized(control, genes, matrices['truth'][counts['truth']:])
            for start in range(0, counts['control'], 256):
                matrices['pred'][counts['pred'] + start:counts['pred'] + start + 256] = matrices['truth'][counts['truth'] + start:counts['truth'] + start + 256]
            pred_ad = attach(matrices['pred'], np.append(pred.labels, [CONTROL] * counts['control']), genes)
            truth_ad = attach(matrices['truth'], np.append(truth.labels, [CONTROL] * counts['control']), genes)
            print(f"H5AD: {counts['pred']} predicted, {counts['truth']} real perturbed, "
                  f"{counts['control']} control cells; {len(genes)} genes; CPU/pdex", flush=True)
            values, cfg = compute_six(pred_ad, truth_ad, targets)
            del pred_ad, truth_ad, matrices
        summary = {'protocol': 'h5ad-reference-panel-lognorm-v1', 'score_type': 'raw_metrics',
                   'version': '0.5.0', 'cell_eval2_version': cell_eval_version,
                   'prediction': str(pred.path), 'truth_data': str(truth.path), 'control_data': str(control.path),
                   'n_pred_cells': counts['pred'], 'n_real_test_cells': counts['truth'],
                   'n_control_cells': counts['control'], 'n_genes': len(genes), 'gene_names': genes,
                   'gene_panel_sha256': hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
                   'perturbations': sorted(targets), 'target_genes': targets,
                   'unmeasured_target_genes': sorted({g for ts in targets.values() for g in ts if g not in genes}),
                   'input_types': {role: source.scale for role, source in sources.items()},
                   'input_layers': {role: source.layer for role, source in sources.items()},
                   'normalized_expression_sha256': hashes,
                   'preprocessing': 'reference gene panel -> linear abundance -> normalize_total(10000) -> log1p',
                   'lognorm_input': 'expm1 produces linear abundance, not raw counts',
                   'config': asdict(cfg), 'metrics': values,
                   'metric_status': {name: 'ok' if value is not None else 'undefined' for name, value in values.items()}}
        return write_summary(summary, output)


def write_summary(summary, output_dir):
    metrics = summary["metrics"]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-path', required=True, type=Path, help='Predicted perturbation H5AD')
    parser.add_argument('--truth-path', required=True, type=Path, help='Real perturbed-cell H5AD')
    parser.add_argument('--control-path', required=True, type=Path, help='Real control-cell H5AD')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory for summary.csv/json')
    args = parser.parse_args(argv)
    try:
        evaluate(args.pred_path, args.truth_path, args.control_path, args.output_dir)
    except (ValueError, OSError, KeyError, ImportError, FloatingPointError) as exc:
        parser.exit(2, f'Evaluation error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
