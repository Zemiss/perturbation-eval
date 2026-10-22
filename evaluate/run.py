"""Evaluate a prediction H5AD against real test cells with six cell-eval2 metrics."""

import argparse
from contextlib import ExitStack
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import tempfile

DEFAULT_DATA_ROOT = Path('/data/dataset/lingshucell/vcc')
LINGSHU_DATA = Path(__file__).resolve().parents[2] / 'Lingshu-Cell' / 'data'
DEFAULT_GENE_MAP = LINGSHU_DATA / 'vcc/gene_id_name.csv'
DEFAULT_GENE_PANEL = LINGSHU_DATA / 'vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json'


def evaluate(pred_path, truth_path=None, control_path=None, output_dir=None, *, data_root=None, protocol="lognorm",
             pert_col=None, control_label=None, gene_map=None, gene_panel=None,
             pred_input_type=None, truth_input_type=None, control_input_type=None,
             pred_layer=None, truth_layer=None, control_layer=None, num_threads=-1):
    from cell_eval2 import __version__ as cell_eval_version
    import numpy as np
    from .data import CONTROL, attach, fill_expression, load_gene_map, open_source, reference_genes
    from .metrics import compute_counts_six, compute_six

    if output_dir is None:
        raise ValueError("output_dir is required")
    if protocol not in {"vcc2026", "lognorm"}:
        raise ValueError("protocol must be vcc2026 or lognorm")
    if num_threads == 0 or num_threads < -1:
        raise ValueError("num_threads must be -1 (all CPUs) or a positive integer")
    dataset_truth = truth_path is None
    dataset_control = control_path is None
    root = Path(data_root or DEFAULT_DATA_ROOT).expanduser().resolve()
    truth_path = truth_path if not dataset_truth else root / 'test/adata_Test.h5ad'
    control_path = control_path if not dataset_control else root / 'train/adata_Training.h5ad'
    if dataset_truth:
        # The current Lingshu and GEARS VCC exporters share this fixed 18,080-gene vocabulary.
        # Read annotation files only; never import or run either model repository.
        if gene_map is None and DEFAULT_GENE_MAP.is_file():
            gene_map = DEFAULT_GENE_MAP
        if gene_panel is None:
            if DEFAULT_GENE_PANEL.is_file():
                gene_panel = DEFAULT_GENE_PANEL
            elif (root / 'gene_names.csv').is_file():
                gene_panel = root / 'gene_names.csv'
    normalize = protocol == "lognorm"
    mapping = load_gene_map(gene_map)

    with ExitStack() as stack:
        sources = {}
        options = {
            'pred': (pred_input_type, pred_layer), 'truth': (truth_input_type, truth_layer),
            'control': (control_input_type, control_layer),
        }
        for role, path in [('pred', pred_path), ('truth', truth_path), ('control', control_path)]:
            scale, layer = options[role]
            sources[role] = open_source(path, role, pert_col=pert_col, control_label=control_label,
                                        gene_map=mapping, input_type=scale, layer=layer,
                                        allow_unlabeled_controls=not dataset_control)
            stack.callback(sources[role].adata.file.close)
        pred, truth, control = [sources[role] for role in ['pred', 'truth', 'control']]
        if set(pred.labels) != set(truth.labels):
            missing = sorted(set(truth.labels) - set(pred.labels))
            extra = sorted(set(pred.labels) - set(truth.labels))
            raise ValueError('Prediction and real perturbation panels must match exactly; '
                             f'missing predictions: {missing[:10]}; absent from truth: {extra[:10]}')
        contexts = {source.context for source in sources.values() if source.context is not None}
        if len(contexts) > 1:
            raise ValueError('Prediction, truth and controls belong to different cell contexts')
        genes = reference_genes(truth, gene_panel, mapping)
        for source in [pred, control]:
            missing = set(g.upper() for g in genes) - set(g.upper() for g in source.genes)
            if missing:
                raise ValueError(f'{source.path.name}: missing {len(missing)} reference genes; '
                                 'use --gene-panel to specify a fixed common reference panel')
        lookup = {gene.upper(): gene for gene in genes}
        targets = {}
        for label in sorted(set(truth.labels)):
            parts = label.split('+')
            components = sorted(set(parts))
            targets[label] = [lookup.get(gene.upper(), gene) for gene in components]
        if not any(gene in genes for ts in targets.values() for gene in ts):
            raise ValueError('No perturbation target matches a measured gene; provide --gene-map '
                             'or gene symbols in var.gene_name and the perturbation labels')
        if not normalize and any(len(ts) != 1 for ts in targets.values()):
            raise ValueError('The cell-eval2 vcc2026 preset supports single-gene perturbations; '
                             'use --protocol lognorm for the multi-gene extension')
        if not normalize and any(source.scale != 'counts' for source in sources.values()):
            raise ValueError('vcc2026 requires raw counts in every input; use --protocol lognorm '
                             'for log-normalized predictions and all models compared with them')
        output = Path(output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        counts = {role: len(source.rows) for role, source in sources.items()}
        hashes = {}
        with tempfile.TemporaryDirectory(prefix='.evaluation-', dir=output) as work:
            matrices = {}
            for role in ['truth', 'pred']:
                matrices[role] = np.memmap(Path(work) / f'{role}.float32', mode='w+', dtype=np.float32,
                                           shape=(counts[role] + counts['control'], len(genes)))
                hashes[role] = fill_expression(sources[role], genes, matrices[role][:counts[role]],
                                               normalize=normalize)
            hashes['control'] = fill_expression(control, genes, matrices['truth'][counts['truth']:],
                                                normalize=normalize)
            # Read the real control pool once, then append these identical rows to both sides.
            for start in range(0, counts['control'], 256):
                matrices['pred'][counts['pred'] + start:counts['pred'] + start + 256] = matrices['truth'][counts['truth'] + start:counts['truth'] + start + 256]
            pred_ad = attach(matrices['pred'], np.append(pred.labels, [CONTROL] * counts['control']), genes)
            truth_ad = attach(matrices['truth'], np.append(truth.labels, [CONTROL] * counts['control']), genes)
            print(f"H5AD: {counts['pred']} predicted, {counts['truth']} real perturbed, "
                  f"{counts['control']} real control cells; {len(genes)} genes; {protocol}; CPU/pdex", flush=True)
            compute = compute_six if normalize else compute_counts_six
            values, cfg = compute(pred_ad, truth_ad, targets, num_threads=num_threads)
            del pred_ad, truth_ad, matrices
        summary = {'protocol': 'h5ad-reference-panel-lognorm-v1' if normalize else 'h5ad-vcc2026-counts-v1',
                   'score_type': 'raw_metrics', 'version': '0.6.0', 'cell_eval2_version': cell_eval_version,
                   'prediction': str(pred.path), 'truth_data': str(truth.path), 'control_data': str(control.path),
                   'control_source': 'dataset_train' if dataset_control else 'separate_h5ad',
                   'data_root': str(root) if dataset_truth or dataset_control else None,
                   'ignored_prediction_controls': pred.adata.n_obs - counts['pred'],
                   'ignored_truth_controls': truth.adata.n_obs - counts['truth'],
                   'n_pred_cells': counts['pred'], 'n_real_test_cells': counts['truth'],
                   'n_control_cells': counts['control'], 'n_genes': len(genes), 'gene_names': genes,
                   'gene_panel_sha256': hashlib.sha256('\n'.join(genes).encode()).hexdigest(),
                   'perturbations': sorted(targets), 'target_genes': targets,
                   'unmeasured_target_genes': sorted({g for ts in targets.values() for g in ts if g not in genes}),
                   'input_types': {role: source.scale for role, source in sources.items()},
                   'input_layers': {role: source.layer for role, source in sources.items()},
                   'label_options': {'pert_col': pert_col, 'control_label': control_label},
                   'expression_sha256': hashes,
                   'gene_map': str(Path(gene_map).expanduser().resolve()) if gene_map else None,
                   'gene_panel': str(Path(gene_panel).expanduser().resolve()) if gene_panel else None,
                   'preprocessing': ('reference gene panel -> linear abundance -> normalize_total(10000) -> log1p'
                                     if normalize else 'reference gene panel -> raw counts -> cell-eval2 vcc2026 preset'),
                   'config': asdict(cfg), 'metrics': values,
                   'metric_status': {name: 'ok' if value is not None else 'undefined' for name, value in values.items()}}
        if normalize:
            summary['normalized_expression_sha256'] = hashes
            summary['lognorm_input'] = 'expm1 produces linear abundance, not raw counts'
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
    parser.add_argument('--data-root', type=Path,
                        help=f'VCC dataset root (default: {DEFAULT_DATA_ROOT}); read test truth and training controls')
    parser.add_argument('--truth-path', type=Path, help='Override dataset test/adata_Test.h5ad')
    parser.add_argument('--control-path', type=Path, help='Override dataset train/adata_Training.h5ad control source')
    parser.add_argument('--output-dir', required=True, type=Path, help='Output directory for summary.csv/json')
    parser.add_argument('--protocol', choices=['vcc2026', 'lognorm'], default='lognorm',
                        help='lognorm: shared counts/lognorm comparison (default); vcc2026: raw counts only')
    parser.add_argument('--pert-col', help='Perturbation obs column; default: auto-detect standard columns')
    parser.add_argument('--control-label', help='Additional real control label; default: standard control aliases')
    parser.add_argument('--gene-map', type=Path, help='Gene ID to symbol TSV; dataset defaults use the adjacent Lingshu mapping')
    parser.add_argument('--gene-panel', type=Path,
                        help='Fixed gene list or Lingshu vocabulary JSON; dataset defaults use the shared VCC vocabulary')
    parser.add_argument('--num-threads', type=int, default=-1, help='CPU threads, -1 for all (default)')
    for role in ['pred', 'truth', 'control']:
        parser.add_argument(f'--{role}-input-type', choices=['counts', 'lognorm'],
                            help=f'{role} expression scale; default: H5AD metadata, otherwise counts')
        parser.add_argument(f'--{role}-layer', help=f'{role} expression layer; default: H5AD metadata, otherwise X')
    args = parser.parse_args(argv)
    try:
        evaluate(**vars(args))
    except (ValueError, OSError, KeyError, ImportError, FloatingPointError) as exc:
        parser.exit(2, f'Evaluation error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
