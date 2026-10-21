"""Prepare one shared VCC truth/control panel before four-path evaluation.

The panel can be defined by a model prediction (legacy behaviour), by a gene list,
by a Chreode native vocabulary, or by any intersection of them. Truth and control
are written once and reused by every model, so Chreode and Lingshu are scored
on exactly the same genes and cells.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from baseline.common import LINGSHU_REPO, gene_symbols, native_genes
from evaluate.data import (CONTROL, LABEL_COLUMNS, gene_names, is_control,
                           open_source, perturbation_labels)


def _normalize_label(label):
    if is_control(label):
        return CONTROL
    parts = [part for part in str(label).split('+') if part and not is_control(part)]
    return '+'.join(parts) if parts else CONTROL


def _read_gene_list(path):
    path = Path(path)
    if path.suffix.lower() == ".h5ad":
        source = ad.read_h5ad(path, backed="r")
        try:
            return gene_names(source)
        finally:
            source.file.close()
    table = pd.read_csv(path, header=None)
    if table.empty:
        raise ValueError(f"Gene list is empty: {path}")
    values = table.iloc[:, 0].astype(str).str.strip().tolist()
    if values and values[0].casefold() in {"gene", "gene_name", "genes", "gene_symbol", "symbol"}:
        values = values[1:]
    genes = [value for value in values if value]
    if not genes:
        raise ValueError(f"Gene list is empty: {path}")
    return genes


def _unique_upper(values):
    seen, ordered = set(), []
    for value in values:
        key = value.upper()
        if key not in seen:
            seen.add(key)
            ordered.append(value)
    return ordered


def _panel_genes(pred_path, panel_path, panel_vocab):
    sources = []
    if pred_path is not None:
        source = open_source(pred_path, "pred")
        try:
            sources.append((source.genes, source.labels, source.context))
        finally:
            source.adata.file.close()
    if panel_path is not None:
        panel_path = Path(panel_path)
        if panel_path.suffix.lower() == ".h5ad":
            source = ad.read_h5ad(panel_path, backed="r")
            try:
                labels = None
                columns = [key for key in LABEL_COLUMNS if key in source.obs]
                if columns:
                    labels = perturbation_labels(source, "panel")
                context = None
                if "cell_type" in source.obs:
                    contexts = set(source.obs.cell_type.astype(str))
                    if len(contexts) == 1:
                        context = next(iter(contexts))
                sources.append((gene_names(source), labels, context))
            finally:
                source.file.close()
        else:
            sources.append((_read_gene_list(panel_path), None, None))
    if panel_vocab is not None:
        sources.append((native_genes(panel_vocab), None, None))
    if not sources:
        raise ValueError("Provide --pred-path, --panel-path or --panel-vocab to define the gene panel")
    genes = _unique_upper(sources[0][0])
    allowed = {gene.upper() for gene in genes}
    labels = sources[0][1]
    context = sources[0][2]
    for other, other_labels, other_context in sources[1:]:
        allowed &= {gene.upper() for gene in other}
        if labels is None and other_labels is not None:
            labels = other_labels
        if context is None and other_context is not None:
            context = other_context
    genes = [gene for gene in genes if gene.upper() in allowed]
    if not genes:
        raise ValueError("Panel sources have no genes in common")
    if labels is not None:
        labels = np.asarray([label for label in labels if not is_control(label)])
        labels = labels if len(labels) else None
    return genes, labels, context


def _downsample(rows, labels, limit, control_limit, rng):
    if (limit <= 0 and control_limit <= 0) or not len(rows):
        return rows
    keep = []
    for label in pd.unique(labels[rows]):
        selected = rows[labels[rows] == label]
        cap = control_limit if is_control(label) else limit
        if cap > 0 and len(selected) > cap:
            selected = rng.choice(selected, size=cap, replace=False)
        keep.append(np.sort(selected))
    return np.sort(np.concatenate(keep))


def _conditions_from_labels(labels, context):
    values = pd.Series(labels, dtype="object")
    values = values[~values.map(is_control)]
    values = values.map(lambda value: value[:-5] if value.endswith("+ctrl") else value)
    counts = values.value_counts()
    table = pd.DataFrame({"cell_type": context or "H1", "target_gene": counts.index,
                          "n_cells": counts.to_numpy(dtype=np.int64)})
    return table.sort_values("target_gene", ignore_index=True)


def prepare_reference(pred_path, truth_path, control_path, output_dir, gene_map, *, layer=None,
                      panel_path=None, panel_vocab=None, max_cells_per_pert=0,
                      max_control_cells=0, conditions_out=None, seed=0):
    """Fix genes, perturbations and optional cell budgets once for all models."""
    output = Path(output_dir).resolve()
    paths = [output / 'truth.h5ad', output / 'control.h5ad']
    if any(path.exists() for path in paths):
        raise ValueError('Reference files already exist; use a new reference directory')
    panel, panel_labels, panel_context = _panel_genes(pred_path, panel_path, panel_vocab)
    mapping = pd.read_csv(gene_map, sep='\t', header=None, names=['id', 'symbol'])
    names = dict(zip(mapping.id.astype(str), mapping.symbol.astype(str)))
    panel = _unique_upper([names.get(gene, gene) for gene in panel])
    if panel_labels is not None:
        panel_labels = np.asarray([_normalize_label(names.get(label, label)) for label in panel_labels])
    rng = np.random.default_rng(seed)
    with ExitStack() as stack:
        sources = []
        for role, path in [('truth', truth_path), ('control', control_path)]:
            source = ad.read_h5ad(path, backed='r')
            stack.callback(source.file.close)
            genes = gene_symbols(source, gene_map)
            labels = perturbation_labels(source, role, control_only=role == "control")
            labels = np.asarray(['+'.join(names.get(part, part) for part in label.split('+'))
                                 for label in labels])
            labels = np.asarray([_normalize_label(label) for label in labels])
            if role == 'truth':
                if panel_labels is None:
                    keep = labels != CONTROL
                else:
                    keep = np.isin(labels, list(panel_labels))
            else:
                keep = labels == CONTROL
            context = panel_context
            if context is not None and 'cell_type' in source.obs:
                keep &= source.obs.cell_type.astype(str).eq(context).to_numpy()
            elif context is None and 'cell_type' in source.obs:
                contexts = set(source.obs.cell_type.astype(str))
                if len(contexts) == 1:
                    context = next(iter(contexts))
            rows = np.flatnonzero(keep)
            if role == 'truth':
                rows = _downsample(rows, labels, max_cells_per_pert, max_control_cells, rng)
            else:
                rows = _downsample(rows, labels, max_control_cells, max_control_cells, rng)
            if not len(rows):
                raise ValueError(f'{role}: no selected cells')
            if role == 'truth' and panel_labels is not None and set(labels[rows]) != set(panel_labels):
                raise ValueError(f'{role}: missing test perturbations from the prediction panel')
            metadata = source.uns.get('evaluation', {})
            selected_layer = layer or metadata.get('layer')
            scale = ('counts' if layer else
                     metadata.get('expression_scale', source.uns.get('vcc', {}).get('expression_scale', 'counts')))
            if scale not in {'counts', 'lognorm'}:
                raise ValueError('Reference expression scale must be counts or lognorm')
            if selected_layer is not None and selected_layer not in source.layers:
                raise ValueError(f'Missing declared layer: {selected_layer}')
            sources.append((source, genes, rows, labels[rows], selected_layer, scale, context))
        shared = {gene.upper() for gene in panel}
        for _, genes, *_ in sources:
            shared &= {gene.upper() for gene in genes}
        genes = [gene for gene in sources[0][1] if gene.upper() in shared]
        if not genes:
            raise ValueError('No common expression genes')
        output.mkdir(parents=True, exist_ok=True)
        context = panel_context or next((item[6] for item in sources if item[6] is not None), None)
        conditions = _conditions_from_labels(sources[0][3], context)
        if conditions.target_gene.duplicated().any() or conditions.target_gene.eq('').any():
            raise ValueError('Invalid perturbation labels after downsampling')
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output) as work:
            pending = []
            for path, (source, source_genes, rows, labels, selected_layer, scale, _) in zip(paths, sources):
                lookup = {gene.upper(): i for i, gene in enumerate(source_genes)}
                columns = [lookup[gene.upper()] for gene in genes]
                matrix = source.X if selected_layer is None else source.layers[selected_layer]
                expression = np.memmap(Path(work) / f'{path.stem}.float32', mode='w+',
                                        dtype=np.float32, shape=(len(rows), len(genes)))
                for start in range(0, len(rows), 256):
                    block = matrix[rows[start:start + 256]][:, columns]
                    block = block.toarray() if sparse.issparse(block) else np.asarray(block)
                    if not np.isfinite(block).all() or (block < 0).any():
                        raise ValueError('Reference expression must be finite and nonnegative')
                    if scale == 'counts' and (block != np.floor(block)).any():
                        raise ValueError('Reference counts contain fractional values; select the counts layer')
                    expression[start:start + len(block)] = block
                obs = pd.DataFrame({'target_gene': labels}, index=source.obs_names[rows].astype(str))
                if context is not None:
                    obs['cell_type'] = context
                data = ad.AnnData(np.asarray(expression), obs=obs, var=pd.DataFrame(index=genes))
                data.uns['evaluation'] = {'expression_scale': scale,
                                          'source_path': str(Path(source.filename).resolve()),
                                          'panel_genes': len(genes)}
                temp = Path(work) / path.name
                data.write_h5ad(temp, compression='gzip')
                pending.append((temp, path))
                del data, expression
            for temp, path in pending:
                temp.replace(path)
        (output / 'panel.txt').write_text('\n'.join(genes) + '\n', encoding='utf-8')
        if conditions_out is not None:
            conditions_out = Path(conditions_out).resolve()
            conditions_out.parent.mkdir(parents=True, exist_ok=True)
            conditions.to_csv(conditions_out, index=False)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-path', type=Path, help='Prediction used to define panel and test labels')
    parser.add_argument('--panel-path', type=Path, help='Gene list or H5AD defining the common panel')
    parser.add_argument('--panel-vocab', type=Path, help='Repository whose native gene vocabulary defines the panel')
    parser.add_argument('--truth-path', type=Path, required=True)
    parser.add_argument('--control-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Prepared dataset reference directory')
    parser.add_argument('--gene-map', type=Path, default=LINGSHU_REPO / 'data/vcc/gene_id_name.csv')
    parser.add_argument('--layer', help='Explicit raw counts layer in both real files, if needed')
    parser.add_argument('--max-cells-per-pert', type=int, default=0,
                        help='Downsample each truth perturbation; 0 keeps all cells')
    parser.add_argument('--max-control-cells', type=int, default=0,
                        help='Downsample control cells; 0 keeps all cells')
    parser.add_argument('--conditions-out', type=Path,
                        help='Write the shared per-perturbation cell counts for all models')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)
    if args.max_cells_per_pert < 0 or args.max_control_cells < 0 or args.seed < 0:
        parser.error('cell budgets and seed must be nonnegative')
    if not any([args.pred_path, args.panel_path, args.panel_vocab]):
        parser.error('provide --pred-path, --panel-path or --panel-vocab')
    for path in prepare_reference(args.pred_path, args.truth_path, args.control_path,
                                  args.output_dir, args.gene_map, layer=args.layer,
                                  panel_path=args.panel_path, panel_vocab=args.panel_vocab,
                                  max_cells_per_pert=args.max_cells_per_pert,
                                  max_control_cells=args.max_control_cells,
                                  conditions_out=args.conditions_out, seed=args.seed):
        print(f'Saved: {path}')
    if args.conditions_out is not None:
        print(f'Saved: {args.conditions_out.resolve()}')
    print(f'Saved: {(args.output_dir.resolve() / "panel.txt")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
