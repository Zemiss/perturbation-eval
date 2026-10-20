"""Prepare one shared VCC truth/control panel before four-path evaluation."""

import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from baseline.common import LINGSHU_REPO, gene_symbols
from evaluate.data import CONTROL, gene_names, is_control, open_source, perturbation_labels


def prepare_reference(pred_path, truth_path, control_path, output_dir, gene_map, *, layer=None):
    """Fix genes and test labels once; the resulting files are shared across models."""
    output = Path(output_dir).resolve()
    paths = [output / 'truth.h5ad', output / 'control.h5ad']
    if any(path.exists() for path in paths):
        raise ValueError('Reference files already exist; use a new reference directory')
    mapping = pd.read_csv(gene_map, sep='\t', header=None, names=['id', 'symbol'])
    names = dict(zip(mapping.id, mapping.symbol))
    with ExitStack() as stack:
        prediction = open_source(pred_path, 'pred')
        stack.callback(prediction.adata.file.close)
        panel = set(prediction.labels)
        sources = []
        for role, path in [('truth', truth_path), ('control', control_path)]:
            source = ad.read_h5ad(path, backed='r')
            stack.callback(source.file.close)
            genes = gene_symbols(source, gene_map)
            labels = perturbation_labels(source, role, control_only=role == 'control')
            labels = np.asarray(['+'.join(names.get(part, part) for part in label.split('+'))
                                 for label in labels])
            labels = np.asarray([CONTROL if is_control(label) else label for label in labels])
            keep = np.isin(labels, list(panel)) if role == 'truth' else labels == CONTROL
            if prediction.context is not None and 'cell_type' in source.obs:
                keep &= source.obs.cell_type.astype(str).eq(prediction.context).to_numpy()
            rows = np.flatnonzero(keep)
            if not len(rows) or (role == 'truth' and set(labels[rows]) != panel):
                raise ValueError(f'{role}: missing controls or test perturbations from the prediction panel')
            metadata = source.uns.get('evaluation', {})
            selected_layer = layer or metadata.get('layer')
            scale = 'counts' if layer else metadata.get('expression_scale', source.uns.get('vcc', {}).get('expression_scale', 'counts'))
            if scale not in {'counts', 'lognorm'}:
                raise ValueError('Reference expression scale must be counts or lognorm')
            if selected_layer is not None and selected_layer not in source.layers:
                raise ValueError(f'Missing declared layer: {selected_layer}')
            sources.append((source, genes, rows, labels[rows], selected_layer, scale))
        shared = {gene.upper() for gene in gene_names(prediction.adata)}
        for _, genes, *_ in sources:
            shared &= {gene.upper() for gene in genes}
        genes = [gene for gene in sources[0][1] if gene.upper() in shared]
        if not genes:
            raise ValueError('No common expression genes')
        output.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.prepare-', dir=output) as work:
            pending = []
            for path, (source, source_genes, rows, labels, selected_layer, scale) in zip(paths, sources):
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
                if 'cell_type' in source.obs:
                    obs['cell_type'] = source.obs.iloc[rows].cell_type.to_numpy()
                data = ad.AnnData(np.asarray(expression), obs=obs, var=pd.DataFrame(index=genes))
                data.uns['evaluation'] = {'expression_scale': scale, 'source_path': str(Path(source.filename).resolve()),
                                          'panel_prediction': str(Path(pred_path).resolve())}
                temp = Path(work) / path.name
                data.write_h5ad(temp, compression='gzip')
                pending.append((temp, path))
                del data, expression
            for temp, path in pending:
                temp.replace(path)
    return paths


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-path', type=Path, required=True, help='Chreode H5AD defining the shared VCC panel')
    parser.add_argument('--truth-path', type=Path, required=True)
    parser.add_argument('--control-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True, help='Prepared dataset reference directory')
    parser.add_argument('--gene-map', type=Path, default=LINGSHU_REPO / 'data/vcc/gene_id_name.csv')
    parser.add_argument('--layer', help='Explicit raw counts layer in both real files, if needed')
    args = parser.parse_args(argv)
    for path in prepare_reference(args.pred_path, args.truth_path, args.control_path,
                                   args.output_dir, args.gene_map, layer=args.layer):
        print(f'Saved: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
