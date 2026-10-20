"""Export per-cell GEARS results as prediction, test truth and control H5AD files."""

import argparse
from pathlib import Path
import pickle

import anndata as ad
import numpy as np
import pandas as pd

from evaluate.data import CONTROL, _check_matrix, _labels, gene_names, is_control


def export_results(results, source, output_dir, reference_dir):
    """Keep full perturbation identities and use controls from processed GEARS data."""
    pred = np.asarray(results['pred'], dtype=np.float32)
    truth = np.asarray(results['truth'], dtype=np.float32)
    labels = _labels(results['pert_cat'], 'GEARS perturbations')
    genes = gene_names(source)
    if pred.shape != truth.shape or pred.shape != (len(labels), len(genes)):
        raise ValueError('GEARS pred/truth must have matching cells-by-reference-genes shape')
    _check_matrix(pred, 'GEARS predictions', nonnegative=False)
    _check_matrix(truth, 'GEARS truth')
    declared = results.get('gene_names')
    if declared is not None:
        declared = _labels(declared, 'GEARS gene_names').tolist()
        lookup = {gene.upper(): i for i, gene in enumerate(declared)}
        if len(lookup) != len(genes) or set(lookup) != {gene.upper() for gene in genes}:
            raise ValueError('GEARS gene_names must match the processed reference gene panel')
        columns = [lookup[gene.upper()] for gene in genes]
        pred, truth = pred[:, columns], truth[:, columns]
    keep = np.asarray([not is_control(label) for label in labels])
    if not keep.any() or 'condition' not in source.obs:
        raise ValueError('GEARS export requires test perturbations and source obs.condition')
    real_labels = _labels(source.obs.condition, 'GEARS source conditions')
    control_rows = np.flatnonzero([is_control(label) for label in real_labels])
    if not len(control_rows):
        raise ValueError('GEARS source has no real control cells')
    pred, truth, labels = pred[keep], truth[keep], labels[keep]
    negative = int(np.count_nonzero(pred < 0))
    pred = np.maximum(pred, 0)
    controls = source.X[control_rows]
    _check_matrix(controls, 'GEARS real controls')
    context = None
    if 'cell_type' in source.obs:
        contexts = set(_labels(source.obs.cell_type, 'GEARS cell_type'))
        if len(contexts) != 1:
            raise ValueError('Prepare a single GEARS cell context before export')
        context = next(iter(contexts))
    output, reference = Path(output_dir).resolve(), Path(reference_dir).resolve()
    if output.is_relative_to(reference) or reference.is_relative_to(output):
        raise ValueError('Prediction and dataset reference directories must be separate')
    paths = [output / 'prediction.h5ad', reference / 'truth.h5ad', reference / 'control.h5ad']
    if any(path.exists() for path in paths):
        raise ValueError('H5AD export already exists; use new output/reference directories')
    matrices = [pred, truth, controls]
    all_labels = [labels, labels, np.repeat(CONTROL, len(control_rows))]
    temporary = []
    try:
        for role, path, matrix, conditions in zip(['pred', 'truth', 'control'], paths, matrices, all_labels):
            path.parent.mkdir(parents=True, exist_ok=True)
            obs = pd.DataFrame({'target_gene': conditions},
                               index=[f'{role}_{i}' for i in range(len(conditions))])
            if context is not None:
                obs['cell_type'] = context
            data = ad.AnnData(matrix, obs=obs, var=pd.DataFrame(index=genes))
            data.uns['evaluation'] = {'expression_scale': 'lognorm', 'producer': 'GEARS',
                                      'gene_order': 'declared' if declared is not None else 'processed-reference-order'}
            if role == 'pred':
                data.uns['evaluation']['negative_values_clipped'] = negative
            temp = path.with_suffix('.tmp.h5ad')
            temporary.append(temp)
            data.write_h5ad(temp, compression='gzip')
        for temp, path in zip(temporary, paths):
            temp.replace(path)
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
    return dict(zip(['prediction', 'truth', 'control'], map(str, paths)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pred-path', type=Path, required=True, help='Trusted legacy GEARS test_res.pkl')
    parser.add_argument('--data-path', type=Path, required=True, help='GEARS perturb_processed.h5ad (lognorm X)')
    parser.add_argument('--output-dir', type=Path, default=Path('/data/xiebo/outputs/gears'))
    parser.add_argument('--reference-dir', type=Path, default=Path('/data/dataset/gears/evaluation'))
    args = parser.parse_args(argv)
    with args.pred_path.open('rb') as handle:
        results = pickle.load(handle)
    source = ad.read_h5ad(args.data_path, backed='r')
    try:
        paths = export_results(results, source, args.output_dir, args.reference_dir)
    finally:
        source.file.close()
    for role, path in paths.items():
        print(f'{role}: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
