"""Run a saved official GEARS model and directly export per-cell H5AD results."""

import argparse
from pathlib import Path
import random
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gears-repo', type=Path, default=Path('/data/xiebo/GEARS'))
    parser.add_argument('--data-dir', type=Path, default=Path('/data/dataset/gears'))
    parser.add_argument('--data-name', default='norman')
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--split', default='simulation')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--test-batch-size', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output-dir', type=Path, default=Path('/data/xiebo/outputs/gears'))
    parser.add_argument('--reference-dir', type=Path, default=Path('/data/dataset/gears/evaluation'))
    parser.add_argument('--run-dir', type=Path, default=Path('/data/xiebo/runs/gears/infer'))
    args = parser.parse_args(argv)
    output, reference, run = [path.resolve() for path in [args.output_dir, args.reference_dir, args.run_dir]]
    if any(a.is_relative_to(b) or b.is_relative_to(a)
           for a, b in [(output, reference), (output, run), (reference, run)]):
        parser.error('Prediction, dataset reference and run directories must be separate')
    if any(path.exists() and any(path.iterdir()) for path in [output, reference, run]):
        parser.error('Use empty output/reference/run directories for a new inference run')
    if args.batch_size < 1 or args.test_batch_size < 1 or args.seed < 0:
        parser.error('Batch sizes must be positive and seed nonnegative')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.gears_repo.resolve()))
    import numpy as np
    import torch
    from gears import GEARS, PertData
    from gears.inference import evaluate
    from baseline.common import git_revision, write_json
    from baseline.gears.export import export_results

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    data = PertData(str(args.data_dir.resolve()))
    data.load(data_name=args.data_name)
    data.prepare_split(split=args.split, seed=args.seed)
    data.get_dataloader(batch_size=args.batch_size, test_batch_size=args.test_batch_size)
    model = GEARS(data, device=args.device, weight_bias_track=False)
    model.load_pretrained(str(args.model_dir.resolve()))
    results = evaluate(data.dataloader['test_loader'], model.best_model,
                       model.config['uncertainty'], args.device)
    results['gene_names'] = data.adata.var['gene_name'].to_numpy(dtype=str)
    paths = export_results(results, data.adata, output, reference)
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / 'run.json', {**{key: str(value) if isinstance(value, Path) else value
                                    for key, value in vars(args).items()},
                                  'gears_revision': git_revision(args.gears_repo), **paths})
    for role, path in paths.items():
        print(f'{role}: {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
