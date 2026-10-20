"""Predict the Lingshu VCC condition panel using an adapted Chreode checkpoint."""

import argparse
from pathlib import Path

import numpy as np

from baseline.common import (CHREODE_REPO, DATA_ROOT, LINGSHU_REPO, encode_h1,
                             fingerprint, load_conditions, write_json, write_prediction)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=CHREODE_REPO)
    parser.add_argument("--lingshu-repo", type=Path, default=LINGSHU_REPO)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--control-data", required=True, type=Path)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--release-dir", type=Path, default=Path("/data/xiebo/models/chreode/pretrained"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/xiebo/models/chreode/vcc/model.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("/data/xiebo/outputs/chreode"))
    parser.add_argument("--run-dir", type=Path, default=Path("/data/xiebo/runs/chreode/vcc_infer"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.seed < 0:
        parser.error("batch-size must be positive and seed nonnegative")
    args.repo, args.lingshu_repo = args.repo.resolve(), args.lingshu_repo.resolve()
    output, release_dir = args.output_dir.resolve(), args.release_dir.resolve()
    run_dir = args.run_dir.resolve()
    if output.is_relative_to(run_dir) or run_dir.is_relative_to(output):
        raise ValueError("--run-dir and --output-dir must be separate directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output exists; use a new --output-dir for each prediction run")
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Run directory exists; use a new --run-dir for each prediction run")
    conditions_path = (args.conditions or args.lingshu_repo / "data/vcc/pert_counts_Test.csv").resolve()
    conditions = load_conditions(conditions_path)
    import torch
    from baseline.chreode.model import enable_repository, load_adapted, sample_population
    enable_repository(args.repo)
    torch.manual_seed(args.seed)
    backbone, model, config = load_adapted(args.checkpoint.resolve(), release_dir, args.device)
    if fingerprint(conditions_path) != config["conditions_sha256"]:
        raise ValueError("Inference conditions differ from the test panel held out during training")
    control_path = (args.control_data if args.control_data.is_absolute()
                    else args.data_root / args.control_data).resolve()
    controls, _, _ = encode_h1(control_path, backbone, config["expression_genes"],
                               args.lingshu_repo / "data/vcc/gene_id_name.csv",
                               condition_col=config["condition_col"], cell_type_col=config["cell_type_col"],
                               layer=config["layer"], batch_size=args.batch_size, controls_only=True)
    action_ids = {name: i for i, name in enumerate(config["action_genes"])}
    native = {name.upper(): i for i, name in enumerate(config["expression_genes"]) if name}
    genes = [name for name in config["action_genes"] if name.upper() in native]
    columns = [native[name.upper()] for name in genes]
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch = run_dir / ".prediction.float32"
    expression = np.memmap(scratch, mode="w+", dtype=np.float32,
                           shape=(int(conditions.n_cells.sum()), len(genes)))
    labels, offset = [], 0
    for row in conditions.itertuples(index=False):
        for start in range(0, row.n_cells, args.batch_size):
            count = min(args.batch_size, row.n_cells - start)
            selected = rng.integers(len(controls), size=count)
            with torch.no_grad():
                z = sample_population(model, controls[selected], [action_ids[row.target_gene]] * count,
                                      k_samples=1, delta=config["delta"], generator=generator)
                decoded = backbone.decode(z)[:, columns].detach().cpu().numpy()
            expression[offset:offset + count] = decoded
            labels.extend([row.target_gene] * count)
            offset += count
        print(f"Generated {row.target_gene}: {row.n_cells} cells", flush=True)
    missing_targets = sorted(set(conditions.target_gene) - set(genes))
    metadata = {"model": "chreode", "method": config["method"], "expression_scale": "lognorm",
                "normalization_target_sum": 10000, "normalization_gene_axis": "full-native-vocabulary",
                "conditions_sha256": fingerprint(conditions_path), "seed": args.seed,
                "target_genes_absent_from_expression": missing_targets}
    output.mkdir(parents=True, exist_ok=True)
    negative = write_prediction(output / "prediction.h5ad", expression, labels, genes, metadata)
    del expression
    scratch.unlink()
    write_json(run_dir / "run.json", {**metadata, "checkpoint": str(args.checkpoint.resolve()),
                                    "checkpoint_sha256": fingerprint(args.checkpoint),
                                    "negative_values_clipped": negative,
                                    "prediction_path": str(output / "prediction.h5ad"),
                                    "control_data": str(control_path), "batch_size": args.batch_size})
    conditions.to_csv(run_dir / "conditions.csv", index=False)
    print(f"Saved: {output / 'prediction.h5ad'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
