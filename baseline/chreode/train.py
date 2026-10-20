"""Post-train a Chreode CRISPRi adapter on explicit H1 training/validation data."""

import argparse
import csv
from pathlib import Path

import numpy as np

from baseline.common import (CHREODE_REPO, DATA_ROOT, LINGSHU_ASSETS, LINGSHU_REPO,
                             check_training_labels, encode_h1, fingerprint, git_revision,
                             lingshu_genes, load_conditions, load_gene_priors, native_genes,
                             write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=CHREODE_REPO)
    parser.add_argument("--lingshu-repo", type=Path, default=LINGSHU_REPO)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--val-data", required=True, type=Path)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--release-dir", type=Path, default=Path("/data/xiebo/models/chreode/pretrained"))
    parser.add_argument("--gene-embeddings", type=Path, default=LINGSHU_ASSETS / "gene_embeddings.npz")
    parser.add_argument("--run-dir", type=Path, default=Path("/data/xiebo/runs/chreode/vcc"))
    parser.add_argument("--checkpoint", type=Path, default=Path("/data/xiebo/models/chreode/vcc/model.pt"))
    parser.add_argument("--condition-col", default="target_gene")
    parser.add_argument("--cell-type-col", default="cell_type")
    parser.add_argument("--layer", help="Explicit raw counts layer; default is X")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--k-samples", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze-base", action="store_true", help="Train only the external adapter")
    args = parser.parse_args(argv)
    if min(args.steps, args.k_samples, args.eval_every) < 1 or args.batch_size < 2:
        parser.error("steps, k-samples, eval-every must be positive; batch-size must be at least 2")
    if not np.isfinite([args.lr, args.delta]).all() or min(args.lr, args.delta) <= 0 or args.seed < 0:
        parser.error("lr and delta must be finite and positive; seed must be nonnegative")
    for key in ["repo", "lingshu_repo", "release_dir", "run_dir", "checkpoint", "gene_embeddings"]:
        setattr(args, key, getattr(args, key).resolve())
    train_path, val_path = [(path if path.is_absolute() else args.data_root / path).resolve()
                            for path in [args.train_data, args.val_data]]
    if train_path == val_path:
        raise ValueError("Training and validation must be separate files")
    if args.checkpoint.exists() or (args.run_dir.exists() and any(args.run_dir.iterdir())):
        raise ValueError("Training output exists; select new --run-dir and --checkpoint paths")
    conditions_path = (args.conditions or args.lingshu_repo / "data/vcc/pert_counts_Test.csv").resolve()
    conditions = load_conditions(conditions_path)
    action_genes, expression_genes = lingshu_genes(args.lingshu_repo), native_genes(args.repo)
    mapping_file = args.lingshu_repo / "data/vcc/gene_id_name.csv"
    priors, missing_priors = load_gene_priors(args.gene_embeddings, action_genes, mapping_file)
    if set(conditions.target_gene) - set(action_genes):
        raise ValueError("Test conditions contain genes outside the VCC action vocabulary")
    if set(conditions.target_gene) & set(missing_priors):
        raise ValueError(f"Test targets lack ESM priors: {sorted(set(conditions.target_gene) & set(missing_priors))}")

    import torch
    from baseline.chreode.model import build_model, enable_repository, sample_population
    enable_repository(args.repo)
    from cellworldmodel.foundation import load_chreode_backbone
    from cellworldmodel.model.intervention import population_loss
    torch.manual_seed(args.seed)
    backbone = load_chreode_backbone(args.release_dir, device=args.device)
    if backbone.n_genes != len(expression_genes):
        raise ValueError("Native vocabulary does not match the released encoder")
    kwargs = dict(condition_col=args.condition_col, cell_type_col=args.cell_type_col, layer=args.layer)
    train_z, train_labels, train_ids = encode_h1(train_path, backbone, expression_genes, mapping_file, **kwargs)
    val_z, val_labels, val_ids = encode_h1(val_path, backbone, expression_genes, mapping_file, **kwargs)
    check_training_labels(train_labels, val_labels, conditions, action_genes)
    if set(train_ids) & set(val_ids):
        raise ValueError("Training and validation share cell identifiers; prepare disjoint cell splits")
    config = {"method": "chreode-vcc-crispri-esm-surgery-v1", "seed": args.seed,
              "release_sha256": fingerprint(args.release_dir / "release.json"),
              "chreode_commit": git_revision(args.repo), "tune_base": not args.freeze_base,
              "action_genes": action_genes, "expression_genes": expression_genes,
              "conditions_sha256": fingerprint(conditions_path),
              "gene_embeddings_sha256": fingerprint(args.gene_embeddings),
              "missing_gene_priors": missing_priors, "delta": args.delta,
              "steps": args.steps, "batch_size": args.batch_size, "k_samples": args.k_samples,
              "learning_rate": args.lr, "input_type": "counts", "condition_col": args.condition_col,
              "cell_type_col": args.cell_type_col, "layer": args.layer,
              "train_data": str(train_path), "val_data": str(val_path),
              "loss": "Sinkhorn divergence (blur=0.05) + multiscale MMD; weights 1,1"}
    model = build_model(backbone, action_genes, priors, tune_base=not args.freeze_base)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    action_ids = {name: i for i, name in enumerate(action_genes)}
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    train_groups = {label: np.flatnonzero(train_labels == label) for label in np.unique(train_labels)}
    val_groups = {label: np.flatnonzero(val_labels == label) for label in np.unique(val_labels)}
    train_targets = sorted(set(train_groups) - {"non-targeting"})
    val_targets = sorted(set(val_groups) - {"non-targeting"})

    def loss_for(states, groups, label, local_rng, local_generator):
        controls = local_rng.choice(groups["non-targeting"], args.batch_size, replace=True)
        targets = local_rng.choice(groups[label], args.batch_size, replace=True)
        prediction = sample_population(model, states[controls], [action_ids[label]] * args.batch_size,
                                       k_samples=args.k_samples,
                                       delta=args.delta, generator=local_generator)
        truth = torch.as_tensor(states[targets], dtype=torch.float32, device=args.device)
        return population_loss(prediction, truth, transport_objective="sinkhorn_divergence")[0]

    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.run_dir / "run.json", config)
    conditions.to_csv(args.run_dir / "conditions.csv", index=False)
    best = float("inf")
    with (args.run_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["step", "train_loss", "val_loss"])
        writer.writeheader()
        for step in range(1, args.steps + 1):
            model.train()
            loss = loss_for(train_z, train_groups, str(rng.choice(train_targets)), rng, generator)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if step == 1 or step % args.eval_every == 0 or step == args.steps:
                model.eval()
                val_rng = np.random.default_rng(args.seed + 100000)
                val_generator = torch.Generator(device=args.device).manual_seed(args.seed + 100000)
                with torch.no_grad():
                    scores = [float(loss_for(val_z, val_groups, label, val_rng, val_generator))
                              for label in val_targets]
                score = float(np.mean(scores))
                if not np.isfinite(score):
                    raise ValueError("Nonfinite validation loss")
                writer.writerow({"step": step, "train_loss": float(loss.detach()), "val_loss": score})
                handle.flush()
                print(f"step={step} train_loss={float(loss.detach()):.6g} val_loss={score:.6g}", flush=True)
                if score < best:
                    best = score
                    temporary = args.checkpoint.with_suffix(".tmp.pt")
                    torch.save({"model_state_dict": model.state_dict(), "config": config,
                                "best_step": step, "validation_loss": score}, temporary)
                    temporary.replace(args.checkpoint)
    print(f"Saved validation-selected checkpoint: {args.checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
