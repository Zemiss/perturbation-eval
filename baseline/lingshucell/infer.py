"""Run the original Lingshu VCC inference and export prediction.h5ad."""

import argparse
import errno
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from baseline.common import (LINGSHU_ASSETS, LINGSHU_REPO, fingerprint, git_revision,
                             lingshu_genes, load_conditions, validate_lingshu_prediction, write_json)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=LINGSHU_REPO)
    parser.add_argument("--assets", type=Path, default=LINGSHU_ASSETS)
    parser.add_argument("--conditions", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/xiebo/outputs/lingshucell"))
    parser.add_argument("--run-dir", type=Path, default=Path("/data/xiebo/runs/lingshucell/vcc_infer"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0, help="Per-task base seed; global seed stays 3407")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print command without loading weights")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.seed < 0:
        parser.error("batch-size must be positive and seed nonnegative")
    repo, assets, output = args.repo.resolve(), args.assets.resolve(), args.output_dir.resolve()
    run_dir = args.run_dir.resolve()
    if output.is_relative_to(run_dir) or run_dir.is_relative_to(output):
        raise ValueError("--run-dir and --output-dir must be separate directories")
    conditions = (args.conditions or repo / "data/vcc/pert_counts_Test.csv").resolve()
    command = [sys.executable, "-m", "workflows.infer.main", "--config-name", "infer/legacy_vcc",
               f"checkpoint.model_path={assets / 'vcc_perturbation_85M.pth'}",
               f"model.config.gene_embedding_file={assets / 'gene_embeddings.npz'}",
               f"sampler.clue_csv={assets / 'whole_dataset_stringent_clue.csv'}",
               f"task.condition_csv={conditions}", f"task.batch_size={args.batch_size}",
               f"task.base_seed={args.seed}", "seed=3407", "sampler.steps=3", "sampler.cfg_scale=2.0",
               f"output.workdir={run_dir}", "output.sample_dir=inference"]
    if args.dry_run:
        print(f"cd {shlex.quote(str(repo))}\n{shlex.join(command)}")
        print(f"Export: {output / 'prediction.h5ad'}")
        return 0
    load_conditions(conditions)
    for name in ["vcc_perturbation_85M.pth", "gene_embeddings.npz", "whole_dataset_stringent_clue.csv"]:
        if not (assets / name).is_file():
            raise FileNotFoundError(assets / name)
    config = {"model": "lingshucell", "source_commit": git_revision(repo),
              "conditions_sha256": fingerprint(conditions), "batch_size": args.batch_size,
              "base_seed": args.seed, "expression_scale": "counts", "command": command,
              "prediction_path": str(output / "prediction.h5ad"),
              "assets_sha256": {name: fingerprint(assets / name) for name in
                                ["vcc_perturbation_85M.pth", "gene_embeddings.npz",
                                 "whole_dataset_stringent_clue.csv"]}}
    manifest = run_dir / "run.json"
    resuming = False
    if run_dir.exists() and any(run_dir.iterdir()):
        if not args.resume or not manifest.is_file():
            raise ValueError("Run directory exists; use a new --run-dir or --resume for the same run")
        import json
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Resume configuration differs from the recorded run")
        resuming = True
    if output.exists() and any(output.iterdir()):
        if not resuming or sorted(path.name for path in output.iterdir()) != ["prediction.h5ad"]:
            raise ValueError("Output exists; use a new --output-dir or resume its matching run")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest, config)
    load_conditions(conditions).to_csv(run_dir / "conditions.csv", index=False)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with (run_dir / "console.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, command)
    source, destination = run_dir / "inference/merged.h5ad", output / "prediction.h5ad"
    validate_lingshu_prediction(source, load_conditions(conditions), lingshu_genes(repo))
    output.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not os.path.samefile(source, destination) and fingerprint(source) != fingerprint(destination):
            raise ValueError("Existing prediction.h5ad does not belong to this inference run")
    else:
        try:
            os.link(source, destination)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise
            temporary = destination.with_suffix(".tmp.h5ad")
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
    print(f"Saved: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
