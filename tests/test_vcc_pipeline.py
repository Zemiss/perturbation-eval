"""Small checks for the shared panel and the two-model comparison."""

import csv
import json
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from baseline.compare import _parse_prediction, compare
from baseline.prepare_reference import prepare_reference


@pytest.fixture
def vcc_case(tmp_path):
    rng = np.random.default_rng(11)
    genes = [f"G{i}" for i in range(40)]
    labels = np.repeat(["G0", "G1", "G2"], 24)
    ctrl = rng.poisson(5, (24, 40)).astype(np.int32)
    truth = (rng.poisson(5, (72, 40)) + np.repeat(rng.integers(0, 8, (3, 40)), 24, axis=0)).astype(np.int32)

    def save(name, matrix, conditions, index):
        data = ad.AnnData(matrix, obs=pd.DataFrame({"cell_type": "H1", "target_gene": conditions},
                                                   index=index),
                          var=pd.DataFrame(index=genes))
        path = tmp_path / f"{name}.h5ad"
        data.write_h5ad(path)
        return path

    truth_path = save("truth", truth, labels, [f"truth_{i}" for i in range(len(labels))])
    control_path = save("control", ctrl, ["non-targeting"] * 24, [f"ctrl_{i}" for i in range(24)])
    pred_paths = {}
    for name, scale in [("chreode", 0.1), ("lingshucell", 0.3)]:
        pred = np.rint(np.maximum(truth + rng.normal(0, scale, truth.shape), 0)).astype(np.int32)
        pred_paths[name] = save(name, pred, labels, [f"{name}_{i}" for i in range(len(labels))])
    mapping = tmp_path / "genes.tsv"
    mapping.write_text("".join(f"ENSG{i}\t{gene}\n" for i, gene in enumerate(genes)))
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(genes) + "\n")
    return truth_path, control_path, pred_paths, mapping, panel, genes


def test_prepare_reference_writes_shared_panel_budget_and_conditions(vcc_case, tmp_path):
    truth, control, _, mapping, panel, genes = vcc_case
    out = tmp_path / "reference"
    conditions = out / "conditions.csv"
    prepare_reference(None, truth, control, out, mapping, panel_path=panel,
                      max_cells_per_pert=16, max_control_cells=10,
                      conditions_out=conditions, seed=3)
    prepared = ad.read_h5ad(out / "truth.h5ad")
    assert prepared.shape[0] == 3 * 16
    assert set(prepared.obs.target_gene) == {"G0", "G1", "G2"}
    assert ad.read_h5ad(out / "control.h5ad").shape[0] == 10
    table = pd.read_csv(conditions)
    assert set(table.target_gene) == {"G0", "G1", "G2"}
    assert table.n_cells.tolist() == [16, 16, 16]
    assert (out / "panel.txt").read_text().split() == genes
    with pytest.raises(ValueError, match="already exist"):
        prepare_reference(None, truth, control, out, mapping, panel_path=panel)


def test_compare_scores_two_models_on_one_panel(vcc_case, tmp_path):
    truth, control, predictions, _, _, _ = vcc_case
    payload = compare(predictions, truth, control, tmp_path / "results")
    assert set(payload["models"]) == set(predictions)
    assert payload["gene_panel_sha256"]
    with (tmp_path / "results/comparison.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert set(rows[0]) == {"model", *payload["metrics"], "average_rank"}
    summary = json.loads((tmp_path / "results/comparison.json").read_text())
    assert summary["n_genes"] == 40
    assert set(summary["ranks"]) == set(predictions)
    assert (tmp_path / "results/comparison.md").is_file()


def test_compare_prediction_parser(tmp_path):
    path = tmp_path / "a.h5ad"
    path.write_bytes(b"x")
    assert _parse_prediction(f"model={path}")[0] == "model"
    with pytest.raises(ValueError, match="NAME=PATH"):
        _parse_prediction(str(path))


def test_new_compare_cli_help():
    result = subprocess.run([sys.executable, "-m", "baseline.compare", "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_prepare_reference_intersects_gene_list_and_native_vocab(tmp_path):
    rng = np.random.default_rng(5)
    genes = [f"G{i}" for i in range(10)]
    labels = np.repeat(["G0", "G1"], 8)
    truth = rng.poisson(4, (16, 10)).astype(np.int32)
    ctrl = rng.poisson(4, (8, 10)).astype(np.int32)

    def save(name, matrix, conditions):
        data = ad.AnnData(matrix,
                          obs=pd.DataFrame({"cell_type": "H1", "target_gene": conditions},
                                           index=[f"{name}_{i}" for i in range(len(conditions))]),
                          var=pd.DataFrame(index=genes))
        path = tmp_path / f"{name}.h5ad"
        data.write_h5ad(path)
        return path

    truth_path = save("truth", truth, labels)
    control_path = save("control", ctrl, ["non-targeting"] * 8)
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(genes[:6]) + "\n")
    repo = tmp_path / "repo"
    (repo / "artifacts").mkdir(parents=True)
    pd.DataFrame({"canonical_index": np.arange(8), "human_symbol": genes[2:]}).to_parquet(
        repo / "artifacts/gene_vocab.parquet")
    mapping = tmp_path / "genes.tsv"
    mapping.write_text("".join(f"ENSG{i}\t{gene}\n" for i, gene in enumerate(genes)))
    out = tmp_path / "reference"
    prepare_reference(None, truth_path, control_path, out, mapping, panel_path=panel,
                      panel_vocab=repo)
    prepared = ad.read_h5ad(out / "truth.h5ad")
    assert list(prepared.var_names) == genes[2:6]
    assert (out / "panel.txt").read_text().split() == genes[2:6]
