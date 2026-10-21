"""Small checks of shared VCC inputs, exports and model command boundaries."""

import json
from pathlib import Path
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from baseline.common import (check_training_labels, encode_h1, load_conditions,
                             load_gene_priors, mapped_counts, write_prediction)


def test_conditions_and_test_perturbation_isolation(tmp_path):
    path = tmp_path / "conditions.csv"
    path.write_text("cell_type,target_gene,n_cells\nH1,G0,3\nH1,G1,2\n")
    table = load_conditions(path)
    assert table.n_cells.sum() == 5
    check_training_labels(["non-targeting", "G2"], ["non-targeting", "G3"],
                          table, ["G0", "G1", "G2", "G3"])
    with pytest.raises(ValueError, match="Test perturbations"):
        check_training_labels(["non-targeting", "G0"], ["non-targeting", "G3"],
                              table, ["G0", "G1", "G2", "G3"])
    path.write_text("cell_type,target_gene,n_cells\nH1,G0,1.5\n")
    with pytest.raises(ValueError, match="positive integers"):
        load_conditions(path)


@pytest.mark.parametrize("sparse_input", [False, True])
def test_raw_mapping_and_control_only_encoding(tmp_path, sparse_input):
    matrix = np.asarray([[2, 5], [9, 1], [3, 4]], dtype=np.float32)
    obs = pd.DataFrame({"target_gene": ["NT", "ENSG0", "NT"], "cell_type": ["H1", "H1", "other"]},
                       index=["control", "test", "other"])
    data = ad.AnnData(sparse.csr_matrix(matrix) if sparse_input else matrix,
                      obs=obs, var=pd.DataFrame(index=["ENSG0", "ENSG1"]))
    path, mapping = tmp_path / "data.h5ad", tmp_path / "mapping.tsv"
    data.write_h5ad(path)
    mapping.write_text("ENSG0\tG0\nENSG1\tG1\n")

    class Encoded:
        def __init__(self, values):
            self.values = values
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return self.values

    class Encoder:
        def encode(self, counts, batch_size):
            return Encoded(counts)

    states, labels, ids = encode_h1(path, Encoder(), ["G1", "G0", "missing"], mapping,
                                   controls_only=True, batch_size=1)
    np.testing.assert_array_equal(states, [[5, 2, 0]])
    assert list(labels) == ["non-targeting"] and ids == ["control"]
    _, all_labels, _ = encode_h1(path, Encoder(), ["G1", "G0"], mapping)
    assert list(all_labels) == ["non-targeting", "G0"]
    data.X = matrix / 3
    with pytest.raises(ValueError, match="raw integer counts"):
        mapped_counts(data, np.array([0]), ["G0", "G1"], ["G0", "G1"])


def test_action_priors_do_not_require_an_expression_gene(tmp_path):
    mapping, embeddings = tmp_path / "mapping.tsv", tmp_path / "embeddings.npz"
    mapping.write_text("ENSG0\tG0\nENSG1\tG1\n")
    np.savez(embeddings, ENSG0=np.array([1., 3.]), ENSG1=np.array([3., 1.]))
    priors, missing = load_gene_priors(embeddings, ["G0", "G1", "G2"], mapping)
    assert priors.shape == (4, 2) and missing == ["G2"]
    np.testing.assert_allclose(priors[:2].mean(axis=0), 0)
    np.testing.assert_array_equal(priors[2:], 0)


def test_h5ad_export_with_mapped_file_and_negative_accounting(tmp_path):
    scratch = tmp_path / "scratch"
    expression = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(3, 2))
    expression[:] = [[1, -1], [2, 3], [4, 5]]
    prediction = tmp_path / "prediction.h5ad"
    negative = write_prediction(prediction, expression, ["G0", "G0", "G1"], ["G0", "G2"],
                                {"expression_scale": "lognorm", "seed": 0})
    assert negative == 1
    result = ad.read_h5ad(prediction)
    assert result.shape == (3, 2)
    assert result.obs.target_gene.tolist() == ["G0", "G0", "G1"]
    assert result.obs.cell_type.eq("H1").all()
    assert result.uns["vcc"]["expression_scale"] == "lognorm"
    assert result.uns["vcc"]["negative_values_clipped"] == 1
    assert result.X.min() == 0


@pytest.mark.parametrize("module", ["baseline.lingshucell.infer", "baseline.chreode.train",
                                     "baseline.chreode.infer"])
def test_baseline_help_without_model_libraries(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--conditions" in result.stdout


@pytest.mark.parametrize("module", ["baseline.gears.infer", "baseline.gears.export",
                                     "baseline.prepare_reference"])
def test_h5ad_workflow_help_without_model_libraries(module):
    result = subprocess.run([sys.executable, "-m", module, "--help"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--output-dir" in result.stdout


def test_lingshu_uses_external_outputs_and_original_vcc_config(tmp_path):
    output = tmp_path / "out"
    run_dir = tmp_path / "run"
    result = subprocess.run([sys.executable, "-m", "baseline.lingshucell.infer", "--dry-run",
                             "--output-dir", str(output), "--run-dir", str(run_dir)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "infer/legacy_vcc" in result.stdout
    assert "sampler.steps=3" in result.stdout and "sampler.cfg_scale=2.0" in result.stdout
    assert "sampler.clue_csv=" in result.stdout
    no_clue = subprocess.run([sys.executable, "-m", "baseline.lingshucell.infer", "--dry-run",
                              "--no-clue", "--output-dir", str(output), "--run-dir", str(run_dir)],
                             capture_output=True, text=True)
    assert no_clue.returncode == 0 and "sampler.clue_csv=" not in no_clue.stdout
    assert str(output / "prediction.h5ad") in result.stdout
    assert f"output.workdir={run_dir}" in result.stdout
    assert not output.exists() and not run_dir.exists()


def test_lingshu_exports_h5ad_and_rejects_changed_resume_config(tmp_path):
    repo, assets, output = tmp_path / "repo", tmp_path / "assets", tmp_path / "out"
    run_dir = tmp_path / "run"
    module = repo / "workflows/infer/main.py"
    module.parent.mkdir(parents=True)
    module.write_text('''import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
options = dict(arg.split("=", 1) for arg in sys.argv[1:] if "=" in arg)
table = pd.read_csv(options["task.condition_csv"])
labels = np.repeat(table.target_gene.to_numpy(), table.n_cells.to_numpy())
directory = Path(options["output.workdir"]) / options["output.sample_dir"]
directory.mkdir(parents=True, exist_ok=True)
ad.AnnData(np.ones((len(labels), 2), dtype=np.int32),
           obs=pd.DataFrame({"cell_type": "H1", "target_gene": labels},
                            index=[str(i) for i in range(len(labels))]),
           var=pd.DataFrame(index=["G0", "G1"])).write_h5ad(directory / "merged.h5ad")
''')
    vocab = repo / "data/vocabs/legacy/tokenizer_vcc_cond_dense_gene1tok.json"
    vocab.parent.mkdir(parents=True)
    vocab.write_text(json.dumps({"target_gene_to_token_id_info": {"gene_names_ordered": ["G0", "G1"]}}))
    assets.mkdir()
    for name in ["vcc_perturbation_85M.pth", "gene_embeddings.npz", "whole_dataset_stringent_clue.csv"]:
        (assets / name).write_bytes(b"fake resource for subprocess boundary test")
    conditions = tmp_path / "conditions.csv"
    conditions.write_text("cell_type,target_gene,n_cells\nH1,G0,3\nH1,G1,2\n")
    command = [sys.executable, "-m", "baseline.lingshucell.infer", "--repo", str(repo),
               "--assets", str(assets), "--conditions", str(conditions), "--output-dir", str(output),
               "--run-dir", str(run_dir)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (output / "prediction.h5ad").samefile(run_dir / "inference/merged.h5ad")
    assert sorted(path.name for path in output.iterdir()) == ["prediction.h5ad"]
    assert all((run_dir / name).is_file() for name in ["run.json", "conditions.csv", "console.log"])
    assert ad.read_h5ad(output / "prediction.h5ad").shape == (5, 2)
    failed = subprocess.run(command + ["--resume", "--batch-size", "64"], capture_output=True, text=True)
    assert failed.returncode != 0 and "Resume configuration differs" in failed.stderr


def test_crispri_adapter_has_gradients_when_model_dependencies_are_available(tmp_path):
    torch = pytest.importorskip("torch")
    repo = Path(__file__).resolve().parents[2] / "Chreode"
    if not repo.is_dir():
        pytest.skip("Chreode source is needed for the optional adapter smoke test")
    from baseline.chreode.model import build_model, enable_repository, sample_population
    enable_repository(repo)

    class Base(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(3, 3)
        def alpha_gate(self, delta):
            return torch.ones_like(delta)
        def forward(self, states, delta, noise):
            return self.proj(states)[:, None, :] + 0.1 * noise

    class Backbone:
        dynamics, latent_dim, device = Base(), 3, torch.device("cpu")

    model = build_model(Backbone(), ["G0", "G1"], np.ones((3, 2), dtype=np.float32))
    generator = torch.Generator().manual_seed(1)
    sample = sample_population(model, np.ones((2, 3)), [0, 1], k_samples=2,
                               delta=1., generator=generator)
    sample.square().mean().backward()
    assert sample.shape == (4, 3)
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
