"""VCC CRISPRi adaptation of the released Chreode surgery interface."""

import sys

import torch


def enable_repository(repository):
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repository / "src"))


class GeneConditionEmbedding(torch.nn.Module):
    """Shared CRISPRi action encoder with fixed ESM priors, without condition memorization."""

    def __init__(self, n_genes, width, priors):
        super().__init__()
        from cellworldmodel.foundation.action import GeneSetPerturbationEncoder
        self.encoder = GeneSetPerturbationEncoder(n_genes, width, gene_priors=priors)
        with torch.no_grad():
            self.encoder.gene_residual.weight.zero_()
        self.encoder.gene_residual.weight.requires_grad_(False)

    def forward(self, ids):
        genes = ids.long()[:, None]
        return self.encoder(gene_ids=genes, signs=-torch.ones_like(genes, dtype=torch.float32),
                            modality_ids=torch.ones_like(genes),
                            strengths=torch.ones_like(genes, dtype=torch.float32),
                            mask=genes != self.encoder.unknown_gene_id)


def build_model(backbone, action_genes, priors, *, tune_base=True):
    from cellworldmodel.model.intervention import FieldSurgeryAdapter, SurgeryModel
    adapter = FieldSurgeryAdapter(backbone.latent_dim, len(action_genes), emb_dim=32)
    adapter.cond_emb = GeneConditionEmbedding(len(action_genes), 32,
                                             torch.as_tensor(priors, dtype=torch.float32))
    return SurgeryModel(backbone.dynamics, adapter, tune_base=tune_base).to(backbone.device)


def sample_population(model, states, target_ids, *, k_samples, delta, generator):
    source = torch.as_tensor(states, dtype=torch.float32, device=next(model.parameters()).device)
    horizon = torch.full((len(source),), delta, device=source.device)
    noise = torch.randn(len(source), k_samples, source.shape[1], device=source.device,
                        generator=generator)
    ids = torch.as_tensor(target_ids, dtype=torch.long, device=source.device)
    return model(source, horizon, noise, ids).reshape(-1, source.shape[1])


def load_adapted(checkpoint, release_dir, device):
    from cellworldmodel.foundation import load_chreode_backbone
    from baseline.common import fingerprint
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = payload["config"]
    if config.get("method") != "chreode-vcc-crispri-esm-surgery-v1":
        raise ValueError("Unsupported VCC checkpoint method")
    if fingerprint(release_dir / "release.json") != config["release_sha256"]:
        raise ValueError("Pretrained release differs from the training release")
    backbone = load_chreode_backbone(release_dir, device=device)
    prior_shape = payload["model_state_dict"]["adapter.cond_emb.encoder.gene_priors"].shape
    model = build_model(backbone, config["action_genes"], torch.zeros(prior_shape),
                        tune_base=config["tune_base"])
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return backbone, model, config
