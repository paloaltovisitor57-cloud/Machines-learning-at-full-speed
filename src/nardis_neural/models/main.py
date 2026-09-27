"""The complete network: experts → dynamic gate → mixture → latent → multi-task heads."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import Batch
from nardis_neural.models.experts import Expert, SequenceExpert, build_expert
from nardis_neural.models.fusion import ExpertMixture, LatentEncoder
from nardis_neural.models.gating import GatingNetwork
from nardis_neural.models.heads import MultiTaskHeads

Tensor = torch.Tensor


@dataclass
class ModelOutput:
    """All outputs live in *normalised target space* (see FeatureNormalizer)."""

    means: dict[str, Tensor]  # regression task -> (B, H)
    logvars: dict[str, Tensor]  # regression task -> (B, H) aleatoric log-variance
    logits: dict[str, Tensor]  # event task -> (B, H)
    quantiles: Tensor | None  # (B, H, Q) return quantiles
    embedding: Tensor  # (B, latent_dim) MarketStateEmbedding
    expert_weights: Tensor  # (B, E)
    expert_names: tuple[str, ...]
    expert_available: Tensor  # (B, E)
    aux_losses: dict[str, Tensor] = field(default_factory=dict)
    timescale_weights: dict[str, Tensor] = field(default_factory=dict)


class NardisNeuralNetwork(nn.Module):
    """Experts → dynamic gate → expert mixture → MarketStateEmbedding → task heads."""

    def __init__(self, config: NeuralConfig) -> None:
        super().__init__()
        self.config = config
        mc = config.model
        self.expert_names: tuple[str, ...] = tuple(mc.enabled_experts)
        self.experts = nn.ModuleDict({name: build_expert(name, config) for name in self.expert_names})
        self.gate = GatingNetwork(len(self.expert_names), mc.d_model, mc.gating, mc.dropout)
        self.mixture = ExpertMixture(len(self.expert_names), mc.d_model, mc.d_model)
        self.latent = LatentEncoder(mc.d_model, mc.latent_dim, mc.dropout)
        self.heads = MultiTaskHeads(
            mc.latent_dim,
            mc.head_hidden_dim,
            len(config.targets.horizons),
            len(config.targets.quantiles),
            mc.dropout,
        )
        self.disabled_experts: set[str] = set()

    def set_disabled_experts(self, names: set[str] | None) -> None:
        """Disable experts at runtime (their gate weight is forced to zero)."""
        unknown = (names or set()) - set(self.expert_names)
        if unknown:
            raise ValueError(f"unknown experts {sorted(unknown)}")
        self.disabled_experts = set(names or set())

    def sequence_experts(self) -> dict[str, SequenceExpert]:
        """The enabled sequence (temporal) experts by name."""
        return {n: e for n, e in self.experts.items() if isinstance(e, SequenceExpert)}

    def run_experts(self, batch: Batch) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """Run every expert: latents (B, E, D), availability (B, E), timescale weights per expert.

        Disabled experts are reported as unavailable.
        """
        latents, avail = [], []
        ts_weights: dict[str, Tensor] = {}
        for name in self.expert_names:
            expert = self.experts[name]
            assert isinstance(expert, Expert)
            out = expert(batch)
            latents.append(out.latent)
            a = out.available
            if name in self.disabled_experts:
                a = torch.zeros_like(a)
            avail.append(a)
            if "timescale_weights" in out.extras:
                ts_weights[name] = out.extras["timescale_weights"]
        return torch.stack(latents, dim=1), torch.stack(avail, dim=1), ts_weights

    def forward(self, batch: Batch) -> ModelOutput:
        """Full forward pass; ``ValueError`` unless ``batch`` is normalised."""
        if not batch.normalized:
            raise ValueError(
                "NardisNeuralNetwork expects a normalised batch (FeatureNormalizer.transform_batch)"
            )
        latents, available, ts_weights = self.run_experts(batch)
        return self.decode(latents, available, ts_weights)

    def decode_modules(self) -> list[nn.Module]:
        """Modules after the experts (gate, mixture, latent, heads) — the MC-dropout stack."""
        return [self.gate, self.mixture, self.latent, self.heads]

    def decode(self, latents: Tensor, available: Tensor, ts_weights: dict[str, Tensor]) -> ModelOutput:
        """Gate → mixture → MarketStateEmbedding → heads, from pre-computed expert latents."""
        gate = self.gate(latents, available)
        fused = self.mixture(latents, gate.weights)
        embedding = self.latent(fused)
        means, logvars, logits, quantiles = self.heads(embedding)
        return ModelOutput(
            means=means,
            logvars=logvars,
            logits=logits,
            quantiles=quantiles,
            embedding=embedding.float(),
            expert_weights=gate.weights,
            expert_names=self.expert_names,
            expert_available=available,
            aux_losses=gate.aux_losses,
            timescale_weights=ts_weights,
        )

    def parameter_count(self) -> int:
        """Total number of parameters."""
        return sum(p.numel() for p in self.parameters())
