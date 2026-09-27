"""Deep ensemble of independently initialised full networks + MC-dropout sampling."""

from __future__ import annotations

import copy
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import Batch
from nardis_neural.models.main import ModelOutput, NardisNeuralNetwork

Tensor = torch.Tensor


def set_mc_dropout(module: nn.Module, enabled: bool, scope: list[nn.Module] | None = None) -> None:
    """Put ``nn.Dropout`` layers (within ``scope``, default everywhere) in train mode, all else eval."""
    module.eval()
    if enabled:
        for root in scope if scope is not None else [module]:
            for m in root.modules():
                if isinstance(m, nn.Dropout):
                    m.train()


@contextmanager
def seeded(seed: int, device: torch.device) -> Iterator[None]:
    """Fork the RNG so MC-dropout masks are reproducible without touching global state."""
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices, device_type=device.type if device.type != "cpu" else "cuda"):
        torch.manual_seed(seed)
        yield


@dataclass
class StackedOutputs:
    """Outputs of S stochastic forward passes, stacked on dim 0."""

    means: dict[str, Tensor]  # (S, B, H)
    logvars: dict[str, Tensor]
    logits: dict[str, Tensor]
    quantiles: Tensor | None  # (S, B, H, Q)
    embeddings: Tensor  # (M, B, D) one per member (deterministic pass)
    expert_weights: Tensor  # (M, B, E)
    member_index: Tensor  # (S,) which member produced each sample
    expert_names: tuple[str, ...]


class DeepEnsemble(nn.Module):
    """``size`` independently initialised :class:`NardisNeuralNetwork` members.

    Each member is created under its own seed and trained independently (separate
    optimiser, data order and optionally bootstrap sample).  This is a real ensemble —
    disagreement between members measures epistemic uncertainty.
    """

    def __init__(self, members: list[NardisNeuralNetwork]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ensemble needs at least one member")
        self.members = nn.ModuleList(members)

    @classmethod
    def create(cls, config: NeuralConfig, seed: int | None = None) -> DeepEnsemble:
        base = config.training.seed if seed is None else seed
        members = []
        for i in range(config.ensemble.size):
            torch.manual_seed(base + 1000 * (i + 1))
            members.append(NardisNeuralNetwork(config))
        return cls(members)

    @property
    def size(self) -> int:
        return len(self.members)

    def member(self, i: int) -> NardisNeuralNetwork:
        m = self.members[i]
        assert isinstance(m, NardisNeuralNetwork)
        return m

    def clone(self) -> DeepEnsemble:
        return copy.deepcopy(self)

    @torch.inference_mode()
    def forward_samples(self, batch: Batch, mc_samples: int = 0, mc_seed: int = 0) -> StackedOutputs:
        """Deterministic pass per member plus ``mc_samples`` MC-dropout passes per member.

        The expert encoders (the expensive part) run once per member; MC dropout is applied
        to the decode stack (gate → mixture → latent → heads).  Encoder-level epistemic
        diversity comes from the independently trained ensemble members.
        """
        outs: list[ModelOutput] = []
        member_idx: list[int] = []
        embeddings, gates = [], []
        for i in range(self.size):
            m = self.member(i)
            set_mc_dropout(m, False)
            latents, available, ts_weights = m.run_experts(batch)
            det = m.decode(latents, available, ts_weights)
            outs.append(det)
            member_idx.append(i)
            embeddings.append(det.embedding)
            gates.append(det.expert_weights)
            if mc_samples > 0:
                set_mc_dropout(m, True, scope=m.decode_modules())
                with seeded(mc_seed + 97 * i, batch.device):
                    for _ in range(mc_samples):
                        outs.append(m.decode(latents, available, ts_weights))
                        member_idx.append(i)
                set_mc_dropout(m, False)
        first = outs[0]
        return StackedOutputs(
            means={k: torch.stack([o.means[k] for o in outs]) for k in first.means},
            logvars={k: torch.stack([o.logvars[k] for o in outs]) for k in first.logvars},
            logits={k: torch.stack([o.logits[k] for o in outs]) for k in first.logits},
            quantiles=None
            if first.quantiles is None
            else torch.stack([o.quantiles for o in outs if o.quantiles is not None]),
            embeddings=torch.stack(embeddings),
            expert_weights=torch.stack(gates),
            member_index=torch.tensor(member_idx, device=batch.device),
            expert_names=first.expert_names,
        )
