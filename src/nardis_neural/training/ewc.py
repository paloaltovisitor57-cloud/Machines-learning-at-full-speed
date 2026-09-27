"""Elastic Weight Consolidation (Kirkpatrick et al., 2017).

A diagonal empirical Fisher information is estimated on the *previous* data (replay
history) at the champion's parameters θ*.  The candidate then pays
``λ/2 · Σ_i F_i (θ_i − θ*_i)²`` for moving parameters that mattered before.

The Fisher is normalised to mean 1 across all parameters so ``ewc_weight`` has the same
meaning regardless of model size or loss scale.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from torch import nn

from nardis_neural.data.datasets import Batch
from nardis_neural.models.main import ModelOutput

Tensor = torch.Tensor


class EWCPenalty:
    def __init__(self, anchor: dict[str, Tensor], fisher: dict[str, Tensor], weight: float) -> None:
        self.anchor = anchor
        self.fisher = fisher
        self.weight = weight

    @classmethod
    def estimate(
        cls,
        model: nn.Module,
        batches: Iterable[Batch],
        loss_fn: Callable[[ModelOutput, Batch], Tensor],
        weight: float,
        max_batches: int = 20,
    ) -> EWCPenalty:
        """``batches`` must already be normalised and on the model's device."""
        params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        fisher = {n: torch.zeros_like(p) for n, p in params.items()}
        was_training = model.training
        model.eval()
        count = 0
        for batch in batches:
            if count >= max_batches:
                break
            model.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch), batch)
            if not torch.isfinite(loss):
                continue
            torch.autograd.backward(loss)
            for n, p in params.items():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            count += 1
        model.zero_grad(set_to_none=True)
        model.train(was_training)
        if count == 0:
            raise ValueError("EWC Fisher estimation saw no valid batches")
        fisher = {n: f / count for n, f in fisher.items()}
        total = sum(float(f.sum()) for f in fisher.values())
        numel = sum(f.numel() for f in fisher.values())
        mean = total / max(numel, 1)
        if mean > 0:
            fisher = {n: f / mean for n, f in fisher.items()}
        anchor = {n: p.detach().clone() for n, p in params.items()}
        return cls(anchor, fisher, weight)

    def to(self, device: torch.device) -> EWCPenalty:
        self.anchor = {k: v.to(device) for k, v in self.anchor.items()}
        self.fisher = {k: v.to(device) for k, v in self.fisher.items()}
        return self

    def __call__(self, model: nn.Module) -> Tensor:
        penalty: Tensor | None = None
        for n, p in model.named_parameters():
            if n not in self.fisher:
                continue
            term = (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
            penalty = term if penalty is None else penalty + term
        if penalty is None:
            return torch.zeros(())
        return 0.5 * self.weight * penalty

    def state_dict(self) -> dict[str, dict[str, Tensor] | float]:
        return {"anchor": self.anchor, "fisher": self.fisher, "weight": self.weight}
