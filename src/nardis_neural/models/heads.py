"""Multi-horizon, multi-task output heads.

Every (task, horizon) pair has its *own* two-layer MLP; the per-horizon weights are held
in batched parameter tensors and evaluated with one einsum, so adding horizons is cheap.

Activations
-----------
* ``return``: identity mean (can be negative)
* ``max_upside``, ``max_drawdown``, ``volatility``: softplus mean (non-negative magnitudes)
* log-variance: soft-bounded to ``[LOGVAR_MIN, LOGVAR_MAX]`` via tanh (heteroscedastic)
* return quantiles: base + cumulative softplus increments → monotone, never crossing
* event heads: raw logits (sigmoid / calibration applied at inference)
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import CLASSIFICATION_TASKS, POSITIVE_TASKS, REGRESSION_TASKS

Tensor = torch.Tensor

LOGVAR_MIN = -9.0
LOGVAR_MAX = 5.0


class HorizonHeads(nn.Module):
    """H independent MLPs ``in_dim → hidden → out_dim`` evaluated in parallel."""

    def __init__(self, in_dim: int, hidden: int, n_horizons: int, out_dim: int, dropout: float) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_horizons, in_dim, hidden))
        self.b1 = nn.Parameter(torch.zeros(n_horizons, hidden))
        self.w2 = nn.Parameter(torch.empty(n_horizons, hidden, out_dim))
        self.b2 = nn.Parameter(torch.zeros(n_horizons, out_dim))
        nn.init.normal_(self.w1, std=1.0 / math.sqrt(in_dim))
        nn.init.normal_(self.w2, std=0.01)
        self.drop = nn.Dropout(dropout)

    def forward(self, z: Tensor) -> Tensor:
        """z (B, in_dim) → (B, H, out_dim)."""
        h = F.gelu(torch.einsum("bi,hio->bho", z, self.w1) + self.b1)
        h = self.drop(h)
        return torch.einsum("bhi,hio->bho", h, self.w2) + self.b2


class MultiTaskHeads(nn.Module):
    """Per-(task, horizon) heads: regression mean/log-variance, event logits, quantiles."""

    def __init__(
        self, latent_dim: int, hidden: int, n_horizons: int, n_quantiles: int, dropout: float
    ) -> None:
        super().__init__()
        self.regression = nn.ModuleDict(
            {t: HorizonHeads(latent_dim, hidden, n_horizons, 2, dropout) for t in REGRESSION_TASKS}
        )
        self.classification = nn.ModuleDict(
            {t: HorizonHeads(latent_dim, hidden, n_horizons, 1, dropout) for t in CLASSIFICATION_TASKS}
        )
        self.n_quantiles = n_quantiles
        self.quantiles = (
            HorizonHeads(latent_dim, hidden, n_horizons, n_quantiles, dropout) if n_quantiles else None
        )
        for task in POSITIVE_TASKS:
            head = self.regression[task]
            assert isinstance(head, HorizonHeads)
            with torch.no_grad():
                head.b2[:, 0].fill_(0.5)  # softplus(0.5) ≈ 0.97 ≈ unit-scale magnitude

    def forward(
        self, z: Tensor
    ) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Tensor], Tensor | None]:
        """Map embeddings (B, latent_dim) to ``(means, logvars, logits, quantiles)``.

        The first three are task → (B, H) float32 dicts; quantiles is a monotone (B, H, Q)
        tensor, or None when no quantiles are configured.
        """
        means: dict[str, Tensor] = {}
        logvars: dict[str, Tensor] = {}
        for task, head in self.regression.items():
            raw = head(z).float()
            mu = raw[..., 0]
            if task in POSITIVE_TASKS:
                mu = F.softplus(mu)
            half = (LOGVAR_MAX - LOGVAR_MIN) / 2
            logvars[task] = LOGVAR_MIN + half + half * torch.tanh(raw[..., 1] / half)
            means[task] = mu
        logits = {task: head(z).float()[..., 0] for task, head in self.classification.items()}
        quantiles = None
        if self.quantiles is not None:
            q = self.quantiles(z).float()
            base = q[..., :1]
            if self.n_quantiles > 1:
                inc = F.softplus(q[..., 1:]) + 1e-4
                quantiles = torch.cat([base, base + torch.cumsum(inc, dim=-1)], dim=-1)
            else:
                quantiles = base
        return means, logvars, logits, quantiles
