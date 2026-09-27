"""Fusion modules: timescale fusion, expert mixture and the shared latent encoder."""

from __future__ import annotations

import torch
from torch import nn

from nardis_neural.models.common import ResidualBlock

Tensor = torch.Tensor


class TimescaleFusion(nn.Module):
    """Masked attention pooling across timescale representations.

    Each timescale summary (B, S, D) gets a learned timescale embedding; a content-based
    score decides how much each resolution contributes *for this observation*.  Missing
    timescales are masked out.  Returns the fused vector and the (B, S) weights.
    """

    def __init__(self, d_model: int, n_timescales: int) -> None:
        super().__init__()
        self.timescale_embedding = nn.Parameter(torch.zeros(n_timescales, d_model))
        self.score = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.mix = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, available: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse tokens (B, S, D) with availability (B, S) into (B, D); also return weights (B, S).

        Weights are all zero for samples with no available timescale.
        """
        x = tokens + self.timescale_embedding.unsqueeze(0)
        scores = self.score(x).squeeze(-1).float()
        any_avail = available.any(dim=1, keepdim=True)
        safe_avail = available | ~any_avail  # avoid all -inf rows
        scores = scores.masked_fill(~safe_avail, -1e9)
        weights = torch.softmax(scores, dim=1).to(x.dtype)
        weights = weights * any_avail.to(x.dtype)
        attended = (weights.unsqueeze(-1) * x).sum(dim=1)
        m = available.unsqueeze(-1).to(x.dtype)
        mean = (x * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        fused: Tensor = self.norm(self.mix(torch.cat([attended, mean], dim=-1)))
        return fused, weights


class ExpertMixture(nn.Module):
    """Gate-weighted mixture of expert-specific adapters: Σ_e w_e · A_e(h_e)."""

    def __init__(self, n_experts: int, d_expert: int, d_out: int) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            nn.Sequential(nn.LayerNorm(d_expert), nn.Linear(d_expert, d_out)) for _ in range(n_experts)
        )

    def forward(self, latents: Tensor, weights: Tensor) -> Tensor:
        """latents (B, E, D), weights (B, E) → (B, d_out)."""
        adapted = torch.stack([a(latents[:, i]) for i, a in enumerate(self.adapters)], dim=1)
        return (weights.unsqueeze(-1).to(adapted.dtype) * adapted).sum(dim=1)


class LatentEncoder(nn.Module):
    """Maps the fused representation to the reusable MarketStateEmbedding."""

    def __init__(self, d_in: int, latent_dim: int, dropout: float, depth: int = 2) -> None:
        super().__init__()
        self.inp = nn.Linear(d_in, latent_dim)
        self.blocks = nn.Sequential(
            *[ResidualBlock(latent_dim, 2 * latent_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, d_in) to the (B, latent_dim) MarketStateEmbedding."""
        out: Tensor = self.norm(self.blocks(self.inp(x)))
        return out
