"""Pathway 4 — deep residual MLP for the immediate (current-state) feature vector."""

from __future__ import annotations

import torch
from torch import nn

from nardis_neural.config import TabularConfig
from nardis_neural.models.common import ResidualBlock

Tensor = torch.Tensor


class ResidualMLP(nn.Module):
    """Input projection → pre-norm residual blocks → LayerNorm → output projection."""

    def __init__(self, in_dim: int, out_dim: int, cfg: TabularConfig, dropout: float) -> None:
        super().__init__()
        self.inp = nn.Linear(in_dim, cfg.width)
        self.blocks = nn.Sequential(
            *[ResidualBlock(cfg.width, 2 * cfg.width, dropout, cfg.activation) for _ in range(cfg.depth)]
        )
        self.norm = nn.LayerNorm(cfg.width)
        self.out = nn.Linear(cfg.width, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Map (B, in_dim) to (B, out_dim)."""
        y: Tensor = self.out(self.norm(self.blocks(self.inp(x))))
        return y
