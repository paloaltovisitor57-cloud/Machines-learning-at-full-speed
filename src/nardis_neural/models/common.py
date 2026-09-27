"""Small building blocks shared by several pathways."""

from __future__ import annotations

import torch
from torch import nn

Tensor = torch.Tensor


def make_activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name}")


class TimeEncoding(nn.Module):
    """Continuous encoding of irregular time offsets.

    ``log1p(seconds)`` compresses the dynamic range (1 s … hours) and a learnable bank of
    Fourier features maps it to ``d_model``.  This is robust to irregular sampling and to
    different bar resolutions, unlike integer positional encodings.
    """

    def __init__(self, d_model: int, n_frequencies: int = 16) -> None:
        super().__init__()
        self.freq = nn.Linear(1, n_frequencies)
        nn.init.normal_(self.freq.weight, std=1.0)
        self.out = nn.Linear(2 * n_frequencies + 1, d_model)

    def forward(self, seconds: Tensor) -> Tensor:
        t = torch.log1p(seconds.clamp_min(0.0)).unsqueeze(-1)
        phase = self.freq(t)
        feats = torch.cat([torch.sin(phase), torch.cos(phase), t / 10.0], dim=-1)
        out: Tensor = self.out(feats)
        return out


def masked_mean(h: Tensor, mask: Tensor) -> Tensor:
    """Mean over time of (B, T, D) with a (B, T) bool mask; zeros for empty rows."""
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def last_valid(h: Tensor, mask: Tensor) -> Tensor:
    """Hidden state at the most recent observed step (index 0 when none observed)."""
    t = mask.shape[1]
    positions = torch.arange(t, device=mask.device).expand_as(mask)
    idx = torch.where(mask, positions, torch.zeros_like(positions)).max(dim=1).values
    return h[torch.arange(h.shape[0], device=h.device), idx]


class ResidualBlock(nn.Module):
    """Pre-norm residual MLP block: x + Drop(W2 act(W1 LN(x)))."""

    def __init__(self, dim: int, hidden: int, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = make_activation(activation)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        y = self.fc2(self.drop1(self.act(self.fc1(self.norm(x)))))
        out: Tensor = x + self.drop2(y)
        return out

