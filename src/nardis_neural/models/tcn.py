"""Pathway 3 — causal temporal convolution network.

Dilated causal convolutions (left padding only) in residual blocks.  Normalisation is a
per-timestep LayerNorm over channels — batch/group norm over time would leak future
statistics into past positions.  Each block's output is a different receptive-field
scale; a learned softmax mixture of all scales forms the output ("multi-scale skip").
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import TCNConfig

Tensor = torch.Tensor


class CausalConv1d(nn.Conv1d):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__(in_ch, out_ch, kernel_size, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(F.pad(x, (self.left_pad, 0)))


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of a (B, C, T) tensor — strictly per timestep."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.norm1 = ChannelLayerNorm(out_ch)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm2 = ChannelLayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: Tensor, m: Tensor) -> Tensor:
        """x (B, C, T); m (B, 1, T) float mask — unobserved steps are held at exactly zero."""
        y = self.drop(self.act(self.norm1(self.conv1(x)))) * m
        y = self.drop(self.act(self.norm2(self.conv2(y)))) * m
        out: Tensor = y + self.residual(x) * m
        return out


class TCNCore(nn.Module):
    def __init__(self, d_model: int, cfg: TCNConfig, dropout: float) -> None:
        super().__init__()
        blocks = []
        in_ch = d_model
        for i, ch in enumerate(cfg.channels):
            blocks.append(TemporalBlock(in_ch, ch, cfg.kernel_size, 2**i, dropout))
            in_ch = ch
        self.blocks = nn.ModuleList(blocks)
        self.scale_proj = nn.ModuleList(nn.Conv1d(ch, d_model, 1) for ch in cfg.channels)
        self.scale_logits = nn.Parameter(torch.zeros(len(cfg.channels)))
        self.norm = nn.LayerNorm(d_model)
        self.receptive_field = 1 + sum(2 * (cfg.kernel_size - 1) * 2**i for i in range(len(cfg.channels)))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        m = mask.unsqueeze(1).to(x.dtype)  # (B, 1, T)
        h = x.transpose(1, 2) * m  # (B, D, T)
        weights = torch.softmax(self.scale_logits, dim=0)
        mixed = torch.zeros_like(h)
        for i, (block, proj) in enumerate(zip(self.blocks, self.scale_proj, strict=True)):
            # unobserved steps stay exactly zero inside every block, identical to causal zero
            # padding, so outputs do not depend on how much left padding precedes a sequence
            h = block(h, m)
            mixed = mixed + weights[i] * proj(h)
        out: Tensor = self.norm(mixed.transpose(1, 2))
        return out * mask.unsqueeze(-1).to(out.dtype)
