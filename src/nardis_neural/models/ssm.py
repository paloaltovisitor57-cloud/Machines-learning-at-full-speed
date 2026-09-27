"""Selective state-space expert (Mamba-style), pure PyTorch.

Each layer: pre-norm → input projection into a signal and a gate → causal depthwise
convolution → **input-dependent** step size Δ, input matrix B and output matrix C →
diagonal selective scan  ``s_t = exp(Δ_t A) s_{t−1} + Δ_t B_t x_t``,  ``y_t = C_t s_t + D x_t``
→ SiLU gate → output projection with a residual connection.

Why it suits market streams: the selection mechanism lets the model decide per step how
much to remember (bursts reset state, quiet periods preserve it), and it is causal and
linear-time.  Unobserved steps get ``Δ = 0`` — the state passes through unchanged — and a
zero input, so padding and gaps are handled exactly (outputs do not depend on how much
left padding precedes a sequence).  The scan runs sequentially over time in plain PyTorch:
cheap for the short windows used here, no custom kernels, runs on CPU, CUDA and MPS.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import SSMConfig

Tensor = torch.Tensor


class SelectiveSSMLayer(nn.Module):
    """One Mamba-style selective state-space layer with a residual connection."""

    def __init__(self, d_model: int, cfg: SSMConfig, dropout: float) -> None:
        super().__init__()
        d_inner = cfg.expand * d_model
        self.d_inner, self.n = d_inner, cfg.state_dim
        self.dt_rank = max(1, math.ceil(d_model / 16))
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, 2 * d_inner)
        self.conv = nn.Conv1d(d_inner, d_inner, cfg.conv_kernel, groups=d_inner)
        self.pad = cfg.conv_kernel - 1
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * cfg.state_dim, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner)
        # Δ initialised in [1e-3, 1e-1] (log-uniform), as in S4/Mamba
        dt = torch.exp(torch.rand(d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.log(torch.arange(1, cfg.state_dim + 1).float()).repeat(d_inner, 1))
        self.D = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Apply causally to (B, T, D); unobserved steps leave the state unchanged."""
        m = mask.unsqueeze(-1).to(x.dtype)
        xs, z = self.in_proj(self.norm(x)).chunk(2, dim=-1)
        xs = xs * m
        xs = F.silu(self.conv(F.pad(xs.transpose(1, 2), (self.pad, 0))).transpose(1, 2)) * m
        dt_in, bmat, cmat = self.x_proj(xs).split([self.dt_rank, self.n, self.n], dim=-1)
        delta = F.softplus(self.dt_proj(dt_in)) * m  # Δ = 0 on unobserved steps → state unchanged
        a = -torch.exp(self.A_log.float())  # (Di, N), strictly negative → stable
        # a plain loop over time: on CPUs it beats parallel scans, which are memory-bound
        state = torch.zeros(x.shape[0], self.d_inner, self.n, device=x.device, dtype=torch.float32)
        ys = []
        for i in range(x.shape[1]):
            d_i = delta[:, i].float().unsqueeze(-1)  # (B, Di, 1)
            b_i = bmat[:, i].float().unsqueeze(1)  # (B, 1, N)
            x_i = xs[:, i].float().unsqueeze(-1)  # (B, Di, 1)
            state = torch.exp(d_i * a) * state + d_i * b_i * x_i
            ys.append((state * cmat[:, i].float().unsqueeze(1)).sum(-1))
        y = torch.stack(ys, dim=1).to(x.dtype) + self.D * xs
        y = y * F.silu(z)
        out: Tensor = x + self.drop(self.out_proj(y)) * m
        return out


class SSMCore(nn.Module):
    """Stack of selective SSM layers followed by LayerNorm."""

    def __init__(self, d_model: int, cfg: SSMConfig, dropout: float) -> None:
        super().__init__()
        self.layers = nn.ModuleList(SelectiveSSMLayer(d_model, cfg, dropout) for _ in range(cfg.layers))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode (B, T, D) with a (B, T) mask; unobserved steps output zero."""
        h = x * mask.unsqueeze(-1).to(x.dtype)
        for layer in self.layers:
            h = layer(h, mask)
        out: Tensor = self.norm(h) * mask.unsqueeze(-1).to(h.dtype)
        return out
