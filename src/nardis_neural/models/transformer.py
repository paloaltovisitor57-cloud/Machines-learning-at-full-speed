"""Pathway 1 — temporal Transformer encoder (pre-LN, causal, masked, time-aware)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import TransformerConfig

Tensor = torch.Tensor


def build_attention_mask(mask: Tensor, causal: bool) -> Tensor:
    """(B, T) key-validity mask → (B, 1, T, T) boolean "may attend" mask.

    Row ``i`` may attend to key ``j`` if ``j`` is observed and (when causal) ``j <= i``.
    The diagonal is always allowed so rows of fully-missing sequences never produce NaN;
    such outputs are discarded downstream by the availability mask.
    """
    b, t = mask.shape
    allowed = mask[:, None, :].expand(b, t, t)
    if causal:
        tri = torch.ones(t, t, dtype=torch.bool, device=mask.device).tril()
        allowed = allowed & tri
    eye = torch.eye(t, dtype=torch.bool, device=mask.device)
    return (allowed | eye).unsqueeze(1)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.head_dim = d_model // heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b, t, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).reshape(b, t, d)
        y: Tensor = self.proj(out)
        return y


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, ff_multiplier: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_multiplier * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_multiplier * d_model, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = x + self.drop1(self.attn(self.norm1(x), attn_mask))
        x = x + self.drop2(self.ff(self.norm2(x)))
        return x


class TemporalTransformerCore(nn.Module):
    """Stack of pre-LN Transformer blocks over already-projected inputs (B, T, D).

    A learned *recency* positional embedding (index counted from the most recent step)
    complements the continuous time encoding added by the input projection, so the
    representation is aligned on "now" regardless of sequence length.
    """

    def __init__(self, d_model: int, cfg: TransformerConfig, dropout: float) -> None:
        super().__init__()
        self.causal = cfg.causal
        self.max_positions = cfg.max_positions
        self.recency_embedding = nn.Embedding(cfg.max_positions, d_model)
        nn.init.normal_(self.recency_embedding.weight, std=0.02)
        self.blocks = nn.ModuleList(
            TransformerBlock(d_model, cfg.heads, cfg.ff_multiplier, dropout) for _ in range(cfg.layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        t = x.shape[1]
        recency = torch.arange(t - 1, -1, -1, device=x.device).clamp_max(self.max_positions - 1)
        x = x + self.recency_embedding(recency).unsqueeze(0)
        attn_mask = build_attention_mask(mask, self.causal)
        for block in self.blocks:
            x = block(x, attn_mask)
        out: Tensor = self.norm(x) * mask.unsqueeze(-1).to(x.dtype)
        return out
