"""Pathway 5 (optional) — relational graph encoder in pure PyTorch.

PyTorch Geometric is deliberately *not* a dependency: the graphs here are small
per-observation ego-graphs (wallet→token, wallet→wallet, token→token), for which
``index_add_``/``scatter_reduce`` message passing is simple, fast and dependency-free.

Two layer types are provided:

* ``sage`` — relational GraphSAGE: ``h_i' = W_self h_i + Σ_r W_r · mean_{j ∈ N_r(i)} h_j``
* ``gat``  — multi-head graph attention with a learned per-relation attention bias.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import GraphModelConfig

Tensor = torch.Tensor


def scatter_mean(src: Tensor, index: Tensor, n: int) -> Tensor:
    out = torch.zeros(n, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    count = torch.zeros(n, dtype=src.dtype, device=src.device)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    return out / count.clamp_min(1.0).unsqueeze(-1)


def scatter_softmax(scores: Tensor, index: Tensor, n: int) -> Tensor:
    """Softmax of (E, H) scores grouped by destination node ``index``."""
    h = scores.shape[1]
    idx = index.unsqueeze(-1).expand(-1, h)
    maxes = torch.full((n, h), float("-inf"), dtype=scores.dtype, device=scores.device)
    maxes = maxes.scatter_reduce(0, idx, scores, reduce="amax", include_self=True)
    ex = torch.exp(scores - maxes[index])
    denom = torch.zeros(n, h, dtype=scores.dtype, device=scores.device).index_add_(0, index, ex)
    return ex / denom[index].clamp_min(1e-12)


class RelationalSAGELayer(nn.Module):
    def __init__(self, dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.self_lin = nn.Linear(dim, dim)
        self.rel_lin = nn.ModuleList(nn.Linear(dim, dim, bias=False) for _ in range(num_relations))
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        n = h.shape[0]
        src, dst = edge_index[0], edge_index[1]
        agg = self.self_lin(h)
        for r, lin in enumerate(self.rel_lin):
            sel = edge_type == r
            if bool(sel.any()):
                agg = agg + lin(scatter_mean(h[src[sel]], dst[sel], n))
        out: Tensor = self.norm(h + self.drop(F.gelu(agg)))
        return out


class RelationalGATLayer(nn.Module):
    def __init__(self, dim: int, heads: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("graph hidden_dim must be divisible by heads")
        self.heads, self.head_dim = heads, dim // heads
        self.lin = nn.Linear(dim, dim, bias=False)
        self.att_src = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.att_dst = nn.Parameter(torch.randn(heads, self.head_dim) * 0.1)
        self.rel_bias = nn.Embedding(num_relations + 1, heads)  # +1 = self loop
        nn.init.zeros_(self.rel_bias.weight)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.num_relations = num_relations

    def forward(self, h: Tensor, edge_index: Tensor, edge_type: Tensor) -> Tensor:
        n = h.shape[0]
        loops = torch.arange(n, device=h.device)
        src = torch.cat([edge_index[0], loops])
        dst = torch.cat([edge_index[1], loops])
        et = torch.cat([edge_type, torch.full_like(loops, self.num_relations)])
        x = self.lin(h).view(n, self.heads, self.head_dim)
        score = (x[src] * self.att_src).sum(-1) + (x[dst] * self.att_dst).sum(-1) + self.rel_bias(et)
        alpha = self.drop(scatter_softmax(F.leaky_relu(score, 0.2), dst, n))
        msg = x[src] * alpha.unsqueeze(-1)
        out = torch.zeros_like(x).index_add_(0, dst, msg).reshape(n, -1)
        res: Tensor = self.norm(h + F.gelu(out))
        return res


class GraphEncoder(nn.Module):
    """Encodes a batch of ego-graphs; returns one vector per sample."""

    def __init__(
        self, node_dim: int, out_dim: int, num_relations: int, cfg: GraphModelConfig, dropout: float
    ) -> None:
        super().__init__()
        self.inp = nn.Linear(node_dim, cfg.hidden_dim)
        layers: list[nn.Module] = []
        for _ in range(cfg.layers):
            if cfg.kind == "sage":
                layers.append(RelationalSAGELayer(cfg.hidden_dim, num_relations, dropout))
            else:
                layers.append(RelationalGATLayer(cfg.hidden_dim, cfg.heads, num_relations, dropout))
        self.layers = nn.ModuleList(layers)
        self.readout = nn.Sequential(nn.Linear(2 * cfg.hidden_dim, out_dim), nn.LayerNorm(out_dim))
        self.num_relations = num_relations

    def forward(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        node_batch: Tensor,
        target_node: Tensor,
        batch_size: int,
    ) -> Tensor:
        h = F.gelu(self.inp(node_features))
        edge_type = edge_type.clamp(0, self.num_relations - 1)
        for layer in self.layers:
            h = layer(h, edge_index, edge_type)
        pooled = (
            scatter_mean(h, node_batch, batch_size) if h.shape[0] else h.new_zeros(batch_size, h.shape[1])
        )
        has = target_node >= 0
        target = torch.zeros(batch_size, h.shape[1], dtype=h.dtype, device=h.device)
        if bool(has.any()):
            target[has] = h[target_node[has]]
        out: Tensor = self.readout(torch.cat([target, pooled], dim=-1))
        return out
