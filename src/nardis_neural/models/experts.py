"""Expert pathways with a common interface: ``forward(batch) -> ExpertOutput``.

Multi-timescale design
----------------------
Each *sequence* expert (Transformer / recurrent / TCN) owns:

* a per-timescale input projection ``[values·mask, log1p(age)·mask, mask] → d_model``
  (timescales may have different feature dimensions),
* a continuous time encoding plus a learned *timescale embedding*,
* one temporal core **shared across timescales** by default
  (``share_encoder_across_timescales``) or independent cores per timescale,
* masked last-step + mean pooling and a :class:`TimescaleFusion` across resolutions.

Sharing the core means every resolution trains the same temporal filters (more data per
parameter, regularisation across resolutions) while the timescale embedding and
continuous time encoding let the core specialise per resolution.  Independent cores are
available when resolutions behave very differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import Batch, SequenceBatch
from nardis_neural.models.common import TimeEncoding, last_valid, masked_mean
from nardis_neural.models.fusion import TimescaleFusion
from nardis_neural.models.graph import GraphEncoder
from nardis_neural.models.recurrent import RecurrentCore
from nardis_neural.models.ssm import SSMCore
from nardis_neural.models.tabular import ResidualMLP
from nardis_neural.models.tcn import TCNCore
from nardis_neural.models.transformer import TemporalTransformerCore

Tensor = torch.Tensor


@dataclass
class ExpertOutput:
    latent: Tensor  # (B, D)
    available: Tensor  # (B,) bool
    extras: dict[str, Tensor] = field(default_factory=dict)


class Expert(nn.Module):
    name: str = "expert"

    def forward(self, batch: Batch) -> ExpertOutput:  # pragma: no cover - interface
        raise NotImplementedError


class TabularExpert(Expert):
    name = "tabular"

    def __init__(self, config: NeuralConfig) -> None:
        super().__init__()
        mc = config.model
        self.mlp = ResidualMLP(config.features.current_dim, mc.d_model, mc.tabular, mc.dropout)
        self.norm = nn.LayerNorm(mc.d_model)

    def forward(self, batch: Batch) -> ExpertOutput:
        latent = self.norm(self.mlp(batch.current))
        available = torch.ones(batch.size, dtype=torch.bool, device=latent.device)
        return ExpertOutput(latent=latent, available=available)


def _make_core(kind: str, config: NeuralConfig) -> nn.Module:
    mc = config.model
    if kind == "transformer":
        return TemporalTransformerCore(mc.d_model, mc.transformer, mc.dropout)
    if kind == "recurrent":
        return RecurrentCore(mc.d_model, mc.recurrent, mc.dropout)
    if kind == "tcn":
        return TCNCore(mc.d_model, mc.tcn, mc.dropout)
    if kind == "ssm":
        return SSMCore(mc.d_model, mc.ssm, mc.dropout)
    raise ValueError(f"unknown sequence expert {kind}")


class SequenceExpert(Expert):
    """Multi-timescale temporal expert built around one core type."""

    def __init__(self, kind: str, config: NeuralConfig) -> None:
        super().__init__()
        self.name = kind
        mc = config.model
        d = mc.d_model
        self.timescales = config.features.timescale_names
        self.shared = mc.share_encoder_across_timescales
        self.input_proj = nn.ModuleDict(
            {ts.name: nn.Linear(ts.feature_dim + 2, d) for ts in config.features.timescales}
        )
        self.time_encoding = TimeEncoding(d)
        self.timescale_embedding = nn.Embedding(len(self.timescales), d)
        nn.init.normal_(self.timescale_embedding.weight, std=0.02)
        self.input_norm = nn.LayerNorm(d)
        self.input_drop = nn.Dropout(mc.dropout)
        if self.shared:
            self.core = _make_core(kind, config)
        else:
            self.cores = nn.ModuleDict({name: _make_core(kind, config) for name in self.timescales})
        self.pool = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.fusion = TimescaleFusion(d, len(self.timescales))
        self.missing = nn.Parameter(torch.zeros(d))

    def core_for(self, timescale: str) -> nn.Module:
        return self.core if self.shared else self.cores[timescale]

    def embed_inputs(self, timescale: str, seq: SequenceBatch) -> Tensor:
        """Project raw (normalised) steps to (B, T, D) including time + timescale info."""
        m = seq.mask.unsqueeze(-1).to(seq.values.dtype)
        age = torch.log1p(seq.time_deltas.clamp_min(0.0)).unsqueeze(-1)
        feats = torch.cat([seq.values * m, age * m, m], dim=-1)
        x = self.input_proj[timescale](feats) + self.time_encoding(seq.time_deltas)
        idx = torch.tensor(self.timescales.index(timescale), device=x.device)
        x = x + self.timescale_embedding(idx)
        out: Tensor = self.input_drop(self.input_norm(x))
        return out

    def encode_timesteps(self, timescale: str, seq: SequenceBatch) -> Tensor:
        """Per-step representations (B, T, D) — used for self-supervised pretraining."""
        x = self.embed_inputs(timescale, seq)
        h: Tensor = self.core_for(timescale)(x, seq.mask)
        return h

    def summarize(self, h: Tensor, mask: Tensor) -> Tensor:
        out: Tensor = self.pool(torch.cat([last_valid(h, mask), masked_mean(h, mask)], dim=-1))
        return out

    def encode_timescales(self, batch: Batch) -> tuple[Tensor, Tensor]:
        """(B, S, D) per-timescale summaries and (B, S) availability."""
        b = batch.size
        d = self.missing.shape[0]
        present = [n for n in self.timescales if n in batch.sequences]
        summaries: dict[str, Tensor] = {}
        if self.shared and len(present) > 1 and not self.training:
            # inference: one core call for all timescales — left-pad to a common length and
            # stack on the batch axis (every core is invariant to extra left padding, so this
            # is exact).  Training keeps per-timescale calls to avoid padding compute.
            t_max = max(batch.sequences[n].mask.shape[1] for n in present)
            xs, ms = [], []
            for name in present:
                seq = batch.sequences[name]
                pad = t_max - seq.mask.shape[1]
                xs.append(F.pad(self.embed_inputs(name, seq), (0, 0, pad, 0)))
                ms.append(F.pad(seq.mask, (pad, 0), value=False))
            x, mask = torch.cat(xs), torch.cat(ms)
            s = self.summarize(self.core(x, mask), mask).view(len(present), b, d)
            summaries = {name: s[i] for i, name in enumerate(present)}
        else:
            for name in present:
                seq = batch.sequences[name]
                summaries[name] = self.summarize(self.encode_timesteps(name, seq), seq.mask)
        tokens, avail = [], []
        for name in self.timescales:
            seq_opt = batch.sequences.get(name)
            if seq_opt is None:
                tokens.append(self.missing.expand(b, d))
                avail.append(torch.zeros(b, dtype=torch.bool, device=batch.device))
                continue
            s_n = summaries[name]
            a = seq_opt.available
            tokens.append(torch.where(a.unsqueeze(-1), s_n, self.missing.to(s_n.dtype).expand_as(s_n)))
            avail.append(a)
        return torch.stack(tokens, dim=1), torch.stack(avail, dim=1)

    def forward(self, batch: Batch) -> ExpertOutput:
        tokens, avail = self.encode_timescales(batch)
        fused, weights = self.fusion(tokens, avail)
        available = avail.any(dim=1)
        latent = torch.where(available.unsqueeze(-1), fused, self.missing.to(fused.dtype).expand_as(fused))
        return ExpertOutput(latent=latent, available=available, extras={"timescale_weights": weights})


class GraphExpert(Expert):
    name = "graph"

    def __init__(self, config: NeuralConfig) -> None:
        super().__init__()
        mc = config.model
        gi = config.features.graph
        self.encoder = GraphEncoder(gi.node_feature_dim, mc.d_model, gi.num_edge_types, mc.graph, mc.dropout)
        self.missing = nn.Parameter(torch.zeros(mc.d_model))

    def forward(self, batch: Batch) -> ExpertOutput:
        b = batch.size
        g = batch.graph
        if g is None:
            return ExpertOutput(
                latent=self.missing.expand(b, -1),
                available=torch.zeros(b, dtype=torch.bool, device=batch.device),
            )
        latent = self.encoder(g.node_features, g.edge_index, g.edge_type, g.node_batch, g.target_node, b)
        available = g.available
        latent = torch.where(available.unsqueeze(-1), latent, self.missing.to(latent.dtype).expand_as(latent))
        return ExpertOutput(latent=latent, available=available)


def build_expert(name: str, config: NeuralConfig) -> Expert:
    if name == "tabular":
        return TabularExpert(config)
    if name == "graph":
        return GraphExpert(config)
    return SequenceExpert(name, config)
