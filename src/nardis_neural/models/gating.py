"""Dynamic mixture-of-experts gating network.

The gate sees every expert's (layer-normalised) latent plus the experts' availability
flags and produces a per-observation softmax over experts.  Expert weights are *never*
assigned by hand.  Anti-collapse regularisers:

* **load balance** — ``E · Σ_e importance_e² − 1`` where ``importance`` is the batch-mean
  gate weight; zero when experts are used equally on average (Switch/Shazeer style),
  while still allowing sharp per-sample routing.
* **entropy** — small bonus on per-sample gate entropy (returned as ``−H``) so the gate
  does not saturate early before evidence accumulates.
* **z-loss** — penalises large gate logits (ST-MoE) for numerical stability.
* **noisy gating** and **expert dropout** during training so every expert keeps receiving
  gradient signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from nardis_neural.config import GatingConfig

Tensor = torch.Tensor


@dataclass
class GateOutput:
    """Gate weights (B, E), raw logits (B, E) and the weighted auxiliary losses."""

    weights: Tensor  # (B, E), rows sum to 1 over available experts
    logits: Tensor  # (B, E) pre-mask logits
    aux_losses: dict[str, Tensor]


class GatingNetwork(nn.Module):
    """Per-observation softmax gate over experts with anti-collapse regularisers."""

    def __init__(self, n_experts: int, d_expert: int, cfg: GatingConfig, dropout: float) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.cfg = cfg
        self.norm = nn.LayerNorm(d_expert)
        self.net = nn.Sequential(
            nn.Linear(n_experts * d_expert + n_experts, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.hidden_dim, n_experts),
        )
        last = self.net[-1]
        assert isinstance(last, nn.Linear)
        nn.init.zeros_(last.weight)  # start from uniform routing; the gate must learn to specialise
        nn.init.zeros_(last.bias)

    def forward(self, latents: Tensor, available: Tensor) -> GateOutput:
        """Route expert latents (B, E, D) with availability (B, E).

        Unavailable experts get zero weight (all experts are used when none is available);
        expert dropout and gate noise apply only in training mode.
        """
        b, e, _ = latents.shape
        if self.training and self.cfg.expert_dropout > 0 and e > 1:
            drop = torch.rand(b, e, device=latents.device) < self.cfg.expert_dropout
            candidate = available & ~drop
            keep_any = candidate.any(dim=1, keepdim=True)
            available = torch.where(keep_any, candidate, available)
        feats = torch.cat([self.norm(latents).reshape(b, -1), available.to(latents.dtype)], dim=-1)
        logits = self.net(feats).float()
        routed = logits / self.cfg.temperature
        if self.training and self.cfg.noise_std > 0:
            routed = routed + torch.randn_like(routed) * self.cfg.noise_std
        any_avail = available.any(dim=1, keepdim=True)
        mask = available | ~any_avail
        routed = routed.masked_fill(~mask, -1e9)
        weights = torch.softmax(routed, dim=-1)

        importance = weights.mean(dim=0)
        load_balance = e * (importance**2).sum() - 1.0
        entropy = -(weights * torch.log(weights.clamp_min(1e-9))).sum(dim=-1).mean()
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        aux = {
            "gate_load_balance": load_balance * self.cfg.load_balance_weight,
            "gate_entropy": -entropy * self.cfg.entropy_weight,
            "gate_z_loss": z_loss * self.cfg.z_loss_weight,
        }
        return GateOutput(weights=weights, logits=logits, aux_losses=aux)
