"""Pathway 2 — recurrent (GRU default / LSTM) sequence encoder.

Arbitrary masks (left padding *and* interior gaps) are handled by compacting observed
steps to the front in chronological order and running a packed RNN, so padding never
enters the recurrent state.  Outputs are scattered back to their original positions.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nardis_neural.config import RecurrentConfig

Tensor = torch.Tensor


class RecurrentCore(nn.Module):
    """GRU/LSTM temporal core that runs only over observed steps."""

    def __init__(self, d_model: int, cfg: RecurrentConfig, dropout: float) -> None:
        super().__init__()
        rnn_cls = nn.GRU if cfg.cell == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=d_model,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.layers,
            batch_first=True,
            dropout=dropout if cfg.layers > 1 else 0.0,
        )
        self.out = nn.Linear(cfg.hidden_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """Encode (B, T, D) with a (B, T) mask; unobserved steps output zero."""
        b, t, d = x.shape
        order = torch.argsort((~mask).to(torch.int8), dim=1, stable=True)
        compact = torch.gather(x, 1, order.unsqueeze(-1).expand(b, t, d))
        lengths = mask.sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(compact, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=t)
        h = self.norm(self.out(out))
        restored = torch.zeros_like(h).scatter(1, order.unsqueeze(-1).expand_as(h), h)
        return restored * mask.unsqueeze(-1).to(restored.dtype)
