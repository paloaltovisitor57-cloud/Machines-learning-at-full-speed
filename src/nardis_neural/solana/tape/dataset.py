"""Causal tape extraction for training snapshots.

A fresh market replays the history strictly in time order; every requested ``(mint, t)``
is extracted after all events before ``t`` and before any event at or after it, the
same rule the feature dataset follows, so a tape never contains the future.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.config import SolanaConfig
from nardis_neural.solana.market import EventStore, SolanaMarket
from nardis_neural.solana.tape.features import TRADE_FEATURES, Tape, TapeSpec, extract_tape, stack_tapes


def replay_tapes(
    store: EventStore, cfg: SolanaConfig, spec: TapeSpec, requests: list[tuple[str, float]]
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.int64], npt.NDArray[np.bool_]]:
    """Tapes for every ``(mint, t)`` in ``requests`` (in request order), built causally."""
    empty = Tape(
        np.zeros((spec.max_trades, len(TRADE_FEATURES)), dtype=np.float32),
        np.zeros(spec.max_trades, dtype=np.int64),
        np.zeros(spec.max_trades, dtype=bool),
    )
    order = sorted(range(len(requests)), key=lambda i: requests[i][1])
    out: list[Tape] = [empty] * len(requests)
    market = SolanaMarket(cfg)
    k = 0

    def take(until: float) -> None:
        nonlocal k
        while k < len(order) and requests[order[k]][1] <= until:
            i = order[k]
            mint, t = requests[i]
            market.advance(t)
            if mint in market.tokens:
                out[i] = extract_tape(market.token(mint), market.wallets, t, spec, cfg)
            k += 1

    for e in store.sorted():
        take(e.t - 1e-9)
        market.ingest(e)
    take(float("inf"))
    return stack_tapes(out)
