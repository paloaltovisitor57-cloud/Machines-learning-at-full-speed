"""The trade tape: the last ``max_trades`` individual trades of a token, as the model sees them.

Aggregates (per-minute bars, holder shares) throw away *who* traded and *in what order*.
The tape keeps both.  Every trade becomes a vector of :data:`TRADE_FEATURES` plus a
**stable wallet bucket** (a hash of the wallet address, so the learned wallet embedding
means the same wallet in every workspace, replay and live session).

Everything is computed from state known at ``now``: trades at or before ``now`` only, and
wallet attributes (reputation, runner skill, cluster, freshness) as the market knows them
at ``now``.  Tapes are left-padded, so the most recent trade is always the last position.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from nardis_neural.solana.config import SolanaConfig
from nardis_neural.solana.events import TokenEventLog
from nardis_neural.solana.wallets import WalletIntel

F32 = npt.NDArray[np.float32]
I64 = npt.NDArray[np.int64]

TRADE_FEATURES: tuple[str, ...] = (
    "is_buy",
    "sol_log",
    "signed_flow",
    "dt_prev_log",
    "age_log",
    "since_now_log",
    "price_vs_now",
    "priority_fee_log",
    "jito",
    "wallet_skill",
    "wallet_tail_skill",
    "is_creator",
    "creator_cluster",
    "fresh_wallet",
    "cluster_size_log",
    "wallet_trades_log",
    "first_trade_in_token",
    "launch_slots",
)

TRADE_FEATURE_DOCS: dict[str, str] = {
    "is_buy": "1 for a buy, 0 for a sell",
    "sol_log": "log1p of the trade's SOL amount",
    "signed_flow": "log1p of SOL, positive for buys and negative for sells",
    "dt_prev_log": "log1p of seconds since the previous trade (since launch for the first)",
    "age_log": "log1p of the token's age at the trade",
    "since_now_log": "log1p of seconds between the trade and now",
    "price_vs_now": "log price after the trade minus log price now",
    "priority_fee_log": "log1p of the priority fee (micro-SOL)",
    "jito": "1 if the trade paid a Jito tip",
    "wallet_skill": "wallet's 60-s reputation skill as known now",
    "wallet_tail_skill": "wallet's runner skill as known now",
    "is_creator": "1 if the trader is the token's creator",
    "creator_cluster": "1 if the trader shares the creator's funding cluster",
    "fresh_wallet": "1 if the trader was funded within fresh_wallet_seconds before the trade",
    "cluster_size_log": "log1p of the trader's funding-cluster size",
    "wallet_trades_log": "log1p of the trader's lifetime trades (bot signal)",
    "first_trade_in_token": "1 on the trader's first trade in this token",
    "launch_slots": "1 if the trade landed within sniper_slots of the launch slot",
}


class TapeSpec(BaseModel):
    """Shape of the trade tape and of the wallet-embedding hash space."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    max_trades: int = Field(default=96, ge=1)
    """Most recent trades kept per snapshot (left-padded when fewer)."""
    wallet_buckets: int = Field(default=131_072, ge=2)
    """Hash buckets of the wallet embedding (bucket 0 is padding)."""


@dataclass(frozen=True)
class Tape:
    """One snapshot's tape: trade features, wallet buckets and the observed-trade mask."""

    x: F32
    """(max_trades, len(TRADE_FEATURES))."""
    wallets: I64
    """(max_trades,) wallet hash buckets, 0 on padding."""
    mask: npt.NDArray[np.bool_]
    """(max_trades,) True where a trade is present."""


def wallet_bucket(address: str, buckets: int) -> int:
    """Stable hash bucket of a wallet address in ``[1, buckets)`` (0 is reserved for padding)."""
    h = int.from_bytes(hashlib.blake2b(address.encode(), digest_size=8).digest(), "little")
    return 1 + h % (buckets - 1)


def extract_tape(
    log: TokenEventLog, wallets: WalletIntel, now: float, spec: TapeSpec, cfg: SolanaConfig | None = None
) -> Tape:
    """The last ``spec.max_trades`` trades at or before ``now``, left-padded."""
    cfg = cfg or SolanaConfig()
    n_max, f = spec.max_trades, len(TRADE_FEATURES)
    x = np.zeros((n_max, f), dtype=np.float32)
    wb = np.zeros(n_max, dtype=np.int64)
    mask = np.zeros(n_max, dtype=bool)
    s = log.swaps
    t_all = s["t"]
    end = int(np.searchsorted(t_all, now, side="right"))
    start = max(0, end - n_max)
    n = end - start
    if n == 0:
        return Tape(x, wb, mask)
    sl = slice(start, end)
    t = t_all[sl]
    wal = s["wallet"][sl].astype(np.int64)
    buy = s["is_buy"][sl].astype(bool)
    sol = s["sol"][sl].astype(np.float64)
    lp = np.log(s["sol_reserve"][sl] / np.maximum(s["token_reserve"][sl], 1e-12))
    lp_now = float(np.log(s["sol_reserve"][end - 1] / max(float(s["token_reserve"][end - 1]), 1e-12)))
    prev_t = np.concatenate([[t_all[start - 1] if start > 0 else log.launch.t], t[:-1]])
    uniq, first_pos = np.unique(s["wallet"][:end], return_index=True)
    first_of = first_pos[np.searchsorted(uniq, wal)]  # each trader's first trade index in this token
    creator = log.creator_id
    croot = wallets.root(creator)
    slots = s["slot"][sl]
    cols = {
        "is_buy": buy.astype(np.float64),
        "sol_log": np.log1p(sol),
        "signed_flow": np.where(buy, 1.0, -1.0) * np.log1p(sol),
        "dt_prev_log": np.log1p(np.maximum(t - prev_t, 0.0)),
        "age_log": np.log1p(np.maximum(t - log.launch.t, 0.0)),
        "since_now_log": np.log1p(np.maximum(now - t, 0.0)),
        "price_vs_now": lp - lp_now,
        "priority_fee_log": np.log1p(s["priority_fee"][sl] * 1e6),
        "jito": (s["jito_tip"][sl] > 0).astype(np.float64),
        "wallet_skill": wallets.skills(wal),
        "wallet_tail_skill": wallets.tail_skills(wal),
        "is_creator": (wal == creator).astype(np.float64),
        "creator_cluster": np.array([wallets.root(int(w)) == croot for w in wal], dtype=np.float64),
        "fresh_wallet": np.array(
            [
                wallets.is_fresh(int(w), float(tt), cfg.fresh_wallet_seconds)
                for w, tt in zip(wal, t, strict=True)
            ],
            dtype=np.float64,
        ),
        "cluster_size_log": np.log1p([wallets.cluster_size(int(w)) for w in wal]),
        "wallet_trades_log": np.log1p([wallets.trades[int(w)] for w in wal]),
        "first_trade_in_token": (first_of == start + np.arange(n)).astype(np.float64),
        "launch_slots": (slots - log.launch_slot <= cfg.sniper_slots).astype(np.float64),
    }
    x[n_max - n :] = np.nan_to_num(np.stack([cols[c] for c in TRADE_FEATURES], axis=1)).astype(np.float32)
    wb[n_max - n :] = [wallet_bucket(wallets.names[int(w)], spec.wallet_buckets) for w in wal]
    mask[n_max - n :] = True
    return Tape(x, wb, mask)


def stack_tapes(tapes: list[Tape]) -> tuple[F32, I64, npt.NDArray[np.bool_]]:
    """Batch tapes into ``(N, T, F)`` features, ``(N, T)`` wallet buckets and ``(N, T)`` masks."""
    return (
        np.stack([tp.x for tp in tapes]).astype(np.float32),
        np.stack([tp.wallets for tp in tapes]).astype(np.int64),
        np.stack([tp.mask for tp in tapes]),
    )
