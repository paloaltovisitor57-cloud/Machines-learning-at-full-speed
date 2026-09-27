"""Executable triple-barrier outcomes.

A mid-price forecast is not an edge; what matters is what a position of a given size
could actually have realised.  For a signal at time ``t``:

1. **entry** at ``t + latency`` against the pool state at that instant — buying
   ``size_sol`` through the real curve/AMM maths (fees + price impact);
2. the position is **marked to liquidation value** (selling the whole bag back into the
   pool, fees + impact included) at every subsequent pool state;
3. **exit** at the first of: take-profit barrier, stop-loss barrier, or the time barrier.
   The exit order also suffers ``latency``: it executes against the pool state
   ``latency`` seconds after the barrier was touched — in a fast crash that is exactly
   where edge disappears.

Barriers can be fixed or scaled by recent realised volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from nardis_neural.solana.amm import AMM_FEE, PUMP_FEE, Pool
from nardis_neural.solana.events import TokenEventLog

F64 = npt.NDArray[np.float64]


class BarrierSpec(BaseModel):
    """Executable triple-barrier parameters: take-profit, stop-loss, time barrier, latency and size."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    take_profit: float = Field(default=0.25, gt=0)
    """Net return (fraction of stake) at which the position is closed for a profit."""
    stop_loss: float = Field(default=0.15, gt=0, lt=1)
    """Net loss (fraction of stake) at which the position is closed."""
    max_hold_seconds: float = Field(default=180.0, gt=0)
    """Maximum holding time in seconds before the position is closed on time."""
    latency_seconds: float = Field(default=1.0, ge=0)
    """Delay in seconds between a signal or exit decision and its fill."""
    size_sol: float = Field(default=1.0, gt=0)
    """Position size in SOL, used for fees and price impact on entry and exit."""
    vol_scaled: bool = False
    """Scale barriers by realised volatility over ``vol_lookback_seconds`` (clipped)."""
    vol_lookback_seconds: float = Field(default=120.0, gt=0)
    """Look-back in seconds for the realised volatility used by ``vol_scaled``."""
    vol_multiple: float = Field(default=3.0, gt=0)
    """Barrier scale is this multiple of realised volatility, clipped to [0.5, 2]."""


@dataclass(frozen=True)
class BarrierOutcome:
    """Executable outcome of one signal under a :class:`BarrierSpec`."""

    net_return: float
    """Realised SOL out / SOL in − 1, after both fees and both price impacts."""
    exit_reason: Literal["take_profit", "stop_loss", "time", "dead"]
    hold_seconds: float
    entry_price: float
    exit_time: float

    @property
    def win(self) -> bool:
        """True when the realised net return is positive."""
        return self.net_return > 0


def _states(log: TokenEventLog) -> tuple[F64, F64, F64, F64]:
    r = log.reserves
    return (
        np.asarray(r["t"], dtype=np.float64),
        np.asarray(r["sol_reserve"], dtype=np.float64),
        np.asarray(r["token_reserve"], dtype=np.float64),
        np.asarray(r["virtual"], dtype=np.float64),
    )


def _pool(sol: float, tok: float, virt: float) -> Pool:
    return Pool(sol, tok, PUMP_FEE if virt > 0 else AMM_FEE, virt)


def liquidation_values(tokens: float, sol: F64, tok: F64, virt: F64) -> F64:
    """SOL received for selling ``tokens`` into each pool state (vectorised Pool.sell)."""
    fee = np.where(virt > 0, PUMP_FEE, AMM_FEE)
    gross = sol * tokens / (tok + tokens)
    return np.asarray(gross * (1 - fee), dtype=np.float64)


def triple_barrier(log: TokenEventLog, t: float, spec: BarrierSpec, data_end: float) -> BarrierOutcome | None:
    """Outcome of a position opened for a signal at ``t``; None if the history is too short."""
    entry_t = t + spec.latency_seconds
    horizon_end = entry_t + spec.max_hold_seconds
    if horizon_end + spec.latency_seconds > data_end:
        return None
    ts, sol, tok, virt = _states(log)
    if len(ts) == 0:
        return None
    i0 = int(np.searchsorted(ts, entry_t, side="right")) - 1
    if i0 < 0:
        return None
    pool = _pool(float(sol[i0]), float(tok[i0]), float(virt[i0]))
    if pool.sol <= 1e-9 or pool.tokens <= 1e-9:
        return BarrierOutcome(-1.0, "dead", 0.0, 0.0, entry_t)
    bought, after = pool.buy(spec.size_sol)
    # our own buy moved the pool; that shift persists in every later state we could sell into
    d_sol, d_tok = after.sol - pool.sol, pool.tokens - after.tokens
    tp, sl = spec.take_profit, spec.stop_loss
    if spec.vol_scaled:
        lo = int(np.searchsorted(ts, t - spec.vol_lookback_seconds, side="right"))
        lp = np.log(sol[lo : i0 + 1] / np.maximum(tok[lo : i0 + 1], 1e-12))
        vol = float(np.sqrt((np.diff(lp) ** 2).sum())) if len(lp) > 1 else 0.0
        scale = float(np.clip(spec.vol_multiple * vol, 0.5, 2.0)) if vol > 0 else 1.0
        tp, sl = tp * scale, min(sl * scale, 0.9)
    j = int(np.searchsorted(ts, horizon_end, side="right"))
    # marks after entry (the entry state itself included: price may already be through a barrier)
    idx = np.arange(i0, j)
    marks = liquidation_values(bought, sol[idx] + d_sol, tok[idx] - d_tok, virt[idx]) / spec.size_sol - 1.0
    hit_tp = np.flatnonzero(marks >= tp)
    hit_sl = np.flatnonzero(marks <= -sl)
    first_tp = int(hit_tp[0]) if len(hit_tp) else None
    first_sl = int(hit_sl[0]) if len(hit_sl) else None
    reason: Literal["take_profit", "stop_loss", "time", "dead"]
    if first_tp is None and first_sl is None:
        touch_t, reason = horizon_end, "time"
    elif first_sl is None or (first_tp is not None and first_tp < first_sl):
        assert first_tp is not None
        touch_t, reason = max(float(ts[idx[first_tp]]), entry_t), "take_profit"
    else:
        assert first_sl is not None
        touch_t, reason = max(float(ts[idx[first_sl]]), entry_t), "stop_loss"
    exit_t = touch_t + spec.latency_seconds if reason != "time" else horizon_end
    k = int(np.searchsorted(ts, exit_t, side="right")) - 1
    value = float(
        liquidation_values(bought, sol[k : k + 1] + d_sol, tok[k : k + 1] - d_tok, virt[k : k + 1])[0]
    )
    return BarrierOutcome(value / spec.size_sol - 1.0, reason, exit_t - entry_t, pool.price, exit_t)
