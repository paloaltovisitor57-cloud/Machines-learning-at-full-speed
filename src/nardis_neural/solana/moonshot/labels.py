"""Executable moonshot outcomes: how far could a ticket bought *now* really have run?

For a signal at ``t`` a ticket of ``size_sol`` is bought at ``t + latency`` through the real
curve/AMM maths (fee + price impact).  From then on the position is marked to its
**liquidation multiple**: SOL received for selling the whole bag, fees and impact included,
divided by SOL paid.  Our own entry's reserve shift is carried forward (scaled through
liquidity adds and pulls), and every exit decision taken at ``s`` fills against the pool at
``s + latency``.

Two labels come out of one path:

* **peak multiple** ``M`` — the best liquidation multiple any exit could have realised
  within ``horizon_seconds``.  When the horizon runs past the end of the data while the
  token is still trading, the label is *right-censored*: we only know ``M ≥ observed``.
  The tail model's likelihood treats it so, which is also how models are trained causally
  (labels truncated at the training cutoff).
* **ladder multiple** — what a concrete take-profit ladder actually returned: sell
  ``ladder_fractions[i]`` of the original bag the first time the mark reaches
  ``ladder[i]``; once the mark has reached ``trail_activation`` sell the rest when it falls
  ``trail_drop`` below its running peak; before that sell everything at the
  ``stop_loss``; whatever is left at the horizon is sold then (or marked at the end of the
  data).  Each tranche is priced exactly for its own size and moves the pool for the next.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nardis_neural.solana.edge.barriers import _pool, _states, liquidation_values
from nardis_neural.solana.events import TokenEventLog

F64 = npt.NDArray[np.float64]
MIN_MULTIPLE = 1e-3


class MoonshotSpec(BaseModel):
    """Ticket size, latency, horizon, entry window and ladder-exit policy of moonshot outcomes."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    size_sol: float = Field(default=0.5, gt=0)
    """Ticket size in SOL, used for fees and price impact on entry and exits."""
    latency_seconds: float = Field(default=1.0, ge=0)
    """Delay in seconds between a signal or exit decision and its fill."""
    horizon_seconds: float = Field(default=6 * 3600.0, gt=0)
    """Seconds after entry over which the peak multiple and ladder exits are measured."""
    resolve_idle_seconds: float = Field(default=1800.0, gt=0)
    """A token with no activity for this long before the end of the data is treated as
    finished: its peak so far is final even if the horizon is incomplete (known causally)."""
    max_entry_age_seconds: float = Field(default=600.0, gt=0)
    """Moonshot tickets are only considered this early in a token's life."""
    min_entry_age_seconds: float = Field(default=20.0, ge=0)
    """Moonshot tickets are only considered once a token is at least this old (seconds)."""
    ladder: list[float] = Field(default_factory=lambda: [2.0, 10.0, 100.0, 1000.0])
    """Multiples of the stake at which ladder tranches are sold (strictly increasing)."""
    ladder_fractions: list[float] = Field(default_factory=lambda: [0.35, 0.15, 0.15, 0.15])
    """Fraction of the position sold at each ladder level (sum at most 1)."""
    trail_activation: float = Field(default=2.0, gt=1)
    """Multiple the ticket must reach before the trailing stop on the remainder activates."""
    trail_drop: float = Field(default=0.6, gt=0, lt=1)
    """Once active, the remainder exits when value falls this fraction below its running peak."""
    stop_loss: float = Field(default=0.5, gt=0, lt=1)
    """Before trail activation, the remainder exits when value falls by this fraction of stake."""
    levels: list[float] = Field(default_factory=lambda: [2.0, 5.0, 10.0, 100.0, 1000.0])
    """Multiples reported as P(M ≥ k)."""
    collapse_drop: float = Field(default=0.5, gt=0, lt=1)
    """A collapse is the ticket's value falling this fraction below its value at entry."""

    @model_validator(mode="after")
    def _check(self) -> MoonshotSpec:
        if len(self.ladder) != len(self.ladder_fractions):
            raise ValueError("ladder and ladder_fractions must have the same length")
        if any(b <= a for a, b in zip(self.ladder, self.ladder[1:], strict=False)):
            raise ValueError("ladder levels must be strictly increasing")
        if any(f < 0 for f in self.ladder_fractions) or sum(self.ladder_fractions) > 1 + 1e-9:
            raise ValueError("ladder fractions must be non-negative and sum to at most 1")
        return self


@dataclass(frozen=True)
class MoonshotOutcome:
    """Executable outcome of one ticket: peak, ladder and final multiples of the stake, with timing."""

    peak_multiple: float
    censored: bool
    """The horizon ran past the end of the data: ``peak_multiple`` is a lower bound."""
    ladder_multiple: float
    """SOL returned / SOL paid under the ladder policy (an open remainder is marked)."""
    final_multiple: float
    time_to_peak: float
    exit_time: float
    """When the ladder position was fully closed (or the end of the observed window)."""
    collapse_time: float | None = None
    """Seconds from entry until the value first fell ``collapse_drop`` below its entry value;
    None if that never happened within the observed window."""
    observed_seconds: float = 0.0
    """Length of the observed window after entry (what a collapse-free label is censored at)."""


def _shift_scale(log: TokenEventLog, ts: F64, sol: F64) -> F64:
    """How much of our entry's reserve shift is still in the pool at each state.

    Swaps leave it in place; a proportional liquidity add or pull scales it with the pool.
    """
    scale = np.ones(len(ts))
    for tl in np.asarray(log.liquidity["t"], dtype=np.float64):
        q = int(np.searchsorted(ts, tl, side="right")) - 1
        if 0 < q < len(ts) and sol[q - 1] > 0:
            scale[q:] *= max(sol[q] / sol[q - 1], 0.0)
    return scale


def moonshot_outcome(
    log: TokenEventLog, t: float, spec: MoonshotSpec, data_end: float
) -> MoonshotOutcome | None:
    """Outcome of a ticket for a signal at ``t`` using only data up to ``data_end``."""
    entry_t = t + spec.latency_seconds
    if entry_t + spec.latency_seconds > data_end:
        return None
    ts, sol, tok, virt = _states(log)
    i0 = int(np.searchsorted(ts, entry_t, side="right")) - 1
    if i0 < 0:
        return None
    pool = _pool(float(sol[i0]), float(tok[i0]), float(virt[i0]))
    if pool.sol <= 1e-9 or pool.tokens <= 1e-9:
        return MoonshotOutcome(0.0, False, 0.0, 0.0, 0.0, entry_t, collapse_time=0.0)
    bought, after = pool.buy(spec.size_sol)
    d_sol, d_tok = after.sol - pool.sol, pool.tokens - after.tokens
    horizon_end = entry_t + spec.horizon_seconds
    censored = horizon_end + spec.latency_seconds > data_end
    end = min(horizon_end, data_end - spec.latency_seconds)
    j = int(np.searchsorted(ts, end, side="right"))
    if censored and ts[j - 1] <= data_end - spec.resolve_idle_seconds:
        censored = False  # the token went quiet long ago: its run is over
    scale = _shift_scale(log, ts, sol)
    # an exit decided at state time s fills against the pool at s + latency
    decide_t = np.concatenate([np.maximum(ts[i0:j], entry_t), [end]])
    fill = np.searchsorted(ts, decide_t + spec.latency_seconds, side="right") - 1
    ds, dt = d_sol * scale[fill], d_tok * scale[fill]
    marks = liquidation_values(bought, sol[fill] + ds, tok[fill] - dt, virt[fill]) / spec.size_sol
    k_peak = int(np.argmax(marks))
    peak = float(max(marks[k_peak], 0.0))
    crashed = np.flatnonzero(marks <= (1 - spec.collapse_drop) * marks[0])
    collapse_time = float(decide_t[crashed[0]] - entry_t) if len(crashed) else None

    # ---- ladder policy on the same path
    run_max = np.maximum.accumulate(marks)
    stop_hits = np.flatnonzero((marks <= 1 - spec.stop_loss) & (run_max < spec.trail_activation))
    trail_hits = np.flatnonzero(
        (run_max >= spec.trail_activation) & (marks <= (1 - spec.trail_drop) * run_max)
    )
    close = min(
        int(stop_hits[0]) if len(stop_hits) else len(marks) - 1,
        int(trail_hits[0]) if len(trail_hits) else len(marks) - 1,
    )
    events: list[tuple[int, float]] = []
    for level, frac in zip(spec.ladder, spec.ladder_fractions, strict=True):
        hit = np.flatnonzero(marks[:close] >= level)
        if len(hit) and frac > 0:
            events.append((int(hit[0]), frac))
    events.sort()
    remaining = 1.0 - sum(f for _, f in events)
    events.append((close, remaining))
    proceeds = 0.0
    shift_sol, shift_tok = 0.0, 0.0  # our own tranche sales move the pool too
    for k, frac in events:
        if frac <= 0:
            continue
        q = frac * bought
        f = fill[k]
        s = sol[f] + ds[k] - shift_sol
        m = tok[f] - dt[k] + shift_tok
        got = float(liquidation_values(q, np.array([s]), np.array([m]), virt[f : f + 1])[0])
        proceeds += got
        shift_sol += s * q / (m + q)
        shift_tok += q
    return MoonshotOutcome(
        peak_multiple=max(peak, MIN_MULTIPLE) if peak > 0 else 0.0,
        censored=bool(censored),
        ladder_multiple=proceeds / spec.size_sol,
        final_multiple=float(marks[-1]),
        time_to_peak=float(decide_t[k_peak] - entry_t),
        exit_time=float(ts[fill[close]]) if fill[close] >= 0 else entry_t,
        collapse_time=collapse_time,
        observed_seconds=float(end - entry_t) if censored else float(spec.horizon_seconds),
    )


def ladder_payoff(peak: F64, spec: MoonshotSpec) -> F64:
    """Model-implied ladder multiple as a function of the peak multiple ``M``.

    Tranches fill at their level; the remainder exits ``trail_drop`` below the peak once
    ``trail_activation`` was reached, otherwise at the stop.  Used to turn a predicted
    distribution of ``M`` into an expected payoff; research reports how it compares with
    the exactly simulated ladder.
    """
    m = np.asarray(peak, dtype=np.float64)
    out = np.zeros_like(m)
    sold = np.zeros_like(m)
    for level, frac in zip(spec.ladder, spec.ladder_fractions, strict=True):
        reached = m >= level
        out += np.where(reached, frac * level, 0.0)
        sold += np.where(reached, frac, 0.0)
    rest = np.where(m >= spec.trail_activation, (1 - spec.trail_drop) * m, 1 - spec.stop_loss)
    return np.asarray(out + (1 - sold) * np.minimum(rest, m), dtype=np.float64)
