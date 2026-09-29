"""Capital allocator: from per-token signals to a sized, constrained book (advice only).

The signals say *how good* a ticket looks.  Compounding capital also needs *how much*:
too little and edge is wasted, too much and one bad hour ends the account.  The allocator
turns every candidate into a recommended stake:

``fraction = lottery Kelly × trust × uncertainty discount × track-record factor × drawdown governor``

then enforces, in order of signal quality:

* per-position cap (share of equity) and a **liquidity cap** (a share of the pool's real
  SOL, so our own impact stays small);
* a cap per **creator family** (serial deployers are one correlated bet);
* a cap on total exposure and on the number of concurrent positions;
* the **drawdown governor** — stakes shrink linearly as equity falls below its peak and stop
  at ``max_drawdown`` — and a **daily loss stop**.

It never places an order; it returns stakes for the trading system to accept or ignore.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel, ConfigDict, Field


class AllocatorConfig(BaseModel):
    """Sizing and risk limits of the capital allocator."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    kelly_scale: float = Field(default=1.0, gt=0)
    """Multiplier on the signal's (already fractional, capped) lottery-Kelly fraction."""
    max_position_fraction: float = Field(default=0.04, gt=0, le=1)
    """Largest stake in one token as a share of equity."""
    liquidity_fraction: float = Field(default=0.02, gt=0, le=1)
    """Largest stake as a share of the pool's real SOL liquidity (bounds our own impact)."""
    max_family_fraction: float = Field(default=0.08, gt=0, le=1)
    """Largest combined stake in one creator family as a share of equity."""
    max_total_exposure: float = Field(default=0.3, gt=0, le=1)
    """Largest share of equity in open positions at once."""
    max_positions: int = Field(default=12, ge=1)
    """Largest number of concurrent positions."""
    min_stake_sol: float = Field(default=0.02, ge=0)
    """Stakes below this are dropped (fees and rent dominate)."""
    uncertainty_scale: float = Field(default=0.15, gt=0)
    """Epistemic spread at which the stake is discounted by a factor e."""
    max_drawdown: float = Field(default=0.35, gt=0, lt=1)
    """Drawdown from peak equity at which new stakes reach zero (linear governor)."""
    daily_loss_stop: float = Field(default=0.15, gt=0, lt=1)
    """No new positions for the rest of the day after losing this share of the day's opening equity."""


@dataclass(frozen=True)
class Signal:
    """What the allocator needs to know about one candidate ticket."""

    mint: str
    t: float
    expected_multiple: float
    lottery_kelly: float
    trust: float = 1.0
    epistemic: float = 0.0
    liquidity_sol: float = math.inf
    family: str = ""
    vetoed: bool = False


@dataclass
class BookState:
    """Equity, open stakes and the governors' reference points."""

    equity: float
    peak_equity: float
    day: int = -1
    day_start_equity: float = 0.0
    open_stakes: dict[str, float] = field(default_factory=dict)
    open_family: dict[str, str] = field(default_factory=dict)

    @property
    def exposure(self) -> float:
        """SOL currently committed to open positions."""
        return sum(self.open_stakes.values())


@dataclass(frozen=True)
class Allocation:
    """Recommended stake for one signal and the binding reason when it was cut."""

    mint: str
    stake_sol: float
    fraction: float
    reason: str


class CapitalAllocator:
    """Sizes candidate tickets under Kelly, uncertainty, liquidity, correlation and drawdown limits."""

    def __init__(self, cfg: AllocatorConfig | None = None, track_record: float = 1.0) -> None:
        self.cfg = cfg or AllocatorConfig()
        self.track_record = track_record
        """Realised / predicted payoff of past tickets (1 = the model has been right), clipped
        to [0.25, 1.25]; it scales every stake."""

    def governor(self, state: BookState) -> float:
        """Stake multiplier from the drawdown governor (1 at the peak, 0 at ``max_drawdown``)."""
        dd = 1.0 - state.equity / max(state.peak_equity, 1e-12)
        return max(0.0, 1.0 - dd / self.cfg.max_drawdown)

    def base_fraction(self, s: Signal) -> float:
        """Uncapped fraction of equity for one signal before portfolio limits."""
        if s.vetoed or s.expected_multiple < 1.0 or s.lottery_kelly <= 0:
            return 0.0
        record = min(max(self.track_record, 0.25), 1.25)
        discount = math.exp(-max(s.epistemic, 0.0) / self.cfg.uncertainty_scale)
        return s.lottery_kelly * self.cfg.kelly_scale * max(min(s.trust, 1.0), 0.0) * discount * record

    def allocate(self, signals: list[Signal], state: BookState) -> list[Allocation]:
        """Stakes for ``signals`` (best expected edge first) given the current book."""
        cfg = self.cfg
        day = int(signals[0].t // 86400) if signals else state.day
        if day != state.day:
            state.day, state.day_start_equity = day, state.equity
        stopped = state.equity < state.day_start_equity * (1 - cfg.daily_loss_stop)
        gov = self.governor(state)
        ranked = sorted(signals, key=lambda s: -(s.expected_multiple - 1.0) * self.base_fraction(s))
        out: list[Allocation] = []
        exposure = state.exposure
        n_open = len(state.open_stakes)
        family_exp: dict[str, float] = {}
        for m, f in state.open_family.items():
            family_exp[f] = family_exp.get(f, 0.0) + state.open_stakes.get(m, 0.0)
        for s in ranked:
            frac = self.base_fraction(s) * gov
            reason = "sized"
            if s.mint in state.open_stakes:
                out.append(Allocation(s.mint, 0.0, 0.0, "already open"))
                continue
            if stopped:
                out.append(Allocation(s.mint, 0.0, 0.0, "daily loss stop"))
                continue
            if frac <= 0:
                why = "drawdown governor" if gov <= 0 else "no edge"
                out.append(Allocation(s.mint, 0.0, 0.0, why))
                continue
            stake = min(frac, cfg.max_position_fraction) * state.equity
            if frac > cfg.max_position_fraction:
                reason = "position cap"
            if stake > cfg.liquidity_fraction * s.liquidity_sol:
                stake, reason = cfg.liquidity_fraction * s.liquidity_sol, "liquidity cap"
            fam_room = (
                cfg.max_family_fraction * state.equity - family_exp.get(s.family, 0.0) if s.family else stake
            )
            if stake > fam_room:
                stake, reason = max(fam_room, 0.0), "family cap"
            room = cfg.max_total_exposure * state.equity - exposure
            if stake > room:
                stake, reason = max(room, 0.0), "exposure cap"
            if n_open >= cfg.max_positions:
                stake, reason = 0.0, "position count cap"
            if stake < cfg.min_stake_sol:
                out.append(Allocation(s.mint, 0.0, 0.0, reason + " (below minimum)" if stake > 0 else reason))
                continue
            exposure += stake
            n_open += 1
            if s.family:
                family_exp[s.family] = family_exp.get(s.family, 0.0) + stake
            out.append(Allocation(s.mint, stake, stake / state.equity, reason))
        return out
