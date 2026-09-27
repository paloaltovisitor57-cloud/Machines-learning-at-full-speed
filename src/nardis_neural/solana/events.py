"""Solana market events and fast columnar per-token logs.

The trading system (or an indexer / Geyser stream) converts decoded on-chain activity into
these small immutable events.  :class:`TokenEventLog` stores them column-wise in growable
NumPy arrays so window statistics are vectorised slices, not Python loops.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.amm import (
    AMM_FEE,
    PUMP_FEE,
    PUMP_VIRTUAL_SOL,
    TOTAL_SUPPLY,
    Pool,
)

Venue = Literal["pump_fun", "pumpswap", "raydium", "orca", "meteora"]
VENUES: tuple[Venue, ...] = ("pump_fun", "pumpswap", "raydium", "orca", "meteora")
SLOT_SECONDS = 0.4


@dataclass(frozen=True)
class TokenLaunch:
    """A new SPL token / pool became tradable."""

    mint: str
    t: float
    creator: str
    venue: Venue = "pump_fun"
    supply: float = TOTAL_SUPPLY
    sol_reserve: float = PUMP_VIRTUAL_SOL
    token_reserve: float = 1_073_000_000.0
    mint_authority_revoked: bool = True
    freeze_authority_revoked: bool = True
    lp_burned_fraction: float = 0.0
    slot: int = -1


@dataclass(frozen=True)
class Swap:
    mint: str
    t: float
    wallet: str
    is_buy: bool
    sol_amount: float
    token_amount: float
    sol_reserve: float
    """Pricing reserves *after* the swap (virtual reserves on a bonding curve)."""
    token_reserve: float
    priority_fee: float = 0.0
    """Priority fee paid, in SOL."""
    jito_tip: float = 0.0
    slot: int = -1


@dataclass(frozen=True)
class LiquidityChange:
    """LP add (positive deltas) or removal (negative deltas)."""

    mint: str
    t: float
    wallet: str
    sol_delta: float
    token_delta: float
    sol_reserve: float
    token_reserve: float
    slot: int = -1


@dataclass(frozen=True)
class Migration:
    """Bonding-curve graduation: trading moves to an AMM pool."""

    mint: str
    t: float
    venue: Venue
    sol_reserve: float
    token_reserve: float
    slot: int = -1


@dataclass(frozen=True)
class Transfer:
    """Native SOL transfer between wallets (used for funding / sybil analysis)."""

    t: float
    source: str
    dest: str
    sol_amount: float
    slot: int = -1


Event = TokenLaunch | Swap | LiquidityChange | Migration | Transfer


def event_time(e: Event) -> float:
    return e.t


def slot_of(t: float, slot: int) -> int:
    return slot if slot >= 0 else int(t / SLOT_SECONDS)


class Columns:
    """Append-only columnar table backed by amortised-growth NumPy arrays."""

    def __init__(self, schema: dict[str, Any], capacity: int = 64) -> None:
        self.schema = schema
        self.n = 0
        self._data: dict[str, npt.NDArray[Any]] = {k: np.zeros(capacity, dtype=d) for k, d in schema.items()}

    def __len__(self) -> int:
        return self.n

    def append(self, **row: Any) -> None:
        if self.n == len(next(iter(self._data.values()))):
            for k in self._data:
                self._data[k] = np.concatenate([self._data[k], np.zeros_like(self._data[k])])
        for k, v in row.items():
            self._data[k][self.n] = v
        self.n += 1

    def __getitem__(self, name: str) -> npt.NDArray[Any]:
        return self._data[name][: self.n]

    def to_dict(self) -> dict[str, npt.NDArray[Any]]:
        return {k: self[k].copy() for k in self.schema}


SWAP_SCHEMA = {
    "t": np.float64,
    "slot": np.int64,
    "wallet": np.int64,
    "is_buy": np.bool_,
    "sol": np.float64,
    "tokens": np.float64,
    "sol_reserve": np.float64,
    "token_reserve": np.float64,
    "priority_fee": np.float64,
    "jito_tip": np.float64,
    "venue": np.int8,
}
RESERVE_SCHEMA = {
    "t": np.float64,
    "sol_reserve": np.float64,
    "token_reserve": np.float64,
    "virtual": np.float64,
}
LIQUIDITY_SCHEMA = {"t": np.float64, "wallet": np.int64, "sol_delta": np.float64, "token_delta": np.float64}


@dataclass
class TokenEventLog:
    """Everything observed so far about one token, plus incremental holder state."""

    launch: TokenLaunch
    creator_id: int
    launch_slot: int
    swaps: Columns = field(default_factory=lambda: Columns(SWAP_SCHEMA))
    reserves: Columns = field(default_factory=lambda: Columns(RESERVE_SCHEMA))
    liquidity: Columns = field(default_factory=lambda: Columns(LIQUIDITY_SCHEMA))
    balances: dict[int, float] = field(default_factory=dict)
    bought: dict[int, float] = field(default_factory=dict)
    sold: dict[int, float] = field(default_factory=dict)
    first_buy_slot: dict[int, int] = field(default_factory=dict)
    venue: Venue = "pump_fun"
    migrated_at: float | None = None
    last_t: float = 0.0

    @property
    def mint(self) -> str:
        return self.launch.mint

    @property
    def virtual_sol(self) -> float:
        return PUMP_VIRTUAL_SOL if self.venue == "pump_fun" else 0.0

    def pool(self) -> Pool:
        if len(self.reserves):
            sol, tok = float(self.reserves["sol_reserve"][-1]), float(self.reserves["token_reserve"][-1])
        else:
            sol, tok = self.launch.sol_reserve, self.launch.token_reserve
        fee = PUMP_FEE if self.venue == "pump_fun" else AMM_FEE
        return Pool(sol, tok, fee, self.virtual_sol)

    def _check_time(self, t: float) -> None:
        if t + 1e-9 < self.last_t:
            raise ValueError(f"{self.mint}: event at {t} is older than last event {self.last_t}")
        self.last_t = t

    def add_swap(self, s: Swap, wallet_id: int) -> None:
        self._check_time(s.t)
        slot = slot_of(s.t, s.slot)
        self.swaps.append(
            t=s.t,
            slot=slot,
            wallet=wallet_id,
            is_buy=s.is_buy,
            sol=s.sol_amount,
            tokens=s.token_amount,
            sol_reserve=s.sol_reserve,
            token_reserve=s.token_reserve,
            priority_fee=s.priority_fee,
            jito_tip=s.jito_tip,
            venue=VENUES.index(self.venue),
        )
        self.reserves.append(
            t=s.t, sol_reserve=s.sol_reserve, token_reserve=s.token_reserve, virtual=self.virtual_sol
        )
        sign = 1.0 if s.is_buy else -1.0
        self.balances[wallet_id] = max(self.balances.get(wallet_id, 0.0) + sign * s.token_amount, 0.0)
        book = self.bought if s.is_buy else self.sold
        book[wallet_id] = book.get(wallet_id, 0.0) + s.token_amount
        if s.is_buy and wallet_id not in self.first_buy_slot:
            self.first_buy_slot[wallet_id] = slot

    def add_liquidity(self, c: LiquidityChange, wallet_id: int) -> None:
        self._check_time(c.t)
        self.liquidity.append(t=c.t, wallet=wallet_id, sol_delta=c.sol_delta, token_delta=c.token_delta)
        self.reserves.append(
            t=c.t, sol_reserve=c.sol_reserve, token_reserve=c.token_reserve, virtual=self.virtual_sol
        )

    def add_migration(self, m: Migration) -> None:
        self._check_time(m.t)
        self.venue = m.venue
        self.migrated_at = m.t
        self.reserves.append(t=m.t, sol_reserve=m.sol_reserve, token_reserve=m.token_reserve, virtual=0.0)
