"""Causal Solana market state and event-history persistence.

:class:`SolanaMarket` is fed events in time order (live, or by replaying history) and
maintains per-token logs, holder balances and wallet intelligence.  Features are only ever
computed from this state, so what the model sees at ``t`` is exactly what was knowable
at ``t``.  :class:`EventStore` is the durable, Parquet-backed event history.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nardis_neural.solana.config import SolanaConfig
from nardis_neural.solana.events import (
    Event,
    LiquidityChange,
    Migration,
    Swap,
    TokenEventLog,
    TokenLaunch,
    Transfer,
    slot_of,
)
from nardis_neural.solana.wallets import WalletIntel

EVENT_TYPES: dict[str, type[Any]] = {
    "launches": TokenLaunch,
    "swaps": Swap,
    "liquidity": LiquidityChange,
    "migrations": Migration,
    "transfers": Transfer,
}
# at equal timestamps: launches first, transfers (funding) before the trades they enable
_ORDER = {TokenLaunch: 0, Transfer: 1, Migration: 2, LiquidityChange: 3, Swap: 4}


def event_sort_key(e: Event) -> tuple[float, int]:
    return (e.t, _ORDER[type(e)])


@dataclass
class EventStore:
    """Time-sortable history of raw events, persisted as one Parquet table per type."""

    events: list[Event] = field(default_factory=list)

    def add(self, e: Event) -> None:
        self.events.append(e)

    def extend(self, events: Iterable[Event]) -> None:
        self.events.extend(events)

    def __len__(self) -> int:
        return len(self.events)

    def sorted(self) -> list[Event]:
        return sorted(self.events, key=event_sort_key)

    def of_type(self, kind: type[Any]) -> list[Any]:
        return [e for e in self.events if isinstance(e, kind)]

    @property
    def end_time(self) -> float:
        return max((e.t for e in self.events), default=0.0)

    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, kind in EVENT_TYPES.items():
            rows = [asdict(e) for e in self.events if isinstance(e, kind)]
            if rows:
                pl.DataFrame(rows).write_parquet(path / f"{name}.parquet")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> EventStore:
        path = Path(directory)
        store = cls()
        for name, kind in EVENT_TYPES.items():
            f = path / f"{name}.parquet"
            if f.exists():
                store.extend(kind(**row) for row in pl.read_parquet(f).iter_rows(named=True))
        return store


def merge_streams(*streams: Iterable[Event]) -> Iterator[Event]:
    """Merge already time-sorted streams into one time-sorted stream."""
    return heapq.merge(*streams, key=event_sort_key)


class SolanaMarket:
    """Incremental market state built only from events seen so far.

    Besides per-token logs it learns wallet reputations *online and causally*: every buy
    is queued, and once ``reputation_horizon`` seconds have passed in market time the
    realised price move (from state that now exists) credits or debits the buyer.  Rugs
    are attributed as they happen: a creator dumping most of its bag or an LP pull marks
    the responsible wallet (and the creator's funding cluster for dev dumps).
    """

    def __init__(self, cfg: SolanaConfig | None = None, wallets: WalletIntel | None = None) -> None:
        self.cfg = cfg or SolanaConfig()
        self.tokens: dict[str, TokenEventLog] = {}
        self.wallets = wallets or WalletIntel(
            *self.cfg.reputation_prior, self.cfg.funding_min_sol, self.cfg.hub_threshold
        )
        self.now = 0.0
        self.events_seen = 0
        horizons = {h.name: h.seconds for h in self.cfg.horizons}
        if self.cfg.reputation_horizon not in horizons:
            raise ValueError(
                f"reputation_horizon {self.cfg.reputation_horizon!r} is not a configured horizon"
            )
        self.reputation_seconds = float(horizons[self.cfg.reputation_horizon])
        self._pending: list[tuple[float, int, str, int, float]] = []
        self._seq = 0
        self._rugged: set[str] = set()
        self.reputation_updates = 0
        self.evicted = 0
        self._gone: OrderedDict[str, None] = OrderedDict()
        """Recently evicted mints: a later 'launch' of one is rejected, not treated as new."""

    # ------------------------------------------------------------------ reputation
    def _log_price_at(self, log: TokenEventLog, t: float) -> float:
        r = log.reserves
        idx = int(np.searchsorted(r["t"], t, side="right")) - 1
        if idx < 0:
            return float(np.log(log.launch.sol_reserve / log.launch.token_reserve))
        return float(np.log(r["sol_reserve"][idx] / max(r["token_reserve"][idx], 1e-12)))

    def advance(self, now: float) -> int:
        """Resolve every queued entry whose reputation horizon has elapsed by ``now``."""
        done = 0
        while self._pending and self._pending[0][0] <= now:
            due, _, mint, wid, entry_lp = heapq.heappop(self._pending)
            ret = self._log_price_at(self.tokens[mint], due) - entry_lp
            self.wallets.update(wid, bool(ret > self.cfg.reputation_success_return))
            done += 1
        self.reputation_updates += done
        self.now = max(self.now, now)
        return done

    def _attribute_rugs(self, log: TokenEventLog, e: Swap | LiquidityChange, wid: int) -> None:
        if log.mint in self._rugged:
            return
        if isinstance(e, Swap) and not e.is_buy and wid == log.creator_id:
            bought = log.bought.get(wid, 0.0)
            if bought > 0 and log.sold.get(wid, 0.0) >= self.cfg.dev_dump_fraction * bought:
                self._rugged.add(log.mint)
                root = self.wallets.root(wid)
                for w in set(log.sold) | set(log.balances) | {wid}:
                    if self.wallets.root(w) == root:
                        self.wallets.mark_rug(w)
        elif isinstance(e, LiquidityChange) and e.sol_delta < 0:
            before = e.sol_reserve - e.sol_delta
            if before > 0 and -e.sol_delta >= self.cfg.rug_liquidity_drop * before:
                self._rugged.add(log.mint)
                self.wallets.mark_rug(wid)

    # ------------------------------------------------------------------ ingestion
    def ingest(self, e: Event) -> None:
        self.advance(e.t)
        if isinstance(e, TokenLaunch):
            if e.mint in self.tokens or e.mint in self._gone:
                raise ValueError(f"token {e.mint} launched twice")
            creator = self.wallets.id(e.creator, e.t)
            self.tokens[e.mint] = TokenEventLog(e, creator, slot_of(e.t, e.slot), venue=e.venue, last_t=e.t)
        elif isinstance(e, Transfer):
            self.wallets.add_transfer(e)
        else:
            log = self.tokens.get(e.mint)
            if log is None:
                raise KeyError(f"event for unknown token {e.mint}; ingest its TokenLaunch first")
            if isinstance(e, Swap):
                wid = self.wallets.id(e.wallet, e.t)
                log.add_swap(e, wid)
                self.wallets.record_trade(wid, e.t)
                if e.is_buy:
                    entry = float(np.log(e.sol_reserve / max(e.token_reserve, 1e-12)))
                    heapq.heappush(
                        self._pending, (e.t + self.reputation_seconds, self._seq, e.mint, wid, entry)
                    )
                    self._seq += 1
                self._attribute_rugs(log, e, wid)
            elif isinstance(e, LiquidityChange):
                wid = self.wallets.id(e.wallet, e.t)
                log.add_liquidity(e, wid)
                self._attribute_rugs(log, e, wid)
            else:
                log.add_migration(e)
        self.now = max(self.now, e.t)
        self.events_seen += 1

    def ingest_many(self, events: Iterable[Event]) -> None:
        for e in events:
            self.ingest(e)

    def rugged(self, mint: str) -> bool:
        return mint in self._rugged

    def token(self, mint: str) -> TokenEventLog:
        if mint not in self.tokens:
            raise KeyError(f"unknown token {mint}")
        return self.tokens[mint]

    def evict(self, now: float, idle_seconds: float, keep: Callable[[str], bool] | None = None) -> list[str]:
        """Forget tokens that have been quiet for ``idle_seconds`` (bounded memory for long streams).

        What the market learned from them stays: wallet reputations, rug marks and funding
        clusters live in :class:`WalletIntel`.  Their pending reputation entries have already
        resolved (the idle period exceeds the reputation horizon).  Later events for an
        evicted token are rejected like any unknown token.
        """
        if idle_seconds <= self.reputation_seconds:
            raise ValueError("idle_seconds must exceed the reputation horizon")
        self.advance(now)
        gone = [
            m
            for m, log in self.tokens.items()
            if now - log.last_t > idle_seconds and (keep is None or not keep(m))
        ]
        for m in gone:
            del self.tokens[m]
            self._rugged.discard(m)
            self._gone[m] = None
        while len(self._gone) > 2_000_000:
            self._gone.popitem(last=False)
        self.evicted += len(gone)
        return gone

    def active_tokens(self, now: float, max_idle_seconds: float = 600.0) -> list[str]:
        """Tokens that traded recently — candidates for assessment."""
        return [m for m, log in self.tokens.items() if now - log.last_t <= max_idle_seconds]
