"""Causal Solana market state and event-history persistence.

:class:`SolanaMarket` is fed events in time order (live, or by replaying history) and
maintains per-token logs, holder balances and wallet intelligence.  Features are only ever
computed from this state, so what the model sees at ``t`` is exactly what was knowable
at ``t``.  :class:`EventStore` is the durable, Parquet-backed event history.
"""

from __future__ import annotations

import heapq
from collections import OrderedDict, deque
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
from nardis_neural.solana.narrative import NarrativeBook
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
    """``(t, rank)``: at equal times launches, then transfers, migrations, liquidity changes, swaps."""
    return (e.t, _ORDER[type(e)])


@dataclass
class EventStore:
    """Time-sortable history of raw events, persisted as one Parquet table per type."""

    events: list[Event] = field(default_factory=list)

    def add(self, e: Event) -> None:
        """Append one event (any order; :meth:`sorted` orders them)."""
        self.events.append(e)

    def extend(self, events: Iterable[Event]) -> None:
        """Append several events."""
        self.events.extend(events)

    def __len__(self) -> int:
        return len(self.events)

    def sorted(self) -> list[Event]:
        """Events in time order (ties broken by :func:`event_sort_key`)."""
        return sorted(self.events, key=event_sort_key)

    def of_type(self, kind: type[Any]) -> list[Any]:
        """Events that are instances of ``kind``, in stored order."""
        return [e for e in self.events if isinstance(e, kind)]

    @property
    def end_time(self) -> float:
        """Latest event time (0 for an empty store)."""
        return max((e.t for e in self.events), default=0.0)

    def save(self, directory: str | Path) -> Path:
        """Write one Parquet file per event type into ``directory``; returns its path."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        for name, kind in EVENT_TYPES.items():
            rows = [asdict(e) for e in self.events if isinstance(e, kind)]
            if rows:
                pl.DataFrame(rows).write_parquet(path / f"{name}.parquet")
        return path

    @classmethod
    def load(cls, directory: str | Path) -> EventStore:
        """Load a store written by :meth:`save`."""
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


@dataclass
class FamilyStats:
    """Track record of a creator family (the creator's funder, or the creator itself)."""

    launches: int = 0
    """Finished (forgotten) launches; tokens still in memory are counted from ``live``."""
    rugs: int = 0
    graduations: int = 0
    best_peak: float = 1.0
    """Best peak multiple (vs launch price) among the family's finished (forgotten) tokens."""
    last_launch_t: float = float("-inf")
    live: list[str] = field(default_factory=list)
    """The family's tokens still held in memory, in launch order."""


class _Window:
    """Rolling sum of (time, value) pairs over the last ``seconds``."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.items: deque[tuple[float, float]] = deque()
        self.total = 0.0

    def add(self, t: float, v: float = 1.0) -> None:
        self.items.append((t, v))
        self.total += v

    def value(self, now: float) -> float:
        while self.items and self.items[0][0] <= now - self.seconds:
            self.total -= self.items.popleft()[1]
        return max(self.total, 0.0)


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
        if wallets is None:
            wallets = WalletIntel(
                *self.cfg.reputation_prior, self.cfg.funding_min_sol, self.cfg.hub_threshold
            )
            wallets.tail_prior = self.cfg.tail_prior
        self.wallets = wallets
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
        self.peak: dict[str, float] = {}
        """Running peak price multiple (vs launch price) of every token in memory."""
        self.families: dict[int, FamilyStats] = {}
        self.family_of: dict[str, int] = {}
        self._tail_pending: list[tuple[float, int, str, int, float, float]] = []
        self._tail_open: dict[str, int] = {}
        self._tail_waiting: dict[str, list[tuple[float, int, int]]] = {}
        """Per token: min-heap of (hit price, key, wallet) of early buys still waiting to hit."""
        self._tail_done: set[int] = set()
        """Keys of early buys already credited as hits (skipped when their horizon falls due)."""
        self._tail_seq = 0
        self.tail_updates = 0
        self.tail_early_hits = 0
        self.narrative = NarrativeBook()
        """Theme-word heat, copycats and recent-runner names (see :mod:`narrative`)."""
        """Tail successes credited the moment the price crossed the target, before the horizon."""
        self.heat = {
            "launches": _Window(600.0),
            "graduations": _Window(3600.0),
            "volume": _Window(300.0),
        }
        self._gone: OrderedDict[str, None] = OrderedDict()
        """Recently evicted mints: a later 'launch' of one is rejected, not treated as new."""

    # ------------------------------------------------------------------ reputation
    def _log_price_at(self, log: TokenEventLog, t: float) -> float:
        r = log.reserves
        idx = int(np.searchsorted(r["t"], t, side="right")) - 1
        if idx < 0:
            return float(np.log(log.launch.sol_reserve / log.launch.token_reserve))
        return float(np.log(r["sol_reserve"][idx] / max(r["token_reserve"][idx], 1e-12)))

    def _max_price(self, log: TokenEventLog, t0: float, t1: float) -> float:
        r = log.reserves
        rt = r["t"]
        lo, hi = int(np.searchsorted(rt, t0, side="left")), int(np.searchsorted(rt, t1, side="right"))
        if hi <= lo:
            return 0.0
        return float((r["sol_reserve"][lo:hi] / np.maximum(r["token_reserve"][lo:hi], 1e-12)).max())

    def _credit_tail_hits(self, mint: str, price: float) -> None:
        """Credit every early buy of ``mint`` whose target price has just been reached.

        A hit is known the moment it happens; only a miss needs the full tail horizon.
        """
        waiting = self._tail_waiting.get(mint)
        while waiting and waiting[0][0] <= price:
            _, key, wid = heapq.heappop(waiting)
            self.wallets.update_tail(wid, True)
            self._tail_done.add(key)
            self.tail_updates += 1
            self.tail_early_hits += 1

    def __setstate__(self, state: dict[str, Any]) -> None:
        # checkpoints written before early tail crediting existed
        state.setdefault("_tail_waiting", {})
        state.setdefault("_tail_done", set())
        state.setdefault("_tail_seq", len(state.get("_tail_pending", ())) + state.get("_seq", 0) + 1)
        state.setdefault("tail_early_hits", 0)
        state.setdefault("narrative", NarrativeBook())
        self.__dict__.update(state)

    def advance(self, now: float) -> int:
        """Resolve every queued entry whose reputation (or tail) horizon has elapsed by ``now``."""
        while self._tail_pending and self._tail_pending[0][0] <= now:
            due, key, mint, wid, entry_t, entry_price = heapq.heappop(self._tail_pending)
            self._tail_open[mint] -= 1
            if not self._tail_open[mint]:
                del self._tail_open[mint]
                self._tail_waiting.pop(mint, None)
            if key in self._tail_done:  # already credited as a hit when the price got there
                self._tail_done.discard(key)
                continue
            log = self.tokens.get(mint)
            if log is not None:
                hit = self._max_price(log, entry_t, due) >= self.cfg.tail_multiple * entry_price
                self.wallets.update_tail(wid, hit)
                self.tail_updates += 1
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
        """Apply one event, in time order, to the market state.

        First resolves reputation and tail entries due by ``e.t``, then updates token logs, wallets,
        rug attribution, peaks and heat.  Raises ``ValueError`` for a repeated (or evicted) launch or
        an out-of-order event, and ``KeyError`` for an event of an unknown token.
        """
        self.advance(e.t)
        if isinstance(e, TokenLaunch):
            if e.mint in self.tokens or e.mint in self._gone:
                raise ValueError(f"token {e.mint} launched twice")
            creator = self.wallets.id(e.creator, e.t)
            self.tokens[e.mint] = TokenEventLog(e, creator, slot_of(e.t, e.slot), venue=e.venue, last_t=e.t)
            self.peak[e.mint] = 1.0
            fam = self._family_key(creator)
            stats = self.families.setdefault(fam, FamilyStats())
            self.family_of[e.mint] = fam
            stats.live.append(e.mint)
            self.heat["launches"].add(e.t)
            self.narrative.on_launch(e.mint, e.t, e.name, e.symbol)
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
                price = e.sol_reserve / max(e.token_reserve, 1e-12)
                launch_price = log.launch.sol_reserve / max(log.launch.token_reserve, 1e-12)
                self.peak[e.mint] = max(self.peak.get(e.mint, 1.0), price / max(launch_price, 1e-30))
                self.narrative.on_peak(e.mint, e.t, self.peak[e.mint])
                self.heat["volume"].add(e.t, e.sol_amount)
                self._credit_tail_hits(e.mint, price)
                if e.is_buy and e.t - log.launch.t <= self.cfg.tail_entry_window:
                    key = self._tail_seq
                    self._tail_seq += 1
                    heapq.heappush(
                        self._tail_pending,
                        (e.t + self.cfg.tail_horizon_seconds, key, e.mint, wid, e.t, price),
                    )
                    heapq.heappush(
                        self._tail_waiting.setdefault(e.mint, []), (self.cfg.tail_multiple * price, key, wid)
                    )
                    self._tail_open[e.mint] = self._tail_open.get(e.mint, 0) + 1
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
                self.heat["graduations"].add(e.t)
        self.now = max(self.now, e.t)
        self.events_seen += 1

    def ingest_many(self, events: Iterable[Event]) -> None:
        """Ingest events in order (see :meth:`ingest`)."""
        for e in events:
            self.ingest(e)

    def rugged(self, mint: str) -> bool:
        """True once a rug (dev dump or LP pull) has been attributed to ``mint``."""
        return mint in self._rugged

    def token(self, mint: str) -> TokenEventLog:
        """The event log of ``mint``; ``KeyError`` if unknown or evicted."""
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
            if now - log.last_t > idle_seconds and m not in self._tail_open and (keep is None or not keep(m))
        ]
        for m in gone:
            log = self.tokens.pop(m)
            self.narrative.forget(m, now)
            fam = self.families.get(self.family_of.pop(m, -1))
            if fam is not None:  # fold the token into its family's finished record
                fam.launches += 1
                fam.last_launch_t = max(fam.last_launch_t, log.launch.t)
                fam.best_peak = max(fam.best_peak, self.peak.get(m, 1.0))
                fam.rugs += int(m in self._rugged)
                fam.graduations += int(log.migrated_at is not None)
                fam.live.remove(m)
            self.peak.pop(m, None)
            self._rugged.discard(m)
            self._gone[m] = None
        while len(self._gone) > 2_000_000:
            self._gone.popitem(last=False)
        self.evicted += len(gone)
        return gone

    # ------------------------------------------------------------------ creators & heat
    def _family_key(self, creator: int) -> int:
        """The creator's funder unless that funder is an exchange-like hub; else the creator."""
        funder = self.wallets.funder.get(creator)
        if funder is not None and self.wallets.funded_count.get(funder, 0) <= self.cfg.hub_threshold:
            return funder
        return creator

    def creator_record(self, mint: str, now: float) -> dict[str, float]:
        """Track record of ``mint``'s creator family as known at ``now``: its earlier launches
        still in memory plus every finished one (only data up to ``now`` is used)."""
        fam = self.families.get(self.family_of.get(mint, -1))
        if fam is None:
            return {"launches": 0.0, "best_peak": 1.0, "rugs": 0.0, "graduations": 0.0, "since_last": 1e7}
        launch_t = self.tokens[mint].launch.t
        prior = [m for m in fam.live if m != mint and self.tokens[m].launch.t < launch_t]
        finished = fam.launches
        launches = finished + len(prior)
        rugs = fam.rugs + sum(m in self._rugged for m in prior)
        grads = fam.graduations + sum(self.tokens[m].migrated_at is not None for m in prior)
        best = max([fam.best_peak if finished else 1.0, *(self.peak.get(m, 1.0) for m in prior)])
        last = max(
            [fam.last_launch_t if finished else float("-inf"), *(self.tokens[m].launch.t for m in prior)]
        )
        return {
            "launches": float(launches),
            "best_peak": float(best),
            "rugs": float(rugs),
            "graduations": float(grads),
            "since_last": float(launch_t - last) if np.isfinite(last) else 1e7,
        }

    def heat_values(self, now: float) -> dict[str, float]:
        """Market heat at ``now``: launches in the last 10 min, graduations in the last hour and SOL
        volume in the last 5 min.
        """
        return {k: w.value(now) for k, w in self.heat.items()}

    def active_tokens(self, now: float, max_idle_seconds: float = 600.0) -> list[str]:
        """Tokens that traded recently — candidates for assessment."""
        return [m for m, log in self.tokens.items() if now - log.last_t <= max_idle_seconds]
