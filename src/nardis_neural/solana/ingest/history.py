"""Stream historical chain activity forward in time, storing nothing.

Works against any endpoint that serves ``getSignaturesForAddress`` and ``getTransaction``
for old slots, for example an Old Faithful RPC or an archival RPC provider.  Only the
watched programs' transactions are fetched, never whole blocks.

Signature listings run newest → oldest, but replay must run oldest → newest.  So:

1. **Boundary pass**: page each program's signatures backwards once, from ``end_time``
   to ``start_time``, and remember only the cursors at every ``segment_seconds`` edge.
   Memory is a few signatures per segment.
2. **Replay pass**: for each segment, oldest first, list its signatures between the two
   cursors, order them by slot, fetch the transactions with a small worker pool, decode
   them and yield the events.  A segment's signatures are the only thing ever held.

Historical replays use no mint-account lookups (``mint_info=None``): today's authority
state would leak the future into old snapshots.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from nardis_neural.solana.events import Event
from nardis_neural.solana.ingest.decoder import TransactionDecoder
from nardis_neural.solana.ingest.pumpfun import PUMP_FUN_PROGRAM
from nardis_neural.solana.ingest.rpc import SolanaRpc
from nardis_neural.solana.ingest.stream import DEFAULT_PROGRAMS
from nardis_neural.solana.market import event_sort_key


@dataclass
class _Edge:
    t: float
    newer: str | None = None
    """Oldest signature at or after the edge (listing ``before`` it starts below the edge)."""
    older: str | None = None
    """Newest signature before the edge (listing ``until`` it stops at the edge)."""


@dataclass
class HistoryWalker:
    """Replays the watched programs' history from ``start_time`` to ``end_time`` (Unix seconds),
    oldest first, one ``segment_seconds`` segment at a time.
    """

    rpc: SolanaRpc
    start_time: float
    end_time: float
    programs: list[str] = field(default_factory=lambda: list(DEFAULT_PROGRAMS))
    segment_seconds: float = 3600.0
    workers: int = 8
    page_size: int = 1000
    seek: bool = True
    """Start listing at ``end_time`` (a signature from a block found by slot search) instead of
    paging back from the chain tip; makes old windows (archival RPC, Old Faithful) reachable."""
    decoder: TransactionDecoder = field(default_factory=TransactionDecoder)
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "segments": 0,
            "signatures": 0,
            "transactions": 0,
            "events": 0,
            "decode_errors": 0,
        }
    )

    def __post_init__(self) -> None:
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be after start_time")

    @staticmethod
    def _now() -> float:
        return time.time()

    def _edges(self) -> list[float]:
        edges, t = [], self.start_time
        while t < self.end_time:
            edges.append(t)
            t += self.segment_seconds
        return [*edges, self.end_time]

    def _boundaries(self, program: str, edges: list[float]) -> list[_Edge]:
        """Cursor signatures around every edge, from one backwards pass over the listing."""
        marks = [_Edge(t) for t in edges]
        k = len(marks) - 1  # next edge to cross, walking backwards in time
        prev: str | None = None
        before: str | None = None
        if self.seek and self.end_time < self._now() - 600:
            before = self.rpc.signature_near(self.end_time + 30.0)
        while k >= 0:
            page = self.rpc.get_signatures(program, before=before, limit=self.page_size)
            for s in page:
                bt = float(s.get("blockTime") or 0.0)
                while k >= 0 and bt < marks[k].t:
                    marks[k].newer, marks[k].older = prev, s["signature"]
                    k -= 1
                if k < 0:
                    break
                prev = s["signature"]
            if len(page) < self.page_size:
                break  # reached the beginning of the program's history
            before = page[-1]["signature"]
        return marks

    def _segment_signatures(self, program: str, lo: _Edge, hi: _Edge) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        before = hi.newer
        while True:
            page = self.rpc.get_signatures(program, before=before, until=lo.older, limit=self.page_size)
            out.extend(s for s in page if lo.t <= float(s.get("blockTime") or 0.0) < hi.t)
            if len(page) < self.page_size:
                return out
            before = page[-1]["signature"]

    def _decode(self, tx: dict[str, Any]) -> list[Event]:
        """Decode one transaction; a transaction the decoder cannot handle is counted and skipped
        rather than aborting the whole segment."""
        try:
            return list(self.decoder.decode(tx))
        except (ArithmeticError, ValueError, KeyError, IndexError, TypeError):
            self.stats["decode_errors"] += 1
            return []

    def events(self) -> Iterator[Event]:
        """Yield decoded events in time order, segment by segment, updating :attr:`stats`.

        Failed transactions and signatures listed by several watched programs are skipped.
        """
        edges = self._edges()
        bounds = {p: self._boundaries(p, edges) for p in self.programs}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for k in range(len(edges) - 1):
                seen: set[str] = set()
                sigs: list[dict[str, Any]] = []
                for p in self.programs:
                    for s in self._segment_signatures(p, bounds[p][k], bounds[p][k + 1]):
                        if s.get("err") is None and s["signature"] not in seen:
                            seen.add(s["signature"])
                            sigs.append(s)
                sigs.sort(key=lambda s: (int(s.get("slot", 0)), float(s.get("blockTime") or 0.0)))
                txs = [tx for tx in pool.map(lambda s: self.rpc.get_transaction(s["signature"]), sigs) if tx]
                events = sorted((e for tx in txs for e in self._decode(tx)), key=event_sort_key)
                self.stats["segments"] += 1
                self.stats["signatures"] += len(sigs)
                self.stats["transactions"] += len(txs)
                self.stats["events"] += len(events)
                yield from events


def chain(*sources: Iterable[Event]) -> Iterator[Event]:
    """Concatenate event sources in order (e.g. a history replay followed by the live stream)."""
    for src in sources:
        yield from src


def clean_history(events: Iterable[Event]) -> tuple[list[Event], dict[str, int]]:
    """Keep only tokens whose pump.fun creation is inside the history, SOL-priced throughout.

    Removes tokens first seen by a trade (their launch is inferred, so their age and early
    features would be wrong), pools that existed before the window (e.g. SOL/USDC touched by
    the same transactions), and tokens traded on curves that report no SOL reserves.
    Events without a mint (SOL transfers) are kept.
    """
    from nardis_neural.solana.events import Swap, TokenLaunch

    ordered = sorted(events, key=event_sort_key)
    non_sol = {e.mint for e in ordered if isinstance(e, Swap) and e.sol_reserve <= 0}
    created = {
        e.mint
        for e in ordered
        if isinstance(e, TokenLaunch) and e.creator != "unknown" and e.venue == "pump_fun"
    } - non_sol
    seen: set[str] = set()
    kept: list[Event] = []
    for e in ordered:
        mint = getattr(e, "mint", None)
        if isinstance(e, TokenLaunch):
            if e.mint not in created or e.creator == "unknown" or e.venue != "pump_fun" or e.mint in seen:
                continue
            seen.add(e.mint)
        elif mint is not None and mint not in seen:
            continue
        kept.append(e)
    stats = {
        "events_in": len(ordered),
        "events_kept": len(kept),
        "tokens": len(created),
        "non_sol_quoted": len(non_sol),
    }
    return kept, stats


def fetch_history(
    rpc: SolanaRpc,
    out: Any,
    start_time: float,
    end_time: float,
    segment_seconds: float = 600.0,
    workers: int = 6,
    programs: list[str] | None = None,
    log: Any = None,
    retries: int = 5,
    retry_wait: float = 30.0,
) -> dict[str, int]:
    """Replay ``[start_time, end_time)`` into ``out`` resumably, then write the cleaned history.

    Every segment is its own seeking :class:`HistoryWalker` saved under ``out/segments`` when
    complete, so an interrupted run resumes where it stopped and a network error costs at most
    one segment.  The merged, :func:`clean_history`-filtered events are saved to ``out``.
    """
    from pathlib import Path

    from nardis_neural.solana.market import EventStore

    say = log or (lambda _m: None)
    root = Path(out)
    seg_root = root / "segments"
    seg_root.mkdir(parents=True, exist_ok=True)
    n = max(1, int(-(-(end_time - start_time) // segment_seconds)))
    for k in range(n):
        d = seg_root / f"seg_{k:04d}"
        if (d / "done").exists():
            continue
        lo, hi = start_time + k * segment_seconds, min(start_time + (k + 1) * segment_seconds, end_time)
        for attempt in range(retries):
            try:
                w = HistoryWalker(
                    rpc,
                    lo,
                    hi,
                    programs=programs or [PUMP_FUN_PROGRAM],
                    segment_seconds=segment_seconds,
                    workers=workers,
                )
                store = EventStore(list(w.events()))
                store.save(d)
                (d / "done").write_text("1")
                say(
                    f"segment {k + 1}/{n}: {w.stats['transactions']} transactions, {w.stats['events']} events"
                )
                break
            except Exception as exc:  # network trouble: retry this segment only
                say(f"segment {k + 1}/{n} attempt {attempt + 1} failed: {type(exc).__name__}")
                if attempt + 1 == retries:
                    raise
                time.sleep(retry_wait)
    merged: list[Event] = []
    for d in sorted(seg_root.glob("seg_*")):
        merged.extend(EventStore.load(d).sorted())
    kept, stats = clean_history(merged)
    EventStore(kept).save(root)
    return stats | {"segments": n}
