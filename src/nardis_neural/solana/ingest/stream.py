"""Live chain streaming into :class:`SolanaBrain` (read-only).

:class:`ChainStreamer` polls ``getSignaturesForAddress`` for the watched programs, keeps a
persistent per-program cursor, fetches new transactions, orders them by slot and decodes
them.  :func:`run_live` drives the brain: ingest → periodic batched assessments (JSONL)
→ outcome resolution → periodic maintenance.  Polling works with any standard RPC
endpoint; a Geyser/websocket source can feed :meth:`TransactionDecoder.decode` directly.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import IO, Any

from nardis_neural.solana.events import Event
from nardis_neural.solana.ingest.decoder import TransactionDecoder
from nardis_neural.solana.ingest.pumpfun import PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM
from nardis_neural.solana.ingest.rpc import SolanaRpc
from nardis_neural.solana.market import event_sort_key

DEFAULT_PROGRAMS: tuple[str, ...] = (PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM)


class ChainStreamer:
    """Polls the watched programs for new transactions and decodes them into events.

    Per-program cursors persist in ``state_file``; a bounded set of seen signatures drops
    duplicates.  The default decoder looks up mint authorities through ``rpc``.
    """

    def __init__(
        self,
        rpc: SolanaRpc,
        decoder: TransactionDecoder | None = None,
        programs: Iterable[str] = DEFAULT_PROGRAMS,
        state_file: str | Path | None = None,
        initial_limit: int = 1000,
        max_backlog: int = 50_000,
        seen_capacity: int = 200_000,
    ) -> None:
        self.rpc = rpc
        self.decoder = decoder or TransactionDecoder(mint_info=rpc.mint_authorities)
        self.programs = list(programs)
        self.state_file = Path(state_file) if state_file is not None else None
        self.initial_limit = initial_limit
        self.max_backlog = max_backlog
        self.gaps = 0
        """Polls whose backlog exceeded ``max_backlog`` (older activity was skipped)."""
        self.cursor: dict[str, str] = {}
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._seen_capacity = seen_capacity
        self.raw_sink: Callable[[dict[str, Any]], object] | None = None
        """Optional callback receiving every fetched raw transaction (archiving / replay)."""
        if self.state_file is not None and self.state_file.exists():
            self.cursor = dict(json.loads(self.state_file.read_text()).get("cursor", {}))

    def _save(self) -> None:
        if self.state_file is not None:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps({"cursor": self.cursor}))

    def _new_signatures(self, program: str) -> list[dict[str, Any]]:
        """Every signature newer than the cursor, oldest first.

        Pages backwards until the cursor is reached, so bursts larger than one page are never
        dropped.  On the very first poll (no cursor) only the latest ``initial_limit`` are
        taken.  If the backlog exceeds ``max_backlog`` the newest part is kept and ``gaps`` is
        incremented, so a lagging consumer is visible instead of silently losing data.
        """
        until = self.cursor.get(program)
        collected: list[dict[str, Any]] = []
        before: str | None = None
        while True:
            limit = 1000 if until is not None else min(1000, self.initial_limit - len(collected))
            if limit <= 0:
                break
            page = self.rpc.get_signatures(program, before=before, until=until, limit=limit)
            collected.extend(page)
            if len(page) < limit:
                break
            if until is not None and len(collected) >= self.max_backlog:
                self.gaps += 1
                break
            before = page[-1]["signature"]
        if collected:
            self.cursor[program] = collected[0]["signature"]
        return list(reversed(collected))

    def poll(self) -> list[Event]:
        """Fetch and decode everything since the last poll; returns events in time order and saves cursors.

        Failed transactions are skipped; every fetched raw transaction goes to ``raw_sink`` if set.
        """
        sigs: list[dict[str, Any]] = []
        for program in self.programs:
            for s in self._new_signatures(program):
                if s.get("err") is None and s["signature"] not in self._seen:
                    self._seen[s["signature"]] = None
                    sigs.append(s)
        while len(self._seen) > self._seen_capacity:
            self._seen.popitem(last=False)
        txs = []
        for s in sorted(sigs, key=lambda s: (int(s.get("slot", 0)), float(s.get("blockTime") or 0))):
            tx = self.rpc.get_transaction(s["signature"])
            if tx is not None:
                txs.append(tx)
                if self.raw_sink is not None:
                    self.raw_sink(tx)
        events = [e for tx in txs for e in self.decoder.decode(tx)]
        self._save()
        return sorted(events, key=event_sort_key)


def run_live(
    brain: Any,
    streamer: ChainStreamer,
    assess_every: float = 10.0,
    maintenance_every: float = 600.0,
    poll_interval: float = 2.0,
    out: IO[str] | None = None,
    max_polls: int | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
) -> dict[str, int]:
    """Stream the chain into ``brain`` (a :class:`SolanaBrain`) until ``max_polls``."""
    stats = {"polls": 0, "events": 0, "assessments": 0, "resolved": 0, "maintenance": 0, "rejected": 0}
    next_assess = clock() + assess_every
    next_maint = clock() + maintenance_every
    while max_polls is None or stats["polls"] < max_polls:
        for e in streamer.poll():
            try:
                brain.ingest(e)
                stats["events"] += 1
            except (KeyError, ValueError):
                stats["rejected"] += 1  # out-of-order or unknown-token event: skip, never crash the loop
        stats["polls"] += 1
        now = clock()
        if now >= next_assess:
            next_assess = now + assess_every
            for report in brain.assess_active():
                stats["assessments"] += 1
                if out is not None:
                    out.write(report.model_dump_json(exclude={"features"}) + "\n")
                    out.flush()
            stats["resolved"] += brain.resolve()
        if now >= next_maint:
            next_maint = now + maintenance_every
            brain.maintenance()
            stats["maintenance"] += 1
        if max_polls is None or stats["polls"] < max_polls:
            sleep(poll_interval)
    brain.save()
    return stats
