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

    Per-program cursors advance only once a poll's transactions are fetched and decoded (a
    failed poll is retried in full), and persist in ``state_file`` when the consumer calls
    :meth:`commit` after checkpointing what it ingested, so a restart replays from the last
    checkpoint.  After ``max_failed_polls`` polls in a row whose fetch failed, a transaction that
    still cannot be fetched is skipped (counted in ``fetch_errors``) so one bad signature or a
    long rate-limit burst cannot stall the stream.  A bounded set of seen signatures drops
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
        workers: int = 6,
        max_failed_polls: int = 3,
    ) -> None:
        self.rpc = rpc
        self.decoder = decoder or TransactionDecoder(mint_info=rpc.mint_authorities)
        self.programs = list(programs)
        self.state_file = Path(state_file) if state_file is not None else None
        self.initial_limit = initial_limit
        self.max_backlog = max_backlog
        self.workers = max(1, workers)
        """Parallel ``getTransaction`` calls (one kept-alive connection each)."""
        self.gaps = 0
        """Polls whose backlog exceeded ``max_backlog`` (older activity was skipped)."""
        self.decode_errors = 0
        """Transactions the decoder could not handle (skipped, the rest of the poll is kept)."""
        self.max_failed_polls = max_failed_polls
        self.fetch_errors = 0
        """Transactions skipped because fetching them kept failing (see ``max_failed_polls``)."""
        self._failed_polls = 0
        self.cursor: dict[str, str] = {}
        """Newest signature consumed per program (in memory; see :meth:`commit`)."""
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._seen_capacity = seen_capacity
        self.raw_sink: Callable[[dict[str, Any]], object] | None = None
        """Optional callback receiving every fetched raw transaction (archiving / replay)."""
        if self.state_file is not None and self.state_file.exists():
            self.cursor = dict(json.loads(self.state_file.read_text()).get("cursor", {}))

    def commit(self, cursor: dict[str, str] | None = None) -> None:
        """Persist ``cursor`` (default: the current one) to ``state_file``.

        Call it right after checkpointing the consumer, with the cursor of the last poll it had
        fully ingested: a restart then resumes exactly where the checkpoint stops.
        """
        if self.state_file is not None:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps({"cursor": dict(self.cursor if cursor is None else cursor)}))
            tmp.replace(self.state_file)

    def _new_signatures(self, program: str) -> tuple[list[dict[str, Any]], bool]:
        """Every signature newer than the cursor, oldest first.

        Pages backwards until the cursor is reached, so bursts larger than one page are never
        dropped.  On the very first poll (no cursor) only the latest ``initial_limit`` are
        taken.  If the backlog exceeds ``max_backlog`` the newest part is kept and the returned
        flag is set (:meth:`poll` counts it in ``gaps`` once the poll succeeds), so a lagging
        consumer is visible instead of silently losing data.  The cursor itself is left to
        :meth:`poll`.
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
                return list(reversed(collected)), True
            before = page[-1]["signature"]
        return list(reversed(collected)), False

    def _decode(self, tx: dict[str, Any]) -> list[Event]:
        """Decode one transaction; one the decoder cannot handle is counted and skipped."""
        try:
            return list(self.decoder.decode(tx))
        except (ArithmeticError, AttributeError, ValueError, KeyError, IndexError, TypeError):
            self.decode_errors += 1
            return []

    def poll(self) -> list[Event]:
        """Fetch and decode everything since the last poll; returns events in time order.

        Failed transactions are skipped; every fetched raw transaction goes to ``raw_sink`` if set.
        The cursors (and the seen-set) advance only when the whole poll succeeded, so an RPC
        failure leaves them untouched and the next poll fetches the same batch again (up to
        ``max_failed_polls`` times, then unfetchable transactions are skipped).  They are not
        written to ``state_file`` here: see :meth:`commit`.
        """
        sigs: list[dict[str, Any]] = []
        newest: dict[str, str] = {}
        batch: set[str] = set()
        gaps = 0
        for program in self.programs:
            listed, gapped = self._new_signatures(program)
            gaps += gapped
            if listed:
                newest[program] = listed[-1]["signature"]
            for s in listed:
                sig = s["signature"]
                if s.get("err") is None and sig not in self._seen and sig not in batch:
                    batch.add(sig)
                    sigs.append(s)
        ordered = sorted(sigs, key=lambda s: (int(s.get("slot", 0)), float(s.get("blockTime") or 0)))
        skip_failures = self._failed_polls >= self.max_failed_polls
        failed: list[str] = []

        def fetch(s: dict[str, Any]) -> dict[str, Any] | None:
            if not skip_failures:
                return self.rpc.get_transaction(s["signature"])
            try:
                return self.rpc.get_transaction(s["signature"])
            except Exception:  # still failing after the RPC's own retries: skip this one
                failed.append(s["signature"])
                return None

        try:
            if self.workers > 1 and len(ordered) > 1:
                from concurrent.futures import ThreadPoolExecutor

                with ThreadPoolExecutor(max_workers=self.workers) as pool:
                    fetched = list(pool.map(fetch, ordered))
            else:
                fetched = [fetch(s) for s in ordered]
        except Exception:
            self._failed_polls += 1
            raise
        self._failed_polls = 0
        self.fetch_errors += len(failed)
        txs = []
        for tx in fetched:  # map() keeps slot order
            if tx is not None:
                txs.append(tx)
                if self.raw_sink is not None:
                    self.raw_sink(tx)
        events = [e for tx in txs for e in self._decode(tx)]
        self.gaps += gaps
        self.cursor.update(newest)
        for s in ordered:
            self._seen[s["signature"]] = None
        while len(self._seen) > self._seen_capacity:
            self._seen.popitem(last=False)
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
    """Stream the chain into ``brain`` (a :class:`SolanaBrain`) until ``max_polls``.

    In streaming mode (``brain.streaming``) idle tokens are evicted at each maintenance, with
    their decoder state, so memory stays bounded.

    The streamer's cursor is committed after every brain checkpoint (maintenance and the final
    save), so a restart resumes from what the saved brain has seen.  The final save runs even on
    Ctrl-C or an error, after the rest of the poll in hand has been ingested.
    """
    stats = {
        "polls": 0,
        "events": 0,
        "assessments": 0,
        "resolved": 0,
        "maintenance": 0,
        "rejected": 0,
        "evicted": 0,
    }
    next_assess = clock() + assess_every
    next_maint = clock() + maintenance_every
    events: list[Event] = []
    done = 0

    def feed() -> None:
        nonlocal done
        while done < len(events):
            e = events[done]
            done += 1  # counted first: an event that raises is not retried by the final save
            try:
                brain.ingest(e)
                stats["events"] += 1
            except (KeyError, ValueError):
                stats["rejected"] += 1  # out-of-order or unknown-token event: skip, never crash the loop

    try:
        while max_polls is None or stats["polls"] < max_polls:
            events, done = streamer.poll(), 0
            feed()
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
                if getattr(brain, "streaming", False):  # bounded memory: forget idle tokens
                    gone = brain.evict()
                    stats["evicted"] += len(gone)
                    streamer.decoder.forget(gone)
                brain.maintenance()  # saves the brain
                streamer.commit()
                stats["maintenance"] += 1
            if max_polls is None or stats["polls"] < max_polls:
                sleep(poll_interval)
    finally:
        feed()  # the cursor already covers this poll: ingest what is left before saving
        brain.save()
        streamer.commit()
    return stats
