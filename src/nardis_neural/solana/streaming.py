"""Train by streaming: feed months of history (or the live chain) through the brain.

``stream_train`` needs no stored dataset.  Events arrive in time order from any source —
:class:`~nardis_neural.solana.ingest.history.HistoryWalker` over an archival RPC such as
Old Faithful, a live :class:`~nardis_neural.solana.ingest.stream.ChainStreamer`, or a saved
event directory — and the brain learns online exactly as it would live:

* a fresh workspace is **bootstrapped** from the first ``warmup_seconds`` of the stream;
* every ``assess_every`` seconds of *market* time active tokens are assessed (feeding the
  moonshot tracker and the neural continual-learning queue) and matured outcomes resolve;
* every ``evict_every`` seconds finished tokens are forgotten (bounded memory), after
  their moonshot rows are labelled;
* every ``maintenance_every`` seconds: neural adaptation / retraining / promotion, risk
  refit and the gated online tail-model refit;
* every ``checkpoint_every`` seconds the compact state is saved; a restarted run skips
  events up to the checkpointed market time and carries on.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.solana.brain import SolanaBrain
from nardis_neural.solana.config import SolanaConfig
from nardis_neural.solana.events import Event, TokenLaunch
from nardis_neural.solana.ingest.decoder import TransactionDecoder
from nardis_neural.solana.market import EventStore


def _quiet(_: str) -> None:
    return None


def stream_train(
    workspace: str | Path,
    events: Iterable[Event],
    cfg: SolanaConfig | None = None,
    neural_cfg: NeuralConfig | None = None,
    *,
    warmup_seconds: float = 6 * 3600.0,
    assess_every: float = 10.0,
    evict_every: float = 600.0,
    maintenance_every: float = 3600.0,
    checkpoint_every: float = 6 * 3600.0,
    evict_idle_seconds: float = 2 * 3600.0,
    decoder: TransactionDecoder | None = None,
    max_events: int | None = None,
    device: torch.device | str | None = None,
    log: Callable[[str], None] = _quiet,
) -> dict[str, Any]:
    root = Path(workspace)
    it = iter(events)
    stats: dict[str, Any] = {
        "events": 0,
        "rejected": 0,
        "skipped_before_resume": 0,
        "assessments": 0,
        "resolved": 0,
        "evicted": 0,
        "maintenance": 0,
        "checkpoints": 0,
    }
    if not (root / "solana.yaml").exists():
        warm = EventStore()
        start: float | None = None
        for e in it:
            warm.add(e)
            if start is None and isinstance(e, TokenLaunch):
                start = e.t  # the warm-up clock starts at the first launch, not at old funding
            if start is not None and e.t - start >= warmup_seconds:
                break
        if not len(warm):
            raise ValueError("the stream produced no events to bootstrap from")
        log(f"bootstrapping from {len(warm)} warm-up events ({warmup_seconds / 3600:.1f} h)")
        SolanaBrain.bootstrap(root, warm, cfg, neural_cfg, device=device, log=log)
        stats["warmup_events"] = len(warm)
    brain = SolanaBrain(root, device=device)
    brain.enable_streaming(evict_idle_seconds)
    resume_t = brain.market.now
    nxt: dict[str, float] = {}
    wall = time.monotonic()

    def due(key: str, t: float, every: float) -> bool:
        if key not in nxt:
            nxt[key] = t + every
        if t >= nxt[key]:
            nxt[key] = t + every
            return True
        return False

    for e in it:
        if e.t <= resume_t:
            stats["skipped_before_resume"] += 1
            continue
        try:
            brain.ingest(e)
        except (KeyError, ValueError):
            stats["rejected"] += 1  # unknown / evicted token or out-of-order event
            continue
        stats["events"] += 1
        t = e.t
        if due("assess", t, assess_every):
            stats["assessments"] += len(brain.assess_active())
            stats["resolved"] += brain.resolve()
        if due("evict", t, evict_every):
            gone = brain.evict()
            stats["evicted"] += len(gone)
            if decoder is not None:
                decoder.forget(gone)
        if due("maintenance", t, maintenance_every):
            status = brain.maintenance()
            stats["maintenance"] += 1
            rate = stats["events"] / max(time.monotonic() - wall, 1e-9)
            log(
                f"t={t:.0f} events={stats['events']} ({rate:.0f}/s) tokens={len(brain.market.tokens)} "
                f"wallets={len(brain.market.wallets)} moonshot={status.get('moonshot_online')}"
            )
        if due("checkpoint", t, checkpoint_every):
            brain.save()
            stats["checkpoints"] += 1
        if max_events is not None and stats["events"] >= max_events:
            break
    brain.save()
    stats |= {
        "market_now": brain.market.now,
        "tokens_in_memory": len(brain.market.tokens),
        "wallets": len(brain.market.wallets),
        "moonshot_buffer_rows": len(brain.tracker.resolved),
        "moonshot_model": brain.moonshot is not None,
    }
    return stats
