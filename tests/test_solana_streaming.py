"""Streaming training: forward history walker over an archival RPC, bounded-memory market,
online moonshot learning and the resumable stream_train driver."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nardis_neural.solana import LaunchSimSpec, SolanaConfig, SolanaMarket, simulate_launches
from nardis_neural.solana.brain import SolanaBrain
from nardis_neural.solana.events import Swap, TokenLaunch
from nardis_neural.solana.ingest.decoder import decode_transactions
from nardis_neural.solana.ingest.encode import events_to_transactions
from nardis_neural.solana.ingest.history import HistoryWalker
from nardis_neural.solana.ingest.pumpfun import PUMP_FUN_PROGRAM
from nardis_neural.solana.ingest.rpc import SolanaRpc
from nardis_neural.solana.moonshot import MoonshotSpec
from nardis_neural.solana.moonshot.online import MoonshotTracker
from nardis_neural.solana.streaming import stream_train
from tests.conftest import make_tiny_config


class FakeArchive:
    """An archival endpoint: newest-first signature listings with before / until / limit."""

    def __init__(self, txs: list[dict[str, Any]]) -> None:
        for i, tx in enumerate(txs):
            tx["transaction"]["signatures"] = [f"sig{i:06d}"]
        self.by_sig = {tx["transaction"]["signatures"][0]: tx for tx in txs}
        self.newest_first = [
            {
                "signature": tx["transaction"]["signatures"][0],
                "slot": tx["slot"],
                "blockTime": tx["blockTime"],
            }
            for tx in reversed(txs)
        ]
        self.index = {s["signature"]: i for i, s in enumerate(self.newest_first)}
        self.calls = {"getSignaturesForAddress": 0, "getTransaction": 0}

    def __call__(self, method: str, params: list[Any]) -> Any:
        self.calls[method] += 1
        if method == "getTransaction":
            return json.loads(json.dumps(self.by_sig[params[0]]))
        if params[0] != PUMP_FUN_PROGRAM:
            return []
        opts = params[1]
        lo = self.index[opts["before"]] + 1 if opts.get("before") else 0
        hi = self.index[opts["until"]] if opts.get("until") else len(self.newest_first)
        return [dict(s, err=None) for s in self.newest_first[lo:hi][: opts["limit"]]]


@pytest.fixture(scope="module")
def history() -> Any:
    store, _ = simulate_launches(LaunchSimSpec(n_tokens=24, seed=8, n_retail=250, duration_seconds=3 * 3600))
    return store


def test_history_walker_replays_a_window_forward(history: Any) -> None:
    txs = events_to_transactions(history.events)
    reference = decode_transactions(json.loads(json.dumps(txs)))
    t0, t1 = float(txs[0]["blockTime"]), float(txs[-1]["blockTime"]) + 1
    start, end = t0 + 1800, t1 - 1800
    archive = FakeArchive(txs)
    walker = HistoryWalker(
        SolanaRpc(transport=archive), start, end, segment_seconds=900, page_size=97, seek=False
    )
    got = list(walker.events())
    in_window = [tx for tx in txs if start <= float(tx["blockTime"]) < end]
    assert walker.stats["transactions"] == len(in_window) and walker.stats["segments"] >= 4
    assert all(a.t <= b.t for a, b in itertools.pairwise(got)), "forward in time"
    assert all(start <= e.t < end + 1 for e in got)
    # the same trades come out as a full decode of that window, nothing fetched outside it
    ref_swaps = sorted(
        (e.mint, round(e.sol_amount, 9)) for e in reference if start <= e.t < end and isinstance(e, Swap)
    )
    got_swaps = sorted((e.mint, round(e.sol_amount, 9)) for e in got if isinstance(e, Swap))
    assert got_swaps == ref_swaps
    assert archive.calls["getTransaction"] == len(in_window)
    with pytest.raises(ValueError):
        HistoryWalker(SolanaRpc(transport=archive), end, start)


def test_market_eviction_keeps_what_was_learned(history: Any) -> None:
    m = SolanaMarket(SolanaConfig())
    events = history.sorted()
    m.ingest_many(events)
    wallets, updates = len(m.wallets), m.reputation_updates
    gone = m.evict(m.now + 7200, idle_seconds=3600)
    assert gone and not set(gone) & set(m.tokens) and m.evicted == len(gone)
    assert len(m.wallets) == wallets and m.reputation_updates >= updates
    launch = next(e for e in events if isinstance(e, TokenLaunch) and e.mint == gone[0])
    with pytest.raises(ValueError):
        m.ingest(TokenLaunch(mint=launch.mint, t=m.now + 1, creator=launch.creator))
    with pytest.raises(ValueError):
        m.evict(m.now, idle_seconds=1.0)


def test_stream_train_bootstraps_bounds_memory_learns_online_and_resumes(
    history: Any, tmp_path: Path
) -> None:
    ncfg = make_tiny_config()
    ncfg.training.epochs = 1
    ncfg.ensemble.size = 1
    ncfg.ensemble.mc_dropout_samples = 0
    cfg = SolanaConfig(sample_interval_seconds=30.0, graph_top_k=6)
    events = history.sorted()
    ws = tmp_path / "ws"
    stats = stream_train(
        ws,
        events,
        cfg,
        ncfg,
        warmup_seconds=2400,
        assess_every=30,
        evict_every=300,
        maintenance_every=1800,
        checkpoint_every=1800,
        evict_idle_seconds=1200,
        device="cpu",
    )
    n_tokens = sum(isinstance(e, TokenLaunch) for e in events)
    assert stats["warmup_events"] > 0 and stats["events"] > 0 and stats["assessments"] > 0
    assert stats["evicted"] > 0 and stats["tokens_in_memory"] < n_tokens
    assert (ws / "stream" / "market.pkl").exists() and (ws / "moonshot" / "online.npz").exists()

    brain = SolanaBrain(ws, device="cpu")
    assert brain.streaming and len(brain.history) == 0
    assert brain.market.now == stats["market_now"]
    assert len(brain.tracker.resolved) + sum(map(len, brain.tracker.pending.values())) > 0
    out = brain.refit_moonshot_online(every_seconds=0, min_tokens=5, members=1, epochs=5)
    assert out is not None and out["status"] == "promoted" and np.isfinite(out["holdout_nll"])
    assert brain.moonshot is not None and brain.moonshot.inputs == "raw"
    a = brain.assess(next(iter(brain.market.tokens)))
    assert "chase_score" in a.moonshot

    again = stream_train(ws, events, cfg, ncfg, device="cpu")
    assert again["events"] == 0 and again["skipped_before_resume"] == len(events)


def test_tracker_labels_censors_and_roundtrips(history: Any, tmp_path: Path) -> None:
    m = SolanaMarket(SolanaConfig())
    events = history.sorted()
    spec = MoonshotSpec(min_entry_age_seconds=0, horizon_seconds=3600)
    tr = MoonshotTracker(spec, sample_every=60)
    for e in events:
        m.ingest(e)
        if isinstance(e, TokenLaunch):
            continue
        mints = [e.mint] if isinstance(e, Swap) and e.mint in m.tokens else []
        if mints:
            ages = np.asarray([m.now - m.token(mints[0]).launch.t])
            tr.observe(mints, np.ones((1, 3), np.float32), ages, m.now)
    assert tr.pending
    data = tr.training_set(m, m.now)
    assert data is not None and data.censored.any(), "still-running tokens enter as censored"
    tr.resolve(m, m.now + 10 * 3600)
    assert not tr.pending and len(tr.resolved) == len(data)
    tr.save(tmp_path / "t.npz")
    back = MoonshotTracker.load(tmp_path / "t.npz", spec)
    assert len(back.resolved) == len(tr.resolved) and back.resolved_tokens == tr.resolved_tokens


def test_clean_history_keeps_only_tokens_created_in_the_window() -> None:
    from nardis_neural.solana.events import Event, Swap, TokenLaunch, Transfer
    from nardis_neural.solana.ingest.history import clean_history

    def swap(mint: str, t: float, sol_res: float = 40.0) -> Swap:
        return Swap(mint, t, "w", True, 0.1, 1e6, sol_res, 8e8)

    events: list[Event] = [
        TokenLaunch("new", 1.0, "creator"),
        swap("new", 2.0),
        TokenLaunch("old", 1.5, "unknown"),  # first seen by a trade: inferred launch
        swap("old", 2.5),
        TokenLaunch("usdc", 1.2, "payer", venue="raydium"),  # pre-existing pool seen once
        swap("usdc", 2.2),
        TokenLaunch("v2", 1.3, "creator"),
        swap("v2", 2.3, sol_res=0.0),  # BuyV2 / SellV2 curve without SOL reserves
        TokenLaunch("new", 3.0, "unknown"),  # repeated inferred launch from a later segment
        Transfer(2.0, "a", "b", 1.0),
    ]
    kept, stats = clean_history(events)
    mints = {getattr(e, "mint", None) for e in kept}
    assert mints == {"new", None} and stats["tokens"] == 1
    assert sum(isinstance(e, TokenLaunch) for e in kept) == 1


def test_fetch_history_resumes_and_saves_a_clean_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana.ingest import history as hist

    calls: list[float] = []

    class FakeWalker:
        def __init__(self, rpc, lo, hi, **kw) -> None:  # type: ignore[no-untyped-def]
            self.lo, self.stats = lo, {"transactions": 1, "events": 2}
            calls.append(lo)

        def events(self):  # type: ignore[no-untyped-def]
            from nardis_neural.solana.events import Swap, TokenLaunch

            m = f"m{int(self.lo)}"
            return [TokenLaunch(m, self.lo + 1, "c"), Swap(m, self.lo + 2, "w", True, 0.1, 1e6, 40.0, 8e8)]

    real = hist.HistoryWalker
    hist.HistoryWalker = FakeWalker  # type: ignore[misc,assignment]
    try:
        stats = hist.fetch_history(SolanaRpc(transport=lambda m, p: None), tmp_path, 0.0, 1800.0, 600.0)
        assert calls == [0.0, 600.0, 1200.0] and stats["tokens"] == 3
        calls.clear()
        hist.fetch_history(SolanaRpc(transport=lambda m, p: None), tmp_path, 0.0, 1800.0, 600.0)
        assert calls == []  # every segment already done
    finally:
        hist.HistoryWalker = real  # type: ignore[misc]
    from nardis_neural.solana.market import EventStore

    assert len(EventStore.load(tmp_path)) == 6


def test_follow_graduates_adds_only_post_graduation_trades(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana.events import Event, Migration, Swap, TokenLaunch
    from nardis_neural.solana.ingest import history as hist

    base: list[Event] = [
        TokenLaunch("g", 0.0, "dev"),
        Swap("g", 10.0, "w1", True, 1.0, 1e6, 40.0, 8e8),
        Migration("g", 100.0, "pumpswap", 85.0, 2e8),
        TokenLaunch("dud", 5.0, "dev2"),
    ]
    calls: list[list[str]] = []

    class FakeWalker:
        def __init__(self, rpc, lo, hi, programs, **kw) -> None:  # type: ignore[no-untyped-def]
            self.stats = {"transactions": 3, "events": 3}
            calls.append(list(programs))
            self.lo = lo

        def events(self):  # type: ignore[no-untyped-def]
            return [
                Swap("g", 10.0, "w1", True, 1.0, 1e6, 40.0, 8e8),  # before graduation: dropped
                Swap("g", self.lo + 50, "w2", True, 2.0, 1e6, 90.0, 1.9e8),  # after: kept
                Swap("g", self.lo + 60, "bot", True, 2.0, 1e6, 5000.0, 5e10),  # not a pump pool: dropped
            ]

    real = hist.HistoryWalker
    hist.HistoryWalker = FakeWalker  # type: ignore[misc,assignment]
    try:
        merged, stats = hist.follow_graduates(
            SolanaRpc(transport=lambda m, p: None), tmp_path, base, 100.0 + 7200.0
        )
        assert calls and all(p == ["g"] for p in calls) and stats["graduates"] == 1
        post = [e for e in merged if isinstance(e, Swap) and e.t > 100.0]
        assert len(post) == 2 and {e.wallet for e in post} == {"w2"}
        assert [e.t for e in merged] == sorted(e.t for e in merged)
        calls.clear()
        hist.follow_graduates(SolanaRpc(transport=lambda m, p: None), tmp_path, base, 100.0 + 7200.0)
        assert calls == []  # resumed: every segment already saved
    finally:
        hist.HistoryWalker = real  # type: ignore[misc]
