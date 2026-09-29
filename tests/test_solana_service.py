import json
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, simulate_launches
from nardis_neural.solana.service import AddonService, make_server
from tests.conftest import make_tiny_config


def _call(port: int, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def served(tmp_path_factory):  # type: ignore[no-untyped-def]
    hist, _ = simulate_launches(LaunchSimSpec(n_tokens=8, seed=2, n_retail=80, duration_seconds=1800))
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    ws = tmp_path_factory.mktemp("svc") / "ws"
    SolanaBrain.bootstrap(ws, hist, SolanaConfig(sample_interval_seconds=30.0), base, device="cpu")
    service = AddonService(SolanaBrain(ws, device="cpu"))
    server = make_server(service, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield service, server.server_address[1]
    server.shutdown()
    server.server_close()


def test_http_api_round_trip(served) -> None:  # type: ignore[no-untyped-def]
    service, port = served
    status, health = _call(port, "GET", "/health")
    assert status == 200 and health["ok"] and health["tokens"] > 0
    mint = next(iter(service.brain.market.tokens))
    status, listed = _call(port, "GET", "/tokens?active_seconds=1e9")
    assert status == 200 and mint in {r["mint"] for r in listed["tokens"]}
    status, assessed = _call(port, "GET", f"/assess?mint={mint}")
    assert status == 200 and assessed["mint"] == mint
    status, advice = _call(
        port, "POST", "/advise_trade", {"trade_id": "n1", "mint": mint, "features": {"nardis_score": 0.9}}
    )
    assert status == 200 and advice["source"] == "prior" and 0 <= advice["p_win"] <= 1
    status, settled = _call(
        port, "POST", "/settle_trade", {"trade_id": "n1", "multiple": 2.5, "peak_multiple": 4}
    )
    assert status == 200 and settled == {"accepted": True, "settled_trades": 1}
    status, hold = _call(
        port, "POST", "/hold_advice", {"mint": mint, "t_signal": service.brain.market.now - 60}
    )
    assert status == 200 and isinstance(hold, dict)
    assert _call(port, "POST", "/save", {})[0] == 200


def test_http_api_rejects_bad_requests(served) -> None:  # type: ignore[no-untyped-def]
    _, port = served
    assert _call(port, "GET", "/nope")[0] == 404
    assert _call(port, "GET", "/assess?mint=unknown")[0] == 400
    assert _call(port, "POST", "/advise_trade", {"mint": "x"})[0] == 400  # no trade_id
    assert _call(port, "GET", "/ranking")[0] == 409  # no tail model installed yet


def test_pushed_transactions_are_ingested(served) -> None:  # type: ignore[no-untyped-def]
    _, port = served
    status, out = _call(port, "POST", "/ingest", {"transactions": [{"slot": 1, "meta": None}]})
    assert status == 200 and out["undecodable_transactions"] == 1
    assert _call(port, "POST", "/ingest", {"transactions": "nope"})[0] == 400


def test_stream_thread_feeds_the_brain(served) -> None:  # type: ignore[no-untyped-def]
    service, _ = served

    class OnePoll:
        def __init__(self) -> None:
            self.calls = 0

        def poll(self) -> list[Any]:
            self.calls += 1
            if self.calls == 2:
                raise ConnectionError("network hiccup")  # must not kill the thread
            return []

    streamer = OnePoll()
    service.stream(streamer, poll_interval=0.01, save_every=1e9, maintenance_every=1e9)
    deadline = time.time() + 5
    while service.stream_stats.get("polls", 0) < 3 and time.time() < deadline:
        time.sleep(0.02)
    service._stop.set()
    assert service.stream_stats["polls"] >= 3 and service.stream_stats["errors"] >= 1


def test_sidecar_keeps_training_on_the_archive(served, tmp_path) -> None:  # type: ignore[no-untyped-def]
    from tests.test_solana_archive import _trades

    service, port = served
    _trades(30, 7, start=10_000).write_parquet(tmp_path / "trades.parquet")
    service._stop.clear()  # an earlier test stopped the service threads
    before = len(service.brain.meta.multiple)
    service.train_on_archive(tmp_path / "trades.parquet", every=1e9)
    deadline = time.time() + 30
    while service.archive_stats["scans"] < 1 and time.time() < deadline:
        time.sleep(0.05)
    status, health = _call(port, "GET", "/health")
    assert status == 200
    assert health["archive"]["last"]["added"] == 30 and len(service.brain.meta.multiple) == before + 30


def test_moonshots_and_alerts(served) -> None:  # type: ignore[no-untyped-def]
    service, port = served
    assert _call(port, "GET", "/moonshots?target=7")[0] == 400  # not a chase target

    class A:  # a minimal assessment as moonshot_ranking returns it
        def __init__(self, mint: str, p10: float) -> None:
            self.mint = mint
            self.flags: list[str] = []
            self.tape: dict[str, float] = {}
            self.moonshot = {"p_ge_10x": p10, "edge_10x": p10 / 0.0323, "chase_score": 1.0}

    mint = next(iter(service.brain.market.tokens))
    ranked = [A(mint, 0.20), A(mint, 0.02)]
    real = service.brain.moonshot_ranking
    service.brain.moonshot_ranking = lambda *a, **k: ranked
    try:
        status, body = _call(port, "GET", "/moonshots?target=10&min_edge=2")
        assert status == 200 and [round(c["p_ge_10x"], 2) for c in body["candidates"]] == [0.2]
        got: list[dict[str, Any]] = []
        fail = {"n": 1}

        def post(url: str, payload: dict[str, Any]) -> None:
            if fail["n"]:
                fail["n"] -= 1
                raise ConnectionError("Nardis restarting")
            got.append(payload)

        service._stop.clear()
        service.alerts("http://nardis/moon", target=10.0, min_edge=2.0, every=0.01, post=post)
        deadline = time.time() + 10
        while service.alert_stats["scans"] < 5 and time.time() < deadline:
            time.sleep(0.02)
        service._stop.set()
        assert len(got) == 1 and got[0]["candidate"]["mint"] == mint  # retried once, then sent once
        assert service.alert_stats["errors"] == 1
    finally:
        service.brain.moonshot_ranking = real


def test_every_request_gets_an_answer(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import http.client

    service, port = served
    status, body = _call(port, "POST", "/ingest", {"transactions": [1, "x"]})
    assert status == 400 and "JSON object" in body["error"]
    bad = {"trade_id": "bad", "mint": "m", "features": [1, 2], "with_market": False}
    status, body = _call(port, "POST", "/advise_trade", bad)
    assert status == 400 and "features must be a JSON object" in body["error"]
    status, body = _call(port, "POST", "/allocate", {"equity_sol": 10, "open_stakes": "m"})
    assert status == 400 and "open_stakes" in body["error"]
    null_meta = {
        "slot": 5,
        "meta": None,
        "transaction": {"message": {"accountKeys": []}, "signatures": ["n"]},
    }
    status, body = _call(port, "POST", "/ingest", {"transactions": [null_meta]})
    assert status == 200 and body["undecodable_transactions"] == 1

    def broken(*a: Any, **k: Any) -> Any:
        raise RuntimeError("internal failure deep in a model")

    monkeypatch.setattr(service.brain, "assess", broken)
    mint = next(iter(service.brain.market.tokens))
    status, body = _call(port, "GET", f"/assess?mint={mint}")
    assert status == 500 and "internal failure" in body["error"], "an internal RuntimeError is not a 409"
    monkeypatch.setattr(service.brain, "assess", lambda *a, **k: 1 / 0)
    assert _call(port, "GET", f"/assess?mint={mint}")[0] == 500
    assert _call(port, "GET", "/health")[0] == 200, "the server keeps running"
    assert _call(port, "GET", "/ranking")[0] == 409  # a missing tail model is still the documented 409

    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.putrequest("POST", "/ingest")
    conn.putheader("Content-Length", str(64 * 1024 * 1024 + 1))  # over MAX_DRAIN_BYTES: not read
    conn.endheaders()
    resp = conn.getresponse()
    assert resp.status == 413 and "error" in json.loads(resp.read())
    conn.close()


def test_an_oversized_body_gets_its_413(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana import service as svc

    _, port = served
    monkeypatch.setattr(svc, "MAX_BODY_BYTES", 1024)
    monkeypatch.setattr(svc, "MAX_DRAIN_BYTES", 4 * 1024 * 1024)
    status, body = _call(port, "POST", "/ingest", {"transactions": [], "pad": "x" * 3 * 1024 * 1024})
    assert status == 413 and "1024" in body["error"], "an ordinary client reads the 413, not a reset"
    assert _call(port, "GET", "/health")[0] == 200


def test_pushed_transactions_are_deduplicated(served) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana.ingest.encode import events_to_transactions

    _, port = served
    later, _ = simulate_launches(
        LaunchSimSpec(
            n_tokens=1, seed=9, n_retail=10, duration_seconds=120, mint_prefix="Dup", start_time=1_760_000_000
        )
    )
    txs = events_to_transactions(later.events)[:5]
    status, first = _call(port, "POST", "/ingest", {"transactions": txs})
    assert status == 200 and first["events"] > 0 and first["duplicates"] == 0
    status, again = _call(port, "POST", "/ingest", {"transactions": [*txs, txs[0]]})
    assert status == 200 and again["events"] == 0 and again["duplicates"] == len(txs) + 1
    assert _call(port, "GET", "/health")[1]["ingest"]["duplicates"] >= len(txs) + 1


def test_an_undecodable_push_is_not_remembered(served) -> None:  # type: ignore[no-untyped-def]
    from nardis_neural.solana.ingest.encode import events_to_transactions

    _, port = served
    later, _ = simulate_launches(
        LaunchSimSpec(
            n_tokens=1,
            seed=11,
            n_retail=10,
            duration_seconds=120,
            mint_prefix="Late",
            start_time=1_760_100_000,
        )
    )
    txs = events_to_transactions(later.events)[:3]
    for i, tx in enumerate(txs):
        tx["transaction"]["signatures"] = [f"late{i}"]  # the encoder numbers from 0 on every call
    early = [tx | {"meta": None} for tx in txs]  # pushed before the full transaction was available
    status, first = _call(port, "POST", "/ingest", {"transactions": early})
    assert status == 200 and first["events"] == 0 and first["duplicates"] == 0
    status, full = _call(port, "POST", "/ingest", {"transactions": txs})
    assert status == 200 and full["duplicates"] == 0 and full["events"] > 0


def test_moonshots_negative_limit_and_health_models(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, port = served

    class A:
        def __init__(self, mint: str) -> None:
            self.mint = mint
            self.flags: list[str] = []
            self.tape: dict[str, float] = {}
            self.moonshot = {"p_ge_10x": 0.2, "edge_10x": 6.0, "chase_score": 1.0}

    mint = next(iter(service.brain.market.tokens))
    monkeypatch.setattr(service.brain, "moonshot_ranking", lambda *a, **k: [A(mint), A(mint)])
    assert _call(port, "GET", "/moonshots?target=10&limit=-1")[1]["candidates"] == []
    assert len(_call(port, "GET", "/moonshots?target=10&limit=1")[1]["candidates"]) == 1
    models = _call(port, "GET", "/health")[1]["models"]
    assert models["risk"] is True and models["runners"] is False


class _Streamer:
    """A streamer that advances its cursor on every poll and records commits."""

    def __init__(self) -> None:
        self.cursor: dict[str, str] = {"prog": "s0"}
        self.commits: list[dict[str, str]] = []
        self.gaps = 2
        self.decoder = _Decoder()
        self.n = 0

    def poll(self) -> list[Any]:
        self.n += 1
        self.cursor = {"prog": f"s{self.n}"}
        return []

    def commit(self, cursor: dict[str, str] | None = None) -> None:
        self.commits.append(dict(cursor or self.cursor))


class _Decoder:
    def __init__(self) -> None:
        self.forgotten: list[str] = []

    def forget(self, mints: list[str]) -> None:
        self.forgotten += mints


def _run_upkeep(service: Any, streamer: Any, until: Any, **kw: Any) -> None:
    service._stop.clear()
    thread = service.stream(streamer, poll_interval=0.01, **kw)
    deadline = time.time() + 10
    while not until() and time.time() < deadline:
        time.sleep(0.02)
    service._stop.set()
    thread.join(10)


def test_upkeep_runs_without_the_stream(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, _ = served
    calls = {"assess": 0, "maintenance": 0, "save": 0}
    brain = service.brain

    def count(key: str, value: Any) -> Any:
        def fn(*a: Any, **k: Any) -> Any:
            calls[key] += 1
            return value

        return fn

    monkeypatch.setattr(brain, "assess_active", count("assess", [1, 2]))
    monkeypatch.setattr(brain, "maintenance", count("maintenance", {}))
    monkeypatch.setattr(brain, "save", count("save", None))
    monkeypatch.setattr(brain, "evict", lambda: ["gone1"])
    monkeypatch.setattr(service, "decoder", _Decoder())
    _run_upkeep(
        service,
        None,
        lambda: calls["maintenance"] >= 2 and calls["save"] >= 3,
        assess_every=0.01,
        maintenance_every=0.05,
        save_every=0.02,
    )
    stats = service.stream_stats
    assert brain.streaming, "bounded-memory mode without the stream too"
    assert calls["assess"] == 1 and stats["assessments"] == 2, "one round while the market clock is frozen"
    assert calls["maintenance"] >= 2 and stats["evicted"] >= 2 and calls["save"] >= 3
    assert "gone1" in service.decoder.forgotten, "evicted tokens leave the /ingest decoder too"
    assert stats["polls"] == 0 and stats["errors"] == 0

    monkeypatch.setattr(brain.market, "now", brain.market.now)
    ticking = count("assess", [1])

    def assess_and_tick(*a: Any, **k: Any) -> Any:
        brain.market.now += 1.0  # events keep arriving through POST /ingest
        return ticking()

    monkeypatch.setattr(brain, "assess_active", assess_and_tick)
    _run_upkeep(service, None, lambda: calls["assess"] >= 4, assess_every=0.01)
    assert calls["assess"] >= 4, "assessment rounds without Nardis polling"


def test_a_frozen_market_is_not_assessed_again(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, _ = served
    brain = service.brain
    monkeypatch.setattr(brain, "maintenance", lambda: {})
    monkeypatch.setattr(brain, "save", lambda: None)
    monkeypatch.setattr(brain, "resolve", lambda: 0)
    monkeypatch.setattr(brain, "pending", [])
    rounds: list[int] = []
    real = brain.assess_active

    def counted(*a: Any, **k: Any) -> Any:
        rounds.append(1)
        return real(*a, **k)

    monkeypatch.setattr(brain, "assess_active", counted)
    service._stop.clear()
    thread = service.stream(None, poll_interval=0.01, assess_every=0.01)
    time.sleep(0.4)
    service._stop.set()
    thread.join(10)
    assert rounds == [1], "no Nardis push, no RPC: the same observations are not queued again"
    assert len({p.mint for p in brain.pending}) == len(brain.pending), "one observation per token"


def test_checkpoints_commit_the_cursor_the_brain_covers(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, port = served
    monkeypatch.setattr(service.brain, "save", lambda: None)
    monkeypatch.setattr(service.brain, "maintenance", lambda: {})
    monkeypatch.setattr(service.brain, "evict", lambda: ["gone2"])
    streamer = _Streamer()
    _run_upkeep(
        service, streamer, lambda: streamer.n >= 3 and streamer.decoder.forgotten, maintenance_every=0.01
    )
    assert streamer.commits and streamer.commits[-1] == {"prog": f"s{streamer.n}"}
    assert "gone2" in streamer.decoder.forgotten, "evicted tokens leave the stream decoder"
    health = _call(port, "GET", "/health")[1]
    assert health["stream"]["gaps"] == 2, "skipped backlog is reported"
    before = len(streamer.commits)
    assert _call(port, "POST", "/save", {})[0] == 200
    service.stop()
    assert len(streamer.commits) == before + 2 and streamer.commits[-1] == {"prog": f"s{streamer.n}"}


def test_checkpoint_waits_for_the_poll_in_progress(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, _ = served
    monkeypatch.setattr(service.brain, "save", lambda: None)
    streamer = _Streamer()
    monkeypatch.setattr(service, "_streamer", streamer)
    monkeypatch.setattr(service, "_consumed", {"prog": "old"})
    with service.lock:
        service._mid_poll = True  # the stream thread is between the chunks of a poll
    saver = threading.Thread(target=service.save)
    saver.start()
    time.sleep(0.2)
    assert saver.is_alive() and streamer.commits == [], "no checkpoint holds part of a poll"
    with service.lock:
        service._mid_poll = False
        service._consumed = {"prog": "new"}
        service._poll_done.notify_all()
    saver.join(5)
    assert streamer.commits == [{"prog": "new"}]


def test_allocate_passes_the_day_start_equity(served, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    service, port = served
    seen: list[Any] = []

    def allocate(*a: Any, **k: Any) -> list[Any]:
        seen.append((a, k))
        return []

    monkeypatch.setattr(service.brain, "allocate", allocate)
    body = {"equity_sol": 10, "open_stakes": {"m": 1}, "day_start_equity_sol": 12}
    assert _call(port, "POST", "/allocate", body) == (200, {"allocations": []})
    assert seen[-1] == ((10.0, {"m": 1.0}, None), {"day_start_equity_sol": 12.0})
    assert _call(port, "POST", "/allocate", {"equity_sol": 10})[0] == 200
    assert seen[-1][1] == {"day_start_equity_sol": None}, "calls without it stay compatible"
