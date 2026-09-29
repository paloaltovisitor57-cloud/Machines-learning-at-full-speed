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
