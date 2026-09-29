"""Local HTTP/JSON service: the addon as a sidecar the trading system calls from any language.

The service owns one :class:`~nardis_neural.solana.brain.SolanaBrain`.  Optionally a background
thread streams the chain into it (read-only RPC), so every answer reflects the live market.
All calls are advice; nothing here can sign or send a transaction.

Endpoints (JSON in, JSON out):

=========================  =====================================================================
``GET  /health``           market clock, tracked tokens, installed models, learner status
``GET  /tokens``           tokens active in the last ``?active_seconds=120``, newest launch first
``GET  /ranking``          moonshot candidates, best ``chase_score`` first (``?limit=20``)
``GET  /assess``           full assessment of one token (``?mint=…``)
``POST /advise_trade``     ``{trade_id, mint, t?, features{}}`` → P(win / 10x / 100x), size, veto
``POST /settle_trade``     ``{trade_id, t_exit, multiple, peak_multiple?}`` → learner updates
``POST /hold_advice``      ``{mint, t_signal}`` → sell-now vs continuation value, crash risk
``POST /allocate``         ``{equity_sol, open_stakes{}, peak_equity_sol?}`` → recommended stakes
``POST /ingest``           ``{transactions: [getTransaction JSON, …]}`` pushed by the trading
                           system (slot order), decoded and fed to the brain
``POST /save``             checkpoint the workspace now
=========================  =====================================================================

Requests are serialised with a lock (the brain is not thread-safe); each call takes
milliseconds, so this is not a bottleneck for one trading system.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

from nardis_neural.solana.metalabel import TradeOutcome, TradeProposal


def _clean(obj: Any) -> Any:
    """JSON-safe copy: NaN / inf become None."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    return obj


class AddonService:
    """Routes requests to the brain under a lock; see the module docstring for the API."""

    def __init__(self, brain: Any) -> None:
        self.brain = brain
        self.lock = threading.RLock()
        self.stream_stats: dict[str, int] = {}
        self.decoder: Any = None
        """Decoder state for pushed transactions (``POST /ingest``)."""
        self._stop = threading.Event()

    # ------------------------------------------------------------------ routing
    def handle(self, method: str, path: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """Dispatch one request; returns ``(http_status, body)``."""
        url = urlsplit(path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        route = (method, url.path.rstrip("/") or "/")
        handlers = {
            ("GET", "/health"): self.health,
            ("GET", "/tokens"): lambda: self.tokens(float(query.get("active_seconds", 120))),
            ("GET", "/ranking"): lambda: self.ranking(int(query.get("limit", 20))),
            ("GET", "/assess"): lambda: self.assess(query.get("mint", "")),
            ("POST", "/advise_trade"): lambda: self.advise_trade(payload),
            ("POST", "/settle_trade"): lambda: self.settle_trade(payload),
            ("POST", "/hold_advice"): lambda: self.hold_advice(payload),
            ("POST", "/allocate"): lambda: self.allocate(payload),
            ("POST", "/ingest"): lambda: self.ingest(payload),
            ("POST", "/save"): self.save,
        }
        fn = handlers.get(route)
        if fn is None:
            return 404, {"error": f"unknown endpoint {method} {url.path}"}
        try:
            with self.lock:
                return 200, _clean(fn())
        except (KeyError, ValueError, TypeError) as exc:
            return 400, {"error": f"{type(exc).__name__}: {exc}"}
        except RuntimeError as exc:
            return 409, {"error": str(exc)}

    # ------------------------------------------------------------------ endpoints
    def health(self) -> dict[str, Any]:
        b = self.brain
        return {
            "ok": True,
            "market_time": b.market.now,
            "tokens": len(b.market.tokens),
            "models": {
                "moonshot": b.moonshot is not None,
                "tape": b.tape_model is not None,
                "stopping": b.stopping is not None,
                "edge": b.edge is not None,
            },
            "learner": {
                "settled_trades": len(b.meta.multiple),
                "pending_trades": len(b.meta.pending),
                "deployed_levels": sorted(b.meta.models),
                "value_model": b.meta.value_model is not None,
            },
            "stream": self.stream_stats,
            "archive": getattr(self, "archive_stats", {}),
        }

    def tokens(self, active_seconds: float) -> dict[str, Any]:
        m = self.brain.market
        now = m.now
        rows = [
            {"mint": mint, "age_seconds": now - m.token(mint).launch.t, "venue": m.token(mint).launch.venue}
            for mint in m.active_tokens(now, active_seconds)
        ]
        rows.sort(key=lambda r: r["age_seconds"])
        return {"time": now, "tokens": rows}

    def ranking(self, limit: int) -> dict[str, Any]:
        rows = []
        for a in self.brain.moonshot_ranking()[: max(limit, 0)]:
            m, t = a.moonshot, a.tape
            rows.append(
                {
                    "mint": a.mint,
                    "chase_score": m.get("chase_score"),
                    "expected_multiple": m.get("expected_multiple_blend", m.get("expected_multiple")),
                    "p_ge_10x": m.get("p_ge_10x"),
                    "p_ge_100x": m.get("p_ge_100x"),
                    "p_ge_1000x": m.get("p_ge_1000x"),
                    "chase_target": m.get("chase_target"),
                    "chase_edge": m.get("chase_edge"),
                    "edge_2x": m.get("edge_2x"),
                    "edge_5x": m.get("edge_5x"),
                    "edge_10x": m.get("edge_10x"),
                    "edge_100x": m.get("edge_100x"),
                    "edge_1000x": m.get("edge_1000x"),
                    "tail_ev": m.get("tail_ev"),
                    "lottery_kelly": m.get("lottery_kelly"),
                    "trust": m.get("trust"),
                    "p_collapse_1m": t.get("p_collapse_1m"),
                    "p_collapse_5m": t.get("p_collapse_5m"),
                    "flags": a.flags,
                }
            )
        return {"time": self.brain.market.now, "candidates": rows}

    def assess(self, mint: str) -> dict[str, Any]:
        if not mint:
            raise ValueError("pass ?mint=<address>")
        if mint not in self.brain.market.tokens:
            raise KeyError(f"token {mint} is not tracked")
        dumped: dict[str, Any] = self.brain.assess(mint).model_dump(mode="json", exclude={"features"})
        return dumped

    def advise_trade(self, p: dict[str, Any]) -> dict[str, Any]:
        proposal = TradeProposal(
            str(p["trade_id"]),
            str(p["mint"]),
            float(p.get("t", self.brain.market.now)),
            {str(k): float(v) for k, v in (p.get("features") or {}).items()},
        )
        with_market = bool(p.get("with_market", True))
        return asdict(self.brain.advise_trade(proposal, with_market=with_market))

    def settle_trade(self, p: dict[str, Any]) -> dict[str, Any]:
        peak = p.get("peak_multiple")
        outcome = TradeOutcome(
            str(p["trade_id"]),
            float(p.get("t_exit", self.brain.market.now)),
            float(p["multiple"]),
            float(peak) if peak is not None else None,
        )
        known = self.brain.settle_trade(outcome)
        return {"accepted": known, "settled_trades": len(self.brain.meta.multiple)}

    def hold_advice(self, p: dict[str, Any]) -> dict[str, Any]:
        mint = str(p["mint"])
        out: dict[str, Any] = dict(self.brain.hold_advice(mint, float(p["t_signal"])))
        if mint in self.brain.market.tokens:
            tape = self.brain.assess(mint).tape
            out |= {k: tape[k] for k in ("p_collapse_1m", "p_collapse_5m", "p_collapse_15m") if k in tape}
        return out

    def allocate(self, p: dict[str, Any]) -> dict[str, Any]:
        allocs = self.brain.allocate(
            float(p["equity_sol"]),
            {str(k): float(v) for k, v in (p.get("open_stakes") or {}).items()},
            float(p["peak_equity_sol"]) if p.get("peak_equity_sol") is not None else None,
        )
        return {"allocations": [asdict(a) for a in allocs]}

    def ingest(self, p: dict[str, Any]) -> dict[str, Any]:
        """Decode pushed transactions (``getTransaction`` JSON, ``jsonParsed`` encoding) into the brain."""
        from nardis_neural.solana.ingest.decoder import TransactionDecoder

        if self.decoder is None:
            self.decoder = TransactionDecoder()
        txs = p["transactions"]
        if not isinstance(txs, list):
            raise ValueError("transactions must be a list")
        accepted = rejected = undecodable = 0
        for tx in sorted(txs, key=lambda x: int(x.get("slot", 0))):
            try:
                events = list(self.decoder.decode(tx))
            except (ArithmeticError, KeyError, IndexError, TypeError, ValueError):
                undecodable += 1
                continue
            for e in events:
                try:
                    self.brain.ingest(e)
                    accepted += 1
                except (KeyError, ValueError):
                    rejected += 1
        return {"events": accepted, "rejected": rejected, "undecodable_transactions": undecodable}

    def save(self) -> dict[str, Any]:
        self.brain.save()
        return {"saved": True}

    # ------------------------------------------------------------------ live stream
    def stream(
        self,
        streamer: Any,
        poll_interval: float = 2.0,
        resolve_every: float = 10.0,
        maintenance_every: float = 600.0,
        save_every: float = 300.0,
        chunk: int = 50,
        bounded_memory: bool = True,
    ) -> threading.Thread:
        """Feed the chain into the brain on a daemon thread until :meth:`stop`.

        With ``bounded_memory`` (default) the brain switches to streaming mode: events are not
        accumulated and tokens idle for two hours are forgotten at each maintenance, so a
        sidecar can run for weeks at constant memory."""
        if bounded_memory and not self.brain.streaming:
            with self.lock:
                self.brain.enable_streaming()
        stats = self.stream_stats
        stats.update({"polls": 0, "events": 0, "rejected": 0, "errors": 0})

        def loop() -> None:
            nxt = {
                "resolve": time.time(),
                "maint": time.time() + maintenance_every,
                "save": time.time() + save_every,
            }
            while not self._stop.is_set():
                try:
                    events = list(streamer.poll())
                    # small chunks, lock released in between: a request never waits behind a
                    # whole poll's backlog
                    for i in range(0, len(events), chunk):
                        with self.lock:
                            for e in events[i : i + chunk]:
                                try:
                                    self.brain.ingest(e)
                                    stats["events"] += 1
                                except (KeyError, ValueError):
                                    stats["rejected"] += 1
                    with self.lock:
                        now = time.time()
                        if now >= nxt["resolve"]:
                            nxt["resolve"] = now + resolve_every
                            self.brain.resolve()
                        if now >= nxt["maint"]:
                            nxt["maint"] = now + maintenance_every
                            self.brain.maintenance()
                            if self.brain.streaming:
                                self.brain.evict()
                        if now >= nxt["save"]:
                            nxt["save"] = now + save_every
                            self.brain.save()
                    stats["polls"] += 1
                except Exception:  # a network hiccup must never kill the sidecar
                    stats["errors"] += 1
                self._stop.wait(poll_interval)

        thread = threading.Thread(target=loop, name="addon-stream", daemon=True)
        thread.start()
        return thread

    def train_on_archive(
        self, path: Any, mapping: Any = None, table: str | None = None, every: float = 600.0
    ) -> threading.Thread:
        """Keep training the meta-learner on the trade archive: rescan every ``every`` seconds
        and learn only the trades it has not seen yet."""
        from nardis_neural.solana.archive import train_from_archive

        self.archive_stats: dict[str, Any] = {"scans": 0, "errors": 0, "last": {}}

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    with self.lock:
                        out = train_from_archive(self.brain.meta, path, mapping, table)
                        if out["added"]:
                            self.brain.meta.save(self.brain.root / "meta")
                    self.archive_stats["last"] = {k: v for k, v in out.items() if k != "refit"}
                    self.archive_stats["scans"] += 1
                except Exception as exc:  # a half-written file must not kill the sidecar
                    self.archive_stats["errors"] += 1
                    self.archive_stats["last_error"] = f"{type(exc).__name__}: {exc}"[:300]
                self._stop.wait(every)

        thread = threading.Thread(target=loop, name="addon-archive", daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """Stop the stream thread and checkpoint."""
        self._stop.set()
        with self.lock:
            self.brain.save()


def make_server(service: AddonService, host: str = "127.0.0.1", port: int = 8787) -> ThreadingHTTPServer:
    """An HTTP server bound to ``host:port`` that routes to ``service``."""

    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reply(*service.handle("GET", self.path, {}))

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError as exc:
                self._reply(400, {"error": f"invalid JSON: {exc}"})
                return
            if not isinstance(payload, dict):
                self._reply(400, {"error": "the body must be a JSON object"})
                return
            self._reply(*service.handle("POST", self.path, payload))

        def log_message(self, format: str, *args: Any) -> None:  # quiet by default
            return

    return ThreadingHTTPServer((host, port), Handler)
