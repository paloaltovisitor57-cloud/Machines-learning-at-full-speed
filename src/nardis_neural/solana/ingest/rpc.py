"""Minimal **read-only** Solana JSON-RPC client (standard library only).

Only query methods are allowed — the client refuses anything that could submit or sign
a transaction, so the ML module can never move funds.  Retries with exponential backoff on
rate limits (HTTP 429) and transient server errors.  HTTP connections are kept alive per
thread (through an ``HTTPS_PROXY`` tunnel when one is set): opening a fresh TLS connection per
request limited throughput to about 12 requests/s on a hosted node, reuse reached 60–100.
A custom ``transport`` can be injected (tests, websockets bridges, provider SDKs).
"""

from __future__ import annotations

import http.client
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.parse
from collections.abc import Callable
from pathlib import Path
from typing import Any

Transport = Callable[[str, list[Any]], Any]

READ_ONLY_METHODS = frozenset(
    {
        "getSignaturesForAddress",
        "getTransaction",
        "getSlot",
        "getBlockTime",
        "getBlock",
        "getAccountInfo",
        "getMultipleAccounts",
        "getTokenLargestAccounts",
        "getTokenSupply",
        "getHealth",
        "getVersion",
    }
)


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE")
    return ssl.create_default_context(cafile=cafile if cafile and Path(cafile).exists() else None)


class _TunnelledHTTPS(http.client.HTTPSConnection):
    """HTTPS to ``host:port`` through an HTTP CONNECT proxy, verified against the system CAs."""

    def __init__(self, proxy_host: str, proxy_port: int, host: str, port: int, timeout: float) -> None:
        super().__init__(proxy_host, proxy_port, timeout=timeout, context=_ssl_context())
        self.set_tunnel(host, port)


class RpcError(RuntimeError):
    """Error object returned by the JSON-RPC endpoint."""

    pass


class SolanaRpc:
    """Read-only Solana JSON-RPC client over HTTP or an injected ``transport``, with retries."""

    def __init__(
        self,
        url: str | None = None,
        timeout: float = 20.0,
        retries: int = 4,
        backoff: float = 0.5,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if url is None and transport is None:
            raise ValueError("pass an RPC url or a transport")
        self.url = url
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff
        self.transport = transport
        self.sleep = sleep
        self._id = 0
        self._local = threading.local()

    def _connection(self) -> http.client.HTTPConnection:
        """This thread's kept-alive connection to the endpoint (created on first use)."""
        conn: http.client.HTTPConnection | None = getattr(self._local, "conn", None)
        if conn is not None:
            return conn
        assert self.url is not None
        u = urllib.parse.urlsplit(self.url)
        secure = u.scheme == "https"
        port = u.port or (443 if secure else 80)
        proxy = os.environ.get("HTTPS_PROXY" if secure else "HTTP_PROXY") or os.environ.get(
            "https_proxy" if secure else "http_proxy"
        )
        host = u.hostname or ""
        no_proxy = [
            h.strip() for h in (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").split(",")
        ]
        if proxy and host not in no_proxy:
            p = urllib.parse.urlsplit(proxy)
            if secure:  # TLS to the endpoint through the proxy's CONNECT tunnel
                conn = _TunnelledHTTPS(p.hostname or "", p.port or 80, host, port, self.timeout)
            else:
                conn = http.client.HTTPConnection(p.hostname or "", p.port or 80, timeout=self.timeout)
                conn.set_tunnel(host, port)
        elif secure:
            conn = http.client.HTTPSConnection(host, port, timeout=self.timeout, context=_ssl_context())
        else:
            conn = http.client.HTTPConnection(host, port, timeout=self.timeout)
        self._local.conn = conn
        return conn

    def _http(self, method: str, params: list[Any]) -> Any:
        assert self.url is not None
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}).encode()
        u = urllib.parse.urlsplit(self.url)
        path = (u.path or "/") + (f"?{u.query}" if u.query else "")
        try:
            conn = self._connection()
            conn.request("POST", path, body, {"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw = resp.read()
        except (OSError, http.client.HTTPException) as exc:
            self._local.conn = None  # reconnect on the next attempt
            raise urllib.error.URLError(exc) from exc
        if resp.status == 429:
            raise RpcError(f"{method}: HTTP 429 rate limited")
        if resp.status >= 500:
            raise urllib.error.URLError(f"{method}: HTTP {resp.status}")
        if resp.status >= 400:
            raise urllib.error.HTTPError(self.url, resp.status, resp.reason, resp.headers, None)
        payload = json.loads(raw)
        if "error" in payload:
            raise RpcError(f"{method}: {payload['error']}")
        return payload.get("result")

    def call(self, method: str, params: list[Any]) -> Any:
        """Call a read-only RPC method and return its ``result``.

        Raises ``PermissionError`` for any method outside :data:`READ_ONLY_METHODS`.  Network errors,
        5xx responses and rate limits (429 / -32005) are retried up to ``retries`` times with
        exponential backoff.
        """
        if method not in READ_ONLY_METHODS:
            raise PermissionError(
                f"{method} is not a read-only RPC method; this client never writes to chain"
            )
        attempt = 0
        while True:
            try:
                if self.transport is not None:
                    return self.transport(method, params)
                return self._http(method, params)
            except (urllib.error.URLError, TimeoutError, ConnectionError, RpcError) as exc:
                retriable = not isinstance(exc, RpcError) or "429" in str(exc) or "-32005" in str(exc)
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500 and exc.code != 429:
                    retriable = False
                attempt += 1
                if not retriable or attempt > self.retries:
                    raise
                self.sleep(self.backoff * 2 ** (attempt - 1))

    # ------------------------------------------------------------------ typed helpers
    def get_signatures(
        self, address: str, before: str | None = None, until: str | None = None, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Up to ``limit`` confirmed signatures of ``address``, newest first.

        ``before`` / ``until`` bound the page by signature (exclusive), for paging backwards.
        """
        opts: dict[str, Any] = {"limit": limit, "commitment": "confirmed"}
        if before:
            opts["before"] = before
        if until:
            opts["until"] = until
        result = self.call("getSignaturesForAddress", [address, opts])
        return list(result or [])

    def get_transaction(self, signature: str) -> dict[str, Any] | None:
        """A confirmed transaction in ``jsonParsed`` encoding (legacy, v0 and v1), or None if unavailable."""
        result = self.call(
            "getTransaction",
            [
                signature,
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 1, "commitment": "confirmed"},
            ],
        )
        return dict(result) if result else None

    def get_slot(self) -> int:
        """The current confirmed slot."""
        return int(self.call("getSlot", [{"commitment": "confirmed"}]))

    def block_time(self, slot: int) -> float | None:
        """Unix time of a slot's block, or None when the slot was skipped or is unavailable."""
        try:
            t = self.call("getBlockTime", [slot])
        except RpcError:
            return None
        return float(t) if t is not None else None

    def block_signatures(self, slot: int) -> list[str] | None:
        """Transaction signatures of a slot's block, or None when the slot was skipped."""
        try:
            block = self.call(
                "getBlock",
                [
                    slot,
                    {
                        "transactionDetails": "signatures",
                        "rewards": False,
                        "maxSupportedTransactionVersion": 1,
                        "commitment": "confirmed",
                    },
                ],
            )
        except RpcError:
            return None
        return list(block.get("signatures") or []) if block else None

    def slot_at(self, t: float, tolerance: float = 5.0, max_steps: int = 40) -> int:
        """A slot whose block time is within ``tolerance`` seconds of Unix time ``t`` (secant
        search from the current slot; skipped slots are stepped over)."""
        hi = self.get_slot()
        hi_t = self.block_time(hi)
        step = 0
        while hi_t is None and step < 50:
            step += 1
            hi_t = self.block_time(hi - step)
        if hi_t is None:
            raise RpcError("no recent block time available")
        slot, slot_t = hi - step, hi_t
        rate = 0.4  # seconds per slot, refined as we go
        for _ in range(max_steps):
            if abs(slot_t - t) <= tolerance:
                return slot
            guess = max(int(slot + (t - slot_t) / rate), 1)
            gt, probe = None, guess
            while gt is None and probe < guess + 50:
                gt = self.block_time(probe)
                probe += 1 if gt is None else 0
            if gt is None:
                raise RpcError(f"no block time around slot {guess}")
            if probe != slot and gt != slot_t:
                rate = min(max(abs((gt - slot_t) / (probe - slot)), 0.2), 1.0)
            slot, slot_t = probe, gt
        return slot

    def signature_near(self, t: float) -> str | None:
        """Any transaction signature from a block at (or just after) Unix time ``t``, usable as a
        ``before`` cursor to start listing signatures at ``t`` instead of at the chain tip."""
        slot = self.slot_at(t)
        for s in range(slot, slot + 50):
            sigs = self.block_signatures(s)
            if sigs:
                return sigs[0]
        return None

    def mint_authorities(self, mint: str) -> tuple[bool, bool] | None:
        """(mint_authority_revoked, freeze_authority_revoked) from the parsed mint account."""
        result = self.call("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        try:
            info = result["value"]["data"]["parsed"]["info"]
        except (KeyError, TypeError):
            return None
        return info.get("mintAuthority") is None, info.get("freezeAuthority") is None
