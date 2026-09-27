"""Minimal **read-only** Solana JSON-RPC client (standard library only).

Only query methods are allowed — the client refuses anything that could submit or sign
a transaction, so the ML module can never move funds.  Retries with exponential backoff on
rate limits (HTTP 429) and transient server errors.  A custom ``transport`` can be
injected (tests, websockets bridges, provider SDKs).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

Transport = Callable[[str, list[Any]], Any]

READ_ONLY_METHODS = frozenset(
    {
        "getSignaturesForAddress",
        "getTransaction",
        "getSlot",
        "getBlockTime",
        "getAccountInfo",
        "getMultipleAccounts",
        "getTokenLargestAccounts",
        "getTokenSupply",
        "getHealth",
        "getVersion",
    }
)


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

    def _http(self, method: str, params: list[Any]) -> Any:
        assert self.url is not None
        self._id += 1
        body = json.dumps({"jsonrpc": "2.0", "id": self._id, "method": method, "params": params}).encode()
        req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            payload = json.loads(resp.read())
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
        """A confirmed transaction in ``jsonParsed`` encoding (v0 included), or None if unavailable."""
        result = self.call(
            "getTransaction",
            [
                signature,
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0, "commitment": "confirmed"},
            ],
        )
        return dict(result) if result else None

    def get_slot(self) -> int:
        """The current confirmed slot."""
        return int(self.call("getSlot", [{"commitment": "confirmed"}]))

    def mint_authorities(self, mint: str) -> tuple[bool, bool] | None:
        """(mint_authority_revoked, freeze_authority_revoked) from the parsed mint account."""
        result = self.call("getAccountInfo", [mint, {"encoding": "jsonParsed"}])
        try:
            info = result["value"]["data"]["parsed"]["info"]
        except (KeyError, TypeError):
            return None
        return info.get("mintAuthority") is None, info.get("freezeAuthority") is None
