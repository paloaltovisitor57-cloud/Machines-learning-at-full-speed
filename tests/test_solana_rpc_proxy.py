"""Read-only RPC client behind a proxy: credentials and ``NO_PROXY`` domain matching (no network)."""

from __future__ import annotations

import base64
import http.client

import pytest

from nardis_neural.solana.ingest.rpc import READ_ONLY_METHODS, SolanaRpc


def _conn(url: str) -> http.client.HTTPConnection:
    return SolanaRpc(url)._connection()  # builds the connection object; nothing is sent


@pytest.fixture(autouse=True)
def _clean_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy"):
        monkeypatch.delenv(name, raising=False)


def test_proxy_credentials_go_into_the_connect_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://al%40ice:s3cr%3At@proxy.local:3128")
    conn = _conn("https://rpc.example.com/path")
    assert (conn.host, conn.port) == ("proxy.local", 3128)
    token = base64.b64encode(b"al@ice:s3cr:t").decode()
    assert conn._tunnel_host == "rpc.example.com"  # type: ignore[attr-defined]
    headers = conn._tunnel_headers  # type: ignore[attr-defined]
    assert headers["Proxy-Authorization"] == f"Basic {token}"
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    assert "Proxy-Authorization" not in _conn("https://rpc.example.com")._tunnel_headers  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("no_proxy", "host", "bypass"),
    [
        ("example.com", "rpc.example.com", True),
        (".example.com", "rpc.example.com", True),
        ("example.com", "example.com", True),
        ("other.org, example.com:443", "api.rpc.example.com", True),
        ("*", "anything.net", True),
        ("example.com", "badexample.com", False),
        ("rpc.example.com", "example.com", False),
        ("", "rpc.example.com", False),
    ],
)
def test_no_proxy_matches_domains_and_suffixes(
    monkeypatch: pytest.MonkeyPatch, no_proxy: str, host: str, bypass: bool
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.local:3128")
    monkeypatch.setenv("NO_PROXY", no_proxy)
    conn = _conn(f"https://{host}")
    assert (conn.host == host) is bypass


def test_client_stays_read_only() -> None:
    assert not {"sendTransaction", "requestAirdrop", "simulateTransaction"} & READ_ONLY_METHODS
