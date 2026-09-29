"""pump.fun program constants and Anchor event (de)serialisation.

pump.fun emits Anchor events through ``Program data: <base64>`` log lines.  Each payload
is an 8-byte discriminator (``sha256("event:<Name>")[:8]``) followed by the Borsh-encoded
struct.  Newer program versions append extra fields (real reserves, fee and creator-fee
details …); the decoders below read the stable leading fields and ignore any trailing
bytes, so they keep working across upgrades that only append fields.

Amounts: SOL in lamports (1e9 per SOL); pump.fun tokens use 6 decimals.
"""

from __future__ import annotations

import base64
import hashlib
import struct
from dataclasses import dataclass

from nardis_neural.solana.ingest.base58 import b58decode, b58encode

PUMP_FUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
PUMP_SWAP_PROGRAM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"
RAYDIUM_AMM_V4_PROGRAM = "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8"
RAYDIUM_CPMM_PROGRAM = "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C"
METEORA_DLMM_PROGRAM = "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9t2dgsbFM"
ORCA_WHIRLPOOL_PROGRAM = "whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc"
SYSTEM_PROGRAM = "11111111111111111111111111111111"
WSOL_MINT = "So11111111111111111111111111111111111111112"
LAMPORTS_PER_SOL = 1_000_000_000
PUMP_TOKEN_DECIMALS = 6

AMM_PROGRAMS: dict[str, str] = {
    PUMP_SWAP_PROGRAM: "pumpswap",
    RAYDIUM_AMM_V4_PROGRAM: "raydium",
    RAYDIUM_CPMM_PROGRAM: "raydium",
    METEORA_DLMM_PROGRAM: "meteora",
    ORCA_WHIRLPOOL_PROGRAM: "orca",
}

JITO_TIP_ACCOUNTS: frozenset[str] = frozenset(
    {
        "96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5",
        "HFqU5x63VTqvQss8hp11i4wVV8bD44PvwucfZ2bU7gRe",
        "Cw8CFyM9FkoMi7K7Crf6HNQqf4uEMzpKw6QNghXLvLkY",
        "ADaUMid9yfUytqMBgopwjb2DTLSokTSzL1zt6iGPaS49",
        "DfXygSm4jCyNCybVYYK6DwvWqjKee8pbDmJGcLWNDXjh",
        "ADuUkR4vqLUMWXxW9gh6D6L8pMSawimctcNZ5pGwDcEt",
        "DttWaMuVvTiduZRnguLF7jNxTgiMBZ1hyAumKUiL2KRL",
        "3AVi9Tg9Uo68tJfuvoKvqKNWKkC5wPdSSdeBnizKZ6jT",
    }
)


def discriminator(name: str) -> bytes:
    """Anchor event discriminator: the first 8 bytes of ``sha256("event:<name>")``."""
    return hashlib.sha256(f"event:{name}".encode()).digest()[:8]


TRADE_DISC = discriminator("TradeEvent")
CREATE_DISC = discriminator("CreateEvent")
COMPLETE_DISC = discriminator("CompleteEvent")


@dataclass(frozen=True)
class PumpTrade:
    """pump.fun ``TradeEvent``: amounts in lamports / raw token units, virtual reserves after the trade."""

    mint: str
    sol_lamports: int
    token_units: int
    is_buy: bool
    user: str
    timestamp: int
    virtual_sol_lamports: int
    virtual_token_units: int


@dataclass(frozen=True)
class PumpCreate:
    """pump.fun ``CreateEvent``: token metadata, mint, bonding curve and creator."""

    name: str
    symbol: str
    uri: str
    mint: str
    bonding_curve: str
    user: str


@dataclass(frozen=True)
class PumpComplete:
    """pump.fun ``CompleteEvent``: the bonding curve of ``mint`` completed."""

    user: str
    mint: str
    bonding_curve: str
    timestamp: int


PumpEvent = PumpTrade | PumpCreate | PumpComplete


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data, self.pos = data, 0

    def take(self, n: int) -> bytes:
        if self.pos + n > len(self.data):
            raise ValueError("truncated event payload")
        out = self.data[self.pos : self.pos + n]
        self.pos += n
        return out

    def pubkey(self) -> str:
        return b58encode(self.take(32))

    def u64(self) -> int:
        return int(struct.unpack("<Q", self.take(8))[0])

    def i64(self) -> int:
        return int(struct.unpack("<q", self.take(8))[0])

    def boolean(self) -> bool:
        return self.take(1) != b"\0"

    def string(self) -> str:
        n = int(struct.unpack("<I", self.take(4))[0])
        return self.take(n).decode("utf-8", errors="replace")


def decode_event(payload: bytes) -> PumpEvent | None:
    """Decode one ``Program data`` payload; None for unrelated events."""
    disc, r = payload[:8], _Reader(payload[8:])
    if disc == TRADE_DISC:
        return PumpTrade(r.pubkey(), r.u64(), r.u64(), r.boolean(), r.pubkey(), r.i64(), r.u64(), r.u64())
    if disc == CREATE_DISC:
        return PumpCreate(r.string(), r.string(), r.string(), r.pubkey(), r.pubkey(), r.pubkey())
    if disc == COMPLETE_DISC:
        return PumpComplete(r.pubkey(), r.pubkey(), r.pubkey(), r.i64())
    return None


def decode_log_events(logs: list[str], program: str | None = PUMP_FUN_PROGRAM) -> list[PumpEvent]:
    """pump.fun events in a transaction's ``Program data:`` log lines; other or corrupt lines are skipped.

    Other programs (routers, forks) emit events with the same Anchor name, hence the same
    discriminator, in different layouts.  When the logs carry ``Program <id> invoke`` lines, only
    data logged while ``program`` is the executing program is decoded.
    """
    out: list[PumpEvent] = []
    stack: list[str] = []
    attributed = program is not None and any(" invoke [" in line for line in logs)
    for line in logs:
        if line.startswith("Program ") and " invoke [" in line:
            stack.append(line.split()[1])
            continue
        if line.startswith("Program ") and (line.endswith(" success") or " failed" in line):
            parts = line.split()
            if len(parts) > 1 and stack and stack[-1] == parts[1]:
                stack.pop()
            continue
        if not line.startswith("Program data: "):
            continue
        if attributed and (not stack or stack[-1] != program):
            continue
        try:
            ev = decode_event(base64.b64decode(line[len("Program data: ") :]))
        except (ValueError, struct.error):
            continue
        if ev is not None:
            out.append(ev)
    return out


SWAP_BUY_DISC = discriminator("BuyEvent")
SWAP_SELL_DISC = discriminator("SellEvent")


@dataclass(frozen=True)
class PumpSwapTrade:
    """PumpSwap ``BuyEvent`` / ``SellEvent``: a trade against one pool, in raw units.

    ``is_base_buy`` is True when the user received the pool's *base* asset.  Pools usually hold
    the token as base and WSOL as quote, but some are the other way round, so the orientation is
    resolved against the transaction's token accounts (see the decoder).
    """

    is_base_buy: bool
    timestamp: int
    base_amount: int
    quote_amount: int
    pool_base_reserves: int
    pool_quote_reserves: int
    pool: str
    user: str
    user_base_account: str
    user_quote_account: str


def decode_pumpswap_event(payload: bytes) -> PumpSwapTrade | None:
    """Decode one PumpSwap ``Program data`` payload; None for other events."""
    disc = payload[:8]
    if disc not in (SWAP_BUY_DISC, SWAP_SELL_DISC):
        return None
    r = _Reader(payload[8:])
    ts = r.i64()
    base_amount = r.u64()  # base_amount_out (buy) / base_amount_in (sell)
    r.u64()  # max_quote_amount_in / min_quote_amount_out
    r.u64()  # user base reserves
    r.u64()  # user quote reserves
    pool_base, pool_quote = r.u64(), r.u64()
    quote_amount = r.u64()  # quote_amount_in / quote_amount_out
    for _ in range(6):  # fee fields and the two quote totals
        r.u64()
    pool, user, base_acct, quote_acct = r.pubkey(), r.pubkey(), r.pubkey(), r.pubkey()
    return PumpSwapTrade(
        disc == SWAP_BUY_DISC,
        ts,
        base_amount,
        quote_amount,
        pool_base,
        pool_quote,
        pool,
        user,
        base_acct,
        quote_acct,
    )


def decode_pumpswap_log_events(logs: list[str]) -> list[PumpSwapTrade]:
    """PumpSwap trades logged by the PumpSwap program itself (attributed like :func:`decode_log_events`)."""
    out: list[PumpSwapTrade] = []
    stack: list[str] = []
    for line in logs:
        if line.startswith("Program ") and " invoke [" in line:
            stack.append(line.split()[1])
            continue
        if line.startswith("Program ") and (line.endswith(" success") or " failed" in line):
            parts = line.split()
            if len(parts) > 1 and stack and stack[-1] == parts[1]:
                stack.pop()
            continue
        if not line.startswith("Program data: ") or not stack or stack[-1] != PUMP_SWAP_PROGRAM:
            continue
        try:
            ev = decode_pumpswap_event(base64.b64decode(line[len("Program data: ") :]))
        except (ValueError, struct.error):
            continue
        if ev is not None:
            out.append(ev)
    return out


# ---------------------------------------------------------------- encoders (fixtures / replay tools)
def _pk(key: str) -> bytes:
    raw = b58decode(key)
    if len(raw) != 32:
        raise ValueError(f"not a 32-byte public key: {key}")
    return raw


def _string(s: str) -> bytes:
    b = s.encode()
    return struct.pack("<I", len(b)) + b


def encode_event(ev: PumpEvent, trailing: bytes = b"") -> str:
    """Base64 ``Program data`` payload for an event (``trailing`` mimics newer appended fields)."""
    if isinstance(ev, PumpTrade):
        body = (
            TRADE_DISC
            + _pk(ev.mint)
            + struct.pack("<QQ?", ev.sol_lamports, ev.token_units, ev.is_buy)
            + _pk(ev.user)
            + struct.pack("<qQQ", ev.timestamp, ev.virtual_sol_lamports, ev.virtual_token_units)
        )
    elif isinstance(ev, PumpCreate):
        body = (
            CREATE_DISC
            + _string(ev.name)
            + _string(ev.symbol)
            + _string(ev.uri)
            + _pk(ev.mint)
            + _pk(ev.bonding_curve)
            + _pk(ev.user)
        )
    else:
        body = (
            COMPLETE_DISC
            + _pk(ev.user)
            + _pk(ev.mint)
            + _pk(ev.bonding_curve)
            + struct.pack("<q", ev.timestamp)
        )
    return base64.b64encode(body + trailing).decode()


def pubkey_from_seed(seed: str) -> str:
    """Deterministic fake public key (tests / simulations only)."""
    return b58encode(hashlib.sha256(seed.encode()).digest())
