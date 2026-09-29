"""Real-chain ingestion: base58, pump.fun event codec, transaction decoding, read-only RPC,
polling streamer and the live loop into a SolanaBrain."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from nardis_neural.cli import app
from nardis_neural.solana import LaunchSimSpec, SolanaMarket, simulate_launches
from nardis_neural.solana.events import LiquidityChange, Migration, Swap, TokenLaunch, Transfer
from nardis_neural.solana.features import SolanaFeatureBuilder
from nardis_neural.solana.ingest import (
    ChainStreamer,
    SolanaRpc,
    TransactionDecoder,
    decode_transactions,
    run_live,
)
from nardis_neural.solana.ingest.base58 import b58decode, b58encode
from nardis_neural.solana.ingest.encode import as_pubkey, events_to_transactions
from nardis_neural.solana.ingest.pumpfun import (
    JITO_TIP_ACCOUNTS,
    ORCA_WHIRLPOOL_PROGRAM,
    PUMP_FUN_PROGRAM,
    SYSTEM_PROGRAM,
    WSOL_MINT,
    PumpComplete,
    PumpCreate,
    PumpTrade,
    decode_event,
    decode_log_events,
    encode_event,
    pubkey_from_seed,
)

MINT = pubkey_from_seed("mint")
USER = pubkey_from_seed("user")
T = 1_750_000_000


# ---------------------------------------------------------------- codec
def test_base58_roundtrip_and_known_vectors() -> None:
    assert b58encode(bytes(32)) == SYSTEM_PROGRAM
    assert b58decode(SYSTEM_PROGRAM) == bytes(32)
    for raw in (b"\0\0abc", b"\xff" * 32, b"hello world"):
        assert b58decode(b58encode(raw)) == raw
    assert len(b58decode(WSOL_MINT)) == 32
    with pytest.raises(ValueError):
        b58decode("0OIl")


def test_pump_event_codec_tolerates_appended_fields() -> None:
    trade = PumpTrade(MINT, 1_500_000_000, 42_000_000, True, USER, T, 31_500_000_000, 1_020_000_000_000_000)
    create = PumpCreate("Dog", "DOG", "ipfs://x", MINT, pubkey_from_seed("curve"), USER)
    complete = PumpComplete(USER, MINT, pubkey_from_seed("curve"), T)
    for ev in (trade, create, complete):
        payload = base64.b64decode(encode_event(ev, trailing=bytes(120)))
        assert decode_event(payload) == ev
    assert decode_event(b"\x01" * 8 + bytes(100)) is None
    logs = [
        "Program log: hi",
        f"Program data: {encode_event(trade)}",
        "Program data: !!notbase64",
        "Program data: AAAA",
    ]
    assert decode_log_events(logs) == [trade]


# ---------------------------------------------------------------- decoder
def _tx(
    logs: list[str],
    ixs: list[dict[str, Any]],
    fee: int = 5000,
    err: Any = None,
    blocktime: int = T,
    pre: list[dict[str, Any]] | None = None,
    post: list[dict[str, Any]] | None = None,
    keys: list[str] | None = None,
    slot: int = 1,
) -> dict[str, Any]:
    return {
        "slot": slot,
        "blockTime": blocktime,
        "meta": {
            "err": err,
            "fee": fee,
            "logMessages": logs,
            "preTokenBalances": pre or [],
            "postTokenBalances": post or [],
            "innerInstructions": [],
        },
        "transaction": {
            "signatures": ["s"],
            "message": {"accountKeys": [USER, *(keys or [])], "instructions": ixs},
        },
    }


def _transfer(src: str, dst: str, lamports: int) -> dict[str, Any]:
    return {
        "programId": SYSTEM_PROGRAM,
        "parsed": {"type": "transfer", "info": {"source": src, "destination": dst, "lamports": lamports}},
    }


PUMP_IX = {"programId": PUMP_FUN_PROGRAM, "accounts": [], "data": ""}


def test_decoder_pump_lifecycle_fees_and_tips() -> None:
    dec = TransactionDecoder()
    create = _tx(
        [f"Program data: {encode_event(PumpCreate('A', 'A', '', MINT, pubkey_from_seed('c'), USER))}"],
        [PUMP_IX],
    )
    trade = PumpTrade(
        MINT, 2_000_000_000, 60_000_000_000_000, True, USER, T, 32_000_000_000, 1_000_000_000_000_000
    )
    buy = _tx(
        [f"Program data: {encode_event(trade)}"],
        [
            PUMP_IX,
            _transfer(USER, sorted(JITO_TIP_ACCOUNTS)[0], 1_000_000),
            _transfer(USER, pubkey_from_seed("friend"), 3 * 10**9),
            _transfer(USER, pubkey_from_seed("x"), 1000),
        ],
        fee=5000 + 250_000,
    )
    done = _tx(
        [f"Program data: {encode_event(PumpComplete(USER, MINT, pubkey_from_seed('c'), T))}"], [PUMP_IX]
    )
    failed = _tx([f"Program data: {encode_event(trade)}"], [PUMP_IX], err={"InstructionError": [0, "Custom"]})
    events = [e for tx in (create, buy, failed, done) for e in dec.decode(tx)]
    kinds = [type(e).__name__ for e in events]
    assert kinds == ["TokenLaunch", "Transfer", "Swap", "Migration"], kinds
    launch, transfer, swap, mig = events
    assert isinstance(launch, TokenLaunch) and launch.venue == "pump_fun" and launch.creator == USER
    assert isinstance(transfer, Transfer) and transfer.sol_amount == 3.0, (
        "dust and tips are not funding transfers"
    )
    assert isinstance(swap, Swap)
    assert swap.is_buy and swap.sol_amount == 2.0 and swap.token_amount == 60_000_000.0
    assert swap.sol_reserve == 32.0 and swap.token_reserve == 1_000_000_000.0
    assert swap.jito_tip == pytest.approx(0.001) and swap.priority_fee == pytest.approx(0.00025)
    assert isinstance(mig, Migration) and mig.sol_reserve == pytest.approx(2.0)
    assert dec.stats.failed == 1
    ts = [e.t for e in events]
    assert ts == sorted(ts), "ordered despite 1 s blockTime"
    assert len(set(ts)) == 3, "one distinct timestamp per successful transaction (events in a tx share it)"
    m = SolanaMarket()
    m.ingest_many(events)
    assert m.token(MINT).migrated_at is not None


def _vault_rows(owner: str, sol: int, tok: int, idx0: int = 1, mint: str = MINT) -> list[dict[str, Any]]:
    return [
        {
            "accountIndex": idx0,
            "mint": WSOL_MINT,
            "owner": owner,
            "uiTokenAmount": {"amount": str(sol), "decimals": 9},
        },
        {
            "accountIndex": idx0 + 1,
            "mint": mint,
            "owner": owner,
            "uiTokenAmount": {"amount": str(tok), "decimals": 6},
        },
    ]


def test_decoder_amm_swaps_liquidity_and_mint_info() -> None:
    calls: list[str] = []

    def mint_info(m: str) -> tuple[bool, bool]:
        calls.append(m)
        return False, True

    dec = TransactionDecoder(mint_info=mint_info)
    pool = pubkey_from_seed("pool")
    keys = [pubkey_from_seed("vs"), pubkey_from_seed("vt")]
    ray = {"programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", "accounts": [], "data": ""}
    add = _tx([], [ray], pre=[], post=_vault_rows(pool, 50 * 10**9, 10**15), keys=keys)
    buy = _tx(
        [],
        [ray],
        pre=_vault_rows(pool, 50 * 10**9, 10**15),
        post=_vault_rows(pool, 51 * 10**9, 98 * 10**13),
        keys=keys,
    )
    pull = _tx(
        [],
        [ray],
        pre=_vault_rows(pool, 51 * 10**9, 98 * 10**13),
        post=_vault_rows(pool, 10**9, 2 * 10**13),
        keys=keys,
    )
    evs = [e for tx in (add, buy, pull) for e in dec.decode(tx)]
    assert [type(e).__name__ for e in evs] == ["TokenLaunch", "LiquidityChange", "Swap", "LiquidityChange"]
    launch, _, swap, pulled = evs
    assert isinstance(launch, TokenLaunch) and launch.venue == "raydium" and not launch.mint_authority_revoked
    assert calls == [MINT]
    assert isinstance(swap, Swap) and swap.is_buy and swap.sol_amount == pytest.approx(1.0)
    assert swap.token_amount == pytest.approx(2e7) and swap.sol_reserve == pytest.approx(51.0)
    assert isinstance(pulled, LiquidityChange) and pulled.sol_delta == pytest.approx(-50.0)
    m = SolanaMarket()
    m.ingest_many(evs)
    assert m.rugged(MINT), "a 98% LP pull is attributed as a rug"


def test_decoder_clmm_prices_from_execution() -> None:
    dec = TransactionDecoder()
    pool = pubkey_from_seed("whirl")
    orca = {"programId": ORCA_WHIRLPOOL_PROGRAM, "accounts": [], "data": ""}
    keys = [pubkey_from_seed("a"), pubkey_from_seed("b")]
    dec.decode(_tx([], [orca], pre=[], post=_vault_rows(pool, 100 * 10**9, 10**12), keys=keys))
    evs = dec.decode(
        _tx(
            [],
            [orca],
            pre=_vault_rows(pool, 100 * 10**9, 10**12),
            post=_vault_rows(pool, 102 * 10**9, 9 * 10**11),
            keys=keys,
        )
    )
    swap = evs[-1]
    assert isinstance(swap, Swap)
    assert swap.sol_reserve / swap.token_reserve == pytest.approx(2.0 / 100_000.0), (
        "execution price, not vault ratio"
    )


def test_simulated_market_roundtrips_through_transactions() -> None:
    store, _ = simulate_launches(LaunchSimSpec(n_tokens=6, seed=9, n_retail=150))
    txs = events_to_transactions(store.events)
    decoded = decode_transactions(json.loads(json.dumps(txs)))  # through real JSON
    orig = [e for e in store.sorted() if isinstance(e, Swap)]
    got = [e for e in decoded if isinstance(e, Swap)]
    assert len(orig) == len(got)
    for a, b in zip(orig, got, strict=True):
        assert as_pubkey(a.mint) == b.mint and as_pubkey(a.wallet) == b.wallet and a.is_buy == b.is_buy
        assert b.sol_reserve == pytest.approx(a.sol_reserve, rel=1e-6)
        assert b.token_amount == pytest.approx(a.token_amount, rel=1e-6, abs=1e-6)
        assert b.jito_tip == pytest.approx(a.jito_tip, abs=1e-9)
    a_m, b_m = SolanaMarket(), SolanaMarket()
    a_m.ingest_many(store.sorted())
    b_m.ingest_many(decoded)
    builder = SolanaFeatureBuilder()
    for mint in a_m.tokens:
        fa = builder.explain(builder.current_features(a_m.token(mint), a_m.wallets, a_m.token(mint).last_t))
        pk = as_pubkey(mint)
        fb = builder.explain(builder.current_features(b_m.token(pk), b_m.wallets, b_m.token(pk).last_t))
        for key in (
            "holders_log",
            "top10_share",
            "dev_share",
            "bundle_share",
            "bonding_progress",
            "n_swaps_log",
        ):
            assert fb[key] == pytest.approx(fa[key], rel=1e-4, abs=1e-6), (mint, key)


# ---------------------------------------------------------------- RPC & streaming
class FakeChain:
    """Serves transactions through the JSON-RPC surface; reveals more on every signature poll."""

    def __init__(self, txs: list[dict[str, Any]], per_poll: int, reveal_on: str = PUMP_FUN_PROGRAM) -> None:
        self.txs = txs
        self.reveal_on = reveal_on
        for i, tx in enumerate(txs):
            tx["transaction"]["signatures"] = [f"sig{i}"]
        self.by_sig = {tx["transaction"]["signatures"][0]: tx for tx in txs}
        self.visible = 0
        self.per_poll = per_poll
        self.per_poll_after_first: int | None = None
        self.calls: list[str] = []

    def __call__(self, method: str, params: list[Any]) -> Any:
        self.calls.append(method)
        if method == "getSignaturesForAddress":
            opts = params[1]
            if opts.get("before") is None and params[0] == self.reveal_on:  # once per streamer poll
                step = (
                    self.per_poll_after_first if self.visible and self.per_poll_after_first else self.per_poll
                )
                self.visible = min(len(self.txs), self.visible + step)
            newest_first = list(reversed(self.txs[: self.visible]))
            sigs = [
                {
                    "signature": tx["transaction"]["signatures"][0],
                    "slot": tx["slot"],
                    "blockTime": tx["blockTime"],
                    "err": None,
                }
                for tx in newest_first
            ]
            if opts.get("until"):
                sigs = sigs[: [s["signature"] for s in sigs].index(opts["until"])]
            if opts.get("before"):
                sigs = sigs[[s["signature"] for s in sigs].index(opts["before"]) + 1 :]
            return sigs[: opts["limit"]]
        if method == "getTransaction":
            return self.by_sig.get(params[0])
        if method == "getAccountInfo":
            return {"value": {"data": {"parsed": {"info": {"mintAuthority": None, "freezeAuthority": "X"}}}}}
        raise AssertionError(method)


def test_rpc_is_read_only_and_retries() -> None:
    rpc = SolanaRpc(transport=lambda m, p: {"value": None})
    with pytest.raises(PermissionError):
        rpc.call("sendTransaction", ["..."])
    with pytest.raises(PermissionError):
        rpc.call("requestAirdrop", [])
    attempts: list[int] = []

    def flaky(method: str, params: list[Any]) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            from nardis_neural.solana.ingest.rpc import RpcError

            raise RpcError("429 Too Many Requests")
        return 123

    sleeps: list[float] = []
    assert SolanaRpc(transport=flaky, sleep=sleeps.append).get_slot() == 123
    assert sleeps == [0.5, 1.0]
    auth = SolanaRpc(transport=FakeChain([], 1)).mint_authorities(MINT)
    assert auth == (True, False)
    with pytest.raises(ValueError):
        SolanaRpc()


def test_streamer_cursor_dedupe_and_order(tmp_path: Path) -> None:
    store, _ = simulate_launches(LaunchSimSpec(n_tokens=3, seed=4, n_retail=80, duration_seconds=1200))
    txs = events_to_transactions([e for e in store.events if not isinstance(e, Transfer) or e.t > T - 10**6])
    chain = FakeChain(txs, per_poll=900)  # first poll fits the initial window; later bursts exceed a page
    chain.per_poll_after_first = len(txs) // 3 + 1
    streamer = ChainStreamer(
        SolanaRpc(transport=chain), state_file=tmp_path / "cursor.json", programs=[PUMP_FUN_PROGRAM, "other"]
    )
    seen: list[Any] = []
    for _ in range(6):
        seen += streamer.poll()
    assert streamer.gaps == 0
    swaps = [e for e in seen if isinstance(e, Swap)]
    assert len(swaps) == sum(isinstance(e, Swap) for e in store.events), (
        "every swap exactly once across programs"
    )
    assert json.loads((tmp_path / "cursor.json").read_text())["cursor"]
    resumed = ChainStreamer(
        SolanaRpc(transport=chain), state_file=tmp_path / "cursor.json", programs=[PUMP_FUN_PROGRAM]
    )
    assert resumed.poll() == [], "a restarted streamer resumes from its cursor"


def test_run_live_loop_mechanics() -> None:
    class FakeBrain:
        def __init__(self) -> None:
            self.ingested = 0
            self.saved = False

        def ingest(self, e: Any) -> None:
            if isinstance(e, Swap) and e.sol_amount < 0:
                raise ValueError("bad")
            self.ingested += 1

        def assess_active(self) -> list[Any]:
            return []

        def resolve(self) -> int:
            return 0

        def maintenance(self) -> dict[str, Any]:
            return {}

        def save(self) -> None:
            self.saved = True

    store, _ = simulate_launches(LaunchSimSpec(n_tokens=2, seed=1, n_retail=50, duration_seconds=900))
    chain = FakeChain(events_to_transactions(store.events), per_poll=500)
    ticks = iter(range(0, 10_000, 7))
    brain = FakeBrain()
    stats = run_live(
        brain,
        ChainStreamer(SolanaRpc(transport=chain)),
        assess_every=10,
        maintenance_every=20,
        max_polls=5,
        sleep=lambda s: None,
        clock=lambda: float(next(ticks)),
    )
    assert stats["polls"] == 5 and stats["events"] == brain.ingested > 0 and brain.saved
    assert stats["maintenance"] >= 1


def test_chain_to_brain_integration(tmp_path: Path) -> None:
    """History → bootstrap; later chain activity → decoded → streamed live → assessments."""
    from nardis_neural.solana import SolanaBrain, SolanaConfig
    from nardis_neural.solana.market import EventStore
    from tests.conftest import make_tiny_config

    hist, _ = simulate_launches(
        LaunchSimSpec(n_tokens=8, seed=21, n_retail=150, duration_seconds=3600, mint_prefix="H")
    )
    hist_decoded = EventStore(decode_transactions(events_to_transactions(hist.events)))
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    SolanaBrain.bootstrap(
        tmp_path / "ws", hist_decoded, SolanaConfig(sample_interval_seconds=30.0), base, device="cpu"
    )
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    live, _ = simulate_launches(
        LaunchSimSpec(
            n_tokens=2,
            seed=22,
            n_retail=150,
            duration_seconds=900,
            mint_prefix="L",
            start_time=1_750_000_000 + 2 * 3600,
        )
    )
    chain = FakeChain(events_to_transactions([e for e in live.events if e.t > 1_750_000_000]), per_poll=400)
    out = tmp_path / "assess.jsonl"
    clock = iter(range(0, 100_000, 11))
    with out.open("w") as fh:
        stats = run_live(
            brain,
            ChainStreamer(SolanaRpc(transport=chain)),
            assess_every=10,
            maintenance_every=10_000,
            out=fh,
            max_polls=6,
            sleep=lambda s: None,
            clock=lambda: float(next(clock)),
        )
    assert stats["events"] > 0 and stats["rejected"] == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows and {"risk", "expected_net_return", "flags"} <= set(rows[0])


def test_decode_cli(tmp_path: Path) -> None:
    store, _ = simulate_launches(LaunchSimSpec(n_tokens=2, seed=3, n_retail=60, duration_seconds=900))
    with (tmp_path / "txs.jsonl").open("w") as fh:
        for tx in events_to_transactions(store.events):
            fh.write(json.dumps(tx) + "\n")
    res = CliRunner().invoke(
        app, ["solana", "decode", "--input", str(tmp_path / "txs.jsonl"), "--out", str(tmp_path / "events")]
    )
    assert res.exit_code == 0, res.output
    assert json.loads(res.output)["events"]["Swap"] == sum(isinstance(e, Swap) for e in store.events)
    res = CliRunner().invoke(
        app, ["solana", "stream", "--workspace", str(tmp_path)], env={"SOLANA_RPC_URL": ""}
    )
    assert res.exit_code != 0


def test_get_transaction_accepts_version_one_transactions() -> None:
    seen: list[Any] = []

    def transport(method: str, params: list[Any]) -> Any:
        seen.append(params)
        return {"slot": 1}

    SolanaRpc(transport=transport).get_transaction("sig")
    assert seen[0][1]["maxSupportedTransactionVersion"] == 1


def test_slot_search_and_seek_cursor_skip_missing_slots() -> None:
    from nardis_neural.solana.ingest.rpc import RpcError

    def transport(method: str, params: list[Any]) -> Any:
        if method == "getSlot":
            return 100_000
        slot = int(params[0])
        if slot % 7 == 0:  # skipped slots
            raise RpcError("slot skipped")
        if method == "getBlockTime":
            return 1_000_000 + int(0.45 * slot)
        if method == "getBlock":
            return {"signatures": [f"sig{slot}"]}
        raise AssertionError(method)

    rpc = SolanaRpc(transport=transport)
    target = 1_000_000 + 0.45 * 40_000
    slot = rpc.slot_at(target)
    assert abs(1_000_000 + int(0.45 * slot) - target) <= 5
    sig = rpc.signature_near(target)
    assert sig is not None and abs(int(sig[3:]) - slot) < 50


def test_one_sided_pool_change_is_not_a_swap() -> None:
    """A token-only vault change (no SOL moved) is not a swap and must not divide by zero."""
    from nardis_neural.solana.ingest.pumpfun import ORCA_WHIRLPOOL_PROGRAM

    dec = TransactionDecoder()
    pool = pubkey_from_seed("pool")
    keys = [pubkey_from_seed("vs"), pubkey_from_seed("vt")]
    orca = {"programId": ORCA_WHIRLPOOL_PROGRAM, "accounts": [], "data": ""}
    seed = _tx([], [orca], pre=[], post=_vault_rows(pool, 50 * 10**9, 10**15), keys=keys)
    one_sided = _tx(
        [],
        [orca],
        pre=_vault_rows(pool, 50 * 10**9, 10**15),
        post=_vault_rows(pool, 50 * 10**9, 2 * 10**15),
        keys=keys,
    )
    evs = [e for tx in (seed, one_sided) for e in dec.decode(tx)]
    assert not any(isinstance(e, Swap) for e in evs)


def test_history_walker_skips_undecodable_transactions() -> None:
    from collections.abc import Mapping

    from nardis_neural.solana.ingest.history import HistoryWalker

    class Broken(TransactionDecoder):
        def decode(self, tx: Mapping[str, Any]) -> list[Any]:
            raise ZeroDivisionError

    walker = HistoryWalker(SolanaRpc(transport=lambda m, p: None), 0.0, 1.0, decoder=Broken(), seek=False)
    assert walker._decode({"slot": 1}) == [] and walker.stats["decode_errors"] == 1


def test_pump_events_are_attributed_to_the_emitting_program() -> None:
    trade = PumpTrade(MINT, 1_500_000_000, 42_000_000, True, USER, T, 31_500_000_000, 1_020_000_000_000_000)
    other = PumpTrade(MINT, 0, 7, False, USER, T, 0, 5)  # same event name from a router program
    logs = [
        "Program Router1111111111111111111111111111111 invoke [1]",
        f"Program data: {encode_event(other)}",
        f"Program {PUMP_FUN_PROGRAM} invoke [2]",
        f"Program data: {encode_event(trade)}",
        f"Program {PUMP_FUN_PROGRAM} success",
        f"Program data: {encode_event(other)}",
        "Program Router1111111111111111111111111111111 success",
    ]
    assert decode_log_events(logs) == [trade]
    assert len(decode_log_events(logs, program=None)) == 3


def test_decoder_skips_non_sol_quoted_curves() -> None:
    """BuyV2 / SellV2 curves report zero SOL reserves; their trades are skipped, not zero-priced."""
    dec = TransactionDecoder()
    v2 = PumpTrade(MINT, 0, 6_000_000, True, USER, T, 0, 972_262_028_209_242)
    later = PumpTrade(MINT, 2_000_000_000, 60_000_000, True, USER, T, 32_000_000_000, 10**15)
    evs = [e for ev in (v2, later) for e in dec.decode(_tx([f"Program data: {encode_event(ev)}"], [PUMP_IX]))]
    assert not any(isinstance(e, Swap) for e in evs) and MINT in dec.non_sol_quoted


def test_first_sight_of_an_existing_pool_is_not_a_creation() -> None:
    dec = TransactionDecoder()
    pool = pubkey_from_seed("pool")
    keys = [pubkey_from_seed("vs"), pubkey_from_seed("vt")]
    ray = {"programId": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8", "accounts": [], "data": ""}
    swap = _tx(
        [],
        [ray],
        pre=_vault_rows(pool, 50 * 10**9, 10**15),
        post=_vault_rows(pool, 51 * 10**9, 98 * 10**13),
        keys=keys,
    )
    launch = next(e for e in dec.decode(swap) if isinstance(e, TokenLaunch))
    assert launch.creator == "unknown"


def _swap_event(
    buy: bool, base_amt: int, quote_amt: int, base_res: int, quote_res: int, accts: list[str]
) -> str:
    import struct

    from nardis_neural.solana.ingest.pumpfun import SWAP_BUY_DISC, SWAP_SELL_DISC, _pk

    body = struct.pack("<q", T) + struct.pack("<7Q", base_amt, 0, 0, 0, base_res, quote_res, quote_amt)
    body += struct.pack("<6Q", 20, 0, 5, 0, 0, 0) + b"".join(_pk(a) for a in accts)
    return base64.b64encode((SWAP_BUY_DISC if buy else SWAP_SELL_DISC) + body + bytes(64)).decode()


def test_pumpswap_events_decode_in_either_pool_orientation() -> None:
    from nardis_neural.solana.ingest.pumpfun import PUMP_SWAP_PROGRAM

    pool, ub, uq = pubkey_from_seed("pool"), pubkey_from_seed("ub"), pubkey_from_seed("uq")
    ix = {"programId": PUMP_SWAP_PROGRAM, "accounts": [], "data": ""}

    def tx(log: str, base_mint: str, quote_mint: str, base_dec: int, quote_dec: int) -> dict[str, Any]:
        rows = [
            {
                "accountIndex": 1,
                "mint": base_mint,
                "owner": USER,
                "uiTokenAmount": {"amount": "0", "decimals": base_dec},
            },
            {
                "accountIndex": 2,
                "mint": quote_mint,
                "owner": USER,
                "uiTokenAmount": {"amount": "0", "decimals": quote_dec},
            },
        ]
        logs = [
            f"Program {PUMP_SWAP_PROGRAM} invoke [1]",
            f"Program data: {log}",
            f"Program {PUMP_SWAP_PROGRAM} success",
        ]
        return _tx(logs, [ix], pre=rows, post=rows, keys=[ub, uq])

    # token is base, WSOL quote: a BuyEvent buys the token
    buy = tx(
        _swap_event(True, 2_000_000, 10**9, 200_000_000 * 10**6, 100 * 10**9, [pool, USER, ub, uq]),
        MINT,
        WSOL_MINT,
        6,
        9,
    )
    # WSOL is base, token quote: a BuyEvent buys WSOL, i.e. sells the token
    sell = tx(
        _swap_event(True, 10**9, 2_000_000, 100 * 10**9, 200_000_000 * 10**6, [pool, USER, ub, uq]),
        WSOL_MINT,
        MINT,
        9,
        6,
    )
    for raw, is_buy in ((buy, True), (sell, False)):
        dec = TransactionDecoder()
        swaps = [e for e in dec.decode(raw) if isinstance(e, Swap)]
        assert len(swaps) == 1
        s = swaps[0]
        assert s.mint == MINT and s.is_buy is is_buy
        assert s.sol_amount == pytest.approx(1.0) and s.token_amount == pytest.approx(2.0)
        assert s.sol_reserve == pytest.approx(100.0) and s.token_reserve == pytest.approx(2e8)
