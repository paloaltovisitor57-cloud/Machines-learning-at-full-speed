"""Render market events as ``getTransaction``-style JSON (jsonParsed encoding).

The inverse of :mod:`decoder` — used to test the decoder end to end (simulated market →
realistic transactions → decoded events) and to build replay fixtures.  Wallet / mint
names that are not valid base58 public keys are mapped to deterministic fake keys.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

from nardis_neural.solana.events import Event, LiquidityChange, Migration, Swap, TokenLaunch, Transfer
from nardis_neural.solana.ingest.base58 import b58decode, b58encode
from nardis_neural.solana.ingest.pumpfun import (
    JITO_TIP_ACCOUNTS,
    LAMPORTS_PER_SOL,
    PUMP_FUN_PROGRAM,
    PUMP_SWAP_PROGRAM,
    PUMP_TOKEN_DECIMALS,
    RAYDIUM_AMM_V4_PROGRAM,
    SYSTEM_PROGRAM,
    WSOL_MINT,
    PumpComplete,
    PumpCreate,
    PumpTrade,
    encode_event,
)
from nardis_neural.solana.market import event_sort_key

JITO = sorted(JITO_TIP_ACCOUNTS)[0]
NEWER_FIELDS = bytes(64)
"""Zero padding standing in for fields newer program versions append to TradeEvent."""


def as_pubkey(name: str) -> str:
    """Valid 32-byte base58 keys pass through; any other name maps to a stable fake key."""
    try:
        if len(b58decode(name)) == 32:
            return name
    except ValueError:
        pass
    return b58encode(hashlib.sha256(name.encode()).digest())


def _sig(i: int) -> str:
    return b58encode(hashlib.sha256(f"sig{i}".encode()).digest() * 2)


def _transfer_ix(src: str, dst: str, sol: float) -> dict[str, Any]:
    return {
        "programId": SYSTEM_PROGRAM,
        "program": "system",
        "parsed": {
            "type": "transfer",
            "info": {"source": src, "destination": dst, "lamports": round(sol * LAMPORTS_PER_SOL)},
        },
    }


class TransactionEncoder:
    def __init__(self) -> None:
        self.n = 0
        self.venue: dict[str, str] = {}
        self.vaults: dict[str, tuple[int, int]] = {}  # mint → (sol lamports, token units) in the AMM pool

    def _tx(
        self,
        e: Event,
        payer: str,
        instructions: list[dict[str, Any]],
        logs: list[str] | None = None,
        fee_sol: float = 0.0,
        pre_tb: list[dict[str, Any]] | None = None,
        post_tb: list[dict[str, Any]] | None = None,
        extra_keys: list[str] | None = None,
    ) -> dict[str, Any]:
        self.n += 1
        return {
            "slot": e.slot if e.slot >= 0 else int(e.t / 0.4),
            "blockTime": int(e.t),
            "meta": {
                "err": None,
                "fee": 5000 + round(fee_sol * LAMPORTS_PER_SOL),
                "logMessages": logs or [],
                "preTokenBalances": pre_tb or [],
                "postTokenBalances": post_tb or [],
                "innerInstructions": [],
            },
            "transaction": {
                "signatures": [_sig(self.n)],
                "message": {
                    "accountKeys": [{"pubkey": payer, "signer": True, "writable": True}]
                    + [{"pubkey": k, "signer": False, "writable": True} for k in extra_keys or []],
                    "instructions": instructions,
                },
            },
        }

    def _pool_tx(
        self, e: Swap | LiquidityChange, payer: str, sol_after: float, tok_after: float, program: str
    ) -> dict[str, Any]:
        mint = as_pubkey(e.mint)
        authority = as_pubkey(f"pool-authority:{e.mint}")
        sol_vault, tok_vault = as_pubkey(f"sol-vault:{e.mint}"), as_pubkey(f"token-vault:{e.mint}")
        pre_sol, pre_tok = self.vaults.get(e.mint, (0, 0))
        post_sol, post_tok = round(sol_after * LAMPORTS_PER_SOL), round(tok_after * 10**PUMP_TOKEN_DECIMALS)
        self.vaults[e.mint] = (post_sol, post_tok)

        def bal(idx: int, m: str, amount: int, dec: int) -> dict[str, Any]:
            return {
                "accountIndex": idx,
                "mint": m,
                "owner": authority,
                "uiTokenAmount": {"amount": str(amount), "decimals": dec},
            }

        pre = [bal(1, WSOL_MINT, pre_sol, 9), bal(2, mint, pre_tok, PUMP_TOKEN_DECIMALS)]
        post = [bal(1, WSOL_MINT, post_sol, 9), bal(2, mint, post_tok, PUMP_TOKEN_DECIMALS)]
        ixs: list[dict[str, Any]] = [{"programId": program, "accounts": [], "data": ""}]
        fee = e.priority_fee if isinstance(e, Swap) else 0.0
        if isinstance(e, Swap) and e.jito_tip > 0:
            ixs.append(_transfer_ix(payer, JITO, e.jito_tip))
        return self._tx(
            e, payer, ixs, fee_sol=fee, pre_tb=pre, post_tb=post, extra_keys=[sol_vault, tok_vault]
        )

    def encode(self, e: Event) -> dict[str, Any] | None:
        if isinstance(e, Transfer):
            src, dst = as_pubkey(e.source), as_pubkey(e.dest)
            return self._tx(e, src, [_transfer_ix(src, dst, e.sol_amount)])
        if isinstance(e, TokenLaunch):
            self.venue[e.mint] = e.venue
            if e.venue != "pump_fun":
                return None  # AMM launches appear through their first liquidity transaction
            creator, mint = as_pubkey(e.creator), as_pubkey(e.mint)
            payload = encode_event(
                PumpCreate(e.mint[:32], e.mint[:10], "", mint, as_pubkey(f"curve:{e.mint}"), creator)
            )
            return self._tx(
                e,
                creator,
                [{"programId": PUMP_FUN_PROGRAM, "accounts": [], "data": ""}],
                logs=[f"Program {PUMP_FUN_PROGRAM} invoke [1]", f"Program data: {payload}"],
            )
        if isinstance(e, Migration):
            self.venue[e.mint] = e.venue
            self.vaults[e.mint] = (
                round(e.sol_reserve * LAMPORTS_PER_SOL),
                round(e.token_reserve * 10**PUMP_TOKEN_DECIMALS),
            )
            payload = encode_event(
                PumpComplete(as_pubkey("migrator"), as_pubkey(e.mint), as_pubkey(f"curve:{e.mint}"), int(e.t))
            )
            return self._tx(
                e,
                as_pubkey("migrator"),
                [{"programId": PUMP_FUN_PROGRAM, "accounts": [], "data": ""}],
                logs=[f"Program data: {payload}"],
            )
        wallet = as_pubkey(e.wallet)
        if isinstance(e, LiquidityChange):
            return self._pool_tx(e, wallet, e.sol_reserve, e.token_reserve, RAYDIUM_AMM_V4_PROGRAM)
        if self.venue.get(e.mint) == "pump_fun":
            trade = PumpTrade(
                as_pubkey(e.mint),
                round(e.sol_amount * LAMPORTS_PER_SOL),
                round(e.token_amount * 10**PUMP_TOKEN_DECIMALS),
                e.is_buy,
                wallet,
                int(e.t),
                round(e.sol_reserve * LAMPORTS_PER_SOL),
                round(e.token_reserve * 10**PUMP_TOKEN_DECIMALS),
            )
            ixs: list[dict[str, Any]] = [{"programId": PUMP_FUN_PROGRAM, "accounts": [], "data": ""}]
            if e.jito_tip > 0:
                ixs.append(_transfer_ix(wallet, JITO, e.jito_tip))
            return self._tx(
                e,
                wallet,
                ixs,
                logs=[f"Program data: {encode_event(trade, trailing=NEWER_FIELDS)}"],
                fee_sol=e.priority_fee,
            )
        program = PUMP_SWAP_PROGRAM if self.venue.get(e.mint) == "pumpswap" else RAYDIUM_AMM_V4_PROGRAM
        return self._pool_tx(e, wallet, e.sol_reserve, e.token_reserve, program)


def events_to_transactions(events: Iterable[Event]) -> list[dict[str, Any]]:
    enc = TransactionEncoder()
    out = []
    for e in sorted(events, key=event_sort_key):
        tx = enc.encode(e)
        if tx is not None:
            out.append(tx)
    return out
