"""Decode Solana transactions (``getTransaction`` JSON, ``jsonParsed`` encoding) into the
market events consumed by :class:`~nardis_neural.solana.market.SolanaMarket`.

What is extracted
-----------------
* **pump.fun** — ``CreateEvent`` → :class:`TokenLaunch`, ``TradeEvent`` → :class:`Swap`
  (virtual reserves after the trade), ``CompleteEvent`` → :class:`Migration`.
* **AMM pools** (PumpSwap, Raydium AMM/CPMM, Meteora, Orca) — venue-agnostic: a pool is
  recognised as a pair of token accounts (one WSOL, one other mint) owned by the same
  non-signer authority inside a transaction that invokes a known AMM program.  Vault
  balance deltas give direction and size: SOL in + tokens out = buy, the reverse = sell,
  both in = liquidity add, both out = liquidity removal.  Post-transaction vault balances
  are the reserves.  For concentrated-liquidity venues (Orca, Meteora) the reserves are
  re-expressed so that ``sol / tokens`` equals the swap's execution price.
* **SOL transfers** above ``min_transfer_sol`` (funding graph / sybil detection).
* **Execution competition** — Jito tips (transfers to the tip accounts) and priority fees
  (``fee − 5000 lamports × signatures``), attached to the transaction's first swap.

Transactions must be fed in slot order.  Timestamps are ``blockTime`` (1 s resolution)
plus a microsecond sequence so events stay strictly ordered.  Failed transactions are
ignored.  Tokens first seen mid-stream get an implicit :class:`TokenLaunch`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from nardis_neural.solana.amm import PUMP_VIRTUAL_SOL, TOTAL_SUPPLY
from nardis_neural.solana.events import Event, LiquidityChange, Migration, Swap, TokenLaunch, Transfer, Venue
from nardis_neural.solana.ingest.pumpfun import (
    AMM_PROGRAMS,
    JITO_TIP_ACCOUNTS,
    LAMPORTS_PER_SOL,
    PUMP_FUN_PROGRAM,
    PUMP_TOKEN_DECIMALS,
    SYSTEM_PROGRAM,
    WSOL_MINT,
    PumpComplete,
    PumpCreate,
    PumpTrade,
    decode_log_events,
)

MintInfo = Callable[[str], tuple[bool, bool] | None]
"""mint → (mint_authority_revoked, freeze_authority_revoked), or None if unknown."""

CLMM_VENUES = frozenset({"orca", "meteora"})
BASE_FEE_LAMPORTS = 5000


@dataclass
class PoolInfo:
    """An AMM pool recognised on chain: mint, authority, token and WSOL vaults, venue."""

    mint: str
    authority: str
    token_vault: str
    sol_vault: str
    venue: Venue


@dataclass
class _Balance:
    mint: str
    owner: str
    amount: int
    decimals: int


@dataclass
class DecodeStats:
    """Counts of decoded transactions, failed ones and emitted events by type."""

    transactions: int = 0
    failed: int = 0
    events: dict[str, int] = field(default_factory=dict)

    def count(self, e: Event) -> None:
        """Count one emitted event under its type name."""
        name = type(e).__name__
        self.events[name] = self.events.get(name, 0) + 1


def _keys(tx: Mapping[str, Any]) -> list[str]:
    keys = tx["transaction"]["message"]["accountKeys"]
    out = [k["pubkey"] if isinstance(k, dict) else str(k) for k in keys]
    loaded = tx.get("meta", {}).get("loadedAddresses") or {}
    return out + list(loaded.get("writable", [])) + list(loaded.get("readonly", []))


def _instructions(tx: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outer = list(tx["transaction"]["message"].get("instructions", []))
    inner = [
        ix
        for grp in (tx.get("meta", {}).get("innerInstructions") or [])
        for ix in grp.get("instructions", [])
    ]
    return outer + inner


def _token_balances(rows: Iterable[Mapping[str, Any]], keys: list[str]) -> dict[str, _Balance]:
    out = {}
    for r in rows or []:
        acct = keys[int(r["accountIndex"])]
        ui = r.get("uiTokenAmount", {})
        out[acct] = _Balance(
            str(r["mint"]), str(r.get("owner", "")), int(ui.get("amount", "0")), int(ui.get("decimals", 0))
        )
    return out


class TransactionDecoder:
    """Stateful decoder of ``getTransaction`` JSON into market events; feed transactions in slot order.

    Keeps per-token venue, pool and virtual-reserve state.  ``mint_info`` looks up authority
    revocations of AMM-launched tokens; ``implicit_launches`` emits a :class:`TokenLaunch` for
    tokens first seen mid-stream.
    """

    def __init__(
        self,
        min_transfer_sol: float = 0.05,
        jito_accounts: frozenset[str] = JITO_TIP_ACCOUNTS,
        mint_info: MintInfo | None = None,
        implicit_launches: bool = True,
    ) -> None:
        self.min_transfer_sol = min_transfer_sol
        self.jito_accounts = jito_accounts
        self.mint_info = mint_info
        self.implicit_launches = implicit_launches
        self.venue: dict[str, Venue] = {}
        self.pools: dict[str, PoolInfo] = {}
        self.virtual: dict[str, tuple[float, float]] = {}
        self.non_sol_quoted: set[str] = set()
        """Mints traded through curves that report no SOL reserves (skipped)."""
        self.stats = DecodeStats()
        self._last_t = 0.0

    def forget(self, mints: Iterable[str]) -> None:
        """Drop per-token decoding state (bounded memory for long streams)."""
        for m in mints:
            self.venue.pop(m, None)
            self.pools.pop(m, None)
            self.virtual.pop(m, None)

    # ------------------------------------------------------------------ helpers
    def _clock(self, block_time: float) -> float:
        self._last_t = max(float(block_time), self._last_t + 1e-6)
        return self._last_t

    def _authorities(self, mint: str) -> tuple[bool, bool]:
        info = self.mint_info(mint) if self.mint_info is not None else None
        return info if info is not None else (True, True)

    def _launch(
        self,
        out: list[Event],
        mint: str,
        t: float,
        creator: str,
        venue: Venue,
        sol: float,
        tok: float,
        lp_burned: float,
        slot: int,
    ) -> None:
        if mint in self.venue:
            return
        mint_ok, freeze_ok = self._authorities(mint) if venue != "pump_fun" else (True, True)
        self.venue[mint] = venue
        out.append(
            TokenLaunch(
                mint=mint,
                t=t,
                creator=creator,
                venue=venue,
                supply=TOTAL_SUPPLY,
                sol_reserve=sol,
                token_reserve=tok,
                mint_authority_revoked=mint_ok,
                freeze_authority_revoked=freeze_ok,
                lp_burned_fraction=lp_burned,
                slot=slot,
            )
        )

    # ------------------------------------------------------------------ main entry
    def decode(self, tx: Mapping[str, Any]) -> list[Event]:
        """Decode one transaction into events (none if it failed) and update :attr:`stats`.

        The transaction's priority fee and Jito tip (in SOL) are attached to its first swap.
        """
        self.stats.transactions += 1
        meta = tx.get("meta") or {}
        if meta.get("err") is not None:
            self.stats.failed += 1
            return []
        keys = _keys(tx)
        payer = keys[0]
        slot = int(tx.get("slot", -1))
        t = self._clock(float(tx.get("blockTime") or 0.0))
        n_sigs = len(tx["transaction"].get("signatures", [])) or 1
        priority_fee = max(int(meta.get("fee", 0)) - BASE_FEE_LAMPORTS * n_sigs, 0) / LAMPORTS_PER_SOL
        instructions = _instructions(tx)
        programs = {str(ix.get("programId", "")) for ix in instructions}
        jito_tip = 0.0
        transfers: list[tuple[str, str, float]] = []
        for ix in instructions:
            parsed = ix.get("parsed")
            if str(ix.get("programId")) != SYSTEM_PROGRAM or not isinstance(parsed, dict):
                continue
            if parsed.get("type") not in ("transfer", "transferWithSeed"):
                continue
            info = parsed.get("info", {})
            sol = int(info.get("lamports", 0)) / LAMPORTS_PER_SOL
            dest = str(info.get("destination", ""))
            if dest in self.jito_accounts:
                jito_tip += sol
            else:
                transfers.append((str(info.get("source", "")), dest, sol))

        events: list[Event] = []
        swaps: list[Event] = []
        pool_vaults = {v for p in self.pools.values() for v in (p.sol_vault, p.token_vault)}
        for src, dst, sol in transfers:
            if sol >= self.min_transfer_sol and dst not in pool_vaults:
                events.append(Transfer(t=t, source=src, dest=dst, sol_amount=sol, slot=slot))

        if PUMP_FUN_PROGRAM in programs:
            for ev in decode_log_events(list(meta.get("logMessages") or [])):
                self._pump(ev, events, swaps, t, slot)
        amm = [AMM_PROGRAMS[p] for p in programs if p in AMM_PROGRAMS]
        if amm:
            self._amm(tx, keys, payer, amm[0], events, swaps, t, slot)  # type: ignore[arg-type]

        if swaps and (priority_fee or jito_tip):
            first = swaps[0]
            assert isinstance(first, Swap)
            idx = events.index(first)
            events[idx] = replace(first, priority_fee=priority_fee, jito_tip=jito_tip)
        for e in events:
            self.stats.count(e)
        return events

    # ------------------------------------------------------------------ pump.fun
    def _pump(self, ev: object, out: list[Event], swaps: list[Event], t: float, slot: int) -> None:
        if isinstance(ev, PumpCreate):
            self._launch(out, ev.mint, t, ev.user, "pump_fun", PUMP_VIRTUAL_SOL, 1_073_000_000.0, 1.0, slot)
        elif isinstance(ev, PumpTrade):
            if ev.virtual_sol_lamports == 0:
                # BuyV2 / SellV2 curves report no SOL amounts or reserves (not SOL-quoted):
                # outside what the SOL-denominated features and labels can describe
                self.non_sol_quoted.add(ev.mint)
                return
            if ev.mint in self.non_sol_quoted:
                return
            if ev.mint not in self.venue:
                if not self.implicit_launches:
                    return
                self._launch(
                    out,
                    ev.mint,
                    t,
                    "unknown",
                    "pump_fun",
                    ev.virtual_sol_lamports / LAMPORTS_PER_SOL,
                    ev.virtual_token_units / 10**PUMP_TOKEN_DECIMALS,
                    1.0,
                    slot,
                )
            sol_res = ev.virtual_sol_lamports / LAMPORTS_PER_SOL
            tok_res = ev.virtual_token_units / 10**PUMP_TOKEN_DECIMALS
            self.virtual[ev.mint] = (sol_res, tok_res)
            s = Swap(
                mint=ev.mint,
                t=t,
                wallet=ev.user,
                is_buy=ev.is_buy,
                sol_amount=ev.sol_lamports / LAMPORTS_PER_SOL,
                token_amount=ev.token_units / 10**PUMP_TOKEN_DECIMALS,
                sol_reserve=sol_res,
                token_reserve=tok_res,
                slot=slot,
            )
            out.append(s)
            swaps.append(s)
        elif isinstance(ev, PumpComplete) and ev.mint in self.venue and self.venue[ev.mint] == "pump_fun":
            sol_res, tok_res = self.virtual.get(ev.mint, (PUMP_VIRTUAL_SOL + 85.0, 2.8e8))
            real_sol = max(sol_res - PUMP_VIRTUAL_SOL, 1e-9)
            self.venue[ev.mint] = "pumpswap"
            out.append(
                Migration(
                    mint=ev.mint,
                    t=t,
                    venue="pumpswap",
                    sol_reserve=real_sol,
                    token_reserve=real_sol * tok_res / sol_res,
                    slot=slot,
                )
            )

    # ------------------------------------------------------------------ AMM pools
    def _amm(
        self,
        tx: Mapping[str, Any],
        keys: list[str],
        payer: str,
        venue: Venue,
        out: list[Event],
        swaps: list[Event],
        t: float,
        slot: int,
    ) -> None:
        meta = tx["meta"]
        pre = _token_balances(meta.get("preTokenBalances"), keys)
        post = _token_balances(meta.get("postTokenBalances"), keys)
        accounts = set(pre) | set(post)
        by_owner: dict[str, dict[str, str]] = {}
        for acct in accounts:
            b = post.get(acct) or pre[acct]
            if b.owner and b.owner != payer:
                by_owner.setdefault(b.owner, {})[b.mint] = acct
        for owner, mints in by_owner.items():
            if WSOL_MINT not in mints:
                continue
            for mint, token_acct in mints.items():
                if mint == WSOL_MINT:
                    continue
                pool = self.pools.get(mint)
                if pool is None or {pool.token_vault, pool.sol_vault} != {token_acct, mints[WSOL_MINT]}:
                    pool = PoolInfo(mint, owner, token_acct, mints[WSOL_MINT], venue)
                    self.pools[mint] = pool
                self._pool_event(pool, pre, post, payer, out, swaps, t, slot)

    def _pool_event(
        self,
        pool: PoolInfo,
        pre: dict[str, _Balance],
        post: dict[str, _Balance],
        payer: str,
        out: list[Event],
        swaps: list[Event],
        t: float,
        slot: int,
    ) -> None:
        def amt(book: dict[str, _Balance], acct: str) -> float:
            b = book.get(acct)  # absent before → account created in this transaction
            return b.amount / 10**b.decimals if b is not None else 0.0

        s0, s1 = amt(pre, pool.sol_vault), amt(post, pool.sol_vault)
        k0, k1 = amt(pre, pool.token_vault), amt(post, pool.token_vault)
        ds, dk = s1 - s0, k1 - k0
        if abs(ds) < 1e-12 and abs(dk) < 1e-12:
            return
        mint = pool.mint
        if mint not in self.venue:
            if not self.implicit_launches:
                return
            self._launch(out, mint, t, payer, pool.venue, max(s0, 1e-9), max(k0, 1e-9), 0.0, slot)
        elif self.venue[mint] == "pump_fun":  # graduation seen through the new pool first
            self.venue[mint] = pool.venue
            out.append(
                Migration(
                    mint=mint,
                    t=t,
                    venue=pool.venue,
                    sol_reserve=max(s0, 1e-9),
                    token_reserve=max(k0, 1e-9),
                    slot=slot,
                )
            )
        sol_res, tok_res = max(s1, 1e-12), max(k1, 1e-12)
        if ds != 0 and dk != 0 and (ds > 0) != (dk > 0):  # opposite, non-zero changes → swap
            is_buy = ds > 0
            sol_amt, tok_amt = abs(ds), abs(dk)
            if pool.venue in CLMM_VENUES and tok_amt > 0:
                tok_res = sol_res / (sol_amt / tok_amt)
            s = Swap(
                mint=mint,
                t=t,
                wallet=payer,
                is_buy=is_buy,
                sol_amount=sol_amt,
                token_amount=tok_amt,
                sol_reserve=sol_res,
                token_reserve=tok_res,
                slot=slot,
            )
            out.append(s)
            swaps.append(s)
        else:
            out.append(
                LiquidityChange(
                    mint=mint,
                    t=t,
                    wallet=payer,
                    sol_delta=ds,
                    token_delta=dk,
                    sol_reserve=sol_res,
                    token_reserve=tok_res,
                    slot=slot,
                )
            )


def decode_transactions(
    txs: Iterable[Mapping[str, Any]], decoder: TransactionDecoder | None = None
) -> list[Event]:
    """Decode an iterable of transactions (sorted by slot) into a flat event list."""
    dec = decoder or TransactionDecoder()
    ordered = sorted(txs, key=lambda x: (int(x.get("slot", 0)), float(x.get("blockTime") or 0)))
    return [e for tx in ordered for e in dec.decode(tx)]
