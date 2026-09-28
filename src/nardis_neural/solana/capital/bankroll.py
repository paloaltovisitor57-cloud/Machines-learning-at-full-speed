"""Event-driven bankroll simulation of a stream of tickets, and its bootstrap risk profile.

Capital is locked from entry until exit; a ticket returns ``stake × multiple`` when it
closes.  Open positions are carried at cost (no mark-to-market optimism).  The simulation
answers the questions a per-ticket average cannot: how fast does the bankroll compound,
how deep are the drawdowns, how long is it underwater, and how often does the sequence of
outcomes ruin the account.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.capital.allocator import BookState, CapitalAllocator, Signal

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class TicketRecord:
    """One historical ticket: the signal at entry and the executable result."""

    signal: Signal
    t_exit: float
    multiple: float
    """SOL returned per SOL staked (ladder, latency, impact and fees included)."""


def simulate_book(
    tickets: list[TicketRecord],
    allocator: CapitalAllocator | None,
    initial_equity: float = 100.0,
    flat_stake: float | None = None,
) -> dict[str, Any]:
    """Run the book through time.  ``allocator=None`` with ``flat_stake`` stakes a fixed amount."""
    state = BookState(equity=initial_equity, peak_equity=initial_equity)
    cash = initial_equity
    exits: list[tuple[float, str, float, float]] = []  # (t_exit, mint, stake, multiple)
    curve = [(tickets[0].signal.t if tickets else 0.0, initial_equity)]
    taken = 0
    staked = 0.0

    def settle(until: float) -> None:
        nonlocal cash
        while exits and exits[0][0] <= until:
            t, mint, stake, mult = heapq.heappop(exits)
            cash += stake * mult
            state.open_stakes.pop(mint, None)
            state.open_family.pop(mint, None)
            state.equity = cash + state.exposure
            state.peak_equity = max(state.peak_equity, state.equity)
            curve.append((t, state.equity))

    for tk in sorted(tickets, key=lambda k: k.signal.t):
        settle(tk.signal.t)
        if allocator is not None:
            stake = allocator.allocate([tk.signal], state)[0].stake_sol
        else:
            stake = min(flat_stake or 0.0, cash) if tk.signal.mint not in state.open_stakes else 0.0
        stake = min(stake, cash)
        if stake <= 0:
            continue
        cash -= stake
        state.open_stakes[tk.signal.mint] = stake
        if tk.signal.family:
            state.open_family[tk.signal.mint] = tk.signal.family
        heapq.heappush(exits, (tk.t_exit, tk.signal.mint, stake, max(tk.multiple, 0.0)))
        taken += 1
        staked += stake
    settle(float("inf"))
    eq = np.asarray([e for _, e in curve], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = 1.0 - eq / np.maximum(peak, 1e-12)
    return {
        "tickets_taken": taken,
        "sol_staked": staked,
        "final_equity": float(eq[-1]),
        "return": float(eq[-1] / initial_equity - 1.0),
        "log_growth": float(np.log(max(eq[-1], 1e-12) / initial_equity)),
        "max_drawdown": float(dd.max()),
        "underwater_share": float((dd > 0.01).mean()),
        "curve": [(float(t), float(e)) for t, e in curve],
    }


def bootstrap_book(
    tickets: list[TicketRecord],
    make_allocator: Any,
    initial_equity: float = 100.0,
    flat_stake: float | None = None,
    draws: int = 300,
    ruin_level: float = 0.5,
    seed: int = 0,
) -> dict[str, float]:
    """Risk profile over alternative histories: outcomes are resampled (with replacement)
    onto the same schedule of signals, so luck in the ordering and in which tickets hit is
    randomised while the signal flow stays realistic."""
    rng = np.random.default_rng(seed)
    mults = np.asarray([t.multiple for t in tickets], dtype=np.float64)
    finals, dds = [], []
    for _ in range(draws):
        pick = rng.integers(0, len(tickets), len(tickets))
        alt = [
            TicketRecord(tk.signal, tk.t_exit, float(mults[j])) for tk, j in zip(tickets, pick, strict=True)
        ]
        res = simulate_book(alt, make_allocator() if make_allocator else None, initial_equity, flat_stake)
        finals.append(res["final_equity"])
        dds.append(res["max_drawdown"])
    f, d = np.asarray(finals), np.asarray(dds)
    return {
        "median_return": float(np.median(f) / initial_equity - 1.0),
        "p05_return": float(np.quantile(f, 0.05) / initial_equity - 1.0),
        "p95_return": float(np.quantile(f, 0.95) / initial_equity - 1.0),
        "prob_loss": float((f < initial_equity).mean()),
        "prob_ruin": float((f < ruin_level * initial_equity).mean()),
        "median_max_drawdown": float(np.median(d)),
        "p95_max_drawdown": float(np.quantile(d, 0.95)),
    }
