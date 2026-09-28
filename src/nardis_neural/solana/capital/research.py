"""Capital research: turn a research test period into bankroll and overfitting evidence.

Given the test-period tickets of a policy (entry signal, exit time, executable multiple),
this compares a **flat stake** with the **capital allocator** on the same starting equity,
adds a bootstrap risk profile for each, and reports the Deflated Sharpe Ratio of the
policy and the Probability of Backtest Overfitting across the configurations that were
compared (e.g. entry thresholds).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.capital.allocator import AllocatorConfig, CapitalAllocator, Signal
from nardis_neural.solana.capital.bankroll import TicketRecord, bootstrap_book, simulate_book
from nardis_neural.solana.capital.overfit import deflated_sharpe, probability_of_backtest_overfitting, sharpe

F64 = npt.NDArray[np.float64]


def ticket_records(
    t_entry: F64,
    mints: npt.NDArray[np.str_],
    t_exit: F64,
    multiple: F64,
    expected: F64,
    kelly: F64,
    epistemic: F64,
    liquidity: F64,
    family: list[str],
    trust: F64 | None = None,
) -> list[TicketRecord]:
    """Pack per-ticket arrays into :class:`TicketRecord` objects."""
    tr = np.ones(len(t_entry)) if trust is None else trust
    return [
        TicketRecord(
            Signal(
                str(mints[i]),
                float(t_entry[i]),
                float(expected[i]),
                float(kelly[i]),
                float(tr[i]),
                float(epistemic[i]),
                float(liquidity[i]),
                family[i],
            ),
            float(max(t_exit[i], t_entry[i])),
            float(multiple[i]),
        )
        for i in range(len(t_entry))
    ]


def capital_report(
    tickets: list[TicketRecord],
    config_returns: dict[str, F64],
    chosen: str,
    flat_stake: float,
    initial_equity: float = 100.0,
    allocator: AllocatorConfig | None = None,
    draws: int = 300,
    pbo_blocks: int = 8,
    haircuts: tuple[float, ...] = (0.6, 0.35),
    aggressive_fraction: float = 0.05,
) -> dict[str, Any]:
    """Flat vs allocator bankroll, bootstrap risk, DSR of ``chosen`` and PBO across configs.

    ``config_returns`` maps each compared configuration to its per-token return in time order
    (0 where that configuration did not trade the token); ``chosen`` names the one deployed.
    """
    cfg = allocator or AllocatorConfig()
    flat = simulate_book(tickets, None, initial_equity, flat_stake)
    sized = simulate_book(tickets, CapitalAllocator(cfg), initial_equity)
    names = list(config_returns)
    matrix = np.stack([config_returns[n] for n in names], axis=1) if names else np.zeros((0, 0))
    chosen_r = config_returns.get(chosen, np.zeros(0))
    traded = chosen_r[chosen_r != 0] if len(chosen_r) else chosen_r
    trial_sharpes = np.asarray([sharpe(config_returns[n][config_returns[n] != 0]) for n in names])
    out: dict[str, Any] = {
        "initial_equity": initial_equity,
        "flat_stake": flat_stake,
        "flat": {k: v for k, v in flat.items() if k != "curve"},
        "allocator": {k: v for k, v in sized.items() if k != "curve"},
        "allocator_config": cfg.model_dump(),
        "flat_bootstrap": bootstrap_book(tickets, None, initial_equity, flat_stake, draws=draws),
        "allocator_bootstrap": bootstrap_book(
            tickets, lambda: CapitalAllocator(cfg), initial_equity, draws=draws
        ),
        "deflated_sharpe": deflated_sharpe(traded, max(len(names), 1), trial_sharpes),
        "pbo": probability_of_backtest_overfitting(matrix, pbo_blocks)
        if matrix.size
        else {"pbo": float("nan")},
        "trials": len(names),
    }
    # stress: the live edge is usually far smaller than the backtest's; scale every payoff
    # down and compare how each sizing survives
    aggressive = aggressive_fraction * initial_equity
    stress = []
    for h in haircuts:
        cut = [TicketRecord(tk.signal, tk.t_exit, tk.multiple * h) for tk in tickets]
        stress.append(
            {
                "haircut": h,
                "mean_multiple": float(np.mean([tk.multiple for tk in cut])) if cut else float("nan"),
                "flat": bootstrap_book(cut, None, initial_equity, flat_stake, draws=draws),
                "flat_aggressive": bootstrap_book(cut, None, initial_equity, aggressive, draws=draws),
                "allocator": bootstrap_book(cut, lambda: CapitalAllocator(cfg), initial_equity, draws=draws),
            }
        )
    out["stress"] = stress
    out["aggressive_flat_stake"] = aggressive
    out["tickets"] = [
        {
            "mint": tk.signal.mint,
            "t_entry": tk.signal.t,
            "t_exit": tk.t_exit,
            "multiple": tk.multiple,
            "expected_multiple": tk.signal.expected_multiple,
            "lottery_kelly": tk.signal.lottery_kelly,
            "epistemic": tk.signal.epistemic,
            "liquidity_sol": tk.signal.liquidity_sol,
            "family": tk.signal.family,
        }
        for tk in tickets
    ]
    return out
