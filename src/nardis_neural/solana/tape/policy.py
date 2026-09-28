"""Entry + exit policies evaluated on executable outcomes.

The tail model and the Tape Transformer both give an entry view; only the tape gives an
exit view (collapse hazard).  This module turns them into concrete, testable policies:

* **entry** — one ticket per token at the first entry-window snapshot where the expected
  ladder payoff is at least the ticket (tail model, tape model or their blend);
* **exit** — the ladder as usual, plus an *alarm*: at the first monitored snapshot after
  entry where P(value halves within the window) ≥ θ, whatever is still held is sold (with
  latency, impact and fees).  θ and the window are chosen on a *tune* period and scored
  once on the untouched test period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.market import SolanaMarket
from nardis_neural.solana.moonshot.labels import MoonshotSpec, moonshot_outcome
from nardis_neural.solana.moonshot.research import ticket_stats

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
THRESHOLDS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def monitor_rows(times: F64, mints: npt.NDArray[np.str_], tokens: set[str], every_seconds: float) -> I64:
    """Snapshot indices of ``tokens`` thinned to at most one per ``every_seconds`` per token."""
    last: dict[str, float] = {}
    keep = []
    for i in np.argsort(times, kind="stable"):
        m = str(mints[i])
        if m in tokens and times[i] - last.get(m, -np.inf) >= every_seconds:
            last[m] = float(times[i])
            keep.append(int(i))
    return np.asarray(keep, dtype=np.int64)


def alarm_times(
    entries: dict[str, float], mon_t: F64, mon_mint: npt.NDArray[np.str_], risk: F64, theta: float
) -> dict[str, float | None]:
    """First monitored time after each entry where ``risk`` ≥ ``theta`` (None: never)."""
    out: dict[str, float | None] = dict.fromkeys(entries)
    order = np.argsort(mon_t, kind="stable")
    for i in order:
        m = str(mon_mint[i])
        if m in out and out[m] is None and mon_t[i] > entries[m] and risk[i] >= theta:
            out[m] = float(mon_t[i])
    return out


@dataclass
class PolicyOutcome:
    """Executable results of one entry/exit policy over a set of tokens."""

    ladder: F64
    peak: F64
    stats: dict[str, float]


def run_policy(
    market: SolanaMarket,
    entries: dict[str, float],
    spec: MoonshotSpec,
    data_end: float,
    exits: dict[str, float | None] | None = None,
) -> PolicyOutcome:
    """Simulate every entry (with optional exit alarms) against the real pool path."""
    ladder, peak = [], []
    for mint, t in entries.items():
        out = moonshot_outcome(
            market.token(mint), t, spec, data_end, exit_at=None if exits is None else exits.get(mint)
        )
        if out is not None:
            ladder.append(out.ladder_multiple)
            peak.append(out.peak_multiple)
    lad, pk = np.asarray(ladder, dtype=np.float64), np.asarray(peak, dtype=np.float64)
    return PolicyOutcome(lad, pk, ticket_stats(lad, pk, spec.size_sol))


def tune_exit(
    market: SolanaMarket,
    entries: dict[str, float],
    spec: MoonshotSpec,
    data_end: float,
    mon_t: F64,
    mon_mint: npt.NDArray[np.str_],
    collapse: F64,
    window_names: list[str],
) -> dict[str, Any]:
    """Grid-search the alarm window and threshold on the tune period (total PnL); keeps
    'no alarm' unless an alarm strictly improves on it."""
    base = run_policy(market, entries, spec, data_end)
    best: dict[str, Any] = {
        "window": None,
        "theta": None,
        "total_pnl_sol": base.stats.get("total_pnl_sol", 0.0),
    }
    grid = []
    for j, name in enumerate(window_names):
        for theta in THRESHOLDS:
            exits = alarm_times(entries, mon_t, mon_mint, collapse[:, j], theta)
            res = run_policy(market, entries, spec, data_end, exits)
            pnl = res.stats.get("total_pnl_sol", 0.0)
            grid.append({"window": name, "theta": theta, "total_pnl_sol": pnl})
            if pnl > best["total_pnl_sol"] + 1e-9:
                best = {"window": name, "window_index": j, "theta": theta, "total_pnl_sol": pnl}
    return {"chosen": best, "no_alarm_pnl_sol": base.stats.get("total_pnl_sol", 0.0), "grid": grid}
