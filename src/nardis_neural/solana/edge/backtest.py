"""Research backtester for executable, cost-aware signals (paper only — no execution).

Every candidate row carries its *executable* triple-barrier outcome (latency, impact and
fees already inside).  A policy selects rows by score; the simulator enforces realistic
constraints — one open position per token, a maximum number of concurrent positions,
positions occupy their slot until the barrier exit — and reports per-trade statistics with
bootstrap confidence intervals.  Baselines use the *same* constraints and trade count, so
"edge" means beating them, not just being positive.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass
class Candidates:
    """Candidate trades: signal time, mint, executable net return, exit time and ticket size in SOL."""

    t: F64
    mint: npt.NDArray[np.str_]
    net: F64
    exit_time: F64
    size_sol: float = 1.0

    def __len__(self) -> int:
        return len(self.t)


def simulate(c: Candidates, selected: npt.NDArray[np.bool_], max_positions: int = 5) -> I64:
    """Indices of rows that become trades, in time order, under position constraints."""
    order = np.argsort(c.t, kind="stable")
    open_until: dict[str, float] = {}
    trades: list[int] = []
    for i in order:
        if not selected[i]:
            continue
        now = c.t[i]
        open_until = {m: e for m, e in open_until.items() if e > now}
        if c.mint[i] in open_until or len(open_until) >= max_positions:
            continue
        open_until[str(c.mint[i])] = c.exit_time[i]
        trades.append(int(i))
    return np.asarray(trades, dtype=np.int64)


def trade_stats(c: Candidates, trades: I64, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Per-trade statistics of ``trades``: hit rate, net-return mean / median / t-stat, bootstrap
    95 % CI of the mean, profit factor, and total PnL / max drawdown in SOL (by exit time).
    """
    r = c.net[trades]
    n = len(r)
    if n == 0:
        return {"trades": 0.0}
    rng = np.random.default_rng(seed)
    boots = rng.choice(r, size=(n_boot, n), replace=True).mean(axis=1) if n > 1 else np.array([r.mean()])
    by_exit = np.argsort(c.exit_time[trades], kind="stable")
    pnl = np.cumsum(r[by_exit] * c.size_sol)
    peak = np.maximum.accumulate(np.concatenate([[0.0], pnl]))[1:]
    gains, losses = r[r > 0].sum(), -r[r <= 0].sum()
    sd = float(r.std(ddof=1)) if n > 1 else 0.0
    return {
        "trades": float(n),
        "hit_rate": float((r > 0).mean()),
        "mean_net": float(r.mean()),
        "median_net": float(np.median(r)),
        "std_net": sd,
        "per_trade_sharpe": float(r.mean() / sd) if sd > 0 else 0.0,
        "t_stat": float(r.mean() / (sd / np.sqrt(n))) if sd > 0 else 0.0,
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "p_mean_le_0": float((boots <= 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf"),
        "total_pnl_sol": float(pnl[-1]),
        "max_drawdown_sol": float((peak - pnl).max()),
    }


def select_threshold(
    c: Candidates, scores: F64, max_positions: int = 5, min_trades: int = 20, quantiles: int = 40
) -> tuple[float, list[dict[str, float]]]:
    """Pick the score threshold maximising the per-trade t-statistic (≥ ``min_trades``)."""
    grid = np.unique(np.quantile(scores, np.linspace(0.0, 0.99, quantiles)))
    curve = []
    best_t, best = float(grid[0]), -np.inf
    for thr in grid:
        trades = simulate(c, scores >= thr, max_positions)
        st = trade_stats(c, trades, n_boot=200)
        st["threshold"] = float(thr)
        curve.append(st)
        if st["trades"] >= min_trades and st.get("t_stat", -np.inf) > best:
            best, best_t = st["t_stat"], float(thr)
    return best_t, curve


def baselines(
    c: Candidates,
    n_trades: int,
    momentum: F64 | None = None,
    max_positions: int = 5,
    draws: int = 30,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Same constraints, same trade budget: random picks, momentum picks, and take-everything."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, Any]] = {}
    everything = simulate(c, np.ones(len(c), dtype=bool), max_positions)
    out["all_candidates"] = trade_stats(c, everything)
    means = []
    for _ in range(draws):
        pick = np.zeros(len(c), dtype=bool)
        pick[rng.permutation(len(c))[: max(n_trades * 3, 1)]] = True
        tr = simulate(c, pick, max_positions)[:n_trades]
        if len(tr):
            means.append(float(c.net[tr].mean()))
    out["random"] = {
        "mean_net": float(np.mean(means)) if means else float("nan"),
        "mean_net_p95": float(np.quantile(means, 0.95)) if means else float("nan"),
    }
    if momentum is not None:
        thr = np.quantile(momentum, max(0.0, 1 - 3 * n_trades / max(len(c), 1)))
        tr = simulate(c, momentum >= thr, max_positions)[:n_trades]
        out["momentum"] = trade_stats(c, tr)
    return out
