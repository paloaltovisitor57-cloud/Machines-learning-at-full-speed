"""Optimal stopping for exits: Longstaff–Schwartz on executable liquidation paths.

Holding a memecoin position is an American option on its own liquidation value: at every
decision time ``t_k`` we may sell (receive ``V_k``, the executable multiple) or continue.
The optimal rule is the Snell envelope

    U_k = max( u(V_k),  E[ U_{k+1} | state_k ] ),       stop at the first k with u(V_k) ≥ C_k,

where ``C_k = E[U_{k+1} | state_k]`` is the **continuation value**.  Longstaff & Schwartz
(2001) estimate ``C`` by regressing realised future values on the current state along
observed paths, backwards in time.  Here:

* **utility** — ``u(V) = log V`` by default, the Kelly-consistent objective: with fat tails,
  maximising E[V] says "always hold for the lottery", while maximising E[log V] is what
  compounds capital (``utility="linear"`` is available for comparison);
* **state** — the token's current features (all of them, causal), time in the position,
  the current log multiple, the running peak and the drawdown from it;
* **fitted policy iteration** — pooled over paths of different lengths: start from "hold to
  the end", regress the realised utility of following the current policy from ``k+1``,
  switch to "stop when u(V_k) ≥ Ĉ(state_k)", recompute, refit; a few rounds converge;
* **regressor** — gradient-boosted trees, exported to plain arrays for persistence;
* **honesty** — the regression is fitted on earlier tokens with paths truncated at the
  cutoff and the policy is scored once on later tokens (in-sample LSM is biased upward).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.edge.trees import TreeEnsemble

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
FLOOR = 1e-3


@dataclass
class StoppingPath:
    """One held position observed at its decision times."""

    mint: str
    t_entry: float
    times: F64
    marks: F64
    """Executable liquidation multiple if the whole position were sold at each decision time."""
    current: F32
    """(K, n_features) causal market state at each decision time."""


def state_matrix(p: StoppingPath) -> F32:
    """Regression state: market features + time held + log multiple, running peak, drawdown."""
    lm = np.log(np.maximum(p.marks, FLOOR))
    peak = np.maximum.accumulate(lm)
    held = np.log1p(np.maximum(p.times - p.t_entry, 0.0))
    extra = np.stack([held, lm, peak, lm - peak], axis=1)
    return np.concatenate([np.asarray(p.current, dtype=np.float64), extra], axis=1).astype(np.float32)


class StoppingModel:
    """Fitted-policy-iteration Longstaff–Schwartz exit model."""

    def __init__(self, utility: str = "log", iterations: int = 4, seed: int = 0) -> None:
        if utility not in ("log", "linear"):
            raise ValueError("utility must be 'log' or 'linear'")
        self.utility, self.iterations, self.seed = utility, iterations, seed
        self.trees: TreeEnsemble | None = None
        self.report: dict[str, Any] = {}

    def u(self, v: F64) -> F64:
        """Utility of a liquidation multiple."""
        v = np.asarray(v, dtype=np.float64)
        return np.log(np.maximum(v, FLOOR)) if self.utility == "log" else v

    def continuation(self, x: F32) -> F64:
        """Estimated E[utility of continuing] for each state row."""
        if self.trees is None:
            raise RuntimeError("model is not fitted")
        return self.trees.predict(np.nan_to_num(np.asarray(x, dtype=np.float64)))

    def stop_index(self, p: StoppingPath) -> int:
        """First decision index at which selling beats continuing (the last index if never)."""
        if len(p.marks) == 0:
            return -1
        if self.trees is None or len(p.marks) == 1:
            return len(p.marks) - 1
        c = self.continuation(state_matrix(p)[:-1])
        hit = np.flatnonzero(self.u(p.marks[:-1]) >= c)
        return int(hit[0]) if len(hit) else len(p.marks) - 1

    def fit(self, paths: list[StoppingPath], max_iter: int = 200) -> dict[str, Any]:
        """Policy iteration: regress the realised utility of the current policy, re-derive it."""
        from sklearn.ensemble import HistGradientBoostingRegressor

        paths = [p for p in paths if len(p.marks) >= 2]
        if not paths:
            raise ValueError("no path has two or more decision times")
        states = [state_matrix(p) for p in paths]
        utils = [self.u(p.marks) for p in paths]
        stops = [np.r_[np.zeros(len(p.marks) - 1, dtype=bool), True] for p in paths]  # hold to the end
        history = []
        for it in range(self.iterations):
            xs, ys = [], []
            for x, uu, s in zip(states, utils, stops, strict=True):
                # realised utility from k+1 onwards under the current policy
                nxt = np.empty(len(uu))
                val = uu[-1]
                for k in range(len(uu) - 1, -1, -1):
                    nxt[k] = val
                    if s[k]:
                        val = uu[k]
                xs.append(x[:-1])
                ys.append(nxt[:-1])
            model = HistGradientBoostingRegressor(
                max_depth=4,
                max_iter=max_iter,
                learning_rate=0.05,
                l2_regularization=1.0,
                random_state=self.seed + it,
            )
            model.fit(np.nan_to_num(np.concatenate(xs).astype(np.float64)), np.concatenate(ys))
            self.trees = TreeEnsemble.export(model)
            changed = 0
            new_stops = []
            for x, uu, s in zip(states, utils, stops, strict=True):
                ns = np.r_[uu[:-1] >= self.continuation(x[:-1]), True]
                changed += int((ns != s).sum())
                new_stops.append(ns)
            stops = new_stops
            realised = [uu[int(np.argmax(s))] for uu, s in zip(utils, stops, strict=True)]
            history.append(
                {
                    "iteration": it,
                    "decisions_changed": changed,
                    "mean_realised_utility": float(np.mean(realised)),
                }
            )
        self.report = {
            "paths": len(paths),
            "decisions": int(sum(len(p.marks) - 1 for p in paths)),
            "utility": self.utility,
            "iterations": history,
        }
        return self.report

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Write the exported trees and ``stopping.json``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        if self.trees is not None:
            self.trees.save(d / "continuation.npz")
        meta = {
            "utility": self.utility,
            "iterations": self.iterations,
            "seed": self.seed,
            "report": self.report,
        }
        (d / "stopping.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, directory: str | Path) -> StoppingModel:
        """Rebuild a model saved by :meth:`save`."""
        d = Path(directory)
        meta = json.loads((d / "stopping.json").read_text())
        model = cls(str(meta["utility"]), int(meta["iterations"]), int(meta["seed"]))
        if (d / "continuation.npz").exists():
            model.trees = TreeEnsemble.load(d / "continuation.npz")
        model.report = dict(meta.get("report", {}))
        return model


# ---------------------------------------------------------------------------- research
def _thin(times: F64, spacing: float) -> npt.NDArray[np.int64]:
    """Indices of ``times`` (sorted) kept so that consecutive kept times are ≥ ``spacing`` apart."""
    keep: list[int] = []
    last = -np.inf
    for i, t in enumerate(times):
        if t - last >= spacing:
            keep.append(i)
            last = float(t)
    return np.asarray(keep, dtype=np.int64)


def build_paths(
    mds: Any, mints: list[str], data_end: float, spacing: float = 30.0
) -> tuple[list[StoppingPath], list[int]]:
    """One position per token, entered at its first entry-window snapshot and marked at the
    token's later snapshots (at most one per ``spacing`` seconds) up to the horizon and to
    ``data_end`` (a sale decided at ``s`` fills at ``s + latency``, which must be observed).
    Returns the paths and the candidate row (into ``mds``) of each entry.
    """
    from nardis_neural.solana.moonshot.labels import position_marks

    spec = mds.spec
    base_ts = np.asarray(mds.base.arrays["timestamp"], dtype=np.float64)
    base_mints = mds.base.mints
    cand_ts, cand_mints = mds.timestamps, mds.mints
    current = mds.base.current
    paths, entries = [], []
    for mint in mints:
        cand = np.flatnonzero(cand_mints == mint)
        if len(cand) == 0:
            continue
        row = int(cand[np.argmin(cand_ts[cand])])
        t0 = float(cand_ts[row])
        rows = np.flatnonzero(base_mints == mint)
        rows = rows[np.argsort(base_ts[rows], kind="stable")]
        tt = base_ts[rows]
        ok = (tt >= t0 + spec.latency_seconds) & (tt <= t0 + spec.horizon_seconds)
        ok &= tt + spec.latency_seconds <= data_end
        rows, tt = rows[ok], tt[ok]
        if len(rows) < 2:
            continue
        keep = _thin(tt, spacing)
        rows, tt = rows[keep], tt[keep]
        priced = position_marks(mds.market.token(mint), t0, spec, tt)
        if priced is None or len(priced[0]) < 2:
            continue
        times, marks = priced
        rows = rows[len(rows) - len(times) :]
        paths.append(StoppingPath(mint, t0, times, marks, np.asarray(current[rows], dtype=np.float32)))
        entries.append(row)
    return paths, entries


def _exit_stats(mult: F64, size_sol: float) -> dict[str, float]:
    m = np.asarray(mult, dtype=np.float64)
    if len(m) == 0:
        return {"tickets": 0.0}
    return {
        "tickets": float(len(m)),
        "total_pnl_sol": float(((m - 1.0) * size_sol).sum()),
        "mean_multiple": float(m.mean()),
        "median_multiple": float(np.median(m)),
        "mean_log_multiple": float(np.log(np.maximum(m, FLOOR)).mean()),
        "share_above_1x": float((m > 1.0).mean()),
    }


def _timer_exit(p: StoppingPath, seconds: float) -> float:
    k = int(np.searchsorted(p.times - p.t_entry, seconds, side="left"))
    return float(p.marks[min(k, len(p.marks) - 1)])


def run_stopping_research(
    store: Any,
    cfg: Any,
    spec: Any = None,
    test_fraction: float = 0.35,
    spacing: float = 30.0,
    iterations: int = 4,
    archetypes: dict[str, str] | None = None,
    seed: int = 0,
    log: Any = None,
) -> tuple[dict[str, Any], StoppingModel]:
    """Fit the exit model on earlier tokens (paths truncated at the cutoff), score it once on
    later tokens against hold-to-horizon, fixed timers, the take-profit ladder and the
    hindsight-perfect exit.  Every token is entered, so only the exit decision is compared.
    Returns the report and a model refitted on every token's full path.
    """
    from nardis_neural.solana.moonshot.labels import MoonshotSpec
    from nardis_neural.solana.moonshot.research import build_moonshot_dataset

    say = log or (lambda _m: None)
    spec = spec or MoonshotSpec()
    mds = build_moonshot_dataset(store, cfg, spec, archetypes=archetypes)
    launch = {str(m): mds.market.token(str(m)).launch.t for m in np.unique(mds.mints)}
    tokens = sorted(launch, key=lambda m: launch[m])
    n_test = max(1, round(len(tokens) * test_fraction))
    if len(tokens) - n_test < 5:
        raise ValueError("not enough tokens for a train / test split")
    train_tokens, test_tokens = tokens[:-n_test], tokens[-n_test:]
    cutoff = float(min(mds.timestamps[mds.mints == m].min() for m in test_tokens))
    train_paths, _ = build_paths(mds, train_tokens, cutoff, spacing)
    test_paths, test_rows = build_paths(mds, test_tokens, mds.data_end, spacing)
    say(
        f"stopping: {len(train_paths)} train paths ({sum(len(p.marks) for p in train_paths)} decisions, "
        f"truncated at the cutoff), {len(test_paths)} test paths"
    )
    policies: dict[str, StoppingModel] = {}
    fits: dict[str, Any] = {}
    for utility in ("log", "linear"):
        m = StoppingModel(utility, iterations, seed)
        fits[utility] = m.fit(train_paths)
        policies[utility] = m

    lab = mds.labels()
    held = np.array([p.marks[-1] for p in test_paths])
    exits: dict[str, F64] = {
        "optimal_stopping_log": np.array([p.marks[policies["log"].stop_index(p)] for p in test_paths]),
        "optimal_stopping_linear": np.array([p.marks[policies["linear"].stop_index(p)] for p in test_paths]),
        "hold_to_horizon": held,
        "ladder": lab.ladder[np.asarray(test_rows, dtype=np.int64)] if test_rows else np.zeros(0),
        "timer_5m": np.array([_timer_exit(p, 300.0) for p in test_paths]),
        "timer_30m": np.array([_timer_exit(p, 1800.0) for p in test_paths]),
        "hindsight_best": np.array([p.marks.max() for p in test_paths]),
    }
    hold_s = np.array(
        [p.times[policies["log"].stop_index(p)] - p.t_entry for p in test_paths], dtype=np.float64
    )
    report: dict[str, Any] = {
        "tokens": {"train": len(train_tokens), "test": len(test_tokens)},
        "cutoff": cutoff,
        "decision_spacing_seconds": spacing,
        "fit": fits,
        "exits": {k: _exit_stats(v, spec.size_sol) for k, v in exits.items()},
        "log_policy_median_hold_seconds": float(np.median(hold_s)) if len(hold_s) else float("nan"),
    }
    full_paths, _ = build_paths(mds, tokens, mds.data_end, spacing)
    production = StoppingModel("log", iterations, seed)
    production.fit(full_paths)
    return report, production


def stopping_markdown(report: dict[str, Any]) -> str:
    """Human-readable summary of :func:`run_stopping_research`."""
    lines = [
        "# Optimal-stopping exit research",
        "",
        f"Train tokens {report['tokens']['train']}, test tokens {report['tokens']['test']}; "
        f"decisions every ≥ {report['decision_spacing_seconds']:g} s. Every test token is entered; "
        "only the exit differs.",
        "",
        "| exit | tickets | PnL (SOL) | mean x | median x | mean log x | share > 1x |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in report["exits"].items():
        if not s.get("tickets"):
            continue
        lines.append(
            f"| {name} | {s['tickets']:.0f} | {s['total_pnl_sol']:+.2f} | {s['mean_multiple']:.2f} | "
            f"{s['median_multiple']:.2f} | {s['mean_log_multiple']:+.3f} | {s['share_above_1x']:.0%} |"
        )
    lines.append("")
    lines.append(
        f"Median holding time of the log-utility policy: {report['log_policy_median_hold_seconds']:.0f} s."
    )
    return "\n".join(lines) + "\n"
