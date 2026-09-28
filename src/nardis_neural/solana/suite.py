"""Robustness suite: the same research on several independent simulated markets.

One market is an anecdote.  ``run_suite`` simulates a market per seed, runs the moonshot
research (and optionally the tape research) on each, and aggregates every headline metric
as mean ± standard deviation across seeds, together with how often each verdict held.
Seeds are independent draws of the whole market (tokens, wallets, archetypes), so the
spread measures how much a result depends on the particular market.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from nardis_neural.solana.config import SolanaConfig
from nardis_neural.solana.moonshot.research import run_moonshot_research
from nardis_neural.solana.simulator import MARKET_PRESETS, LaunchSimSpec, simulate_launches

Logger = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


def _moonshot_metrics(report: dict[str, Any]) -> dict[str, float]:
    out = {"test_nll": report["test_nll"], "test_nll_marginal": report["test_nll_marginal"]}
    for c in report["calibration"]:
        out[f"auc_{c['level']:g}x"] = c["auc"]
    for name, stats in (
        ("model", report["portfolio"]),
        ("every_launch", report["baselines"]["every_launch"]),
    ):
        out[f"{name}_pnl_sol"] = stats.get("total_pnl_sol", 0.0)
        out[f"{name}_mean_multiple"] = stats.get("mean_multiple", float("nan"))
    if "runners" in report:
        r = report["runners"]
        out["runners_ticketed_share"] = r["ticketed"] / r["in_test"] if r["in_test"] else float("nan")
    return out


def _tape_metrics(report: dict[str, Any]) -> dict[str, float]:
    out = {"tape_test_nll": report["test_nll"]["tape"], "tail_test_nll": report["test_nll"]["tail"]}
    for c in report["collapse"]:
        out[f"collapse_auc_{c['window_seconds']:g}s"] = c["auc"]
    ex = report.get("exit_policy", {}).get("test", {})
    if ex:
        out["ladder_only_pnl_sol"] = ex["ladder_only"].get("total_pnl_sol", 0.0)
        out["learned_exit_pnl_sol"] = ex["learned_exit"].get("total_pnl_sol", 0.0)
    return out


def aggregate(rows: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Mean, standard deviation, min, max and count of every metric across seeds (NaNs skipped)."""
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        v = np.asarray([r.get(k, np.nan) for r in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        if not len(v):
            continue
        out[k] = {
            "mean": float(v.mean()),
            "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()),
            "max": float(v.max()),
            "n": float(len(v)),
        }
    return out


def run_suite(
    seeds: list[int],
    market: str = "degen",
    n_tokens: int = 150,
    hours: float = 8.0,
    tape: bool = False,
    cfg: SolanaConfig | None = None,
    log: Logger = _quiet,
) -> dict[str, Any]:
    """Simulate one market per seed, run the research on each and aggregate across seeds."""
    from nardis_neural.solana.tape.research import run_tape_research

    cfg = cfg or SolanaConfig()
    ncfg = cfg.neural_config()
    per_seed: list[dict[str, Any]] = []
    verdicts: dict[str, list[bool]] = {}
    for seed in seeds:
        log(f"seed {seed}: simulating {n_tokens} launches over {hours:g} h ({market})")
        store, arch = simulate_launches(
            LaunchSimSpec(
                n_tokens=n_tokens,
                seed=seed,
                duration_seconds=hours * 3600,
                archetype_weights=dict(MARKET_PRESETS[market]),
            )
        )
        moon = run_moonshot_research(store, cfg, ncfg, archetypes=arch, seed=seed)
        metrics = _moonshot_metrics(moon.report)
        for k, v in moon.report["verdict"].items():
            verdicts.setdefault(f"moonshot.{k}", []).append(bool(v))
        if tape:
            tr = run_tape_research(store, cfg, ncfg, archetypes=arch, seed=seed, log=log)
            metrics |= _tape_metrics(tr.report)
            for k, v in tr.report["verdict"].items():
                verdicts.setdefault(f"tape.{k}", []).append(bool(v))
        per_seed.append({"seed": seed} | metrics)
        log(f"seed {seed}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if isinstance(v, float)))
    return {
        "seeds": seeds,
        "market": market,
        "tokens": n_tokens,
        "hours": hours,
        "per_seed": per_seed,
        "aggregate": aggregate([{k: v for k, v in r.items() if k != "seed"} for r in per_seed]),
        "verdict_rate": {k: float(np.mean(v)) for k, v in verdicts.items()},
    }


def suite_markdown(result: dict[str, Any]) -> str:
    """Mean ± sd table of every metric and how often each verdict held."""
    lines = [
        f"# Robustness suite: {len(result['seeds'])} independent `{result['market']}` markets",
        "",
        f"{result['tokens']} launches over {result['hours']:g} h per market; seeds {result['seeds']}.",
        "",
        "| metric | mean | sd | min | max | seeds |",
        "|---|---|---|---|---|---|",
    ]
    for k, a in result["aggregate"].items():
        lines.append(
            f"| {k} | {a['mean']:.3f} | {a['std']:.3f} | {a['min']:.3f} | {a['max']:.3f} | {int(a['n'])} |"
        )
    lines += ["", "| verdict | held in share of seeds |", "|---|---|"]
    for k, v in result["verdict_rate"].items():
        lines.append(f"| {k} | {v:.0%} |")
    return "\n".join(lines) + "\n"
