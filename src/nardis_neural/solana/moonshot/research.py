"""Moonshot research: can the tail model find the rare launches that run 100–1000x?

Protocol (every step causal):

1. Snapshots of every token are built by the causal replay; only *early* ones
   (``min_entry_age ≤ age ≤ max_entry_age``) are ticket candidates.
2. Optionally the neural ensemble and risk model are retrained walk-forward so that their
   out-of-fold outputs can feed the tail model (``inputs="neural"``); ``inputs="raw"``
   uses the on-chain features only and is much cheaper.
3. Tokens are split by launch time.  The tail model is trained on the earlier tokens with
   labels **truncated at the cutoff** (the first test entry): a runner that was still
   running at the cutoff is a censored row, exactly what a live system would have known.
4. On the later tokens the model is scored once: tail calibration of P(M ≥ k) for every
   level, ranking AUC, NLL against a marginal (feature-free) tail model, and a one-ticket-
   per-token portfolio that buys when the expected ladder payoff exceeds the ticket,
   compared with buying every launch, random tickets and momentum under the same budget.
5. A production model is refitted on all tokens with labels up to the end of the data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.solana.config import CURRENT_FEATURES, SolanaConfig
from nardis_neural.solana.dataset import SolanaDataset, build_solana_dataset, hindsight_market
from nardis_neural.solana.edge.research import OOFPredictions, edge_features, walk_forward_oof
from nardis_neural.solana.market import EventStore, SolanaMarket
from nardis_neural.solana.moonshot.labels import MoonshotSpec, moonshot_outcome
from nardis_neural.solana.moonshot.tail import TailModel
from nardis_neural.training.metrics import brier_score, roc_auc

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
Logger = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


@dataclass
class MoonshotLabels:
    """Per-candidate peak, ladder and final multiples; ``valid`` marks rows that have a label."""

    valid: npt.NDArray[np.bool_]
    peak: F64
    censored: npt.NDArray[np.bool_]
    ladder: F64
    final: F64
    collapse_time: F64 = field(default_factory=lambda: np.zeros(0))
    """Seconds from entry to the first collapse; NaN when none was observed."""
    observed: F64 = field(default_factory=lambda: np.zeros(0))
    """Seconds of path observed after entry (collapse labels are censored there)."""


@dataclass
class MoonshotDataset:
    """Entry-window snapshots (moonshot ticket candidates) with a hindsight market to label them."""

    base: SolanaDataset
    rows: I64
    """Indices into ``base`` of the early snapshots that are ticket candidates."""
    age: F64
    market: SolanaMarket
    spec: MoonshotSpec
    data_end: float
    _cache: dict[float, MoonshotLabels] = field(default_factory=dict)

    @property
    def timestamps(self) -> F64:
        """Snapshot time of every candidate."""
        return np.asarray(self.base.arrays["timestamp"], dtype=np.float64)[self.rows]

    @property
    def mints(self) -> npt.NDArray[np.str_]:
        """Mint of every candidate."""
        return self.base.mints[self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def labels(self, data_end: float | None = None) -> MoonshotLabels:
        """Outcomes of every candidate using only data up to ``data_end`` (default: all)."""
        end = self.data_end if data_end is None else float(data_end)
        if end not in self._cache:
            n = len(self.rows)
            valid = np.zeros(n, dtype=bool)
            peak, ladder, final = np.zeros(n), np.zeros(n), np.zeros(n)
            cens = np.zeros(n, dtype=bool)
            collapse, observed = np.full(n, np.nan), np.zeros(n)
            for i, (mint, t) in enumerate(zip(self.mints, self.timestamps, strict=True)):
                if t >= end:
                    continue
                out = moonshot_outcome(self.market.token(str(mint)), float(t), self.spec, end)
                if out is None:
                    continue
                valid[i] = True
                peak[i], ladder[i], final[i] = out.peak_multiple, out.ladder_multiple, out.final_multiple
                cens[i] = out.censored
                collapse[i] = np.nan if out.collapse_time is None else out.collapse_time
                observed[i] = out.observed_seconds
            self._cache[end] = MoonshotLabels(valid, peak, cens, ladder, final, collapse, observed)
        return self._cache[end]


def build_moonshot_dataset(
    store: EventStore,
    cfg: SolanaConfig,
    spec: MoonshotSpec,
    neural_cfg: NeuralConfig | None = None,
    archetypes: dict[str, str] | None = None,
    base: SolanaDataset | None = None,
) -> MoonshotDataset:
    """Pick the entry-window snapshots of ``base`` (built from ``store`` if omitted) as ticket
    candidates; raises ``ValueError`` if there are none.
    """
    base = base or build_solana_dataset(store, cfg, neural_cfg, archetypes=archetypes)
    market = hindsight_market(store, cfg)
    ts = np.asarray(base.arrays["timestamp"], dtype=np.float64)
    launch = np.asarray([market.token(str(m)).launch.t for m in base.mints])
    age = ts - launch
    rows = np.flatnonzero((age >= spec.min_entry_age_seconds) & (age <= spec.max_entry_age_seconds))
    if len(rows) == 0:
        raise ValueError("no snapshot falls inside the moonshot entry window")
    return MoonshotDataset(base, rows.astype(np.int64), age[rows], market, spec, store.end_time)


def moonshot_features(
    mds: MoonshotDataset, oof: OOFPredictions | None, horizons: list[str]
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Tail-model inputs: raw current features, or the edge meta-feature stack when ``oof`` is given."""
    current = mds.base.current[mds.rows]
    if oof is None:
        return np.nan_to_num(current).astype(np.float32), list(CURRENT_FEATURES)
    preds = {k: np.asarray(v)[mds.rows] for k, v in oof.preds.items()}
    return edge_features(preds, oof.risk[mds.rows], current, horizons, oof.experts)


# ---------------------------------------------------------------------------- portfolio
def first_signal(order_t: F64, mints: npt.NDArray[np.str_], selected: npt.NDArray[np.bool_]) -> I64:
    """Index of the first selected row of every token (one ticket per token)."""
    seen: set[str] = set()
    out: list[int] = []
    for i in np.argsort(order_t, kind="stable"):
        if selected[i] and str(mints[i]) not in seen:
            seen.add(str(mints[i]))
            out.append(int(i))
    return np.asarray(out, dtype=np.int64)


def ticket_stats(
    ladder: F64, peak: F64, size_sol: float, n_boot: int = 2000, seed: int = 0
) -> dict[str, float]:
    """Statistics of a set of tickets from their ladder and peak multiples: PnL in SOL, mean / median
    multiple with a bootstrap 95 % CI of the mean, hit rates and profit concentration.
    """
    n = len(ladder)
    if n == 0:
        return {"tickets": 0.0}
    pnl = (ladder - 1.0) * size_sol
    rng = np.random.default_rng(seed)
    boots = rng.choice(ladder, size=(n_boot, n), replace=True).mean(axis=1)
    profits = np.sort(pnl[pnl > 0])[::-1]
    return {
        "tickets": float(n),
        "total_pnl_sol": float(pnl.sum()),
        "mean_multiple": float(ladder.mean()),
        "median_multiple": float(np.median(ladder)),
        "ci95_low": float(np.quantile(boots, 0.025)),
        "ci95_high": float(np.quantile(boots, 0.975)),
        "hit_2x": float((ladder >= 2).mean()),
        "hit_10x": float((ladder >= 10).mean()),
        "peak_10x": float((peak >= 10).mean()),
        "peak_100x": float((peak >= 100).mean()),
        "best_multiple": float(ladder.max()),
        "best_peak": float(peak.max()),
        "top_ticket_profit_share": float(profits[0] / profits.sum()) if len(profits) else 0.0,
        "log_growth_per_ticket": float(np.log(np.maximum(ladder, 1e-3)).mean()),
    }


# ---------------------------------------------------------------------------- research
@dataclass
class MoonshotResearch:
    """Result of :func:`run_moonshot_research`: report, production tail model and dataset."""

    report: dict[str, Any]
    model: TailModel
    dataset: MoonshotDataset


def _calibration(pred_sf: F64, peak: F64, censored: npt.NDArray[np.bool_], levels: list[float]) -> list[Any]:
    rows = []
    for j, k in enumerate(levels):
        reached = peak >= k
        known = reached | ~censored  # a censored row that has not reached k is unresolved
        if not known.any():
            continue
        p, y = pred_sf[known, j], reached[known].astype(np.float64)
        rows.append(
            {
                "level": k,
                "rows": int(known.sum()),
                "predicted": float(p.mean()),
                "observed": float(y.mean()),
                "auc": roc_auc(p, y) if 0 < y.sum() < len(y) else float("nan"),
                "brier": brier_score(p, y),
            }
        )
    return rows


def run_moonshot_research(
    store: EventStore,
    cfg: SolanaConfig,
    ncfg: NeuralConfig,
    spec: MoonshotSpec | None = None,
    inputs: str = "raw",
    n_folds: int = 4,
    test_fraction: float = 0.35,
    min_expected_multiple: float = 1.0,
    members: int = 5,
    archetypes: dict[str, str] | None = None,
    device: torch.device | None = None,
    log: Logger = _quiet,
    seed: int = 0,
) -> MoonshotResearch:
    """Run the causal moonshot protocol (see the module docstring) on an event history.

    Returns the out-of-sample report together with a tail model refitted on every token.  Raises
    ``ValueError`` for unknown ``inputs`` or too few tokens for the split.
    """
    if inputs not in ("raw", "neural"):
        raise ValueError("inputs must be 'raw' or 'neural'")
    spec = spec or MoonshotSpec()
    mds = build_moonshot_dataset(store, cfg, spec, ncfg, archetypes)
    log(f"moonshot dataset: {len(mds)} early snapshots of {len(set(mds.mints))} tokens")
    oof = None
    eligible = np.ones(len(mds), dtype=bool)
    if inputs == "neural":
        oof = walk_forward_oof(mds.base, cfg, ncfg, 0.0, n_folds, device=device, log=log)
        eligible = oof.covered[mds.rows]
    x, names = moonshot_features(mds, oof, ncfg.horizon_names)
    ts, mints = mds.timestamps, mds.mints

    # ---- token split by launch time; train labels truncated at the cutoff
    launch = {str(m): mds.market.token(str(m)).launch.t for m in np.unique(mints[eligible])}
    tokens = sorted(launch, key=lambda m: launch[m])
    n_test = max(1, round(len(tokens) * test_fraction))
    if len(tokens) - n_test < 5:
        raise ValueError("not enough tokens for a train / test split")
    test_tokens = set(tokens[-n_test:])
    is_test = np.array([str(m) in test_tokens for m in mints]) & eligible
    cutoff = float(ts[is_test].min())
    train_lab = mds.labels(cutoff)
    train = np.flatnonzero(eligible & ~is_test & train_lab.valid & (ts + spec.latency_seconds < cutoff))
    log(
        f"train: {len(train)} rows of {len(tokens) - n_test} tokens "
        f"({train_lab.censored[train].mean():.0%} censored at the cutoff); test: {n_test} tokens"
    )
    model = TailModel(x.shape[1], spec, members=members, feature_names=names, inputs=inputs, seed=seed)
    fit_rep = model.fit(x[train], train_lab.peak[train], train_lab.censored[train], ts[train], mints[train])
    marginal = TailModel(1, spec, members=1, seed=seed)
    ones = np.ones((len(x), 1), dtype=np.float32)
    marginal.fit(ones[train], train_lab.peak[train], train_lab.censored[train], ts[train], mints[train])

    lab = mds.labels()
    test = np.flatnonzero(is_test & lab.valid)
    pred = model.predict(x[test])
    peak, cens, ladder = lab.peak[test], lab.censored[test], lab.ladder[test]
    nll = float(model.nll(x[test], peak, cens).mean())
    nll_marginal = float(marginal.nll(ones[test], peak, cens).mean())

    # ---- one ticket per token
    select = (pred.expected_multiple >= min_expected_multiple) & (pred.lottery_kelly > 0)
    picked = first_signal(ts[test], mints[test], select)
    everything = first_signal(ts[test], mints[test], np.ones(len(test), dtype=bool))
    stats = ticket_stats(ladder[picked], peak[picked], spec.size_sol)
    rng = np.random.default_rng(seed)
    by_token: dict[str, I64] = {}
    for i, m in enumerate(mints[test]):
        by_token.setdefault(str(m), np.empty(0, dtype=np.int64))
        by_token[str(m)] = np.append(by_token[str(m)], i)
    budget = max(len(picked), 1)
    draws = []
    for _ in range(500):
        chosen = rng.choice(list(by_token), size=min(budget, len(by_token)), replace=False)
        idx = np.array([rng.choice(by_token[m]) for m in chosen])
        draws.append(((ladder[idx] - 1.0) * spec.size_sol).sum())
    ret60 = mds.base.current[mds.rows[test], list(CURRENT_FEATURES).index("ret_60s")]
    mom_rows = first_signal(ts[test], mints[test], mds.age[test] >= 60)
    mom = mom_rows[np.argsort(-ret60[mom_rows], kind="stable")[:budget]]
    sensitivity = []
    for thr in (1.0, 1.5, 2.0, 3.0, 5.0, 10.0):
        sel = first_signal(ts[test], mints[test], (pred.expected_multiple >= thr) & (pred.lottery_kelly > 0))
        st = ticket_stats(ladder[sel], peak[sel], spec.size_sol)
        sensitivity.append({"min_expected_multiple": thr} | st)
    report: dict[str, Any] = {
        "spec": spec.model_dump(),
        "inputs": inputs,
        "rows": {"candidates": len(mds), "train": len(train), "test": len(test)},
        "tokens": {"train": len(tokens) - n_test, "test": n_test},
        "cutoff": cutoff,
        "censored_train_share": float(train_lab.censored[train].mean()),
        "tail_model": fit_rep,
        "test_nll": nll,
        "test_nll_marginal": nll_marginal,
        "calibration": _calibration(pred.survival, peak, cens, spec.levels),
        "expected_vs_realized": {
            "mean_expected_multiple": float(pred.expected_multiple.mean()),
            "mean_realized_ladder": float(ladder.mean()),
        },
        "portfolio": stats,
        "threshold_sensitivity": sensitivity,
        "baselines": {
            "every_launch": ticket_stats(ladder[everything], peak[everything], spec.size_sol),
            "momentum": ticket_stats(ladder[mom], peak[mom], spec.size_sol),
            "random": {
                "total_pnl_sol_mean": float(np.mean(draws)),
                "total_pnl_sol_p95": float(np.quantile(draws, 0.95)),
            },
        },
    }
    if "archetype" in mds.base.extra:
        arch = np.asarray(mds.base.extra["archetype"])[mds.rows[test]]
        runners = {str(m) for m, a in zip(mints[test], arch, strict=True) if a == "runner"}
        caught = {str(m) for m in mints[test][picked]} & runners
        report["runners"] = {
            "in_test": len(runners),
            "ticketed": len(caught),
            "ticket_multiples": [float(ladder[i]) for i in picked if str(mints[test][i]) in runners],
            "tickets_by_archetype": {str(a): int((arch[picked] == a).sum()) for a in np.unique(arch[picked])}
            if len(picked)
            else {},
        }
    rand = report["baselines"]["random"]
    report["verdict"] = {
        "beats_marginal_nll": nll < nll_marginal,
        "profitable": bool(stats.get("total_pnl_sol", -np.inf) > 0),
        "beats_every_launch_per_ticket": bool(
            stats.get("mean_multiple", -np.inf)
            > report["baselines"]["every_launch"].get("mean_multiple", np.inf)
        ),
        "beats_random_p95": bool(stats.get("total_pnl_sol", -np.inf) > rand["total_pnl_sol_p95"]),
        "beats_momentum": bool(
            stats.get("total_pnl_sol", -np.inf) > report["baselines"]["momentum"].get("total_pnl_sol", np.inf)
        ),
    }
    log("refitting the production tail model on every token")
    rows_all = np.flatnonzero(eligible & lab.valid)
    final = TailModel(x.shape[1], spec, members=members, feature_names=names, inputs=inputs, seed=seed)
    final.fit(x[rows_all], lab.peak[rows_all], lab.censored[rows_all], ts[rows_all], mints[rows_all])
    return MoonshotResearch(report, final, mds)


def moonshot_markdown(report: dict[str, Any]) -> str:
    """Render a moonshot research report as Markdown."""
    p, b, v = report["portfolio"], report["baselines"], report["verdict"]
    lines = [
        "# Moonshot research report (out-of-sample tokens)",
        "",
        f"- inputs: `{report['inputs']}` · rows {report['rows']} · tokens {report['tokens']}",
        f"- train labels censored at the cutoff: {report['censored_train_share']:.0%}",
        f"- test NLL of log peak multiple: {report['test_nll']:.3f} "
        f"(feature-free tail model: {report['test_nll_marginal']:.3f})",
        "",
        "## Tail calibration: P(peak multiple ≥ k)",
        "",
        "| k | resolved rows | predicted | observed | AUC | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["calibration"]:
        lines.append(
            f"| {c['level']:g}x | {c['rows']} | {c['predicted']:.3f} | {c['observed']:.3f} | "
            f"{c['auc']:.3f} | {c['brier']:.3f} |"
        )
    lines += [
        "",
        "## One ticket per token (ladder exits, latency, impact, fees)",
        "",
        "| policy | tickets | total PnL (SOL) | mean multiple | 95% CI | median | ≥10x | best |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, s: dict[str, Any]) -> str:
        if not s or s.get("tickets", 0) == 0:
            return f"| {name} | 0 | | | | | | |"
        return (
            f"| {name} | {int(s['tickets'])} | {s['total_pnl_sol']:+.2f} | {s['mean_multiple']:.2f}x | "
            f"[{s['ci95_low']:.2f}, {s['ci95_high']:.2f}] | {s['median_multiple']:.2f}x | "
            f"{s['hit_10x']:.0%} | {s['best_multiple']:.1f}x |"
        )

    lines.append(row("tail model (E[payoff] > ticket)", p))
    lines.append(row("momentum (same budget)", b["momentum"]))
    lines.append(row("every launch", b["every_launch"]))
    lines.append(
        f"| random (same budget) | | {b['random']['total_pnl_sol_mean']:+.2f} "
        f"(p95 {b['random']['total_pnl_sol_p95']:+.2f}) | | | | | |"
    )
    lines += [
        "",
        "Sensitivity to the entry threshold (test period, shown for transparency, never used to choose it):",
        "",
        "| E[payoff] ≥ | tickets | total PnL (SOL) | mean multiple | ≥10x |",
        "|---|---|---|---|---|",
    ]
    for s_ in report.get("threshold_sensitivity", []):
        if s_.get("tickets", 0):
            lines.append(
                f"| {s_['min_expected_multiple']:g}x | {int(s_['tickets'])} | {s_['total_pnl_sol']:+.2f} | "
                f"{s_['mean_multiple']:.2f}x | {s_['hit_10x']:.0%} |"
            )
        else:
            lines.append(f"| {s_['min_expected_multiple']:g}x | 0 | | | |")
    if "runners" in report:
        r = report["runners"]
        lines += ["", f"Runners in the test tokens: {r['in_test']}, ticketed: {r['ticketed']}."]
    lines += [
        "",
        f"**Verdict:** beats feature-free NLL: {v['beats_marginal_nll']} · profitable: {v['profitable']} · "
        f"beats every-launch per ticket: {v['beats_every_launch_per_ticket']} · beats random p95: "
        f"{v['beats_random_p95']} · beats momentum: {v['beats_momentum']}",
        "",
        "Fat-tailed results are dominated by a handful of tickets; read the ticket count and the "
        "top-ticket profit share before the mean. Paper research only.",
    ]
    return "\n".join(lines) + "\n"
