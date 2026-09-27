"""Edge research: walk-forward out-of-fold predictions → meta-labeling → honest backtest.

Protocol (every step is causal):

1. Build snapshots with *executable* triple-barrier outcomes (:mod:`barriers`).
2. **Walk-forward** retraining of the neural ensemble and the launch-risk model: each fold
   trains only on data that ended (plus an embargo covering labels *and* holding periods)
   before the fold starts, then predicts the fold.  These out-of-fold (OOF) outputs are
   what a live system would have seen.
3. The OOF period is cut chronologically into **fit / tune / test**:
   the edge model learns on *fit*, the entry threshold is chosen on *tune*
   (max per-trade t-statistic), and everything is reported once on the untouched *test*
   period against random, momentum and take-everything baselines under the same
   position constraints and trade budget.
4. A production edge model is then refitted on all OOF rows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import Array, select_rows
from nardis_neural.data.splits import chronological_split, walk_forward_splits
from nardis_neural.solana.config import CURRENT_FEATURES, RISK_LABELS, SolanaConfig
from nardis_neural.solana.dataset import SolanaDataset, build_solana_dataset, hindsight_market
from nardis_neural.solana.edge.backtest import Candidates, baselines, select_threshold, simulate, trade_stats
from nardis_neural.solana.edge.barriers import BarrierSpec, triple_barrier
from nardis_neural.solana.edge.model import EdgeModel
from nardis_neural.solana.market import EventStore
from nardis_neural.solana.risk import SolanaRiskModel
from nardis_neural.training.pipeline import train_engine

F64 = npt.NDArray[np.float64]
Logger = Callable[[str], None]
HORIZON_KEYS = (
    "return.mean",
    "return.std",
    "upside.prob",
    "downside.prob",
    "max_upside.mean",
    "max_drawdown.mean",
    "volatility.mean",
)
SCALAR_KEYS = (
    "confidence",
    "epistemic",
    "aleatoric",
    "ood_score",
    "disagreement",
    "classification_epistemic",
)


def _quiet(_: str) -> None:
    return None


# ---------------------------------------------------------------------------- dataset
@dataclass
class EdgeDataset:
    """Snapshots with a complete executable barrier outcome: net return, exit reason, hold and exit time."""

    base: SolanaDataset
    net: F64
    reason: npt.NDArray[np.str_]
    hold: F64
    exit_time: F64

    @property
    def timestamps(self) -> F64:
        """Snapshot times in seconds."""
        return np.asarray(self.base.arrays["timestamp"], dtype=np.float64)

    def __len__(self) -> int:
        return len(self.net)


def subset_dataset(ds: SolanaDataset, keep: npt.NDArray[np.int64]) -> SolanaDataset:
    """Rows ``keep`` of a :class:`SolanaDataset` (every array, list and extra)."""
    return SolanaDataset(
        arrays=select_rows(ds.arrays, keep),
        observations=[ds.observations[i] for i in keep],
        outcomes=[ds.outcomes[i] for i in keep],
        mints=ds.mints[keep],
        risk=ds.risk[keep],
        net_returns=ds.net_returns[keep],
        round_trip_cost=ds.round_trip_cost[keep],
        extra={k: np.asarray(v)[keep] for k, v in ds.extra.items()},
    )


def build_edge_dataset(
    store: EventStore,
    cfg: SolanaConfig,
    spec: BarrierSpec,
    neural_cfg: NeuralConfig | None = None,
    archetypes: dict[str, str] | None = None,
) -> EdgeDataset:
    """Leakage-free snapshots labelled with executable triple-barrier outcomes (from the hindsight
    market); snapshots without a complete outcome are dropped.
    """
    base = build_solana_dataset(store, cfg, neural_cfg, archetypes=archetypes)
    full = hindsight_market(store, cfg)
    end = store.end_time
    rows, net, reason, hold, exit_t = [], [], [], [], []
    ts = np.asarray(base.arrays["timestamp"], dtype=np.float64)
    for i, (mint, t) in enumerate(zip(base.mints, ts, strict=True)):
        out = triple_barrier(full.token(str(mint)), float(t), spec, end)
        if out is None:
            continue
        rows.append(i)
        net.append(out.net_return)
        reason.append(out.exit_reason)
        hold.append(out.hold_seconds)
        exit_t.append(out.exit_time)
    if not rows:
        raise ValueError("no snapshot has a complete barrier outcome")
    keep = np.asarray(rows, dtype=np.int64)
    return EdgeDataset(
        subset_dataset(base, keep), np.asarray(net), np.asarray(reason), np.asarray(hold), np.asarray(exit_t)
    )


# ---------------------------------------------------------------------------- features
def edge_features(
    preds: dict[str, Array],
    risk: F64 | None,
    current: npt.NDArray[Any],
    horizons: list[str],
    experts: list[str],
) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Meta-features: neural forecasts per horizon, uncertainty, experts, risk, raw features."""
    cols: list[npt.NDArray[Any]] = []
    names: list[str] = []
    for key in HORIZON_KEYS:
        v = np.asarray(preds[key], dtype=np.float64)
        cols.append(v)
        names += [f"{key}.{h}" for h in horizons]
    z = np.asarray(preds["return.mean"]) / np.maximum(np.asarray(preds["return.std"]), 1e-8)
    cols.append(z)
    names += [f"return.z.{h}" for h in horizons]
    for key in SCALAR_KEYS:
        cols.append(np.asarray(preds[key], dtype=np.float64)[:, None])
        names.append(key)
    cols.append(np.asarray(preds["expert_weights"], dtype=np.float64))
    names += [f"gate.{e}" for e in experts]
    n = len(current)
    cols.append(np.full((n, len(RISK_LABELS)), 0.5) if risk is None else np.asarray(risk, dtype=np.float64))
    names += [f"risk.{k}" for k in RISK_LABELS]
    cols.append(np.asarray(current, dtype=np.float64))
    names += list(CURRENT_FEATURES)
    return np.nan_to_num(np.concatenate(cols, axis=1)).astype(np.float32), names


# ---------------------------------------------------------------------------- walk-forward OOF
@dataclass
class OOFPredictions:
    """Walk-forward out-of-fold neural outputs and risk probabilities (0.5 where unavailable), the
    rows the folds covered and per-fold statistics.
    """

    preds: dict[str, Array]
    risk: F64
    covered: npt.NDArray[np.bool_]
    experts: list[str]
    folds: list[dict[str, float]] = field(default_factory=list)


def walk_forward_oof(
    ds: SolanaDataset,
    cfg: SolanaConfig,
    ncfg: NeuralConfig,
    hold_seconds: float = 0.0,
    n_folds: int = 4,
    min_train_fraction: float = 0.4,
    device: torch.device | None = None,
    log: Logger = _quiet,
) -> OOFPredictions:
    """Walk-forward neural + risk predictions; ``hold_seconds`` extends the embargo."""
    store = ds.store()
    ts = np.asarray(ds.arrays["timestamp"], dtype=np.float64)
    n = len(ts)
    embargo = max(ncfg.embargo_seconds, cfg.risk_horizon_seconds, hold_seconds)
    splits = walk_forward_splits(ts, n_folds, min_train_fraction, embargo_seconds=embargo)
    preds: dict[str, Array] = {}
    risk = np.full((n, len(RISK_LABELS)), np.nan)
    covered = np.zeros(n, dtype=bool)
    experts: list[str] = []
    folds = []
    for k, sp in enumerate(splits):
        inner = chronological_split(ts[sp.train], ncfg.training.validation_fraction, embargo_seconds=embargo)
        tr, va = sp.train[inner.train], sp.train[inner.validation]
        engine, rep = train_engine(ncfg, store, tr, va, device=device, fit_regimes=False)
        experts = list(engine.expert_names)
        out = engine.predict_dataset(MarketDataset(store, ncfg, sp.validation))
        for key, v in out.items():
            arr = np.asarray(v)
            if arr.dtype.kind not in "fiub":
                continue
            if key not in preds:
                preds[key] = np.full((n, *arr.shape[1:]), np.nan)
            preds[key][sp.validation] = arr
        train_rows = np.sort(sp.train)
        emb_train = engine.predict_dataset(MarketDataset(store, ncfg, train_rows))["embedding"]
        x_train = SolanaRiskModel.inputs(emb_train, ds.current[train_rows])
        labelled = ~np.isnan(ds.risk[train_rows]).any(axis=1)
        if labelled.sum() >= 30 and np.nansum(ds.risk[train_rows], axis=0).min() >= 3:
            rm = SolanaRiskModel(x_train.shape[1], members=2)
            rm.fit(x_train, ds.risk[train_rows], ts[train_rows], groups=ds.mints[train_rows], epochs=40)
            risk[sp.validation] = rm.predict(
                SolanaRiskModel.inputs(out["embedding"], ds.current[sp.validation])
            )[0]
        covered[sp.validation] = True
        folds.append(
            {
                "fold": float(k),
                "train": float(len(tr)),
                "predict": float(len(sp.validation)),
                "val_downside_auc": float(rep.validation_metrics.get("downside.auc", np.nan)),
            }
        )
        log(f"fold {k}: trained on {len(tr)} rows, predicted {len(sp.validation)} out-of-fold")
    risk = np.where(np.isnan(risk), 0.5, risk)
    return OOFPredictions(preds, risk, covered, experts, folds)


# ---------------------------------------------------------------------------- research
@dataclass
class EdgeResearch:
    """Result of :func:`run_edge_research`: report, production edge model, dataset and OOF outputs."""

    report: dict[str, Any]
    model: EdgeModel
    dataset: EdgeDataset
    oof: OOFPredictions


def run_edge_research(
    store: EventStore,
    cfg: SolanaConfig,
    ncfg: NeuralConfig,
    spec: BarrierSpec | None = None,
    n_folds: int = 4,
    max_positions: int = 5,
    min_trades: int = 20,
    archetypes: dict[str, str] | None = None,
    device: torch.device | None = None,
    log: Logger = _quiet,
) -> EdgeResearch:
    """Run the causal edge protocol (see the module docstring) on an event history.

    Returns the out-of-sample report together with a production edge model refitted on every
    out-of-fold row.
    """
    spec = spec or BarrierSpec(size_sol=cfg.trade_size_sol)
    eds = build_edge_dataset(store, cfg, spec, ncfg, archetypes)
    log(
        f"edge dataset: {len(eds)} snapshots with executable outcomes, "
        f"base win rate {(eds.net > 0).mean():.3f}"
    )
    oof = walk_forward_oof(
        eds.base,
        cfg,
        ncfg,
        spec.max_hold_seconds + 2 * spec.latency_seconds,
        n_folds,
        device=device,
        log=log,
    )
    rows = np.flatnonzero(oof.covered)
    ts = eds.timestamps[rows]
    x, names = edge_features(
        {k: v[rows] for k, v in oof.preds.items()},
        oof.risk[rows],
        eds.base.current[rows],
        ncfg.horizon_names,
        oof.experts,
    )
    net = eds.net[rows]
    order = np.argsort(ts, kind="stable")
    a, b = int(len(order) * 0.5), int(len(order) * 0.75)
    fit_i, tune_i, test_i = order[:a], order[a:b], order[b:]
    gap = spec.max_hold_seconds + 2 * spec.latency_seconds
    tune_i = tune_i[ts[tune_i] >= ts[fit_i].max() + gap]
    test_i = test_i[ts[test_i] >= ts[tune_i].max() + gap] if len(tune_i) else test_i

    model = EdgeModel(x.shape[1], feature_names=names)
    fit_report = model.fit(x[fit_i], net[fit_i], ts[fit_i])

    def cands(idx: npt.NDArray[np.int64]) -> Candidates:
        r = rows[idx]
        return Candidates(eds.timestamps[r], eds.base.mints[r], eds.net[r], eds.exit_time[r], spec.size_sol)

    tune_scores = model.predict(x[tune_i]).edge_score
    threshold, curve = select_threshold(cands(tune_i), tune_scores, max_positions, min_trades)
    test_c = cands(test_i)
    test_pred = model.predict(x[test_i])
    trades = simulate(test_c, test_pred.edge_score >= threshold, max_positions)
    stats = trade_stats(test_c, trades)
    mom = eds.base.current[rows[test_i], list(CURRENT_FEATURES).index("ret_60s")].astype(np.float64)
    base = baselines(test_c, max(len(trades), 1), momentum=mom, max_positions=max_positions)
    report: dict[str, Any] = {
        "barrier": spec.model_dump(),
        "rows": {
            "total": len(eds),
            "oof": len(rows),
            "fit": len(fit_i),
            "tune": len(tune_i),
            "test": len(test_i),
        },
        "folds": oof.folds,
        "edge_model_validation": fit_report,
        "threshold": threshold,
        "tune_curve": curve,
        "test": stats,
        "test_baselines": base,
        "test_ic": float(np.corrcoef(test_pred.expected_net, test_c.net)[0, 1])
        if len(test_i) > 2
        else float("nan"),
    }
    if "archetype" in eds.base.extra and len(trades):
        arch = np.asarray(eds.base.extra["archetype"])[rows[test_i]][trades]
        report["test_trades_by_archetype"] = {
            str(a_): {
                "trades": int((arch == a_).sum()),
                "mean_net": float(test_c.net[trades][arch == a_].mean()),
            }
            for a_ in np.unique(arch)
        }
    beat_random = stats.get("mean_net", -np.inf) > base["random"].get("mean_net_p95", np.inf)
    report["verdict"] = {
        "positive_after_costs": bool(stats.get("ci95_low", -1) > 0),
        "beats_random_p95": bool(beat_random),
        "beats_momentum": bool(
            stats.get("mean_net", -np.inf) > base.get("momentum", {}).get("mean_net", np.inf)
        ),
        "beats_momentum_risk_adjusted": bool(
            stats.get("t_stat", -np.inf) > base.get("momentum", {}).get("t_stat", np.inf)
        ),
    }
    final = EdgeModel(x.shape[1], feature_names=names)
    final.fit(x, net, ts)
    return EdgeResearch(report, final, eds, oof)


def research_markdown(report: dict[str, Any]) -> str:
    """Render an edge research report as Markdown: test-period table versus baselines and the verdict."""
    t, b = report["test"], report["test_baselines"]
    v = report["verdict"]
    lines = [
        "# Edge research report (out-of-sample test period)",
        "",
        f"- rows: {report['rows']}",
        f"- entry threshold (chosen on tune period): `{report['threshold']:.4f}`",
        f"- test information coefficient (expected vs realised net): {report['test_ic']:.3f}",
        "",
        "| policy | trades | hit rate | mean net | 95% CI | t-stat | total PnL (SOL) | max DD (SOL) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def row(name: str, s: dict[str, Any]) -> str:
        if not s or s.get("trades", 0) == 0:
            return f"| {name} | 0 | | | | | | |"
        return (
            f"| {name} | {int(s['trades'])} | {s['hit_rate']:.1%} | {s['mean_net']:+.2%} | "
            f"[{s['ci95_low']:+.2%}, {s['ci95_high']:+.2%}] | {s['t_stat']:.2f} | "
            f"{s['total_pnl_sol']:+.2f} | "
            f"{s['max_drawdown_sol']:.2f} |"
        )

    lines.append(row("edge model", t))
    lines.append(row("momentum (same budget)", b.get("momentum", {})))
    lines.append(row("take every candidate", b["all_candidates"]))
    lines.append(
        f"| random (same budget, mean of draws) | | | {b['random']['mean_net']:+.2%} | "
        f"p95 {b['random']['mean_net_p95']:+.2%} | | | |"
    )
    lines += [
        "",
        f"**Verdict:** positive after costs (CI above 0): {v['positive_after_costs']} · beats random p95: "
        f"{v['beats_random_p95']} · beats momentum (mean): {v['beats_momentum']} · "
        f"beats momentum (t-stat): {v['beats_momentum_risk_adjusted']}",
        "",
        "Outcomes are executable triple-barrier results (latency, AMM impact and fees on both legs). "
        "Paper research only — past or simulated edge does not guarantee future results.",
    ]
    return "\n".join(lines) + "\n"
