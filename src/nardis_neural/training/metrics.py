"""Evaluation metrics shared by training, calibration, shadow mode and promotion.

Metrics operate on NumPy arrays in *real* units.  Predictions use the flat key layout
produced by :meth:`nardis_neural.inference.engine.NeuralEngine.predict_arrays`:
``"<task>.mean"``, ``"<task>.std"`` (N, H) for regression tasks, ``"<task>.prob"`` for
event tasks, ``"return.quantiles"`` (N, H, Q).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from nardis_neural.config import CLASSIFICATION_TASKS, REGRESSION_TASKS, NeuralConfig

Array = npt.NDArray[Any]
EPS = 1e-7


def brier_score(prob: Array, labels: Array) -> float:
    """Mean squared error between probabilities and 0/1 labels (NaN if empty)."""
    return float(np.mean((prob - labels) ** 2)) if len(prob) else math.nan


def log_loss(prob: Array, labels: Array) -> float:
    """Mean binary cross-entropy with probabilities clipped away from 0 and 1 (NaN if empty)."""
    if not len(prob):
        return math.nan
    p = np.clip(prob, EPS, 1 - EPS)
    return float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))


def reliability_bins(prob: Array, labels: Array, n_bins: int = 10) -> list[dict[str, float]]:
    """Count, mean confidence and observed frequency per equal-width probability bin."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob, edges[1:-1]), 0, n_bins - 1)
    out = []
    for b in range(n_bins):
        sel = idx == b
        cnt = int(sel.sum())
        out.append(
            {
                "lower": float(edges[b]),
                "upper": float(edges[b + 1]),
                "count": float(cnt),
                "mean_confidence": float(prob[sel].mean()) if cnt else math.nan,
                "observed_frequency": float(labels[sel].mean()) if cnt else math.nan,
            }
        )
    return out


def expected_calibration_error(prob: Array, labels: Array, n_bins: int = 10) -> float:
    """Count-weighted mean |confidence − frequency| over reliability bins (NaN if empty)."""
    if not len(prob):
        return math.nan
    total = len(prob)
    ece = 0.0
    for b in reliability_bins(prob, labels, n_bins):
        if b["count"]:
            ece += b["count"] / total * abs(b["mean_confidence"] - b["observed_frequency"])
    return float(ece)


def roc_auc(prob: Array, labels: Array) -> float:
    """ROC AUC from the rank-sum statistic (NaN unless both classes are present)."""
    pos = labels > 0.5
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    ranks = stats.rankdata(prob)
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def rank_correlation(a: Array, b: Array) -> float:
    """Spearman correlation; 0 for fewer than 3 points, constant inputs or undefined results."""
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = stats.spearmanr(a, b).statistic
    return float(r) if np.isfinite(r) else 0.0


def gaussian_nll_np(mean: Array, std: Array, y: Array) -> float:
    """Mean Gaussian negative log-likelihood of ``y`` under N(mean, std²), including log 2π."""
    var = np.maximum(std, 1e-8) ** 2
    return float(np.mean(0.5 * (np.log(2 * np.pi * var) + (y - mean) ** 2 / var)))


def labels_from_targets(targets: Mapping[str, Array], config: NeuralConfig) -> dict[str, Array]:
    """Upside/downside event labels (N, H) from real-unit targets and per-horizon thresholds."""
    up = np.array([h.upside_threshold for h in config.targets.horizons])
    down = np.array([h.downside_threshold for h in config.targets.horizons])
    return {
        "upside": (targets["return"] > up).astype(np.float64),
        "downside": (targets["max_drawdown"] > down).astype(np.float64),
    }


def evaluate_predictions(
    preds: Mapping[str, Array],
    targets: Mapping[str, Array],
    config: NeuralConfig,
    mask: Array | None = None,
    tail_quantile: float = 0.9,
    per_horizon: bool = True,
) -> dict[str, float]:
    """Flat metric dict.  ``targets`` maps regression task → (N, H) real-unit values.

    Aggregate keys (``return.rmse``, ``upside.log_loss``, …) average across horizons;
    ``<metric>.<horizon>`` keys are per horizon.
    """
    horizons = config.horizon_names
    n = len(next(iter(targets.values())))
    m = np.ones((n, len(horizons)), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    labels = labels_from_targets(targets, config)
    out: dict[str, float] = {"n": float(n)}
    acc: dict[str, list[float]] = {}

    def put(key: str, h: int, value: float) -> None:
        if per_horizon:
            out[f"{key}.{horizons[h]}"] = value
        acc.setdefault(key, []).append(value)

    for h in range(len(horizons)):
        sel = m[:, h]
        if sel.sum() < 2:
            continue
        for task in REGRESSION_TASKS:
            if f"{task}.mean" not in preds:
                continue
            y = targets[task][sel, h]
            mu = preds[f"{task}.mean"][sel, h]
            err = mu - y
            put(f"{task}.mae", h, float(np.mean(np.abs(err))))
            put(f"{task}.rmse", h, float(np.sqrt(np.mean(err**2))))
            put(f"{task}.rank_corr", h, rank_correlation(mu, y))
            if f"{task}.std" in preds:
                sd = preds[f"{task}.std"][sel, h]
                put(f"{task}.nll", h, gaussian_nll_np(mu, sd, y))
                put(f"{task}.coverage90", h, float(np.mean(np.abs(err) <= 1.645 * sd)))
                put(f"{task}.unc_error_corr", h, rank_correlation(sd, np.abs(err)))
        y_ret = targets["return"][sel, h]
        if "return.mean" in preds:
            y_mag = np.abs(y_ret)
            thr = np.quantile(y_mag, tail_quantile)
            tail = y_mag >= thr
            put(
                "tail.return_mae", h, float(np.mean(np.abs(preds["return.mean"][sel, h][tail] - y_ret[tail])))
            )
        if "max_drawdown.mean" in preds:
            dd = targets["max_drawdown"][sel, h]
            thr = np.quantile(dd, tail_quantile)
            tail = dd >= thr
            put(
                "tail.drawdown_mae",
                h,
                float(np.mean(np.abs(preds["max_drawdown.mean"][sel, h][tail] - dd[tail]))),
            )
        if "return.quantiles" in preds:
            q = preds["return.quantiles"][sel, h]
            levels = np.asarray(config.targets.quantiles)
            diff = y_ret[:, None] - q
            put("return.pinball", h, float(np.mean(np.maximum(levels * diff, (levels - 1) * diff))))
        for task in CLASSIFICATION_TASKS:
            if f"{task}.prob" not in preds:
                continue
            p = preds[f"{task}.prob"][sel, h]
            y = labels[task][sel, h]
            put(f"{task}.log_loss", h, log_loss(p, y))
            put(f"{task}.brier", h, brier_score(p, y))
            put(f"{task}.ece", h, expected_calibration_error(p, y, config.calibration.n_bins))
            put(f"{task}.auc", h, roc_auc(p, y))
            put(f"{task}.base_rate", h, float(y.mean()))
            if task == "downside":
                pos = y > 0.5
                if pos.any():
                    put("downside.tail_recall", h, float(np.mean(p[pos] > 0.5)))
    for key, vals in acc.items():
        finite = [v for v in vals if np.isfinite(v)]
        out[key] = float(np.mean(finite)) if finite else math.nan
    return out


def summarize_groups(
    preds: Mapping[str, Array],
    targets: Mapping[str, Array],
    groups: Array,
    config: NeuralConfig,
    mask: Array | None = None,
    min_count: int = 10,
    keys: Sequence[str] = (
        "return.rmse",
        "return.mae",
        "upside.log_loss",
        "downside.log_loss",
        "upside.brier",
    ),
) -> dict[str, dict[str, float]]:
    """Aggregate metrics per group label (regime, time window …)."""
    out: dict[str, dict[str, float]] = {}
    for g in np.unique(groups):
        sel = groups == g
        if sel.sum() < min_count:
            continue
        p = {k: v[sel] for k, v in preds.items() if hasattr(v, "shape") and len(v) == len(groups)}
        t = {k: v[sel] for k, v in targets.items()}
        mm = None if mask is None else np.asarray(mask)[sel]
        full = evaluate_predictions(p, t, config, mm, per_horizon=False)
        out[str(g)] = {k: full[k] for k in keys if k in full} | {"n": float(sel.sum())}
    return out
