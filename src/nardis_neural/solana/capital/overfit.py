"""Is the backtest edge real?  Multiple-testing-aware statistics.

Every threshold, window and model compared on the same history is a *trial*; the best of
many trials looks good by luck.  Two standard corrections (Bailey & López de Prado):

* **Deflated Sharpe Ratio** (DSR): the probability that the true Sharpe ratio of the chosen
  strategy is above the Sharpe ratio one would expect from the *best of N* unskilled
  trials, accounting for the sample length and for the skew and fat tails of the returns;
* **Probability of Backtest Overfitting** (PBO) by combinatorially symmetric
  cross-validation: split time into S blocks, and for every way of choosing half of them as
  in-sample, pick the best configuration in-sample and see where it ranks out-of-sample.
  PBO is the share of splits in which the in-sample winner is below the out-of-sample median.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

F64 = npt.NDArray[np.float64]
EULER_GAMMA = 0.5772156649015329


def sharpe(returns: F64) -> float:
    """Per-observation Sharpe ratio (mean / sample standard deviation); 0 when undefined."""
    r = np.asarray(returns, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    sd = float(r.std(ddof=1))
    return float(r.mean() / sd) if sd > 0 else 0.0


def expected_max_sharpe(n_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe ratio of ``n_trials`` independent unskilled strategies."""
    if n_trials <= 1:
        return 0.0
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(max(sharpe_variance, 0.0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def deflated_sharpe(returns: F64, n_trials: int, trial_sharpes: F64 | None = None) -> dict[str, float]:
    """Deflated Sharpe Ratio of ``returns`` after ``n_trials`` strategies were tried.

    ``trial_sharpes`` (the Sharpe ratios of every trial) estimates the cross-trial variance;
    without it the variance of a Sharpe estimate under the null, 1/T, is used.
    """
    r = np.asarray(returns, dtype=np.float64)
    t = len(r)
    sr = sharpe(r)
    if t < 3:
        return {"sharpe": sr, "benchmark_sharpe": float("nan"), "dsr": float("nan"), "n": float(t)}
    var = (
        float(np.var(trial_sharpes, ddof=1))
        if trial_sharpes is not None and len(trial_sharpes) > 1
        else 1.0 / t
    )
    sr0 = expected_max_sharpe(n_trials, var)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = math.sqrt(max(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr, 1e-12))
    dsr = float(stats.norm.cdf((sr - sr0) * math.sqrt(t - 1) / denom))
    return {"sharpe": sr, "benchmark_sharpe": sr0, "dsr": dsr, "skew": skew, "kurtosis": kurt, "n": float(t)}


def probability_of_backtest_overfitting(performance: F64, blocks: int = 8) -> dict[str, Any]:
    """PBO by combinatorially symmetric cross-validation.

    ``performance`` is ``(T, N)``: the per-period (or per-ticket, in time order) return of
    each of N configurations.  Rows are cut into ``blocks`` contiguous groups; every choice of
    half of them is an in-sample set, the rest out-of-sample.
    """
    m = np.asarray(performance, dtype=np.float64)
    t, n = m.shape
    if n < 2 or t < blocks or blocks < 2 or blocks % 2:
        return {"pbo": float("nan"), "splits": 0}
    groups = np.array_split(np.arange(t), blocks)
    logits = []
    degradation = []
    for is_blocks in combinations(range(blocks), blocks // 2):
        is_rows = np.concatenate([groups[b] for b in is_blocks])
        oos_rows = np.concatenate([groups[b] for b in range(blocks) if b not in is_blocks])
        is_perf = m[is_rows].mean(axis=0)
        oos_perf = m[oos_rows].mean(axis=0)
        best = int(np.argmax(is_perf))
        rank = float(stats.rankdata(oos_perf)[best]) / (n + 1)  # relative OOS rank in (0, 1)
        logits.append(math.log(rank / (1 - rank)))
        degradation.append((float(is_perf[best]), float(oos_perf[best])))
    lg = np.asarray(logits)
    deg = np.asarray(degradation)
    return {
        "pbo": float((lg <= 0).mean()),
        "splits": len(lg),
        "mean_logit": float(lg.mean()),
        "in_sample_best_mean": float(deg[:, 0].mean()),
        "out_of_sample_of_is_best_mean": float(deg[:, 1].mean()),
    }
