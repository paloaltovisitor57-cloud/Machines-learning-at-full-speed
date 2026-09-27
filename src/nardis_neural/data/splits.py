"""Leakage-free chronological splitting.

Labels look ``max_horizon`` seconds into the future, so a training sample observed just
before the validation period has a label that overlaps validation time.  Every splitter
therefore applies an *embargo*: training samples must satisfy
``t_train <= t_val_start - embargo_seconds``.  Samples are never shuffled across time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Idx = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Split:
    """Row indices of a chronological train / validation / optional test split."""

    train: Idx
    validation: Idx
    test: Idx | None = None

    def check_no_leakage(self, timestamps: npt.NDArray[np.float64], embargo: float = 0.0) -> None:
        """Assert each split ends at least ``embargo`` seconds before the next one starts."""
        both = len(self.train) and len(self.validation)
        if both and timestamps[self.train].max() > timestamps[self.validation].min() - embargo:
            raise AssertionError("training data overlaps validation period")
        test = self.test if self.test is not None else np.zeros(0, dtype=np.int64)
        has_test = len(test) and len(self.validation)
        if has_test and timestamps[self.validation].max() > timestamps[test].min() - embargo:
            raise AssertionError("validation data overlaps test period")


def _embargoed(order: Idx, timestamps: npt.NDArray[np.float64], cutoff: float, embargo: float) -> Idx:
    return order[timestamps[order] <= cutoff - embargo]


def chronological_split(
    timestamps: npt.NDArray[np.float64],
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    embargo_seconds: float = 0.0,
) -> Split:
    """Oldest data → train, then validation, then (optional) test."""
    ts = np.asarray(timestamps, dtype=np.float64)
    n = len(ts)
    if n < 3:
        raise ValueError("need at least 3 samples to split")
    order = np.argsort(ts, kind="stable").astype(np.int64)
    n_test = round(n * test_fraction)
    n_val = max(1, round(n * validation_fraction))
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ValueError("split fractions leave no training data")
    val = order[n_train : n_train + n_val]
    test = order[n_train + n_val :] if n_test else None
    train = _embargoed(order[:n_train], ts, float(ts[val].min()), embargo_seconds)
    if test is not None:
        val = _embargoed(val, ts, float(ts[test].min()), embargo_seconds)
    if len(train) == 0:
        raise ValueError("embargo removed all training samples; reduce embargo or add data")
    return Split(train=train, validation=val, test=test)


def walk_forward_splits(
    timestamps: npt.NDArray[np.float64],
    n_folds: int = 4,
    min_train_fraction: float = 0.4,
    embargo_seconds: float = 0.0,
    expanding: bool = True,
    window_fraction: float | None = None,
) -> list[Split]:
    """Walk-forward validation.

    The time-ordered data after the first ``min_train_fraction`` is cut into ``n_folds``
    consecutive validation blocks.  Fold ``k`` trains on everything before block ``k``
    (expanding window) or on a fixed-size rolling window (``expanding=False``).
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(ts, kind="stable").astype(np.int64)
    n = len(order)
    start = int(n * min_train_fraction)
    if start < 1 or n - start < n_folds:
        raise ValueError("not enough data for walk-forward splits")
    bounds = np.linspace(start, n, n_folds + 1).astype(int)
    window = int(n * window_fraction) if window_fraction is not None else start
    splits: list[Split] = []
    for k in range(n_folds):
        val = order[bounds[k] : bounds[k + 1]]
        if len(val) == 0:
            continue
        lo = 0 if expanding else max(0, bounds[k] - window)
        train = _embargoed(order[lo : bounds[k]], ts, float(ts[val].min()), embargo_seconds)
        if len(train) == 0:
            continue
        splits.append(Split(train=train, validation=val))
    return splits


def rolling_window_splits(
    timestamps: npt.NDArray[np.float64],
    train_seconds: float,
    validation_seconds: float,
    step_seconds: float | None = None,
    embargo_seconds: float = 0.0,
) -> list[Split]:
    """Fixed-duration rolling windows in wall-clock time."""
    ts = np.asarray(timestamps, dtype=np.float64)
    order = np.argsort(ts, kind="stable").astype(np.int64)
    sorted_ts = ts[order]
    step = step_seconds if step_seconds is not None else validation_seconds
    t0, t_end = float(sorted_ts[0]), float(sorted_ts[-1])
    splits: list[Split] = []
    cursor = t0 + train_seconds
    while cursor < t_end:
        tr_mask = (sorted_ts >= cursor - train_seconds) & (sorted_ts <= cursor - embargo_seconds)
        va_mask = (sorted_ts > cursor) & (sorted_ts <= cursor + validation_seconds)
        if tr_mask.any() and va_mask.any():
            splits.append(Split(train=order[tr_mask], validation=order[va_mask]))
        cursor += step
    return splits
