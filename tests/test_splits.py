"""Time-series safety: no random leakage, embargo respected, walk-forward is causal."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from nardis_neural.data.splits import chronological_split, rolling_window_splits, walk_forward_splits


@settings(max_examples=60, deadline=None)
@given(
    ts=st.lists(st.floats(0, 1e6, allow_nan=False), min_size=10, max_size=300),
    frac=st.floats(0.1, 0.5),
    embargo=st.floats(0, 1000),
)
def test_chronological_split_never_leaks(ts: list[float], frac: float, embargo: float) -> None:
    t = np.asarray(ts)
    try:
        split = chronological_split(t, frac, embargo_seconds=embargo)
    except ValueError:
        return  # embargo can legitimately remove all training data
    assert t[split.train].max() <= t[split.validation].min() - embargo
    assert not set(split.train) & set(split.validation)
    split.check_no_leakage(t, embargo)


def test_split_with_test_and_shuffled_input() -> None:
    rng = np.random.default_rng(0)
    t = rng.permutation(np.arange(1000.0))
    split = chronological_split(t, 0.2, 0.1, embargo_seconds=5)
    assert split.test is not None
    assert t[split.train].max() <= t[split.validation].min() - 5
    assert t[split.validation].max() <= t[split.test].min() - 5
    assert len(split.test) == 100


def test_leakage_check_detects_overlap() -> None:
    from nardis_neural.data.splits import Split

    t = np.arange(10.0)
    bad = Split(train=np.array([0, 5]), validation=np.array([3, 4]))
    with pytest.raises(AssertionError):
        bad.check_no_leakage(t)


def test_walk_forward_expanding_and_rolling() -> None:
    t = np.arange(1000.0)
    folds = walk_forward_splits(t, n_folds=4, min_train_fraction=0.4, embargo_seconds=10)
    assert len(folds) == 4
    prev_val_end = -1.0
    for f in folds:
        assert t[f.train].max() <= t[f.validation].min() - 10
        assert t[f.validation].min() > prev_val_end
        prev_val_end = t[f.validation].max()
    assert len(folds[-1].train) > len(folds[0].train)  # expanding
    rolling = walk_forward_splits(t, n_folds=4, expanding=False, window_fraction=0.2)
    assert all(len(f.train) <= 200 for f in rolling)


def test_rolling_windows_in_time() -> None:
    t = np.arange(0, 1000.0, 1.0)
    splits = rolling_window_splits(t, train_seconds=200, validation_seconds=50, embargo_seconds=5)
    assert len(splits) >= 10
    for s in splits:
        assert t[s.train].max() <= t[s.validation].min() - 5
        assert t[s.train].max() - t[s.train].min() <= 200


def test_training_pipeline_split_has_no_future_rows(base_store) -> None:  # type: ignore[no-untyped-def]
    from tests.conftest import make_tiny_config

    cfg = make_tiny_config()
    split = chronological_split(
        base_store.timestamps, cfg.training.validation_fraction, embargo_seconds=cfg.embargo_seconds
    )
    ts = base_store.timestamps
    # every training label window (t + max horizon) ends before validation starts
    assert (ts[split.train] + cfg.targets.max_horizon_seconds).max() <= ts[split.validation].min()
