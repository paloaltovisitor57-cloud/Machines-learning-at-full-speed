"""Robustness suite: per-seed research and cross-seed aggregation."""

from __future__ import annotations

import numpy as np
import pytest

from nardis_neural.solana.suite import aggregate, run_suite, suite_markdown


def test_aggregate_skips_missing_values() -> None:
    agg = aggregate([{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": float("nan")}])
    assert agg["a"]["mean"] == 2.0 and agg["a"]["std"] == pytest.approx(np.sqrt(2.0))
    assert agg["b"]["n"] == 1 and agg["b"]["std"] == 0.0


@pytest.mark.slow
def test_suite_runs_two_small_markets() -> None:
    result = run_suite([3, 4], market="degen", n_tokens=24, hours=2.0)
    assert len(result["per_seed"]) == 2 and "test_nll" in result["aggregate"]
    assert result["aggregate"]["test_nll"]["n"] == 2
    assert all(0 <= v <= 1 for v in result["verdict_rate"].values())
    assert "Robustness suite" in suite_markdown(result)
