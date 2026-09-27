from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nardis_neural.config import NeuralConfig, ReplayConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import select_rows
from nardis_neural.data.replay import Experience, ExperienceReplayBuffer, experiences_from_arrays
from nardis_neural.data.sequences import arrays_to_observations, arrays_to_outcomes


def _exp(
    i: int,
    ts: float | None = None,
    rare: bool = False,
    regime: int | None = None,
    priority: float | None = None,
) -> Experience:
    row = {
        "observation_id": np.array([f"e{i}"]),
        "timestamp": np.array([float(i if ts is None else ts)]),
        "current": np.zeros((1, 2), np.float32),
        "target.return": np.full((1, 3), 0.3 if rare else 0.01, np.float32),
        "target.max_upside": np.zeros((1, 3), np.float32),
        "target.max_drawdown": np.zeros((1, 3), np.float32),
        "target.volatility": np.zeros((1, 3), np.float32),
        "target_mask": np.ones((1, 3), bool),
    }
    return Experience(
        observation_id=f"e{i}",
        timestamp=float(i if ts is None else ts),
        row=row,
        is_rare=rare,
        regime=regime,
        priority=priority,
    )


def _buf(**kw: Any) -> ExperienceReplayBuffer:
    return ExperienceReplayBuffer(
        ReplayConfig(**({"recent_capacity": 10, "historical_capacity": 20, "rare_capacity": 5} | kw))
    )


def test_capacities_and_historical_retention() -> None:
    buf = _buf()
    for i in range(500):
        buf.add(_exp(i, rare=(i % 50 == 0)))
    assert len(buf.recent) == 10 and [e.seq for e in buf.recent] == list(range(490, 500))
    assert len(buf.historical) == 20
    hist_ts = np.array([e.timestamp for e in buf.historical])
    assert hist_ts.min() < 250, "reservoir keeps a spread of old data instead of discarding it"
    assert len(buf.rare) == 5 and all(e.is_rare for e in buf.rare)
    assert min(e.timestamp for e in buf.rare) < 490, "rare events outlive the recent window"
    assert len(buf) == len({e.seq for e in buf.all()}) <= 35
    ts = [e.timestamp for e in buf.all()]
    assert ts == sorted(ts)


def test_recency_sampling_prefers_new() -> None:
    buf = _buf(recent_capacity=100, recency_half_life=10.0)
    for i in range(100):
        buf.add(_exp(i))
    got, w = buf.sample(2000, "recency")
    assert np.mean([e.timestamp for e in got]) > 80 and (w == 1).all()
    uni, _ = buf.sample(2000, "uniform")
    assert abs(np.mean([e.timestamp for e in uni]) - 49.5) < 5


def test_prioritized_sampling_and_importance_weights() -> None:
    buf = _buf(recent_capacity=50)
    for i in range(50):
        buf.add(_exp(i, priority=10.0 if i == 7 else 0.1))
    got, w = buf.sample(1000, "prioritized")
    frac7 = np.mean([e.seq == 7 for e in got])
    assert frac7 > 0.2
    assert w.max() == pytest.approx(1.0) and w[[e.seq == 7 for e in got]].max() < 1.0
    buf.update_priorities([7], [0.1])
    got2, _ = buf.sample(1000, "prioritized")
    assert np.mean([e.seq == 7 for e in got2]) < 0.1
    unknown = _exp(99)
    buf.add(unknown)
    assert unknown.priority == buf.max_priority, "new experiences get max priority (PER)"


def test_regime_balanced_and_rare_sampling() -> None:
    buf = _buf(recent_capacity=200)
    for i in range(200):
        buf.add(_exp(i, regime=0 if i < 180 else 1, rare=i in (5, 6)))
    got, _ = buf.sample(4000, "regime_balanced")
    assert 0.4 < np.mean([e.regime == 1 for e in got]) < 0.6
    rare, _ = buf.sample(100, "rare")
    assert all(e.is_rare for e in rare)


def test_sample_mixture_respects_eligibility() -> None:
    buf = _buf(recent_capacity=30)
    for i in range(80):
        buf.add(_exp(i, rare=i % 10 == 0))
    eligible = {e.seq for e in buf.all() if e.timestamp < 60}
    mix = buf.sample_mixture(300, {"recent": 0.5, "historical": 0.3, "rare": 0.1, "difficult": 0.1}, eligible)
    assert len(mix) == 300 and all(e.seq in eligible for e in mix)
    with pytest.raises(ValueError):
        buf.sample_mixture(10, {"recent": 0.0})


def test_persistence_roundtrip(
    tmp_path: Path, tiny_config: NeuralConfig, later_arrays: dict[str, Any]
) -> None:
    rows = select_rows(later_arrays, np.arange(40))
    obs = arrays_to_observations(rows, tiny_config)
    outs = arrays_to_outcomes(rows, tiny_config)
    buf = ExperienceReplayBuffer(tiny_config.replay.model_copy(update={"recent_capacity": 15}))
    for i, (o, oc) in enumerate(zip(obs, outs, strict=True)):
        pred = {
            "expected_returns": {"30s": 0.0, "2m": 0.0, "5m": 0.0},
            "return_std": {"30s": 0.01, "2m": 0.02, "5m": 0.03},
        }
        buf.add(
            Experience.create(
                o,
                oc,
                tiny_config,
                model_version="v1",
                prediction=pred,
                uncertainty=0.1,
                embedding=[0.1 * i] * 4 if i % 2 else None,
                regime=i % 3,
            )
        )
    buf.save(tmp_path / "replay")
    back = ExperienceReplayBuffer.load(tmp_path / "replay", buf.cfg)
    assert [e.seq for e in back.recent] == [e.seq for e in buf.recent]
    assert [e.seq for e in back.historical] == [e.seq for e in buf.historical]
    assert back.next_seq == buf.next_seq and back.seen_historical == buf.seen_historical
    a, b = buf.all()[3], back.all()[3]
    assert a.observation_id == b.observation_id and a.model_version == b.model_version == "v1"
    assert a.priority == pytest.approx(b.priority) and a.regime == b.regime
    assert (a.embedding is None) == (b.embedding is None)
    np.testing.assert_array_equal(a.row["seq.fast.values"], b.row["seq.fast.values"])
    store = back.to_store()
    ds = MarketDataset(store, tiny_config)
    batch = ds[np.arange(len(ds))]
    assert batch.targets is not None and batch.size == len(back)
    empty = ExperienceReplayBuffer.load(tmp_path / "nothing", buf.cfg)
    assert len(empty) == 0


def test_experience_creation_and_priority(tiny_config: NeuralConfig, later_arrays: dict[str, Any]) -> None:
    rows = select_rows(later_arrays, np.arange(2))
    o = arrays_to_observations(rows, tiny_config)
    oc = arrays_to_outcomes(rows, tiny_config)
    y = float(rows["target.return"][0, 0])
    pred = {"expected_returns": {"30s": y + 0.02}, "return_std": {"30s": 0.01}}
    e = Experience.create(o[0], oc[0], tiny_config, prediction=pred)
    assert e.priority == pytest.approx(2.0, rel=1e-3)
    with pytest.raises(ValueError):
        Experience.create(o[0], oc[1], tiny_config)
    bulk = experiences_from_arrays(rows, tiny_config)
    assert "regime" not in bulk[0].row, "ground-truth diagnostics never enter replay rows"
    assert set(bulk[0].row) == set(e.row)
