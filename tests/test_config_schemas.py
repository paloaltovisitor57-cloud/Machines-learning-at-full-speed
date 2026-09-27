from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from nardis_neural.config import NeuralConfig, TimescaleConfig, load_config
from nardis_neural.data.loaders import seq_key
from nardis_neural.data.sequences import (
    arrays_to_observations,
    build_bars,
    observations_to_arrays,
    outcomes_to_arrays,
    pad_sequence,
)
from nardis_neural.schemas import GraphInput, NeuralObservation, NeuralOutcome, SequenceInput


def test_default_config_is_valid_and_roundtrips(tmp_path: Path) -> None:
    cfg = NeuralConfig()
    assert cfg.model.enabled_experts == ["transformer", "recurrent", "tcn", "tabular"]
    assert cfg.horizon_names == ["30s", "2m", "5m"]
    assert cfg.embargo_seconds == 300
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert load_config(path) == cfg
    assert load_config(None) == NeuralConfig()


def test_repo_default_yaml_matches_code_defaults() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    assert load_config(path) == NeuralConfig()


@pytest.mark.parametrize(
    "patch",
    [
        {"model": {"experts": ["nope"]}},
        {"model": {"experts": []}},
        {"model": {"d_model": 30, "transformer": {"heads": 4}}},
        {"model": {"graph": {"enabled": True}}},
        {"targets": {"horizons": []}},
        {"targets": {"quantiles": [0.9, 0.1]}},
        {"loss": {"task_weights": {"bogus": 1.0}}},
        {"ensemble": {"size": 2, "embedding_member": 3}},
        {"unknown_section": {}},
    ],
)
def test_invalid_configs_rejected(patch: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        NeuralConfig.model_validate(patch)


def test_observation_validation() -> None:
    with pytest.raises(ValidationError):
        NeuralObservation(observation_id="a", timestamp=0, current_features=[[1.0]])
    with pytest.raises(ValidationError):
        SequenceInput(values=np.zeros((3, 2)), time_deltas=np.zeros(4))
    with pytest.raises(ValidationError):
        GraphInput(node_features=np.zeros((2, 3)), edge_index=[[0, 5], [1, 0]])
    g = GraphInput(node_features=np.zeros((2, 3)), edge_index=[[0], [1]], edge_type=[1])
    assert g.edge_index.shape == (2, 1)


def test_observation_dict_roundtrip() -> None:
    obs = NeuralObservation(
        observation_id="x",
        timestamp=12.5,
        current_features=[1.0, 2.0],
        sequences={"fast": SequenceInput(values=[[1.0, 2.0], [3.0, 4.0]], mask=[True, False])},
        graph=GraphInput(node_features=[[1.0], [2.0]], edge_index=[[1], [0]]),
        metadata={"source": "test"},
    )
    back = NeuralObservation.from_dict(obs.to_dict())
    assert back.observation_id == "x"
    np.testing.assert_array_equal(back.sequences["fast"].values, obs.sequences["fast"].values)
    assert back.graph is not None and back.graph.target_node == 0


@settings(max_examples=40, deadline=None)
@given(length=st.integers(0, 40), max_len=st.integers(1, 20))
def test_pad_sequence_left_pads_and_truncates(length: int, max_len: int) -> None:
    ts = TimescaleConfig(name="f", feature_dim=3, max_len=max_len, resolution_seconds=2.0)
    vals = np.arange(length * 3, dtype=np.float32).reshape(length, 3) + 1
    v, m, d = pad_sequence(vals, ts)
    keep = min(length, max_len)
    assert v.shape == (max_len, 3) and m.sum() == keep
    if keep:
        np.testing.assert_array_equal(v[-keep:], vals[-keep:])  # most recent rows kept, at the end
        assert d[-1] == 0.0 and np.all(np.diff(d[-keep:]) < 0)
    assert np.all(v[~m] == 0)


def test_pad_sequence_marks_nan_rows_missing() -> None:
    ts = TimescaleConfig(name="f", feature_dim=2, max_len=4, resolution_seconds=1.0)
    vals = np.array([[1, 2], [np.nan, np.nan], [3, np.nan]], dtype=np.float32)
    _, m, _ = pad_sequence(vals, ts)
    assert m.tolist() == [False, True, False, True]


def test_build_bars_is_causal() -> None:
    ts = TimescaleConfig(name="b", feature_dim=1, max_len=5, resolution_seconds=10.0)
    times = np.array([1.0, 12.0, 25.0, 41.0, 49.0, 55.0, 70.0])
    vals = np.arange(len(times), dtype=np.float32)[:, None]
    seq = build_bars(times, vals, observation_time=50.0, ts=ts)
    # events after t=50 (55, 70) must never influence any bar
    seq_future_changed = build_bars(times, np.where(times > 50, 999.0, vals[:, 0])[:, None], 50.0, ts)
    np.testing.assert_array_equal(seq.values, seq_future_changed.values)
    assert seq.mask is not None and seq.mask.tolist() == [True, True, True, False, True]
    assert seq.values[-1, 0] == pytest.approx(3.5)  # mean of events at 41 and 49


def test_observation_array_roundtrip(tiny_config: NeuralConfig) -> None:
    rng = np.random.default_rng(0)
    obs = []
    for i in range(4):
        seqs = {
            "fast": SequenceInput(values=rng.normal(size=(5 + i, 6))),
            "medium": SequenceInput(values=rng.normal(size=(20, 6))),
        }
        obs.append(
            NeuralObservation(
                observation_id=f"o{i}",
                timestamp=float(i),
                current_features=rng.normal(size=16),
                sequences=seqs,
            )
        )
    arrays = observations_to_arrays(obs, tiny_config)
    assert arrays[seq_key("fast", "mask")].sum(axis=1).tolist() == [5, 6, 7, 8]
    assert arrays[seq_key("medium", "mask")].sum(axis=1).tolist() == [12] * 4  # truncated to max_len
    assert not arrays[seq_key("slow", "mask")].any()  # missing timescale
    back = arrays_to_observations(arrays, tiny_config)
    np.testing.assert_allclose(back[2].sequences["fast"].values, obs[2].sequences["fast"].values, rtol=1e-6)
    assert "slow" not in back[0].sequences


def test_observation_dimension_errors(tiny_config: NeuralConfig) -> None:
    bad = NeuralObservation(observation_id="a", timestamp=0, current_features=np.zeros(3))
    with pytest.raises(ValueError, match="current_features"):
        observations_to_arrays([bad], tiny_config)
    bad_seq = NeuralObservation(
        observation_id="a",
        timestamp=0,
        current_features=np.zeros(16),
        sequences={"unknown": SequenceInput(values=np.zeros((2, 6)))},
    )
    with pytest.raises(ValueError, match="unknown timescales"):
        observations_to_arrays([bad_seq], tiny_config)


def test_outcomes_partial_horizons_masked(tiny_config: NeuralConfig) -> None:
    oc = NeuralOutcome(
        observation_id="a",
        returns={"30s": 0.01, "2m": 0.02},
        max_upside={"30s": 0.02, "2m": 0.03},
        max_drawdown={"30s": 0.0, "2m": 0.01},
        volatility={"30s": 0.01, "2m": 0.01},
    )
    arr = outcomes_to_arrays([oc], tiny_config)
    assert arr["target_mask"].tolist() == [[True, True, False]]
