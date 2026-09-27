from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import (
    BatchIndexSampler,
    MarketDataset,
    compute_sampling_weights,
    make_loader,
)
from nardis_neural.data.loaders import (
    ArrayStore,
    concat_arrays,
    fingerprint_arrays,
    load_store,
    select_rows,
    seq_key,
    target_key,
)
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.synthetic import SyntheticSpec, generate_synthetic
from tests.conftest import make_tiny_config


def test_synthetic_generator_structure(
    synthetic_arrays: dict[str, Any], session_config: NeuralConfig
) -> None:
    store = ArrayStore(synthetic_arrays)
    store.validate(session_config)
    ts = store.timestamps
    assert np.all(np.diff(ts) >= 0), "synthetic data must be time ordered"
    assert set(np.unique(synthetic_arrays["regime"]).tolist()) == {0, 1, 2, 3}
    slow_avail = synthetic_arrays[seq_key("slow", "mask")].any(axis=1)
    assert 0 < slow_avail.mean() < 1, "some observations must lack the slow sequence"
    assert np.isnan(synthetic_arrays["current"]).any(), "missing current features are simulated"
    assert (synthetic_arrays[target_key("max_drawdown")] >= 0).all()
    assert (synthetic_arrays[target_key("max_upside")] >= synthetic_arrays[target_key("return")] - 1e-6).all()


def test_synthetic_is_reproducible(tiny_config: NeuralConfig) -> None:
    a = generate_synthetic(tiny_config, SyntheticSpec(n_observations=50, seed=5))
    b = generate_synthetic(tiny_config, SyntheticSpec(n_observations=50, seed=5))
    assert fingerprint_arrays(a) == fingerprint_arrays(b)


@pytest.mark.parametrize("fmt", ["npy", "npz", "pt", "parquet"])
def test_store_formats_roundtrip(tmp_path: Path, fmt: str, tiny_config: NeuralConfig) -> None:
    arrays = generate_synthetic(tiny_config, SyntheticSpec(n_observations=60, seed=1))
    store = ArrayStore(arrays)
    path = tmp_path / {"npy": "ds", "npz": "ds.npz", "pt": "ds.pt", "parquet": "ds.parquet"}[fmt]
    if fmt == "npy":
        store.save(path)
    elif fmt == "npz":
        store.save_npz(path)
    elif fmt == "pt":
        store.save_torch(path)
    else:
        store.to_parquet(path, tiny_config, row_group_size=16)
    loaded = load_store(path, tiny_config, cache_dir=tmp_path / "cache" if fmt == "parquet" else None)
    assert len(loaded) == 60
    for key in ("current", seq_key("fast", "values"), seq_key("fast", "mask"), target_key("return")):
        np.testing.assert_allclose(
            np.nan_to_num(np.asarray(loaded[key], dtype=np.float64)),
            np.nan_to_num(np.asarray(arrays[key], dtype=np.float64)),
            rtol=1e-6,
        )
    np.testing.assert_array_equal(np.asarray(loaded["observation_id"]), arrays["observation_id"])
    if fmt in ("npy", "parquet"):
        assert isinstance(loaded["current"], np.memmap), "large datasets are memory-mapped, not loaded"


def test_parquet_variable_length_sequences(tmp_path: Path, tiny_config: NeuralConfig) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = {
        "observation_id": ["a", "b"],
        "timestamp": [1.0, 2.0],
        "current": [[0.0] * 16, [1.0] * 16],
        "seq_fast_values": [[[1.0] * 6] * 3, [[2.0] * 6] * 40],
    }
    pq.write_table(pa.table(rows), tmp_path / "v.parquet")
    store = load_store(tmp_path / "v.parquet", tiny_config)
    assert store[seq_key("fast", "mask")].sum(axis=1).tolist() == [3, 16]
    assert not store.has_targets


def test_graph_ragged_select_and_concat(tiny_config: NeuralConfig) -> None:
    cfg = make_tiny_config(graph=True)
    arrays = generate_synthetic(cfg, SyntheticSpec(n_observations=30, seed=2, graph=True))
    sel = select_rows(arrays, np.array([5, 1, 7]))
    offs = sel["graph.node_offsets"]
    assert len(offs) == 4 and offs[-1] == len(sel["graph.node_features"])
    for j, i in enumerate([5, 1, 7]):
        a, b = arrays["graph.node_offsets"][i], arrays["graph.node_offsets"][i + 1]
        np.testing.assert_array_equal(
            sel["graph.node_features"][offs[j] : offs[j + 1]], arrays["graph.node_features"][a:b]
        )
    both = concat_arrays([sel, sel])
    assert both["graph.node_offsets"][-1] == 2 * offs[-1]
    ds = MarketDataset(ArrayStore(arrays), cfg)
    batch = ds[np.arange(10)]
    assert batch.graph is not None
    assert batch.graph.edge_index.max() < batch.graph.node_features.shape[0]
    ok = batch.graph.target_node >= 0
    assert (batch.graph.node_batch[batch.graph.target_node[ok]] == torch.arange(10)[ok]).all()


def test_dataset_batches_and_masks(base_store: ArrayStore, tiny_config: NeuralConfig) -> None:
    ds = MarketDataset(base_store, tiny_config, np.arange(100))
    batch = ds[np.array([3, 1, 2])]
    assert batch.size == 3 and batch.observation_ids[0] == str(base_store["observation_id"][3])
    assert batch.sequences["fast"].values.shape == (3, 16, 6)
    assert batch.targets is not None
    labels = batch.targets.labels["upside"]
    ret = batch.targets.regression["return"]
    thr = torch.tensor([h.upside_threshold for h in tiny_config.targets.horizons])
    assert torch.equal(labels, (ret > thr).float())
    loader = make_loader(ds, 32, shuffle=True, seed=1)
    seen = np.concatenate([np.array(b.observation_ids) for b in loader])
    assert len(seen) == 100 and len(set(seen)) == 100


def test_sampler_weighted_and_max_batches() -> None:
    w = np.zeros(10)
    w[3] = 1.0
    s = BatchIndexSampler(10, 5, weights=w, seed=0)
    assert all((chunk == 3).all() for chunk in s)
    assert len(BatchIndexSampler(100, 10, max_batches=3)) == 3
    assert len(list(BatchIndexSampler(100, 10, max_batches=3))) == 3


def test_sampling_weights_balanced_and_rare(base_store: ArrayStore, tiny_config: NeuralConfig) -> None:
    ds = MarketDataset(base_store, tiny_config)
    assert compute_sampling_weights(ds, tiny_config) is None
    tiny_config.training.balanced_sampling = True
    tiny_config.training.rare_event_oversampling = 3.0
    w = compute_sampling_weights(ds, tiny_config)
    assert w is not None and w.shape == (len(ds),) and w.max() > w.min()


def test_normalizer_uses_training_rows_only(
    base_store: ArrayStore, tiny_config: NeuralConfig, tmp_path: Path
) -> None:
    train = np.arange(600)
    norm = FeatureNormalizer.fit(base_store, train, tiny_config)
    cur = np.asarray(base_store["current"], dtype=np.float64)
    np.testing.assert_allclose(norm.current_center, np.nanmedian(cur[train], axis=0), rtol=1e-6)
    # corrupting non-training rows must not change the fitted statistics
    arrays = dict(base_store.arrays)
    arrays["current"] = np.asarray(arrays["current"]).copy()
    arrays["current"][600:] = 1e6
    norm2 = FeatureNormalizer.fit(ArrayStore(arrays), train, tiny_config)
    np.testing.assert_allclose(norm.current_center, norm2.current_center)
    np.testing.assert_allclose(norm.target_scale["return"], norm2.target_scale["return"])
    # persistence
    back = FeatureNormalizer.from_dict(norm.to_dict())
    ds = MarketDataset(base_store, tiny_config)
    b1, b2 = norm.transform_batch(ds[np.arange(8)]), back.transform_batch(ds[np.arange(8)])
    assert torch.allclose(b1.current, b2.current)
    assert b1.normalized and torch.isfinite(b1.current).all()
    assert (b1.current.abs() <= tiny_config.normalization.clip).all()
    assert b1.targets is not None and b1.targets.regression["max_drawdown"].min() >= 0
    mean, var = norm.denormalize(
        "return", b1.targets.regression["return"], torch.ones_like(b1.targets.regression["return"])
    )
    assert var is not None
    raw = ds[np.arange(8)].targets
    assert raw is not None
    assert torch.allclose(mean, raw.regression["return"], atol=1e-6)


def test_standard_normalization(base_store: ArrayStore, tiny_config: NeuralConfig) -> None:
    tiny_config.normalization.method = "standard"
    norm = FeatureNormalizer.fit(base_store, np.arange(500), tiny_config)
    cur = np.asarray(base_store["current"][:500], dtype=np.float64)
    np.testing.assert_allclose(norm.current_center, np.nanmean(cur, axis=0), rtol=1e-6)
