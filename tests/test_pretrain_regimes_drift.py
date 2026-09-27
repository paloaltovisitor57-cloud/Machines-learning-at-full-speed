from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import torch

from nardis_neural.config import DriftConfig, NeuralConfig, RegimeConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import ArrayStore, select_rows, target_key
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.monitoring.drift import (
    drift_report,
    error_drift,
    input_drift,
    population_stability_index,
    univariate_drift,
)
from nardis_neural.regimes.clustering import RegimeClusterer, cluster_statistics
from nardis_neural.regimes.embeddings import EmbeddingTable, extract_embeddings
from nardis_neural.synthetic import SyntheticSpec, generate_synthetic
from nardis_neural.training.pipeline import load_pretrained_encoders, train_engine
from nardis_neural.training.pretraining import (
    load_pretrained,
    nt_xent,
    pretrain_encoders,
    save_pretrained,
)


# ---------------------------------------------------------------- self-supervised pretraining
def _unlabelled(cfg: NeuralConfig, n: int = 600) -> ArrayStore:
    arrays = generate_synthetic(cfg, SyntheticSpec(n_observations=n, seed=9))
    return ArrayStore({k: v for k, v in arrays.items() if not k.startswith("target")})


def test_nt_xent_prefers_aligned_views() -> None:
    z = torch.randn(16, 8)
    assert nt_xent(z, z + 0.01 * torch.randn_like(z), 0.2) < nt_xent(z, torch.randn(16, 8), 0.2)


def test_pretraining_runs_on_unlabelled_data_and_transfers(tmp_path: Path, tiny_config: NeuralConfig) -> None:
    store = _unlabelled(tiny_config)
    assert not store.has_targets
    norm = FeatureNormalizer.fit(store, np.arange(len(store)), tiny_config)
    ds = MarketDataset(store, tiny_config)
    tiny_config.pretrain.epochs = 3
    model, result = pretrain_encoders(
        tiny_config, ds, norm, device=torch.device("cpu"), max_batches_per_epoch=6
    )
    hist = result.history
    assert len(hist) == 3
    for key in ("transformer.masked_timestep", "recurrent.masked_feature", "tcn.contrastive"):
        assert key in hist[0], key
    assert hist[-1]["loss"] < hist[0]["loss"], "self-supervised loss must decrease"
    save_pretrained(tmp_path / "pre.pt", model, result, tiny_config)
    state = load_pretrained(tmp_path / "pre.pt")
    assert state and all(k.startswith("experts.") for k in state)
    fresh = NardisNeuralNetwork(tiny_config)
    n = load_pretrained_encoders(fresh, state)
    assert n == len(state)
    key = next(iter(state))
    assert torch.equal(fresh.state_dict()[key], state[key])


def test_pretraining_subset_of_tasks(tiny_config: NeuralConfig) -> None:
    tiny_config.pretrain.tasks = ["contrastive"]
    tiny_config.model.experts = ["tcn", "tabular"]
    store = _unlabelled(tiny_config, 200)
    norm = FeatureNormalizer.fit(store, np.arange(len(store)), tiny_config)
    _, res = pretrain_encoders(
        tiny_config,
        MarketDataset(store, tiny_config),
        norm,
        device=torch.device("cpu"),
        epochs=1,
        max_batches_per_epoch=2,
    )
    assert set(res.history[0]) == {"epoch", "loss", "tcn.contrastive"}
    tiny_config.model.experts = ["tabular"]
    with pytest.raises(ValueError):
        pretrain_encoders(tiny_config, MarketDataset(store, tiny_config), norm, device=torch.device("cpu"))


def test_supervised_training_from_pretrained_state(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    store = _unlabelled(tiny_config, 300)
    norm = FeatureNormalizer.fit(store, np.arange(len(store)), tiny_config)
    model, _ = pretrain_encoders(
        tiny_config,
        MarketDataset(store, tiny_config),
        norm,
        device=torch.device("cpu"),
        epochs=1,
        max_batches_per_epoch=2,
    )
    state = {k: v for k, v in model.state_dict().items() if k.startswith("experts.")}
    tiny_config.training.epochs = 1
    tiny_config.ensemble.size = 1
    tiny_config.training.max_train_batches_per_epoch = 3
    engine, _ = train_engine(tiny_config, base_store, pretrained_state=state, device=torch.device("cpu"))
    assert engine.ensemble.size == 1


# ---------------------------------------------------------------- embeddings & regimes
def test_embedding_extraction_and_persistence(
    tmp_path: Path, trained_engine: NeuralEngine, later_arrays: dict[str, Any]
) -> None:
    ds = MarketDataset(ArrayStore(select_rows(later_arrays, np.arange(300))), trained_engine.config)
    table = extract_embeddings(trained_engine, ds)
    assert len(table) == 300 and table.embeddings.shape == (300, trained_engine.config.model.latent_dim)
    assert (
        table.expert_weights.shape == (300, 5) and "return" in table.targets and "true_regime" in table.extra
    )
    table.save(tmp_path / "e.parquet")
    df = pl.read_parquet(tmp_path / "e.parquet")
    assert (
        df.height == 300
        and "emb_0" in df.columns
        and "gate_tcn" in df.columns
        and "return.mean.30s" in df.columns
    )
    table.save(tmp_path / "e.npz")
    back = EmbeddingTable.load_npz(tmp_path / "e.npz")
    np.testing.assert_array_equal(back.embeddings, table.embeddings)
    assert back.expert_names == table.expert_names
    with pytest.raises(ValueError):
        table.save(tmp_path / "e.csv")


@pytest.mark.parametrize("method", ["kmeans", "gmm", "hdbscan"])
def test_regime_clustering_methods(method: str) -> None:
    rng = np.random.default_rng(0)
    centers = np.array([[0.0, 0, 0], [8, 8, 8], [-8, 8, 0]])
    x = np.concatenate([c + rng.normal(size=(150, 3)) for c in centers])
    truth = np.repeat(np.arange(3), 150)
    cfg = RegimeConfig(method=method, n_clusters=3, hdbscan_min_cluster_size=20)
    model, labels = RegimeClusterer.fit(x, cfg)
    valid = labels >= 0
    assert valid.mean() > 0.8
    # clusters must recover the planted structure (up to label permutation)
    for k in np.unique(labels[valid]):
        assert np.bincount(truth[labels == k]).max() / (labels == k).sum() > 0.9
    back = RegimeClusterer.from_dict(model.to_dict())
    np.testing.assert_array_equal(back.predict(x), model.predict(x))


def test_regime_auto_selection_and_stats() -> None:
    rng = np.random.default_rng(1)
    x = np.concatenate([c + rng.normal(size=(100, 2)) * 0.3 for c in ([0, 0], [5, 5], [0, 5], [5, 0])])
    model, labels = RegimeClusterer.fit(x, RegimeConfig(method="kmeans", auto_select=True, k_min=2, k_max=6))
    assert model.n_clusters == 4 and len(model.selection_scores) == 5
    gmm, _ = RegimeClusterer.fit(x, RegimeConfig(method="gmm", auto_select=True, k_min=2, k_max=6))
    assert gmm.n_clusters == 4
    w = rng.dirichlet(np.ones(3), size=len(x))
    pred = rng.normal(size=(len(x), 2))
    stats = cluster_statistics(
        labels,
        w,
        ["a", "b", "c"],
        ["h1", "h2"],
        pred,
        pred + 0.1,
        np.arange(len(x), dtype=float),
        extra={"confidence": np.ones(len(x))},
    )
    assert len(stats) == 4
    s0 = next(iter(stats.values()))
    assert set(s0["expert_weights"]) == {"a", "b", "c"} and s0["return_mae"]["h1"] == pytest.approx(0.1)
    assert s0["dominant_expert"] in {"a", "b", "c"} and "first_seen" in s0 and s0["confidence"] == 1.0


def test_engine_regimes_align_with_hidden_regimes(
    trained_engine: NeuralEngine, later_arrays: dict[str, Any]
) -> None:
    ds = MarketDataset(ArrayStore(select_rows(later_arrays, np.arange(600))), trained_engine.config)
    table = extract_embeddings(trained_engine, ds)
    labels = table.predictions["regime"]
    truth = table.extra["true_regime"]
    # learned clusters are not hand-defined but should carry information about the hidden regime
    from sklearn.metrics import adjusted_mutual_info_score

    assert adjusted_mutual_info_score(truth, labels) > 0.02


# ---------------------------------------------------------------- drift
def test_univariate_drift_statistics() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=5000), rng.normal(size=5000)
    assert population_stability_index(a, b) < 0.02
    assert population_stability_index(a, b + 1.0) > 0.3
    cfg = DriftConfig()
    same = univariate_drift(a, b, "x", cfg)
    moved = univariate_drift(a, b * 2 + 1, "x", cfg)
    assert not same.drifted and moved.drifted
    assert moved.mean_shift == pytest.approx(1.0, abs=0.1) and moved.std_ratio == pytest.approx(2.0, abs=0.1)
    const = population_stability_index(np.zeros(100), np.ones(100))
    assert const > 1.0
    err = error_drift(np.abs(a) * 0.1, np.abs(b) * 0.2, cfg)
    assert err.drifted and err.stats["ratio"] == pytest.approx(2.0, rel=0.1)


def test_input_drift_on_synthetic_shift(tiny_config: NeuralConfig) -> None:
    ref = generate_synthetic(tiny_config, SyntheticSpec(n_observations=800, seed=1))
    same = generate_synthetic(tiny_config, SyntheticSpec(n_observations=800, seed=2))
    shifted = generate_synthetic(tiny_config, SyntheticSpec(n_observations=800, seed=2, drift_shift=4.0))
    assert not input_drift(ref, same, tiny_config).drifted
    rep = input_drift(ref, shifted, tiny_config)
    assert rep.drifted and rep.fraction_drifted > 0.25


def test_full_drift_report(trained_engine: NeuralEngine, base_store: ArrayStore) -> None:
    cfg = trained_engine.config
    ref = MarketDataset(base_store, cfg, np.arange(600))
    shifted_arrays = generate_synthetic(cfg, SyntheticSpec(n_observations=600, seed=7, drift_shift=5.0))
    cur = MarketDataset(ArrayStore(shifted_arrays), cfg)
    rep = drift_report(trained_engine, ref, cur)
    assert rep.input.drifted
    assert rep.embedding is not None and rep.embedding.stats["mean_distance_ratio"] > 1.0
    assert rep.prediction is not None and rep.prediction.features
    assert rep.error is not None and rep.error.stats["ratio"] > 1.0
    assert rep.any_drift and rep.summary()["input"] is True
    same = drift_report(trained_engine, ref, MarketDataset(base_store, cfg, np.arange(600, 1200)))
    assert same.input.fraction_drifted < rep.input.fraction_drifted
    assert np.isfinite(shifted_arrays[target_key("return")]).all()
