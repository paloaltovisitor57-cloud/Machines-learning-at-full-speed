"""Shared fixtures: a tiny but complete configuration, synthetic data and a trained engine."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nardis_neural.config import NeuralConfig, TimescaleConfig
from nardis_neural.data.datasets import Batch, MarketDataset
from nardis_neural.data.loaders import ArrayStore, select_rows
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.synthetic import SyntheticSpec, generate_synthetic
from nardis_neural.training.continual import ContinualLearner
from nardis_neural.training.pipeline import train_engine

torch.set_num_threads(2)


def make_tiny_config(graph: bool = False) -> NeuralConfig:
    c = NeuralConfig()
    c.features.timescales = [
        TimescaleConfig(name="fast", feature_dim=6, max_len=16, resolution_seconds=1),
        TimescaleConfig(name="medium", feature_dim=6, max_len=12, resolution_seconds=5),
        TimescaleConfig(name="slow", feature_dim=7, max_len=8, resolution_seconds=30),
    ]
    m = c.model
    m.d_model = 16
    m.latent_dim = 16
    m.head_hidden_dim = 16
    m.transformer.layers = 1
    m.transformer.heads = 2
    m.recurrent.hidden_dim = 16
    m.tcn.channels = [16, 16]
    m.ssm.state_dim = 8
    m.ssm.layers = 1
    m.tabular.width = 32
    m.tabular.depth = 1
    m.graph.hidden_dim = 16
    if graph:
        m.experts = ["transformer", "recurrent", "tcn", "ssm", "tabular", "graph"]
        m.graph.enabled = True
    c.ensemble.size = 2
    c.ensemble.mc_dropout_samples = 1
    c.training.epochs = 3
    c.training.batch_size = 64
    c.training.learning_rate = 3e-3
    c.training.device = "cpu"
    c.replay.rare_return_threshold = 0.15
    c.replay.recent_capacity = 400
    c.replay.historical_capacity = 400
    c.replay.rare_capacity = 100
    c.continual.min_new_samples = 150
    c.continual.adapt_samples = 400
    c.continual.adapt_epochs = 2
    c.continual.full_retrain_min_new_samples = 300
    c.continual.ewc_fisher_batches = 3
    c.promotion.min_observations = 50
    c.promotion.min_regime_observations = 10
    c.regimes.n_clusters = 3
    return c


@pytest.fixture
def tiny_config() -> NeuralConfig:
    return make_tiny_config()


@pytest.fixture(scope="session")
def session_config() -> NeuralConfig:
    return make_tiny_config()


@pytest.fixture(scope="session")
def synthetic_arrays(session_config: NeuralConfig) -> dict[str, Any]:
    return generate_synthetic(session_config, SyntheticSpec(n_observations=2400, seed=3))


@pytest.fixture(scope="session")
def base_store(synthetic_arrays: dict[str, Any]) -> ArrayStore:
    return ArrayStore(select_rows(synthetic_arrays, np.arange(1200)))


@pytest.fixture(scope="session")
def later_arrays(synthetic_arrays: dict[str, Any]) -> dict[str, Any]:
    return select_rows(synthetic_arrays, np.arange(1200, 2400))


@pytest.fixture(scope="session")
def trained_engine(session_config: NeuralConfig, base_store: ArrayStore) -> NeuralEngine:
    engine, _ = train_engine(session_config, base_store, device=torch.device("cpu"))
    return engine


@pytest.fixture
def normalizer(tiny_config: NeuralConfig, base_store: ArrayStore) -> FeatureNormalizer:
    return FeatureNormalizer.fit(base_store, np.arange(900), tiny_config)


@pytest.fixture
def norm_batch(tiny_config: NeuralConfig, base_store: ArrayStore, normalizer: FeatureNormalizer) -> Batch:
    ds = MarketDataset(base_store, tiny_config)
    return normalizer.transform_batch(ds[np.arange(32)])


@pytest.fixture(scope="session")
def workspace_template(tmp_path_factory: pytest.TempPathFactory, trained_engine: NeuralEngine) -> Path:
    root = tmp_path_factory.mktemp("ws_template") / "ws"
    ContinualLearner.initialize(root, trained_engine, trained_engine.config, device="cpu")
    return root


@pytest.fixture
def workspace(tmp_path: Path, workspace_template: Path) -> Iterator[Path]:
    dst = tmp_path / "ws"
    shutil.copytree(workspace_template, dst)
    yield dst
