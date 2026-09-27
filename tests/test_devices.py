"""Device coverage: CPU always; CUDA and Apple MPS when present (skipped otherwise)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import ArrayStore
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.training.pipeline import build_objective
from nardis_neural.training.trainer import Trainer, amp_settings, autocast_context

DEVICES = [
    pytest.param("cpu", id="cpu"),
    pytest.param(
        "cuda",
        id="cuda",
        marks=[
            pytest.mark.cuda,
            pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
        ],
    ),
    pytest.param(
        "mps",
        id="mps",
        marks=[
            pytest.mark.mps,
            pytest.mark.skipif(
                not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                reason="MPS unavailable",
            ),
        ],
    ),
]


@pytest.mark.parametrize("device_name", DEVICES)
def test_forward_backward_train_on_device(
    device_name: str, tiny_config: NeuralConfig, base_store: ArrayStore
) -> None:
    device = torch.device(device_name)
    norm = FeatureNormalizer.fit(base_store, np.arange(500), tiny_config)
    ds = MarketDataset(base_store, tiny_config, np.arange(500))
    model = NardisNeuralNetwork(tiny_config).to(device)
    batch = norm.transform_batch(ds[np.arange(16)].to(device))
    enabled, dtype, _ = amp_settings(device, tiny_config.training.mixed_precision)
    with autocast_context(device, enabled, dtype):
        out = model(batch)
    loss = sum(v.float().mean() for v in out.means.values())
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    tiny_config.training.max_train_batches_per_epoch = 2
    res = Trainer(tiny_config, device).fit(
        NardisNeuralNetwork(tiny_config),
        norm,
        ds,
        MarketDataset(base_store, tiny_config, np.arange(600, 700)),
        build_objective(tiny_config, ds),
        epochs=1,
    )
    assert np.isfinite(res.best_val_loss)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_mixed_precision_matches_fp32(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    device = torch.device("cuda")
    norm = FeatureNormalizer.fit(base_store, np.arange(500), tiny_config)
    batch = norm.transform_batch(MarketDataset(base_store, tiny_config)[np.arange(32)].to(device))
    model = NardisNeuralNetwork(tiny_config).to(device).eval()
    ref = model(batch).means["return"]
    enabled, dtype, _ = amp_settings(device, "auto")
    with autocast_context(device, enabled, dtype):
        amp = model(batch).means["return"]
    assert torch.allclose(ref, amp.float(), atol=5e-2)
