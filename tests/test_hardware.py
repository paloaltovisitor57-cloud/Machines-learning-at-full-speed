"""Hardware profiles: one code path from laptop CPU to large GPU."""

from __future__ import annotations

import pytest
import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.hardware import PROFILES, HardwareInfo, apply_profile, detect
from nardis_neural.models.main import NardisNeuralNetwork


def _params(cfg: NeuralConfig) -> int:
    return sum(p.numel() for p in NardisNeuralNetwork(cfg).parameters())


def test_profiles_scale_monotonically_and_stay_valid() -> None:
    threads = torch.get_num_threads()
    hw = HardwareInfo("cpu", 4, None, None, "cpu-lite")
    try:
        sizes = [_params(apply_profile(NeuralConfig(), p, hw)) for p in PROFILES]
    finally:
        torch.set_num_threads(threads)
    assert sizes == sorted(sizes) and sizes[-1] > 20 * sizes[0]
    lite = apply_profile(NeuralConfig(), "cpu-lite", hw)
    assert lite.training.device == "cpu" and lite.ensemble.mc_dropout_samples == 0
    frontier = apply_profile(
        NeuralConfig(), "gpu-frontier", HardwareInfo("cuda", 16, "x", 24.0, "gpu-frontier")
    )
    assert frontier.training.device == "cuda" and frontier.model.ssm.layers == 4
    assert apply_profile(NeuralConfig(), "auto", hw).model.d_model == lite.model.d_model
    with pytest.raises(ValueError):
        apply_profile(NeuralConfig(), "supercomputer", hw)


def test_detect_reports_a_known_profile() -> None:
    hw = detect()
    assert hw.recommended_profile in PROFILES and hw.cpu_cores >= 1
    assert set(hw.to_dict()) >= {"device", "recommended_profile"}
