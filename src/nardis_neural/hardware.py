"""Hardware detection and compute profiles.

The same code runs on a laptop CPU and on a large GPU; only the model's width, depth,
ensemble size and MC-dropout budget change.  ``auto`` picks a profile from the detected
hardware:

=============  ==============================================  ======================
profile        model                                           for
=============  ==============================================  ======================
cpu-lite       d=32, 1-layer experts, 2 members, no MC         2–4 core laptops
cpu            d=48, 2-layer experts, 3 members, 1 MC pass     ≥ 8 cores
gpu            d=128, 3-layer experts, 5 members, 2 MC passes  consumer GPUs / Apple
gpu-frontier   d=256, 6-layer transformer, 4-layer SSM,        ≥ 16 GB VRAM
               5 members, 4 MC passes, bf16/fp16
=============  ==============================================  ======================
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch

from nardis_neural.config import NeuralConfig

PROFILES = ("cpu-lite", "cpu", "gpu", "gpu-frontier")


@dataclass(frozen=True)
class HardwareInfo:
    device: str
    cpu_cores: int
    gpu_name: str | None
    gpu_memory_gb: float | None
    recommended_profile: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect() -> HardwareInfo:
    cores = os.cpu_count() or 1
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        mem = props.total_memory / 2**30
        return HardwareInfo("cuda", cores, props.name, round(mem, 1), "gpu-frontier" if mem >= 16 else "gpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return HardwareInfo("mps", cores, "Apple GPU", None, "gpu")
    return HardwareInfo("cpu", cores, None, None, "cpu" if cores >= 8 else "cpu-lite")


def _widths(cfg: NeuralConfig, d: int, layers: int, heads: int, ssm_layers: int, ssm_state: int) -> None:
    m = cfg.model
    m.d_model = d
    m.latent_dim = d
    m.head_hidden_dim = d
    m.transformer.layers = layers
    m.transformer.heads = heads
    m.recurrent.hidden_dim = d
    m.recurrent.layers = 1 if layers < 3 else 2
    m.tcn.channels = [d] * (2 if layers < 2 else layers + 1)
    m.ssm.layers = ssm_layers
    m.ssm.state_dim = ssm_state
    m.tabular.width = 2 * d
    m.tabular.depth = 2 if layers < 3 else 3
    m.gating.hidden_dim = d


def apply_profile(cfg: NeuralConfig, profile: str = "auto", hw: HardwareInfo | None = None) -> NeuralConfig:
    """Return a copy of ``cfg`` scaled to ``profile`` (``auto`` = detected hardware)."""
    hw = hw or detect()
    name = hw.recommended_profile if profile == "auto" else profile
    if name not in PROFILES:
        raise ValueError(f"unknown profile {name!r}; choose from {PROFILES} or 'auto'")
    out = cfg.model_copy(deep=True)
    t, e = out.training, out.ensemble
    if name == "cpu-lite":
        _widths(out, 32, 1, 4, 1, 8)
        e.size, e.mc_dropout_samples, t.batch_size = 2, 0, 128
    elif name == "cpu":
        _widths(out, 48, 2, 4, 1, 16)
        e.size, e.mc_dropout_samples, t.batch_size = 3, 1, 128
    elif name == "gpu":
        _widths(out, 128, 3, 8, 2, 16)
        e.size, e.mc_dropout_samples, t.batch_size = 5, 2, 256
    else:
        _widths(out, 256, 6, 8, 4, 32)
        e.size, e.mc_dropout_samples, t.batch_size = 5, 4, 512
    e.embedding_member = min(e.embedding_member, e.size - 1)
    t.device = hw.device if name.startswith("gpu") else "cpu"
    t.mixed_precision = "auto"
    if not name.startswith("gpu"):
        torch.set_num_threads(max(1, hw.cpu_cores))
    return NeuralConfig.model_validate(out.model_dump())
