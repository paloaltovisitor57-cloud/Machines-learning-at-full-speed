"""Inference latency / throughput / memory measurement."""

from __future__ import annotations

import resource
import sys
import time
from typing import Any

import numpy as np
import torch

from nardis_neural.data.datasets import arrays_to_batch
from nardis_neural.data.loaders import select_rows
from nardis_neural.data.sequences import observations_to_arrays
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.schemas import NeuralObservation


def _rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru / (1024 * 1024) if sys.platform == "darwin" else ru / 1024


def benchmark_engine(
    engine: NeuralEngine,
    observations: list[NeuralObservation],
    batch_sizes: tuple[int, ...] = (1, 32, 256),
    repeats: int = 5,
    mc_samples: int | None = None,
) -> dict[str, Any]:
    """Measure end-to-end latency for single observations and throughput for batches."""
    if not observations:
        raise ValueError("need observations to benchmark")
    arrays = observations_to_arrays(observations, engine.config)
    n = len(observations)
    if engine.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    engine.predict(observations[0])  # warm-up
    single = []
    for i in range(min(repeats * 4, n)):
        t0 = time.perf_counter()
        engine.predict(observations[i])
        single.append(time.perf_counter() - t0)
    results: dict[str, Any] = {
        "device": str(engine.device),
        "ensemble_size": engine.ensemble.size,
        "mc_dropout_samples": engine.config.ensemble.mc_dropout_samples if mc_samples is None else mc_samples,
        "parameters_per_member": engine.ensemble.member(0).parameter_count(),
        "single_observation_ms": {
            "p50": float(np.percentile(single, 50) * 1e3),
            "p95": float(np.percentile(single, 95) * 1e3),
            "mean": float(np.mean(single) * 1e3),
        },
        "batches": {},
    }
    for bs in batch_sizes:
        idx = np.arange(bs) % n
        batch = arrays_to_batch(select_rows(arrays, idx), engine.config)
        engine.forward_arrays(batch, mc_samples)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            engine.forward_arrays(batch, mc_samples)
            if engine.device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        med = float(np.median(times))
        results["batches"][str(bs)] = {"latency_ms": med * 1e3, "throughput_obs_per_s": bs / med}
    results["peak_rss_mb"] = _rss_mb()
    if engine.device.type == "cuda":
        results["cuda_peak_mb"] = torch.cuda.max_memory_allocated() / 2**20
    return results
