"""Candidate creation with strict isolation from the champion."""

from __future__ import annotations

from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.lifecycle.checkpoints import new_version_id


def create_candidate(champion: NeuralEngine, origin: str = "adapt") -> NeuralEngine:
    """Deep-copy the champion into a new, independently trainable candidate.

    The candidate shares no tensors, normaliser, calibrator or metadata object with the
    champion, so training it can never mutate production weights.
    """
    candidate = champion.clone(new_version_id("cand"), origin=origin)
    assert_isolated(champion, candidate)
    return candidate


def assert_isolated(champion: NeuralEngine, candidate: NeuralEngine) -> None:
    """Raise AssertionError if the candidate shares parameter storage, normaliser/calibration
    objects or its version id with the champion.
    """
    champ_ptrs = {p.data_ptr() for p in champion.ensemble.parameters()}
    shared = [n for n, p in candidate.ensemble.named_parameters() if p.data_ptr() in champ_ptrs]
    if shared:
        raise AssertionError(f"candidate shares parameter storage with champion: {shared[:3]}")
    if candidate.normalizer is champion.normalizer or candidate.calibration is champion.calibration:
        raise AssertionError("candidate shares normaliser/calibration objects with champion")
    if candidate.version == champion.version:
        raise AssertionError("candidate must have a new version id")
