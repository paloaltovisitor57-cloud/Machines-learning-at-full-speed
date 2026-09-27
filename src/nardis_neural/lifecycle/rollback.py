"""Manual and (opt-in) automatic rollback to a previous champion.

A rollback only moves the registry's champion pointer: every model directory is complete
and immutable, so restoring it restores weights, normaliser, calibration state,
configuration and ensemble membership together.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from nardis_neural.config import LifecycleConfig
from nardis_neural.lifecycle.champion import ModelRegistry


def rollback(registry: ModelRegistry, to_version: str | None = None, reason: str = "manual rollback") -> str:
    """Restore ``to_version`` (default: the previous champion). Returns the new champion."""
    target = to_version or registry.previous_champion()
    if target is None:
        raise RuntimeError("no previous champion available for rollback")
    registry.restore(target, reason)
    return target


class AutoRollbackMonitor:
    """Watches live champion error against its promotion-time baseline.

    Disabled unless ``LifecycleConfig.auto_rollback`` is True.  Triggers when the mean
    absolute return error over the last ``auto_rollback_min_observations`` resolved
    predictions exceeds ``baseline · (1 + auto_rollback_degradation)``.
    """

    def __init__(self, cfg: LifecycleConfig, baseline_mae: float | None = None) -> None:
        self.cfg = cfg
        self.baseline_mae = baseline_mae
        self.errors: deque[float] = deque(maxlen=cfg.auto_rollback_min_observations)

    def reset(self, baseline_mae: float | None) -> None:
        """Set a new baseline MAE and discard the collected live errors."""
        self.baseline_mae = baseline_mae
        self.errors.clear()

    def update(self, abs_error: float) -> None:
        """Record one live absolute return error (non-finite values are ignored)."""
        if np.isfinite(abs_error):
            self.errors.append(float(abs_error))

    @property
    def live_mae(self) -> float | None:
        """Mean of the recorded live errors, or None when there are none."""
        return float(np.mean(self.errors)) if self.errors else None

    def should_rollback(self) -> bool:
        """True when enabled, the error window is full and live MAE exceeds the degraded baseline."""
        if not self.cfg.auto_rollback or self.baseline_mae is None:
            return False
        if len(self.errors) < self.cfg.auto_rollback_min_observations:
            return False
        live = self.live_mae
        return live is not None and live > self.baseline_mae * (1 + self.cfg.auto_rollback_degradation)
