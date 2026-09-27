"""Portable gradient-boosted trees.

scikit-learn's ``HistGradientBoostingRegressor`` is trained normally, then its trees are
exported to plain NumPy arrays and evaluated by a tiny NumPy predictor.  Checkpoints thus
contain only arrays + JSON (no pickled objects) and load without scikit-learn internals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass
class TreeEnsemble:
    """Gradient-boosted regression trees as padded NumPy arrays, one ``(trees, nodes)`` array per field."""

    feature: npt.NDArray[np.int64]  # (T, M) padded
    threshold: F64
    left: npt.NDArray[np.int64]
    right: npt.NDArray[np.int64]
    is_leaf: npt.NDArray[np.bool_]
    value: F64
    missing_left: npt.NDArray[np.bool_]
    baseline: float

    @classmethod
    def fit(cls, x: npt.NDArray[Any], y: F64, seed: int = 0, **params: Any) -> TreeEnsemble:
        """Train a shallow, early-stopped ``HistGradientBoostingRegressor`` (``params`` override the
        defaults) and export it.
        """
        from sklearn.ensemble import HistGradientBoostingRegressor

        cfg = {
            "max_depth": 3,
            "learning_rate": 0.04,
            "max_iter": 400,
            "l2_regularization": 1.0,
            "min_samples_leaf": 40,
            "early_stopping": True,
            "validation_fraction": 0.2,
        } | params
        model = HistGradientBoostingRegressor(random_state=seed, **cfg).fit(x, y)
        return cls.export(model)

    @classmethod
    def export(cls, model: Any) -> TreeEnsemble:
        """Convert a fitted ``HistGradientBoostingRegressor`` to arrays (reads its private predictors)."""
        trees = [p[0].nodes for p in model._predictors]
        m = max(len(t) for t in trees)
        t_count = len(trees)

        def pad(field: str, dtype: Any, fill: Any) -> Any:
            out = np.full((t_count, m), fill, dtype=dtype)
            for i, t in enumerate(trees):
                out[i, : len(t)] = t[field]
            return out

        return cls(
            feature=pad("feature_idx", np.int64, 0),
            threshold=pad("num_threshold", np.float64, 0.0),
            left=pad("left", np.int64, 0),
            right=pad("right", np.int64, 0),
            is_leaf=pad("is_leaf", np.bool_, True),
            value=pad("value", np.float64, 0.0),
            missing_left=pad("missing_go_to_left", np.bool_, True),
            baseline=float(np.asarray(model._baseline_prediction).ravel()[0]),
        )

    def predict(self, x: npt.NDArray[Any]) -> F64:
        """Baseline plus the leaf values of every tree; NaN inputs follow each split's missing branch."""
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        rows = np.arange(n)
        total = np.full(n, self.baseline)
        for t in range(self.feature.shape[0]):
            node = np.zeros(n, dtype=np.int64)
            active = ~self.is_leaf[t, node]
            while active.any():
                nd = node[active]
                v = x[rows[active], self.feature[t, nd]]
                go_left = np.where(np.isnan(v), self.missing_left[t, nd], v <= self.threshold[t, nd])
                node[active] = np.where(go_left, self.left[t, nd], self.right[t, nd])
                active = ~self.is_leaf[t, node]
            total += self.value[t, node]
        return total

    def save(self, file: str | Path) -> None:
        """Write the arrays to an ``.npz`` file."""
        np.savez(
            file,
            feature=self.feature,
            threshold=self.threshold,
            left=self.left,
            right=self.right,
            is_leaf=self.is_leaf,
            value=self.value,
            missing_left=self.missing_left,
            baseline=np.asarray([self.baseline]),
        )

    @classmethod
    def load(cls, file: str | Path) -> TreeEnsemble:
        """Load an ensemble written by :meth:`save` (no pickles)."""
        with np.load(file, allow_pickle=False) as z:
            return cls(
                z["feature"],
                z["threshold"],
                z["left"],
                z["right"],
                z["is_leaf"],
                z["value"],
                z["missing_left"],
                float(z["baseline"][0]),
            )
