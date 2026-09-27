"""Post-hoc probability calibration fitted on *validation* data only.

Methods: temperature scaling (1 parameter on the logit), Platt/sigmoid scaling (affine
on the logit, logistic regression) and isotonic regression (monotone, non-parametric).
State is plain JSON (isotonic stores its knots, applied with ``np.interp``) so no pickled
scikit-learn objects end up in checkpoints.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from nardis_neural.config import CLASSIFICATION_TASKS, CalibrationConfig
from nardis_neural.training.metrics import brier_score, expected_calibration_error, log_loss, reliability_bins

F64 = npt.NDArray[np.float64]
EPS = 1e-6


def _logit(p: F64) -> F64:
    p = np.clip(p, EPS, 1 - EPS)
    return np.asarray(np.log(p) - np.log1p(-p), dtype=np.float64)


def _sigmoid(x: F64) -> F64:
    return np.asarray(1.0 / (1.0 + np.exp(-x)), dtype=np.float64)


@dataclass
class BinaryCalibrator:
    """Maps raw probabilities of one binary event at one horizon to calibrated ones."""

    method: str = "none"
    params: dict[str, Any] = field(default_factory=dict)

    def fit(self, prob: F64, labels: F64, method: str, min_samples: int = 50) -> BinaryCalibrator:
        """Fit ``method`` (``temperature``, ``platt``, ``isotonic`` or ``none``) in place; return self.

        Falls back to the identity with fewer than ``min_samples`` points or a single class.
        """
        prob = np.asarray(prob, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.float64)
        if method == "none" or len(prob) < min_samples or labels.min() == labels.max():
            self.method, self.params = "none", {}
            return self
        z = _logit(prob)
        if method == "temperature":

            def nll(log_t: float) -> float:
                q = np.clip(_sigmoid(z / math.exp(log_t)), EPS, 1 - EPS)
                return float(-np.mean(labels * np.log(q) + (1 - labels) * np.log(1 - q)))

            res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
            self.method, self.params = "temperature", {"temperature": float(math.exp(res.x))}
        elif method == "platt":
            from sklearn.linear_model import LogisticRegression

            lr = LogisticRegression(C=1e4, max_iter=1000)
            lr.fit(z.reshape(-1, 1), labels.astype(int))
            self.method = "platt"
            self.params = {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
        elif method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            iso.fit(prob, labels)
            self.method = "isotonic"
            self.params = {
                "x": [float(v) for v in iso.X_thresholds_],
                "y": [float(v) for v in iso.y_thresholds_],
            }
        else:
            raise ValueError(f"unknown calibration method {method}")
        return self

    def transform(self, prob: F64) -> F64:
        """Apply the fitted calibration to probabilities (identity for ``none``)."""
        prob = np.asarray(prob, dtype=np.float64)
        if self.method == "temperature":
            return _sigmoid(_logit(prob) / self.params["temperature"])
        if self.method == "platt":
            return _sigmoid(self.params["a"] * _logit(prob) + self.params["b"])
        if self.method == "isotonic":
            out = np.interp(prob, np.asarray(self.params["x"]), np.asarray(self.params["y"]))
            return np.asarray(np.clip(out, EPS, 1 - EPS), dtype=np.float64)
        return prob

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state."""
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BinaryCalibrator:
        """Rebuild a calibrator from :meth:`to_dict` output."""
        return cls(method=str(d["method"]), params=dict(d["params"]))


@dataclass
class CalibrationSet:
    """One calibrator per (event task, horizon) plus a before/after quality report."""

    horizons: list[str]
    calibrators: dict[str, list[BinaryCalibrator]] = field(default_factory=dict)
    report: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def identity(cls, horizons: list[str]) -> CalibrationSet:
        """Identity (uncalibrated) calibrators for every event task and horizon."""
        return cls(horizons, {t: [BinaryCalibrator() for _ in horizons] for t in CLASSIFICATION_TASKS})

    def fit(
        self,
        probs: dict[str, F64],
        labels: dict[str, F64],
        mask: npt.NDArray[np.bool_],
        cfg: CalibrationConfig,
    ) -> CalibrationSet:
        """probs/labels: task → (N, H) arrays from the validation split."""
        self.report = {}
        for task in CLASSIFICATION_TASKS:
            cals = []
            for h, name in enumerate(self.horizons):
                sel = mask[:, h]
                p, y = probs[task][sel, h], labels[task][sel, h]
                cal = BinaryCalibrator().fit(p, y, cfg.method, cfg.min_samples)
                q = cal.transform(p)
                self.report[f"{task}.{name}"] = {
                    "method": cal.method,
                    "n": int(sel.sum()),
                    "brier_before": brier_score(p, y),
                    "brier_after": brier_score(q, y),
                    "ece_before": expected_calibration_error(p, y, cfg.n_bins),
                    "ece_after": expected_calibration_error(q, y, cfg.n_bins),
                    "log_loss_before": log_loss(p, y),
                    "log_loss_after": log_loss(q, y),
                    "reliability_after": reliability_bins(q, y, cfg.n_bins),
                }
                cals.append(cal)
            self.calibrators[task] = cals
        return self

    def apply(self, task: str, prob: F64) -> F64:
        """prob: (N, H) → calibrated (N, H)."""
        out = np.empty_like(np.asarray(prob, dtype=np.float64))
        for h, cal in enumerate(self.calibrators[task]):
            out[:, h] = cal.transform(prob[:, h])
        return out

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state, including the fit report."""
        return {
            "horizons": self.horizons,
            "calibrators": {t: [c.to_dict() for c in cs] for t, cs in self.calibrators.items()},
            "report": self.report,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CalibrationSet:
        """Rebuild a calibration set from :meth:`to_dict` output."""
        return cls(
            horizons=list(d["horizons"]),
            calibrators={
                t: [BinaryCalibrator.from_dict(c) for c in cs] for t, cs in d["calibrators"].items()
            },
            report=dict(d.get("report", {})),
        )
