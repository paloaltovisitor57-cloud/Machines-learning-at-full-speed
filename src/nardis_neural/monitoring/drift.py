"""Drift detection on four levels, each with statistics suited to it.

1. **Raw input drift** — per-feature PSI (quantile bins of the reference), two-sample KS
   test, Wasserstein distance (in reference std units) and moment shifts.  A feature
   drifts when PSI exceeds its threshold, or when KS is significant *and* the Wasserstein
   distance is material (KS alone is over-sensitive on large samples).
2. **Latent embedding drift** — Mahalanobis distance of embeddings to the training
   distribution: ratio of mean distances, KS test on the distance distributions, and the
   norm of the mean shift in whitened space.
3. **Prediction drift** — the same univariate battery on model outputs.
4. **Model-error drift** — ratio of current to reference absolute error plus KS on error
   distributions (only when outcomes are known).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field
from scipy import stats

from nardis_neural.config import DriftConfig, NeuralConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import KEY_CURRENT, Array, seq_key
from nardis_neural.inference.engine import NeuralEngine

F64 = npt.NDArray[np.float64]


def population_stability_index(ref: F64, cur: F64, bins: int = 10) -> float:
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return math.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        # (near-)constant reference: compare the share of values equal to the mode
        mode = edges[0]
        p = np.array([np.mean(ref == mode), np.mean(ref != mode)])
        q = np.array([np.mean(cur == mode), np.mean(cur != mode)])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
        p = np.histogram(ref, edges)[0] / len(ref)
        q = np.histogram(cur, edges)[0] / len(cur)
    p = np.clip(p, 1e-4, None)
    q = np.clip(q, 1e-4, None)
    return float(np.sum((q - p) * np.log(q / p)))


class FeatureDrift(BaseModel):
    name: str
    psi: float
    ks_statistic: float
    ks_pvalue: float
    wasserstein: float
    mean_shift: float
    std_ratio: float
    drifted: bool


def univariate_drift(ref: F64, cur: F64, name: str, cfg: DriftConfig) -> FeatureDrift:
    r = ref[np.isfinite(ref)]
    c = cur[np.isfinite(cur)]
    if len(r) < 2 or len(c) < 2:
        return FeatureDrift(
            name=name,
            psi=math.nan,
            ks_statistic=math.nan,
            ks_pvalue=1.0,
            wasserstein=math.nan,
            mean_shift=math.nan,
            std_ratio=math.nan,
            drifted=False,
        )
    sd = float(r.std()) or 1.0
    ks = stats.ks_2samp(r, c)
    w = float(stats.wasserstein_distance(r, c)) / sd
    psi = population_stability_index(r, c, cfg.psi_bins)
    drifted = (np.isfinite(psi) and psi > cfg.psi_threshold) or (
        float(ks.pvalue) < cfg.ks_pvalue_threshold and w > cfg.wasserstein_threshold
    )
    return FeatureDrift(
        name=name,
        psi=psi,
        ks_statistic=float(ks.statistic),
        ks_pvalue=float(ks.pvalue),
        wasserstein=w,
        mean_shift=float((c.mean() - r.mean()) / sd),
        std_ratio=float(c.std() / sd),
        drifted=bool(drifted),
    )


class DriftSection(BaseModel):
    drifted: bool
    fraction_drifted: float = 0.0
    features: list[FeatureDrift] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)


class DriftReport(BaseModel):
    input: DriftSection
    embedding: DriftSection | None = None
    prediction: DriftSection | None = None
    error: DriftSection | None = None
    n_reference: int
    n_current: int

    @property
    def any_drift(self) -> bool:
        return any(
            s is not None and s.drifted for s in (self.input, self.embedding, self.prediction, self.error)
        )

    def summary(self) -> dict[str, Any]:
        return {
            "any_drift": self.any_drift,
            "input": self.input.drifted,
            "embedding": None if self.embedding is None else self.embedding.drifted,
            "prediction": None if self.prediction is None else self.prediction.drifted,
            "error": None if self.error is None else self.error.drifted,
        }


def multivariate_section(ref: F64, cur: F64, names: Sequence[str], cfg: DriftConfig) -> DriftSection:
    feats = [univariate_drift(ref[:, j], cur[:, j], names[j], cfg) for j in range(ref.shape[1])]
    frac = float(np.mean([f.drifted for f in feats])) if feats else 0.0
    return DriftSection(drifted=frac >= cfg.feature_fraction_threshold, fraction_drifted=frac, features=feats)


def input_features(arrays: dict[str, Array], config: NeuralConfig) -> tuple[F64, list[str]]:
    """Current features + masked means of each sequence's features."""
    cols = [np.asarray(arrays[KEY_CURRENT], dtype=np.float64)]
    names = [f"current_{j}" for j in range(cols[0].shape[1])]
    for ts in config.features.timescales:
        vk = seq_key(ts.name, "values")
        if vk not in arrays:
            continue
        v = np.asarray(arrays[vk], dtype=np.float64)
        m = np.asarray(arrays[seq_key(ts.name, "mask")], dtype=bool)[..., None]
        cnt = m.sum(axis=1)
        mean = np.where(cnt > 0, (v * m).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        cols.append(mean)
        names += [f"{ts.name}_mean_{j}" for j in range(v.shape[2])]
    return np.concatenate(cols, axis=1), names


def input_drift(ref: dict[str, Array], cur: dict[str, Array], config: NeuralConfig) -> DriftSection:
    r, names = input_features(ref, config)
    c, _ = input_features(cur, config)
    return multivariate_section(r, c, names, config.drift)


def embedding_drift(
    ref_distances: F64, cur_distances: F64, ref_emb: F64, cur_emb: F64, precision: F64, cfg: DriftConfig
) -> DriftSection:
    ratio = float(np.mean(cur_distances) / max(np.mean(ref_distances), 1e-12))
    ks = stats.ks_2samp(ref_distances, cur_distances)
    shift = cur_emb.mean(axis=0) - ref_emb.mean(axis=0)
    whitened = float(np.sqrt(max(float(shift @ precision @ shift), 0.0)))
    return DriftSection(
        drifted=ratio > cfg.embedding_distance_threshold,
        stats={
            "mean_distance_ratio": ratio,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "whitened_mean_shift": whitened,
        },
    )


def error_drift(ref_err: F64, cur_err: F64, cfg: DriftConfig) -> DriftSection:
    ratio = float(np.mean(cur_err) / max(np.mean(ref_err), 1e-12))
    ks = stats.ks_2samp(ref_err, cur_err)
    return DriftSection(
        drifted=ratio > cfg.error_ratio_threshold,
        stats={
            "reference_mae": float(np.mean(ref_err)),
            "current_mae": float(np.mean(cur_err)),
            "ratio": ratio,
            "ks_statistic": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
        },
    )


PREDICTION_COLUMNS = (
    "return.mean",
    "upside.prob",
    "downside.prob",
    "volatility.mean",
    "confidence",
    "ood_score",
)


def prediction_matrix(preds: dict[str, Array], horizons: Sequence[str]) -> tuple[F64, list[str]]:
    cols, names = [], []
    for key in PREDICTION_COLUMNS:
        if key not in preds:
            continue
        v = np.asarray(preds[key], dtype=np.float64)
        if v.ndim == 1:
            cols.append(v[:, None])
            names.append(key)
        else:
            cols.append(v)
            names += [f"{key}.{h}" for h in horizons]
    return np.concatenate(cols, axis=1), names


def drift_report(engine: NeuralEngine, reference: MarketDataset, current: MarketDataset) -> DriftReport:
    """Full four-level drift report of ``current`` against ``reference`` data."""
    config = engine.config
    cfg = config.drift
    ref_rows = reference.store.select(reference.indices)
    cur_rows = current.store.select(current.indices)
    section_input = input_drift(ref_rows, cur_rows, config)
    ref_pred = engine.predict_dataset(reference)
    cur_pred = engine.predict_dataset(current)
    emb_section = None
    if engine.ood is not None:
        emb_section = embedding_drift(
            engine.ood.embedding_distance(ref_pred["embedding"]),
            engine.ood.embedding_distance(cur_pred["embedding"]),
            ref_pred["embedding"],
            cur_pred["embedding"],
            engine.ood.precision,
            cfg,
        )
    pr, names = prediction_matrix(ref_pred, config.horizon_names)
    pc, _ = prediction_matrix(cur_pred, config.horizon_names)
    pred_section = multivariate_section(pr, pc, names, cfg)
    err_section = None
    if reference.store.has_targets and current.store.has_targets:
        from nardis_neural.training.pipeline import dataset_targets

        rt, rm = dataset_targets(reference)
        ct, cm = dataset_targets(current)
        ref_err = np.abs(ref_pred["return.mean"] - rt["return"])[rm]
        cur_err = np.abs(cur_pred["return.mean"] - ct["return"])[cm]
        err_section = error_drift(ref_err, cur_err, cfg)
    return DriftReport(
        input=section_input,
        embedding=emb_section,
        prediction=pred_section,
        error=err_section,
        n_reference=len(reference),
        n_current=len(current),
    )
