"""Out-of-distribution scoring.

Three complementary signals, each normalised by its ``reference_quantile`` on the
training distribution (so ≈1 marks the edge of familiar territory):

1. **embedding** — Mahalanobis distance of the MarketStateEmbedding to the training
   embedding distribution (shrinkage covariance for stability),
2. **input** — RMS z-score of the raw current features under the training normaliser,
3. **disagreement** — ensemble epistemic uncertainty.

``ood_score`` is their weighted mean; values > 1 reduce confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.config import OODConfig

F64 = npt.NDArray[np.float64]


def shrunk_precision(x: F64, shrinkage: float) -> tuple[F64, F64]:
    mean = x.mean(axis=0)
    xc = x - mean
    cov = xc.T @ xc / max(len(x) - 1, 1)
    target = np.eye(cov.shape[0]) * np.trace(cov) / cov.shape[0]
    cov = (1 - shrinkage) * cov + shrinkage * target + 1e-6 * np.eye(cov.shape[0])
    return mean, np.linalg.inv(cov)


def mahalanobis(x: F64, mean: F64, precision: F64) -> F64:
    d = x - mean
    sq = np.asarray(np.einsum("ij,jk,ik->i", d, precision, d), dtype=np.float64)
    return np.asarray(np.sqrt(np.maximum(sq, 0.0)), dtype=np.float64)


@dataclass
class OODDetector:
    mean: F64
    precision: F64
    ref_embedding: float
    ref_input: float
    ref_disagreement: float
    median_embedding: float
    weights: tuple[float, float, float]

    @classmethod
    def fit(cls, embeddings: F64, input_rms: F64, epistemic: F64, cfg: OODConfig) -> OODDetector:
        emb = np.asarray(embeddings, dtype=np.float64)
        mean, prec = shrunk_precision(emb, cfg.shrinkage)
        d = mahalanobis(emb, mean, prec)
        q = cfg.reference_quantile

        def ref(v: F64) -> float:
            val = float(np.quantile(v, q)) if len(v) else 1.0
            return val if val > 1e-8 else 1.0

        return cls(
            mean=mean,
            precision=prec,
            ref_embedding=ref(d),
            ref_input=ref(np.asarray(input_rms, dtype=np.float64)),
            ref_disagreement=ref(np.asarray(epistemic, dtype=np.float64)),
            median_embedding=float(np.median(d)) if len(d) else 1.0,
            weights=(cfg.embedding_weight, cfg.input_weight, cfg.disagreement_weight),
        )

    def embedding_distance(self, embeddings: F64) -> F64:
        return mahalanobis(np.asarray(embeddings, dtype=np.float64), self.mean, self.precision)

    def score(self, embeddings: F64, input_rms: F64, epistemic: F64) -> dict[str, F64]:
        comp = {
            "embedding": self.embedding_distance(embeddings) / self.ref_embedding,
            "input": np.asarray(input_rms, dtype=np.float64) / self.ref_input,
            "disagreement": np.asarray(epistemic, dtype=np.float64) / self.ref_disagreement,
        }
        w = np.asarray(self.weights, dtype=np.float64)
        w = w / w.sum() if w.sum() > 0 else np.full(3, 1 / 3)
        comp["score"] = w[0] * comp["embedding"] + w[1] * comp["input"] + w[2] * comp["disagreement"]
        return comp

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "precision": self.precision.tolist(),
            "ref_embedding": self.ref_embedding,
            "ref_input": self.ref_input,
            "ref_disagreement": self.ref_disagreement,
            "median_embedding": self.median_embedding,
            "weights": list(self.weights),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OODDetector:
        w = d["weights"]
        return cls(
            mean=np.asarray(d["mean"], dtype=np.float64),
            precision=np.asarray(d["precision"], dtype=np.float64),
            ref_embedding=float(d["ref_embedding"]),
            ref_input=float(d["ref_input"]),
            ref_disagreement=float(d["ref_disagreement"]),
            median_embedding=float(d["median_embedding"]),
            weights=(float(w[0]), float(w[1]), float(w[2])),
        )
