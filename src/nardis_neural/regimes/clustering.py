"""Unsupervised regime discovery on MarketStateEmbeddings.

No regime is defined by hand: clusters emerge from the learned latent space via KMeans,
a Gaussian mixture, or HDBSCAN (scikit-learn's implementation, no extra dependency).
Fitted models are stored as plain arrays so they can be embedded in checkpoints and
applied to new embeddings at inference time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.config import RegimeConfig

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def _log_gauss(x: F64, means: F64, covs: F64, weights: F64) -> F64:
    """(N, K) log p(x, k) for full-covariance Gaussians."""
    n, d = x.shape
    out = np.empty((n, len(means)))
    for k in range(len(means)):
        chol = np.linalg.cholesky(covs[k] + 1e-6 * np.eye(d))
        sol = np.linalg.solve(chol, (x - means[k]).T)
        maha = (sol**2).sum(axis=0)
        logdet = 2 * np.log(np.diag(chol)).sum()
        out[:, k] = np.log(weights[k] + 1e-300) - 0.5 * (d * np.log(2 * np.pi) + logdet + maha)
    return out


@dataclass
class RegimeClusterer:
    """Fitted regime model (KMeans, Gaussian mixture or HDBSCAN) stored as plain arrays."""

    method: str
    n_clusters: int = 0
    centers: F64 = field(default_factory=lambda: np.zeros((0, 0)))
    covariances: F64 | None = None
    weights: F64 | None = None
    noise_radius: F64 | None = None
    selection_scores: dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, embeddings: F64, cfg: RegimeConfig) -> tuple[RegimeClusterer, I64]:
        """Fit ``cfg.method`` and return the model with labels for every embedding.

        Fitting uses at most ``max_fit_samples`` random rows.  With ``auto_select`` the number
        of clusters is chosen by silhouette score (KMeans) or BIC (GMM).
        """
        x = np.asarray(embeddings, dtype=np.float64)
        rng = np.random.default_rng(cfg.seed)
        fit_x = (
            x if len(x) <= cfg.max_fit_samples else x[rng.choice(len(x), cfg.max_fit_samples, replace=False)]
        )
        if cfg.method == "kmeans":
            model = cls._fit_kmeans(fit_x, cfg)
        elif cfg.method == "gmm":
            model = cls._fit_gmm(fit_x, cfg)
        else:
            model = cls._fit_hdbscan(fit_x, cfg)
        return model, model.predict(x)

    @classmethod
    def _choose_k(cls, cfg: RegimeConfig, n: int) -> list[int]:
        if not cfg.auto_select:
            return [min(cfg.n_clusters, max(n - 1, 1))]
        return [k for k in range(cfg.k_min, cfg.k_max + 1) if k < n]

    @classmethod
    def _fit_kmeans(cls, x: F64, cfg: RegimeConfig) -> RegimeClusterer:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        best: tuple[float, Any] | None = None
        scores: dict[str, float] = {}
        for k in cls._choose_k(cfg, len(x)):
            km = KMeans(n_clusters=k, n_init=5, random_state=cfg.seed).fit(x)
            if cfg.auto_select and k > 1:
                sample = min(len(x), 5000)
                s = float(silhouette_score(x, km.labels_, sample_size=sample, random_state=cfg.seed))
            else:
                s = 0.0
            scores[str(k)] = s
            if best is None or s > best[0]:
                best = (s, km)
        assert best is not None
        km = best[1]
        return cls("kmeans", int(km.n_clusters), np.asarray(km.cluster_centers_), selection_scores=scores)

    @classmethod
    def _fit_gmm(cls, x: F64, cfg: RegimeConfig) -> RegimeClusterer:
        from sklearn.mixture import GaussianMixture

        best: tuple[float, Any] | None = None
        scores: dict[str, float] = {}
        for k in cls._choose_k(cfg, len(x)):
            g = GaussianMixture(n_components=k, covariance_type="full", reg_covar=1e-4, random_state=cfg.seed)
            g.fit(x)
            bic = float(g.bic(x))
            scores[str(k)] = bic
            if best is None or -bic > best[0]:
                best = (-bic, g)
        assert best is not None
        g = best[1]
        return cls(
            "gmm",
            int(g.n_components),
            np.asarray(g.means_),
            covariances=np.asarray(g.covariances_),
            weights=np.asarray(g.weights_),
            selection_scores=scores,
        )

    @classmethod
    def _fit_hdbscan(cls, x: F64, cfg: RegimeConfig) -> RegimeClusterer:
        from sklearn.cluster import HDBSCAN

        labels = HDBSCAN(min_cluster_size=cfg.hdbscan_min_cluster_size, copy=True).fit_predict(x)
        ks = sorted(k for k in set(labels.tolist()) if k >= 0)
        if not ks:  # everything is noise → single cluster fallback
            return cls("hdbscan", 1, x.mean(axis=0, keepdims=True), noise_radius=np.array([np.inf]))
        centers = np.stack([x[labels == k].mean(axis=0) for k in ks])
        radius = np.array(
            [np.quantile(np.linalg.norm(x[labels == k] - centers[i], axis=1), 0.95) for i, k in enumerate(ks)]
        )
        return cls("hdbscan", len(ks), centers, noise_radius=radius)

    def predict(self, embeddings: F64) -> I64:
        """Regime label per embedding: nearest centre, or most likely component for GMM.

        HDBSCAN labels points farther than their cluster's radius as noise (-1).
        """
        x = np.asarray(embeddings, dtype=np.float64)
        if len(x) == 0:
            return np.zeros(0, dtype=np.int64)
        if self.method == "gmm":
            assert self.covariances is not None and self.weights is not None
            return np.asarray(
                _log_gauss(x, self.centers, self.covariances, self.weights).argmax(axis=1), np.int64
            )
        dist = np.linalg.norm(x[:, None, :] - self.centers[None, :, :], axis=-1)
        labels = dist.argmin(axis=1).astype(np.int64)
        if self.method == "hdbscan" and self.noise_radius is not None:
            too_far = dist[np.arange(len(x)), labels] > self.noise_radius[labels]
            labels[too_far] = -1
        return labels

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form, the inverse of from_dict()."""
        return {
            "method": self.method,
            "n_clusters": self.n_clusters,
            "centers": self.centers.tolist(),
            "covariances": None if self.covariances is None else self.covariances.tolist(),
            "weights": None if self.weights is None else self.weights.tolist(),
            "noise_radius": None if self.noise_radius is None else self.noise_radius.tolist(),
            "selection_scores": self.selection_scores,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RegimeClusterer:
        """Rebuild a clusterer from to_dict() output."""

        def arr(v: Any) -> F64 | None:
            return None if v is None else np.asarray(v, dtype=np.float64)

        centers = arr(d["centers"])
        assert centers is not None
        return cls(
            method=str(d["method"]),
            n_clusters=int(d["n_clusters"]),
            centers=centers,
            covariances=arr(d.get("covariances")),
            weights=arr(d.get("weights")),
            noise_radius=arr(d.get("noise_radius")),
            selection_scores=dict(d.get("selection_scores", {})),
        )


def cluster_statistics(
    labels: I64,
    expert_weights: F64,
    expert_names: list[str],
    horizons: list[str],
    predicted_return: F64 | None = None,
    realized_return: F64 | None = None,
    timestamps: F64 | None = None,
    extra: dict[str, F64] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per-cluster summary: size, expert usage, prediction and error statistics."""
    out: dict[str, dict[str, Any]] = {}
    n = len(labels)
    for k in sorted(set(labels.tolist())):
        sel = labels == k
        stats: dict[str, Any] = {"count": int(sel.sum()), "fraction": float(sel.mean()) if n else 0.0}
        stats["expert_weights"] = {
            e: float(expert_weights[sel, i].mean()) for i, e in enumerate(expert_names)
        }
        stats["dominant_expert"] = expert_names[int(expert_weights[sel].mean(axis=0).argmax())]
        if predicted_return is not None:
            stats["predicted_return"] = {
                h: float(predicted_return[sel, j].mean()) for j, h in enumerate(horizons)
            }
        if realized_return is not None:
            stats["realized_return"] = {
                h: float(realized_return[sel, j].mean()) for j, h in enumerate(horizons)
            }
            stats["realized_volatility"] = {
                h: float(realized_return[sel, j].std()) for j, h in enumerate(horizons)
            }
            if predicted_return is not None:
                err = predicted_return[sel] - realized_return[sel]
                stats["return_mae"] = {h: float(np.abs(err[:, j]).mean()) for j, h in enumerate(horizons)}
                stats["return_bias"] = {h: float(err[:, j].mean()) for j, h in enumerate(horizons)}
        if timestamps is not None:
            stats["first_seen"] = float(timestamps[sel].min())
            stats["last_seen"] = float(timestamps[sel].max())
        for name, values in (extra or {}).items():
            stats[name] = float(np.asarray(values)[sel].mean())
        out[str(k)] = stats
    return out
