"""Feature and target normalisation fitted strictly on training rows.

``FeatureNormalizer.fit`` receives the store *and the training indices*; it never sees
validation/test rows.  The fitted state is JSON-serialisable and stored in every
checkpoint so inference is reproducible.

Regression targets are *scaled but not centred* so that non-negative magnitudes
(drawdown, upside, volatility) stay non-negative and the model's softplus heads remain
valid.  Models therefore operate entirely in normalised space; the inference engine maps
predictions back to real units with :meth:`FeatureNormalizer.denormalize`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig, NormalizationConfig
from nardis_neural.data.datasets import Batch, SequenceBatch, TargetBatch
from nardis_neural.data.loaders import KEY_CURRENT, ArrayStore, seq_key, target_key

F64 = npt.NDArray[np.float64]


def _center_scale(x: F64, method: str) -> tuple[F64, F64]:
    """Column-wise centre/scale ignoring NaN.  x: (rows, features)."""
    if x.shape[0] == 0:
        return np.zeros(x.shape[1]), np.ones(x.shape[1])
    with np.errstate(all="ignore"):
        if method == "robust":
            center = np.nanmedian(x, axis=0)
            q75, q25 = np.nanpercentile(x, [75, 25], axis=0)
            scale = (q75 - q25) / 1.349
            std = np.nanstd(x, axis=0)
            # fall back to std for (near-)degenerate IQR such as sparse / binary features
            scale = np.where(scale < 1e-8, std, scale)
        else:
            center = np.nanmean(x, axis=0)
            scale = np.nanstd(x, axis=0)
    center = np.nan_to_num(center, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    return center.astype(np.float64), scale.astype(np.float64)


@dataclass
class FeatureNormalizer:
    """Centre/scale statistics for current features and sequences, and target scales."""

    method: str
    clip: float
    current_center: F64
    current_scale: F64
    seq_center: dict[str, F64]
    seq_scale: dict[str, F64]
    target_scale: dict[str, F64]
    fitted_rows: int = 0

    # ------------------------------------------------------------------ fitting
    @classmethod
    def fit(
        cls,
        store: ArrayStore,
        train_indices: npt.NDArray[np.int64],
        config: NeuralConfig,
        seed: int = 0,
    ) -> FeatureNormalizer:
        """Fit on ``train_indices`` only (subsampled to ``max_fit_rows``).

        Sequence statistics use observed steps only; each target scale is the per-horizon RMS
        of the 1–99 % clipped training values.
        """
        nc: NormalizationConfig = config.normalization
        idx = np.asarray(train_indices, dtype=np.int64)
        if len(idx) == 0:
            raise ValueError("cannot fit normaliser on an empty training set")
        rng = np.random.default_rng(seed)
        if len(idx) > nc.max_fit_rows:
            idx = np.sort(rng.choice(idx, size=nc.max_fit_rows, replace=False))
        else:
            idx = np.sort(idx)
        cur = np.asarray(store[KEY_CURRENT][idx], dtype=np.float64)
        c_center, c_scale = _center_scale(cur, nc.method)
        seq_center: dict[str, F64] = {}
        seq_scale: dict[str, F64] = {}
        for ts in config.features.timescales:
            vk = seq_key(ts.name, "values")
            if vk not in store:
                seq_center[ts.name] = np.zeros(ts.feature_dim)
                seq_scale[ts.name] = np.ones(ts.feature_dim)
                continue
            vals = np.asarray(store[vk][idx], dtype=np.float64)
            mask = np.asarray(store[seq_key(ts.name, "mask")][idx], dtype=np.bool_)
            flat = vals[mask]
            if flat.shape[0] > nc.max_fit_rows:
                flat = flat[rng.choice(flat.shape[0], size=nc.max_fit_rows, replace=False)]
            seq_center[ts.name], seq_scale[ts.name] = _center_scale(flat, nc.method)
        tscale: dict[str, F64] = {}
        for task in REGRESSION_TASKS:
            tk = target_key(task)
            if tk not in store:
                tscale[task] = np.ones(len(config.targets.horizons))
                continue
            y = np.asarray(store[tk][idx], dtype=np.float64)
            y = np.where(np.isfinite(y), y, np.nan)
            lo, hi = np.nanpercentile(y, [1, 99], axis=0)
            yc = np.clip(y, lo, hi)
            rms = np.sqrt(np.nanmean(yc**2, axis=0))
            tscale[task] = np.where(np.isfinite(rms) & (rms > 1e-8), rms, 1.0)
        return cls(
            method=nc.method,
            clip=nc.clip,
            current_center=c_center,
            current_scale=c_scale,
            seq_center=seq_center,
            seq_scale=seq_scale,
            target_scale=tscale,
            fitted_rows=len(idx),
        )

    # ------------------------------------------------------------------ transforms
    def _t(self, arr: F64, like: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(arr, dtype=torch.float32, device=like.device)

    def _norm(self, x: torch.Tensor, center: F64, scale: F64) -> torch.Tensor:
        z = (x.float() - self._t(center, x)) / self._t(scale, x)
        z = torch.nan_to_num(z, nan=0.0, posinf=self.clip, neginf=-self.clip)
        return z.clamp(-self.clip, self.clip)

    def transform_batch(self, batch: Batch) -> Batch:
        """Return a normalised copy of ``batch`` (unchanged if it is already normalised).

        Features are centred, scaled and clipped to ``±clip`` (NaN → 0), unobserved steps are
        zeroed, regression targets are divided by their scale and graph node features get a
        signed ``log1p``.
        """
        if batch.normalized:
            return batch
        seqs: dict[str, SequenceBatch] = {}
        for name, s in batch.sequences.items():
            v = self._norm(s.values, self.seq_center[name], self.seq_scale[name])
            v = v * s.mask.unsqueeze(-1).to(v.dtype)
            seqs[name] = SequenceBatch(values=v, mask=s.mask, time_deltas=s.time_deltas)
        targets = batch.targets
        if targets is not None:
            reg = {
                task: targets.regression[task] / self._t(self.target_scale[task], targets.regression[task])
                for task in targets.regression
            }
            targets = TargetBatch(regression=reg, labels=targets.labels, mask=targets.mask)
        graph = batch.graph
        if graph is not None:
            nf = torch.nan_to_num(graph.node_features, nan=0.0).clamp(-1e4, 1e4)
            nf = torch.sign(nf) * torch.log1p(nf.abs())  # scale-free signed log for graph features
            graph = replace(graph, node_features=nf)
        return replace(
            batch,
            current=self._norm(batch.current, self.current_center, self.current_scale),
            sequences=seqs,
            targets=targets,
            graph=graph,
            normalized=True,
        )

    def denormalize(
        self, task: str, mean: torch.Tensor, var: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Map normalised mean (…, H) and variance back to real units."""
        s = self._t(self.target_scale[task], mean)
        return mean * s, None if var is None else var * s**2

    def target_scale_tensor(self, task: str, like: torch.Tensor) -> torch.Tensor:
        """Per-horizon scale of ``task`` as float32 on ``like``'s device."""
        return self._t(self.target_scale[task], like)

    def current_zscore(self, current: torch.Tensor) -> torch.Tensor:
        """Unclipped z-scores of current features (for input-space OOD)."""
        z = (current.float() - self._t(self.current_center, current)) / self._t(self.current_scale, current)
        return torch.nan_to_num(z, nan=0.0, posinf=1e3, neginf=-1e3)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable state, stored in checkpoints."""
        return {
            "method": self.method,
            "clip": self.clip,
            "current_center": self.current_center.tolist(),
            "current_scale": self.current_scale.tolist(),
            "seq_center": {k: v.tolist() for k, v in self.seq_center.items()},
            "seq_scale": {k: v.tolist() for k, v in self.seq_scale.items()},
            "target_scale": {k: v.tolist() for k, v in self.target_scale.items()},
            "fitted_rows": self.fitted_rows,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureNormalizer:
        """Rebuild a normaliser from :meth:`to_dict` output."""
        return cls(
            method=str(d["method"]),
            clip=float(d["clip"]),
            current_center=np.asarray(d["current_center"], dtype=np.float64),
            current_scale=np.asarray(d["current_scale"], dtype=np.float64),
            seq_center={k: np.asarray(v, dtype=np.float64) for k, v in d["seq_center"].items()},
            seq_scale={k: np.asarray(v, dtype=np.float64) for k, v in d["seq_scale"].items()},
            target_scale={k: np.asarray(v, dtype=np.float64) for k, v in d["target_scale"].items()},
            fitted_rows=int(d.get("fitted_rows", 0)),
        )
