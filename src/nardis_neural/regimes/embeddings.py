"""Extract MarketStateEmbeddings (plus expert weights, predictions, targets) from datasets
and persist them to Parquet or NumPy for exploratory analysis / downstream models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from nardis_neural.config import REGRESSION_TASKS
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import KEY_REGIME
from nardis_neural.inference.engine import NeuralEngine

Array = npt.NDArray[Any]


@dataclass
class EmbeddingTable:
    observation_ids: Array
    timestamps: Array
    embeddings: Array  # (N, D)
    expert_weights: Array  # (N, E)
    expert_names: list[str]
    horizons: list[str]
    predictions: dict[str, Array] = field(default_factory=dict)
    targets: dict[str, Array] = field(default_factory=dict)
    extra: dict[str, Array] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observation_ids)

    def to_polars(self) -> pl.DataFrame:
        cols: dict[str, Any] = {
            "observation_id": self.observation_ids.astype(str),
            "timestamp": self.timestamps.astype(np.float64),
        }
        for j in range(self.embeddings.shape[1]):
            cols[f"emb_{j}"] = self.embeddings[:, j].astype(np.float32)
        for i, name in enumerate(self.expert_names):
            cols[f"gate_{name}"] = self.expert_weights[:, i].astype(np.float32)
        for key, arr in {**self.predictions, **self.targets}.items():
            a = np.asarray(arr)
            if a.ndim == 2 and a.shape[1] == len(self.horizons):
                for h, hn in enumerate(self.horizons):
                    cols[f"{key}.{hn}"] = a[:, h]
            elif a.ndim == 1:
                cols[key] = a
        for key, arr in self.extra.items():
            cols[key] = np.asarray(arr)
        return pl.DataFrame(cols)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".parquet":
            self.to_polars().write_parquet(p)
        elif p.suffix == ".npz":
            payload: dict[str, Any] = {
                "observation_ids": self.observation_ids.astype(str),
                "timestamps": self.timestamps,
                "embeddings": self.embeddings,
                "expert_weights": self.expert_weights,
                "expert_names": np.asarray(self.expert_names),
                "horizons": np.asarray(self.horizons),
            }
            payload |= {f"pred__{k}": v for k, v in self.predictions.items()}
            payload |= {f"target__{k}": v for k, v in self.targets.items()}
            payload |= {f"extra__{k}": v for k, v in self.extra.items()}
            np.savez(p, **payload)
        else:
            raise ValueError("embedding output must end with .parquet or .npz")
        return p

    @classmethod
    def load_npz(cls, path: str | Path) -> EmbeddingTable:
        with np.load(path, allow_pickle=False) as d:
            return cls(
                observation_ids=d["observation_ids"],
                timestamps=d["timestamps"],
                embeddings=d["embeddings"],
                expert_weights=d["expert_weights"],
                expert_names=[str(x) for x in d["expert_names"]],
                horizons=[str(x) for x in d["horizons"]],
                predictions={k[6:]: d[k] for k in d.files if k.startswith("pred__")},
                targets={k[8:]: d[k] for k in d.files if k.startswith("target__")},
                extra={k[7:]: d[k] for k in d.files if k.startswith("extra__")},
            )


def extract_embeddings(
    engine: NeuralEngine, dataset: MarketDataset, batch_size: int | None = None
) -> EmbeddingTable:
    preds = engine.predict_dataset(dataset, batch_size=batch_size)
    keep = [
        "return.mean",
        "return.std",
        "upside.prob",
        "downside.prob",
        "max_drawdown.mean",
        "volatility.mean",
        "confidence",
        "ood_score",
        "epistemic",
        "regime",
    ]
    table = EmbeddingTable(
        observation_ids=preds["observation_id"],
        timestamps=preds["timestamp"],
        embeddings=preds["embedding"],
        expert_weights=preds["expert_weights"],
        expert_names=list(engine.expert_names),
        horizons=engine.config.horizon_names,
        predictions={k: preds[k] for k in keep if k in preds},
    )
    if dataset.store.has_targets:
        table.targets = {t: dataset.target_array(t) for t in REGRESSION_TASKS}
    if KEY_REGIME in dataset.store:
        table.extra["true_regime"] = np.asarray(dataset.store[KEY_REGIME][dataset.indices])
    return table
