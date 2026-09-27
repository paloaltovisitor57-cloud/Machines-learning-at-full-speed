"""Tensor batches, collation and the map-style dataset used for training/inference."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig
from nardis_neural.data.loaders import (
    KEY_CURRENT,
    KEY_ID,
    KEY_REGIME,
    KEY_TARGET_MASK,
    KEY_TIMESTAMP,
    KEY_WEIGHT,
    Array,
    ArrayStore,
    has_graph,
    seq_key,
    target_key,
)

Tensor = torch.Tensor


@dataclass
class SequenceBatch:
    values: Tensor  # (B, T, F) float
    mask: Tensor  # (B, T) bool, True = observed
    time_deltas: Tensor  # (B, T) seconds before observation time

    def to(self, device: torch.device) -> SequenceBatch:
        return SequenceBatch(
            self.values.to(device, non_blocking=True),
            self.mask.to(device, non_blocking=True),
            self.time_deltas.to(device, non_blocking=True),
        )

    @property
    def available(self) -> Tensor:
        """(B,) True where at least one step is observed."""
        return self.mask.any(dim=1)


@dataclass
class GraphBatch:
    node_features: Tensor  # (N_total, F_node)
    edge_index: Tensor  # (2, E_total) global node indices
    edge_type: Tensor  # (E_total,)
    node_batch: Tensor  # (N_total,) sample index of each node
    target_node: Tensor  # (B,) global index of target node, -1 when the sample has no graph

    def to(self, device: torch.device) -> GraphBatch:
        return GraphBatch(
            self.node_features.to(device),
            self.edge_index.to(device),
            self.edge_type.to(device),
            self.node_batch.to(device),
            self.target_node.to(device),
        )

    @property
    def available(self) -> Tensor:
        return self.target_node >= 0


@dataclass
class TargetBatch:
    regression: dict[str, Tensor]  # task -> (B, H)
    labels: dict[str, Tensor]  # task -> (B, H) float {0, 1}
    mask: Tensor  # (B, H) bool

    def to(self, device: torch.device) -> TargetBatch:
        return TargetBatch(
            {k: v.to(device) for k, v in self.regression.items()},
            {k: v.to(device) for k, v in self.labels.items()},
            self.mask.to(device),
        )


@dataclass
class Batch:
    current: Tensor  # (B, F)
    sequences: dict[str, SequenceBatch]
    observation_ids: list[str]
    timestamps: npt.NDArray[np.float64]
    graph: GraphBatch | None = None
    targets: TargetBatch | None = None
    weights: Tensor | None = None  # (B,) per-sample loss weights
    regimes: npt.NDArray[np.int64] | None = None
    normalized: bool = False
    extras: dict[str, Tensor] = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(self.current.shape[0])

    @property
    def device(self) -> torch.device:
        return self.current.device

    def to(self, device: torch.device) -> Batch:
        return replace(
            self,
            current=self.current.to(device, non_blocking=True),
            sequences={k: v.to(device) for k, v in self.sequences.items()},
            graph=None if self.graph is None else self.graph.to(device),
            targets=None if self.targets is None else self.targets.to(device),
            weights=None if self.weights is None else self.weights.to(device),
            extras={k: v.to(device) for k, v in self.extras.items()},
        )


def derive_labels(regression: Mapping[str, Tensor], config: NeuralConfig) -> dict[str, Tensor]:
    """Binary event labels from raw (un-normalised) regression targets."""
    up = torch.tensor([h.upside_threshold for h in config.targets.horizons], dtype=torch.float32)
    down = torch.tensor([h.downside_threshold for h in config.targets.horizons], dtype=torch.float32)
    ret = regression["return"]
    dd = regression["max_drawdown"]
    return {
        "upside": (ret > up.to(ret.device)).float(),
        "downside": (dd > down.to(dd.device)).float(),
    }


def arrays_to_batch(arrays: Mapping[str, Array], config: NeuralConfig) -> Batch:
    """Collate a dict of row arrays (see :mod:`nardis_neural.data.loaders`) into a Batch."""
    current = torch.from_numpy(np.ascontiguousarray(arrays[KEY_CURRENT], dtype=np.float32))
    b = current.shape[0]
    sequences: dict[str, SequenceBatch] = {}
    for ts in config.features.timescales:
        vk = seq_key(ts.name, "values")
        if vk not in arrays:
            continue
        sequences[ts.name] = SequenceBatch(
            values=torch.from_numpy(np.ascontiguousarray(arrays[vk], dtype=np.float32)),
            mask=torch.from_numpy(np.ascontiguousarray(arrays[seq_key(ts.name, "mask")], dtype=np.bool_)),
            time_deltas=torch.from_numpy(
                np.ascontiguousarray(arrays[seq_key(ts.name, "time_deltas")], dtype=np.float32)
            ),
        )
    targets: TargetBatch | None = None
    if all(target_key(t) in arrays for t in REGRESSION_TASKS):
        reg = {
            t: torch.from_numpy(np.ascontiguousarray(arrays[target_key(t)], dtype=np.float32))
            for t in REGRESSION_TASKS
        }
        if KEY_TARGET_MASK in arrays:
            tmask = torch.from_numpy(np.ascontiguousarray(arrays[KEY_TARGET_MASK], dtype=np.bool_))
        else:
            tmask = torch.ones(b, len(config.targets.horizons), dtype=torch.bool)
        for t in REGRESSION_TASKS:
            finite = torch.isfinite(reg[t])
            tmask = tmask & finite
            reg[t] = torch.nan_to_num(reg[t])
        targets = TargetBatch(regression=reg, labels=derive_labels(reg, config), mask=tmask)
    graph: GraphBatch | None = None
    if config.model.graph.enabled and has_graph(arrays):
        graph = _collate_graph(arrays, b)
    weights = None
    if KEY_WEIGHT in arrays:
        weights = torch.from_numpy(np.ascontiguousarray(arrays[KEY_WEIGHT], dtype=np.float32))
    regimes = np.asarray(arrays[KEY_REGIME], dtype=np.int64) if KEY_REGIME in arrays else None
    return Batch(
        current=current,
        sequences=sequences,
        observation_ids=[str(x) for x in np.asarray(arrays[KEY_ID])],
        timestamps=np.asarray(arrays[KEY_TIMESTAMP], dtype=np.float64),
        graph=graph,
        targets=targets,
        weights=weights,
        regimes=regimes,
    )


def _collate_graph(arrays: Mapping[str, Array], b: int) -> GraphBatch:
    node_off = np.asarray(arrays["graph.node_offsets"], dtype=np.int64)
    edge_off = np.asarray(arrays["graph.edge_offsets"], dtype=np.int64)
    n_nodes = np.diff(node_off)
    n_edges = np.diff(edge_off)
    node_batch = np.repeat(np.arange(b), n_nodes)
    edge_shift = np.repeat(node_off[:-1], n_edges)
    edge_index = np.asarray(arrays["graph.edge_index"], dtype=np.int64).reshape(-1, 2) + edge_shift[:, None]
    local_target = np.asarray(arrays["graph.target_node"], dtype=np.int64)
    target = np.where((local_target >= 0) & (n_nodes > 0), node_off[:-1] + local_target, -1)
    return GraphBatch(
        node_features=torch.from_numpy(np.ascontiguousarray(arrays["graph.node_features"], dtype=np.float32)),
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index.T)),
        edge_type=torch.from_numpy(np.ascontiguousarray(arrays["graph.edge_type"], dtype=np.int64)),
        node_batch=torch.from_numpy(node_batch),
        target_node=torch.from_numpy(target),
    )


class MarketDataset(Dataset[Batch]):
    """Map-style dataset whose ``__getitem__`` takes an *array of indices* and returns a Batch.

    Used with :class:`BatchIndexSampler` and ``DataLoader(batch_size=None)`` so a whole
    batch is gathered with one vectorised (memmap-friendly) fancy-index per array.
    """

    def __init__(
        self,
        store: ArrayStore,
        config: NeuralConfig,
        indices: npt.NDArray[np.int64] | None = None,
        sample_weights: npt.NDArray[np.float32] | None = None,
    ) -> None:
        self.store = store
        self.config = config
        self.indices = (
            np.arange(len(store), dtype=np.int64) if indices is None else np.asarray(indices, np.int64)
        )
        if sample_weights is not None and len(sample_weights) != len(self.indices):
            raise ValueError("sample_weights must align with indices")
        self.sample_weights = sample_weights

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, positions: npt.NDArray[np.int64]) -> Batch:
        pos = np.asarray(positions, dtype=np.int64)
        rows = self.store.select(self.indices[pos])
        if self.sample_weights is not None:
            rows[KEY_WEIGHT] = np.asarray(self.sample_weights[pos], dtype=np.float32)
        return arrays_to_batch(rows, self.config)

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
        return self.store.timestamps[self.indices]

    def subset(self, positions: npt.NDArray[np.int64]) -> MarketDataset:
        w = None if self.sample_weights is None else self.sample_weights[positions]
        return MarketDataset(self.store, self.config, self.indices[positions], w)

    def target_array(self, task: str) -> npt.NDArray[np.float32]:
        return np.asarray(self.store[target_key(task)][self.indices], dtype=np.float32)


class BatchIndexSampler(Sampler[npt.NDArray[np.int64]]):
    """Yields arrays of dataset positions.  Optional weighted sampling with replacement."""

    def __init__(
        self,
        n: int,
        batch_size: int,
        shuffle: bool = True,
        weights: npt.NDArray[np.float64] | None = None,
        seed: int = 0,
        drop_last: bool = False,
        max_batches: int | None = None,
    ) -> None:
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.weights = weights
        self.seed = seed
        self.drop_last = drop_last
        self.max_batches = max_batches

    def __iter__(self) -> Iterator[npt.NDArray[np.int64]]:
        rng = np.random.default_rng(self.seed)
        if self.weights is not None:
            p = self.weights / self.weights.sum()
            order = rng.choice(self.n, size=self.n, replace=True, p=p)
        elif self.shuffle:
            order = rng.permutation(self.n)
        else:
            order = np.arange(self.n)
        for count, start in enumerate(range(0, self.n, self.batch_size)):
            if self.max_batches is not None and count >= self.max_batches:
                break
            chunk = order[start : start + self.batch_size]
            if len(chunk) == 0 or (self.drop_last and len(chunk) < self.batch_size):
                break
            yield chunk.astype(np.int64)

    def __len__(self) -> int:
        full = self.n // self.batch_size if self.drop_last else -(-self.n // self.batch_size)
        return full if self.max_batches is None else min(full, self.max_batches)


def _identity(batch: Any) -> Any:
    return batch


def make_loader(
    dataset: MarketDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    seed: int = 0,
    weights: npt.NDArray[np.float64] | None = None,
    max_batches: int | None = None,
) -> DataLoader[Batch]:
    sampler = BatchIndexSampler(
        len(dataset), batch_size, shuffle=shuffle, weights=weights, seed=seed, max_batches=max_batches
    )
    return DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=_identity,
        pin_memory=False,
        persistent_workers=num_workers > 0,
    )


def compute_sampling_weights(dataset: MarketDataset, config: NeuralConfig) -> npt.NDArray[np.float64] | None:
    """Class-balanced and/or rare-event oversampling weights (None when disabled)."""
    tc = config.training
    if not tc.balanced_sampling and tc.rare_event_oversampling <= 1.0:
        return None
    if not dataset.store.has_targets:
        return None
    ret = dataset.target_array("return")
    dd = dataset.target_array("max_drawdown")
    w = np.ones(len(dataset), dtype=np.float64)
    if tc.balanced_sampling:
        up_thr = np.array([h.upside_threshold for h in config.targets.horizons])
        down_thr = np.array([h.downside_threshold for h in config.targets.horizons])
        pos = (ret > up_thr).any(axis=1) | (dd > down_thr).any(axis=1)
        frac = pos.mean()
        if 0 < frac < 1:
            w *= np.where(pos, 0.5 / frac, 0.5 / (1 - frac))
    if tc.rare_event_oversampling > 1.0:
        mag = np.maximum(np.abs(ret).max(axis=1), dd.max(axis=1))
        thr = np.quantile(mag, tc.rare_event_return_quantile)
        w *= np.where(mag >= thr, tc.rare_event_oversampling, 1.0)
    return w
