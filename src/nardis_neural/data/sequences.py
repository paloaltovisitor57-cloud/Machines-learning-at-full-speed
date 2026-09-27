"""Sequence padding, observation → array conversion and causal bar construction."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig, TimescaleConfig
from nardis_neural.data.loaders import (
    ID_WIDTH,
    KEY_CURRENT,
    KEY_ID,
    KEY_TARGET_MASK,
    KEY_TIMESTAMP,
    Array,
    seq_key,
    target_key,
)
from nardis_neural.schemas import NeuralObservation, NeuralOutcome, SequenceInput

F32 = npt.NDArray[np.float32]


def pad_sequence(
    values: F32,
    ts: TimescaleConfig,
    time_deltas: F32 | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[F32, npt.NDArray[np.bool_], F32]:
    """Left-pad (or truncate to the most recent rows) a ``(T, F)`` sequence to ``max_len``.

    Returns ``(values, mask, time_deltas)``.  Rows that are entirely NaN are treated as
    missing.  Masked rows are zero-filled.  Missing time deltas are derived from the
    timescale resolution assuming the last row is at the observation time.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1, ts.feature_dim)
    t = values.shape[0]
    if time_deltas is None:
        time_deltas = (np.arange(t - 1, -1, -1, dtype=np.float32) * ts.resolution_seconds).astype(np.float32)
    row_mask = ~np.all(np.isnan(values), axis=1) if t else np.zeros(0, dtype=np.bool_)
    if mask is not None:
        row_mask = row_mask & np.asarray(mask, dtype=np.bool_)
    if t > ts.max_len:
        values, time_deltas, row_mask = (
            values[-ts.max_len :],
            time_deltas[-ts.max_len :],
            row_mask[-ts.max_len :],
        )
        t = ts.max_len
    out_v = np.zeros((ts.max_len, ts.feature_dim), dtype=np.float32)
    out_m = np.zeros(ts.max_len, dtype=np.bool_)
    out_d = np.zeros(ts.max_len, dtype=np.float32)
    if t:
        out_v[-t:] = np.where(row_mask[:, None], values, 0.0)
        out_m[-t:] = row_mask
        out_d[-t:] = np.where(row_mask, np.maximum(time_deltas, 0.0), 0.0)
    return out_v, out_m, out_d


def observations_to_arrays(
    observations: Sequence[NeuralObservation], config: NeuralConfig
) -> dict[str, Array]:
    """Convert typed observations into the canonical padded array layout."""
    n = len(observations)
    fc = config.features
    arrays: dict[str, Array] = {
        KEY_ID: np.asarray([o.observation_id for o in observations], dtype=f"<U{ID_WIDTH}"),
        KEY_TIMESTAMP: np.asarray([o.timestamp for o in observations], dtype=np.float64),
        KEY_CURRENT: np.zeros((n, fc.current_dim), dtype=np.float32),
    }
    for ts in fc.timescales:
        arrays[seq_key(ts.name, "values")] = np.zeros((n, ts.max_len, ts.feature_dim), np.float32)
        arrays[seq_key(ts.name, "mask")] = np.zeros((n, ts.max_len), np.bool_)
        arrays[seq_key(ts.name, "time_deltas")] = np.zeros((n, ts.max_len), np.float32)
    for i, obs in enumerate(observations):
        if obs.current_features.shape[0] != fc.current_dim:
            raise ValueError(
                f"observation {obs.observation_id}: current_features has {obs.current_features.shape[0]} "
                f"values, expected {fc.current_dim}"
            )
        arrays[KEY_CURRENT][i] = obs.current_features
        unknown = set(obs.sequences) - set(fc.timescale_names)
        if unknown:
            raise ValueError(f"observation {obs.observation_id}: unknown timescales {sorted(unknown)}")
        for ts in fc.timescales:
            seq = obs.sequences.get(ts.name)
            if seq is None:
                continue
            if seq.values.shape[1] != ts.feature_dim:
                raise ValueError(
                    f"observation {obs.observation_id}: sequence {ts.name} has {seq.values.shape[1]} "
                    f"features, expected {ts.feature_dim}"
                )
            v, m, d = pad_sequence(seq.values, ts, seq.time_deltas, seq.mask)
            arrays[seq_key(ts.name, "values")][i] = v
            arrays[seq_key(ts.name, "mask")][i] = m
            arrays[seq_key(ts.name, "time_deltas")][i] = d
    if config.model.graph.enabled:
        _graphs_to_arrays(observations, config, arrays)
    return arrays


def _graphs_to_arrays(
    observations: Sequence[NeuralObservation], config: NeuralConfig, arrays: dict[str, Array]
) -> None:
    fdim = config.features.graph.node_feature_dim
    nf, ei, et, tn = [], [], [], []
    n_nodes, n_edges = [0], [0]
    for obs in observations:
        g = obs.graph
        if g is None:
            n_nodes.append(0)
            n_edges.append(0)
            tn.append(-1)
            continue
        if g.node_features.shape[1] != fdim:
            raise ValueError(f"graph node features must have {fdim} columns")
        nf.append(g.node_features)
        ei.append(g.edge_index.T)
        etype = g.edge_type if g.edge_type is not None else np.zeros(g.edge_index.shape[1], np.int64)
        et.append(etype)
        n_nodes.append(g.node_features.shape[0])
        n_edges.append(g.edge_index.shape[1])
        tn.append(g.target_node)
    arrays["graph.node_features"] = (
        np.concatenate(nf).astype(np.float32) if nf else np.zeros((0, fdim), np.float32)
    )
    arrays["graph.edge_index"] = np.concatenate(ei).astype(np.int64) if ei else np.zeros((0, 2), np.int64)
    arrays["graph.edge_type"] = np.concatenate(et).astype(np.int64) if et else np.zeros((0,), np.int64)
    arrays["graph.node_offsets"] = np.cumsum(n_nodes).astype(np.int64)
    arrays["graph.edge_offsets"] = np.cumsum(n_edges).astype(np.int64)
    arrays["graph.target_node"] = np.asarray(tn, dtype=np.int64)


def outcomes_to_arrays(outcomes: Sequence[NeuralOutcome], config: NeuralConfig) -> dict[str, Array]:
    """Targets + mask for a list of outcomes (missing horizons are masked out)."""
    horizons = config.horizon_names
    n = len(outcomes)
    out: dict[str, Array] = {
        target_key(t): np.zeros((n, len(horizons)), np.float32) for t in REGRESSION_TASKS
    }
    mask = np.ones((n, len(horizons)), dtype=np.bool_)
    for i, oc in enumerate(outcomes):
        for task in REGRESSION_TASKS:
            vals = oc.task_values(task)
            for h, name in enumerate(horizons):
                v = vals.get(name)
                if v is None or not np.isfinite(v):
                    mask[i, h] = False
                else:
                    out[target_key(task)][i, h] = float(v)
    out[KEY_TARGET_MASK] = mask
    return out


def build_bars(
    event_times: npt.NDArray[np.float64],
    event_values: F32,
    observation_time: float,
    ts: TimescaleConfig,
    aggregation: Literal["mean", "last", "sum"] = "mean",
) -> SequenceInput:
    """Causally aggregate irregular events into fixed-resolution bars ending at ``observation_time``.

    Only events with ``time <= observation_time`` are used, so this can never leak the
    future.  Bars without events are marked missing.  Bar ``k`` (0 = oldest) covers
    ``(end - (L-k) * res, end - (L-k-1) * res]`` where ``end = observation_time``.
    """
    times = np.asarray(event_times, dtype=np.float64)
    vals = np.asarray(event_values, dtype=np.float32).reshape(len(times), -1)
    keep = times <= observation_time
    times, vals = times[keep], vals[keep]
    length, res = ts.max_len, ts.resolution_seconds
    age = observation_time - times
    bucket = length - 1 - np.floor(age / res).astype(np.int64)
    in_window = bucket >= 0
    bucket, vals = bucket[in_window], vals[in_window]
    f = vals.shape[1] if vals.ndim == 2 else ts.feature_dim
    out = np.full((length, f), np.nan, dtype=np.float32)
    counts = np.bincount(bucket, minlength=length).astype(np.float32)
    if len(bucket):
        if aggregation in ("mean", "sum"):
            sums = np.zeros((length, f), dtype=np.float64)
            np.add.at(sums, bucket, vals)
            filled = counts > 0
            agg = sums if aggregation == "sum" else sums / np.maximum(counts, 1.0)[:, None]
            out[filled] = agg[filled].astype(np.float32)
        else:
            order = np.argsort(times[in_window], kind="stable")
            out[bucket[order]] = vals[order]
    mask = counts > 0
    deltas = ((length - 1 - np.arange(length)) * res).astype(np.float32)
    return SequenceInput(values=out, time_deltas=deltas, mask=mask)


def arrays_to_observations(arrays: dict[str, Array], config: NeuralConfig) -> list[NeuralObservation]:
    """Inverse of :func:`observations_to_arrays` (only observed steps are kept)."""
    from nardis_neural.schemas import GraphInput

    n = len(arrays[KEY_ID])
    out = []
    for i in range(n):
        seqs = {}
        for ts in config.features.timescales:
            vk = seq_key(ts.name, "values")
            if vk not in arrays:
                continue
            m = np.asarray(arrays[seq_key(ts.name, "mask")][i], dtype=bool)
            if not m.any():
                continue
            seqs[ts.name] = SequenceInput(
                values=np.asarray(arrays[vk][i])[m],
                time_deltas=np.asarray(arrays[seq_key(ts.name, "time_deltas")][i])[m],
            )
        graph = None
        if "graph.node_offsets" in arrays and int(arrays["graph.target_node"][i]) >= 0:
            a, b = int(arrays["graph.node_offsets"][i]), int(arrays["graph.node_offsets"][i + 1])
            c, d = int(arrays["graph.edge_offsets"][i]), int(arrays["graph.edge_offsets"][i + 1])
            graph = GraphInput(
                node_features=np.asarray(arrays["graph.node_features"][a:b]),
                edge_index=np.asarray(arrays["graph.edge_index"][c:d]).T,
                edge_type=np.asarray(arrays["graph.edge_type"][c:d]),
                target_node=int(arrays["graph.target_node"][i]),
            )
        out.append(
            NeuralObservation(
                observation_id=str(arrays[KEY_ID][i]),
                timestamp=float(arrays[KEY_TIMESTAMP][i]),
                current_features=np.asarray(arrays[KEY_CURRENT][i]),
                sequences=seqs,
                graph=graph,
            )
        )
    return out


def arrays_to_outcomes(arrays: dict[str, Array], config: NeuralConfig) -> list[NeuralOutcome]:
    horizons = config.horizon_names
    n = len(arrays[KEY_ID])
    mask = np.asarray(arrays.get(KEY_TARGET_MASK, np.ones((n, len(horizons)), dtype=bool)), dtype=bool)
    out = []
    for i in range(n):
        vals = {
            t: {h: float(arrays[target_key(t)][i, j]) for j, h in enumerate(horizons) if mask[i, j]}
            for t in REGRESSION_TASKS
        }
        out.append(
            NeuralOutcome(
                observation_id=str(arrays[KEY_ID][i]),
                returns=vals["return"],
                max_upside=vals["max_upside"],
                max_drawdown=vals["max_drawdown"],
                volatility=vals["volatility"],
            )
        )
    return out
