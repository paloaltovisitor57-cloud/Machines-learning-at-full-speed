"""Columnar storage of observations and loaders for Parquet / NumPy / PyTorch files.

Canonical in-memory/on-disk layout is a flat mapping ``key -> ndarray`` where the first
axis indexes observations (``N``).  The native on-disk format is a directory with one
``.npy`` file per key, opened with ``mmap_mode="r"`` so arbitrarily large datasets never
need to fit in RAM.  Parquet and ``.pt`` files are converted (streaming, row-batch by
row-batch for Parquet) into this layout.

Keys
----
``observation_id`` (N,) unicode, ``timestamp`` (N,) float64, ``current`` (N, F),
``seq.<name>.values`` (N, T, F), ``seq.<name>.mask`` (N, T) bool,
``seq.<name>.time_deltas`` (N, T), ``target.<task>`` (N, H), ``target_mask`` (N, H) bool,
``regime`` (N,) int64 (optional ground truth for diagnostics), ``sample_weight`` (N,).
Optional ragged graph arrays: ``graph.node_offsets`` (N+1), ``graph.edge_offsets`` (N+1),
``graph.target_node`` (N,), ``graph.node_features`` (sum nodes, F_node),
``graph.edge_index`` (sum edges, 2) local indices, ``graph.edge_type`` (sum edges,).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig

Array = npt.NDArray[Any]

KEY_ID = "observation_id"
KEY_TIMESTAMP = "timestamp"
KEY_CURRENT = "current"
KEY_TARGET_MASK = "target_mask"
KEY_REGIME = "regime"
KEY_WEIGHT = "sample_weight"
GRAPH_RAGGED_KEYS = ("graph.node_features", "graph.edge_index", "graph.edge_type")
GRAPH_OFFSET_KEYS = ("graph.node_offsets", "graph.edge_offsets")
ID_WIDTH = 64


def seq_key(name: str, part: str) -> str:
    return f"seq.{name}.{part}"


def target_key(task: str) -> str:
    return f"target.{task}"


def has_graph(arrays: Mapping[str, Array]) -> bool:
    return "graph.node_offsets" in arrays


def _per_sample_keys(arrays: Mapping[str, Array]) -> list[str]:
    return [k for k in arrays if k not in GRAPH_RAGGED_KEYS and k not in GRAPH_OFFSET_KEYS]


def select_rows(arrays: Mapping[str, Array], indices: Array) -> dict[str, Array]:
    """Materialise rows ``indices`` (any order) including ragged graph arrays."""
    idx = np.asarray(indices, dtype=np.int64)
    out: dict[str, Array] = {}
    for key in _per_sample_keys(arrays):
        out[key] = np.asarray(arrays[key][idx])
    if has_graph(arrays):
        node_off = np.asarray(arrays["graph.node_offsets"])
        edge_off = np.asarray(arrays["graph.edge_offsets"])
        nf, ei, et = [], [], []
        n_nodes = [0]
        n_edges = [0]
        for i in idx:
            a, b = int(node_off[i]), int(node_off[i + 1])
            c, d = int(edge_off[i]), int(edge_off[i + 1])
            nf.append(np.asarray(arrays["graph.node_features"][a:b]))
            ei.append(np.asarray(arrays["graph.edge_index"][c:d]))
            et.append(np.asarray(arrays["graph.edge_type"][c:d]))
            n_nodes.append(b - a)
            n_edges.append(d - c)
        fdim = arrays["graph.node_features"].shape[1]
        out["graph.node_features"] = (
            np.concatenate(nf).astype(np.float32) if nf else np.zeros((0, fdim), np.float32)
        )
        out["graph.edge_index"] = np.concatenate(ei).astype(np.int64) if ei else np.zeros((0, 2), np.int64)
        out["graph.edge_type"] = np.concatenate(et).astype(np.int64) if et else np.zeros((0,), np.int64)
        out["graph.node_offsets"] = np.cumsum(n_nodes).astype(np.int64)
        out["graph.edge_offsets"] = np.cumsum(n_edges).astype(np.int64)
    return out


def concat_arrays(parts: Sequence[Mapping[str, Array]]) -> dict[str, Array]:
    """Concatenate several array dicts that share the same keys."""
    if not parts:
        raise ValueError("nothing to concatenate")
    keys = list(parts[0].keys())
    for p in parts[1:]:
        if set(p.keys()) != set(keys):
            raise ValueError("array dicts have different keys")
    out: dict[str, Array] = {}
    for key in keys:
        if key in GRAPH_OFFSET_KEYS:
            continue
        out[key] = np.concatenate([np.asarray(p[key]) for p in parts], axis=0)
    if has_graph(parts[0]):
        for off_key in GRAPH_OFFSET_KEYS:
            sizes = [np.diff(np.asarray(p[off_key])) for p in parts]
            out[off_key] = np.concatenate([[0], np.cumsum(np.concatenate(sizes))]).astype(np.int64)
    return out


class ArrayStore:
    """A (possibly memory-mapped) columnar collection of observations."""

    def __init__(self, arrays: Mapping[str, Array], path: Path | None = None) -> None:
        self.arrays: dict[str, Array] = dict(arrays)
        self.path = path
        if KEY_ID not in self.arrays or KEY_TIMESTAMP not in self.arrays:
            raise ValueError("arrays must contain observation_id and timestamp")
        n = len(self.arrays[KEY_ID])
        for key in _per_sample_keys(self.arrays):
            if len(self.arrays[key]) != n:
                raise ValueError(f"array {key!r} has {len(self.arrays[key])} rows, expected {n}")

    def __len__(self) -> int:
        return len(self.arrays[KEY_ID])

    def __contains__(self, key: str) -> bool:
        return key in self.arrays

    def __getitem__(self, key: str) -> Array:
        return self.arrays[key]

    @property
    def timestamps(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.arrays[KEY_TIMESTAMP], dtype=np.float64)

    @property
    def has_targets(self) -> bool:
        return all(target_key(t) in self.arrays for t in REGRESSION_TASKS)

    def select(self, indices: Array) -> dict[str, Array]:
        return select_rows(self.arrays, indices)

    def subset(self, indices: Array) -> ArrayStore:
        return ArrayStore(self.select(indices))

    def validate(self, config: NeuralConfig) -> None:
        n = len(self)
        cur = self.arrays[KEY_CURRENT]
        if cur.shape != (n, config.features.current_dim):
            raise ValueError(f"current has shape {cur.shape}, expected {(n, config.features.current_dim)}")
        for ts in config.features.timescales:
            key = seq_key(ts.name, "values")
            if key not in self.arrays:
                continue
            shape = self.arrays[key].shape
            if shape != (n, ts.max_len, ts.feature_dim):
                raise ValueError(f"{key} has shape {shape}, expected {(n, ts.max_len, ts.feature_dim)}")
        h = len(config.targets.horizons)
        for task in REGRESSION_TASKS:
            key = target_key(task)
            if key in self.arrays and self.arrays[key].shape != (n, h):
                raise ValueError(f"{key} has shape {self.arrays[key].shape}, expected {(n, h)}")

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        manifest = {"n": len(self), "keys": sorted(self.arrays)}
        for key, arr in self.arrays.items():
            np.save(path / f"{key}.npy", np.asarray(arr), allow_pickle=False)
        (path / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load(cls, directory: str | Path, mmap: bool = True) -> ArrayStore:
        path = Path(directory)
        manifest = json.loads((path / "manifest.json").read_text())
        arrays = {
            key: np.load(path / f"{key}.npy", mmap_mode="r" if mmap else None, allow_pickle=False)
            for key in manifest["keys"]
        }
        return cls(arrays, path=path)

    @classmethod
    def from_npz(cls, file: str | Path) -> ArrayStore:
        with np.load(file, allow_pickle=False) as data:
            return cls({k: data[k] for k in data.files})

    def save_npz(self, file: str | Path) -> None:
        payload: dict[str, Any] = {k: np.asarray(v) for k, v in self.arrays.items()}
        np.savez(file, **payload)

    @classmethod
    def from_torch(cls, file: str | Path) -> ArrayStore:
        raw = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
        if not isinstance(raw, dict):
            raise ValueError("a .pt dataset must contain a dict of tensors")
        arrays: dict[str, Array] = {}
        for key, value in raw.items():
            if isinstance(value, torch.Tensor):
                arrays[str(key)] = value.numpy()
            else:
                arrays[str(key)] = np.asarray(value)
        if arrays[KEY_ID].dtype.kind not in "U":
            arrays[KEY_ID] = arrays[KEY_ID].astype(str).astype(f"<U{ID_WIDTH}")
        return cls(arrays)

    def save_torch(self, file: str | Path) -> None:
        payload: dict[str, Any] = {}
        for key, arr in self.arrays.items():
            a = np.asarray(arr)
            payload[key] = a.tolist() if a.dtype.kind == "U" else torch.from_numpy(np.ascontiguousarray(a))
        torch.save(payload, file)

    # ------------------------------------------------------------------ parquet
    def to_parquet(self, file: str | Path, config: NeuralConfig, row_group_size: int = 4096) -> None:
        """Write the canonical Parquet layout (graph data is not stored in Parquet)."""
        horizons = config.horizon_names
        writer: pq.ParquetWriter | None = None
        try:
            for start in range(0, len(self), row_group_size):
                idx = np.arange(start, min(start + row_group_size, len(self)))
                rows = self.select(idx)
                cols: dict[str, Any] = {
                    KEY_ID: pa.array(rows[KEY_ID].astype(str)),
                    KEY_TIMESTAMP: pa.array(rows[KEY_TIMESTAMP].astype(np.float64)),
                    KEY_CURRENT: pa.array(list(rows[KEY_CURRENT].astype(np.float32))),
                }
                for ts in config.features.timescales:
                    vk = seq_key(ts.name, "values")
                    if vk not in rows:
                        continue
                    vals, mask = rows[vk], rows[seq_key(ts.name, "mask")]
                    dts = rows[seq_key(ts.name, "time_deltas")]
                    seq_vals, seq_dts = [], []
                    for i in range(len(idx)):
                        m = mask[i]
                        seq_vals.append([list(r) for r in vals[i][m].astype(np.float32)])
                        seq_dts.append(list(dts[i][m].astype(np.float32)))
                    cols[f"seq_{ts.name}_values"] = pa.array(seq_vals, type=pa.list_(pa.list_(pa.float32())))
                    cols[f"seq_{ts.name}_time_deltas"] = pa.array(seq_dts, type=pa.list_(pa.float32()))
                for task in REGRESSION_TASKS:
                    tk = target_key(task)
                    if tk not in rows:
                        continue
                    tmask = rows.get(KEY_TARGET_MASK)
                    for h, hname in enumerate(horizons):
                        col = rows[tk][:, h].astype(np.float64)
                        valid = None if tmask is None else ~tmask[:, h]
                        cols[f"target_{task}_{hname}"] = pa.array(col, mask=valid)
                if KEY_REGIME in rows:
                    cols[KEY_REGIME] = pa.array(rows[KEY_REGIME].astype(np.int64))
                if KEY_WEIGHT in rows:
                    cols[KEY_WEIGHT] = pa.array(rows[KEY_WEIGHT].astype(np.float32))
                table = pa.table(cols)
                if writer is None:
                    writer = pq.ParquetWriter(str(file), table.schema)
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    @classmethod
    def from_parquet(
        cls,
        file: str | Path,
        config: NeuralConfig,
        cache_dir: str | Path | None = None,
        batch_rows: int = 4096,
    ) -> ArrayStore:
        """Stream a Parquet file into (memory-mapped) arrays.

        Sequences may have any length; they are left-padded / truncated to ``max_len``.
        When ``cache_dir`` is given, arrays are written to ``.npy`` memmaps there so the
        dataset never has to be fully resident in memory.
        """
        from nardis_neural.data.sequences import pad_sequence

        pf = pq.ParquetFile(str(file))
        n = pf.metadata.num_rows
        names = set(pf.schema_arrow.names)
        horizons = config.horizon_names
        shapes: dict[str, tuple[tuple[int, ...], Any]] = {
            KEY_ID: ((n,), f"<U{ID_WIDTH}"),
            KEY_TIMESTAMP: ((n,), np.float64),
            KEY_CURRENT: ((n, config.features.current_dim), np.float32),
        }
        present_ts = [ts for ts in config.features.timescales if f"seq_{ts.name}_values" in names]
        for ts in present_ts:
            shapes[seq_key(ts.name, "values")] = ((n, ts.max_len, ts.feature_dim), np.float32)
            shapes[seq_key(ts.name, "mask")] = ((n, ts.max_len), np.bool_)
            shapes[seq_key(ts.name, "time_deltas")] = ((n, ts.max_len), np.float32)
        tasks = [t for t in REGRESSION_TASKS if all(f"target_{t}_{h}" in names for h in horizons)]
        for task in tasks:
            shapes[target_key(task)] = ((n, len(horizons)), np.float32)
        if tasks:
            shapes[KEY_TARGET_MASK] = ((n, len(horizons)), np.bool_)
        if KEY_REGIME in names:
            shapes[KEY_REGIME] = ((n,), np.int64)
        if KEY_WEIGHT in names:
            shapes[KEY_WEIGHT] = ((n,), np.float32)

        arrays: dict[str, Array] = {}
        cache = Path(cache_dir) if cache_dir is not None else None
        if cache is not None:
            cache.mkdir(parents=True, exist_ok=True)
        for key, (shape, dtype) in shapes.items():
            if cache is not None:
                arrays[key] = np.lib.format.open_memmap(
                    cache / f"{key}.npy", mode="w+", dtype=dtype, shape=shape
                )
            else:
                arrays[key] = np.zeros(shape, dtype=dtype)

        row = 0
        for rb in pf.iter_batches(batch_size=batch_rows):
            b = rb.num_rows
            sl = slice(row, row + b)
            arrays[KEY_ID][sl] = np.asarray(rb.column(KEY_ID).to_pylist(), dtype=str)
            arrays[KEY_TIMESTAMP][sl] = rb.column(KEY_TIMESTAMP).to_numpy(zero_copy_only=False)
            cur = rb.column(KEY_CURRENT).to_pylist()
            arrays[KEY_CURRENT][sl] = np.asarray(cur, dtype=np.float32)
            for ts in present_ts:
                vals = rb.column(f"seq_{ts.name}_values").to_pylist()
                dt_col = f"seq_{ts.name}_time_deltas"
                dts = rb.column(dt_col).to_pylist() if dt_col in names else [None] * b
                for i in range(b):
                    v = np.asarray(vals[i] or [], dtype=np.float32).reshape(-1, ts.feature_dim)
                    d = None if dts[i] is None else np.asarray(dts[i], dtype=np.float32)
                    pv, pm, pd = pad_sequence(v, ts, time_deltas=d)
                    arrays[seq_key(ts.name, "values")][row + i] = pv
                    arrays[seq_key(ts.name, "mask")][row + i] = pm
                    arrays[seq_key(ts.name, "time_deltas")][row + i] = pd
            for task in tasks:
                for h, hname in enumerate(horizons):
                    col = rb.column(f"target_{task}_{hname}")
                    vals_np = col.to_numpy(zero_copy_only=False).astype(np.float64)
                    valid = ~np.isnan(vals_np)
                    arrays[target_key(task)][sl, h] = np.nan_to_num(vals_np).astype(np.float32)
                    if task == tasks[0]:
                        arrays[KEY_TARGET_MASK][sl, h] = valid
                    else:
                        arrays[KEY_TARGET_MASK][sl, h] &= valid
            if KEY_REGIME in names:
                arrays[KEY_REGIME][sl] = rb.column(KEY_REGIME).to_numpy(zero_copy_only=False)
            if KEY_WEIGHT in names:
                arrays[KEY_WEIGHT][sl] = rb.column(KEY_WEIGHT).to_numpy(zero_copy_only=False)
            row += b
        if cache is not None:
            for arr in arrays.values():
                if isinstance(arr, np.memmap):
                    arr.flush()
            (cache / "manifest.json").write_text(json.dumps({"n": n, "keys": sorted(arrays)}, indent=2))
            return cls.load(cache, mmap=True)
        return cls(arrays)


def load_store(path: str | Path, config: NeuralConfig, cache_dir: str | Path | None = None) -> ArrayStore:
    """Load a dataset from a directory of ``.npy`` files, ``.npz``, ``.pt`` or ``.parquet``."""
    p = Path(path)
    if p.is_dir():
        store = ArrayStore.load(p)
    elif p.suffix == ".npz":
        store = ArrayStore.from_npz(p)
    elif p.suffix in {".pt", ".pth"}:
        store = ArrayStore.from_torch(p)
    elif p.suffix == ".parquet":
        store = ArrayStore.from_parquet(p, config, cache_dir=cache_dir)
    else:
        raise ValueError(f"unsupported dataset path: {p}")
    store.validate(config)
    return store


def fingerprint_arrays(arrays: Mapping[str, Array], max_rows: int = 4096) -> str:
    """Stable content hash of a dataset (ids, timestamps and a sample of features)."""
    import hashlib

    h = hashlib.sha256()
    n = len(arrays[KEY_ID])
    h.update(str(n).encode())
    rows = np.linspace(0, max(n - 1, 0), num=min(n, max_rows)).astype(np.int64) if n else np.array([], int)
    for key in sorted(k for k in arrays if k not in GRAPH_RAGGED_KEYS and k not in GRAPH_OFFSET_KEYS):
        arr = np.asarray(arrays[key][rows]) if n else np.asarray(arrays[key])
        h.update(key.encode())
        h.update(str(arr.shape).encode())
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()[:16]


def iter_index_chunks(n: int, chunk: int) -> Iterable[npt.NDArray[np.int64]]:
    for start in range(0, n, chunk):
        yield np.arange(start, min(start + chunk, n), dtype=np.int64)
