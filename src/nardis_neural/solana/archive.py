"""Train the meta-learner from the trading system's own trade archive.

The trading system keeps its trades in SQLite (write buffer) and Parquet (permanent archive).
This module reads either, normalises the columns and replays every settled trade, in exit-time
order, into :class:`~nardis_neural.solana.metalabel.MetaLearner`.  Re-running it only adds
trades the learner has not seen (by trade id), so it can be called on a timer and the learner
keeps training itself as the archive grows.

Columns are found by common names (see :data:`ALIASES`) or mapped explicitly.  The realised
multiple can come from a ``multiple`` column, from SOL in / SOL out, or from a fractional
return.  Every other numeric column becomes a feature, unless a feature list is given.
Timestamps may be Unix seconds or milliseconds, or datetimes.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from nardis_neural.solana.metalabel import MetaLearner, TradeOutcome, TradeProposal

ALIASES: dict[str, tuple[str, ...]] = {
    "trade_id": ("trade_id", "id", "trade", "signature", "tx", "position_id", "uuid"),
    "mint": ("mint", "token", "token_address", "token_mint", "mint_address", "address"),
    "t_entry": ("t_entry", "entry_time", "entry_ts", "open_time", "opened_at", "entry_timestamp", "buy_time"),
    "t_exit": ("t_exit", "exit_time", "exit_ts", "close_time", "closed_at", "exit_timestamp", "sell_time"),
    "multiple": ("multiple", "exit_multiple", "realized_multiple", "realised_multiple", "multiplier"),
    "sol_in": ("sol_in", "entry_sol", "cost_sol", "size_sol", "stake_sol", "buy_sol", "amount_in_sol"),
    "sol_out": ("sol_out", "exit_sol", "proceeds_sol", "sell_sol", "amount_out_sol"),
    "return": ("return", "ret", "roi", "pnl_pct", "return_pct", "pnl_ratio"),
    "peak_multiple": ("peak_multiple", "max_multiple", "peak", "ath_multiple", "max_x"),
}
"""Column names recognised for each field (case-insensitive)."""

_RESERVED = set(ALIASES)


@dataclass
class ArchiveMapping:
    """Explicit column names; anything left ``None`` is found through :data:`ALIASES`."""

    columns: dict[str, str] = field(default_factory=dict)
    """Field → column, e.g. ``{"mint": "token_ca", "t_entry": "bought_at"}``."""
    features: list[str] | None = None
    """Feature columns; default: every other numeric column."""
    return_is_percent: bool = False
    """Set when the ``return`` column is in percent (50 = +50 %) rather than a fraction."""


def read_table(path: str | Path, table: str | None = None) -> pl.DataFrame:
    """Read a Parquet file or directory (recursively, partitions included) or an SQLite table."""
    p = Path(path)
    if p.suffix.lower() in (".db", ".sqlite", ".sqlite3"):
        with sqlite3.connect(f"file:{p}?mode=ro", uri=True) as con:
            names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            name = table or (names[0] if len(names) == 1 else None)
            if name is None or name not in names:
                raise ValueError(f"choose the trades table with --table (tables: {', '.join(names)})")
            cur = con.execute(f'SELECT * FROM "{name}"')
            cols = [d[0] for d in cur.description]
            return pl.DataFrame(cur.fetchall(), schema=cols, orient="row", infer_schema_length=None)
    if p.is_dir():
        files = sorted(p.rglob("*.parquet"))
        if not files:
            raise ValueError(f"no .parquet files under {p}")
        return pl.concat([pl.read_parquet(f) for f in files], how="diagonal_relaxed")
    return pl.read_parquet(p)


def _find(df: pl.DataFrame, fld: str, mapping: ArchiveMapping) -> str | None:
    if fld in mapping.columns:
        col = mapping.columns[fld]
        if col not in df.columns:
            raise ValueError(f"mapped column {col!r} for {fld} is not in the archive")
        return col
    lower = {c.lower(): c for c in df.columns}
    for alias in ALIASES[fld]:
        if alias in lower:
            return lower[alias]
    return None


def _seconds(s: pl.Series) -> np.ndarray[Any, np.dtype[np.float64]]:
    if s.dtype.is_temporal():
        return np.asarray(s.cast(pl.Datetime("us")).cast(pl.Int64), dtype=np.float64) / 1e6
    v = np.asarray(s.cast(pl.Float64), dtype=np.float64)
    return np.where(v > 1e11, v / 1000.0, v)  # milliseconds → seconds


def normalise_trades(df: pl.DataFrame, mapping: ArchiveMapping | None = None) -> pl.DataFrame:
    """One row per settled trade: ``trade_id, mint, t_entry, t_exit, multiple, peak_multiple`` +
    feature columns.  Open trades (no exit or no result) are left out."""
    mp = mapping or ArchiveMapping()
    cols = {f: _find(df, f, mp) for f in ALIASES}
    for need in ("trade_id", "mint", "t_entry"):
        if cols[need] is None:
            raise ValueError(f"cannot find the {need} column; map it (tried {', '.join(ALIASES[need])})")
    n = df.height
    t_entry = _seconds(df[cols["t_entry"]])  # type: ignore[index]
    t_exit = _seconds(df[cols["t_exit"]]) if cols["t_exit"] else np.full(n, np.nan)
    if cols["multiple"]:
        mult = np.asarray(df[cols["multiple"]].cast(pl.Float64), dtype=np.float64)
    elif cols["sol_in"] and cols["sol_out"]:
        sol_in = np.asarray(df[cols["sol_in"]].cast(pl.Float64), dtype=np.float64)
        sol_out = np.asarray(df[cols["sol_out"]].cast(pl.Float64), dtype=np.float64)
        mult = np.where(sol_in > 0, sol_out / np.where(sol_in > 0, sol_in, 1.0), np.nan)
    elif cols["return"]:
        r = np.asarray(df[cols["return"]].cast(pl.Float64), dtype=np.float64)
        mult = 1.0 + (r / 100.0 if mp.return_is_percent else r)
    else:
        raise ValueError("cannot find the result: need a multiple, sol_in + sol_out, or a return column")
    peak = (
        np.asarray(df[cols["peak_multiple"]].cast(pl.Float64), dtype=np.float64)
        if cols["peak_multiple"]
        else np.full(n, np.nan)
    )
    used = {c for c in cols.values() if c}
    if mp.features is not None:
        missing = [f for f in mp.features if f not in df.columns]
        if missing:
            raise ValueError(f"feature columns not in the archive: {missing}")
        feats = list(mp.features)
    else:
        feats = [
            c
            for c in df.columns
            if c not in used
            and c.lower() not in _RESERVED
            and (df[c].dtype.is_numeric() or df[c].dtype == pl.Boolean)
        ]
    out = pl.DataFrame(
        {
            "trade_id": df[cols["trade_id"]].cast(pl.Utf8),  # type: ignore[index]
            "mint": df[cols["mint"]].cast(pl.Utf8),  # type: ignore[index]
            "t_entry": t_entry,
            "t_exit": np.where(np.isnan(t_exit), t_entry, t_exit),
            "multiple": mult,
            "peak_multiple": peak,
        }
    )
    for c in feats:
        out = out.with_columns(df[c].cast(pl.Float64).alias(f"f::{c}"))
    return out.filter(pl.col("multiple").is_finite() & (pl.col("multiple") >= 0)).sort("t_exit")


def train_from_archive(
    learner: MetaLearner,
    path: str | Path,
    mapping: ArchiveMapping | None = None,
    table: str | None = None,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Replay every settled trade the learner has not seen, in exit-time order, then refit."""
    say = log or (lambda _m: None)
    trades = normalise_trades(read_table(path, table), mapping)
    feat_cols = [c for c in trades.columns if c.startswith("f::")]
    added = skipped = 0
    every, learner.refit_every = learner.refit_every, 1 << 62  # one refit at the end, not per batch
    try:
        added, skipped = _replay(learner, trades, feat_cols)
    finally:
        learner.refit_every = every
    report: dict[str, Any] = {}
    if added and len(learner.multiple) >= learner.min_trades:
        report = learner.refit()
    say(f"archive: {added} new trades learned, {skipped} already known, {len(learner.multiple)} in total")
    return {
        "added": added,
        "already_known": skipped,
        "total": len(learner.multiple),
        "features": [c[3:] for c in feat_cols],
        "refit": report,
    }


def _replay(learner: MetaLearner, trades: pl.DataFrame, feat_cols: list[str]) -> tuple[int, int]:
    added = skipped = 0
    for row in trades.iter_rows(named=True):
        tid = str(row["trade_id"])
        if learner.knows(tid):
            skipped += 1
            continue
        feats = {c[3:]: float(row[c]) for c in feat_cols if row[c] is not None and np.isfinite(float(row[c]))}
        learner.advise(TradeProposal(tid, str(row["mint"]), float(row["t_entry"]), feats), record=True)
        peak = row["peak_multiple"]
        learner.settle(
            TradeOutcome(
                tid,
                float(row["t_exit"]),
                float(row["multiple"]),
                float(peak) if peak is not None and np.isfinite(peak) else None,
            )
        )
        added += 1
    return added, skipped
