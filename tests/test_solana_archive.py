import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from nardis_neural.solana.archive import ArchiveMapping, normalise_trades, read_table, train_from_archive
from nardis_neural.solana.metalabel import MetaLearner


def _trades(n: int, seed: int, start: int = 0) -> pl.DataFrame:
    """A Nardis-like archive: ms timestamps, SOL in / out, a signal that drives the result."""
    rng = np.random.default_rng(seed)
    score = rng.normal(size=n)
    win = rng.random(n) < 1 / (1 + np.exp(-(2 * score - 0.5)))
    mult = np.where(win, np.exp(rng.normal(0.6 + 0.4 * score, 0.4)), rng.uniform(0.2, 0.95, n))
    t = (1_790_000_000 + np.arange(start, start + n) * 60) * 1000  # milliseconds
    return pl.DataFrame(
        {
            "id": [f"tr{start + i}" for i in range(n)],
            "token_address": [f"mint{start + i}" for i in range(n)],
            "entry_time": t,
            "exit_time": t + 90_000,
            "entry_sol": np.full(n, 0.5),
            "exit_sol": 0.5 * mult,
            "nardis_score": score,
            "noise": rng.normal(size=n),
            "note": ["x"] * n,
        }
    )


def test_normalise_finds_columns_and_converts_units() -> None:
    df = _trades(5, 0)
    out = normalise_trades(df)
    assert out.columns[:6] == ["trade_id", "mint", "t_entry", "t_exit", "multiple", "peak_multiple"]
    assert {"f::nardis_score", "f::noise"} <= set(out.columns) and "f::note" not in out.columns
    assert out["t_entry"][0] == pytest.approx(1_790_000_000)  # ms → s
    np.testing.assert_allclose(
        out.sort("trade_id")["multiple"].to_numpy()[:1], (df["exit_sol"] / 0.5).to_numpy()[:1]
    )
    with pytest.raises(ValueError, match="mint"):
        normalise_trades(df.drop("token_address"))


def test_trains_itself_from_a_partitioned_parquet_archive_and_only_adds_new_trades(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    for day, part in enumerate((_trades(400, 1), _trades(400, 2, start=400))):
        (root / f"date=2026-09-{day + 1:02d}").mkdir(parents=True)
        part.write_parquet(root / f"date=2026-09-{day + 1:02d}" / "trades.parquet")
    learner = MetaLearner(min_trades=100)
    first = train_from_archive(learner, root)
    assert first["added"] == 800 and first["refit"]["levels"]["1.0"]["deployed"]
    again = train_from_archive(learner, root)
    assert again["added"] == 0 and again["already_known"] == 800
    _trades(50, 3, start=800).write_parquet(root / "date=2026-09-01" / "more.parquet")
    assert train_from_archive(learner, root)["added"] == 50
    learner.save(tmp_path / "meta")
    assert MetaLearner.load(tmp_path / "meta").knows("tr849")


def test_reads_the_sqlite_write_buffer_with_an_explicit_mapping(tmp_path: Path) -> None:
    db = tmp_path / "buffer.db"
    df = _trades(20, 4).rename({"token_address": "ca"})
    with sqlite3.connect(db) as con:
        con.execute(
            "CREATE TABLE trades (id TEXT, ca TEXT, entry_time INT, exit_time INT, entry_sol REAL, "
            "exit_sol REAL, nardis_score REAL, noise REAL, note TEXT)"
        )
        con.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)", df.rows())
        con.execute("CREATE TABLE other (x INT)")
    with pytest.raises(ValueError, match="--table"):
        read_table(db)
    learner = MetaLearner()
    mapping = ArchiveMapping({"mint": "ca"}, features=["nardis_score"])
    out = train_from_archive(learner, db, mapping, table="trades")
    assert out["added"] == 20 and out["features"] == ["nardis_score"]


def test_a_corrupt_parquet_file_is_skipped_counted_and_given_up_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nardis_neural.solana import archive

    monkeypatch.setattr(archive, "_read_failures", {})
    root = tmp_path / "archive"
    root.mkdir()
    _trades(30, 5).write_parquet(root / "good.parquet")
    (root / "bad.parquet").write_bytes(b"not parquet at all" * 4)
    learner = MetaLearner()
    first = train_from_archive(learner, root)
    assert first["added"] == 30 and first["unreadable_files"] == 1
    for _ in range(archive.MAX_READ_FAILURES + 2):
        assert train_from_archive(learner, root)["unreadable_files"] == 1
    assert list(archive._read_failures.values()) == [archive.MAX_READ_FAILURES], "no longer retried"
    _trades(5, 6, start=30).write_parquet(root / "bad.parquet")  # repaired: read again
    fixed = train_from_archive(learner, root)
    assert fixed["added"] == 5 and fixed["unreadable_files"] == 0
    (root / "good.parquet").unlink()
    (root / "bad.parquet").write_bytes(b"broken again")
    with pytest.raises(ValueError, match="no readable"):
        read_table(root)


def test_learn_trades_learns_an_already_read_table_once_with_a_single_refit() -> None:
    from nardis_neural.solana.archive import learn_trades

    trades = normalise_trades(_trades(120, 7))
    learner = MetaLearner(min_trades=50, refit_every=10)
    refits: list[int] = []
    original = learner.refit

    def counted() -> dict[str, Any]:
        refits.append(len(learner.multiple))
        return original()

    learner.refit = counted  # type: ignore[method-assign]
    logged: list[str] = []
    out = learn_trades(learner, trades, log=logged.append)
    assert out["added"] == 120 and out["already_known"] == 0 and out["total"] == 120
    assert out["features"] == ["nardis_score", "noise"] and out["refit"]
    assert refits == [120], "one refit at the end, not one per refit_every trades"
    assert learner.refit_every == 10 and logged
    again = learn_trades(learner, trades)
    assert again["added"] == 0 and again["already_known"] == 120 and again["refit"] == {}
