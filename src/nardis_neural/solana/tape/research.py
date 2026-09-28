"""Tape research: does reading the raw tape beat the aggregate-feature tail model?

Same causal protocol as :mod:`moonshot.research`: entry-window snapshots, tokens split by
launch time, training labels truncated (censored) at the cutoff, one scoring pass on the
later tokens.  The Tape Transformer and the raw-feature tail model are trained on the
*same* rows and scored on the *same* test rows, so the comparison is direct:

* tail: NLL of the log peak multiple, calibration and ranking AUC per level;
* collapse: for every window, predicted vs observed collapse rate, AUC and Brier score on
  rows whose outcome for that window is known;
* one ticket per token (expected ladder payoff ≥ ticket) for both models, against buying
  every launch.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.config import NeuralConfig
from nardis_neural.solana.config import CURRENT_FEATURES, SolanaConfig
from nardis_neural.solana.market import EventStore
from nardis_neural.solana.moonshot.labels import MoonshotSpec
from nardis_neural.solana.moonshot.research import (
    MoonshotDataset,
    _calibration,
    build_moonshot_dataset,
    first_signal,
    ticket_stats,
)
from nardis_neural.solana.moonshot.tail import TailModel
from nardis_neural.solana.tape.dataset import replay_tapes
from nardis_neural.solana.tape.features import TapeSpec
from nardis_neural.solana.tape.model import TapeModel
from nardis_neural.training.metrics import brier_score, roc_auc

Logger = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


@dataclass
class TapeResearch:
    """Report, production model (refitted on every token) and the candidate dataset."""

    report: dict[str, Any]
    model: TapeModel
    dataset: MoonshotDataset


def _collapse_metrics(
    pred: npt.NDArray[np.float64],
    collapse_time: npt.NDArray[np.float64],
    observed: npt.NDArray[np.float64],
    bins: tuple[float, ...],
) -> list[dict[str, float]]:
    rows = []
    for j, edge in enumerate(bins):
        hit = np.isfinite(collapse_time) & (collapse_time <= edge)
        known = hit | (observed >= edge)
        if not known.any():
            continue
        p, y = pred[known, j], hit[known].astype(np.float64)
        rows.append(
            {
                "window_seconds": edge,
                "rows": int(known.sum()),
                "predicted": float(p.mean()),
                "observed": float(y.mean()),
                "auc": roc_auc(p, y) if 0 < y.sum() < len(y) else float("nan"),
                "brier": brier_score(p, y),
            }
        )
    return rows


def run_tape_research(
    store: EventStore,
    cfg: SolanaConfig,
    ncfg: NeuralConfig,
    spec: MoonshotSpec | None = None,
    tape: TapeSpec | None = None,
    test_fraction: float = 0.35,
    members: int = 3,
    epochs: int = 40,
    min_expected_multiple: float = 1.0,
    archetypes: dict[str, str] | None = None,
    log: Logger = _quiet,
    seed: int = 0,
) -> TapeResearch:
    """Build tapes causally, train and score the Tape Transformer against the tail model."""
    spec = spec or MoonshotSpec()
    tape = tape or TapeSpec()
    mds = build_moonshot_dataset(store, cfg, spec, ncfg, archetypes)
    ts, mints = mds.timestamps, mds.mints
    log(f"tape dataset: {len(mds)} entry-window snapshots of {len(set(mints.tolist()))} tokens")
    tx, tw, tm = replay_tapes(store, cfg, tape, list(zip(mints.tolist(), ts.tolist(), strict=True)))
    cur = mds.base.current[mds.rows]

    launch = {str(m): mds.market.token(str(m)).launch.t for m in np.unique(mints)}
    tokens = sorted(launch, key=launch.__getitem__)
    n_test = max(1, round(len(tokens) * test_fraction))
    if len(tokens) - n_test < 5:
        raise ValueError("not enough tokens for a train / test split")
    test_tokens = set(tokens[-n_test:])
    is_test = np.array([str(m) in test_tokens for m in mints.tolist()])
    cutoff = float(ts[is_test].min())
    tl = mds.labels(cutoff)
    train = np.flatnonzero(~is_test & tl.valid & (ts + spec.latency_seconds < cutoff))
    log(f"train: {len(train)} rows of {len(tokens) - n_test} tokens (labels censored at the cutoff)")

    model = TapeModel(
        cur.shape[1], spec, tape, members=members, seed=seed, feature_names=list(CURRENT_FEATURES)
    )
    fit = model.fit(
        tx[train], tw[train], tm[train], cur[train],
        tl.peak[train], tl.censored[train], tl.collapse_time[train], tl.observed[train],
        ts[train], mints[train], epochs=epochs,
    )  # fmt: skip
    log(f"tape model: {fit['parameters_per_member']:,} parameters per member")
    tail = TailModel(cur.shape[1], spec, members=members, feature_names=list(CURRENT_FEATURES), seed=seed)
    tail.fit(cur[train], tl.peak[train], tl.censored[train], ts[train], mints[train])

    lab = mds.labels()
    test = np.flatnonzero(is_test & lab.valid)
    peak, cens = lab.peak[test], lab.censored[test]
    pred = model.predict(tx[test], tw[test], tm[test], cur[test])
    tail_pred = tail.predict(cur[test])
    ladder = lab.ladder[test]

    def portfolio(expected: npt.NDArray[np.float64], kelly: npt.NDArray[np.float64]) -> dict[str, float]:
        picked = first_signal(ts[test], mints[test], (expected >= min_expected_multiple) & (kelly > 0))
        return ticket_stats(ladder[picked], peak[picked], spec.size_sol)

    every = first_signal(ts[test], mints[test], np.ones(len(test), dtype=bool))
    report: dict[str, Any] = {
        "spec": spec.model_dump(),
        "tape": tape.model_dump(),
        "rows": {"candidates": len(mds), "train": len(train), "test": len(test)},
        "tokens": {"train": len(tokens) - n_test, "test": n_test},
        "tape_model": fit,
        "test_nll": {
            "tape": float(model.nll(tx[test], tw[test], tm[test], cur[test], peak, cens).mean()),
            "tail": float(tail.nll(cur[test], peak, cens).mean()),
        },
        "calibration": {
            "tape": _calibration(pred.tail.survival, peak, cens, spec.levels),
            "tail": _calibration(tail_pred.survival, peak, cens, spec.levels),
        },
        "collapse": _collapse_metrics(
            pred.collapse, lab.collapse_time[test], lab.observed[test], model.collapse_bins
        ),
        "portfolio": {
            "tape": portfolio(pred.tail.expected_multiple, pred.tail.lottery_kelly),
            "tail": portfolio(tail_pred.expected_multiple, tail_pred.lottery_kelly),
            "every_launch": ticket_stats(ladder[every], peak[every], spec.size_sol),
        },
    }
    report["verdict"] = {
        "tape_beats_tail_nll": report["test_nll"]["tape"] < report["test_nll"]["tail"],
        "tape_more_pnl_than_tail": report["portfolio"]["tape"].get("total_pnl_sol", -np.inf)
        > report["portfolio"]["tail"].get("total_pnl_sol", np.inf),
    }
    log("refitting the production tape model on every token")
    rows_all = np.flatnonzero(lab.valid)
    final = TapeModel(
        cur.shape[1], spec, tape, members=members, seed=seed, feature_names=list(CURRENT_FEATURES)
    )
    final.fit(
        tx[rows_all], tw[rows_all], tm[rows_all], cur[rows_all],
        lab.peak[rows_all], lab.censored[rows_all], lab.collapse_time[rows_all], lab.observed[rows_all],
        ts[rows_all], mints[rows_all], epochs=epochs,
    )  # fmt: skip
    return TapeResearch(report, final, mds)


def tape_markdown(report: dict[str, Any]) -> str:
    """Human-readable research report (tail vs tape, collapse windows, tickets)."""
    nll = report["test_nll"]
    lines = [
        "# Tape Transformer research report (out-of-sample tokens)",
        "",
        f"- rows {report['rows']} · tokens {report['tokens']} · "
        f"{report['tape_model'].get('parameters_per_member', 0):,} parameters per member",
        f"- test NLL of log peak multiple: tape {nll['tape']:.3f} · raw-feature tail model {nll['tail']:.3f}",
        "",
        "## P(peak ≥ k): AUC (tape / tail) and calibration",
        "",
        "| k | resolved rows | tape predicted | tail predicted | observed | AUC tape | AUC tail |",
        "|---|---|---|---|---|---|---|",
    ]
    tail_by = {c["level"]: c for c in report["calibration"]["tail"]}
    for c in report["calibration"]["tape"]:
        t = tail_by.get(c["level"], {})
        t_pred, t_auc = t.get("predicted", float("nan")), t.get("auc", float("nan"))
        lines.append(
            f"| {c['level']:g}x | {c['rows']} | {c['predicted']:.3f} | {t_pred:.3f} | "
            f"{c['observed']:.3f} | {c['auc']:.3f} | {t_auc:.3f} |"
        )
    lines += [
        "",
        "## Collapse (value halves) within each window",
        "",
        "| window | known rows | predicted | observed | AUC | Brier |",
        "|---|---|---|---|---|---|",
    ]
    for c in report["collapse"]:
        lines.append(
            f"| {c['window_seconds']:g} s | {c['rows']} | {c['predicted']:.3f} | {c['observed']:.3f} | "
            f"{c['auc']:.3f} | {c['brier']:.3f} |"
        )
    lines += [
        "",
        "## One ticket per token",
        "",
        "| policy | tickets | total PnL (SOL) | mean multiple | median | best |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in report["portfolio"].items():
        if s.get("tickets", 0):
            lines.append(
                f"| {name} | {int(s['tickets'])} | {s['total_pnl_sol']:+.2f} | {s['mean_multiple']:.2f}x | "
                f"{s['median_multiple']:.2f}x | {s['best_multiple']:.1f}x |"
            )
        else:
            lines.append(f"| {name} | 0 | | | | |")
    v = report["verdict"]
    lines += [
        "",
        f"**Verdict:** tape beats tail NLL: {v['tape_beats_tail_nll']} · tape earns more than tail: "
        f"{v['tape_more_pnl_than_tail']}",
        "",
        "Paper research only; fat-tailed results rest on a handful of tokens.",
    ]
    return "\n".join(lines) + "\n"
