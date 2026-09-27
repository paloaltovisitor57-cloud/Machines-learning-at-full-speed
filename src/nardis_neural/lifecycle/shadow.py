"""Shadow mode: the challenger sees exactly the champion's observations, its predictions
are recorded, and it *never* influences what is returned to the caller.

When outcomes arrive, records are resolved and the two models are compared on
regression error, log loss, Brier score, calibration, ranking, uncertainty quality,
tail-event behaviour, per horizon, per regime and per time window.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.config import CLASSIFICATION_TASKS, REGRESSION_TASKS, NeuralConfig
from nardis_neural.data.datasets import MarketDataset, make_loader
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.training.metrics import evaluate_predictions, summarize_groups

Array = npt.NDArray[Any]
SHADOW_KEYS: tuple[str, ...] = (
    *(f"{t}.mean" for t in REGRESSION_TASKS),
    *(f"{t}.std" for t in REGRESSION_TASKS),
    *(f"{t}.prob" for t in CLASSIFICATION_TASKS),
    "return.quantiles",
)


@dataclass
class ShadowRecord:
    observation_id: str
    timestamp: float
    regime: int
    champion_version: str
    challenger_version: str
    champion: dict[str, list[Any]]
    challenger: dict[str, list[Any]]
    targets: dict[str, list[float]] | None = None
    mask: list[bool] | None = None

    @property
    def resolved(self) -> bool:
        return self.targets is not None


@dataclass
class ShadowReport:
    champion_version: str
    challenger_version: str
    n: int
    champion: dict[str, float] = field(default_factory=dict)
    challenger: dict[str, float] = field(default_factory=dict)
    by_regime: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    by_window: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _slice(out: dict[str, Array], i: int) -> dict[str, list[Any]]:
    return {k: np.asarray(out[k][i]).tolist() for k in SHADOW_KEYS if k in out}


class ShadowEvaluator:
    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else None
        self.records: dict[str, ShadowRecord] = {}
        if self.directory is not None and (self.directory / "records.jsonl").exists():
            self.load()

    # ------------------------------------------------------------------ recording
    def record(
        self,
        champion_out: dict[str, Array],
        challenger_out: dict[str, Array],
        champion_version: str,
        challenger_version: str,
    ) -> None:
        ids = champion_out["observation_id"]
        if not np.array_equal(ids, challenger_out["observation_id"]):
            raise ValueError("shadow models must receive identical observations")
        for i, oid in enumerate(ids):
            self.records[str(oid)] = ShadowRecord(
                observation_id=str(oid),
                timestamp=float(champion_out["timestamp"][i]),
                regime=int(champion_out["regime"][i]) if "regime" in champion_out else -1,
                champion_version=champion_version,
                challenger_version=challenger_version,
                champion=_slice(champion_out, i),
                challenger=_slice(challenger_out, i),
            )

    def resolve(self, observation_id: str, targets: dict[str, Array], mask: Array) -> bool:
        rec = self.records.get(observation_id)
        if rec is None:
            return False
        rec.targets = {t: np.asarray(targets[t], dtype=float).tolist() for t in REGRESSION_TASKS}
        rec.mask = np.asarray(mask, dtype=bool).tolist()
        return True

    def resolved_records(self, champion_version: str, challenger_version: str) -> list[ShadowRecord]:
        return sorted(
            (
                r
                for r in self.records.values()
                if r.resolved
                and r.champion_version == champion_version
                and r.challenger_version == challenger_version
            ),
            key=lambda r: r.timestamp,
        )

    def clear(self, challenger_version: str | None = None) -> None:
        if challenger_version is None:
            self.records.clear()
        else:
            self.records = {
                k: r for k, r in self.records.items() if r.challenger_version != challenger_version
            }
        self.save()

    # ------------------------------------------------------------------ comparison
    def report(self, config: NeuralConfig, champion_version: str, challenger_version: str) -> ShadowReport:
        recs = self.resolved_records(champion_version, challenger_version)
        rep = ShadowReport(champion_version, challenger_version, n=len(recs))
        if len(recs) < 2:
            return rep

        def stack(side: str) -> dict[str, Array]:
            keys = getattr(recs[0], side).keys()
            return {k: np.asarray([getattr(r, side)[k] for r in recs], dtype=np.float64) for k in keys}

        champ, chall = stack("champion"), stack("challenger")
        targets = {
            t: np.asarray([r.targets[t] for r in recs if r.targets], dtype=np.float64)
            for t in REGRESSION_TASKS
        }
        mask = np.asarray([r.mask for r in recs], dtype=bool)
        rep.champion = evaluate_predictions(champ, targets, config, mask)
        rep.challenger = evaluate_predictions(chall, targets, config, mask)
        regimes = np.asarray([r.regime for r in recs])
        min_n = config.promotion.min_regime_observations
        g_champ = summarize_groups(champ, targets, regimes, config, mask, min_count=min_n)
        g_chall = summarize_groups(chall, targets, regimes, config, mask, min_count=min_n)
        rep.by_regime = {
            g: {"champion": g_champ[g], "challenger": g_chall[g]} for g in g_champ if g in g_chall
        }
        windows = np.array_split(np.arange(len(recs)), config.promotion.time_windows)
        for w, idx in enumerate(windows):
            if len(idx) < 5:
                continue
            sub_champ = {k: v[idx] for k, v in champ.items()}
            sub_chall = {k: v[idx] for k, v in chall.items()}
            sub_targets = {k: v[idx] for k, v in targets.items()}
            rep.by_window.append(
                {
                    "window": w,
                    "start": recs[int(idx[0])].timestamp,
                    "end": recs[int(idx[-1])].timestamp,
                    "champion": evaluate_predictions(
                        sub_champ, sub_targets, config, mask[idx], per_horizon=False
                    ),
                    "challenger": evaluate_predictions(
                        sub_chall, sub_targets, config, mask[idx], per_horizon=False
                    ),
                }
            )
        return rep

    def evaluate_dataset(
        self, champion: NeuralEngine, challenger: NeuralEngine, dataset: MarketDataset, batch_size: int = 512
    ) -> ShadowReport:
        """Offline shadow run: feed identical labelled batches to both, record and resolve."""
        from nardis_neural.training.pipeline import dataset_targets

        targets, mask = dataset_targets(dataset)
        ids = [str(x) for x in np.asarray(dataset.store["observation_id"])[dataset.indices]]
        pos = {oid: i for i, oid in enumerate(ids)}
        for batch in make_loader(dataset, batch_size, shuffle=False):
            a = champion.forward_arrays(batch)
            b = challenger.forward_arrays(batch)
            self.record(a, b, champion.version, challenger.version)
            for oid in batch.observation_ids:
                i = pos[oid]
                self.resolve(oid, {t: targets[t][i] for t in REGRESSION_TASKS}, mask[i])
        self.save()
        return self.report(champion.config, champion.version, challenger.version)

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        if self.directory is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        tmp = self.directory / "records.jsonl.tmp"
        with tmp.open("w") as fh:
            for r in self.records.values():
                fh.write(json.dumps(asdict(r)) + "\n")
        tmp.replace(self.directory / "records.jsonl")

    def load(self) -> None:
        assert self.directory is not None
        self.records = {}
        with (self.directory / "records.jsonl").open() as fh:
            for line in fh:
                if line.strip():
                    rec = ShadowRecord(**json.loads(line))
                    self.records[rec.observation_id] = rec
