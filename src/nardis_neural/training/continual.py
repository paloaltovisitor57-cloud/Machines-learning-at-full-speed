"""Continual learning orchestration.

    observation ─▶ champion predicts (challenger shadows) ─▶ outcome known
        ─▶ labelled Experience ─▶ replay buffer ─▶ enough new data?
        ─▶ candidate = clone(champion) ─▶ fine-tune on replay mixture
           (+ distillation from champion, + EWC) ─▶ offline validation
        ─▶ challenger in shadow mode ─▶ multi-gate promotion ─▶ champion (or rejection)

The production champion is never trained or mutated: candidates are deep copies, the
teacher used for distillation is a frozen copy, and model directories are immutable.

Workspace layout::

    workspace/
      config.yaml          learner configuration
      registry.json        lifecycle state + audit log
      models/<version>/    immutable checkpoints
      replay/              persisted experience replay buffer
      shadow/records.jsonl shadow predictions awaiting / with outcomes
      reports/             adaptation, retraining and promotion reports
      state.json           counters
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import BaseModel, Field

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import KEY_TARGET_MASK, Array, ArrayStore, concat_arrays, target_key
from nardis_neural.data.replay import (
    Experience,
    ExperienceReplayBuffer,
    canonical_row,
    experiences_from_arrays,
)
from nardis_neural.data.sequences import observations_to_arrays, outcomes_to_arrays
from nardis_neural.data.splits import chronological_split
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.lifecycle.candidate import create_candidate
from nardis_neural.lifecycle.champion import ModelRegistry
from nardis_neural.lifecycle.checkpoints import new_version_id
from nardis_neural.lifecycle.promotion import PromotionDecision, evaluate_promotion
from nardis_neural.lifecycle.rollback import AutoRollbackMonitor, rollback
from nardis_neural.lifecycle.shadow import ShadowEvaluator, ShadowReport
from nardis_neural.monitoring.drift import input_drift
from nardis_neural.schemas import NeuralObservation, NeuralOutcome, NeuralPrediction
from nardis_neural.training.pipeline import evaluate_engine, train_engine
from nardis_neural.training.trainer import resolve_device

OFFLINE_KEYS = ("return.rmse", "upside.log_loss", "downside.log_loss")


class LearnerState(BaseModel):
    """Counters and promotion baseline MAE, persisted in ``state.json``."""

    new_since_adapt: int = 0
    new_since_full: int = 0
    last_adapt_seq: int = -1
    last_full_seq: int = -1
    adaptations: int = 0
    full_retrains: int = 0
    promotions: int = 0
    rollbacks: int = 0
    promotion_baseline_mae: float | None = None


class CandidateReport(BaseModel):
    """Outcome of an adaptation or full retrain: offline comparison and resulting status."""

    kind: str
    candidate_version: str
    champion_version: str
    status: str
    n_train: int
    n_validation: int
    degradation: float
    reason: str
    champion_metrics: dict[str, float] = Field(default_factory=dict)
    candidate_metrics: dict[str, float] = Field(default_factory=dict)
    seconds: float = 0.0
    created_at: float = Field(default_factory=time.time)


def split_holdout(
    indices: np.ndarray[Any, np.dtype[np.int64]], timestamps: np.ndarray[Any, np.dtype[np.float64]]
) -> tuple[np.ndarray[Any, np.dtype[np.int64]], np.ndarray[Any, np.dtype[np.int64]]]:
    """Chronological halves of a validation window: (fit, holdout).

    The earlier half drives early stopping and calibration; the later half is never seen
    by the candidate and is used only for the offline champion-vs-candidate comparison.
    Windows too small to split are returned as (window, window).
    """
    order = indices[np.argsort(timestamps[indices], kind="stable")]
    if len(order) < 20:
        return order, order
    mid = len(order) // 2
    return order[:mid], order[mid:]


def offline_degradation(champion: dict[str, float], candidate: dict[str, float]) -> float:
    """Mean relative change of key validation losses (positive = candidate worse)."""
    ratios = [
        candidate[k] / champion[k] - 1.0
        for k in OFFLINE_KEYS
        if k in champion and k in candidate and np.isfinite(champion[k]) and champion[k] > 0
    ]
    return float(np.mean(ratios)) if ratios else 0.0


class ContinualLearner:
    """Stateful continual-learning service around a model registry workspace."""

    def __init__(
        self,
        workspace: str | Path,
        config: NeuralConfig | None = None,
        device: torch.device | str | None = None,
        autosave_every: int = 1000,
        max_pending: int = 200_000,
    ) -> None:
        self.root = Path(workspace)
        self.registry = ModelRegistry(self.root)
        cfg_file = self.root / "config.yaml"
        if config is not None:
            self.config = config
        elif cfg_file.exists():
            self.config = NeuralConfig.load(cfg_file)
        else:
            raise FileNotFoundError(f"{cfg_file} missing; pass a config or use ContinualLearner.initialize")
        self.device = (
            torch.device(device) if device is not None else resolve_device(self.config.training.device)
        )
        self.buffer = ExperienceReplayBuffer.load(self.root / "replay", self.config.replay)
        self.shadow = ShadowEvaluator(self.root / "shadow")
        state_file = self.root / "state.json"
        self.state = (
            LearnerState.model_validate(json.loads(state_file.read_text()))
            if state_file.exists()
            else LearnerState()
        )
        self.rollback_monitor = AutoRollbackMonitor(self.config.lifecycle, self.state.promotion_baseline_mae)
        self._engines: dict[str, NeuralEngine] = {}
        self.pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.max_pending = max_pending
        self.autosave_every = autosave_every
        self._since_save = 0

    # ------------------------------------------------------------------ setup
    @classmethod
    def initialize(
        cls,
        workspace: str | Path,
        champion: NeuralEngine,
        config: NeuralConfig | None = None,
        device: torch.device | str | None = None,
    ) -> ContinualLearner:
        """Create a workspace with ``champion`` registered as champion (unless one exists),
        save the config and return a learner on it.
        """
        root = Path(workspace)
        root.mkdir(parents=True, exist_ok=True)
        cfg = config or champion.config
        cfg.save(root / "config.yaml")
        registry = ModelRegistry(root)
        if registry.champion_version is None:
            registry.register(champion, status="champion", reason="initial champion")
        learner = cls(root, cfg, device)
        learner.save()
        return learner

    def engine(self, version: str) -> NeuralEngine:
        """Engine of ``version``, loaded from the registry once and cached."""
        if version not in self._engines:
            self._engines[version] = self.registry.load_engine(version, self.device)
        return self._engines[version]

    @property
    def champion(self) -> NeuralEngine:
        """Current champion engine; raises RuntimeError if the workspace has none."""
        v = self.registry.champion_version
        if v is None:
            raise RuntimeError("workspace has no champion")
        return self.engine(v)

    @property
    def challenger(self) -> NeuralEngine | None:
        """Current challenger engine, or None."""
        v = self.registry.challenger_version
        return None if v is None else self.engine(v)

    # ------------------------------------------------------------------ serving
    def predict_batch(self, observations: Sequence[NeuralObservation]) -> list[NeuralPrediction]:
        """Champion predictions.  A challenger, if present, runs in shadow on the *same*
        observations; its outputs are recorded but never returned."""
        if not observations:
            return []
        champion = self.champion
        arrays = observations_to_arrays(observations, champion.config)
        champ_out = champion.predict_arrays(arrays)
        challenger = self.challenger
        if challenger is not None:
            chall_out = challenger.predict_arrays(arrays)
            self.shadow.record(champ_out, chall_out, champion.version, challenger.version)
        preds = champion.to_predictions(champ_out)
        for p in preds:
            self.record_prediction(p)
        return preds

    def predict(self, observation: NeuralObservation) -> NeuralPrediction:
        """Champion prediction for one observation (see predict_batch())."""
        return self.predict_batch([observation])[0]

    def record_prediction(self, prediction: NeuralPrediction) -> None:
        """Remember a prediction so the eventual outcome can be linked to it."""
        self.pending[prediction.observation_id] = {
            "model_version": prediction.model_version,
            "expected_returns": prediction.expected_returns,
            "return_std": prediction.return_std,
            "upside_probabilities": prediction.upside_probabilities,
            "downside_probabilities": prediction.downside_probabilities,
            "confidence": prediction.confidence,
            "ood_score": prediction.ood_score,
            "total_uncertainty": prediction.total_uncertainty,
            "embedding": prediction.market_embedding,
            "regime": prediction.regime_cluster,
        }
        while len(self.pending) > self.max_pending:
            self.pending.popitem(last=False)

    # ------------------------------------------------------------------ learning signal
    def add_experience(
        self,
        observation: NeuralObservation,
        outcome: NeuralOutcome,
        prediction: NeuralPrediction | None = None,
    ) -> Experience:
        """Add a labelled outcome to replay and resolve its shadow record; returns the experience.

        The outcome is linked to its pending prediction (or ``prediction``).  If the champion
        made that prediction its live error is tracked, which may trigger an automatic
        rollback.  State is saved every ``autosave_every`` experiences.
        """
        if prediction is not None:
            self.record_prediction(prediction)
        pred = self.pending.pop(observation.observation_id, None)
        exp = Experience.create(
            observation,
            outcome,
            self.config,
            model_version=None if pred is None else pred["model_version"],
            prediction=pred,
            uncertainty=None if pred is None else pred["total_uncertainty"],
            embedding=None if pred is None else pred["embedding"],
            regime=None if pred is None else pred["regime"],
        )
        self.buffer.add(exp)
        self.state.new_since_adapt += 1
        self.state.new_since_full += 1
        targets = outcomes_to_arrays([outcome], self.config)
        self.shadow.resolve(
            observation.observation_id,
            {t: targets[target_key(t)][0] for t in REGRESSION_TASKS},
            targets[KEY_TARGET_MASK][0],
        )
        if pred is not None and pred["model_version"] == self.registry.champion_version:
            self._track_live_error(pred, exp)
        self._since_save += 1
        if self._since_save >= self.autosave_every:
            self.save()
        return exp

    def add_labelled_arrays(self, arrays: dict[str, Array], shadow: bool = True) -> int:
        """Bulk-ingest a labelled dataset (e.g. from Nardis' outcome logs) into replay.

        With ``shadow=True`` and an active challenger, both models first predict the rows
        so the shadow comparison accumulates evidence as well."""
        if shadow and self.challenger is not None:
            champ_out = self.champion.predict_arrays(arrays)
            chall_out = self.challenger.predict_arrays(arrays)
            self.shadow.record(champ_out, chall_out, self.champion.version, self.challenger.version)
        exps = experiences_from_arrays(arrays, self.config)
        for e in exps:
            self.buffer.add(e)
            self.shadow.resolve(
                e.observation_id, {t: e.target(t) for t in REGRESSION_TASKS}, e.row[KEY_TARGET_MASK][0]
            )
        self.state.new_since_adapt += len(exps)
        self.state.new_since_full += len(exps)
        self.save()
        return len(exps)

    def _track_live_error(self, pred: dict[str, Any], exp: Experience) -> None:
        mask = np.asarray(exp.row[KEY_TARGET_MASK][0], dtype=bool)
        y = exp.target("return")
        errs = [
            abs(y[h] - pred["expected_returns"][name])
            for h, name in enumerate(self.config.horizon_names)
            if mask[h] and name in pred["expected_returns"]
        ]
        if errs:
            self.rollback_monitor.update(float(np.mean(errs)))
        if self.rollback_monitor.should_rollback():
            self.rollback(
                reason=f"automatic: live MAE {self.rollback_monitor.live_mae:.5g} exceeded baseline"
            )

    # ------------------------------------------------------------------ adaptation
    def adapt_if_needed(self) -> CandidateReport | None:
        """Run adapt() once ``min_new_samples`` new experiences have arrived; otherwise None."""
        if self.state.new_since_adapt < self.config.continual.min_new_samples:
            return None
        return self.adapt()

    def adapt(self) -> CandidateReport:
        """Fine-tune a clone of the champion on a replay mixture (distillation + EWC)."""
        t0 = time.time()
        cc = self.config.continual
        champion = self.champion
        exps = self.buffer.all()
        new = [e for e in exps if e.seq > self.state.last_adapt_seq]
        if len(new) < 10:
            raise ValueError(f"only {len(new)} new experiences; need at least 10 to adapt")
        n_val = max(5, round(len(new) * cc.adapt_validation_fraction))
        val = new[-n_val:]
        cutoff = val[0].timestamp - champion.config.embargo_seconds
        eligible = {e.seq for e in exps if e.timestamp <= cutoff and e.seq not in {v.seq for v in val}}
        if not eligible:
            raise ValueError("no experiences old enough to train on without overlapping validation")
        train = self.buffer.sample_mixture(
            cc.adapt_samples,
            {
                "recent": cc.recent_fraction,
                "historical": cc.historical_fraction,
                "rare": cc.rare_fraction,
                "difficult": cc.difficult_fraction,
            },
            eligible=eligible,
            now=cutoff,
        )
        store = ArrayStore(concat_arrays([e.row for e in train + val]))
        train_idx = np.arange(len(train), dtype=np.int64)
        val_idx = np.arange(len(train), len(train) + len(val), dtype=np.int64)
        candidate = create_candidate(champion, origin="adapt")
        ewc_ds = None
        if cc.ewc_enabled and cc.ewc_weight > 0:
            hist = [e for e in self.buffer.pool("historical") if e.seq in eligible]
            if len(hist) < 32:
                hist = [e for e in exps if e.seq in eligible]
            hist = hist[-4096:]
            ewc_ds = MarketDataset(ArrayStore(concat_arrays([e.row for e in hist])), champion.config)
        fit_idx, holdout_idx = split_holdout(val_idx, store.timestamps)
        engine, _ = train_engine(
            champion.config,
            store,
            train_idx,
            fit_idx,
            version=candidate.version,
            parent_version=champion.version,
            origin="adapt",
            device=self.device,
            init_ensemble=candidate.ensemble,
            normalizer=candidate.normalizer,
            epochs=cc.adapt_epochs,
            learning_rate=cc.adapt_learning_rate,
            teacher=champion.ensemble,
            distillation_weight=cc.distillation_weight,
            ewc_dataset=ewc_ds,
            ewc_weight=cc.ewc_weight if cc.ewc_enabled else 0.0,
        )
        holdout = MarketDataset(store, champion.config, holdout_idx)
        champ_metrics, _ = evaluate_engine(champion, holdout)
        cand_metrics, _ = evaluate_engine(engine, holdout)
        rep = self._register_candidate("adapt", engine, champ_metrics, cand_metrics, len(train), len(val), t0)
        self.state.new_since_adapt = 0
        self.state.last_adapt_seq = max(e.seq for e in exps)
        self.state.adaptations += 1
        self.save()
        return rep

    # ------------------------------------------------------------------ full retraining
    def full_retrain_if_needed(self, external: ArrayStore | None = None) -> CandidateReport | None:
        """Run full_retrain() when enough new samples arrived or, if enabled, replay input drift
        is detected; otherwise None.
        """
        cc = self.config.continual
        due = self.state.new_since_full >= cc.full_retrain_min_new_samples
        if not due and cc.full_retrain_on_drift:
            due = self.replay_input_drift()
        return self.full_retrain(external) if due else None

    def replay_input_drift(self, min_samples: int = 200) -> bool:
        """True if recent replay inputs drifted from the historical pool.

        False when either pool has fewer than ``min_samples`` experiences.
        """
        recent = list(self.buffer.recent)
        hist = self.buffer.historical
        if len(recent) < min_samples or len(hist) < min_samples:
            return False
        ref = ExperienceReplayBuffer.to_arrays(hist)
        cur = ExperienceReplayBuffer.to_arrays(recent)
        return input_drift(ref, cur, self.config).drifted

    def full_retrain(self, external: ArrayStore | None = None) -> CandidateReport:
        """Train a fresh ensemble (new normaliser) on weighted recent + historical + rare
        + difficult data, optionally together with an external dataset."""
        t0 = time.time()
        cc = self.config.continual
        champion = self.champion
        exps = self.buffer.all()
        parts: list[dict[str, Array]] = []
        weights: list[np.ndarray[Any, np.dtype[np.float64]]] = []
        if exps:
            parts.append(ExperienceReplayBuffer.to_arrays(exps))
            seqs = np.array([e.seq for e in exps])
            recent_cut = np.sort(seqs)[-min(cc.full_retrain_recent_window, len(seqs))]
            prio = np.array([e.priority or 0.0 for e in exps])
            hard = prio >= np.quantile(prio, 0.8) if len(prio) else np.zeros(0, bool)
            w = np.where(seqs >= recent_cut, cc.full_retrain_recent_weight, cc.full_retrain_historical_weight)
            w = w * np.where([e.is_rare for e in exps], cc.full_retrain_rare_weight, 1.0)
            w = w * np.where(hard, cc.full_retrain_difficult_weight, 1.0)
            weights.append(w.astype(np.float64))
        if external is not None:
            ext = canonical_row(external.select(np.arange(len(external))), self.config)
            if KEY_TARGET_MASK not in ext:
                ext[KEY_TARGET_MASK] = np.ones((len(external), len(self.config.targets.horizons)), dtype=bool)
            parts.append(ext)
            weights.append(np.full(len(external), cc.full_retrain_historical_weight))
        if not parts:
            raise ValueError("no data available for full retraining")
        arrays = concat_arrays(parts) if len(parts) > 1 else parts[0]
        store = ArrayStore(arrays)
        sw = np.concatenate(weights)
        sw = (sw / max(sw.mean(), 1e-12)).astype(np.float32)
        cfg = champion.config
        split = chronological_split(
            store.timestamps, cfg.training.validation_fraction, embargo_seconds=cfg.embargo_seconds
        )
        fit_idx, holdout_idx = split_holdout(split.validation, store.timestamps)
        engine, _ = train_engine(
            cfg,
            store,
            split.train,
            fit_idx,
            version=new_version_id("full"),
            parent_version=champion.version,
            origin="full_retrain",
            device=self.device,
            epochs=cc.full_retrain_epochs,
            sample_weights=sw[split.train],
        )
        holdout = MarketDataset(store, cfg, holdout_idx)
        champ_metrics, _ = evaluate_engine(champion, holdout)
        cand_metrics, _ = evaluate_engine(engine, holdout)
        rep = self._register_candidate(
            "full_retrain",
            engine,
            champ_metrics,
            cand_metrics,
            len(split.train),
            len(split.validation),
            t0,
        )
        self.state.new_since_full = 0
        self.state.last_full_seq = max((e.seq for e in exps), default=-1)
        self.state.full_retrains += 1
        self.save()
        return rep

    def _register_candidate(
        self,
        kind: str,
        engine: NeuralEngine,
        champ_metrics: dict[str, float],
        cand_metrics: dict[str, float],
        n_train: int,
        n_val: int,
        t0: float,
    ) -> CandidateReport:
        champion_version = self.champion.version
        self.registry.register(engine, status="candidate", reason=f"{kind} candidate")
        self._engines[engine.version] = engine
        deg = offline_degradation(champ_metrics, cand_metrics)
        limit = self.config.continual.offline_max_degradation
        if deg > limit:
            status, reason = "failed", f"offline validation degradation {deg:.3f} > {limit}"
            self.registry.set_status(engine.version, "failed", reason)
        else:
            old = self.registry.challenger_version
            status, reason = "challenger", f"offline degradation {deg:.3f} ≤ {limit}; entering shadow mode"
            self.registry.set_status(engine.version, "challenger", reason)
            if old is not None:
                self.shadow.clear(old)
        rep = CandidateReport(
            kind=kind,
            candidate_version=engine.version,
            champion_version=champion_version,
            status=status,
            n_train=n_train,
            n_validation=n_val,
            degradation=deg,
            reason=reason,
            champion_metrics={k: champ_metrics[k] for k in OFFLINE_KEYS if k in champ_metrics},
            candidate_metrics={k: cand_metrics[k] for k in OFFLINE_KEYS if k in cand_metrics},
            seconds=time.time() - t0,
        )
        self._write_report(f"{kind}-{engine.version}.json", rep.model_dump(mode="json"))
        return rep

    # ------------------------------------------------------------------ shadow / promotion
    def shadow_report(self) -> ShadowReport | None:
        """Shadow comparison of champion and challenger, or None without a challenger."""
        ch = self.registry.challenger_version
        champ = self.registry.champion_version
        if ch is None or champ is None:
            return None
        return self.shadow.report(self.config, champ, ch)

    def evaluate_promotion(self) -> PromotionDecision | None:
        """Evaluate the promotion gates on the shadow report and write JSON and Markdown
        reports to ``reports/``; None without a challenger.
        """
        report = self.shadow_report()
        if report is None:
            return None
        decision = evaluate_promotion(report, self.config.promotion)
        stamp = int(time.time())
        self._write_report(
            f"promotion-{decision.challenger_version}-{stamp}.json", decision.model_dump(mode="json")
        )
        (self.root / "reports" / f"promotion-{decision.challenger_version}-{stamp}.md").write_text(
            decision.to_markdown()
        )
        return decision

    def promote_if_ready(self) -> PromotionDecision | None:
        """Evaluate promotion and promote the challenger if it passes; returns the decision."""
        decision = self.evaluate_promotion()
        if decision is None or not decision.promote:
            return decision
        self.promote(decision.challenger_version, decision)
        return decision

    def promote(self, version: str, decision: PromotionDecision | None = None) -> None:
        """Make ``version`` champion, prune old weights, reset the rollback baseline and save.

        The new baseline is the challenger's shadow return MAE from ``decision`` (None for a
        manual promotion); the version's shadow records are cleared.
        """
        reason = "manual promotion" if decision is None else decision.reason
        summary: dict[str, Any] = {"reason": reason}
        if decision is not None:
            summary["gates"] = {g.name: g.passed for g in decision.gates}
        self.registry.promote(version, reason=reason, report=summary)
        # long-running learners would otherwise keep every model version on disk
        self.registry.prune(self.config.lifecycle.keep_champions)
        baseline = None
        if decision is not None:
            baseline = decision.shadow_report.get("challenger", {}).get("return.mae")
        self.state.promotion_baseline_mae = baseline
        self.rollback_monitor.reset(baseline)
        self.shadow.clear(version)
        self.state.promotions += 1
        self.save()

    def rollback(self, to_version: str | None = None, reason: str = "manual rollback") -> str:
        """Restore ``to_version`` (default: previous champion), clear the rollback baseline and
        save; returns the new champion version.
        """
        target = rollback(self.registry, to_version, reason)
        self.state.rollbacks += 1
        self.state.promotion_baseline_mae = None
        self.rollback_monitor.reset(None)
        self.save()
        return target

    # ------------------------------------------------------------------ persistence
    def _write_report(self, name: str, payload: dict[str, Any]) -> None:
        d = self.root / "reports"
        d.mkdir(exist_ok=True)
        (d / name).write_text(json.dumps(payload, indent=2, default=float))

    def save(self) -> None:
        """Persist the replay buffer, shadow records and learner state."""
        self.buffer.save(self.root / "replay")
        self.shadow.save()
        (self.root / "state.json").write_text(self.state.model_dump_json(indent=2))
        self._since_save = 0

    def status(self) -> dict[str, Any]:
        """Champion, challenger, replay pool sizes, shadow records, counters and version statuses."""
        return {
            "champion": self.registry.champion_version,
            "challenger": self.registry.challenger_version,
            "replay_size": len(self.buffer),
            "replay_pools": {
                "recent": len(self.buffer.recent),
                "historical": len(self.buffer.historical),
                "rare": len(self.buffer.rare),
            },
            "shadow_records": len(self.shadow.records),
            "state": self.state.model_dump(),
            "versions": {v: self.registry.entry(v).status for v in self.registry.versions()},
        }
