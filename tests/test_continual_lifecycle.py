from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nardis_neural.config import LifecycleConfig, NeuralConfig, PromotionConfig
from nardis_neural.data.datasets import Batch, MarketDataset
from nardis_neural.data.loaders import ArrayStore, select_rows
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.data.sequences import arrays_to_observations, arrays_to_outcomes
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.lifecycle.candidate import assert_isolated, create_candidate
from nardis_neural.lifecycle.champion import ModelRegistry
from nardis_neural.lifecycle.promotion import evaluate_promotion
from nardis_neural.lifecycle.rollback import AutoRollbackMonitor, rollback
from nardis_neural.lifecycle.shadow import ShadowReport
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.training.continual import ContinualLearner, offline_degradation
from nardis_neural.training.distillation import DistillationLoss
from nardis_neural.training.ewc import EWCPenalty
from nardis_neural.training.losses import MultiTaskLoss
from nardis_neural.training.trainer import Trainer


def _param_hash(engine: NeuralEngine) -> str:
    h = hashlib.sha256()
    for k, v in sorted(engine.ensemble.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _perturbed(engine: NeuralEngine, version: str, scale: float = 0.05) -> NeuralEngine:
    other = engine.clone(version)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for p in other.ensemble.parameters():
            p.add_(torch.randn(p.shape, generator=gen) * scale)
    return other


# ---------------------------------------------------------------- isolation
def test_candidate_isolation(trained_engine: NeuralEngine) -> None:
    before = _param_hash(trained_engine)
    cand = create_candidate(trained_engine)
    assert cand.metadata.parent_version == trained_engine.version and cand.version != trained_engine.version
    with torch.no_grad():
        for p in cand.ensemble.parameters():
            p.mul_(0.0)
    cand.normalizer.current_center[:] = 123.0
    assert _param_hash(trained_engine) == before
    assert not np.any(trained_engine.normalizer.current_center == 123.0)
    with pytest.raises(AssertionError):
        assert_isolated(trained_engine, trained_engine)


def test_adaptation_trains_candidate_without_touching_champion(
    workspace: Path, later_arrays: dict[str, Any]
) -> None:
    learner = ContinualLearner(workspace, device="cpu")
    champion = learner.champion
    champ_hash = _param_hash(champion)
    member_file = learner.registry.champion_path() / "members" / "member_0.pt"
    file_hash = _file_hash(member_file)
    assert learner.adapt_if_needed() is None, "below min_new_samples nothing happens"
    learner.add_labelled_arrays(select_rows(later_arrays, np.arange(400)))
    report = learner.adapt_if_needed()
    assert report is not None
    assert report.kind == "adapt" and report.status in {"challenger", "failed"}
    assert report.champion_version == champion.version and report.n_train > 0 and report.n_validation >= 5
    assert _param_hash(champion) == champ_hash, "champion weights mutated during adaptation"
    assert _file_hash(member_file) == file_hash, "champion checkpoint changed on disk"
    assert learner.registry.champion_version == champion.version
    cand = learner.engine(report.candidate_version)
    assert cand.metadata.origin == "adapt" and cand.metadata.parent_version == champion.version
    assert _param_hash(cand) != champ_hash, "candidate must actually be fine-tuned"
    assert cand.normalizer.to_dict() == champion.normalizer.to_dict(), "adaptation keeps the input space"
    assert any(p.name.startswith("adapt-") for p in (workspace / "reports").iterdir())
    assert learner.state.new_since_adapt == 0 and learner.state.adaptations == 1


def test_full_retrain_builds_fresh_weighted_candidate(workspace: Path, later_arrays: dict[str, Any]) -> None:
    learner = ContinualLearner(workspace, device="cpu")
    assert learner.full_retrain_if_needed() is None
    learner.add_labelled_arrays(select_rows(later_arrays, np.arange(500)))
    report = learner.full_retrain_if_needed()
    assert report is not None and report.kind == "full_retrain"
    cand = learner.engine(report.candidate_version)
    assert cand.metadata.origin == "full_retrain"
    assert cand.normalizer.fitted_rows > 0
    assert cand.normalizer.to_dict() != learner.champion.normalizer.to_dict(), (
        "full retrain refits normalisation"
    )
    assert learner.state.new_since_full == 0 and learner.state.full_retrains == 1


def test_offline_degradation_metric() -> None:
    champ = {"return.rmse": 1.0, "upside.log_loss": 0.5, "downside.log_loss": 0.5}
    assert offline_degradation(champ, champ) == 0.0
    worse = {k: v * 1.5 for k, v in champ.items()}
    assert offline_degradation(champ, worse) == pytest.approx(0.5)


# ---------------------------------------------------------------- distillation / EWC
def test_distillation_teacher_is_frozen_and_detached(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    student = NardisNeuralNetwork(tiny_config).eval()
    teacher = copy.deepcopy(student)
    dist = DistillationLoss(teacher, temperature=2.0, weight=1.0)
    same = dist(student(norm_batch), norm_batch)
    assert all(v.item() < 1e-5 for v in same.values()), "identical teacher/student → zero distillation loss"
    with torch.no_grad():
        for p in student.heads.parameters():
            p.add_(0.1)
    comps = dist(student(norm_batch), norm_batch)
    assert comps["distill_cls"] > 0 and comps["distill_reg"] > 0
    sum(comps.values()).backward()  # type: ignore[union-attr]
    assert all(p.grad is None for p in teacher.parameters()), "teacher must receive no gradient"
    assert not any(p.requires_grad for p in teacher.parameters())
    assert any(p.grad is not None for p in student.parameters())


def test_ewc_penalty(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    model = NardisNeuralNetwork(tiny_config)
    norm = FeatureNormalizer.fit(base_store, np.arange(500), tiny_config)
    trainer = Trainer(tiny_config, torch.device("cpu"))
    loss = MultiTaskLoss(tiny_config)
    batches = trainer.batches(MarketDataset(base_store, tiny_config, np.arange(500)), norm, 64, shuffle=False)
    ewc = EWCPenalty.estimate(
        model, batches, lambda out, b: loss(out, b.targets)[0], weight=5.0, max_batches=3
    )
    total = sum(float(f.sum()) for f in ewc.fisher.values())
    numel = sum(f.numel() for f in ewc.fisher.values())
    assert total / numel == pytest.approx(1.0, rel=1e-4), "Fisher normalised to mean 1"
    assert ewc(model).item() == pytest.approx(0.0, abs=1e-8)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    pen = ewc(model)
    assert pen.item() > 0
    torch.autograd.backward(pen)
    assert any(p.grad is not None for p in model.parameters())


# ---------------------------------------------------------------- shadow & promotion
def _with_challenger(workspace: Path, scale: float = 0.05) -> tuple[ContinualLearner, NeuralEngine]:
    learner = ContinualLearner(workspace, device="cpu")
    challenger = _perturbed(learner.champion, "challenger-v1", scale)
    learner.registry.register(challenger, "candidate")
    learner.registry.set_status(challenger.version, "challenger", "test")
    return learner, challenger


def test_shadow_never_replaces_champion_outputs(workspace: Path, later_arrays: dict[str, Any]) -> None:
    learner, challenger = _with_challenger(workspace)
    rows = select_rows(later_arrays, np.arange(5))
    obs = arrays_to_observations(rows, learner.config)
    served = learner.predict_batch(obs)
    direct = learner.champion.predict_batch(obs)
    shadow_only = challenger.predict_batch(obs)
    for s, d, c in zip(served, direct, shadow_only, strict=True):
        assert s.model_version == learner.champion.version
        assert s.expected_returns == d.expected_returns
        assert s.expected_returns != c.expected_returns
    rec = learner.shadow.records[obs[0].observation_id]
    assert rec.challenger_version == challenger.version and not rec.resolved
    assert rec.challenger["return.mean"] == pytest.approx(
        list(shadow_only[0].expected_returns.values()), rel=1e-5
    )
    for o, oc in zip(obs, arrays_to_outcomes(rows, learner.config), strict=True):
        learner.add_experience(o, oc)
    assert learner.shadow.records[obs[0].observation_id].resolved


def test_shadow_report_and_promotion_flow(workspace: Path, later_arrays: dict[str, Any]) -> None:
    learner, challenger = _with_challenger(workspace, scale=0.0)  # identical quality → gates at the boundary
    old_champion = learner.champion.version
    rows = select_rows(later_arrays, np.arange(300))
    learner.add_labelled_arrays(rows)
    report = learner.shadow_report()
    assert report is not None and report.n == 300
    assert report.by_window and report.by_regime
    assert report.champion["return.rmse"] == pytest.approx(report.challenger["return.rmse"])
    decision = learner.promote_if_ready()
    assert decision is not None
    assert decision.promote, decision.reason
    assert learner.registry.champion_version == challenger.version
    assert learner.registry.entry(old_champion).status == "retired"
    assert list((workspace / "reports").glob("promotion-*.md"))
    assert learner.state.promotion_baseline_mae is not None
    assert learner.shadow.records == {}


def _report(n: int, champ: dict[str, float], chall: dict[str, float], windows: int = 4) -> ShadowReport:
    wins = [{"window": i, "champion": champ, "challenger": chall} for i in range(windows)]
    regimes = {"0": {"champion": champ, "challenger": chall}}
    return ShadowReport("champ", "chall", n, champ, chall, regimes, wins)


BASE = {
    "return.rmse": 1.0,
    "upside.log_loss": 0.6,
    "downside.log_loss": 0.6,
    "upside.brier": 0.2,
    "downside.brier": 0.2,
    "upside.ece": 0.05,
    "downside.ece": 0.05,
    "return.rank_corr": 0.3,
    "tail.return_mae": 2.0,
    "tail.drawdown_mae": 2.0,
    "return.nll": 0.5,
}


def test_promotion_gates() -> None:
    cfg = PromotionConfig(min_observations=100)
    better = {k: v * 0.9 if k != "return.rank_corr" else v + 0.05 for k, v in BASE.items()}
    ok = evaluate_promotion(_report(500, BASE, better), cfg)
    assert ok.promote and all(g.passed for g in ok.gates)
    assert {g.name for g in ok.gates} >= {
        "min_observations",
        "return_rmse",
        "log_loss",
        "brier",
        "calibration_ece",
        "rank_corr",
        "tail_mae",
        "uncertainty_nll",
        "time_consistency",
        "regime_consistency",
    }
    assert "PROMOTE" in ok.to_markdown()
    too_few = evaluate_promotion(_report(20, BASE, better), cfg)
    assert not too_few.promote and "min_observations" in too_few.reason
    worse_rmse = dict(better, **{"return.rmse": 1.2})
    rej = evaluate_promotion(_report(500, BASE, worse_rmse), cfg)
    assert not rej.promote and "return_rmse" in rej.reason
    # a single metric improvement is not enough: many soft gates failing blocks promotion
    mixed = {
        **BASE,
        "return.rmse": 0.9,
        "upside.ece": 0.5,
        "downside.ece": 0.5,
        "return.rank_corr": 0.0,
        "return.nll": 5.0,
        "upside.brier": 0.5,
        "downside.brier": 0.5,
    }
    assert not evaluate_promotion(_report(500, BASE, mixed), cfg).promote
    regime_bad = _report(500, BASE, better)
    regime_bad.by_regime["1"] = {"champion": BASE, "challenger": {**BASE, "return.rmse": 3.0}}
    gates = {g.name: g for g in evaluate_promotion(regime_bad, cfg).gates}
    assert not gates["regime_consistency"].passed


# ---------------------------------------------------------------- rollback / registry
def test_manual_rollback_restores_everything(workspace: Path, later_arrays: dict[str, Any]) -> None:
    learner, challenger = _with_challenger(workspace)
    original = learner.champion
    rows = select_rows(later_arrays, np.arange(10))
    before = original.predict_arrays(rows)
    learner.promote(challenger.version)
    assert learner.registry.champion_version == challenger.version
    restored = learner.rollback(reason="test")
    assert restored == original.version
    fresh = NeuralEngine.load(workspace, device="cpu")
    assert fresh.version == original.version
    after = fresh.predict_arrays(rows)
    for key in ("return.mean", "upside.prob", "embedding", "confidence"):
        np.testing.assert_array_equal(before[key], after[key])
    assert fresh.normalizer.to_dict() == original.normalizer.to_dict()
    assert fresh.calibration.to_dict()["calibrators"] == original.calibration.to_dict()["calibrators"]
    assert fresh.ensemble.size == original.ensemble.size and fresh.config == original.config
    events = [e["event"] for e in learner.registry.state.events]
    assert "rollback" in events and "promote" in events
    assert learner.registry.entry(challenger.version).status == "retired"


def test_rollback_requires_previous(workspace: Path) -> None:
    reg = ModelRegistry(workspace)
    with pytest.raises(RuntimeError):
        rollback(reg)


def test_auto_rollback_off_by_default_and_triggers_when_enabled() -> None:
    assert LifecycleConfig().auto_rollback is False
    off = AutoRollbackMonitor(LifecycleConfig(auto_rollback_min_observations=5), baseline_mae=0.1)
    for _ in range(10):
        off.update(1.0)
    assert not off.should_rollback()
    on = AutoRollbackMonitor(LifecycleConfig(auto_rollback=True, auto_rollback_min_observations=5), 0.1)
    for _ in range(4):
        on.update(1.0)
    assert not on.should_rollback(), "needs enough observations"
    on.update(1.0)
    assert on.should_rollback()
    on.reset(0.1)
    for _ in range(5):
        on.update(0.11)
    assert not on.should_rollback()


def test_auto_rollback_in_learner(workspace: Path, later_arrays: dict[str, Any]) -> None:
    learner, challenger = _with_challenger(workspace, scale=0.5)
    original = learner.champion.version
    learner.promote(challenger.version)
    learner.config.lifecycle.auto_rollback = True
    learner.config.lifecycle.auto_rollback_min_observations = 20
    learner.rollback_monitor = AutoRollbackMonitor(learner.config.lifecycle, baseline_mae=1e-6)
    rows = select_rows(later_arrays, np.arange(30))
    obs = arrays_to_observations(rows, learner.config)
    for o, oc in zip(obs, arrays_to_outcomes(rows, learner.config), strict=True):
        learner.predict(o)
        learner.add_experience(o, oc)
    assert learner.registry.champion_version == original
    assert learner.state.rollbacks == 1


def test_registry_rules(workspace: Path, trained_engine: NeuralEngine) -> None:
    reg = ModelRegistry(workspace)
    with pytest.raises(ValueError, match="immutable"):
        reg.register(trained_engine)
    with pytest.raises(ValueError):
        reg.set_status(trained_engine.version, "retired")
    with pytest.raises(ValueError):
        reg.set_status(trained_engine.version, "champion")
    failed = trained_engine.clone("failed-v")
    reg.register(failed, "candidate")
    reg.set_status("failed-v", "failed", "bad")
    removed = reg.prune(keep_champions=1)
    assert removed == ["failed-v"] and not (workspace / "models" / "failed-v").exists()
    assert reg.entry("failed-v").deleted
    with pytest.raises(FileNotFoundError):
        reg.load_engine("failed-v")
    assert ModelRegistry(workspace).entry("failed-v").history[-1].status == "failed"


def test_learner_persists_state(workspace: Path, later_arrays: dict[str, Any]) -> None:
    learner = ContinualLearner(workspace, device="cpu")
    rows = select_rows(later_arrays, np.arange(20))
    for o, oc in zip(
        arrays_to_observations(rows, learner.config), arrays_to_outcomes(rows, learner.config), strict=True
    ):
        learner.predict(o)
        learner.add_experience(o, oc)
    learner.save()
    again = ContinualLearner(workspace, device="cpu")
    assert len(again.buffer) == 20 and again.state.new_since_adapt == 20
    exp = again.buffer.all()[0]
    assert (
        exp.model_version == learner.champion.version
        and exp.embedding is not None
        and exp.priority is not None
    )
    status = again.status()
    assert status["champion"] == learner.champion.version and status["replay_size"] == 20
