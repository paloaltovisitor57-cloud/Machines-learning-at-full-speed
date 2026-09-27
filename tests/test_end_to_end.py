"""Synthetic end-to-end integration test of the whole ML system.

synthetic data → dataset → train ensemble → predict → uncertainty → new experiences
→ replay → candidate adaptation → shadow comparison → promotion → checkpoint → reload
→ identical deterministic inference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nardis_neural import ContinualLearner, NeuralEngine, NeuralObservation, NeuralOutcome
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.loaders import ArrayStore, select_rows
from nardis_neural.data.sequences import arrays_to_observations, arrays_to_outcomes
from nardis_neural.synthetic import SyntheticSpec, generate_synthetic
from nardis_neural.training.pipeline import evaluate_engine, train_engine
from tests.conftest import make_tiny_config


@pytest.mark.slow
def test_full_pipeline(tmp_path: Path) -> None:
    cfg = make_tiny_config()
    cfg.training.epochs = 4
    cfg.continual.min_new_samples = 300
    cfg.continual.adapt_samples = 600
    cfg.continual.offline_max_degradation = 1.0  # synthetic smoke data: let the shadow gates decide
    cfg.promotion.min_observations = 200

    # 1. synthetic data → dataset
    arrays = generate_synthetic(cfg, SyntheticSpec(n_observations=3000, seed=11))
    history = ArrayStore(select_rows(arrays, np.arange(1500)))
    live = select_rows(arrays, np.arange(1500, 2200))
    future = select_rows(arrays, np.arange(2200, 3000))
    assert history.timestamps.max() <= live["timestamp"].min() <= future["timestamp"].min()

    # 2. train an ensemble (chronological split + embargo, calibration, OOD, regimes)
    engine, report = train_engine(cfg, history, device=torch.device("cpu"))
    assert engine.ensemble.size == 2 and len(report.member_results) == 2
    m = report.validation_metrics
    assert m["downside.auc"] > 0.65 and m["return.rank_corr"] > 0.2, "the ensemble must learn real structure"

    # 3. predict + uncertainty through the public API
    observations: list[NeuralObservation] = arrays_to_observations(live, cfg)
    outcomes: list[NeuralOutcome] = arrays_to_outcomes(live, cfg)
    pred = engine.predict(observations[0])
    assert pred.epistemic_uncertainty > 0 and 0 < pred.confidence <= 1
    assert pred.total_uncertainty >= pred.aleatoric_uncertainty
    assert abs(sum(pred.expert_weights.values()) - 1) < 1e-4

    # 4. continual learning: serve, receive outcomes, fill the replay buffer
    learner = ContinualLearner.initialize(tmp_path / "ws", engine, cfg, device="cpu")
    champion_version = learner.champion.version
    preds = learner.predict_batch(observations)
    assert all(p.model_version == champion_version for p in preds)
    for obs, outcome in zip(observations, outcomes, strict=True):
        learner.add_experience(obs, outcome)
    assert len(learner.buffer) == len(observations)
    assert learner.buffer.all()[0].model_version == champion_version

    # 5. candidate adaptation (clone + replay mixture + distillation + EWC)
    adaptation = learner.adapt_if_needed()
    assert adaptation is not None and adaptation.status == "challenger", adaptation
    challenger = learner.challenger
    assert challenger is not None and challenger.metadata.parent_version == champion_version

    # 6. shadow comparison on genuinely new data: the champion keeps serving
    new_obs = arrays_to_observations(future, cfg)
    new_out = arrays_to_outcomes(future, cfg)
    served = learner.predict_batch(new_obs)
    assert all(p.model_version == champion_version for p in served)
    for obs, outcome in zip(new_obs, new_out, strict=True):
        learner.add_experience(obs, outcome)
    shadow = learner.shadow_report()
    assert shadow is not None and shadow.n == len(new_obs)

    # 7. promotion decision (auditable); force it if the gates reject so the rest is exercised
    decision = learner.promote_if_ready()
    assert decision is not None and decision.gates
    if not decision.promote:
        learner.promote(challenger.version)
    assert learner.registry.champion_version == challenger.version

    # 8. checkpoint → reload → identical deterministic inference
    reloaded = NeuralEngine.load(tmp_path / "ws", device="cpu")
    assert reloaded.version == challenger.version
    probe = select_rows(future, np.arange(64))
    a = learner.champion.predict_arrays(probe)
    b = reloaded.predict_arrays(probe)
    c = reloaded.predict_arrays(probe)
    for key in (
        "return.mean",
        "return.std",
        "upside.prob",
        "downside.prob",
        "embedding",
        "expert_weights",
        "confidence",
        "ood_score",
        "epistemic",
        "regime",
    ):
        np.testing.assert_array_equal(a[key], b[key])
        np.testing.assert_array_equal(b[key], c[key])
    metrics, _ = evaluate_engine(reloaded, MarketDataset(ArrayStore(future), cfg))
    assert np.isfinite(metrics["return.rmse"])

    # 9. rollback restores the original champion exactly
    learner.rollback(reason="e2e")
    assert NeuralEngine.load(tmp_path / "ws", device="cpu").version == champion_version
