from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from nardis_neural.config import CalibrationConfig, NeuralConfig, OODConfig
from nardis_neural.data.datasets import Batch, MarketDataset
from nardis_neural.data.loaders import select_rows
from nardis_neural.data.sequences import arrays_to_observations
from nardis_neural.inference.calibration import BinaryCalibrator, CalibrationSet
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.inference.ood import OODDetector
from nardis_neural.inference.uncertainty import (
    aggregate_classification,
    aggregate_regression,
    confidence_score,
    member_disagreement,
)
from nardis_neural.lifecycle.checkpoints import load_checkpoint
from nardis_neural.models.ensemble import DeepEnsemble, set_mc_dropout
from nardis_neural.schemas import NeuralPrediction
from nardis_neural.training.metrics import expected_calibration_error


# ---------------------------------------------------------------- ensemble / uncertainty
def test_ensemble_members_are_independent(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    ens = DeepEnsemble.create(tiny_config)
    assert ens.size == 2
    p0 = dict(ens.member(0).named_parameters())
    p1 = dict(ens.member(1).named_parameters())
    assert not torch.equal(p0["latent.inp.weight"], p1["latent.inp.weight"])
    assert p0["latent.inp.weight"].data_ptr() != p1["latent.inp.weight"].data_ptr()
    stacked = ens.forward_samples(norm_batch, mc_samples=2, mc_seed=1)
    assert stacked.means["return"].shape[0] == 2 * (1 + 2)
    assert stacked.embeddings.shape[0] == 2
    assert stacked.member_index.tolist() == [0, 0, 0, 1, 1, 1]
    dis = member_disagreement(stacked.means["return"], stacked.member_index)
    assert dis.shape == (norm_batch.size,) and (dis > 0).all()


def test_mc_dropout_is_seeded_and_isolated(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    ens = DeepEnsemble.create(tiny_config)
    a = ens.forward_samples(norm_batch, mc_samples=2, mc_seed=7).means["return"]
    b = ens.forward_samples(norm_batch, mc_samples=2, mc_seed=7).means["return"]
    c = ens.forward_samples(norm_batch, mc_samples=2, mc_seed=8).means["return"]
    assert torch.equal(a, b)
    assert not torch.equal(a[1], c[1])
    assert not torch.equal(a[0], a[1]), "MC dropout samples must differ from the deterministic pass"
    m = ens.member(0)
    set_mc_dropout(m, True)
    assert not m.gate.training and any(x.training for x in m.modules() if isinstance(x, torch.nn.Dropout))
    set_mc_dropout(m, False)
    assert not any(x.training for x in m.modules())


def test_uncertainty_decomposition() -> None:
    means = torch.tensor([[[1.0]], [[3.0]]])
    logvars = torch.log(torch.tensor([[[0.5]], [[1.5]]]))
    agg = aggregate_regression(means, logvars)
    assert agg.mean.item() == 2.0 and agg.epistemic_var.item() == pytest.approx(1.0)
    assert agg.aleatoric_var.item() == pytest.approx(1.0) and agg.total_var.item() == pytest.approx(2.0)
    agree = aggregate_classification(torch.zeros(4, 1, 1))
    disagree = aggregate_classification(torch.tensor([[[-6.0]], [[6.0]], [[-6.0]], [[6.0]]]))
    assert agree.mutual_information.item() == pytest.approx(0.0, abs=1e-6)
    assert disagree.mutual_information.item() > 0.6  # members confident but contradicting
    assert agree.aleatoric_entropy.item() == pytest.approx(math.log(2), rel=1e-4)
    conf, c_epi, c_ood = confidence_score(
        torch.tensor([0.0, 1.0, 1.0]), 1.0, torch.tensor([0.5, 0.5, 3.0]), 1.0
    )
    assert conf[0] == 1.0 and c_epi[1] == pytest.approx(0.5) and conf[2] < conf[1]
    assert c_ood[0] == 1.0 and c_ood[2] < 1.0


# ---------------------------------------------------------------- calibration
@pytest.mark.parametrize("method", ["temperature", "platt", "isotonic"])
def test_calibration_methods_fix_overconfidence(method: str) -> None:
    rng = np.random.default_rng(0)
    true_p = rng.random(6000)
    y = (rng.random(6000) < true_p).astype(float)
    logits = np.log(true_p / (1 - true_p)) * 3.0  # overconfident
    raw = 1 / (1 + np.exp(-logits))
    cal = BinaryCalibrator().fit(raw[:3000], y[:3000], method)
    before = expected_calibration_error(raw[3000:], y[3000:])
    after = expected_calibration_error(cal.transform(raw[3000:]), y[3000:])
    assert after < before * 0.5
    back = BinaryCalibrator.from_dict(json.loads(json.dumps(cal.to_dict())))
    np.testing.assert_allclose(back.transform(raw[:50]), cal.transform(raw[:50]))


def test_calibration_set_and_degenerate_cases(tiny_config: NeuralConfig) -> None:
    rng = np.random.default_rng(1)
    n, h = 500, 3
    probs = {t: rng.random((n, h)) for t in ("upside", "downside")}
    labels = {t: (rng.random((n, h)) < probs[t]).astype(float) for t in probs}
    labels["downside"][:, 2] = 0.0  # single class → identity
    cs = CalibrationSet(tiny_config.horizon_names).fit(
        probs, labels, np.ones((n, h), bool), tiny_config.calibration
    )
    assert cs.calibrators["downside"][2].method == "none"
    assert "ece_after" in cs.report["upside.30s"] and "reliability_after" in cs.report["upside.30s"]
    out = cs.apply("upside", probs["upside"])
    assert out.shape == (n, h) and ((out > 0) & (out < 1)).all()
    tiny = BinaryCalibrator().fit(
        probs["upside"][:10, 0], labels["upside"][:10, 0], "temperature", CalibrationConfig().min_samples
    )
    assert tiny.method == "none"


# ---------------------------------------------------------------- OOD
def test_ood_scores_shifted_embeddings_higher() -> None:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(2000, 8))
    det = OODDetector.fit(
        emb, np.abs(rng.normal(size=2000)), np.abs(rng.normal(size=2000)) * 0.1, OODConfig()
    )
    in_dist = det.score(rng.normal(size=(200, 8)), np.abs(rng.normal(size=200)), np.full(200, 0.05))
    shifted = det.score(rng.normal(size=(200, 8)) + 6.0, np.abs(rng.normal(size=200)) * 5, np.full(200, 0.5))
    assert np.median(in_dist["score"]) < 1.0 < np.median(shifted["score"])
    assert (shifted["embedding"] > in_dist["embedding"].max()).mean() > 0.9
    back = OODDetector.from_dict(json.loads(json.dumps(det.to_dict())))
    np.testing.assert_allclose(
        back.score(emb[:5], np.ones(5), np.ones(5))["score"],
        det.score(emb[:5], np.ones(5), np.ones(5))["score"],
    )


# ---------------------------------------------------------------- engine
def test_engine_prediction_schema(trained_engine: NeuralEngine, later_arrays: dict[str, Any]) -> None:
    cfg = trained_engine.config
    obs = arrays_to_observations(select_rows(later_arrays, np.arange(3)), cfg)
    pred = trained_engine.predict(obs[0])
    assert isinstance(pred, NeuralPrediction)
    assert pred.observation_id == obs[0].observation_id and pred.model_version == trained_engine.version
    for field in (
        "expected_returns",
        "upside_probabilities",
        "downside_probabilities",
        "maximum_upside",
        "maximum_drawdown",
        "predicted_volatility",
        "return_std",
    ):
        assert set(getattr(pred, field)) == {"30s", "2m", "5m"}
    assert all(0 <= p <= 1 for p in pred.upside_probabilities.values())
    assert all(v >= 0 for v in pred.maximum_drawdown.values())
    assert 0 <= pred.confidence <= 1 and pred.ood_score >= 0
    assert pred.epistemic_uncertainty >= 0 and pred.aleatoric_uncertainty > 0
    assert pred.total_uncertainty >= pred.aleatoric_uncertainty - 1e-6
    assert len(pred.market_embedding) == cfg.model.latent_dim
    assert sum(pred.expert_weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert set(pred.expert_weights) == {"transformer", "recurrent", "tcn", "tabular"}
    assert pred.return_quantiles["5m"]["q10"] < pred.return_quantiles["5m"]["q90"]
    assert pred.regime_cluster is not None
    forbidden = {"action", "signal", "buy", "sell", "order", "side", "size", "position"}
    assert not forbidden & set(pred.model_dump())


def test_engine_learned_something(trained_engine: NeuralEngine) -> None:
    m = trained_engine.metadata.training_stats["validation_metrics"]
    assert m["upside.auc"] > 0.55 and m["downside.auc"] > 0.65
    assert m["return.rank_corr"] > 0.2


def test_deterministic_inference(trained_engine: NeuralEngine, later_arrays: dict[str, Any]) -> None:
    rows = select_rows(later_arrays, np.arange(16))
    a = trained_engine.predict_arrays(rows)
    b = trained_engine.predict_arrays(rows)
    for key in ("return.mean", "upside.prob", "embedding", "confidence", "ood_score", "expert_weights"):
        np.testing.assert_array_equal(a[key], b[key])


def test_ood_lowers_confidence(trained_engine: NeuralEngine, later_arrays: dict[str, Any]) -> None:
    rows = select_rows(later_arrays, np.arange(64))
    normal = trained_engine.predict_arrays(rows)
    weird = dict(rows)
    weird["current"] = np.asarray(rows["current"]) * 50 + 20
    shifted = trained_engine.predict_arrays(weird)
    assert np.median(shifted["ood_score"]) > np.median(normal["ood_score"])
    assert np.median(shifted["confidence"]) < np.median(normal["confidence"])


def test_checkpoint_roundtrip_identical(
    tmp_path: Path, trained_engine: NeuralEngine, later_arrays: dict[str, Any]
) -> None:
    path = trained_engine.save(tmp_path / "model")
    manifest = json.loads((path / "manifest.json").read_text())
    for key in (
        "version",
        "created_at",
        "data_fingerprint",
        "ensemble",
        "training_stats",
        "reference",
        "target_definitions",
        "horizons",
        "expert_names",
        "format_version",
        "git_commit",
    ):
        assert key in manifest, key
    assert {"config.yaml", "normalizer.json", "calibration.json", "ood.json", "regimes.json", "members"} <= {
        p.name for p in path.iterdir()
    }
    with pytest.raises(FileExistsError):
        trained_engine.save(path)
    loaded = NeuralEngine.load(path, device="cpu")
    rows = select_rows(later_arrays, np.arange(20))
    a, b = trained_engine.predict_arrays(rows), loaded.predict_arrays(rows)
    for key in (
        "return.mean",
        "return.std",
        "upside.prob",
        "downside.prob",
        "embedding",
        "confidence",
        "regime",
    ):
        np.testing.assert_array_equal(a[key], b[key])
    contents = load_checkpoint(path)
    assert contents.normalizer.to_dict() == trained_engine.normalizer.to_dict()
    assert (
        contents.calibration.to_dict()["calibrators"] == trained_engine.calibration.to_dict()["calibrators"]
    )
    assert contents.metadata.ensemble["size"] == trained_engine.ensemble.size


def test_predict_dataset_order_and_clone_isolation(trained_engine: NeuralEngine, base_store: Any) -> None:
    ds = MarketDataset(base_store, trained_engine.config, np.array([10, 3, 7, 1]))
    out = trained_engine.predict_dataset(ds, batch_size=3)
    assert out["observation_id"].tolist() == [str(base_store["observation_id"][i]) for i in (10, 3, 7, 1)]
    clone = trained_engine.clone("clone-v")
    assert clone.metadata.parent_version == trained_engine.version
    with torch.no_grad():
        next(clone.ensemble.parameters()).add_(1.0)
    assert not torch.equal(next(clone.ensemble.parameters()), next(trained_engine.ensemble.parameters()))


def test_describe(trained_engine: NeuralEngine) -> None:
    info = trained_engine.describe()
    assert info["ensemble_size"] == 2 and info["has_ood"] and info["has_regimes"]
