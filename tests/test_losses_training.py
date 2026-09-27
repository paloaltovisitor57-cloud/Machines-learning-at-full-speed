from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import Batch, MarketDataset, TargetBatch
from nardis_neural.data.loaders import ArrayStore
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.models.heads import HorizonHeads
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.training.losses import (
    MultiTaskLoss,
    estimate_pos_weight,
    focal_loss,
    gaussian_nll,
    pinball,
)
from nardis_neural.training.metrics import (
    brier_score,
    evaluate_predictions,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from nardis_neural.training.pipeline import build_objective
from nardis_neural.training.scheduling import build_optimizer, build_scheduler, warmup_cosine
from nardis_neural.training.trainer import NonFiniteLossError, Trainer, amp_settings, resolve_device


# ---------------------------------------------------------------- losses
def test_gaussian_nll_is_minimised_at_true_variance() -> None:
    y = torch.randn(20000) * 2.0
    mean = torch.zeros_like(y)
    nll = [gaussian_nll(mean, torch.full_like(y, math.log(v)), y).mean().item() for v in (1.0, 4.0, 9.0)]
    assert nll[1] < nll[0] and nll[1] < nll[2]


def test_pinball_and_focal() -> None:
    q = torch.tensor([[[-1.0, 0.0, 1.0]]])
    y = torch.tensor([[0.0]])
    levels = torch.tensor([0.1, 0.5, 0.9])
    # a well-placed 10%/90% fan costs 0.1 per side; a mis-ordered one costs 0.9 per side
    assert pinball(q, y, levels).item() == pytest.approx((0.1 + 0 + 0.1) / 3)
    assert pinball(q.flip(-1), y, levels).item() == pytest.approx((0.9 + 0 + 0.9) / 3)
    logits = torch.tensor([4.0, -4.0])
    labels = torch.tensor([1.0, 0.0])
    easy = focal_loss(logits, labels, gamma=2.0, alpha=None)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    assert (easy < bce).all(), "focal loss down-weights easy examples"


def _loss_inputs(model: NardisNeuralNetwork, batch: Batch) -> tuple[object, TargetBatch]:
    out = model(batch)
    assert batch.targets is not None
    return out, batch.targets


@pytest.mark.parametrize("reg", ["gaussian", "huber", "mse"])
@pytest.mark.parametrize("cls", ["bce", "weighted_bce", "focal"])
def test_multitask_loss_variants(tiny_config: NeuralConfig, norm_batch: Batch, reg: str, cls: str) -> None:
    tiny_config.loss.regression_loss = reg  # type: ignore[assignment]
    tiny_config.loss.classification_loss = cls  # type: ignore[assignment]
    model = NardisNeuralNetwork(tiny_config)
    assert norm_batch.targets is not None
    pw = estimate_pos_weight(norm_batch.targets.labels, norm_batch.targets.mask)
    loss_fn = MultiTaskLoss(tiny_config, pw)
    total, comps = loss_fn(model(norm_batch), norm_batch.targets)
    assert torch.isfinite(total)
    for key in (
        "return",
        "max_upside",
        "max_drawdown",
        "volatility",
        "upside",
        "downside",
        "quantile",
        "supervised",
        "total",
        "gate_load_balance",
    ):
        assert key in comps, key
    total.backward()
    heads = model.heads.regression["return"]
    assert isinstance(heads, HorizonHeads)
    assert heads.w2.grad is not None and heads.w2.grad[:, :, 1].abs().sum() > 0, (
        "log-variance head must train"
    )


def test_learned_uncertainty_weighting(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    tiny_config.loss.learned_uncertainty_weighting = True
    model = NardisNeuralNetwork(tiny_config)
    loss_fn = MultiTaskLoss(tiny_config)
    assert loss_fn.log_vars is not None
    assert norm_batch.targets is not None
    total, _ = loss_fn(model(norm_batch), norm_batch.targets)
    total.backward()
    assert all(p.grad is not None for p in loss_fn.log_vars.values())


def test_target_mask_and_sample_weights(tiny_config: NeuralConfig, norm_batch: Batch) -> None:
    model = NardisNeuralNetwork(tiny_config).eval()
    out = model(norm_batch)
    t = norm_batch.targets
    assert t is not None
    loss_fn = MultiTaskLoss(tiny_config)
    _, base = loss_fn(out, t)
    corrupted = TargetBatch(
        {k: torch.where(t.mask, v, torch.full_like(v, 1e6)) for k, v in t.regression.items()},
        t.labels,
        t.mask.clone(),
    )
    corrupted.mask[:, 0] = False
    corrupted.regression["return"][:, 0] = 1e6
    _, masked = loss_fn(out, corrupted)
    assert math.isfinite(masked["return"].item()) and masked["return"].item() < 1e3
    w = torch.zeros(norm_batch.size)
    w[0] = 1.0
    _, one = loss_fn(out, t, w)
    single = TargetBatch(
        {k: v[:1] for k, v in t.regression.items()}, {k: v[:1] for k, v in t.labels.items()}, t.mask[:1]
    )
    sliced = type(out)(
        **{
            **out.__dict__,
            "means": {k: v[:1] for k, v in out.means.items()},
            "logvars": {k: v[:1] for k, v in out.logvars.items()},
            "logits": {k: v[:1] for k, v in out.logits.items()},
            "quantiles": None if out.quantiles is None else out.quantiles[:1],
        }
    )
    _, ref = loss_fn(sliced, single)
    assert one["return"].item() == pytest.approx(ref["return"].item(), rel=1e-5)
    assert base["return"].item() != pytest.approx(one["return"].item())


# ---------------------------------------------------------------- metrics
def test_classification_metrics() -> None:
    rng = np.random.default_rng(0)
    p = rng.random(20000)
    y = (rng.random(20000) < p).astype(float)
    assert expected_calibration_error(p, y) < 0.02
    assert expected_calibration_error(np.clip(p * 0.3, 0, 1), y) > 0.2
    assert roc_auc(p, y) > 0.7
    assert roc_auc(np.array([0.1, 0.9]), np.array([0.0, 1.0])) == 1.0
    assert math.isnan(roc_auc(p[:5], np.ones(5)))
    assert brier_score(y, y) == 0 and log_loss(np.full(4, 0.5), np.array([0, 1, 0, 1.0])) == pytest.approx(
        math.log(2)
    )


def test_evaluate_predictions_keys(tiny_config: NeuralConfig) -> None:
    n, h = 200, 3
    rng = np.random.default_rng(1)
    targets = {
        t: np.abs(rng.normal(size=(n, h))) * 0.05
        for t in ("return", "max_upside", "max_drawdown", "volatility")
    }
    preds = {f"{t}.mean": v + rng.normal(size=v.shape) * 0.01 for t, v in targets.items()}
    preds |= {f"{t}.std": np.full((n, h), 0.01) for t in targets}
    preds |= {"upside.prob": rng.random((n, h)), "downside.prob": rng.random((n, h))}
    m = evaluate_predictions(preds, targets, tiny_config)
    for key in (
        "return.rmse",
        "return.rmse.30s",
        "return.rank_corr",
        "return.nll",
        "return.coverage90",
        "upside.log_loss",
        "upside.brier",
        "upside.ece",
        "upside.auc",
        "tail.return_mae",
        "tail.drawdown_mae",
    ):
        assert key in m and math.isfinite(m[key]), key
    assert m["return.rank_corr"] > 0.9


# ---------------------------------------------------------------- trainer
def _datasets(store: ArrayStore, cfg: NeuralConfig) -> tuple[MarketDataset, MarketDataset, FeatureNormalizer]:
    norm = FeatureNormalizer.fit(store, np.arange(700), cfg)
    return MarketDataset(store, cfg, np.arange(700)), MarketDataset(store, cfg, np.arange(800, 1000)), norm


def test_trainer_reduces_loss_and_early_stops(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    tiny_config.training.epochs = 6
    tiny_config.training.early_stopping_patience = 1
    tiny_config.training.early_stopping_min_delta = 10.0  # impossible improvement → stop early
    tr, va, norm = _datasets(base_store, tiny_config)
    model = NardisNeuralNetwork(tiny_config)
    res = Trainer(tiny_config, torch.device("cpu")).fit(model, norm, tr, va, build_objective(tiny_config, tr))
    assert res.stopped_early and res.epochs_run == 2
    tiny_config.training.early_stopping_min_delta = 0.0
    tiny_config.training.early_stopping_patience = 5
    tiny_config.training.epochs = 4
    res = Trainer(tiny_config, torch.device("cpu")).fit(
        NardisNeuralNetwork(tiny_config), norm, tr, va, build_objective(tiny_config, tr)
    )
    losses = [h["train_loss"] for h in res.history]
    assert losses[-1] < losses[0]
    assert res.best_val_loss == min(h["val_loss"] for h in res.history)
    assert {"lr", "val.return", "train.upside", "seconds"} <= set(res.history[0])


def test_trainer_grad_accumulation_and_schedulers(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    tr, va, norm = _datasets(base_store, tiny_config)
    tiny_config.training.epochs = 1
    tiny_config.training.grad_accumulation_steps = 3
    tiny_config.training.max_train_batches_per_epoch = 4
    for sched in ("cosine", "onecycle", "plateau", "constant"):
        tiny_config.training.scheduler = sched
        res = Trainer(tiny_config, torch.device("cpu")).fit(
            NardisNeuralNetwork(tiny_config), norm, tr, va, build_objective(tiny_config, tr)
        )
        assert res.history[0]["train_batches"] == 4
        assert math.isfinite(res.best_val_loss)


def test_optimizers_and_warmup(tiny_config: NeuralConfig) -> None:
    model = NardisNeuralNetwork(tiny_config)
    for opt in ("adamw", "adam", "sgd"):
        tiny_config.training.optimizer = opt
        o = build_optimizer(model.parameters(), tiny_config.training)
        sched, mode = build_scheduler(o, tiny_config.training, 100)
        assert sched is not None and mode == "batch"
    fn = warmup_cosine(100, 0.1)
    assert fn(0) < fn(9) <= 1.0 and fn(99) < 0.2


def test_nonfinite_detection(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    tr, va, norm = _datasets(base_store, tiny_config)
    tiny_config.training.max_nonfinite_steps = 2
    model = NardisNeuralNetwork(tiny_config)
    with torch.no_grad():
        head = model.heads.regression["return"]
        assert isinstance(head, HorizonHeads)
        head.b2.fill_(float("nan"))
    with pytest.raises(NonFiniteLossError):
        Trainer(tiny_config, torch.device("cpu")).fit(model, norm, tr, va, build_objective(tiny_config, tr))


def test_checkpoint_resume(tmp_path: Path, tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    tr, va, norm = _datasets(base_store, tiny_config)
    tiny_config.training.max_train_batches_per_epoch = 3
    trainer = Trainer(tiny_config, torch.device("cpu"))
    model = NardisNeuralNetwork(tiny_config)
    first = trainer.fit(model, norm, tr, va, build_objective(tiny_config, tr), epochs=1, run_dir=tmp_path)
    assert (tmp_path / "trainer_state.pt").exists() and (tmp_path / "metrics.jsonl").exists()
    model2 = NardisNeuralNetwork(tiny_config)
    resumed = trainer.fit(
        model2, norm, tr, va, build_objective(tiny_config, tr), epochs=3, run_dir=tmp_path, resume=True
    )
    assert [h["epoch"] for h in resumed.history] == [0.0, 1.0, 2.0]
    assert resumed.history[0] == first.history[0]
    assert len((tmp_path / "metrics.jsonl").read_text().strip().splitlines()) == 3


def test_training_is_deterministic(tiny_config: NeuralConfig, base_store: ArrayStore) -> None:
    tr, va, norm = _datasets(base_store, tiny_config)
    tiny_config.training.max_train_batches_per_epoch = 3
    runs = []
    for _ in range(2):
        torch.manual_seed(123)
        model = NardisNeuralNetwork(tiny_config)
        res = Trainer(tiny_config, torch.device("cpu")).fit(
            model, norm, tr, va, build_objective(tiny_config, tr), epochs=1, seed=5
        )
        runs.append((res.history[0]["train_loss"], next(iter(model.parameters())).detach().clone()))
    assert runs[0][0] == runs[1][0]
    assert torch.equal(runs[0][1], runs[1][1])


def test_device_and_amp_settings() -> None:
    assert resolve_device("cpu").type == "cpu"
    assert resolve_device("auto").type in {"cpu", "cuda", "mps"}
    assert amp_settings(torch.device("cpu"), "auto") == (False, torch.float32, False)
    assert amp_settings(torch.device("mps"), "auto")[0] is False
    assert amp_settings(torch.device("mps"), "fp16") == (True, torch.float16, False)
    enabled, dtype, scaler = amp_settings(torch.device("cuda"), "fp16")
    assert enabled and dtype == torch.float16 and scaler
    assert amp_settings(torch.device("cuda"), "bf16") == (True, torch.bfloat16, False)
