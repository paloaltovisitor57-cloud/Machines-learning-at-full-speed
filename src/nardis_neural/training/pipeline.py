"""High-level training pipeline producing a complete, calibrated :class:`NeuralEngine`.

    split (chronological + embargo) → fit normaliser on train only → train each ensemble
    member independently → calibrate on validation → reference statistics → OOD detector
    on training embeddings → regime clusters → validation metrics → engine

The same function serves initial training, continual adaptation (warm start from a
cloned champion with distillation/EWC) and full retraining.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig
from nardis_neural.data.datasets import MarketDataset, compute_sampling_weights
from nardis_neural.data.loaders import KEY_TARGET_MASK, ArrayStore, fingerprint_arrays
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.data.splits import chronological_split
from nardis_neural.inference.calibration import CalibrationSet
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.inference.ood import OODDetector
from nardis_neural.lifecycle.checkpoints import ModelMetadata, new_version_id
from nardis_neural.models.ensemble import DeepEnsemble
from nardis_neural.models.main import ModelOutput, NardisNeuralNetwork
from nardis_neural.regimes.clustering import RegimeClusterer
from nardis_neural.training.distillation import DistillationLoss
from nardis_neural.training.ewc import EWCPenalty
from nardis_neural.training.losses import MultiTaskLoss, estimate_pos_weight
from nardis_neural.training.metrics import evaluate_predictions, labels_from_targets
from nardis_neural.training.trainer import Trainer, TrainingObjective, TrainResult

Array = npt.NDArray[Any]
Logger = Callable[[str], None]


def _silent(_: str) -> None:
    return None


@dataclass
class TrainingReport:
    """Summary of one train_engine() run: member results, validation metrics, calibration."""

    version: str
    member_results: list[TrainResult] = field(default_factory=list)
    validation_metrics: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    n_train: int = 0
    n_validation: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly summary without the calibration report (stored as training stats)."""
        return {
            "version": self.version,
            "n_train": self.n_train,
            "n_validation": self.n_validation,
            "seconds": self.seconds,
            "members": [r.summary() for r in self.member_results],
            "validation_metrics": self.validation_metrics,
        }


def dataset_targets(dataset: MarketDataset) -> tuple[dict[str, Array], Array]:
    """Real-unit regression targets (N, H) and validity mask; non-finite targets are masked
    and zeroed.
    """
    targets = {t: dataset.target_array(t) for t in REGRESSION_TASKS}
    store = dataset.store
    if KEY_TARGET_MASK in store:
        mask = np.asarray(store[KEY_TARGET_MASK][dataset.indices], dtype=bool)
    else:
        mask = np.ones_like(targets["return"], dtype=bool)
    for t in REGRESSION_TASKS:
        mask &= np.isfinite(targets[t])
        targets[t] = np.nan_to_num(targets[t])
    return targets, mask


def evaluate_engine(
    engine: NeuralEngine, dataset: MarketDataset, mc_samples: int | None = None
) -> tuple[dict[str, float], dict[str, Array]]:
    """Predict ``dataset`` with ``engine`` and score it; returns (metrics, predictions)."""
    preds = engine.predict_dataset(dataset, mc_samples=mc_samples)
    targets, mask = dataset_targets(dataset)
    return evaluate_predictions(preds, targets, engine.config, mask), preds


def load_pretrained_encoders(member: NardisNeuralNetwork, state: dict[str, torch.Tensor]) -> int:
    """Copy self-supervised encoder weights (``experts.*``) into a member; returns #tensors loaded."""
    own = member.state_dict()
    loadable = {
        k: v for k, v in state.items() if k.startswith("experts.") and k in own and own[k].shape == v.shape
    }
    own.update(loadable)
    member.load_state_dict(own)
    return len(loadable)


def build_objective(
    config: NeuralConfig,
    train_ds: MarketDataset | None = None,
    teacher: NardisNeuralNetwork | None = None,
    distillation_weight: float = 0.0,
    ewc: EWCPenalty | None = None,
) -> TrainingObjective:
    """Training objective: multi-task loss, optional distillation from a copy of ``teacher``
    and optional EWC.

    For weighted BCE, ``pos_weight`` is estimated from ``train_ds`` (``auto``) or taken from
    the config.
    """
    pos_weight = None
    if config.loss.classification_loss == "weighted_bce":
        h = len(config.targets.horizons)
        if config.loss.pos_weight == "auto" and train_ds is not None and len(train_ds):
            targets, mask = dataset_targets(train_ds)
            labels = labels_from_targets(targets, config)
            pos_weight = estimate_pos_weight(
                {k: torch.as_tensor(v, dtype=torch.float32) for k, v in labels.items()},
                torch.as_tensor(mask),
            )
        elif isinstance(config.loss.pos_weight, float):
            pos_weight = {t: torch.full((h,), config.loss.pos_weight) for t in ("upside", "downside")}
    loss = MultiTaskLoss(config, pos_weight)
    distill = None
    if teacher is not None and distillation_weight > 0:
        distill = DistillationLoss(
            copy.deepcopy(teacher), config.continual.distillation_temperature, distillation_weight
        )
    return TrainingObjective(loss, distill, ewc)


def estimate_ewc(
    model: NardisNeuralNetwork,
    config: NeuralConfig,
    normalizer: FeatureNormalizer,
    dataset: MarketDataset,
    trainer: Trainer,
    weight: float,
) -> EWCPenalty:
    """EWC penalty anchored at ``model``'s current parameters, Fisher estimated on ``dataset``."""
    loss = MultiTaskLoss(config).to(trainer.device)
    model.to(trainer.device)

    def loss_fn(out: ModelOutput, batch: Any) -> torch.Tensor:
        total: torch.Tensor = loss(out, batch.targets, None)[0]
        return total

    batches = trainer.batches(dataset, normalizer, config.training.batch_size, shuffle=True, seed=17)
    return EWCPenalty.estimate(model, batches, loss_fn, weight, config.continual.ewc_fisher_batches)


def finalize_engine(
    engine: NeuralEngine,
    train_ds: MarketDataset,
    val_ds: MarketDataset,
    fit_regimes: bool = True,
    max_reference_rows: int = 20_000,
) -> dict[str, float]:
    """Calibrate on validation, fit reference stats, OOD and regimes; return val metrics."""
    config = engine.config
    val_preds = engine.predict_dataset(val_ds)
    targets, mask = dataset_targets(val_ds)
    labels = labels_from_targets(targets, config)
    raw = {t: val_preds[f"{t}.prob_raw"] for t in ("upside", "downside")}
    engine.calibration = CalibrationSet(config.horizon_names).fit(raw, labels, mask, config.calibration)
    for t in ("upside", "downside"):
        val_preds[f"{t}.prob"] = engine.calibration.apply(t, raw[t])
    engine.metadata.reference = {
        "epistemic_median": float(np.median(val_preds["epistemic"])),
        "epistemic_p90": float(np.quantile(val_preds["epistemic"], 0.9)),
        "aleatoric_median": float(np.median(val_preds["aleatoric"])),
    }
    rng = np.random.default_rng(0)
    pos = np.arange(len(train_ds))
    if len(pos) > max_reference_rows:
        pos = np.sort(rng.choice(pos, max_reference_rows, replace=False))
    ref_preds = engine.predict_dataset(train_ds.subset(pos))
    engine.ood = OODDetector.fit(
        ref_preds["embedding"], ref_preds["input_rms"], ref_preds["epistemic"], config.ood
    )
    if fit_regimes:
        engine.regimes, _ = RegimeClusterer.fit(ref_preds["embedding"], config.regimes)
    metrics = evaluate_predictions(val_preds, targets, config, mask)
    metrics.update(
        {f"calibration.{k}.ece_after": float(v["ece_after"]) for k, v in engine.calibration.report.items()}
    )
    return metrics


def train_engine(
    config: NeuralConfig,
    store: ArrayStore,
    train_indices: npt.NDArray[np.int64] | None = None,
    val_indices: npt.NDArray[np.int64] | None = None,
    *,
    version: str | None = None,
    parent_version: str | None = None,
    origin: str = "train",
    device: torch.device | None = None,
    init_ensemble: DeepEnsemble | None = None,
    normalizer: FeatureNormalizer | None = None,
    epochs: int | None = None,
    learning_rate: float | None = None,
    teacher: DeepEnsemble | None = None,
    distillation_weight: float = 0.0,
    ewc_dataset: MarketDataset | None = None,
    ewc_weight: float = 0.0,
    sample_weights: npt.NDArray[np.float32] | None = None,
    pretrained_state: dict[str, torch.Tensor] | None = None,
    run_dir: str | Path | None = None,
    resume: bool = False,
    fit_regimes: bool = True,
    log: Logger = _silent,
) -> tuple[NeuralEngine, TrainingReport]:
    """Train, calibrate and assemble a complete NeuralEngine; returns it with its report.

    Without explicit indices the store is split chronologically with an embargo.  Each
    ensemble member is trained independently (optionally bootstrapped, warm-started,
    distilled and EWC-regularised).  With ``run_dir``, per-member metrics and resumable
    trainer state are written there; nothing is registered or saved as a checkpoint.
    """
    t0 = time.time()
    tc = config.training
    trainer = Trainer(config, device)
    if train_indices is None or val_indices is None:
        split = chronological_split(
            store.timestamps, tc.validation_fraction, embargo_seconds=config.embargo_seconds
        )
        train_indices, val_indices = split.train, split.validation
    train_indices = np.asarray(train_indices, dtype=np.int64)
    val_indices = np.asarray(val_indices, dtype=np.int64)
    if normalizer is None:
        normalizer = FeatureNormalizer.fit(store, train_indices, config, seed=tc.seed)
    ensemble = init_ensemble if init_ensemble is not None else DeepEnsemble.create(config)
    if pretrained_state is not None:
        for i in range(ensemble.size):
            n = load_pretrained_encoders(ensemble.member(i), pretrained_state)
            log(f"member {i}: loaded {n} pretrained encoder tensors")
    version = version or new_version_id()
    report = TrainingReport(version=version, n_train=len(train_indices), n_validation=len(val_indices))
    val_ds = MarketDataset(store, config, val_indices)
    for i in range(ensemble.size):
        member = ensemble.member(i)
        member_seed = tc.seed + 1000 * (i + 1)
        idx = train_indices
        w = sample_weights
        if tc.bootstrap_members and init_ensemble is None:
            boot = np.random.default_rng(member_seed).integers(0, len(idx), len(idx))
            idx = idx[boot]
            w = None if w is None else w[boot]
        train_ds = MarketDataset(store, config, idx, w)
        ewc = None
        if ewc_dataset is not None and ewc_weight > 0:
            ewc = estimate_ewc(member, config, normalizer, ewc_dataset, trainer, ewc_weight)
        teacher_member = None if teacher is None else teacher.member(i % teacher.size)
        objective = build_objective(config, train_ds, teacher_member, distillation_weight, ewc)
        member_run = None if run_dir is None else Path(run_dir) / f"member_{i}"
        result = trainer.fit(
            member,
            normalizer,
            train_ds,
            val_ds,
            objective,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=member_seed,
            run_dir=member_run,
            resume=resume,
            sampling_weights=compute_sampling_weights(train_ds, config),
        )
        log(
            f"member {i}: epochs={result.epochs_run} best_epoch={result.best_epoch} "
            f"best_val={result.best_val_loss:.4f} ({result.seconds:.1f}s)"
        )
        report.member_results.append(result)
    metadata = ModelMetadata(
        version=version,
        parent_version=parent_version,
        data_fingerprint=fingerprint_arrays(store.select(np.sort(train_indices))),
        origin=origin,
        ensemble={
            "size": ensemble.size,
            "member_seeds": [tc.seed + 1000 * (i + 1) for i in range(ensemble.size)],
            "embedding_member": config.ensemble.embedding_member,
            "mc_dropout_samples": config.ensemble.mc_dropout_samples,
        },
    )
    engine = NeuralEngine(
        config,
        ensemble,
        normalizer,
        CalibrationSet.identity(config.horizon_names),
        metadata,
        device=trainer.device,
    )
    train_ds_full = MarketDataset(store, config, train_indices)
    report.validation_metrics = finalize_engine(engine, train_ds_full, val_ds, fit_regimes=fit_regimes)
    report.calibration = engine.calibration.report
    report.seconds = time.time() - t0
    engine.metadata.training_stats = report.to_dict()
    return engine, report
