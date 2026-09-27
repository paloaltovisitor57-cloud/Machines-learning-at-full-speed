"""Transparent PyTorch training engine.

One :class:`Trainer.fit` call trains one network (one ensemble member).  Features: device
selection (CPU / CUDA / MPS), safe mixed precision, gradient accumulation and clipping,
LR scheduling, early stopping on validation loss, per-epoch checkpoints with exact
resume, deterministic seeding, NaN/Inf detection, structured JSONL metrics.
"""

from __future__ import annotations

import json
import math
import random
import time
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import Batch, BatchIndexSampler, MarketDataset, make_loader
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.models.main import ModelOutput, NardisNeuralNetwork
from nardis_neural.training.distillation import DistillationLoss
from nardis_neural.training.ewc import EWCPenalty
from nardis_neural.training.losses import MultiTaskLoss
from nardis_neural.training.scheduling import build_optimizer, build_scheduler

Tensor = torch.Tensor


class NonFiniteLossError(RuntimeError):
    """Raised when too many consecutive steps produce NaN/Inf losses or gradients."""


# ---------------------------------------------------------------------------- devices
def resolve_device(spec: str = "auto") -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def amp_settings(device: torch.device, mode: str) -> tuple[bool, torch.dtype, bool]:
    """(enabled, dtype, use_grad_scaler).  ``auto`` enables AMP only where it is safe."""
    if mode == "off" or device.type == "cpu":
        return False, torch.float32, False
    if device.type == "cuda":
        bf16_ok = torch.cuda.is_bf16_supported()
        if mode == "bf16" or (mode == "auto" and bf16_ok):
            return True, torch.bfloat16, False
        return True, torch.float16, True
    if device.type == "mps":
        if mode == "fp16":
            return True, torch.float16, False
        return False, torch.float32, False  # auto: MPS autocast is still conservative
    return False, torch.float32, False


def autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype) -> AbstractContextManager[Any]:
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seed legacy global RNG used by third-party code
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------- objective
class TrainingObjective(nn.Module):
    """supervised multi-task loss (+ distillation) (+ EWC)."""

    def __init__(
        self,
        loss: MultiTaskLoss,
        distillation: DistillationLoss | None = None,
        ewc: EWCPenalty | None = None,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.distillation = distillation
        self.ewc = ewc

    def to_device(self, device: torch.device) -> TrainingObjective:
        self.to(device)
        if self.distillation is not None:
            self.distillation.to(device)
        if self.ewc is not None:
            self.ewc.to(device)
        return self

    def compute(self, model: nn.Module, batch: Batch, out: ModelOutput) -> tuple[Tensor, dict[str, Tensor]]:
        if batch.targets is None:
            raise ValueError("training batches must contain targets")
        total, comps = self.loss(out, batch.targets, batch.weights)
        if self.distillation is not None:
            for k, v in self.distillation(out, batch).items():
                comps[k] = v
                total = total + v
        if self.ewc is not None:
            pen = self.ewc(model)
            comps["ewc"] = pen
            total = total + pen
        comps["total"] = total
        return total, comps


# ---------------------------------------------------------------------------- results
@dataclass
class TrainResult:
    history: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = -1
    best_val_loss: float = math.inf
    epochs_run: int = 0
    stopped_early: bool = False
    nonfinite_steps: int = 0
    seconds: float = 0.0

    def summary(self) -> dict[str, float]:
        return {
            "best_epoch": float(self.best_epoch),
            "best_val_loss": self.best_val_loss,
            "epochs_run": float(self.epochs_run),
            "stopped_early": float(self.stopped_early),
            "nonfinite_steps": float(self.nonfinite_steps),
            "seconds": self.seconds,
        }


def _mean_components(acc: dict[str, list[float]]) -> dict[str, float]:
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


# ---------------------------------------------------------------------------- trainer
class Trainer:
    def __init__(self, config: NeuralConfig, device: torch.device | None = None) -> None:
        self.config = config
        self.tc = config.training
        self.device = device or resolve_device(self.tc.device)
        self.amp_enabled, self.amp_dtype, self.use_scaler = amp_settings(self.device, self.tc.mixed_precision)

    def batches(
        self,
        dataset: MarketDataset,
        normalizer: FeatureNormalizer,
        batch_size: int,
        shuffle: bool,
        seed: int = 0,
        weights: npt.NDArray[np.float64] | None = None,
        max_batches: int | None = None,
    ) -> Iterator[Batch]:
        loader = make_loader(
            dataset,
            batch_size,
            shuffle,
            self.tc.num_workers,
            seed=seed,
            weights=weights,
            max_batches=max_batches,
        )
        for batch in loader:
            yield normalizer.transform_batch(batch.to(self.device))

    @torch.no_grad()
    def evaluate_loss(
        self,
        model: nn.Module,
        normalizer: FeatureNormalizer,
        dataset: MarketDataset,
        objective: TrainingObjective,
    ) -> dict[str, float]:
        model.eval()
        acc: dict[str, list[float]] = {}
        weights: list[float] = []
        for batch in self.batches(dataset, normalizer, self.tc.eval_batch_size, shuffle=False):
            with autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                out = model(batch)
            _, comps = objective.compute(model, batch, out)
            for k, v in comps.items():
                acc.setdefault(k, []).append(float(v.detach()))
            weights.append(batch.size)
        w = np.asarray(weights, dtype=np.float64)
        return {k: float(np.average(v, weights=w)) for k, v in acc.items()}

    def fit(
        self,
        model: NardisNeuralNetwork,
        normalizer: FeatureNormalizer,
        train_ds: MarketDataset,
        val_ds: MarketDataset | None,
        objective: TrainingObjective,
        epochs: int | None = None,
        learning_rate: float | None = None,
        seed: int | None = None,
        run_dir: str | Path | None = None,
        resume: bool = False,
        sampling_weights: npt.NDArray[np.float64] | None = None,
        on_epoch_end: Callable[[dict[str, float]], None] | None = None,
    ) -> TrainResult:
        tc = self.tc
        epochs = tc.epochs if epochs is None else epochs
        lr = tc.learning_rate if learning_rate is None else learning_rate
        seed = tc.seed if seed is None else seed
        set_seed(seed, tc.deterministic)
        model.to(self.device)
        objective.to_device(self.device)
        params = list(model.parameters()) + [p for p in objective.loss.parameters() if p.requires_grad]
        optimizer = build_optimizer(params, tc, lr)
        steps_per_epoch = len(
            BatchIndexSampler(len(train_ds), tc.batch_size, max_batches=tc.max_train_batches_per_epoch)
        )
        total_steps = max(1, epochs * math.ceil(steps_per_epoch / tc.grad_accumulation_steps))
        scheduler, step_mode = build_scheduler(optimizer, tc, total_steps, lr)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_scaler)

        result = TrainResult()
        best_state: dict[str, Tensor] | None = None
        patience = 0
        start_epoch = 0
        run_path = Path(run_dir) if run_dir is not None else None
        ckpt_file = run_path / "trainer_state.pt" if run_path is not None else None
        if run_path is not None:
            run_path.mkdir(parents=True, exist_ok=True)
        if resume and ckpt_file is not None and ckpt_file.exists():
            state = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            model.load_state_dict(state["model"])
            objective.loss.load_state_dict(state["loss"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None and state.get("scheduler") is not None:
                scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = int(state["epoch"]) + 1
            best_state = state["best_state"]
            patience = int(state["patience"])
            result = TrainResult(**state["result"])
            random.setstate(state["rng_python"])
            np.random.set_state(state["rng_numpy"])  # noqa: NPY002
            torch.set_rng_state(state["rng_torch"])

        t0 = time.time()
        consecutive_bad = 0
        for epoch in range(start_epoch, epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            acc: dict[str, list[float]] = {}
            n_batches = 0
            for step, batch in enumerate(
                self.batches(
                    train_ds,
                    normalizer,
                    tc.batch_size,
                    shuffle=True,
                    seed=seed + epoch * 7919,
                    weights=sampling_weights,
                    max_batches=tc.max_train_batches_per_epoch,
                )
            ):
                with autocast_context(self.device, self.amp_enabled, self.amp_dtype):
                    out = model(batch)
                loss, comps = objective.compute(model, batch, out)
                if not torch.isfinite(loss):
                    result.nonfinite_steps += 1
                    consecutive_bad += 1
                    optimizer.zero_grad(set_to_none=True)
                    if consecutive_bad >= tc.max_nonfinite_steps:
                        raise NonFiniteLossError(f"{consecutive_bad} consecutive non-finite losses")
                    continue
                torch.autograd.backward(scaler.scale(loss / tc.grad_accumulation_steps))
                boundary = (step + 1) % tc.grad_accumulation_steps == 0
                if boundary:
                    scaler.unscale_(optimizer)
                    grads_ok = all(torch.isfinite(p.grad).all() for p in params if p.grad is not None)
                    if not grads_ok:
                        result.nonfinite_steps += 1
                        consecutive_bad += 1
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        if consecutive_bad >= tc.max_nonfinite_steps:
                            raise NonFiniteLossError(f"{consecutive_bad} consecutive non-finite gradients")
                        continue
                    if tc.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, tc.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None and step_mode == "batch":
                        scheduler.step()
                    consecutive_bad = 0
                for k, v in comps.items():
                    acc.setdefault(k, []).append(float(v.detach()))
                n_batches += 1

            train_metrics = _mean_components(acc)
            record: dict[str, float] = {
                "epoch": float(epoch),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": train_metrics.get("total", math.nan),
                "train_batches": float(n_batches),
                "nonfinite_steps": float(result.nonfinite_steps),
            }
            record.update({f"train.{k}": v for k, v in train_metrics.items()})
            if val_ds is not None and len(val_ds):
                val_metrics = self.evaluate_loss(model, normalizer, val_ds, objective)
                val_loss = val_metrics.get("supervised", val_metrics["total"])
                record["val_loss"] = val_loss
                record.update({f"val.{k}": v for k, v in val_metrics.items()})
            else:
                val_loss = train_metrics.get("supervised", math.inf)
                record["val_loss"] = val_loss
            if scheduler is not None and step_mode == "plateau":
                assert isinstance(scheduler, ReduceLROnPlateau)
                scheduler.step(val_loss)
            elif scheduler is not None and step_mode == "epoch":
                scheduler.step()

            improved = val_loss < result.best_val_loss - tc.early_stopping_min_delta
            if improved:
                result.best_val_loss = val_loss
                result.best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            result.epochs_run = epoch + 1
            record["seconds"] = time.time() - t0
            result.history.append(record)
            if run_path is not None:
                with (run_path / "metrics.jsonl").open("a") as fh:
                    fh.write(json.dumps(record) + "\n")
            if on_epoch_end is not None:
                on_epoch_end(record)
            if ckpt_file is not None:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "loss": objective.loss.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": None if scheduler is None else scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "epoch": epoch,
                        "best_state": best_state,
                        "patience": patience,
                        "result": {k: v for k, v in result.__dict__.items() if k != "history"}
                        | {"history": result.history},
                        "rng_python": random.getstate(),
                        "rng_numpy": np.random.get_state(),  # noqa: NPY002
                        "rng_torch": torch.get_rng_state(),
                    },
                    ckpt_file,
                )
            if patience >= tc.early_stopping_patience:
                result.stopped_early = True
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        result.seconds += time.time() - t0
        model.eval()
        return result
