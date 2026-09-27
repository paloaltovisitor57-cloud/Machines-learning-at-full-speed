"""Optimiser and learning-rate schedule construction."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch.optim import SGD, Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler, OneCycleLR, ReduceLROnPlateau

from nardis_neural.config import TrainingConfig

StepMode = Literal["batch", "epoch", "plateau"]


def build_optimizer(
    params: Iterable[torch.nn.Parameter], cfg: TrainingConfig, lr: float | None = None
) -> Optimizer:
    lr = cfg.learning_rate if lr is None else lr
    plist = [p for p in params if p.requires_grad]
    if cfg.optimizer == "adamw":
        return AdamW(plist, lr=lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return Adam(plist, lr=lr, weight_decay=cfg.weight_decay)
    return SGD(plist, lr=lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)


def warmup_cosine(total_steps: int, warmup_fraction: float, floor: float = 0.05) -> Callable[[int], float]:
    warmup = max(1, int(total_steps * warmup_fraction))

    def fn(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))

    return fn


def build_scheduler(
    optimizer: Optimizer, cfg: TrainingConfig, total_steps: int, lr: float | None = None
) -> tuple[LRScheduler | ReduceLROnPlateau | None, StepMode]:
    lr = cfg.learning_rate if lr is None else lr
    total_steps = max(1, total_steps)
    if cfg.scheduler == "cosine":
        return LambdaLR(optimizer, warmup_cosine(total_steps, cfg.warmup_fraction)), "batch"
    if cfg.scheduler == "onecycle":
        return (
            OneCycleLR(
                optimizer, max_lr=lr, total_steps=total_steps, pct_start=max(cfg.warmup_fraction, 0.01)
            ),
            "batch",
        )
    if cfg.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer, factor=0.5, patience=max(1, cfg.early_stopping_patience // 2)
        ), "plateau"
    return None, "epoch"
