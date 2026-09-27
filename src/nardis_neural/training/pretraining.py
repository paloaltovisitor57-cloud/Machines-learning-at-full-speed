"""Self-supervised pretraining of the temporal encoders on unlabelled sequences.

Tasks (any subset, configured in ``PretrainConfig.tasks``):

* ``masked_timestep`` — whole observed steps are blanked (values → 0 = the normalised
  mean, the step stays "present"); the encoder must reconstruct them.  Because the
  encoders are causal this is a forecasting-style objective from the past.
* ``masked_feature`` — random individual (step, feature) entries are blanked and
  reconstructed, forcing cross-feature reasoning.
* ``contrastive`` — two stochastic augmentations (jitter, scaling, step dropout) of each
  sample are pulled together and pushed away from other samples (NT-Xent / InfoNCE) in a
  projection of the expert's fused multi-timescale representation.

Only the sequence experts (Transformer, recurrent, TCN) are trained; their weights are
later copied into supervised members with ``load_pretrained_encoders``.  Arbitrarily
large unlabelled datasets can be streamed through :class:`MarketDataset`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import NeuralConfig, PretrainConfig
from nardis_neural.data.datasets import Batch, MarketDataset, SequenceBatch, make_loader
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.training.trainer import resolve_device, set_seed

Tensor = torch.Tensor


class PretrainHeads(nn.Module):
    """Reconstruction heads per (sequence expert, timescale) and projection heads per expert."""

    def __init__(self, model: NardisNeuralNetwork, config: NeuralConfig) -> None:
        super().__init__()
        d = config.model.d_model
        self.reconstruct = nn.ModuleDict(
            {
                f"{e}__{ts.name}": nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, ts.feature_dim))
                for e in model.sequence_experts()
                for ts in config.features.timescales
            }
        )
        self.project = nn.ModuleDict(
            {
                e: nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, max(8, d // 2)))
                for e in model.sequence_experts()
            }
        )


def nt_xent(z1: Tensor, z2: Tensor, temperature: float) -> Tensor:
    """NT-Xent (InfoNCE) loss with ``z1[i]`` and ``z2[i]`` as the positive pair."""
    z = F.normalize(torch.cat([z1, z2]), dim=-1)
    sim = z @ z.T / temperature
    n = z1.shape[0]
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), -1e9)
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def augment(batch: Batch, cfg: PretrainConfig, generator: torch.Generator | None = None) -> Batch:
    """Copy of the batch with sequences jittered, rescaled and ~10% of steps dropped."""
    seqs = {}
    for name, s in batch.sequences.items():
        noise = (
            torch.randn(s.values.shape, device=s.values.device, generator=generator) * cfg.augmentation_noise
        )
        scale = 1.0 + 0.1 * torch.randn(s.values.shape[0], 1, 1, device=s.values.device, generator=generator)
        drop = torch.rand(s.mask.shape, device=s.mask.device, generator=generator) < 0.1
        mask = s.mask & ~drop
        seqs[name] = SequenceBatch((s.values * scale + noise) * mask.unsqueeze(-1), mask, s.time_deltas)
    return replace(batch, sequences=seqs)


@dataclass
class PretrainResult:
    """Per-epoch pretraining loss history and wall-clock seconds."""

    history: list[dict[str, float]] = field(default_factory=list)
    seconds: float = 0.0


def pretrain_step(
    model: NardisNeuralNetwork, heads: PretrainHeads, batch: Batch, cfg: PretrainConfig
) -> tuple[Tensor, dict[str, float]]:
    """Self-supervised loss of one normalised batch over all experts, timescales and tasks.

    Returns the total loss tensor and each component as a float.
    """
    comps: dict[str, Tensor] = {}
    total = torch.zeros((), device=batch.device)
    experts = model.sequence_experts()
    for ename, expert in experts.items():
        for ts_name, seq in batch.sequences.items():
            if not bool(seq.mask.any()):
                continue
            for task in ("masked_timestep", "masked_feature"):
                if task not in cfg.tasks:
                    continue
                if task == "masked_timestep":
                    step_sel = (torch.rand(seq.mask.shape, device=batch.device) < cfg.mask_ratio) & seq.mask
                    entry_sel = step_sel.unsqueeze(-1).expand_as(seq.values)
                else:
                    entry_sel = (
                        torch.rand(seq.values.shape, device=batch.device) < cfg.mask_ratio
                    ) & seq.mask.unsqueeze(-1)
                if not bool(entry_sel.any()):
                    continue
                corrupted = SequenceBatch(seq.values.masked_fill(entry_sel, 0.0), seq.mask, seq.time_deltas)
                h = expert.encode_timesteps(ts_name, corrupted)
                recon = heads.reconstruct[f"{ename}__{ts_name}"](h)
                loss = F.mse_loss(recon[entry_sel], seq.values[entry_sel])
                key = f"{ename}.{task}"
                comps[key] = comps.get(key, torch.zeros((), device=batch.device)) + loss
                total = total + loss
        if "contrastive" in cfg.tasks and batch.size > 1:
            v1, v2 = augment(batch, cfg), augment(batch, cfg)
            z1 = heads.project[ename](expert(v1).latent)
            z2 = heads.project[ename](expert(v2).latent)
            loss = cfg.contrastive_weight * nt_xent(z1, z2, cfg.contrastive_temperature)
            comps[f"{ename}.contrastive"] = loss
            total = total + loss
    return total, {k: float(v.detach()) for k, v in comps.items()}


def pretrain_encoders(
    config: NeuralConfig,
    dataset: MarketDataset,
    normalizer: FeatureNormalizer,
    model: NardisNeuralNetwork | None = None,
    device: torch.device | None = None,
    epochs: int | None = None,
    max_batches_per_epoch: int | None = None,
    log: Callable[[str], None] | None = None,
) -> tuple[NardisNeuralNetwork, PretrainResult]:
    """Pretrain the sequence experts of ``model`` (a new network by default) in place.

    Returns the model in eval mode and the loss history.
    """
    pc = config.pretrain
    dev = device or resolve_device(config.training.device)
    set_seed(config.training.seed)
    model = model or NardisNeuralNetwork(config)
    if not model.sequence_experts():
        raise ValueError("pretraining needs at least one sequence expert (transformer/recurrent/tcn)")
    model.to(dev).train()
    heads = PretrainHeads(model, config).to(dev)
    params = [p for e in model.sequence_experts().values() for p in e.parameters()] + list(heads.parameters())
    opt = torch.optim.AdamW(params, lr=pc.learning_rate, weight_decay=config.training.weight_decay)
    result = PretrainResult()
    t0 = time.time()
    for epoch in range(epochs or pc.epochs):
        acc: dict[str, list[float]] = {}
        for batch in make_loader(dataset, pc.batch_size, True, seed=epoch, max_batches=max_batches_per_epoch):
            b = normalizer.transform_batch(batch.to(dev))
            loss, comps = pretrain_step(model, heads, b, pc)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(params, config.training.grad_clip_norm or 1.0)
            opt.step()
            acc.setdefault("loss", []).append(float(loss.detach()))
            for k, v in comps.items():
                acc.setdefault(k, []).append(v)
        record = {"epoch": float(epoch)} | {k: float(np.mean(v)) for k, v in acc.items()}
        result.history.append(record)
        if log is not None:
            log(f"pretrain epoch {epoch}: loss={record.get('loss', float('nan')):.4f}")
    result.seconds = time.time() - t0
    model.eval()
    return model, result


def save_pretrained(
    path: str | Path, model: NardisNeuralNetwork, result: PretrainResult, config: NeuralConfig
) -> None:
    """Save the ``experts.*`` weights, loss history and config to ``path`` with torch.save."""
    payload: dict[str, Any] = {
        "state_dict": {
            k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("experts.")
        },
        "history": result.history,
        "config": config.to_dict(),
    }
    torch.save(payload, path)


def load_pretrained(path: str | Path) -> dict[str, Tensor]:
    """Load the expert state dict written by save_pretrained()."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["state_dict"]
    assert isinstance(state, dict)
    return state
