"""Configurable multi-task objective.

Regression (per task, per horizon, in normalised target space):

* ``gaussian`` — heteroscedastic Gaussian NLL ``½(log σ² + (y−μ)²/σ²)``
* ``huber`` / ``mse`` — point losses; the log-variance heads are then trained with a
  Gaussian NLL on a *detached* mean (``variance_loss_weight``) so aleatoric uncertainty
  is always available without distorting the point estimate.
* pinball (quantile) loss on the return quantile heads (``quantile_loss_weight``).

Classification: ``bce``, ``weighted_bce`` (pos_weight) or ``focal``.

Task weighting: static ``task_weights`` or learned homoscedastic uncertainty weighting
(Kendall et al. 2018): ``L = Σ_t exp(−s_t)·L_t + s_t``.

Every component is returned separately for logging.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.config import CLASSIFICATION_TASKS, REGRESSION_TASKS, LossConfig, NeuralConfig
from nardis_neural.data.datasets import TargetBatch
from nardis_neural.models.main import ModelOutput

Tensor = torch.Tensor


def gaussian_nll(mean: Tensor, logvar: Tensor, target: Tensor) -> Tensor:
    return 0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))


def huber(mean: Tensor, target: Tensor, delta: float) -> Tensor:
    return F.huber_loss(mean, target, reduction="none", delta=delta)


def pinball(quantile_preds: Tensor, target: Tensor, quantiles: Tensor) -> Tensor:
    """quantile_preds (B, H, Q), target (B, H) → (B, H) mean pinball loss over Q."""
    diff = target.unsqueeze(-1) - quantile_preds
    loss = torch.maximum(quantiles * diff, (quantiles - 1.0) * diff)
    return loss.mean(dim=-1)


def focal_loss(logits: Tensor, labels: Tensor, gamma: float, alpha: float | None) -> Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * labels + (1 - p) * (1 - labels)
    loss = (1 - p_t) ** gamma * bce
    if alpha is not None:
        loss = (alpha * labels + (1 - alpha) * (1 - labels)) * loss
    return loss


def masked_weighted_mean(values: Tensor, mask: Tensor, sample_weights: Tensor | None) -> Tensor:
    """Mean of (B, H) ``values`` over valid entries with optional (B,) sample weights."""
    w = mask.to(values.dtype)
    if sample_weights is not None:
        w = w * sample_weights.unsqueeze(-1).to(values.dtype)
    return (values * w).sum() / w.sum().clamp_min(1e-8)


class MultiTaskLoss(nn.Module):
    def __init__(self, config: NeuralConfig, pos_weight: dict[str, Tensor] | None = None) -> None:
        super().__init__()
        self.cfg: LossConfig = config.loss
        self.quantile_levels: Tensor
        self.register_buffer("quantile_levels", torch.tensor(config.targets.quantiles, dtype=torch.float32))
        self.tasks = [*REGRESSION_TASKS, *CLASSIFICATION_TASKS]
        self.static_weights = {t: float(self.cfg.task_weights.get(t, 1.0)) for t in self.tasks}
        self.log_vars = (
            nn.ParameterDict({t: nn.Parameter(torch.zeros(())) for t in self.tasks})
            if self.cfg.learned_uncertainty_weighting
            else None
        )
        self.pos_weight: dict[str, Tensor] = pos_weight or {}

    def regression_term(self, out: ModelOutput, targets: TargetBatch, task: str) -> Tensor:
        mean, logvar, y = out.means[task], out.logvars[task], targets.regression[task]
        if self.cfg.regression_loss == "gaussian":
            return gaussian_nll(mean, logvar, y)
        point = (
            huber(mean, y, self.cfg.huber_delta) if self.cfg.regression_loss == "huber" else (mean - y) ** 2
        )
        return point + self.cfg.variance_loss_weight * gaussian_nll(mean.detach(), logvar, y)

    def classification_term(self, logits: Tensor, labels: Tensor, task: str) -> Tensor:
        kind = self.cfg.classification_loss
        if kind == "focal":
            return focal_loss(logits, labels, self.cfg.focal_gamma, self.cfg.focal_alpha)
        pw = self.pos_weight.get(task) if kind == "weighted_bce" else None
        return F.binary_cross_entropy_with_logits(
            logits, labels, reduction="none", pos_weight=None if pw is None else pw.to(logits.device)
        )

    def forward(
        self, out: ModelOutput, targets: TargetBatch, sample_weights: Tensor | None = None
    ) -> tuple[Tensor, dict[str, Tensor]]:
        mask = targets.mask
        components: dict[str, Tensor] = {}
        for task in REGRESSION_TASKS:
            components[task] = masked_weighted_mean(
                self.regression_term(out, targets, task), mask, sample_weights
            )
        for task in CLASSIFICATION_TASKS:
            per = self.classification_term(out.logits[task], targets.labels[task], task)
            components[task] = masked_weighted_mean(per, mask, sample_weights)
        total = torch.zeros((), device=mask.device)
        for task in self.tasks:
            if self.log_vars is not None:
                s = self.log_vars[task]
                total = total + torch.exp(-s) * components[task] + s
            else:
                total = total + self.static_weights[task] * components[task]
        if out.quantiles is not None and self.cfg.quantile_loss_weight > 0:
            q = pinball(
                out.quantiles, targets.regression["return"], self.quantile_levels.to(out.quantiles.dtype)
            )
            components["quantile"] = masked_weighted_mean(q, mask, sample_weights)
            total = total + self.cfg.quantile_loss_weight * components["quantile"]
        components["supervised"] = total
        for name, aux in out.aux_losses.items():
            components[name] = aux
            total = total + aux
        components["total"] = total
        return total, components


def estimate_pos_weight(
    labels: dict[str, Tensor], mask: Tensor, max_weight: float = 20.0
) -> dict[str, Tensor]:
    """neg/pos ratio per (task, horizon) for weighted BCE."""
    out: dict[str, Tensor] = {}
    m = mask.float()
    for task, y in labels.items():
        pos = (y * m).sum(dim=0)
        neg = ((1 - y) * m).sum(dim=0)
        out[task] = (neg / pos.clamp_min(1.0)).clamp(1.0 / max_weight, max_weight)
    return out
