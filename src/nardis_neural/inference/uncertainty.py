"""Uncertainty decomposition over ensemble members × MC-dropout samples.

Regression (law of total variance over the S stochastic predictions):

* aleatoric  = E_s[σ²_s]        — noise the model believes is irreducible
* epistemic  = Var_s[μ_s]       — disagreement between members / dropout masks
* total      = aleatoric + epistemic

Classification (information-theoretic decomposition):

* total      = H[E_s p_s]                         (entropy of the mean probability)
* aleatoric  = E_s H[p_s]                         (expected entropy)
* epistemic  = total − aleatoric                  (mutual information, BALD)

``member_disagreement`` uses only the deterministic pass of each ensemble member.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass
class RegressionUncertainty:
    """Mean with aleatoric and epistemic variance, each (B, H)."""

    mean: Tensor  # (B, H)
    aleatoric_var: Tensor
    epistemic_var: Tensor

    @property
    def total_var(self) -> Tensor:
        """Aleatoric + epistemic variance."""
        return self.aleatoric_var + self.epistemic_var


@dataclass
class ClassificationUncertainty:
    """Mean event probability and its entropy decomposition, each (B, H)."""

    prob: Tensor  # (B, H) mean probability (uncalibrated)
    total_entropy: Tensor
    aleatoric_entropy: Tensor
    mutual_information: Tensor


def aggregate_regression(means: Tensor, logvars: Tensor) -> RegressionUncertainty:
    """means/logvars: (S, B, H)."""
    mean = means.mean(dim=0)
    aleatoric = torch.exp(logvars).mean(dim=0)
    epistemic = means.var(dim=0, unbiased=False) if means.shape[0] > 1 else torch.zeros_like(mean)
    return RegressionUncertainty(mean=mean, aleatoric_var=aleatoric, epistemic_var=epistemic)


def binary_entropy(p: Tensor) -> Tensor:
    """Elementwise Bernoulli entropy in nats (``p`` clamped away from 0 and 1)."""
    p = p.clamp(1e-7, 1 - 1e-7)
    return -(p * torch.log(p) + (1 - p) * torch.log(1 - p))


def aggregate_classification(logits: Tensor) -> ClassificationUncertainty:
    """logits: (S, B, H)."""
    probs = torch.sigmoid(logits)
    mean_p = probs.mean(dim=0)
    total = binary_entropy(mean_p)
    aleatoric = binary_entropy(probs).mean(dim=0)
    mi = (total - aleatoric).clamp_min(0.0)
    return ClassificationUncertainty(mean_p, total, aleatoric, mi)


def member_disagreement(means: Tensor, member_index: Tensor) -> Tensor:
    """Std across members' deterministic predictions, averaged over horizons → (B,)."""
    first = torch.zeros_like(member_index, dtype=torch.bool)
    seen: set[int] = set()
    for s, m in enumerate(member_index.tolist()):
        if m not in seen:
            seen.add(m)
            first[s] = True
    det = means[first]
    if det.shape[0] < 2:
        return torch.zeros(means.shape[1], device=means.device)
    return det.std(dim=0, unbiased=False).mean(dim=-1)


def confidence_score(
    epistemic: Tensor, epistemic_reference: float, ood_score: Tensor, ood_penalty: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Confidence in [0, 1] = c_epistemic · c_ood.

    * ``c_epistemic = 1 / (1 + epistemic / reference)`` — 0.5 at the median validation
      epistemic uncertainty, → 1 when members agree perfectly.
    * ``c_ood = exp(−penalty · max(0, ood − 1))`` — untouched inside the reference
      quantile of the training distribution, decays outside it.  OOD reduces confidence.
    """
    ref = max(epistemic_reference, 1e-8)
    c_epi = 1.0 / (1.0 + epistemic / ref)
    c_ood = torch.exp(-ood_penalty * (ood_score - 1.0).clamp_min(0.0))
    return c_epi * c_ood, c_epi, c_ood
