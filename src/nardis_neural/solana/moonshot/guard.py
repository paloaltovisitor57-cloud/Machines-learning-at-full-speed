"""Manipulation guard for the moonshot view.

A fat-tail model is exactly what manipulators target: fake volume, bundled supply and
staged "smart money" all look like the early footprint of a runner.  The learned model sees
these features, but a model fitted on past data can still be walked into a trap that was
rare in its training set.  The guard therefore sits *on top* of the model and only ever
lowers its optimism:

* **trust** ∈ [0, 1] multiplies the predicted tail probabilities above 1x.  It combines the
  launch-risk model's P(rug), live authorities, unburned LP, wash trading, bundled and
  creator-cluster supply, rug-linked wallets, concentration, dev selling, how far the input
  lies outside the training data and how much the ensemble disagrees;
* **vetoes** are hard, human-readable reasons after which the token gets no chase score.

The factors are hand-set, conservative and monotone (more red flags never raise trust).
They are documented in docs/MOONSHOT.md and can be tightened in ``GuardConfig``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


class GuardConfig(BaseModel):
    """Hard-veto thresholds of the moonshot manipulation guard."""

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)

    veto_rug_probability: float = Field(default=0.6, ge=0, le=1)
    """Veto when the predicted rug probability is at least this."""
    veto_bundle_share: float = Field(default=0.2, ge=0, le=1)
    """Veto when bundled early wallets hold at least this share of supply."""
    veto_creator_cluster_share: float = Field(default=0.3, ge=0, le=1)
    """Veto when the creator's wallet cluster holds at least this share of supply."""
    veto_bot_share: float = Field(default=0.8, ge=0, le=1)
    """Veto when at least this share of the last 60 s's traders are bots (wash trading)."""
    veto_ood_score: float = Field(default=4.0, gt=0)
    """Veto when the out-of-distribution score of the market state is at least this."""
    veto_out_of_range_share: float = Field(default=0.25, gt=0, le=1)
    """Veto when this share of the inputs lies outside anything seen in training."""


@dataclass(frozen=True)
class GuardVerdict:
    """Guard result: ``trust`` in [0, 1], human-readable vetoes and every trust factor."""

    trust: float
    vetoes: list[str]
    factors: dict[str, float]


def _ramp(value: float, start: float, end: float, floor: float) -> float:
    """1 below ``start``, falling linearly to ``floor`` at ``end`` and beyond."""
    if value <= start:
        return 1.0
    if value >= end:
        return floor
    return 1.0 - (1.0 - floor) * (value - start) / (end - start)


def assess_manipulation(
    f: dict[str, float],
    rug_probability: float | None,
    ood_score: float,
    epistemic: float,
    out_of_range_share: float,
    cfg: GuardConfig | None = None,
) -> GuardVerdict:
    """Trust and vetoes for one token from its named features, P(rug), OOD score, ensemble
    disagreement and out-of-range input share.

    ``trust`` is the product of monotone factors in [0, 1]; a ``rug_probability`` of None (no risk
    model) adds neither a penalty nor a veto.
    """
    cfg = cfg or GuardConfig()
    factors = {
        "rug": 1.0 - rug_probability if rug_probability is not None else 1.0,
        "mint_authority": 1.0 if f["mint_authority_revoked"] >= 0.5 else 0.25,
        "freeze_authority": 1.0 if f["freeze_authority_revoked"] >= 0.5 else 0.25,
        "lp": 1.0 if f["migrated"] >= 0.5 else 0.5 + 0.5 * f["lp_burned_fraction"],
        "wash": _ramp(f["bot_share_60s"], 0.3, 0.8, 0.2),
        "bundle": _ramp(f["bundle_share"], 0.03, 0.2, 0.2),
        "creator_cluster": _ramp(f["creator_cluster_share"], 0.05, 0.3, 0.2),
        "rug_wallets": _ramp(f["rug_associated_share"], 0.05, 0.4, 0.2),
        "concentration": _ramp(f["top10_share"], 0.35, 0.8, 0.3),
        "dev_selling": _ramp(f["dev_sold_fraction"], 0.2, 0.9, 0.3),
        "ood": math.exp(-max(0.0, ood_score - 1.0) / 2.0),
        "out_of_range": _ramp(out_of_range_share, 0.02, cfg.veto_out_of_range_share, 0.3),
        "epistemic": _ramp(epistemic, 0.1, 0.4, 0.4),
    }
    trust = float(min(max(math.prod(factors.values()), 0.0), 1.0))
    vetoes = []
    if rug_probability is not None and rug_probability >= cfg.veto_rug_probability:
        vetoes.append(f"rug probability {rug_probability:.0%}")
    if f["mint_authority_revoked"] < 0.5:
        vetoes.append("mint authority is live")
    if f["freeze_authority_revoked"] < 0.5:
        vetoes.append("freeze authority is live")
    if f["bundle_share"] >= cfg.veto_bundle_share:
        vetoes.append(f"bundled supply {f['bundle_share']:.0%}")
    if f["creator_cluster_share"] >= cfg.veto_creator_cluster_share:
        vetoes.append(f"creator cluster holds {f['creator_cluster_share']:.0%}")
    if f["bot_share_60s"] >= cfg.veto_bot_share:
        vetoes.append(f"wash trading: bots are {f['bot_share_60s']:.0%} of volume")
    if ood_score >= cfg.veto_ood_score:
        vetoes.append(f"market state far out of distribution (ood={ood_score:.1f})")
    if out_of_range_share >= cfg.veto_out_of_range_share:
        vetoes.append(f"{out_of_range_share:.0%} of inputs outside the training range")
    return GuardVerdict(trust, vetoes, factors)
