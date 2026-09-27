"""Multi-criteria, auditable promotion decisions.

A challenger is promoted only if every *required* gate passes and at least
``min_passed_fraction`` of all gates pass.  Gates cover regression error, log loss,
Brier score, calibration, ranking quality, tail behaviour, uncertainty quality,
consistency across time windows and across regimes, and a minimum sample size.
"""

from __future__ import annotations

import math
import time
from typing import Any

from pydantic import BaseModel, Field

from nardis_neural.config import PromotionConfig
from nardis_neural.lifecycle.shadow import ShadowReport


class GateResult(BaseModel):
    name: str
    passed: bool
    required: bool
    champion: float | None = None
    challenger: float | None = None
    threshold: str = ""
    detail: str = ""


class PromotionDecision(BaseModel):
    promote: bool
    champion_version: str
    challenger_version: str
    n_observations: int
    gates: list[GateResult]
    reason: str
    created_at: float = Field(default_factory=time.time)
    shadow_report: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            f"# Promotion decision: {'PROMOTE' if self.promote else 'REJECT'}",
            "",
            f"- champion: `{self.champion_version}`",
            f"- challenger: `{self.challenger_version}`",
            f"- resolved shadow observations: {self.n_observations}",
            f"- reason: {self.reason}",
            "",
            "| gate | required | champion | challenger | threshold | passed |",
            "|---|---|---|---|---|---|",
        ]
        for g in self.gates:
            c = "" if g.champion is None else f"{g.champion:.5g}"
            h = "" if g.challenger is None else f"{g.challenger:.5g}"
            lines.append(
                f"| {g.name} | {g.required} | {c} | {h} | {g.threshold} | {'✅' if g.passed else '❌'} |"
            )
        return "\n".join(lines) + "\n"


def _mean(d: dict[str, float], keys: list[str]) -> float:
    vals = [d[k] for k in keys if k in d and math.isfinite(d[k])]
    return sum(vals) / len(vals) if vals else math.nan


def _ratio_gate(name: str, champ: float, chall: float, max_ratio: float, required: bool) -> GateResult:
    if not (math.isfinite(champ) and math.isfinite(chall)):
        return GateResult(name=name, passed=False, required=required, detail="metric unavailable")
    ok = chall <= champ * max_ratio + 1e-12
    return GateResult(
        name=name,
        passed=ok,
        required=required,
        champion=champ,
        challenger=chall,
        threshold=f"≤ {max_ratio}× champion",
    )


def evaluate_promotion(report: ShadowReport, cfg: PromotionConfig) -> PromotionDecision:
    req = set(cfg.required_gates)
    c, h = report.champion, report.challenger
    gates: list[GateResult] = [
        GateResult(
            name="min_observations",
            passed=report.n >= cfg.min_observations,
            required="min_observations" in req,
            champion=float(report.n),
            challenger=float(report.n),
            threshold=f"≥ {cfg.min_observations}",
        )
    ]
    if report.n >= 2 and c and h:
        gates.append(
            _ratio_gate(
                "return_rmse",
                c.get("return.rmse", math.nan),
                h.get("return.rmse", math.nan),
                cfg.max_rmse_ratio,
                "return_rmse" in req,
            )
        )
        ll = ["upside.log_loss", "downside.log_loss"]
        gates.append(
            _ratio_gate("log_loss", _mean(c, ll), _mean(h, ll), cfg.max_log_loss_ratio, "log_loss" in req)
        )
        br = ["upside.brier", "downside.brier"]
        gates.append(_ratio_gate("brier", _mean(c, br), _mean(h, br), cfg.max_brier_ratio, "brier" in req))
        ece = ["upside.ece", "downside.ece"]
        ce, he = _mean(c, ece), _mean(h, ece)
        gates.append(
            GateResult(
                name="calibration_ece",
                passed=math.isfinite(he) and he <= ce + cfg.max_ece_increase,
                required="calibration_ece" in req,
                champion=ce,
                challenger=he,
                threshold=f"≤ champion + {cfg.max_ece_increase}",
            )
        )
        rc, rh = c.get("return.rank_corr", math.nan), h.get("return.rank_corr", math.nan)
        gates.append(
            GateResult(
                name="rank_corr",
                passed=math.isfinite(rh) and rh >= rc + cfg.min_rank_corr_delta,
                required="rank_corr" in req,
                champion=rc,
                challenger=rh,
                threshold=f"≥ champion + {cfg.min_rank_corr_delta}",
            )
        )
        tail = ["tail.return_mae", "tail.drawdown_mae"]
        gates.append(
            _ratio_gate("tail_mae", _mean(c, tail), _mean(h, tail), cfg.max_tail_mae_ratio, "tail_mae" in req)
        )
        nc, nh = c.get("return.nll", math.nan), h.get("return.nll", math.nan)
        gates.append(
            GateResult(
                name="uncertainty_nll",
                passed=math.isfinite(nh) and nh <= nc + cfg.max_nll_increase,
                required="uncertainty_nll" in req,
                champion=nc,
                challenger=nh,
                threshold=f"≤ champion + {cfg.max_nll_increase}",
            )
        )
        wins = 0
        for w in report.by_window:
            score_c = w["champion"].get("return.rmse", math.inf) * (1 + _mean(w["champion"], ll))
            score_h = w["challenger"].get("return.rmse", math.inf) * (1 + _mean(w["challenger"], ll))
            wins += int(score_h <= score_c)
        frac = wins / len(report.by_window) if report.by_window else 0.0
        gates.append(
            GateResult(
                name="time_consistency",
                passed=bool(report.by_window) and frac >= cfg.min_window_win_fraction,
                required="time_consistency" in req,
                champion=None,
                challenger=frac,
                threshold=f"win fraction ≥ {cfg.min_window_win_fraction}",
                detail=f"{wins}/{len(report.by_window)} windows",
            )
        )
        worst = 0.0
        worst_regime = ""
        for regime, pair in report.by_regime.items():
            for key in ("return.rmse", "upside.log_loss", "downside.log_loss"):
                a, b = pair["champion"].get(key, math.nan), pair["challenger"].get(key, math.nan)
                if math.isfinite(a) and math.isfinite(b) and a > 0:
                    deg = b / a - 1.0
                    if deg > worst:
                        worst, worst_regime = deg, f"{regime}:{key}"
        gates.append(
            GateResult(
                name="regime_consistency",
                passed=worst <= cfg.max_regime_degradation,
                required="regime_consistency" in req,
                champion=0.0,
                challenger=worst,
                threshold=f"worst regime degradation ≤ {cfg.max_regime_degradation}",
                detail=worst_regime or f"{len(report.by_regime)} regimes checked",
            )
        )
    required_ok = all(g.passed for g in gates if g.required) and all(
        any(g.name == r for g in gates) for r in req
    )
    frac_passed = sum(g.passed for g in gates) / len(gates)
    promote = required_ok and frac_passed >= cfg.min_passed_fraction
    failed = [g.name for g in gates if not g.passed]
    if promote:
        reason = f"all required gates passed; {frac_passed:.0%} of gates passed"
    elif not required_ok:
        reason = "required gate(s) failed or missing: " + ", ".join(
            [g.name for g in gates if g.required and not g.passed]
            + [r for r in req if all(g.name != r for g in gates)]
        )
    else:
        reason = f"only {frac_passed:.0%} of gates passed (< {cfg.min_passed_fraction:.0%}); failed: {failed}"
    return PromotionDecision(
        promote=promote,
        champion_version=report.champion_version,
        challenger_version=report.challenger_version,
        n_observations=report.n,
        gates=gates,
        reason=reason,
        shadow_report=report.to_dict(),
    )
