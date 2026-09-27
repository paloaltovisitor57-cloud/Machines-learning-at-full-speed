"""SolanaBrain — the complete Solana ML module behind one small API.

    brain = SolanaBrain.bootstrap("workspaces/sol", history)   # or SolanaBrain("workspaces/sol")
    brain.ingest(event)                 # swaps, launches, migrations, LP changes, transfers
    report = brain.assess(mint)         # forecasts + risk + cost-aware edge + red flags
    brain.resolve()                     # label matured assessments → continual learning
    brain.maintenance()                 # adapt / retrain / promote / refit risk model

Every output is a probabilistic assessment.  There are no BUY/SELL decisions, no keys and
no transaction submission — the trading system decides what to do with the numbers.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, Field

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.sequences import observations_to_arrays
from nardis_neural.schemas import NeuralObservation, NeuralPrediction
from nardis_neural.solana.config import RISK_LABELS, SolanaConfig
from nardis_neural.solana.dataset import SolanaDataset, build_solana_dataset
from nardis_neural.solana.events import Event
from nardis_neural.solana.features import SolanaFeatureBuilder
from nardis_neural.solana.labels import SolanaLabeler
from nardis_neural.solana.market import EventStore, SolanaMarket
from nardis_neural.solana.risk import SolanaRiskModel
from nardis_neural.training.continual import ContinualLearner
from nardis_neural.training.pipeline import train_engine

F32 = npt.NDArray[np.float32]


class SolanaAssessment(BaseModel):
    """Everything the ML module knows about one token right now (no trade decision)."""

    mint: str
    timestamp: float
    venue: str
    model_version: str
    prediction: NeuralPrediction
    risk: dict[str, float]
    """Calibrated P(rug), P(graduation), P(dev_dump) within the risk horizon."""
    risk_uncertainty: dict[str, float]
    round_trip_cost: float
    """Fractional cost of entering and exiting ``trade_size_sol`` right now."""
    expected_net_return: dict[str, float]
    """E[simple return] − round-trip cost, per horizon."""
    prob_net_positive: dict[str, float]
    """P(return beats the round-trip cost) under the predictive distribution."""
    flags: list[str] = Field(default_factory=list)
    features: dict[str, float] = Field(default_factory=dict)


def _normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def red_flags(f: dict[str, float], pred: NeuralPrediction, risk: dict[str, float]) -> list[str]:
    """Human-readable explanations of on-chain red flags present in the features."""
    out = []
    if f["mint_authority_revoked"] < 0.5:
        out.append("mint authority is live: supply can still be inflated")
    if f["freeze_authority_revoked"] < 0.5:
        out.append("freeze authority is live: holder accounts can be frozen")
    if f["top10_share"] > 0.35:
        out.append(f"top-10 wallets hold {f['top10_share']:.0%} of supply")
    if f["dev_share"] > 0.05:
        out.append(f"creator still holds {f['dev_share']:.1%} of supply")
    if f["dev_sold_fraction"] > 0.5:
        out.append(f"creator has sold {f['dev_sold_fraction']:.0%} of its bag")
    if f["bundle_share"] > 0.05:
        out.append(f"bundled launch: {f['bundle_share']:.1%} held by co-funded early wallets")
    if f["creator_cluster_share"] > 0.1:
        out.append(f"creator's funding cluster holds {f['creator_cluster_share']:.1%}")
    if f["sniper_share"] > 0.1:
        out.append(f"snipers hold {f['sniper_share']:.1%}")
    if f["fresh_wallet_share_60s"] > 0.5:
        out.append("most recent buyers are freshly funded wallets")
    if f["rug_associated_share"] > 0.2:
        out.append("wallets linked to earlier rugs are trading it")
    if f["bot_share_60s"] > 0.5:
        out.append("volume dominated by high-frequency bots (possible wash trading)")
    if f["round_trip_cost"] > 0.05:
        out.append(f"thin liquidity: round trip costs {f['round_trip_cost']:.1%}")
    if pred.ood_score > 1.0:
        out.append(f"market state is out of distribution (ood={pred.ood_score:.2f})")
    if risk.get("rug", 0.0) > 0.5:
        out.append(f"high rug probability ({risk['rug']:.0%})")
    return out


@dataclass
class _Pending:
    observation: NeuralObservation
    prediction: NeuralPrediction
    risk_x: F32
    t: float
    mint: str


class SolanaBrain:
    def __init__(self, workspace: str | Path, device: torch.device | str | None = None) -> None:
        self.root = Path(workspace)
        self.cfg = SolanaConfig.load(self.root / "solana.yaml")
        self.learner = ContinualLearner(self.root, device=device)
        self.market = SolanaMarket(self.cfg)
        self.history = EventStore()
        hist = self.root / "events"
        if hist.exists():
            loaded = EventStore.load(hist)
            self.market.ingest_many(loaded.sorted())
            self.history = loaded
        self.builder = SolanaFeatureBuilder(self.cfg)
        self.labeler = SolanaLabeler(self.cfg)
        self.risk = (
            SolanaRiskModel.load(self.root / "risk") if (self.root / "risk" / "risk.json").exists() else None
        )
        self.pending: list[_Pending] = []
        self.risk_x: list[F32] = []
        self.risk_y: list[F32] = []
        self.risk_t: list[float] = []
        self.risk_fitted_n = 0
        self._load_risk_samples()

    # ------------------------------------------------------------------ bootstrap
    @classmethod
    def bootstrap(
        cls,
        workspace: str | Path,
        history: EventStore,
        cfg: SolanaConfig | None = None,
        neural_cfg: NeuralConfig | None = None,
        device: torch.device | str | None = None,
        log: Callable[[str], None] | None = None,
    ) -> SolanaBrain:
        """Train the neural ensemble and the risk model from historical events."""
        root = Path(workspace)
        root.mkdir(parents=True, exist_ok=True)
        cfg = cfg or SolanaConfig()
        ncfg = cfg.neural_config(neural_cfg)
        say = log or (lambda _: None)
        ds = build_solana_dataset(history, cfg, ncfg)
        say(f"built {len(ds)} leakage-free snapshots from {len(history)} events")
        dev = torch.device(device) if device is not None else None
        engine, report = train_engine(ncfg, ds.store(), device=dev, log=say)
        m = report.validation_metrics
        say(
            f"neural ensemble: return rank-corr {m.get('return.rank_corr', float('nan')):.3f}, "
            f"downside AUC {m.get('downside.auc', float('nan')):.3f}"
        )
        cfg.save(root / "solana.yaml")
        ContinualLearner.initialize(root, engine, ncfg, device=engine.device)
        history.save(root / "events")
        embeddings = engine.predict_dataset(MarketDataset(ds.store(), ncfg))["embedding"]
        risk = fit_risk_model(embeddings, ds)
        risk.save(root / "risk")
        say(f"risk model: { {k: round(v['auc'], 3) for k, v in risk.report['metrics'].items()} }")
        return cls(root, device=device)

    # ------------------------------------------------------------------ streaming
    def ingest(self, event: Event) -> None:
        self.market.ingest(event)
        self.history.add(event)

    def ingest_many(self, events: Iterable[Event]) -> None:
        for e in events:
            self.ingest(e)

    def assess(self, mint: str) -> SolanaAssessment:
        return self.assess_many([mint])[0]

    def assess_many(self, mints: list[str]) -> list[SolanaAssessment]:
        """Assess several tokens at the current market time with one batched forward pass."""
        if not mints:
            return []
        now = self.market.now
        observations = [self.builder.observation(self.market, m, now) for m in mints]
        engine = self.learner.champion
        out = engine.predict_arrays(observations_to_arrays(observations, engine.config))
        preds = engine.to_predictions(out)
        # shadow the challenger on identical inputs (recorded, never returned)
        challenger = self.learner.challenger
        if challenger is not None:
            chall = challenger.predict_arrays(observations_to_arrays(observations, challenger.config))
            self.learner.shadow.record(out, chall, engine.version, challenger.version)
        current = np.stack([o.current_features for o in observations])
        x = SolanaRiskModel.inputs(out["embedding"], current)
        probs, unc = self.risk.predict(x) if self.risk is not None else (None, None)
        reports = []
        for i, (mint, obs, pred) in enumerate(zip(mints, observations, preds, strict=True)):
            self.learner.record_prediction(pred)
            feats = SolanaFeatureBuilder.explain(obs.current_features.astype(np.float64))
            risk: dict[str, float] = {}
            risk_unc: dict[str, float] = {}
            if probs is not None and unc is not None:
                risk = {k: float(probs[i, j]) for j, k in enumerate(RISK_LABELS)}
                risk_unc = {k: float(unc[i, j]) for j, k in enumerate(RISK_LABELS)}
            cost = feats["round_trip_cost"]
            exp_net, p_net = {}, {}
            for h in pred.horizons:
                mu, sd = pred.expected_returns[h], max(pred.return_std[h], 1e-6)
                exp_net[h] = float(math.expm1(mu + 0.5 * sd**2) - cost)
                p_net[h] = _normal_sf((math.log1p(cost) - mu) / sd)
            self.pending.append(_Pending(obs, pred, x[i], now, mint))
            reports.append(
                SolanaAssessment(
                    mint=mint,
                    timestamp=now,
                    venue=self.market.token(mint).venue,
                    model_version=pred.model_version,
                    prediction=pred,
                    risk=risk,
                    risk_uncertainty=risk_unc,
                    round_trip_cost=cost,
                    expected_net_return=exp_net,
                    prob_net_positive=p_net,
                    flags=red_flags(feats, pred, risk),
                    features=feats,
                )
            )
        return reports

    def assess_active(
        self, max_idle_seconds: float = 120.0, min_age_seconds: float | None = None
    ) -> list[SolanaAssessment]:
        min_age = self.cfg.min_token_age_seconds if min_age_seconds is None else min_age_seconds
        now = self.market.now
        mints = [
            m
            for m in self.market.active_tokens(now, max_idle_seconds)
            if now - self.market.token(m).launch.t >= min_age
        ]
        return self.assess_many(mints)

    def resolve(self) -> int:
        """Label every assessment whose longest horizon has elapsed and learn from it."""
        now = self.market.now
        horizon = max(h.seconds for h in self.cfg.horizons)
        still, done = [], 0
        for p in self.pending:
            if p.t + max(horizon, self.cfg.risk_horizon_seconds) > now:
                still.append(p)
                continue
            labels = self.labeler.label(self.market.token(p.mint), p.t, now)
            if labels is None:
                continue
            self.learner.add_experience(p.observation, labels.outcome, p.prediction)
            if labels.risk is not None:
                self.risk_x.append(p.risk_x)
                self.risk_y.append(np.asarray([labels.risk[k] for k in RISK_LABELS], dtype=np.float32))
                self.risk_t.append(p.t)
            done += 1
        self.pending = still
        return done

    def maintenance(self, risk_refit_min_new: int = 200) -> dict[str, Any]:
        adapted = self.learner.adapt_if_needed()
        retrained = self.learner.full_retrain_if_needed()
        decision = self.learner.promote_if_ready()
        refit = False
        if len(self.risk_y) - self.risk_fitted_n >= risk_refit_min_new:
            y = np.stack(self.risk_y)
            if y.sum(axis=0).min() >= 3:  # every risk label needs some positive examples
                model = SolanaRiskModel(self.risk_x[0].shape[0])
                model.fit(np.stack(self.risk_x), y, np.asarray(self.risk_t))
                self.risk = model
                model.save(self.root / "risk")
                self.risk_fitted_n = len(self.risk_y)
                refit = True
        self.save()
        return {
            "adapted": None if adapted is None else adapted.status,
            "full_retrain": None if retrained is None else retrained.status,
            "promoted": None if decision is None else decision.promote,
            "risk_refit": refit,
            "champion": self.learner.registry.champion_version,
            "pending": len(self.pending),
        }

    # ------------------------------------------------------------------ persistence
    def _load_risk_samples(self) -> None:
        f = self.root / "risk_samples.npz"
        if f.exists():
            with np.load(f) as z:
                self.risk_x = list(z["x"])
                self.risk_y = list(z["y"])
                self.risk_t = list(z["t"])

    def save(self) -> None:
        self.learner.save()
        self.history.save(self.root / "events")
        if self.risk_y:
            np.savez(
                self.root / "risk_samples.npz",
                x=np.stack(self.risk_x),
                y=np.stack(self.risk_y),
                t=np.asarray(self.risk_t),
            )
        (self.root / "solana_state.json").write_text(
            json.dumps(
                {"market_now": self.market.now, "events": len(self.history), "pending": len(self.pending)}
            )
        )


def fit_risk_model(embeddings: npt.NDArray[Any], ds: SolanaDataset, members: int = 3) -> SolanaRiskModel:
    x = SolanaRiskModel.inputs(embeddings, ds.current)
    model = SolanaRiskModel(x.shape[1], members=members)
    model.fit(x, ds.risk, np.asarray(ds.arrays["timestamp"], dtype=np.float64), groups=ds.mints)
    return model
