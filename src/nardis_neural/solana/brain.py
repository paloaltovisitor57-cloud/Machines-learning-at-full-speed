"""SolanaBrain — the complete Solana ML module behind one small API.

    brain = SolanaBrain.bootstrap("workspaces/sol", history)   # or SolanaBrain("workspaces/sol")
    brain.ingest(event)                 # swaps, launches, migrations, LP changes, transfers
    report = brain.assess(mint)         # forecasts + risk + edge + fat-tail view + red flags
    brain.resolve()                     # label matured assessments → continual learning
    brain.maintenance()                 # adapt / retrain / promote / refit risk model

Every output is a probabilistic assessment.  There are no BUY/SELL decisions, no keys and
no transaction submission — the trading system decides what to do with the numbers.
"""

from __future__ import annotations

import json
import math
import pickle
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict, Field

from nardis_neural.config import NeuralConfig
from nardis_neural.data.datasets import MarketDataset
from nardis_neural.data.sequences import observations_to_arrays
from nardis_neural.schemas import NeuralObservation, NeuralPrediction
from nardis_neural.solana.capital.allocator import (
    Allocation,
    AllocatorConfig,
    BookState,
    CapitalAllocator,
    Signal,
)
from nardis_neural.solana.config import CURRENT_FEATURES, RISK_LABELS, SolanaConfig
from nardis_neural.solana.dataset import SolanaDataset, build_solana_dataset
from nardis_neural.solana.edge.barriers import BarrierSpec
from nardis_neural.solana.edge.model import EdgeModel
from nardis_neural.solana.edge.research import edge_features, research_markdown, run_edge_research
from nardis_neural.solana.events import Event
from nardis_neural.solana.features import SolanaFeatureBuilder
from nardis_neural.solana.forward import ForwardLedger
from nardis_neural.solana.labels import SolanaLabeler
from nardis_neural.solana.market import EventStore, SolanaMarket
from nardis_neural.solana.moonshot import MoonshotSpec, TailModel, moonshot_markdown, run_moonshot_research
from nardis_neural.solana.moonshot.guard import GuardConfig, assess_manipulation
from nardis_neural.solana.moonshot.labels import position_marks
from nardis_neural.solana.moonshot.online import MoonshotTracker
from nardis_neural.solana.risk import SolanaRiskModel
from nardis_neural.solana.stopping import (
    StoppingModel,
    StoppingPath,
    run_stopping_research,
    state_matrix,
    stopping_markdown,
)
from nardis_neural.solana.tape.features import TapeSpec, extract_tape, stack_tapes
from nardis_neural.solana.tape.model import TapeModel, window_label
from nardis_neural.solana.tape.research import run_tape_research, tape_markdown
from nardis_neural.training.continual import ContinualLearner
from nardis_neural.training.pipeline import train_engine

F32 = npt.NDArray[np.float32]


class SolanaAssessment(BaseModel):
    """Everything the ML module knows about one token right now (no trade decision)."""

    model_config = ConfigDict(use_attribute_docstrings=True)

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
    edge: dict[str, float] = Field(default_factory=dict)
    """Meta-labeling edge estimate for an executable round trip (latency, impact, fees):
    p_win, expected_net, uncertainty, edge_score (lower confidence bound), kelly_fraction,
    threshold and above_threshold (1.0 / 0.0).  Empty until ``fit_edge`` has been run."""
    moonshot: dict[str, float] = Field(default_factory=dict)
    """Fat-tail view of a ticket bought now: calibrated P(peak ≥ k) for every level
    (``p_ge_10x`` …), median and expected ladder multiple, lottery-Kelly fraction, tail
    index, epistemic spread, ``in_entry_window``, and the manipulation guard's ``trust``,
    ``vetoed`` (1.0 / 0.0), ``chase_score`` (trust × expected multiple — the blend of the tail
    and tape models when both are installed, see ``expected_multiple_blend`` — 0 when vetoed)
    and ``chase_rank`` within the assessed batch (1 = best).  Empty until ``fit_moonshot``."""
    tape: dict[str, float] = Field(default_factory=dict)
    """Tape Transformer view (reads the raw trade tape with learned wallet embeddings):
    calibrated ``p_ge_*x``, ``expected_multiple``, ``median_multiple``, ``lottery_kelly``,
    ``tail_index``, ``epistemic``, and the exit signal ``p_collapse_1m`` / ``_5m`` / ``_15m``
    / ``_1h`` (probability the value halves within that window).  Empty until ``fit_tape``."""


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


def _blend_views(moon: list[dict[str, float]], tape: list[dict[str, float]]) -> None:
    """With both models installed, rank by the blend of their expected payoffs.

    Across independent simulated markets neither the tail model nor the Tape Transformer
    ranked the tail best every time, while their average was never the worst, so
    ``chase_score`` (and ``chase_rank``) use ``expected_multiple_blend`` when available.
    """
    if not moon or not moon[0] or not tape or not tape[0]:
        return
    for m, t in zip(moon, tape, strict=True):
        m["expected_multiple_blend"] = 0.5 * (m["expected_multiple"] + t["expected_multiple"])
        if not m["vetoed"]:
            m["chase_score"] = m["trust"] * m["expected_multiple_blend"]
    order = sorted(range(len(moon)), key=lambda j: -moon[j]["chase_score"])
    for rank, j in enumerate(order, start=1):
        moon[j]["chase_rank"] = float(rank)


class SolanaBrain:
    """Live Solana intelligence over one workspace: causal market, neural ensemble, risk, edge and
    moonshot models.

    Loads whatever the workspace holds (``solana.yaml``, the neural champion, the event history
    or a streaming checkpoint, risk / edge / tail models, the online moonshot buffer).
    :meth:`bootstrap` creates a workspace from historical events.
    """

    def __init__(self, workspace: str | Path, device: torch.device | str | None = None) -> None:
        self.root = Path(workspace)
        self.cfg = SolanaConfig.load(self.root / "solana.yaml")
        self.learner = ContinualLearner(self.root, device=device)
        self.market = SolanaMarket(self.cfg)
        self.history = EventStore()
        self.streaming = False
        """Streaming mode: no event history is kept; the compact market state is checkpointed."""
        self.evict_idle_seconds = 2 * 3600.0
        self.max_risk_samples = 50_000
        state = self._state()
        market_file = self.root / "stream" / "market.pkl"
        hist = self.root / "events"
        if market_file.exists():
            with market_file.open("rb") as fh:  # our own checkpoint, written by save()
                self.market = pickle.load(fh)
            self.streaming = True
            self.evict_idle_seconds = float(state.get("evict_idle_seconds", self.evict_idle_seconds))
        elif hist.exists():
            loaded = EventStore.load(hist)
            self.market.ingest_many(loaded.sorted())
            self.history = loaded
        self.builder = SolanaFeatureBuilder(self.cfg)
        self.labeler = SolanaLabeler(self.cfg)
        self.risk = (
            SolanaRiskModel.load(self.root / "risk") if (self.root / "risk" / "risk.json").exists() else None
        )
        self.edge: EdgeModel | None = None
        self.edge_meta: dict[str, Any] = {}
        if (self.root / "edge" / "edge.json").exists():
            self.edge = EdgeModel.load(self.root / "edge")
            self.edge_meta = json.loads((self.root / "edge" / "research.json").read_text())
        self.moonshot: TailModel | None = None
        self.tape_model: TapeModel | None = (
            TapeModel.load(self.root / "tape") if (self.root / "tape" / "tape.json").exists() else None
        )
        """Tape Transformer (tail + collapse), installed by :meth:`fit_tape`."""
        self.stopping: StoppingModel | None = (
            StoppingModel.load(self.root / "stopping")
            if (self.root / "stopping" / "stopping.json").exists()
            else None
        )
        """Optimal-stopping exit model (continuation value), installed by :meth:`fit_stopping`."""
        self.guard = GuardConfig()
        """Thresholds of the moonshot manipulation guard (hard vetoes)."""
        if (self.root / "moonshot" / "tail.json").exists():
            self.moonshot = TailModel.load(self.root / "moonshot")
        spec = self.moonshot.spec if self.moonshot is not None else MoonshotSpec()
        online = self.root / "moonshot" / "online.npz"
        self.tracker = MoonshotTracker.load(online, spec) if online.exists() else MoonshotTracker(spec)
        """Samples, labels and buffers moonshot rows as events stream past (online learning)."""
        self.forward = ForwardLedger.load(
            self.root / "forward",
            self.moonshot.spec if self.moonshot is not None else MoonshotSpec(),
            self._ledger_alarm(),
        )
        """Paper tickets opened and settled from the assessments (forward test)."""
        self.moonshot_last_refit = float(state.get("moonshot_last_refit", -np.inf))
        self.pending: list[_Pending] = []
        self._vetoes: dict[str, list[str]] = {}
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
    def _state(self) -> dict[str, Any]:
        f = self.root / "solana_state.json"
        return dict(json.loads(f.read_text())) if f.exists() else {}

    def enable_streaming(self, evict_idle_seconds: float = 2 * 3600.0) -> None:
        """Bounded-memory mode for long streams (months of history or live).

        Events are no longer accumulated; tokens quiet for ``evict_idle_seconds`` are
        forgotten (after their moonshot rows are labelled); :meth:`save` checkpoints the
        compact market state instead of the event history.
        """
        self.streaming = True
        self.evict_idle_seconds = evict_idle_seconds
        self.history = EventStore()

    def ingest(self, event: Event) -> None:
        """Feed one event, in time order, into the market (and the event history unless streaming).

        Raises like :meth:`SolanaMarket.ingest` (``ValueError`` / ``KeyError``) for events it rejects.
        """
        self.market.ingest(event)
        if not self.streaming:
            self.history.add(event)

    def evict(self) -> list[str]:
        """Label finished moonshot rows, then drop tokens idle for ``evict_idle_seconds``."""
        now = self.market.now
        self.tracker.resolve(self.market, now)
        busy = {p.mint for p in self.pending} | set(self.tracker.pending) | set(self.forward.open)
        idle = max(self.evict_idle_seconds, self.market.reputation_seconds + 1.0)
        return self.market.evict(now, idle, keep=busy.__contains__)

    def ingest_many(self, events: Iterable[Event]) -> None:
        """Ingest events in order (see :meth:`ingest`)."""
        for e in events:
            self.ingest(e)

    def assess(self, mint: str) -> SolanaAssessment:
        """Assess one token at the current market time (see :meth:`assess_many`)."""
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
        ages = np.asarray([now - self.market.token(m).launch.t for m in mints], dtype=np.float64)
        self.tracker.observe(mints, current, ages, now)
        x = SolanaRiskModel.inputs(out["embedding"], current)
        probs, unc = self.risk.predict(x) if self.risk is not None else (None, None)
        edge = None
        if self.edge is not None:
            ex, _ = edge_features(out, probs, current, engine.config.horizon_names, list(engine.expert_names))
            edge = self.edge.predict(ex)
            threshold = float(self.edge_meta.get("threshold", 0.0))
        moon = self._moonshot_view(mints, out, probs, current, now)
        tape_views = self._tape_view(mints, current, now)
        _blend_views(moon, tape_views)
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
                    flags=red_flags(feats, pred, risk)
                    + [f"moonshot veto: {r}" for r in self._vetoes.pop(mint, [])],
                    edge={}
                    if edge is None
                    else {
                        "p_win": float(edge.p_win[i]),
                        "expected_net": float(edge.expected_net[i]),
                        "uncertainty": float(edge.uncertainty[i]),
                        "edge_score": float(edge.edge_score[i]),
                        "kelly_fraction": float(edge.kelly[i]),
                        "threshold": threshold,
                        "above_threshold": float(edge.edge_score[i] >= threshold),
                    },
                    moonshot=moon[i],
                    tape=tape_views[i],
                    features=feats,
                )
            )
        self.forward.observe(reports, now)
        return reports

    def _moonshot_view(
        self,
        mints: list[str],
        out: dict[str, Any],
        probs: npt.NDArray[Any] | None,
        current: npt.NDArray[Any],
        now: float,
    ) -> list[dict[str, float]]:
        model = self.moonshot
        if model is None:
            return [{} for _ in mints]
        if model.inputs == "neural":
            engine = self.learner.champion
            x, _ = edge_features(out, probs, current, engine.config.horizon_names, list(engine.expert_names))
        else:
            x = np.nan_to_num(current).astype(np.float32)
        tp = model.predict(x)
        oor = model.out_of_range_share(x)
        ood = np.asarray(out.get("ood_score", np.zeros(len(mints))), dtype=np.float64).reshape(-1)
        rug_col = RISK_LABELS.index("rug")
        spec = model.spec
        views = []
        for i, mint in enumerate(mints):
            age = now - self.market.token(mint).launch.t
            feats = SolanaFeatureBuilder.explain(current[i].astype(np.float64))
            epistemic = float(tp.survival_std[i].max())
            verdict = assess_manipulation(
                feats,
                float(probs[i, rug_col]) if probs is not None else None,
                float(ood[i]),
                epistemic,
                float(oor[i]),
                self.guard,
            )
            vetoed = bool(verdict.vetoes)
            v = {f"p_ge_{k:g}x": float(tp.survival[i, j]) for j, k in enumerate(spec.levels)}
            v |= {
                "median_multiple": float(tp.median_multiple[i]),
                "expected_multiple": float(tp.expected_multiple[i]),
                "lottery_kelly": 0.0 if vetoed else float(tp.lottery_kelly[i]) * verdict.trust,
                "tail_index": float(tp.tail_index[i]),
                "epistemic": epistemic,
                "out_of_range_share": float(oor[i]),
                "in_entry_window": float(spec.min_entry_age_seconds <= age <= spec.max_entry_age_seconds),
                "trust": verdict.trust,
                "vetoed": float(vetoed),
                "chase_score": 0.0 if vetoed else verdict.trust * float(tp.expected_multiple[i]),
            }
            v |= {f"guard.{name}": value for name, value in verdict.factors.items()}
            views.append(v)
            self._vetoes[mint] = verdict.vetoes
        order = sorted(range(len(views)), key=lambda j: -views[j]["chase_score"])
        for rank, j in enumerate(order, start=1):
            views[j]["chase_rank"] = float(rank)
        return views

    def assess_active(
        self, max_idle_seconds: float = 120.0, min_age_seconds: float | None = None
    ) -> list[SolanaAssessment]:
        """Assess every token that traded within ``max_idle_seconds`` and is at least ``min_age_seconds``
        old (default ``cfg.min_token_age_seconds``).
        """
        min_age = self.cfg.min_token_age_seconds if min_age_seconds is None else min_age_seconds
        now = self.market.now
        mints = [
            m
            for m in self.market.active_tokens(now, max_idle_seconds)
            if now - self.market.token(m).launch.t >= min_age
        ]
        return self.assess_many(mints)

    def moonshot_ranking(
        self, max_idle_seconds: float = 120.0, include_vetoed: bool = False
    ) -> list[SolanaAssessment]:
        """Active tokens inside the moonshot entry window, best ``chase_score`` first.

        A ranking for the trading system to consume — not an order.  Vetoed tokens are left
        out unless ``include_vetoed``.
        """
        if self.moonshot is None:
            raise RuntimeError("no tail model installed: run fit_moonshot / solana moonshot-research")
        spec = self.moonshot.spec
        now = self.market.now
        mints = [
            m
            for m in self.market.active_tokens(now, max_idle_seconds)
            if spec.min_entry_age_seconds <= now - self.market.token(m).launch.t <= spec.max_entry_age_seconds
        ]
        found = self.assess_many(mints)
        keep = [a for a in found if include_vetoed or not a.moonshot["vetoed"]]
        return sorted(keep, key=lambda a: -a.moonshot["chase_score"])

    def allocate(
        self,
        equity_sol: float,
        open_stakes: dict[str, float] | None = None,
        peak_equity_sol: float | None = None,
        cfg: AllocatorConfig | None = None,
        max_idle_seconds: float = 120.0,
    ) -> list[Allocation]:
        """Recommended stakes for the current moonshot opportunities (advice, never orders).

        Uses the manipulation-guarded ranking, the tail and tape views, the pool's real
        liquidity, the creator family, and the forward ledger's live track record.
        """
        signals = []
        for a in self.moonshot_ranking(max_idle_seconds):
            m = a.moonshot
            fam = self.market.family_of.get(a.mint)
            signals.append(
                Signal(
                    a.mint,
                    a.timestamp,
                    float(m.get("expected_multiple_blend", m["expected_multiple"])),
                    float(m["lottery_kelly"]),
                    1.0,  # the guard's trust is already inside lottery_kelly
                    float(m["epistemic"]),
                    float(np.expm1(a.features.get("liquidity_sol_log", 0.0))),
                    str(fam) if fam is not None else a.mint,
                    bool(m["vetoed"]),
                )
            )
        state = BookState(
            equity=equity_sol,
            peak_equity=peak_equity_sol if peak_equity_sol is not None else equity_sol,
            open_stakes=dict(open_stakes or {}),
        )
        return CapitalAllocator(cfg, track_record=self.forward.track_record()).allocate(signals, state)

    def resolve(self) -> int:
        """Label every assessment whose longest horizon has elapsed and learn from it."""
        now = self.market.now
        horizon = max(h.seconds for h in self.cfg.horizons)
        still, done = [], 0
        for p in self.pending:
            if p.t + max(horizon, self.cfg.risk_horizon_seconds) > now:
                still.append(p)
                continue
            if p.mint not in self.market.tokens:
                continue  # evicted: nothing left to label it with
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
        self.forward.settle(self.market, now)
        if len(self.risk_y) > self.max_risk_samples:  # rolling window for long streams
            cut = len(self.risk_y) - self.max_risk_samples
            del self.risk_x[:cut], self.risk_y[:cut], self.risk_t[:cut]
            self.risk_fitted_n = max(0, self.risk_fitted_n - cut)
        return done

    def maintenance(self, risk_refit_min_new: int = 200) -> dict[str, Any]:
        """Periodic upkeep: neural adaptation / retraining / promotion, risk refit, online tail refit,
        then :meth:`save`; returns a status summary.

        The risk model is refitted once ``risk_refit_min_new`` new labelled samples have arrived and
        every risk label has at least 3 positives.
        """
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
        moon = self.refit_moonshot_online()
        self.save()
        return {
            "moonshot_online": moon,
            "adapted": None if adapted is None else adapted.status,
            "full_retrain": None if retrained is None else retrained.status,
            "promoted": None if decision is None else decision.promote,
            "risk_refit": refit,
            "champion": self.learner.registry.champion_version,
            "pending": len(self.pending),
            "forward": self.forward.summary(),
        }

    def refit_moonshot_online(
        self,
        every_seconds: float = 6 * 3600.0,
        min_tokens: int = 40,
        members: int = 3,
        epochs: int = 60,
        tolerance: float = 0.02,
    ) -> dict[str, Any] | None:
        """Retrain the raw-input tail model from the streamed buffer, behind a hold-out gate.

        The candidate is fitted on the earliest 80 % of buffered tokens and must match the
        installed model's NLL (within ``tolerance``) on the latest 20 %; open rows enter as
        right-censored.  A model fitted by ``moonshot-research`` on neural inputs is left alone.
        """
        now = self.market.now
        if self.moonshot is not None and self.moonshot.inputs != "raw":
            return {"status": "skipped", "reason": "neural-input model is managed by moonshot-research"}
        if now - self.moonshot_last_refit < every_seconds:
            return None
        self.tracker.resolve(self.market, now)
        data = self.tracker.training_set(self.market, now)
        if data is None or data.tokens < min_tokens:
            return {"status": "waiting", "tokens": 0 if data is None else data.tokens}
        self.moonshot_last_refit = now
        first: dict[str, float] = {}
        for m, t in zip(data.mints.tolist(), data.t.tolist(), strict=True):
            first[m] = min(first.get(m, np.inf), t)
        ordered = sorted(first, key=first.__getitem__)
        recent = set(ordered[int(len(ordered) * 0.8) :])
        hold = np.array([m in recent for m in data.mints.tolist()])
        tr = np.flatnonzero(~hold)
        ho = np.flatnonzero(hold)
        cand = TailModel(
            data.x.shape[1],
            self.tracker.spec,
            members=members,
            feature_names=list(CURRENT_FEATURES),
            inputs="raw",
        )
        cand.fit(data.x[tr], data.peak[tr], data.censored[tr], data.t[tr], data.mints[tr], epochs=epochs)
        new = float(cand.nll(data.x[ho], data.peak[ho], data.censored[ho]).mean())
        old = (
            float(self.moonshot.nll(data.x[ho], data.peak[ho], data.censored[ho]).mean())
            if self.moonshot is not None and self.moonshot.d_in == data.x.shape[1]
            else float("inf")
        )
        accepted = new <= old + tolerance
        if accepted:
            cand.report |= {"online": True, "holdout_nll": new, "previous_holdout_nll": old}
            cand.save(self.root / "moonshot")
            self.moonshot = cand
        return {
            "status": "promoted" if accepted else "rejected",
            "tokens": data.tokens,
            "rows": len(data),
            "holdout_nll": new,
            "previous_holdout_nll": old,
        }

    # ------------------------------------------------------------------ edge research
    def fit_edge(
        self,
        spec: BarrierSpec | None = None,
        n_folds: int = 4,
        max_positions: int = 5,
        history: EventStore | None = None,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Walk-forward edge research on the workspace history; installs the edge model."""
        engine = self.learner.champion
        research = run_edge_research(
            history or self.history,
            self.cfg,
            engine.config,
            spec or BarrierSpec(size_sol=self.cfg.trade_size_sol),
            n_folds=n_folds,
            max_positions=max_positions,
            device=engine.device,
            log=log or (lambda _: None),
        )
        research.model.save(self.root / "edge")
        report_json = json.dumps(research.report, indent=2, default=float)
        (self.root / "edge" / "research.json").write_text(report_json)
        (self.root / "edge" / "REPORT.md").write_text(research_markdown(research.report))
        self.edge, self.edge_meta = research.model, research.report
        return research.report

    def _ledger_alarm(self) -> tuple[str, float] | None:
        """Exit alarm chosen by the last tape research (window key, threshold), if any."""
        f = self.root / "tape" / "research.json"
        if not f.exists():
            return None
        chosen = json.loads(f.read_text()).get("exit_policy", {}).get("tune", {}).get("chosen", {})
        if chosen.get("theta") is None:
            return None
        return str(chosen["window"]), float(chosen["theta"])

    def _tape_view(self, mints: list[str], current: npt.NDArray[Any], now: float) -> list[dict[str, float]]:
        """Tape Transformer outputs for each mint (empty dicts when no tape model is installed)."""
        model = self.tape_model
        if model is None:
            return [{} for _ in mints]
        tapes = [
            extract_tape(self.market.token(m), self.market.wallets, now, model.tape, self.cfg) for m in mints
        ]
        tx, tw, tm = stack_tapes(tapes)
        pred = model.predict(tx, tw, tm, current)
        views = []
        for i in range(len(mints)):
            v = {f"p_ge_{k:g}x": float(pred.tail.survival[i, j]) for j, k in enumerate(model.spec.levels)}
            v |= {
                "expected_multiple": float(pred.tail.expected_multiple[i]),
                "median_multiple": float(pred.tail.median_multiple[i]),
                "lottery_kelly": float(pred.tail.lottery_kelly[i]),
                "tail_index": float(pred.tail.tail_index[i]),
                "epistemic": float(pred.tail.survival_std[i].max()),
            }
            v |= {
                f"p_collapse_{window_label(b)}": float(pred.collapse[i, j])
                for j, b in enumerate(model.collapse_bins)
            }
            views.append(v)
        return views

    def fit_tape(
        self,
        spec: MoonshotSpec | None = None,
        tape: TapeSpec | None = None,
        test_fraction: float = 0.35,
        members: int = 3,
        epochs: int = 40,
        history: EventStore | None = None,
        archetypes: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
        d: int = 64,
        layers: int = 2,
    ) -> dict[str, Any]:
        """Tape Transformer research on the workspace history; installs the refitted model."""
        research = run_tape_research(
            history or self.history,
            self.cfg,
            self.learner.champion.config,
            spec or (self.moonshot.spec if self.moonshot is not None else MoonshotSpec()),
            tape,
            test_fraction=test_fraction,
            members=members,
            epochs=epochs,
            d=d,
            layers=layers,
            archetypes=archetypes,
            log=log or (lambda _: None),
        )
        out = self.root / "tape"
        research.model.save(out)
        (out / "research.json").write_text(json.dumps(research.report, indent=2, default=float))
        (out / "REPORT.md").write_text(tape_markdown(research.report))
        self.tape_model = research.model
        self.forward.alarm = self._ledger_alarm()
        return research.report

    def fit_stopping(
        self,
        spec: MoonshotSpec | None = None,
        test_fraction: float = 0.35,
        spacing: float = 30.0,
        history: EventStore | None = None,
        archetypes: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Optimal-stopping exit research on the workspace history; installs the refitted model."""
        report, model = run_stopping_research(
            history or self.history,
            self.cfg,
            spec or (self.moonshot.spec if self.moonshot is not None else MoonshotSpec()),
            test_fraction=test_fraction,
            spacing=spacing,
            archetypes=archetypes,
            log=log,
        )
        d = self.root / "stopping"
        model.report = model.report | {"research": report, "spacing": spacing}
        model.save(d)
        (d / "REPORT.md").write_text(stopping_markdown(report))
        self.stopping = model
        return report

    def hold_advice(self, mint: str, t_signal: float) -> dict[str, float]:
        """Sell-or-hold advice for a ticket signalled at ``t_signal`` (an estimate, not an order).

        Marks the position at the model's decision spacing up to now, then compares the
        utility of liquidating now with the estimated continuation value.  ``advantage > 0``
        means holding is worth more than selling now.  Empty without a fitted model.
        """
        if self.stopping is None or self.stopping.trees is None:
            return {}
        spec = self.moonshot.spec if self.moonshot is not None else MoonshotSpec()
        now = self.market.now
        spacing = float(self.stopping.report.get("spacing", 30.0))
        grid = np.r_[np.arange(t_signal + spec.latency_seconds, now, spacing), now]
        priced = position_marks(self.market.token(mint), t_signal, spec, grid)
        if priced is None or len(priced[0]) == 0:
            return {}
        times, marks = priced
        cur = self.builder.observation(self.market, mint, now).current_features.astype(np.float32)
        path = StoppingPath(mint, t_signal, times, marks, np.repeat(cur[None, :], len(times), axis=0))
        cont = float(self.stopping.continuation(state_matrix(path)[-1:])[0])
        sell = float(self.stopping.u(marks[-1:])[0])
        return {
            "liquidation_multiple": float(marks[-1]),
            "sell_now_utility": sell,
            "continuation_utility": cont,
            "advantage": cont - sell,
        }

    def fit_moonshot(
        self,
        spec: MoonshotSpec | None = None,
        inputs: str = "raw",
        n_folds: int = 4,
        test_fraction: float = 0.35,
        min_expected_multiple: float = 1.0,
        history: EventStore | None = None,
        archetypes: dict[str, str] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """Fat-tail research on the workspace history; installs the tail model."""
        engine = self.learner.champion
        research = run_moonshot_research(
            history or self.history,
            self.cfg,
            engine.config,
            spec or MoonshotSpec(),
            inputs=inputs,
            n_folds=n_folds,
            test_fraction=test_fraction,
            min_expected_multiple=min_expected_multiple,
            archetypes=archetypes,
            device=engine.device,
            log=log or (lambda _: None),
        )
        d = self.root / "moonshot"
        research.model.save(d)
        (d / "research.json").write_text(json.dumps(research.report, indent=2, default=float))
        (d / "REPORT.md").write_text(moonshot_markdown(research.report))
        self.moonshot = research.model
        self.forward.spec = research.model.spec
        return research.report

    # ------------------------------------------------------------------ persistence
    def _load_risk_samples(self) -> None:
        f = self.root / "risk_samples.npz"
        if f.exists():
            with np.load(f) as z:
                self.risk_x = list(z["x"])
                self.risk_y = list(z["y"])
                self.risk_t = list(z["t"])

    def save(self) -> None:
        """Checkpoint the workspace: learner, event history (or the pickled market when streaming),
        moonshot buffer, risk samples and ``solana_state.json``.
        """
        self.learner.save()
        if self.streaming:
            target = self.root / "stream" / "market.pkl"
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(self.market, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(target)  # atomic: a crash never leaves a torn checkpoint
        else:
            self.history.save(self.root / "events")
        (self.root / "moonshot").mkdir(exist_ok=True)
        self.tracker.save(self.root / "moonshot" / "online.npz")
        self.forward.save()
        if self.risk_y:
            np.savez(
                self.root / "risk_samples.npz",
                x=np.stack(self.risk_x),
                y=np.stack(self.risk_y),
                t=np.asarray(self.risk_t),
            )
        (self.root / "solana_state.json").write_text(
            json.dumps(
                {
                    "market_now": self.market.now,
                    "events": len(self.history),
                    "pending": len(self.pending),
                    "streaming": self.streaming,
                    "evict_idle_seconds": self.evict_idle_seconds,
                    "moonshot_last_refit": self.moonshot_last_refit,
                    "tokens_in_memory": len(self.market.tokens),
                    "wallets": len(self.market.wallets),
                    "evicted": self.market.evicted,
                }
            )
        )


def fit_risk_model(embeddings: npt.NDArray[Any], ds: SolanaDataset, members: int = 3) -> SolanaRiskModel:
    """Fit a :class:`SolanaRiskModel` on embeddings plus current features of ``ds``, validated on
    held-out later-launched tokens (token-disjoint).
    """
    x = SolanaRiskModel.inputs(embeddings, ds.current)
    model = SolanaRiskModel(x.shape[1], members=members)
    model.fit(x, ds.risk, np.asarray(ds.arrays["timestamp"], dtype=np.float64), groups=ds.mints)
    return model
