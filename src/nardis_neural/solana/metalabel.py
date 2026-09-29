"""Meta-labeling: learn from the trading system's own trades.

The trading system (Nardis) decides *what* to trade.  This module learns, from the outcomes of
its own past trades, *how good each new proposal is likely to be*, and answers in
milliseconds before the trade:

* ``p_win``: probability that the trade returns more than it cost;
* ``chase``: for every fixed chase target (2x, 5x, 10x, 100x, 1000x, see
  :mod:`nardis_neural.solana.chase`) the probability of reaching it, its break-even probability,
  the edge ratio, the lift over the trading system's average trade, and the ``chase_target``:
  the most ambitious multiple that is proven (≥ 3 real hits) and +EV for this proposal.
  Reaching a target is judged on the peak multiple when the trading system reports it, else
  on the realised multiple;
* ``p_10x`` / ``p_100x``: shortcuts for two of the chase probabilities;
* ``expected_multiple``;
* ``size_multiplier``: scale for the trading system's own stake, above 1 for proposals that
  look better than its average trade and below 1 for worse ones, tilted up (at most 1.5x) for
  a proven tail edge, capped at 2;
* ``veto`` with a reason, when the proposal matches a pattern that has been losing.

Inputs are whatever named numbers the trading system sends with each proposal, optionally
joined with this addon's own market view of the token at that moment.

Safeguards:

* **cold start**: until ``min_trades`` trades have settled, answers come from Bayesian base rates
  (Jeffreys prior) of the trades seen so far, never from a model;
* **champion / challenger**: every refit is trained on the older 80 % of trades and scored on the
  newest 20 %, in time order; a model is deployed only if it beats the base rate there
  (log loss), otherwise the base rate stays;
* **no lookahead**: labels only come from settled outcomes, and features are frozen at advice time;
* **unproven tails stay at break-even**: until a chase target has ``MIN_CHASE_HITS`` real hits among
  settled trades, its probability (``p_10x``, ``p_100x`` and the chase's ``p_{k}x``) is capped at
  that target's break-even probability, so a base rate on a handful of trades (0.5 for every
  level with no trades at all) never shows up as a huge ``edge_{k}x``; ``tail_ev`` banks proven
  targets only (with none proven, every position is valued at the loss multiple);
* **bounded bookkeeping**: at most ``max_pending`` proposals wait for their outcome (20 000 by
  default); beyond that the oldest recorded one is dropped, and settling it then returns False.
  No clock is involved, so proposals may stamp ``t`` in any unit.  Settling a trade id already
  in the history is refused, never counted twice.

It never places, sizes or cancels anything itself; every output is advice.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.chase import CHASE_TARGETS, DEFAULT_LOSS_MULTIPLE, break_even, chase_profile
from nardis_neural.solana.edge.trees import TreeEnsemble

F64 = npt.NDArray[np.float64]
LEVELS = (1.0, *CHASE_TARGETS)
"""Outcome thresholds learned: win (> 1x) and every chase target (2x, 5x, 10x, 100x, 1000x)."""
MIN_CHASE_HITS = 3
"""Real hits a chase target needs before it is *proven* (as in :func:`chase_profile`)."""


@dataclass
class TradeProposal:
    """A trade the trading system is about to make."""

    trade_id: str
    mint: str
    t: float
    features: dict[str, float] = field(default_factory=dict)
    """The trading system's own signal values (any names)."""


@dataclass
class TradeOutcome:
    """The settled result of a proposal."""

    trade_id: str
    t_exit: float
    multiple: float
    """SOL returned per SOL staked, fees included."""
    peak_multiple: float | None = None
    """Best multiple the position reached, if the trading system tracks it (used for tail labels)."""


@dataclass
class TradeAdvice:
    """What the learner thinks of a proposal (advice only)."""

    trade_id: str
    p_win: float
    p_10x: float
    p_100x: float
    expected_multiple: float
    size_multiplier: float
    veto: bool
    reason: str
    evidence: int
    """Settled trades the answer rests on."""
    source: str
    """``prior`` (base rates) or ``learned`` (deployed model)."""
    chase: dict[str, float] = field(default_factory=dict)
    """The chase of 2x / 5x / 10x / 100x / 1000x for this proposal (see :func:`chase_profile`)."""


def _sigmoid(z: F64) -> F64:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _log_loss(p: F64, y: F64) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


class MetaLearner:
    """Online meta-labeling of the trading system's proposals."""

    def __init__(
        self,
        min_trades: int = 50,
        refit_every: int = 25,
        min_positives: int = 8,
        max_size: float = 2.0,
        veto_ratio: float = 0.5,
        seed: int = 0,
        max_pending: int = 20_000,
    ) -> None:
        self.min_trades, self.refit_every, self.min_positives = min_trades, refit_every, min_positives
        self.max_size, self.veto_ratio, self.seed = max_size, veto_ratio, seed
        self.max_pending = max_pending
        """Proposals kept waiting for an outcome; the oldest recorded one is dropped beyond this."""
        self.feature_names: list[str] = []
        self.pending: dict[str, tuple[float, dict[str, float]]] = {}
        """Unsettled proposals, oldest recorded first."""
        self.x: list[dict[str, float]] = []
        self.t: list[float] = []
        self.multiple: list[float] = []
        self.peak: list[float] = []
        self.trade_ids: list[str] = []
        self._known: set[str] = set()
        self.models: dict[float, TreeEnsemble] = {}
        self.value_model: TreeEnsemble | None = None
        self.smear = 1.0
        """Duan smearing factor: mean exp(residual) of the value model on held-out trades."""
        self.report: dict[str, Any] = {}
        self._since_fit = 0
        self._generation = 0
        """Tag of the tree files written by the last :meth:`save`."""

    # ------------------------------------------------------------------ data
    def _matrix(self, rows: list[dict[str, float]]) -> F64:
        m = np.full((len(rows), len(self.feature_names)), np.nan)
        for i, r in enumerate(rows):
            for j, name in enumerate(self.feature_names):
                v = r.get(name)
                if v is not None and math.isfinite(float(v)):
                    m[i, j] = float(v)
        return m

    def _labels(self, level: float) -> F64:
        mult = np.asarray(self.multiple)
        if level <= 1.0:
            return (mult > level).astype(np.float64)
        best = np.maximum(np.asarray(self.peak), mult)
        return (best >= level).astype(np.float64)

    def _prior(self, level: float) -> float:
        """Jeffreys-prior base rate of reaching ``level`` among settled trades."""
        n = len(self.multiple)
        k = float(self._labels(level).sum()) if n else 0.0
        return (k + 0.5) / (n + 1.0)

    # ------------------------------------------------------------------ advice
    def advise(
        self, proposal: TradeProposal, market: dict[str, float] | None = None, record: bool = True
    ) -> TradeAdvice:
        """Score a proposal; with ``record`` it is kept pending until :meth:`settle` (or until
        ``max_pending`` newer proposals push it out).  A trade id already settled is not recorded
        again."""
        feats = dict(proposal.features)
        if market:
            feats |= {f"addon_{k}": float(v) for k, v in market.items()}
        if record and proposal.trade_id not in self._known:
            self.pending.pop(proposal.trade_id, None)  # re-advised: now the newest
            self.pending[proposal.trade_id] = (proposal.t, feats)
            while len(self.pending) > self.max_pending:
                del self.pending[next(iter(self.pending))]
        n = len(self.multiple)
        base = {lv: self._prior(lv) for lv in LEVELS}
        base_mean = float(np.mean(self.multiple)) if n else 1.0
        p = dict(base)
        exp_mult = base_mean
        source = "prior"
        if self.models or self.value_model is not None:
            x = self._matrix([feats])
            for lv, model in self.models.items():
                p[lv] = float(_sigmoid(model.predict(x))[0])
                source = "learned"
            if self.value_model is not None:
                # E[1 + m] = exp(E[log(1 + m)]) · E[exp(residual)] (Duan's smearing estimator)
                exp_mult = float(np.exp(self.value_model.predict(x))[0] * self.smear) - 1.0
                source = "learned"
        losers = [m for m in self.multiple if m <= 1.0]
        loss = float(np.clip(np.mean(losers), 0.0, 0.99)) if len(losers) >= 20 else DEFAULT_LOSS_MULTIPLE
        hits = {lv: int(self._labels(lv).sum()) if n else 0 for lv in CHASE_TARGETS}
        # an unproven target is never reported above break-even (see the module docstring)
        chase_base = dict(base)
        for lv in CHASE_TARGETS:
            if hits[lv] < MIN_CHASE_HITS:
                cap = break_even(lv, loss)
                p[lv], chase_base[lv] = min(p[lv], cap), min(base[lv], cap)
        # a higher target can never be more likely than a lower one, or than winning at all
        prev = p[1.0]
        for lv in CHASE_TARGETS:
            p[lv] = min(p[lv], prev)
            prev = p[lv]
        chase = chase_profile({lv: p[lv] for lv in CHASE_TARGETS}, loss, chase_base, hits, MIN_CHASE_HITS)
        # the ladder banks proven targets only: rungs held at break-even would still add (1 - L) each
        proven = {lv: p[lv] if hits[lv] >= MIN_CHASE_HITS else 0.0 for lv in CHASE_TARGETS}
        chase["tail_ev"] = chase_profile(proven, loss)["tail_ev"]
        ratio_p = p[1.0] / max(base[1.0], 1e-6)
        ratio_m = exp_mult / max(base_mean, 1e-6) if base_mean > 0 else 1.0
        size = 1.0
        if source == "learned":
            size = ratio_p * ratio_m
            if chase["chase_target"] > 0:  # a proven, +EV shot at a big multiple earns a tilt
                size *= min(1.0 + 0.25 * math.log2(max(chase["chase_edge"], 1.0)), 1.5)
            size = float(np.clip(size, 0.0, self.max_size))
        veto, reason = False, "no settled history" if n == 0 else "base rates only"
        if source == "learned":
            reason = "learned"
            if p[1.0] < self.veto_ratio * base[1.0] and exp_mult < 1.0:
                veto = True
                reason = (
                    f"similar trades have been losing: P(win) {p[1.0]:.0%} against {base[1.0]:.0%} "
                    f"on average, expected {exp_mult:.2f}x"
                )
        if chase["chase_target"] > 0 and not veto:
            reason += f"; chase {chase['chase_target']:g}x (edge {chase['chase_edge']:.1f}x break-even)"
        return TradeAdvice(
            proposal.trade_id,
            p[1.0],
            p[10.0],
            p[100.0],
            exp_mult,
            size,
            veto,
            reason,
            n,
            source,
            chase,
        )

    # ------------------------------------------------------------------ learning
    def knows(self, trade_id: str) -> bool:
        """True when this trade has already been settled into the history."""
        return trade_id in self._known

    def settle(self, outcome: TradeOutcome) -> bool:
        """Record a settled trade; refits when due.  Returns False for unknown, dropped (see
        ``max_pending``) or already settled trade ids (a trade is learned once)."""
        entry = self.pending.pop(outcome.trade_id, None)
        if entry is None or outcome.trade_id in self._known:
            return False
        t, feats = entry
        for k in feats:
            if k not in self.feature_names:
                self.feature_names.append(k)
        self.x.append(feats)
        self.t.append(t)
        self.multiple.append(max(float(outcome.multiple), 0.0))
        self.trade_ids.append(outcome.trade_id)
        self._known.add(outcome.trade_id)
        self.peak.append(float(outcome.peak_multiple) if outcome.peak_multiple is not None else 0.0)
        self._since_fit += 1
        if len(self.multiple) >= self.min_trades and self._since_fit >= self.refit_every:
            self.refit()
        return True

    def refit(self) -> dict[str, Any]:
        """Champion / challenger refit on all settled trades (see the module docstring)."""
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        self._since_fit = 0
        order = np.argsort(np.asarray(self.t), kind="stable")
        x = self._matrix([self.x[i] for i in order])
        n = len(order)
        cut = int(n * 0.8)
        rep: dict[str, Any] = {"trades": n, "features": list(self.feature_names), "levels": {}}
        params = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 150, "l2_regularization": 1.0}
        models: dict[float, TreeEnsemble] = {}
        for lv in LEVELS:
            y = self._labels(lv)[order]
            pos_train = int(y[:cut].sum())
            entry: dict[str, Any] = {"positives": int(y.sum()), "deployed": False}
            if n < self.min_trades or pos_train < self.min_positives or cut - pos_train < self.min_positives:
                entry["why"] = "not enough positives and negatives yet"
                rep["levels"][str(lv)] = entry
                continue
            clf = HistGradientBoostingClassifier(random_state=self.seed, **params).fit(x[:cut], y[:cut])
            p_new = clf.predict_proba(x[cut:])[:, 1]
            base = (y[:cut].sum() + 0.5) / (cut + 1.0)
            loss_model, loss_base = _log_loss(p_new, y[cut:]), _log_loss(np.full(n - cut, base), y[cut:])
            entry |= {"holdout_log_loss": loss_model, "holdout_base_log_loss": loss_base}
            if loss_model < loss_base:
                full = HistGradientBoostingClassifier(random_state=self.seed, **params).fit(x, y)
                models[lv] = TreeEnsemble.export(full)
                entry["deployed"] = True
            else:
                entry["why"] = "did not beat the base rate on the newest trades"
            rep["levels"][str(lv)] = entry
        self.models = models
        # expected multiple: log1p target (bounded influence of rare giants), same champion rule
        y_val = np.log1p(np.asarray(self.multiple)[order])
        self.value_model = None
        if n >= self.min_trades:
            reg = HistGradientBoostingRegressor(random_state=self.seed, **params).fit(x[:cut], y_val[:cut])
            resid = y_val[cut:] - reg.predict(x[cut:])
            err_model = float(np.mean(resid**2))
            err_base = float(np.mean((y_val[:cut].mean() - y_val[cut:]) ** 2))
            rep["value"] = {
                "holdout_mse": err_model,
                "holdout_base_mse": err_base,
                "deployed": err_model < err_base,
            }
            if err_model < err_base:
                full_reg = HistGradientBoostingRegressor(random_state=self.seed, **params).fit(x, y_val)
                self.value_model = TreeEnsemble.export(full_reg)
                self.smear = float(np.mean(np.exp(resid)))
        self.report = rep
        return rep

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Write ``meta.json`` (history, pending proposals, report) and the deployed trees.

        Crash-safe: the trees of each save go to new files tagged with a generation number, then
        ``meta.json`` (which names that generation) is replaced atomically, and only then are the
        previous generation's files removed.  A crash at any point leaves a loadable directory.
        """
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        gen = self._generation + 1
        for lv, model in self.models.items():
            model.save(d / f"level_{lv:g}.g{gen}.npz")
        if self.value_model is not None:
            self.value_model.save(d / f"value.g{gen}.npz")
        meta = {
            "config": {
                "min_trades": self.min_trades,
                "refit_every": self.refit_every,
                "min_positives": self.min_positives,
                "max_size": self.max_size,
                "veto_ratio": self.veto_ratio,
                "seed": self.seed,
                "max_pending": self.max_pending,
            },
            "generation": gen,
            "value": self.value_model is not None,
            "feature_names": self.feature_names,
            "history": {
                "x": self.x,
                "t": self.t,
                "multiple": self.multiple,
                "peak": self.peak,
                "trade_ids": self.trade_ids,
            },
            "pending": {k: [t, f] for k, (t, f) in self.pending.items()},
            "levels": list(self.models),
            "smear": self.smear,
            "since_fit": self._since_fit,
            "report": self.report,
        }
        tmp = d / "meta.json.tmp"
        tmp.write_text(json.dumps(meta, default=float))
        tmp.replace(d / "meta.json")
        self._generation = gen
        keep = {f"level_{lv:g}.g{gen}.npz" for lv in self.models} | {f"value.g{gen}.npz"}
        for f in [*d.glob("level_*.npz"), *d.glob("value*.npz")]:
            if f.name not in keep:
                f.unlink()

    @classmethod
    def load(cls, directory: str | Path) -> MetaLearner:
        """Rebuild a learner saved by :meth:`save` (also the older layout without generations)."""
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        config = dict(meta["config"])
        config.pop("pending_ttl_seconds", None)  # written by an interim version; replaced by max_pending
        m = cls(**config)
        m.feature_names = list(meta["feature_names"])
        h = meta["history"]
        m.x, m.t, m.multiple, m.peak = list(h["x"]), list(h["t"]), list(h["multiple"]), list(h["peak"])
        m.trade_ids = [str(i) for i in h.get("trade_ids", [])]
        m._known = set(m.trade_ids)
        m.pending = {k: (float(v[0]), dict(v[1])) for k, v in meta["pending"].items()}
        gen = meta.get("generation")
        tag = "" if gen is None else f".g{int(gen)}"  # files of the older layout have no tag
        m._generation = 0 if gen is None else int(gen)
        m.models = {
            float(lv): TreeEnsemble.load(d / f"level_{float(lv):g}{tag}.npz") for lv in meta["levels"]
        }
        value = d / f"value{tag}.npz"
        if meta.get("value", value.exists()):
            m.value_model = TreeEnsemble.load(value)
        m._since_fit = int(meta.get("since_fit", 0))
        m.smear = float(meta.get("smear", 1.0))
        m.report = dict(meta.get("report", {}))
        return m


def advice_dict(a: TradeAdvice) -> dict[str, Any]:
    """Plain dict of an advice (for JSON transport to the trading system)."""
    return asdict(a)
