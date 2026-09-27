"""Meta-labeling edge model.

Second-stage model (López de Prado's meta-labeling) trained on the neural brain's
**out-of-fold** outputs plus the launch-risk probabilities and raw on-chain features.  It
answers the only question that matters for a trade: *after latency, impact and fees,
does this setup make money?*

* P(win) — the executable triple-barrier outcome beats zero: a bootstrap MLP ensemble,
  isotonic-calibrated on a later validation window;
* E[net] — a **stack of three diverse learners** averaged: the MLP ensemble's Huber net
  head, a ridge regression, and gradient-boosted trees (exported to plain arrays);
* σ — disagreement between the three learners.

``edge_score = E[net] − λ·σ`` is a lower confidence bound: it prefers setups all learners
agree on, the standard defence against chasing noise.  (On two independent simulated
markets the stacked LCB beat each single learner and momentum on out-of-sample t-stat.)  ``kelly`` gives a
capped fractional-Kelly fraction from P(win) and the empirical win/loss ratio.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from torch import nn

from nardis_neural.inference.calibration import BinaryCalibrator
from nardis_neural.solana.edge.trees import TreeEnsemble
from nardis_neural.training.metrics import brier_score, rank_correlation, roc_auc

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
NET_CLIP = (-1.0, 3.0)


class _EdgeNet(nn.Module):
    def __init__(self, d_in: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.win = nn.Linear(hidden, 1)
        self.net = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.win(h).squeeze(-1), self.net(h).squeeze(-1)


@dataclass
class EdgePrediction:
    """Per-row edge outputs: calibrated P(win), stacked E[net], learner disagreement σ, the
    lower-confidence ``edge_score`` and the capped fractional-Kelly fraction.
    """

    p_win: F64
    expected_net: F64
    uncertainty: F64
    edge_score: F64
    kelly: F64


class EdgeModel:
    """Meta-labeling edge model: bootstrap MLP ensemble, ridge and boosted trees on out-of-fold
    meta-features.
    """

    def __init__(
        self,
        d_in: int,
        members: int = 5,
        hidden: int = 64,
        dropout: float = 0.15,
        seed: int = 0,
        lcb_lambda: float = 1.0,
        kelly_fraction: float = 0.25,
        max_kelly: float = 0.2,
        feature_names: list[str] | None = None,
    ) -> None:
        self.d_in, self.hidden, self.dropout, self.seed = d_in, hidden, dropout, seed
        self.lcb_lambda, self.kelly_fraction, self.max_kelly = lcb_lambda, kelly_fraction, max_kelly
        self.feature_names = feature_names or [f"x{i}" for i in range(d_in)]
        self.members: list[_EdgeNet] = []
        for i in range(members):
            torch.manual_seed(seed + 211 * i)
            self.members.append(_EdgeNet(d_in, hidden, dropout))
        self.mean = np.zeros(d_in, dtype=np.float32)
        self.std = np.ones(d_in, dtype=np.float32)
        self.calibrator = BinaryCalibrator()
        self.win_loss_ratio = 1.0
        self.ridge_coef = np.zeros(d_in, dtype=np.float64)
        self.ridge_intercept = 0.0
        self.trees: TreeEnsemble | None = None
        self.report: dict[str, Any] = {}

    def _x(self, x: F32) -> torch.Tensor:
        z = (np.nan_to_num(x) - self.mean) / self.std
        return torch.from_numpy(np.clip(z, -8, 8).astype(np.float32))

    def fit(
        self,
        x: F32,
        net: F64,
        timestamps: F64,
        validation_fraction: float = 0.25,
        epochs: int = 80,
        lr: float = 1e-3,
        patience: int = 10,
        weight_decay: float = 1e-3,
    ) -> dict[str, Any]:
        """Fit on meta-features and executable net returns; returns the validation report.

        The latest ``validation_fraction`` of rows by ``timestamps`` is held out for early stopping,
        isotonic calibration of P(win) and the report.  Raises ``ValueError`` below 30 training rows.
        """
        order = np.argsort(timestamps, kind="stable")
        n_val = max(10, int(len(order) * validation_fraction))
        tr, va = order[:-n_val], order[-n_val:]
        if len(tr) < 30:
            raise ValueError("not enough rows to fit the edge model")
        self.mean = np.nan_to_num(x[tr]).mean(axis=0).astype(np.float32)
        self.std = (np.nan_to_num(x[tr]).std(axis=0) + 1e-6).astype(np.float32)
        y_net = np.clip(net, *NET_CLIP).astype(np.float32)
        y_win = (net > 0).astype(np.float32)
        wins, losses = net[tr][net[tr] > 0], -net[tr][net[tr] <= 0]
        self.win_loss_ratio = (
            float(wins.mean() / max(losses.mean(), 1e-6)) if len(wins) and len(losses) else 1.0
        )
        xt, xv = self._x(x[tr]), self._x(x[va])
        yw_t, yn_t = torch.from_numpy(y_win[tr]), torch.from_numpy(y_net[tr])
        yw_v, yn_v = torch.from_numpy(y_win[va]), torch.from_numpy(y_net[va])
        pos = float(y_win[tr].mean())
        pos_weight = torch.tensor(min(max((1 - pos) / max(pos, 1e-3), 0.2), 10.0))

        def loss_fn(net_: _EdgeNet, xb: torch.Tensor, w: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
            lw, ln = net_(xb)
            return (
                F.binary_cross_entropy_with_logits(lw, w, pos_weight=pos_weight)
                + F.huber_loss(ln, n, delta=0.2) * 5
            )

        for i, net_ in enumerate(self.members):
            gen = torch.Generator().manual_seed(self.seed + i)
            # each member sees a bootstrap resample → genuine ensemble diversity
            boot = torch.randint(0, len(xt), (len(xt),), generator=gen)
            xb_all, wb_all, nb_all = xt[boot], yw_t[boot], yn_t[boot]
            opt = torch.optim.AdamW(net_.parameters(), lr=lr, weight_decay=weight_decay)
            best, best_state, bad = np.inf, copy.deepcopy(net_.state_dict()), 0
            for _ in range(epochs):
                net_.train()
                perm = torch.randperm(len(xb_all), generator=gen)
                for s in range(0, len(perm), 256):
                    idx = perm[s : s + 256]
                    loss = loss_fn(net_, xb_all[idx], wb_all[idx], nb_all[idx])
                    opt.zero_grad()
                    torch.autograd.backward(loss)
                    opt.step()
                net_.eval()
                with torch.no_grad():
                    v = float(loss_fn(net_, xv, yw_v, yn_v))
                if v < best - 1e-4:
                    best, best_state, bad = v, copy.deepcopy(net_.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            net_.load_state_dict(best_state)
        from sklearn.linear_model import Ridge

        ridge = Ridge(alpha=30.0).fit(self._x(x[tr]).numpy(), y_net[tr])
        self.ridge_coef = np.asarray(ridge.coef_, dtype=np.float64)
        self.ridge_intercept = float(ridge.intercept_)
        self.trees = TreeEnsemble.fit(np.nan_to_num(x[tr]), y_net[tr].astype(np.float64), seed=self.seed)
        raw_p, _, _ = self._raw(x[va])
        mu = self.predict_components(x[va]).mean(axis=0)
        self.calibrator = BinaryCalibrator().fit(
            raw_p, y_win[va].astype(np.float64), "isotonic", min_samples=30
        )
        p = self.calibrator.transform(raw_p)
        top = mu >= np.quantile(mu, 0.9)
        self.report = {
            "n_train": len(tr),
            "n_validation": len(va),
            "win_auc": roc_auc(p, y_win[va]),
            "win_brier": brier_score(p, y_win[va]),
            "base_win_rate": float(y_win[va].mean()),
            "net_ic": rank_correlation(mu, net[va]),
            "top_decile_realized_net": float(net[va][top].mean()) if top.any() else float("nan"),
            "component_ic": {
                name: rank_correlation(c, net[va])
                for name, c in zip(("mlp", "ridge", "trees"), self.predict_components(x[va]), strict=True)
            },
            "all_realized_net": float(net[va].mean()),
            "win_loss_ratio": self.win_loss_ratio,
        }
        return self.report

    @torch.no_grad()
    def _raw(self, x: F32) -> tuple[F64, F64, F64]:
        xt = self._x(x)
        outs = [m.eval()(xt) for m in self.members]
        p = torch.stack([torch.sigmoid(w) for w, _ in outs]).mean(0).numpy().astype(np.float64)
        nets = torch.stack([n for _, n in outs]).numpy().astype(np.float64)
        return p, nets.mean(0), nets.std(0)

    def predict_components(self, x: F32) -> F64:
        """(3, N) expected-net predictions of the MLP ensemble, ridge and trees."""
        _, mlp, _ = self._raw(x)
        ridge = self._x(x).numpy().astype(np.float64) @ self.ridge_coef + self.ridge_intercept
        trees = self.trees.predict(np.nan_to_num(x)) if self.trees is not None else mlp
        return np.stack([mlp, ridge, trees])

    def predict(self, x: F32) -> EdgePrediction:
        """Calibrated P(win), stacked E[net], disagreement σ, ``edge_score = E[net] − λσ``, capped Kelly."""
        raw_p, _, _ = self._raw(x)
        comps = self.predict_components(x)
        mu, sd = comps.mean(axis=0), comps.std(axis=0)
        p = self.calibrator.transform(raw_p)
        b = max(self.win_loss_ratio, 1e-6)
        kelly = np.clip((p - (1 - p) / b) * self.kelly_fraction, 0.0, self.max_kelly)
        return EdgePrediction(p, mu, sd, mu - self.lcb_lambda * sd, kelly)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Write members, scaler, trees and ``edge.json`` metadata to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({f"m{i}": m.state_dict() for i, m in enumerate(self.members)}, d / "members.pt")
        np.savez(d / "scaler.npz", mean=self.mean, std=self.std, ridge_coef=self.ridge_coef)
        if self.trees is not None:
            self.trees.save(d / "trees.npz")
        meta = {
            "d_in": self.d_in,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "seed": self.seed,
            "members": len(self.members),
            "lcb_lambda": self.lcb_lambda,
            "kelly_fraction": self.kelly_fraction,
            "max_kelly": self.max_kelly,
            "feature_names": self.feature_names,
            "win_loss_ratio": self.win_loss_ratio,
            "ridge_intercept": self.ridge_intercept,
            "calibrator": self.calibrator.to_dict(),
            "report": self.report,
        }
        (d / "edge.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, directory: str | Path) -> EdgeModel:
        """Load a model written by :meth:`save`."""
        d = Path(directory)
        meta = json.loads((d / "edge.json").read_text())
        model = cls(
            int(meta["d_in"]),
            int(meta["members"]),
            int(meta["hidden"]),
            float(meta["dropout"]),
            int(meta["seed"]),
            float(meta["lcb_lambda"]),
            float(meta["kelly_fraction"]),
            float(meta["max_kelly"]),
            list(meta["feature_names"]),
        )
        states = torch.load(d / "members.pt", map_location="cpu", weights_only=True)
        for i, m in enumerate(model.members):
            m.load_state_dict(states[f"m{i}"])
        with np.load(d / "scaler.npz") as z:
            model.mean, model.std, model.ridge_coef = z["mean"], z["std"], z["ridge_coef"]
        model.ridge_intercept = float(meta["ridge_intercept"])
        if (d / "trees.npz").exists():
            model.trees = TreeEnsemble.load(d / "trees.npz")
        model.win_loss_ratio = float(meta["win_loss_ratio"])
        model.calibrator = BinaryCalibrator.from_dict(meta["calibrator"])
        model.report = dict(meta.get("report", {}))
        return model
