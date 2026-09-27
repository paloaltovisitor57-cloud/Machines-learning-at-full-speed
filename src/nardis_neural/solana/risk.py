"""Solana launch-risk model: P(rug), P(graduation), P(dev dump) within the risk horizon.

Inputs are the neural engine's MarketStateEmbedding concatenated with the raw named
Solana features, so the model can use both learned market context and explicit
on-chain red flags.  It is a small deep ensemble of multi-label MLPs:

* independently initialised members → epistemic uncertainty per label,
* ``pos_weight`` class balancing (rugs and graduations are rare),
* chronological train/validation split, early stopping,
* per-label temperature calibration fitted on the validation split only.
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
from nardis_neural.solana.config import RISK_LABELS
from nardis_neural.training.metrics import brier_score, roc_auc

F32 = npt.NDArray[np.float32]


class _RiskNet(nn.Module):
    def __init__(self, d_in: int, hidden: int, n_out: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


@dataclass
class RiskFitReport:
    n_train: int
    n_validation: int
    metrics: dict[str, dict[str, float]]


class SolanaRiskModel:
    def __init__(
        self, d_in: int, members: int = 3, hidden: int = 64, dropout: float = 0.1, seed: int = 0
    ) -> None:
        self.d_in, self.hidden, self.dropout, self.seed = d_in, hidden, dropout, seed
        self.labels = RISK_LABELS
        self.members: list[_RiskNet] = []
        for i in range(members):
            torch.manual_seed(seed + 101 * i)
            self.members.append(_RiskNet(d_in, hidden, len(self.labels), dropout))
        self.mean = np.zeros(d_in, dtype=np.float32)
        self.std = np.ones(d_in, dtype=np.float32)
        self.calibrators = [BinaryCalibrator() for _ in self.labels]
        self.report: dict[str, Any] = {}

    @staticmethod
    def inputs(embedding: npt.NDArray[Any], current: npt.NDArray[Any]) -> F32:
        return np.asarray(np.nan_to_num(np.concatenate([embedding, current], axis=1)), dtype=np.float32)

    def _x(self, x: F32) -> torch.Tensor:
        return torch.from_numpy(np.clip((x - self.mean) / self.std, -8, 8).astype(np.float32))

    def fit(
        self,
        x: F32,
        y: F32,
        timestamps: npt.NDArray[np.float64],
        epochs: int = 60,
        lr: float = 2e-3,
        validation_fraction: float = 0.25,
        patience: int = 8,
        groups: npt.NDArray[Any] | None = None,
    ) -> RiskFitReport:
        """``groups`` (e.g. mint per row) makes validation *token-disjoint*: the latest-launched
        tokens are held out and training uses only earlier-launched tokens' rows observed
        before the validation period starts — no token contributes to both sides."""
        ok = ~np.isnan(y).any(axis=1)
        x, y, ts = x[ok], y[ok], timestamps[ok]
        if groups is not None:
            g = np.asarray(groups)[ok]
            names = np.unique(g)
            first = np.array([ts[g == n].min() for n in names])
            held = names[np.argsort(first)][-max(1, round(len(names) * validation_fraction)) :]
            is_val = np.isin(g, held)
            va = np.flatnonzero(is_val)
            tr = np.flatnonzero(~is_val & (ts < ts[va].min()))
        else:
            order = np.argsort(ts, kind="stable")
            n_val = max(1, int(len(order) * validation_fraction))
            tr, va = order[:-n_val], order[-n_val:]
        if len(tr) < 10:
            raise ValueError("not enough labelled risk samples to train")
        self.mean = x[tr].mean(axis=0)
        self.std = x[tr].std(axis=0) + 1e-6
        xt, yt = self._x(x[tr]), torch.from_numpy(y[tr])
        xv, yv = self._x(x[va]), torch.from_numpy(y[va])
        pos = yt.mean(dim=0).clamp(1e-3, 1 - 1e-3)
        pos_weight = ((1 - pos) / pos).clamp(max=20.0)
        for i, net in enumerate(self.members):
            gen = torch.Generator().manual_seed(self.seed + i)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
            best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
            for _ in range(epochs):
                net.train()
                perm = torch.randperm(len(xt), generator=gen)
                for start in range(0, len(xt), 256):
                    idx = perm[start : start + 256]
                    loss = F.binary_cross_entropy_with_logits(net(xt[idx]), yt[idx], pos_weight=pos_weight)
                    opt.zero_grad()
                    torch.autograd.backward(loss)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    v = float(F.binary_cross_entropy_with_logits(net(xv), yv, pos_weight=pos_weight))
                if v < best - 1e-4:
                    best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            net.load_state_dict(best_state)
        raw = self._raw_probs(x[va])
        metrics: dict[str, dict[str, float]] = {}
        for j, name in enumerate(self.labels):
            self.calibrators[j] = BinaryCalibrator().fit(raw[:, j], y[va, j], "temperature", min_samples=30)
            p = self.calibrators[j].transform(raw[:, j])
            metrics[name] = {
                "auc": roc_auc(p, y[va, j]),
                "brier": brier_score(p, y[va, j]),
                "base_rate": float(y[va, j].mean()),
                "train_base_rate": float(y[tr, j].mean()),
            }
        self.report = {"n_train": len(tr), "n_validation": len(va), "metrics": metrics}
        return RiskFitReport(len(tr), len(va), metrics)

    @torch.no_grad()
    def _member_probs(self, x: F32) -> npt.NDArray[np.float64]:
        xt = self._x(x)
        return np.stack([torch.sigmoid(m.eval()(xt)).numpy() for m in self.members]).astype(np.float64)

    def _raw_probs(self, x: F32) -> npt.NDArray[np.float64]:
        return np.asarray(self._member_probs(x).mean(axis=0), dtype=np.float64)

    def predict(self, x: F32) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Calibrated probabilities (N, L) and member disagreement (std, N, L)."""
        members = self._member_probs(x)
        raw = members.mean(axis=0)
        cal = np.stack([c.transform(raw[:, j]) for j, c in enumerate(self.calibrators)], axis=1)
        return cal, members.std(axis=0)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({f"m{i}": m.state_dict() for i, m in enumerate(self.members)}, d / "members.pt")
        np.savez(d / "scaler.npz", mean=self.mean, std=self.std)
        meta = {
            "d_in": self.d_in,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "seed": self.seed,
            "members": len(self.members),
            "labels": list(self.labels),
            "calibrators": [c.to_dict() for c in self.calibrators],
            "report": self.report,
        }
        (d / "risk.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, directory: str | Path) -> SolanaRiskModel:
        d = Path(directory)
        meta = json.loads((d / "risk.json").read_text())
        model = cls(
            int(meta["d_in"]),
            int(meta["members"]),
            int(meta["hidden"]),
            float(meta["dropout"]),
            int(meta["seed"]),
        )
        states = torch.load(d / "members.pt", map_location="cpu", weights_only=True)
        for i, m in enumerate(model.members):
            m.load_state_dict(states[f"m{i}"])
        with np.load(d / "scaler.npz") as z:
            model.mean, model.std = z["mean"], z["std"]
        model.calibrators = [BinaryCalibrator.from_dict(c) for c in meta["calibrators"]]
        model.report = dict(meta.get("report", {}))
        return model
