"""Conditional power-law tail model of a ticket's peak multiple.

The quantity that decides a moonshot is not the expected return but the *shape of the
right tail*: how likely is 10x, 100x, 1000x?  Returns of new tokens are fat-tailed, so the
model works on ``y = log M`` with a **mixture of logistic distributions**

    P(M ≥ k | x) = Σ_j π_j(x) · σ((μ_j(x) − log k) / s_j(x))

Each component is a log-logistic in ``M`` whose survival decays like a power law,
``P(M ≥ k) ∝ k^(−1/s_j)``, so the network learns *per token* both where the bulk sits and
how heavy the tail is (a "dud" component, a "normal pump" component and a "runner"
component emerge on their own).

* **Censoring**: rows whose horizon ran past the data contribute ``log P(M ≥ observed)``
  instead of a density, so the model can be trained at any cutoff without look-ahead.
* **Token weighting**: snapshots of the same token are strongly correlated, so every token
  gets the same total weight and bootstrap resampling is done by token.
* **Deep ensemble**: members are trained on token-bootstrap resamples; the predictive
  survival is the members' average and their spread is the epistemic uncertainty.
* **Decision helpers**: expected ladder payoff by quadrature over the predicted
  distribution, and a *lottery Kelly* fraction that maximises expected log-wealth, the
  sizing rule that survives a strategy whose typical ticket goes to zero.
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

from nardis_neural.solana.moonshot.labels import MIN_MULTIPLE, MoonshotSpec, ladder_payoff

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
GRID = np.exp(np.linspace(np.log(MIN_MULTIPLE), np.log(1e5), 361))
KELLY_GRID = np.concatenate([[0.0], np.geomspace(1e-4, 0.5, 60)])


class _TailNet(nn.Module):
    def __init__(self, d_in: int, hidden: int, components: int, dropout: float) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden, 3 * components)
        with torch.no_grad():  # start components spread over the plausible range of log M
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()
            self.head.bias[components : 2 * components] = torch.linspace(-1.0, 2.5, components)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logit, mu, raw_s = self.head(self.body(x)).chunk(3, dim=-1)
        return F.log_softmax(logit, dim=-1), mu, F.softplus(raw_s) + 0.03


def censored_nll(
    log_pi: torch.Tensor, mu: torch.Tensor, s: torch.Tensor, y: torch.Tensor, censored: torch.Tensor
) -> torch.Tensor:
    """Per-row negative log-likelihood of a logistic mixture on ``y`` with right-censoring."""
    z = (y.unsqueeze(-1) - mu) / s
    log_pdf = -z - torch.log(s) - 2 * F.softplus(-z)
    log_sf = -F.softplus(z)  # log σ(−z)
    comp = torch.where(censored.unsqueeze(-1), log_sf, log_pdf)
    return -torch.logsumexp(log_pi + comp, dim=-1)


@dataclass
class TailPrediction:
    survival: F64
    """(N, len(levels)) ensemble-mean P(M ≥ k)."""
    survival_std: F64
    """(N, len(levels)) spread of P(M ≥ k) across members (epistemic)."""
    median_multiple: F64
    expected_multiple: F64
    """E[ladder payoff] per SOL under the predicted distribution."""
    lottery_kelly: F64
    tail_index: F64
    """Power-law exponent of the predicted far tail (smaller = heavier)."""


class TailModel:
    def __init__(
        self,
        d_in: int,
        spec: MoonshotSpec | None = None,
        members: int = 5,
        components: int = 3,
        hidden: int = 64,
        dropout: float = 0.1,
        seed: int = 0,
        kelly_multiplier: float = 0.25,
        max_kelly: float = 0.05,
        feature_names: list[str] | None = None,
        inputs: str = "raw",
    ) -> None:
        self.inputs = inputs
        """``raw`` (on-chain features only) or ``neural`` (the edge meta-feature stack)."""
        self.d_in, self.components, self.hidden, self.dropout = d_in, components, hidden, dropout
        self.spec = spec or MoonshotSpec()
        self.seed, self.kelly_multiplier, self.max_kelly = seed, kelly_multiplier, max_kelly
        self.feature_names = feature_names or [f"x{i}" for i in range(d_in)]
        self.members: list[_TailNet] = []
        for i in range(members):
            torch.manual_seed(seed + 131 * i)
            self.members.append(_TailNet(d_in, hidden, components, dropout))
        self.mean = np.zeros(d_in, dtype=np.float32)
        self.std = np.ones(d_in, dtype=np.float32)
        self.report: dict[str, Any] = {}

    def _x(self, x: npt.NDArray[Any]) -> torch.Tensor:
        z = (np.nan_to_num(np.asarray(x, dtype=np.float32)) - self.mean) / self.std
        return torch.from_numpy(np.clip(z, -8, 8).astype(np.float32))

    # ------------------------------------------------------------------ training
    def fit(
        self,
        x: npt.NDArray[Any],
        peak: F64,
        censored: npt.NDArray[np.bool_],
        timestamps: F64,
        groups: npt.NDArray[Any],
        validation_fraction: float = 0.25,
        epochs: int = 150,
        lr: float = 2e-3,
        patience: int = 15,
        weight_decay: float = 1e-3,
    ) -> dict[str, Any]:
        """Fit on rows ordered in time; the latest groups (tokens) are held out for early stopping."""
        groups = np.asarray(groups)
        first_seen: dict[Any, float] = {}
        for g, t in zip(groups, timestamps, strict=True):
            first_seen[g] = min(first_seen.get(g, np.inf), float(t))
        ordered = sorted(first_seen, key=lambda g: first_seen[g])
        n_val = max(1, int(len(ordered) * validation_fraction)) if len(ordered) > 3 else 0
        val_groups = set(ordered[len(ordered) - n_val :])
        va = np.flatnonzero([g in val_groups for g in groups])
        tr = np.flatnonzero([g not in val_groups for g in groups])
        if len(tr) < 20:
            raise ValueError("not enough rows to fit the tail model")
        xs = np.nan_to_num(np.asarray(x, dtype=np.float32)[tr])
        self.mean = xs.mean(axis=0).astype(np.float32)
        self.std = (xs.std(axis=0) + 1e-6).astype(np.float32)
        y = np.log(np.maximum(peak, MIN_MULTIPLE)).astype(np.float32)
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        w = (1.0 / counts[inv]).astype(np.float32)  # every token carries the same total weight
        xt, yt, wt = self._x(x), torch.from_numpy(y), torch.from_numpy(w)
        ct = torch.from_numpy(np.asarray(censored, dtype=bool))
        tr_groups = np.unique(inv[tr])

        def loss_on(net: _TailNet, idx: torch.Tensor) -> torch.Tensor:
            log_pi, mu, s = net(xt[idx])
            nll = censored_nll(log_pi, mu, s, yt[idx], ct[idx])
            return (nll * wt[idx]).sum() / wt[idx].sum()

        val_idx = torch.from_numpy(va)
        for i, net in enumerate(self.members):
            gen = torch.Generator().manual_seed(self.seed + i)
            rng = np.random.default_rng(self.seed + i)
            boot_groups = rng.choice(tr_groups, size=len(tr_groups), replace=True)
            by_group = {g: tr[inv[tr] == g] for g in np.unique(boot_groups)}
            boot = torch.from_numpy(np.concatenate([by_group[g] for g in boot_groups]))
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
            best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
            for _ in range(epochs):
                net.train()
                perm = boot[torch.randperm(len(boot), generator=gen)]
                for s in range(0, len(perm), 256):
                    loss = loss_on(net, perm[s : s + 256])
                    opt.zero_grad()
                    torch.autograd.backward(loss)
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                if len(va) == 0:
                    continue
                net.eval()
                with torch.no_grad():
                    v = float(loss_on(net, val_idx))
                if v < best - 1e-4:
                    best, best_state, bad = v, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
            if len(va):
                net.load_state_dict(best_state)
        self.report = {"n_train": len(tr), "n_validation": len(va), "tokens": len(ordered)}
        if len(va):
            with torch.no_grad():
                nll = [float(loss_on(m.eval(), val_idx)) for m in self.members]
            self.report["validation_nll"] = float(np.mean(nll))
        return self.report

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _params(self, x: npt.NDArray[Any]) -> tuple[F64, F64, F64]:
        xt = self._x(x)
        outs = [m.eval()(xt) for m in self.members]
        pi = torch.stack([o[0].exp() for o in outs]).double().numpy()  # (E, N, K)
        mu = torch.stack([o[1] for o in outs]).double().numpy()
        s = torch.stack([o[2] for o in outs]).double().numpy()
        return pi, mu, s

    @staticmethod
    def _survival(pi: F64, mu: F64, s: F64, k: F64) -> F64:
        """(E, N, len(k)) member survival functions at multiples ``k``."""
        z = (mu[..., None, :] - np.log(k)[:, None]) / s[..., None, :]
        sig = 0.5 * (1 + np.tanh(0.5 * z))
        return np.asarray((pi[..., None, :] * sig).sum(-1), dtype=np.float64)

    def survival(self, x: npt.NDArray[Any], k: list[float] | F64) -> F64:
        pi, mu, s = self._params(x)
        return np.asarray(self._survival(pi, mu, s, np.asarray(k, dtype=np.float64)).mean(0))

    def nll(self, x: npt.NDArray[Any], peak: F64, censored: npt.NDArray[np.bool_]) -> F64:
        """Per-row NLL of the ensemble mixture (members averaged in probability space)."""
        y = np.log(np.maximum(peak, MIN_MULTIPLE))
        pi, mu, s = self._params(x)
        z = (y[None, :, None] - mu) / s
        pdf = (pi * np.exp(-z - np.log(s) - 2 * np.logaddexp(0, -z))).sum(-1).mean(0)
        sf = (pi * 0.5 * (1 + np.tanh(-0.5 * z))).sum(-1).mean(0)
        return np.asarray(-np.log(np.maximum(np.where(censored, sf, pdf), 1e-300)))

    def predict(self, x: npt.NDArray[Any]) -> TailPrediction:
        pi, mu, s = self._params(x)
        levels = np.asarray(self.spec.levels, dtype=np.float64)
        member_sf = self._survival(pi, mu, s, levels)
        grid_sf = self._survival(pi, mu, s, GRID).mean(0)  # (N, G)
        mass = np.concatenate(
            [1 - grid_sf[:, :1], grid_sf[:, :-1] - grid_sf[:, 1:], grid_sf[:, -1:]], axis=1
        ).clip(0, None)
        mid = np.concatenate([[GRID[0]], np.sqrt(GRID[:-1] * GRID[1:]), [GRID[-1]]])
        pay = ladder_payoff(mid, self.spec)  # (G + 1,)
        expected = mass @ pay
        median = np.array([np.interp(-0.5, -row, GRID) for row in grid_sf])
        # lottery Kelly: fraction of bankroll maximising E[log(1 + f (payoff − 1))]
        growth = np.log1p(KELLY_GRID[:, None] * (pay[None, :] - 1)) @ mass.T  # (F, N)
        best = KELLY_GRID[np.argmax(growth, axis=0)]
        kelly = np.clip(best * self.kelly_multiplier, 0.0, self.max_kelly)
        # far-tail exponent: the heaviest component with non-negligible weight dominates
        w_pi = pi.mean(0)
        s_eff = np.where(w_pi > 0.02, s.mean(0), 0.0).max(axis=1)
        return TailPrediction(
            survival=member_sf.mean(0),
            survival_std=member_sf.std(0),
            median_multiple=median,
            expected_multiple=expected,
            lottery_kelly=kelly,
            tail_index=1.0 / np.maximum(s_eff, 1e-3),
        )

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({f"m{i}": m.state_dict() for i, m in enumerate(self.members)}, d / "tail.pt")
        np.savez(d / "tail_scaler.npz", mean=self.mean, std=self.std)
        meta = {
            "d_in": self.d_in,
            "members": len(self.members),
            "components": self.components,
            "hidden": self.hidden,
            "dropout": self.dropout,
            "seed": self.seed,
            "kelly_multiplier": self.kelly_multiplier,
            "max_kelly": self.max_kelly,
            "feature_names": self.feature_names,
            "inputs": self.inputs,
            "spec": self.spec.model_dump(),
            "report": self.report,
        }
        (d / "tail.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, directory: str | Path) -> TailModel:
        d = Path(directory)
        meta = json.loads((d / "tail.json").read_text())
        model = cls(
            int(meta["d_in"]),
            MoonshotSpec.model_validate(meta["spec"]),
            members=int(meta["members"]),
            components=int(meta["components"]),
            hidden=int(meta["hidden"]),
            dropout=float(meta["dropout"]),
            seed=int(meta["seed"]),
            kelly_multiplier=float(meta["kelly_multiplier"]),
            max_kelly=float(meta["max_kelly"]),
            feature_names=list(meta["feature_names"]),
            inputs=str(meta.get("inputs", "raw")),
        )
        states = torch.load(d / "tail.pt", map_location="cpu", weights_only=True)
        for i, m in enumerate(model.members):
            m.load_state_dict(states[f"m{i}"])
        with np.load(d / "tail_scaler.npz") as z:
            model.mean, model.std = z["mean"], z["std"]
        model.report = dict(meta.get("report", {}))
        return model
