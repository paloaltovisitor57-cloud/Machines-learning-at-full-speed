"""Tape Transformer: reads the raw trade tape and predicts the tail and the collapse.

Architecture (per ensemble member, ~2.4 M parameters with the default hash space, most of
them in the wallet table; runs on a laptop CPU, an M1 or a small cloud VM):

* each trade → ``Linear(trade features) + Linear(wallet embedding[hash bucket])`` plus a
  learned **recency** embedding (position counted from the most recent trade, so left
  padding changes nothing);
* a learned **summary token** is appended and a pre-LayerNorm Transformer encoder attends
  over the tape (padding masked; the summary token keeps empty tapes well defined);
* the summary token, the masked mean of the trades and an MLP of the current features
  are fused into one vector;
* two heads:

  - **tail**: a mixture of logistics on ``log`` peak multiple, trained with the censored
    likelihood of :mod:`moonshot.tail` (same decision maths: calibrated P(≥ k), expected
    ladder payoff, lottery Kelly);
  - **collapse**: a discrete-time hazard over ``collapse_bins`` (1 min, 5 min, 15 min, 1 h
    by default): the probability that the ticket's value halves within each window,
    trained with the censored survival likelihood.  This is the exit signal.

Members are trained on token-bootstrap resamples with early stopping on the most recent
tokens; every token carries the same total weight.
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

from nardis_neural.solana.moonshot.labels import MIN_MULTIPLE, MoonshotSpec
from nardis_neural.solana.moonshot.tail import (
    TailPrediction,
    censored_nll,
    fit_calibration_ratio,
    mixture_nll,
    mixture_survival,
    summarize_tail,
)
from nardis_neural.solana.tape.features import TRADE_FEATURES, TapeSpec

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
B = npt.NDArray[np.bool_]
DEFAULT_BINS = (60.0, 300.0, 900.0, 3600.0)


def window_label(seconds: float) -> str:
    """Compact name of a time window: 60 → ``1m``, 3600 → ``1h``, 45 → ``45s``."""
    if seconds % 3600 == 0:
        return f"{int(seconds // 3600)}h"
    if seconds % 60 == 0:
        return f"{int(seconds // 60)}m"
    return f"{seconds:g}s"


class TapeNet(nn.Module):
    """One ensemble member: trade/wallet embeddings, Transformer over the tape, two heads."""

    def __init__(
        self,
        n_trade: int,
        n_current: int,
        buckets: int,
        max_trades: int,
        d: int = 64,
        wallet_dim: int = 16,
        heads: int = 4,
        layers: int = 2,
        components: int = 3,
        n_bins: int = 4,
        dropout: float = 0.1,
        wallet_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.wallet_dropout = wallet_dropout
        self.trade = nn.Linear(n_trade, d)
        self.wallet = nn.Embedding(buckets, wallet_dim, padding_idx=0)
        self.wallet_proj = nn.Linear(wallet_dim, d, bias=False)
        self.recency = nn.Parameter(torch.zeros(max_trades + 1, d))
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.normal_(self.recency, std=0.02)
        nn.init.normal_(self.summary, std=0.02)
        # unit-scale identities: a small init leaves wallets drowned out by the trade features
        nn.init.normal_(self.wallet.weight, std=1.0)
        with torch.no_grad():
            self.wallet.weight[0].zero_()
        layer = nn.TransformerEncoderLayer(
            d, heads, 2 * d, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.current = nn.Sequential(nn.Linear(n_current, d), nn.GELU(), nn.Linear(d, d))
        self.fuse = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, d), nn.GELU(), nn.Dropout(dropout))
        self.tail = nn.Linear(d, 3 * components)
        self.hazard = nn.Linear(d, n_bins)
        with torch.no_grad():  # tail components start spread over plausible log multiples
            self.tail.weight.mul_(0.1)
            self.tail.bias.zero_()
            self.tail.bias[components : 2 * components] = torch.linspace(-1.0, 2.5, components)
            self.hazard.bias.fill_(-2.0)

    def forward(
        self, x: torch.Tensor, wallets: torch.Tensor, mask: torch.Tensor, current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log π, μ, s) of the tail mixture and the collapse hazard logits for a batch."""
        b, t, _ = x.shape
        if self.training and self.wallet_dropout > 0:  # hide identities so no single wallet is a crutch
            wallets = wallets.masked_fill(
                torch.rand(wallets.shape, device=wallets.device) < self.wallet_dropout, 0
            )
        h = self.trade(x) + self.wallet_proj(self.wallet(wallets))
        h = torch.cat([h, self.summary.expand(b, 1, -1)], dim=1) + self.recency[-(t + 1) :]
        ignore = torch.cat([~mask, torch.zeros(b, 1, dtype=torch.bool, device=mask.device)], dim=1)
        h = self.norm(self.encoder(h, src_key_padding_mask=ignore))
        m = mask.unsqueeze(-1).to(h.dtype)
        mean = (h[:, :-1] * m).sum(1) / m.sum(1).clamp(min=1.0)
        z = self.fuse(torch.cat([h[:, -1], mean, self.current(current)], dim=-1))
        logit, mu, raw_s = self.tail(z).chunk(3, dim=-1)
        return F.log_softmax(logit, dim=-1), mu, F.softplus(raw_s) + 0.03, self.hazard(z)


def hazard_targets(collapse_time: F64, observed: F64, bins: tuple[float, ...]) -> tuple[F32, F32]:
    """Discrete-time survival targets: (event one-hot, survived-bin mask), each (N, len(bins)).

    A row that collapsed in bin ``b`` survived every earlier bin and has its event in ``b``;
    a row without a collapse survived every bin it was fully observed through.
    """
    edges = np.asarray(bins, dtype=np.float64)
    n, j = len(collapse_time), len(edges)
    event = np.zeros((n, j), dtype=np.float32)
    survive = np.zeros((n, j), dtype=np.float32)
    has = np.isfinite(collapse_time) & (collapse_time <= edges[-1])
    b = np.searchsorted(edges, np.where(has, collapse_time, 0.0), side="left")
    idx = np.arange(j)
    event[has, b[has]] = 1.0
    survive[has] = (idx[None, :] < b[has, None]).astype(np.float32)
    survive[~has] = (edges[None, :] <= np.maximum(observed[~has], 0.0)[:, None]).astype(np.float32)
    # a collapse later than the last edge still means every bin was survived
    late = np.isfinite(collapse_time) & (collapse_time > edges[-1])
    survive[late] = 1.0
    return event, survive


def hazard_nll(logits: torch.Tensor, event: torch.Tensor, survive: torch.Tensor) -> torch.Tensor:
    """Per-row negative log-likelihood of the discrete-time hazard model."""
    log_surv = -F.softplus(logits)  # log(1 − h)
    log_event = -F.softplus(-logits)  # log h
    return -(survive * log_surv + event * log_event).sum(-1)


@dataclass
class TapePrediction:
    """Tail view (same fields as the tail model) plus the collapse curve."""

    tail: TailPrediction
    collapse: F64
    """(N, len(bins)) P(value halves within each window), ensemble mean."""
    collapse_std: F64


class TapeModel:
    """Deep ensemble of :class:`TapeNet` with input scaling, tail calibration and persistence."""

    def __init__(
        self,
        n_current: int,
        spec: MoonshotSpec | None = None,
        tape: TapeSpec | None = None,
        members: int = 3,
        d: int = 64,
        layers: int = 2,
        collapse_bins: tuple[float, ...] = DEFAULT_BINS,
        seed: int = 0,
        kelly_multiplier: float = 0.25,
        max_kelly: float = 0.05,
        feature_names: list[str] | None = None,
        min_wallet_tokens: int = 3,
    ) -> None:
        self.n_current, self.d, self.layers, self.seed = n_current, d, layers, seed
        self.min_wallet_tokens = min_wallet_tokens
        """A wallet gets its own embedding only after trading this many distinct training tokens;
        rarer wallets share the 'unknown wallet' slot (their trades still count through features)."""
        self.spec = spec or MoonshotSpec()
        self.tape = tape or TapeSpec()
        self.collapse_bins = tuple(float(b) for b in collapse_bins)
        self.kelly_multiplier, self.max_kelly = kelly_multiplier, max_kelly
        self.feature_names = feature_names or [f"x{i}" for i in range(n_current)]
        self.members: list[TapeNet] = []
        for i in range(members):
            torch.manual_seed(seed + 101 * i)
            self.members.append(self._net())
        n_trade = len(TRADE_FEATURES)
        self.trade_mean = np.zeros(n_trade, dtype=np.float32)
        self.trade_std = np.ones(n_trade, dtype=np.float32)
        self.cur_mean = np.zeros(n_current, dtype=np.float32)
        self.cur_std = np.ones(n_current, dtype=np.float32)
        self.cur_lo = np.full(n_current, -np.inf, dtype=np.float32)
        self.cur_hi = np.full(n_current, np.inf, dtype=np.float32)
        self.calibration = np.ones(len(self.spec.levels), dtype=np.float64)
        self.known = np.ones(self.tape.wallet_buckets, dtype=bool)
        """Per hash bucket: True when the wallet was seen often enough to have an embedding."""
        self.report: dict[str, Any] = {}

    def _net(self) -> TapeNet:
        return TapeNet(
            len(TRADE_FEATURES),
            self.n_current,
            self.tape.wallet_buckets,
            self.tape.max_trades,
            d=self.d,
            layers=self.layers,
            n_bins=len(self.collapse_bins),
        )

    @property
    def parameters_per_member(self) -> int:
        """Trainable parameters of one ensemble member."""
        return sum(p.numel() for p in self.members[0].parameters())

    # ------------------------------------------------------------------ inputs
    def _inputs(self, tx: F32, tw: I64, tm: B, cur: npt.NDArray[Any]) -> tuple[torch.Tensor, ...]:
        x = np.nan_to_num((np.asarray(tx, dtype=np.float32) - self.trade_mean) / self.trade_std)
        x = np.clip(x, -8, 8) * np.asarray(tm, dtype=np.float32)[..., None]
        c = np.clip(np.nan_to_num(np.asarray(cur, dtype=np.float32)), self.cur_lo, self.cur_hi)
        c = np.clip((c - self.cur_mean) / self.cur_std, -8, 8)
        wb = np.asarray(tw, dtype=np.int64) % self.tape.wallet_buckets
        wb = np.where(self.known[wb], wb, 0)  # unseen or rare wallets → the shared unknown slot
        return (
            torch.from_numpy(x.astype(np.float32)),
            torch.from_numpy(wb),
            torch.from_numpy(np.asarray(tm, dtype=bool)),
            torch.from_numpy(c.astype(np.float32)),
        )

    # ------------------------------------------------------------------ training
    def fit(
        self,
        tx: F32,
        tw: I64,
        tm: B,
        cur: npt.NDArray[Any],
        peak: F64,
        censored: B,
        collapse_time: F64,
        observed: F64,
        timestamps: F64,
        groups: npt.NDArray[Any],
        validation_fraction: float = 0.25,
        epochs: int = 40,
        lr: float = 1e-3,
        patience: int = 6,
        batch_size: int = 256,
        weight_decay: float = 1e-2,
    ) -> dict[str, Any]:
        """Fit every member; the latest ``validation_fraction`` of tokens drive early stopping and
        the tail calibration."""
        groups = np.asarray(groups)
        first: dict[Any, float] = {}
        for g, t in zip(groups.tolist(), np.asarray(timestamps).tolist(), strict=True):
            first[g] = min(first.get(g, np.inf), t)
        ordered = sorted(first, key=first.__getitem__)
        n_val = max(1, int(len(ordered) * validation_fraction)) if len(ordered) > 3 else 0
        val_groups = set(ordered[len(ordered) - n_val :])
        is_val = np.array([g in val_groups for g in groups.tolist()])
        tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
        if len(tr) < 20:
            raise ValueError("not enough rows to fit the tape model")
        valid_trades = np.asarray(tx)[tr][np.asarray(tm)[tr]]
        if len(valid_trades):
            self.trade_mean = valid_trades.mean(axis=0).astype(np.float32)
            self.trade_std = (valid_trades.std(axis=0) + 1e-6).astype(np.float32)
        seen: dict[int, set[Any]] = {}
        for row in tr:
            for b in np.unique(np.asarray(tw)[row][np.asarray(tm)[row]]).tolist():
                seen.setdefault(b, set()).add(groups[row])
        self.known = np.zeros(self.tape.wallet_buckets, dtype=bool)
        for b, gs in seen.items():
            self.known[b % self.tape.wallet_buckets] = len(gs) >= self.min_wallet_tokens
        self.known[0] = True
        c = np.nan_to_num(np.asarray(cur, dtype=np.float32)[tr])
        self.cur_lo = np.quantile(c, 0.001, axis=0).astype(np.float32)
        self.cur_hi = np.quantile(c, 0.999, axis=0).astype(np.float32)
        self.cur_mean = c.mean(axis=0).astype(np.float32)
        self.cur_std = (c.std(axis=0) + 1e-6).astype(np.float32)

        xt, wt, mt, ct = self._inputs(tx, tw, tm, cur)
        y = torch.from_numpy(np.log(np.maximum(peak, MIN_MULTIPLE)).astype(np.float32))
        cens = torch.from_numpy(np.asarray(censored, dtype=bool))
        ev_np, sv_np = hazard_targets(
            np.asarray(collapse_time, dtype=np.float64), np.asarray(observed), self.collapse_bins
        )
        ev, sv = torch.from_numpy(ev_np), torch.from_numpy(sv_np)
        _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
        w_np = (1.0 / counts[inv]).astype(np.float32)
        w = torch.from_numpy(w_np)

        def loss_on(net: TapeNet, idx: torch.Tensor) -> torch.Tensor:
            log_pi, mu, s, hz = net(xt[idx], wt[idx], mt[idx], ct[idx])
            per_row = censored_nll(log_pi, mu, s, y[idx], cens[idx]) + hazard_nll(hz, ev[idx], sv[idx])
            return (per_row * w[idx]).sum() / w[idx].sum()

        tr_groups = np.unique(inv[tr])
        val_idx = torch.from_numpy(va)
        best_losses = []
        for i, net in enumerate(self.members):
            rng = np.random.default_rng(self.seed + i)
            gen = torch.Generator().manual_seed(self.seed + i)
            picks = rng.choice(tr_groups, size=len(tr_groups), replace=True)
            by_group = {g: tr[inv[tr] == g] for g in np.unique(picks)}
            boot = torch.from_numpy(np.concatenate([by_group[g] for g in picks]))
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
            best, best_state, bad = np.inf, copy.deepcopy(net.state_dict()), 0
            for _ in range(epochs):
                net.train()
                perm = boot[torch.randperm(len(boot), generator=gen)]
                for s0 in range(0, len(perm), batch_size):
                    loss = loss_on(net, perm[s0 : s0 + batch_size])
                    opt.zero_grad()
                    torch.autograd.backward(loss)
                    nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step()
                if not len(va):
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
                best_losses.append(best)
        self.report = {
            "n_train": len(tr),
            "n_validation": len(va),
            "tokens": len(ordered),
            "parameters_per_member": self.parameters_per_member,
            "known_wallets": int(self.known[1:].sum()),
        }
        if len(va):
            self.report["validation_loss"] = float(np.mean(best_losses))
            pi, mu, s, _ = self._forward(tx[va], tw[va], tm[va], cur[va])
            levels = np.asarray(self.spec.levels, dtype=np.float64)
            p = mixture_survival(pi, mu, s, levels).mean(0)
            self.calibration = fit_calibration_ratio(p, peak[va], censored[va], w_np[va], levels)
            self.report["calibration_ratio"] = dict(
                zip([f"{k:g}x" for k in self.spec.levels], self.calibration.tolist(), strict=True)
            )
        return self.report

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _forward(
        self, tx: F32, tw: I64, tm: B, cur: npt.NDArray[Any], batch_size: int = 1024
    ) -> tuple[F64, F64, F64, F64]:
        """Member outputs: π, μ, s of shape (E, N, K) and collapse hazards (E, N, bins)."""
        xt, wt, mt, ct = self._inputs(tx, tw, tm, cur)
        outs: list[list[torch.Tensor]] = [[], [], [], []]
        for net in self.members:
            net.eval()
            parts: list[list[torch.Tensor]] = [[], [], [], []]
            for s0 in range(0, len(xt), batch_size):
                sl = slice(s0, s0 + batch_size)
                log_pi, mu, s, hz = net(xt[sl], wt[sl], mt[sl], ct[sl])
                for j, v in enumerate((log_pi.exp(), mu, s, torch.sigmoid(hz))):
                    parts[j].append(v)
            for j in range(4):
                outs[j].append(torch.cat(parts[j]) if parts[j] else torch.zeros(0))
        pi, mu, s, hz = (torch.stack(o).double().numpy() for o in outs)
        return pi, mu, s, hz

    def predict(self, tx: F32, tw: I64, tm: B, cur: npt.NDArray[Any]) -> TapePrediction:
        """Calibrated tail view and collapse curve for a batch of tapes."""
        pi, mu, s, hz = self._forward(tx, tw, tm, cur)
        tail = summarize_tail(pi, mu, s, self.spec, self.calibration, self.kelly_multiplier, self.max_kelly)
        collapse = 1.0 - np.cumprod(1.0 - hz, axis=-1)  # (E, N, bins)
        return TapePrediction(tail, collapse.mean(0), collapse.std(0))

    def nll(self, tx: F32, tw: I64, tm: B, cur: npt.NDArray[Any], peak: F64, censored: B) -> F64:
        """Per-row NLL of the log peak multiple under the ensemble (comparable to the tail model)."""
        pi, mu, s, _ = self._forward(tx, tw, tm, cur)
        return mixture_nll(pi, mu, s, peak, censored)

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> None:
        """Write members, scalers, calibration and ``tape.json`` to ``directory``."""
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        torch.save({f"m{i}": m.state_dict() for i, m in enumerate(self.members)}, d / "tape.pt")
        np.savez(
            d / "tape_scaler.npz",
            trade_mean=self.trade_mean,
            trade_std=self.trade_std,
            cur_mean=self.cur_mean,
            cur_std=self.cur_std,
            cur_lo=self.cur_lo,
            cur_hi=self.cur_hi,
            calibration=self.calibration,
            known=self.known,
        )
        meta = {
            "n_current": self.n_current,
            "members": len(self.members),
            "d": self.d,
            "layers": self.layers,
            "collapse_bins": list(self.collapse_bins),
            "seed": self.seed,
            "kelly_multiplier": self.kelly_multiplier,
            "max_kelly": self.max_kelly,
            "feature_names": self.feature_names,
            "min_wallet_tokens": self.min_wallet_tokens,
            "spec": self.spec.model_dump(),
            "tape": self.tape.model_dump(),
            "report": self.report,
        }
        (d / "tape.json").write_text(json.dumps(meta, indent=2, default=float))

    @classmethod
    def load(cls, directory: str | Path) -> TapeModel:
        """Rebuild a model saved by :meth:`save`."""
        d = Path(directory)
        meta = json.loads((d / "tape.json").read_text())
        model = cls(
            int(meta["n_current"]),
            MoonshotSpec.model_validate(meta["spec"]),
            TapeSpec.model_validate(meta["tape"]),
            members=int(meta["members"]),
            d=int(meta["d"]),
            layers=int(meta["layers"]),
            collapse_bins=tuple(meta["collapse_bins"]),
            seed=int(meta["seed"]),
            kelly_multiplier=float(meta["kelly_multiplier"]),
            max_kelly=float(meta["max_kelly"]),
            feature_names=list(meta["feature_names"]),
            min_wallet_tokens=int(meta.get("min_wallet_tokens", 3)),
        )
        states = torch.load(d / "tape.pt", map_location="cpu", weights_only=True)
        for i, m in enumerate(model.members):
            m.load_state_dict(states[f"m{i}"])
        with np.load(d / "tape_scaler.npz") as z:
            model.trade_mean, model.trade_std = z["trade_mean"], z["trade_std"]
            model.cur_mean, model.cur_std = z["cur_mean"], z["cur_std"]
            model.cur_lo, model.cur_hi = z["cur_lo"], z["cur_hi"]
            model.calibration = z["calibration"]
            if "known" in z:
                model.known = z["known"]
        model.report = dict(meta.get("report", {}))
        return model
