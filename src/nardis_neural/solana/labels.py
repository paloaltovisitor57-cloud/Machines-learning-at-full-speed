"""Hindsight labelling of Solana snapshots.

Labels are computed from the *complete* event history (they describe the future of a
snapshot), while features are computed from a causally replayed market — the two never
mix.  Produces:

* :class:`NeuralOutcome` per horizon: log return, max upside, max drawdown, realised
  volatility (partial horizons are omitted when the history ends too early);
* cost-aware returns: return minus the round-trip cost of ``trade_size_sol`` at the
  snapshot's pool state;
* risk events within ``risk_horizon_seconds``: ``rug`` (liquidity pulled or price
  collapse), ``graduation`` (bonding curve completes), ``dev_dump`` (creator sells a large
  part of its bag).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nardis_neural.schemas import NeuralOutcome
from nardis_neural.solana.amm import AMM_FEE, PUMP_FEE, Pool, round_trip_cost
from nardis_neural.solana.config import SolanaConfig
from nardis_neural.solana.events import TokenEventLog

F64 = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SnapshotLabels:
    """Hindsight labels of one snapshot: outcome, net returns per horizon, risk events (None when
    the risk horizon is incomplete) and round-trip cost.
    """

    outcome: NeuralOutcome
    net_returns: dict[str, float]
    risk: dict[str, float] | None
    round_trip_cost: float


class SolanaLabeler:
    """Labels snapshots from a token's complete event log (hindsight; never features)."""

    def __init__(self, cfg: SolanaConfig | None = None) -> None:
        self.cfg = cfg or SolanaConfig()

    @staticmethod
    def _timeline(log: TokenEventLog) -> tuple[F64, F64, F64]:
        r = log.reserves
        t = np.concatenate([[log.launch.t], r["t"]])
        lp = np.concatenate(
            [
                [np.log(log.launch.sol_reserve / log.launch.token_reserve)],
                np.log(r["sol_reserve"] / np.maximum(r["token_reserve"], 1e-12)),
            ]
        )
        base = log.launch.sol_reserve - (30.0 if log.launch.venue == "pump_fun" else 0.0)
        liq = np.concatenate([[max(base, 0.0)], np.maximum(r["sol_reserve"] - r["virtual"], 0.0)])
        return t, lp, liq

    def _pool_at(self, log: TokenEventLog, t: float) -> Pool:
        r = log.reserves
        idx = int(np.searchsorted(r["t"], t, side="right")) - 1
        if idx < 0:
            sol, tok, virt = log.launch.sol_reserve, log.launch.token_reserve, 30.0
        else:
            sol, tok, virt = (
                float(r["sol_reserve"][idx]),
                float(r["token_reserve"][idx]),
                float(r["virtual"][idx]),
            )
        fee = PUMP_FEE if virt > 0 else AMM_FEE
        return Pool(sol, tok, fee, virt)

    def label(self, full: TokenEventLog, t: float, data_end: float) -> SnapshotLabels | None:
        """Labels for a snapshot of ``full.mint`` at time ``t``; None if no horizon is complete."""
        tl, lp, _ = self._timeline(full)
        i0 = int(np.searchsorted(tl, t, side="right")) - 1
        lp0 = lp[max(i0, 0)]
        rets: dict[str, float] = {}
        ups: dict[str, float] = {}
        dds: dict[str, float] = {}
        vols: dict[str, float] = {}
        for h in self.cfg.horizons:
            end = t + h.seconds
            if end > data_end:
                continue
            j = int(np.searchsorted(tl, end, side="right"))
            path = np.concatenate([[lp0], lp[i0 + 1 : j]]) if i0 + 1 < j else np.array([lp0])
            rets[h.name] = float(path[-1] - lp0)
            ups[h.name] = float(max(path.max() - lp0, 0.0))
            dds[h.name] = float(max(lp0 - path.min(), 0.0))
            vols[h.name] = float(np.sqrt((np.diff(path) ** 2).sum()))
        if not rets:
            return None
        cost = round_trip_cost(self._pool_at(full, t), self.cfg.trade_size_sol)
        net = {k: float(np.expm1(v) - cost) for k, v in rets.items()}
        outcome = NeuralOutcome(
            observation_id=f"{full.mint}@{t:.3f}",
            returns=rets,
            max_upside=ups,
            max_drawdown=dds,
            volatility=vols,
            resolved_at=t + max(self.cfg.horizons[i].seconds for i in range(len(self.cfg.horizons))),
        )
        return SnapshotLabels(outcome, net, self.risk_labels(full, t, data_end), cost)

    def risk_labels(self, full: TokenEventLog, t: float, data_end: float) -> dict[str, float] | None:
        """``rug`` / ``graduation`` / ``dev_dump`` flags (0 or 1) within ``risk_horizon_seconds`` after
        ``t``; None if the history ends first.  A graduation in the window cancels a rug.
        """
        cfg = self.cfg
        end = t + cfg.risk_horizon_seconds
        if end > data_end:
            return None
        tl, lp, liq = self._timeline(full)
        i0 = max(int(np.searchsorted(tl, t, side="right")) - 1, 0)
        j = int(np.searchsorted(tl, end, side="right"))
        window_lp = lp[i0:j] if j > i0 else lp[i0 : i0 + 1]
        window_liq = liq[i0:j] if j > i0 else liq[i0 : i0 + 1]
        price_crash = window_lp.min() <= lp[i0] + np.log(1 - cfg.rug_price_drop)
        liq_pull = liq[i0] > 0.5 and window_liq.min() <= liq[i0] * (1 - cfg.rug_liquidity_drop)
        graduated = full.migrated_at is not None and t < full.migrated_at <= end
        s = full.swaps
        creator = s["wallet"] == full.creator_id
        before = creator & (s["t"] <= t)
        held = float(np.where(s["is_buy"][before], s["tokens"][before], -s["tokens"][before]).sum())
        in_window = creator & (s["t"] > t) & (s["t"] <= end) & ~s["is_buy"]
        dumped = held > 0 and float(s["tokens"][in_window].sum()) >= cfg.dev_dump_fraction * held
        rug = bool((price_crash or liq_pull) and not graduated)
        return {"rug": float(rug), "graduation": float(graduated), "dev_dump": float(dumped)}
