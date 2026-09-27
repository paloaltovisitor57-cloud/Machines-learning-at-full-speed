"""Solana feature engineering: market state → :class:`NeuralObservation`.

Three views of every token, all computed from state known at ``now``:

* **current features** — the named vector :data:`CURRENT_FEATURES` (lifecycle and pool
  state, price action, order flow, holder concentration, insider/sniper/bundle
  exposure, smart-money and bot activity, execution competition, token safety);
* **trade bars** at each configured resolution (:data:`BAR_FEATURES`), with prices
  forward-filled through quiet periods so "nobody traded" is information, not a gap;
* **wallet graph** — the token plus its most active wallets, linked by trades, funding
  transfers and shared funding clusters (:data:`NODE_FEATURES`, :data:`EDGE_TYPES`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.schemas import GraphInput, NeuralObservation, SequenceInput
from nardis_neural.solana.amm import bonding_progress, round_trip_cost
from nardis_neural.solana.config import (
    BAR_FEATURES,
    CURRENT_FEATURES,
    NODE_FEATURES,
    BarSpec,
    SolanaConfig,
)
from nardis_neural.solana.events import TokenEventLog
from nardis_neural.solana.market import SolanaMarket
from nardis_neural.solana.wallets import WalletIntel

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
DUST_TOKENS = 1.0


def slog(x: F64 | float) -> Any:
    """Signed log1p — compresses heavy-tailed SOL flows while keeping direction."""
    return np.sign(x) * np.log1p(np.abs(x))


def gini(values: F64) -> float:
    v = np.sort(values[values > 0])
    n = len(v)
    if n < 2 or v.sum() <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((n + 1 - 2 * (cum / cum[-1]).sum()) / n)


class SolanaFeatureBuilder:
    def __init__(self, cfg: SolanaConfig | None = None) -> None:
        self.cfg = cfg or SolanaConfig()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _log_prices(log: TokenEventLog) -> F64:
        s = log.swaps
        return np.asarray(np.log(s["sol_reserve"] / np.maximum(s["token_reserve"], 1e-12)), dtype=np.float64)

    @staticmethod
    def _log_price_at(log: TokenEventLog, times: F64) -> F64:
        r = log.reserves
        rt = r["t"]
        idx = np.searchsorted(rt, times, side="right") - 1
        launch = np.log(log.launch.sol_reserve / log.launch.token_reserve)
        if len(rt) == 0:
            return np.full(len(times), launch, dtype=np.float64)
        p = np.log(r["sol_reserve"] / np.maximum(r["token_reserve"], 1e-12))
        return np.asarray(np.where(idx >= 0, p[np.clip(idx, 0, None)], launch), dtype=np.float64)

    @staticmethod
    def _liquidity_at(log: TokenEventLog, times: F64) -> F64:
        r = log.reserves
        idx = np.searchsorted(r["t"], times, side="right") - 1
        base = log.launch.sol_reserve - (30.0 if log.launch.venue == "pump_fun" else 0.0)
        if len(r) == 0:
            return np.full(len(times), max(base, 0.0))
        liq = np.maximum(r["sol_reserve"] - r["virtual"], 0.0)
        return np.asarray(np.where(idx >= 0, liq[np.clip(idx, 0, None)], max(base, 0.0)), dtype=np.float64)

    def _check_now(self, log: TokenEventLog, now: float) -> None:
        if now + 1e-9 < log.last_t:
            raise ValueError(
                f"{log.mint}: market state already contains events after now={now} (last={log.last_t}); "
                "features must be built causally at the current market time"
            )

    # ------------------------------------------------------------------ current features
    def current_features(self, log: TokenEventLog, wallets: WalletIntel, now: float) -> F64:
        self._check_now(log, now)
        cfg = self.cfg
        s = log.swaps
        t = s["t"]
        n = len(t)
        f: dict[str, float] = {}
        age = max(now - log.launch.t, 0.0)
        f["age_log"] = np.log1p(age)
        f["n_swaps_log"] = np.log1p(n)
        f["since_last_trade_log"] = np.log1p(now - t[-1]) if n else np.log1p(age)
        pool = log.pool()
        pump = log.venue == "pump_fun"
        f["bonding_progress"] = bonding_progress(pool) if pump else 1.0
        f["migrated"] = float(log.migrated_at is not None)
        liq_now, liq_300 = self._liquidity_at(log, np.array([now, now - 300.0]))
        f["liquidity_sol_log"] = np.log1p(liq_now)
        f["liquidity_change_300s"] = float(np.clip((liq_now - liq_300) / max(liq_300, 1e-6), -1.0, 5.0))
        f["market_cap_sol_log"] = np.log1p(pool.price * log.launch.supply)
        f["round_trip_cost"] = round_trip_cost(pool, cfg.trade_size_sol)

        logp = self._log_prices(log)
        lp_now = float(self._log_price_at(log, np.array([now]))[0])
        f["drawdown_from_ath"] = lp_now - float(logp.max()) if n else 0.0
        f["runup_from_low"] = lp_now - float(logp.min()) if n else 0.0
        past = self._log_price_at(log, now - np.array([5.0, 30.0, 60.0, 300.0]))
        for k, v in zip(("ret_5s", "ret_30s", "ret_60s", "ret_300s"), lp_now - past, strict=True):
            f[k] = float(v)
        dlp = np.diff(logp, prepend=logp[0] if n else 0.0)
        for w in (30, 300):
            sel = t > now - w
            f[f"rv_{w}s"] = float(np.sqrt((dlp[sel] ** 2).sum()))

        wal = s["wallet"]
        buy = s["is_buy"]
        sol = s["sol"]
        w60 = t > now - 60
        w300 = t > now - 300
        b60, s60 = w60 & buy, w60 & ~buy
        nb, ns = int(b60.sum()), int(s60.sum())
        buy_sol, sell_sol = float(sol[b60].sum()), float(sol[s60].sum())
        f["buys_60s_log"] = np.log1p(nb)
        f["sells_60s_log"] = np.log1p(ns)
        f["buy_ratio_60s"] = nb / (nb + ns) if nb + ns else 0.5
        f["buy_sol_share_60s"] = buy_sol / (buy_sol + sell_sol) if buy_sol + sell_sol > 0 else 0.5
        f["net_flow_60s"] = float(slog(buy_sol - sell_sol))
        f["net_flow_300s"] = float(slog(sol[w300 & buy].sum() - sol[w300 & ~buy].sum()))
        f["volume_60s_log"] = np.log1p(buy_sol + sell_sol)
        f["unique_buyers_60s_log"] = np.log1p(len(np.unique(wal[b60])))
        f["unique_sellers_60s_log"] = np.log1p(len(np.unique(wal[s60])))
        if n:
            uniq, first_idx = np.unique(wal, return_index=True)
            first_t = dict(zip(uniq.tolist(), t[first_idx].tolist(), strict=True))
            recent = wal[w60]
            f["new_trader_share_60s"] = (
                float(np.mean([first_t[int(w)] > now - 60 for w in recent])) if len(recent) else 0.0
            )
        else:
            f["new_trader_share_60s"] = 0.0
        f["avg_buy_size_60s_log"] = np.log1p(buy_sol / nb) if nb else 0.0

        supply = log.launch.supply
        holders = {w: b for w, b in log.balances.items() if b > DUST_TOKENS}
        bal = np.asarray(list(holders.values()), dtype=np.float64)
        f["holders_log"] = np.log1p(len(bal))
        f["top10_share"] = float(np.sort(bal)[-10:].sum() / supply) if len(bal) else 0.0
        f["holder_hhi"] = float(((bal / bal.sum()) ** 2).sum()) if len(bal) and bal.sum() > 0 else 0.0
        f["holder_gini"] = gini(bal)
        creator = log.creator_id
        f["dev_share"] = log.balances.get(creator, 0.0) / supply
        dev_bought = log.bought.get(creator, 0.0)
        f["dev_sold_fraction"] = min(log.sold.get(creator, 0.0) / dev_bought, 1.0) if dev_bought > 0 else 0.0
        snipers = [w for w, sl in log.first_buy_slot.items() if sl <= log.launch_slot + cfg.sniper_slots]
        f["sniper_share"] = sum(log.balances.get(w, 0.0) for w in snipers) / supply
        early = [w for w, sl in log.first_buy_slot.items() if sl <= log.launch_slot + cfg.bundle_slots]
        roots = [wallets.root(w) for w in early]
        creator_root = wallets.root(creator)
        bundled = [
            w
            for w, r in zip(early, roots, strict=True)
            if w != creator and (r == creator_root or roots.count(r) > 1)
        ]
        f["bundle_share"] = sum(log.balances.get(w, 0.0) for w in bundled) / supply
        f["creator_cluster_share"] = (
            sum(b for w, b in holders.items() if wallets.root(w) == creator_root) / supply
        )
        buyers60 = wal[b60]
        f["fresh_wallet_share_60s"] = (
            float(np.mean([wallets.is_fresh(int(w), now, cfg.fresh_wallet_seconds) for w in buyers60]))
            if len(buyers60)
            else 0.0
        )

        w3 = wal[w300]
        if len(w3):
            skills = wallets.skills(w3)
            sign = np.where(buy[w300], 1.0, -1.0)
            f["smart_flow_300s"] = float(slog((sign * sol[w300] * skills).sum()))
            smart_buyers = {
                int(w) for w, sk, bb in zip(w3, skills, buy[w300], strict=True) if bb and sk > 0.2
            }
            f["smart_buyers_300s_log"] = np.log1p(len(smart_buyers))
            rugged = np.asarray([wallets.rugs[int(w)] > 0 for w in w3])
            vol = sol[w300]
            f["rug_associated_share"] = float(vol[rugged].sum() / vol.sum()) if vol.sum() > 0 else 0.0
        else:
            f["smart_flow_300s"] = f["smart_buyers_300s_log"] = f["rug_associated_share"] = 0.0
        w6 = wal[w60]
        f["bot_share_60s"] = (
            float(np.mean([wallets.trades[int(w)] >= cfg.bot_trade_threshold for w in w6]))
            if len(w6)
            else 0.0
        )
        fees = s["priority_fee"][w60]
        f["priority_fee_60s_log"] = np.log1p(fees.mean() * 1e6) if len(fees) else 0.0
        f["jito_share_60s"] = float((s["jito_tip"][w60] > 0).mean()) if len(fees) else 0.0
        f["slot_density_60s"] = len(fees) / 150.0
        f.update(self._dynamics(log, wallets, now, bal_holders=holders))
        f["mint_authority_revoked"] = float(log.launch.mint_authority_revoked)
        f["freeze_authority_revoked"] = float(log.launch.freeze_authority_revoked)
        f["lp_burned_fraction"] = float(log.launch.lp_burned_fraction)
        vec = np.asarray([f[name] for name in CURRENT_FEATURES], dtype=np.float64)
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

    def _dynamics(
        self, log: TokenEventLog, wallets: WalletIntel, now: float, bal_holders: dict[int, float]
    ) -> dict[str, float]:
        """Rates of change that tend to lead price: curve velocity, holder growth, who is
        buying (smart money) and who is selling (top holders), and buy acceleration."""
        s = log.swaps
        t, wal, buy, sol, tok = s["t"], s["wallet"], s["is_buy"], s["sol"], s["tokens"]
        out: dict[str, float] = {}
        if log.venue == "pump_fun":
            liq_now, liq_60 = self._liquidity_at(log, np.array([now, now - 60.0]))
            out["bonding_velocity_60s"] = float((liq_now - liq_60) / 85.0)
        else:
            out["bonding_velocity_60s"] = 0.0
        past = t <= now - 60
        if past.any():
            ids = wal[past]
            signed = np.where(buy[past], tok[past], -tok[past])
            _, inv = np.unique(ids, return_inverse=True)
            held = np.bincount(inv, weights=signed)
            holders_60 = int((held > DUST_TOKENS).sum())
        else:
            holders_60 = 0
        out["holders_growth_60s"] = float(np.log1p(len(bal_holders)) - np.log1p(holders_60))
        w60 = t > now - 60
        buyers = np.unique(wal[w60 & buy])
        out["smart_buyer_share_60s"] = float((wallets.skills(buyers) > 0.2).mean()) if len(buyers) else 0.0
        top = {w for w, _ in sorted(bal_holders.items(), key=lambda kv: -kv[1])[:10]}
        sells = w60 & ~buy
        sell_sol = float(sol[sells].sum())
        top_sell = float(sum(v for w, v in zip(wal[sells], sol[sells], strict=True) if int(w) in top))
        out["top_holder_sell_share_60s"] = top_sell / sell_sol if sell_sol > 0 else 0.0
        recent = int((buy & (t > now - 30)).sum())
        prior = int((buy & (t > now - 60) & (t <= now - 30)).sum())
        out["buy_acceleration_30s"] = float(np.log1p(recent) - np.log1p(prior))
        return out

    # ------------------------------------------------------------------ bars
    def bars(self, log: TokenEventLog, now: float, spec: BarSpec) -> SequenceInput:
        """OHLCV-style bars ending at ``now`` from swaps with ``t <= now``."""
        self._check_now(log, now)
        length, res = spec.length, spec.resolution_seconds
        s = log.swaps
        t = s["t"]
        start = now - length * res
        lo = int(np.searchsorted(t, start, side="right"))
        hi = int(np.searchsorted(t, now, side="right"))
        tt = t[lo:hi]
        logp_all = self._log_prices(log)
        lp = logp_all[lo:hi]
        prev_lp = float(logp_all[lo - 1]) if lo > 0 else None
        bucket = np.clip(length - 1 - np.floor((now - tt) / res).astype(np.int64), 0, length - 1)
        sol = s["sol"][lo:hi]
        buy = s["is_buy"][lo:hi]
        wal = s["wallet"][lo:hi]
        count = np.bincount(bucket, minlength=length).astype(np.float64)
        vol = np.bincount(bucket, weights=sol, minlength=length)
        buy_vol = np.bincount(bucket, weights=sol * buy, minlength=length)
        fee = np.bincount(bucket, weights=s["priority_fee"][lo:hi], minlength=length)
        uniq = np.zeros(length)
        if len(tt):
            keys = np.unique(bucket * (int(wal.max()) + 1) + wal)
            uniq = np.bincount(keys // (int(wal.max()) + 1), minlength=length).astype(np.float64)
        high = np.full(length, -np.inf)
        low = np.full(length, np.inf)
        np.maximum.at(high, bucket, lp)
        np.minimum.at(low, bucket, lp)
        last_idx = np.full(length, -1)
        np.maximum.at(last_idx, bucket, np.arange(len(tt)))
        prev_all = np.concatenate(
            [[prev_lp if prev_lp is not None else (lp[0] if len(lp) else 0.0)], lp[:-1]]
        )
        rv = np.bincount(bucket, weights=(lp - prev_all) ** 2, minlength=length)

        close = np.full(length, np.nan)
        has = last_idx >= 0
        close[has] = lp[last_idx[has]]
        carry = prev_lp
        prev_close = np.full(length, np.nan)
        for j in range(length):
            prev_close[j] = np.nan if carry is None else carry
            if not np.isnan(close[j]):
                carry = float(close[j])
            elif carry is not None:
                close[j] = carry
        bar_end = np.asarray(now - (length - 1 - np.arange(length)) * res, dtype=np.float64)
        first_trade = t[0] if len(t) else np.inf
        mask = (bar_end >= min(first_trade, log.launch.t)) & ~np.isnan(close)
        ret = np.where(np.isnan(prev_close), 0.0, close - np.nan_to_num(prev_close, nan=0.0))
        liq = self._liquidity_at(log, bar_end)
        values = np.stack(
            [
                np.where(mask, ret, 0.0),
                np.sqrt(rv),
                np.log1p(vol),
                np.where(vol > 0, buy_vol / np.maximum(vol, 1e-12), 0.5),
                np.log1p(count),
                np.log1p(uniq),
                slog(2 * buy_vol - vol),
                np.log1p(liq),
                np.log1p(np.where(count > 0, fee / np.maximum(count, 1), 0.0) * 1e6),
                np.where(has, high - low, 0.0),
            ],
            axis=1,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        values[~mask] = np.nan
        deltas = ((length - 1 - np.arange(length)) * res).astype(np.float32)
        assert values.shape[1] == len(BAR_FEATURES)
        return SequenceInput(values=values, time_deltas=deltas, mask=mask)

    # ------------------------------------------------------------------ graph
    def graph(self, log: TokenEventLog, wallets: WalletIntel, now: float) -> GraphInput:
        cfg = self.cfg
        s = log.swaps
        t = s["t"]
        sel = (t > now - cfg.graph_window_seconds) & (t <= now)
        wal = s["wallet"][sel]
        sol = s["sol"][sel]
        pool = log.pool()
        token_node = np.zeros(len(NODE_FEATURES))
        token_node[0] = 1.0
        token_node[1] = np.log1p(sol.sum())
        token_node[2] = bonding_progress(pool) if log.venue == "pump_fun" else 1.0
        token_node[3] = np.log1p(max(now - log.launch.t, 0.0)) / 10.0
        nodes = [token_node]
        if len(wal):
            uniq, inv = np.unique(wal, return_inverse=True)
            vol = np.bincount(inv, weights=sol)
            top = uniq[np.argsort(-vol)[: cfg.graph_top_k]]
            vol_of = dict(zip(uniq.tolist(), vol.tolist(), strict=True))
        else:
            top = np.zeros(0, dtype=np.int64)
            vol_of = {}
        snipers = {w for w, sl in log.first_buy_slot.items() if sl <= log.launch_slot + cfg.sniper_slots}
        index = {int(w): i + 1 for i, w in enumerate(top)}
        for w in top.tolist():
            nodes.append(
                np.array(
                    [
                        0.0,
                        np.log1p(vol_of[w]),
                        log.balances.get(w, 0.0) / log.launch.supply * 10.0,
                        wallets.score(w),
                        np.log1p(wallets.evidence(w)),
                        float(w == log.creator_id),
                        float(w in snipers),
                        np.log1p(wallets.cluster_size(w)),
                    ]
                )
            )
        src, dst, etype = [], [], []
        for w, i in index.items():
            src.append(i)
            dst.append(0)
            etype.append(0)
            funder = wallets.funder.get(w)
            if funder is not None and funder in index and wallets.funded_at.get(w, np.inf) <= now:
                src.append(index[funder])
                dst.append(i)
                etype.append(1)
        members = list(index)
        roots = {w: wallets.root(w) for w in members}
        for a_pos, a in enumerate(members):
            for b in members[a_pos + 1 :]:
                if roots[a] == roots[b]:
                    src += [index[a], index[b]]
                    dst += [index[b], index[a]]
                    etype += [2, 2]
        return GraphInput(
            node_features=np.stack(nodes).astype(np.float32),
            edge_index=np.asarray([src, dst], dtype=np.int64).reshape(2, -1),
            edge_type=np.asarray(etype, dtype=np.int64),
            target_node=0,
        )

    # ------------------------------------------------------------------ observation
    def observation(self, market: SolanaMarket, mint: str, now: float | None = None) -> NeuralObservation:
        log = market.token(mint)
        now = market.now if now is None else now
        current = self.current_features(log, market.wallets, now)
        seqs = {b.name: self.bars(log, now, b) for b in self.cfg.bars}
        graph = self.graph(log, market.wallets, now) if self.cfg.graph_enabled else None
        return NeuralObservation(
            observation_id=f"{mint}@{now:.3f}",
            timestamp=now,
            current_features=current.astype(np.float32),
            sequences=seqs,
            graph=graph,
            metadata={"mint": mint, "venue": log.venue},
        )

    @staticmethod
    def explain(current: F64) -> dict[str, float]:
        return {name: float(v) for name, v in zip(CURRENT_FEATURES, current, strict=True)}
