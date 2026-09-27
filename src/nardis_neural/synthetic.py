"""Synthetic multi-regime market generator — for *testing the ML system only*.

It does not model real profitability.  It produces data with genuine, learnable
structure so that every component can be exercised end-to-end:

* a hidden Markov regime per token (calm, momentum-up, dump, choppy reversal),
* AR(1) returns whose sign of autocorrelation depends on the regime (momentum vs
  reversals), fat-tailed Student-t shocks, order-flow imbalance that *leads* returns in
  trending regimes, volume bursts linked to |return|,
* bars at every configured resolution built causally from the 1-second path,
* genuinely missing data (young tokens have short histories, random missing bars, some
  observations without a slow sequence, sporadic NaN current features),
* future outcomes per horizon: log return, max upside, max drawdown, realised volatility.

Feature dimensions adapt to any :class:`~nardis_neural.config.NeuralConfig`: extra
columns are filled with noise, surplus base features are dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from nardis_neural.config import NeuralConfig
from nardis_neural.data.loaders import (
    ID_WIDTH,
    KEY_CURRENT,
    KEY_ID,
    KEY_REGIME,
    KEY_TARGET_MASK,
    KEY_TIMESTAMP,
    Array,
    seq_key,
    target_key,
)

F64 = npt.NDArray[np.float64]

# regime: (drift, sigma, ar coefficient, imbalance→return coupling, base log volume)
REGIMES = np.array(
    [
        [0.00000, 0.0007, -0.05, 0.0, 1.0],  # calm
        [0.00030, 0.0015, 0.25, 0.8, 2.0],  # momentum up
        [-0.00035, 0.0020, 0.20, 0.8, 2.3],  # dump
        [0.00000, 0.0025, -0.35, -0.4, 2.6],  # choppy reversal
    ]
)
N_REGIMES = len(REGIMES)


@dataclass
class SyntheticSpec:
    n_observations: int = 2000
    n_tokens: int = 24
    episode_seconds: int = 1800
    seed: int = 0
    start_time: float = 1_700_000_000.0
    token_spacing_seconds: float = 300.0
    missing_slow_prob: float = 0.1
    missing_bar_prob: float = 0.02
    nan_feature_prob: float = 0.005
    regime_switch_prob: float = 1.0 / 180.0
    graph: bool = False
    drift_shift: float = 0.0
    """Adds this many sigmas of extra drift + volatility — used to simulate distribution shift."""


def _simulate_paths(spec: SyntheticSpec, length: int, rng: np.random.Generator) -> dict[str, F64]:
    k = spec.n_tokens
    regime = np.zeros((k, length), dtype=np.int64)
    r = np.zeros((k, length))
    imb = np.zeros((k, length))
    vol = np.zeros((k, length))
    state = rng.integers(0, N_REGIMES, size=k)
    prev_r = np.zeros(k)
    prev_imb = np.zeros(k)
    shocks = rng.standard_t(df=4, size=(k, length)) / np.sqrt(2.0)
    imb_noise = rng.normal(size=(k, length))
    vol_noise = rng.normal(size=(k, length))
    switch = rng.random((k, length)) < spec.regime_switch_prob
    new_state = rng.integers(0, N_REGIMES, size=(k, length))
    shift = spec.drift_shift
    for t in range(length):
        state = np.where(switch[:, t], new_state[:, t], state)
        p = REGIMES[state]
        mu, sigma, ar, coupling, base_vol = p[:, 0], p[:, 1] * (1 + 0.5 * shift), p[:, 2], p[:, 3], p[:, 4]
        mu = mu + shift * 0.0003
        imb_t = np.tanh(3.0 * np.sign(mu) * (np.abs(mu) > 0) + 0.8 * imb_noise[:, t] + 0.5 * prev_imb)
        rt = mu + ar * prev_r + coupling * 0.5 * sigma * prev_imb + sigma * shocks[:, t]
        regime[:, t] = state
        r[:, t] = rt
        imb[:, t] = imb_t
        vol[:, t] = np.exp(base_vol + 120.0 * np.abs(rt) + 0.3 * vol_noise[:, t])
        prev_r, prev_imb = rt, imb_t
    logp = np.concatenate(
        [np.zeros((k, 1)), np.cumsum(r, axis=1)], axis=1
    )  # logp[:, t] = price after t steps
    return {"regime": regime.astype(np.float64), "r": r, "imb": imb, "vol": vol, "logp": logp}


def _fit_width(base: F64, width: int, rng: np.random.Generator) -> F64:
    """Truncate or pad (with noise) the last axis to ``width`` columns."""
    have = base.shape[-1]
    if have >= width:
        return base[..., :width]
    pad = rng.normal(scale=1.0, size=(*base.shape[:-1], width - have))
    return np.concatenate([base, pad], axis=-1)


def generate_synthetic(config: NeuralConfig, spec: SyntheticSpec | None = None) -> dict[str, Array]:
    """Generate a labelled dataset in the canonical array layout, sorted by timestamp."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    horizons = config.targets.horizons
    max_h = int(np.ceil(max(h.seconds for h in horizons)))
    length = spec.episode_seconds + max_h + 2
    paths = _simulate_paths(spec, length, rng)
    r, imb, vol, logp = paths["r"], paths["imb"], paths["vol"], paths["logp"]
    k = spec.n_tokens
    # cumulative sums for O(1) window aggregates; index c[:, t] = sum of steps < t
    c_r2 = np.concatenate([np.zeros((k, 1)), np.cumsum(r**2, axis=1)], axis=1)
    c_vol = np.concatenate([np.zeros((k, 1)), np.cumsum(vol, axis=1)], axis=1)
    c_imb = np.concatenate([np.zeros((k, 1)), np.cumsum(imb, axis=1)], axis=1)
    c_big = np.concatenate([np.zeros((k, 1)), np.cumsum(np.abs(r) > 0.002, axis=1)], axis=1)

    n = spec.n_observations
    tok = rng.integers(0, k, size=n)
    t_obs = rng.integers(10, spec.episode_seconds, size=n)  # obs at time t sees steps [0, t)

    def window(c: F64, end: npt.NDArray[np.int64], span: int) -> F64:
        start = np.maximum(end - span, 0)
        return c[tok, end] - c[tok, start]

    def ret_back(span: int) -> F64:
        return logp[tok, t_obs] - logp[tok, np.maximum(t_obs - span, 0)]

    age = t_obs.astype(np.float64)
    run_max = np.array([logp[a, max(b - 120, 0) : b + 1].max() for a, b in zip(tok, t_obs, strict=True)])
    run_min = np.array([logp[a, max(b - 120, 0) : b + 1].min() for a, b in zip(tok, t_obs, strict=True)])
    cur_base = np.stack(
        [
            ret_back(1),
            ret_back(5),
            ret_back(30),
            ret_back(60),
            ret_back(300),
            np.sqrt(window(c_r2, t_obs, 30)),
            np.sqrt(window(c_r2, t_obs, 120)),
            np.log1p(window(c_vol, t_obs, 5)) - np.log1p(window(c_vol, t_obs, 60) / 12.0),
            window(c_imb, t_obs, 10) / 10.0,
            window(c_imb, t_obs, 60) / 60.0,
            np.log1p(age),
            logp[tok, t_obs] - run_max,
            logp[tok, t_obs] - run_min,
            window(c_big, t_obs, 60) / 60.0,
        ],
        axis=1,
    )
    current = _fit_width(cur_base, config.features.current_dim, rng).astype(np.float32)
    nan_mask = rng.random(current.shape) < spec.nan_feature_prob
    current[nan_mask] = np.nan

    timestamps = spec.start_time + tok * spec.token_spacing_seconds + t_obs.astype(np.float64)
    arrays: dict[str, Array] = {
        KEY_ID: np.asarray(
            [f"tok{a}-t{b}-{i}" for i, (a, b) in enumerate(zip(tok, t_obs, strict=True))]
        ).astype(f"<U{ID_WIDTH}"),
        KEY_TIMESTAMP: timestamps,
        KEY_CURRENT: current,
        KEY_REGIME: paths["regime"][tok, np.maximum(t_obs - 1, 0)].astype(np.int64),
    }

    for ts in config.features.timescales:
        res = max(1, round(ts.resolution_seconds))
        big_l = ts.max_len
        offsets = (big_l - 1 - np.arange(big_l)) * res  # bar j ends offsets[j] seconds before t
        ends = t_obs[:, None] - offsets[None, :]  # (n, L)
        starts = ends - res
        valid = starts >= 0
        e = np.clip(ends, 0, length)
        s = np.clip(starts, 0, length)
        tk = tok[:, None]
        bar_ret = logp[tk, e] - logp[tk, s]
        bar_rv = np.sqrt(np.maximum(c_r2[tk, e] - c_r2[tk, s], 0))
        bar_vol = np.log1p(c_vol[tk, e] - c_vol[tk, s])
        bar_imb = (c_imb[tk, e] - c_imb[tk, s]) / res
        rel = logp[tk, e] - logp[tok, t_obs][:, None]
        intensity = (c_big[tk, e] - c_big[tk, s]) / res
        base = np.stack([bar_ret, bar_rv, bar_vol, bar_imb, rel, intensity, np.abs(bar_ret)], axis=-1)
        values = _fit_width(base, ts.feature_dim, rng).astype(np.float32)
        valid = valid & (rng.random(valid.shape) >= spec.missing_bar_prob)
        if ts.name == config.features.timescales[-1].name and len(config.features.timescales) > 1:
            drop_all = rng.random(n) < spec.missing_slow_prob
            valid[drop_all] = False
        values[~valid] = 0.0
        deltas = np.where(valid, offsets[None, :].astype(np.float32), 0.0).astype(np.float32)
        arrays[seq_key(ts.name, "values")] = values
        arrays[seq_key(ts.name, "mask")] = valid
        arrays[seq_key(ts.name, "time_deltas")] = deltas

    h_count = len(horizons)
    targets = {
        name: np.zeros((n, h_count), dtype=np.float32) for name in ("return", "max_upside", "max_drawdown")
    }
    targets["volatility"] = np.zeros((n, h_count), dtype=np.float32)
    for j, hz in enumerate(horizons):
        hs = round(hz.seconds)
        steps = t_obs[:, None] + np.arange(1, hs + 1)[None, :]
        future = logp[tok[:, None], steps] - logp[tok, t_obs][:, None]
        targets["return"][:, j] = future[:, -1]
        targets["max_upside"][:, j] = np.maximum(future.max(axis=1), 0.0)
        targets["max_drawdown"][:, j] = np.maximum(-future.min(axis=1), 0.0)
        targets["volatility"][:, j] = np.sqrt(c_r2[tok, t_obs + hs] - c_r2[tok, t_obs])
    for name, arr in targets.items():
        arrays[target_key(name)] = arr
    arrays[KEY_TARGET_MASK] = np.ones((n, h_count), dtype=np.bool_)

    if spec.graph:
        _add_graphs(arrays, config, rng, imb, tok, t_obs, paths["regime"])

    order = np.argsort(timestamps, kind="stable")
    from nardis_neural.data.loaders import select_rows

    return select_rows(arrays, order)


def _add_graphs(
    arrays: dict[str, Array],
    config: NeuralConfig,
    rng: np.random.Generator,
    imb: F64,
    tok: npt.NDArray[np.int64],
    t_obs: npt.NDArray[np.int64],
    regime: F64,
) -> None:
    fdim = config.features.graph.node_feature_dim
    n_rel = config.features.graph.num_edge_types
    n = len(tok)
    nf, ei, et, n_nodes, n_edges, target = [], [], [], [0], [0], []
    for i in range(n):
        if rng.random() < 0.1:  # some observations have no graph at all
            n_nodes.append(0)
            n_edges.append(0)
            target.append(-1)
            continue
        w = int(rng.integers(3, 9))
        reg = int(regime[tok[i], max(t_obs[i] - 1, 0)])
        flow = float(imb[tok[i], max(t_obs[i] - 10, 0) : t_obs[i]].mean())
        token_feat = np.concatenate([[flow, reg / 3.0, w / 8.0], rng.normal(size=fdim - 3)])[:fdim]
        wallets = rng.normal(size=(w, fdim))
        wallets[:, 0] = flow + rng.normal(scale=0.5, size=w)  # wallets' net buy pressure
        wallets[:, 1] = rng.random(w) < (0.6 if reg == 1 else 0.2)  # "smart money" flag
        nodes = np.vstack([token_feat[None, :], wallets])
        src = np.arange(1, w + 1)
        edges = [np.stack([src, np.zeros(w, dtype=np.int64)], axis=1)]
        types = [np.zeros(w, dtype=np.int64)]
        m = int(rng.integers(0, w))
        if m and n_rel > 1:
            a = rng.integers(1, w + 1, size=m)
            b = rng.integers(1, w + 1, size=m)
            edges.append(np.stack([a, b], axis=1))
            types.append(np.ones(m, dtype=np.int64))
        e = np.concatenate(edges).astype(np.int64)
        nf.append(nodes.astype(np.float32))
        ei.append(e)
        et.append(np.concatenate(types))
        n_nodes.append(nodes.shape[0])
        n_edges.append(e.shape[0])
        target.append(0)
    arrays["graph.node_features"] = np.concatenate(nf) if nf else np.zeros((0, fdim), np.float32)
    arrays["graph.edge_index"] = np.concatenate(ei) if ei else np.zeros((0, 2), np.int64)
    arrays["graph.edge_type"] = np.concatenate(et) if et else np.zeros((0,), np.int64)
    arrays["graph.node_offsets"] = np.cumsum(n_nodes).astype(np.int64)
    arrays["graph.edge_offsets"] = np.cumsum(n_edges).astype(np.int64)
    arrays["graph.target_node"] = np.asarray(target, dtype=np.int64)
