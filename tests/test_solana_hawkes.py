"""Criticality engine: exact kernel sums, Hawkes recovery, features and simulator herding."""

from __future__ import annotations

import numpy as np

from nardis_neural.solana import LaunchSimSpec, SolanaConfig, SolanaMarket, simulate_launches
from nardis_neural.solana.features import SolanaFeatureBuilder
from nardis_neural.solana.hawkes import EMPTY, excitation_sums, fit_hawkes, simulate_hawkes


def test_excitation_sums_are_exact_with_ties_and_long_spans() -> None:
    rng = np.random.default_rng(0)
    t = np.sort(np.round(rng.uniform(0, 5000, 400), 0))  # many ties, spans ≫ 600/β
    betas = np.array([0.005, 0.4, 8.0])
    d = t[:, None] - t[None, :]
    ref = np.stack([np.where(d > 0, np.exp(-b * np.where(d > 0, d, 0.0)), 0.0).sum(1) for b in betas])
    np.testing.assert_allclose(excitation_sums(t, betas), ref, rtol=1e-10, atol=1e-12)


def test_hawkes_recovers_branching_and_orders_criticality() -> None:
    rng = np.random.default_rng(1)
    est = {}
    for n_true in (0.1, 0.5, 0.8):
        fits = [fit_hawkes(simulate_hawkes(0.3, n_true, 0.5, 600, rng), 0, 600) for _ in range(8)]
        est[n_true] = float(np.mean([f.branching for f in fits]))
        assert abs(est[n_true] - n_true) < 0.15, (n_true, est[n_true])
        share = float(np.mean([f.endogenous_share for f in fits]))
        assert abs(share - n_true) < 0.2, "the endogenous share tracks n for a stationary process"
    assert est[0.1] < est[0.5] < est[0.8]
    poisson = rng.uniform(0, 600, 300)
    assert fit_hawkes(poisson, 0, 600).branching < 0.2, "independent arrivals are not a cascade"
    assert fit_hawkes(np.array([1.0, 2.0]), 0, 10) is EMPTY


def test_herding_makes_buy_flow_self_exciting() -> None:
    def branching(herding: bool) -> float:
        store, arch = simulate_launches(
            LaunchSimSpec(
                n_tokens=6,
                seed=9,
                n_retail=200,
                duration_seconds=3600,
                archetype_weights={"graduate": 1.0},
                herding=herding,
            )
        )
        m = SolanaMarket(SolanaConfig())
        m.ingest_many(store.sorted())
        out = []
        for mint in arch:
            s = m.token(mint).swaps
            t = s["t"][s["is_buy"]]
            out.append(fit_hawkes(t, t[0], t[0] + 900).branching)
        return float(np.mean(out))

    assert branching(True) > branching(False) + 0.15


def test_criticality_features_are_finite_and_causal() -> None:
    store, _ = simulate_launches(
        LaunchSimSpec(n_tokens=4, seed=2, n_retail=120, duration_seconds=1800, herding=True)
    )
    m = SolanaMarket(SolanaConfig())
    m.ingest_many(store.sorted())
    b = SolanaFeatureBuilder(SolanaConfig())
    for mint in m.tokens:
        log = m.token(mint)
        f = b.explain(b.current_features(log, m.wallets, log.last_t, m))
        for k in (
            "buy_branching_ratio",
            "sell_branching_ratio",
            "endogenous_buy_share",
            "herding_timescale_log",
        ):
            assert np.isfinite(f[k]) and f[k] >= 0
        assert 0 <= f["endogenous_buy_share"] <= 1


def test_detrending_separates_a_fading_rush_from_a_cascade() -> None:
    from nardis_neural.solana.hawkes import TRENDS

    rng = np.random.default_rng(5)

    def fading(rate0: float, tau: float, horizon: float) -> np.ndarray:
        t, out = 0.0, []
        while True:  # inhomogeneous Poisson by thinning: no excitation at all
            t += rng.exponential(1 / rate0)
            if t > horizon:
                break
            if rng.random() < np.exp(-t / tau):
                out.append(t)
        return np.asarray(out)

    rush = [fading(3.0, 60.0, 400.0) for _ in range(6)]
    plain = np.mean([fit_hawkes(e, 0, 400).branching for e in rush])
    detrended = np.mean([fit_hawkes(e, 0, 400, trends=TRENDS).branching for e in rush])
    assert plain > 0.6 and detrended < 0.3, (plain, detrended)
    cascades = [simulate_hawkes(0.3, 0.8, 0.5, 600, rng) for _ in range(6)]
    assert np.mean([fit_hawkes(e, 0, 600, trends=TRENDS).branching for e in cascades]) > 0.6


def test_two_kernel_fit_splits_reflexes_from_herding() -> None:
    rng = np.random.default_rng(7)
    slow = np.geomspace(0.03, 1.0, 5)
    cascades = [simulate_hawkes(0.3, 0.7, 0.3, 900, rng) for _ in range(5)]
    fits = [fit_hawkes(e, 0, 900, betas=slow, fast_beta=4.0) for e in cascades]
    total = np.mean([f.branching for f in fits])
    herding = np.mean([f.branching - f.reflex for f in fits])
    assert abs(total - 0.7) < 0.2 and herding > 0.4, "a slow cascade is attributed to the herding kernel"
    reflexes = [simulate_hawkes(0.3, 0.7, 4.0, 900, rng) for _ in range(5)]
    fr = [fit_hawkes(e, 0, 900, betas=slow, fast_beta=4.0) for e in reflexes]
    assert np.mean([f.reflex for f in fr]) > np.mean([f.reflex for f in fits]), (
        "fast cascades load the reflex kernel"
    )
