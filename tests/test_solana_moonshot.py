"""Moonshot engine: executable peak-multiple labels with censoring, the censored power-law tail
model, runner launches in the simulator and SolanaBrain integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, SolanaMarket, simulate_launches
from nardis_neural.solana.amm import round_trip_cost
from nardis_neural.solana.config import CURRENT_FEATURES
from nardis_neural.solana.moonshot import MoonshotSpec, TailModel, ladder_payoff, moonshot_outcome
from nardis_neural.solana.simulator import MARKET_PRESETS
from tests.conftest import make_tiny_config
from tests.test_solana_edge import T0, _market

SPEC = MoonshotSpec(size_sol=0.2, horizon_seconds=3600, min_entry_age_seconds=0)


# ---------------------------------------------------------------- labels
def test_flat_path_peak_is_the_round_trip() -> None:
    m = _market([(1, "a", True, 2.0)])
    log = m.token("TOK")
    out = moonshot_outcome(log, T0 + 5, SPEC, data_end=T0 + 10_000)
    assert out is not None and not out.censored
    expected = 1 - round_trip_cost(log.pool(), SPEC.size_sol)
    assert out.peak_multiple == pytest.approx(expected, rel=1e-6)
    assert out.ladder_multiple == pytest.approx(expected, rel=1e-6), "nothing fires: sold at the horizon"


def test_pump_peak_ladder_and_censoring() -> None:
    pump = [(1, "a", True, 0.5)] + [(10 + i, f"b{i}", True, 3.0) for i in range(25)]
    dump = [(60 + i, f"b{i}", False, 1.0) for i in range(25)]
    m = _market(pump + dump)
    log = m.token("TOK")
    out = moonshot_outcome(log, T0 + 2, SPEC, data_end=T0 + 10_000)
    assert out is not None and not out.censored
    assert out.peak_multiple > 5, "the executable peak captures the pump"
    assert 1.0 < out.ladder_multiple < out.peak_multiple, "ladder banks part of the run, not the top"
    assert out.final_multiple < 1.0 and 0 < out.time_to_peak < 60
    # the horizon is not over and the token traded recently → only a lower bound is known
    live = moonshot_outcome(log, T0 + 2, SPEC, data_end=T0 + 200)
    assert live is not None and live.censored
    # long quiet before the end of the data → its run is over, even with the horizon incomplete
    quiet = moonshot_outcome(log, T0 + 2, SPEC.model_copy(update={"resolve_idle_seconds": 100}), T0 + 200)
    assert quiet is not None and not quiet.censored
    assert moonshot_outcome(log, T0 + 300, SPEC, data_end=T0 + 300) is None


def test_ladder_payoff_function_and_spec_validation() -> None:
    spec = MoonshotSpec()  # ladder 2/10/100/1000 selling 35/15/15/15 %, trail 60 %, stop 50 %
    got = ladder_payoff(np.array([0.3, 1.5, 3.0, 150.0]), spec)
    np.testing.assert_allclose(got, [0.3, 0.5, 0.7 + 0.65 * 1.2, 0.7 + 1.5 + 15 + 0.35 * 60])
    assert (np.diff(ladder_payoff(np.geomspace(2, 1e4, 50), spec)) >= -1e-12).all()
    with pytest.raises(ValidationError):
        MoonshotSpec(ladder=[2, 10], ladder_fractions=[0.5])
    with pytest.raises(ValidationError):
        MoonshotSpec(ladder=[10, 2], ladder_fractions=[0.2, 0.2])
    with pytest.raises(ValidationError):
        MoonshotSpec(ladder=[2, 10], ladder_fractions=[0.7, 0.7])


# ---------------------------------------------------------------- tail model
def _tail_data(n_tokens: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_tokens), 4)
    x = rng.normal(size=(n_tokens, 2))[groups].astype(np.float32)
    s = np.where(x[:, 1] > 0, 1.2, 0.25)  # heavy tail when x1 > 0
    u = rng.uniform(1e-6, 1 - 1e-6, size=len(x))
    log_m = 0.3 * x[:, 0] + s * np.log(u / (1 - u))
    return x, np.exp(log_m), groups, np.arange(len(x), dtype=np.float64)


def test_tail_model_learns_heavy_tails_and_roundtrips(tmp_path: Path) -> None:
    x, m, groups, ts = _tail_data(600, 0)
    model = TailModel(2, MoonshotSpec(), members=2, hidden=32)
    rep = model.fit(x, m, np.zeros(len(m), dtype=bool), ts, groups, epochs=60)
    assert rep["tokens"] == 600 and np.isfinite(rep["validation_nll"])
    probe = np.array([[0.0, 1.0], [0.0, -1.0]], dtype=np.float32)
    sf = model.survival(probe, [10.0, 100.0])
    assert sf[0, 0] > 3 * sf[1, 0] and sf[0, 1] > sf[1, 1], "heavier tail where x1 > 0"
    pred = model.predict(probe)
    assert (np.diff(pred.survival, axis=1) <= 1e-9).all(), "P(M ≥ k) decreases in k"
    assert pred.tail_index[0] < pred.tail_index[1]
    assert (pred.lottery_kelly >= 0).all() and (pred.lottery_kelly <= model.max_kelly).all()
    x_t, m_t, _, _ = _tail_data(200, 1)
    marginal = TailModel(1, members=1, hidden=32)
    marginal.fit(np.ones((len(m), 1)), m, np.zeros(len(m), dtype=bool), ts, groups, epochs=60)
    cens = np.zeros(len(m_t), dtype=bool)
    assert model.nll(x_t, m_t, cens).mean() < marginal.nll(np.ones((len(m_t), 1)), m_t, cens).mean()
    model.save(tmp_path / "tail")
    back = TailModel.load(tmp_path / "tail")
    np.testing.assert_allclose(back.predict(probe).survival, pred.survival, rtol=1e-6)
    assert back.feature_names == model.feature_names and back.inputs == "raw"
    # crafted extreme inputs are clamped to the training range and flagged
    crafted = np.array([[0.0, 1e6]], dtype=np.float32)
    clamped = np.array([[0.0, float(model.hi[1])]], dtype=np.float32)
    np.testing.assert_allclose(model.survival(crafted, [10.0]), model.survival(clamped, [10.0]))
    assert model.out_of_range_share(crafted)[0] == 0.5 and model.out_of_range_share(probe).max() == 0.0
    assert set(rep["calibration_ratio"]) == {"2x", "5x", "10x", "100x", "1000x"}
    np.testing.assert_allclose(back.calibration, model.calibration)


def test_calibration_shrinks_an_overconfident_tail() -> None:
    x, m, groups, ts = _tail_data(400, 3)
    model = TailModel(2, members=1, hidden=16)
    model.fit(x, m, np.zeros(len(m), dtype=bool), ts, groups, epochs=5)
    probe = x[:50]
    raw = model.survival(probe, [10.0], calibrated=False)[:, 0]
    heavy = np.full(len(m), 1e4)  # held-out outcomes far above the model: ratio must go up
    model.calibration = model._fit_calibration(
        x, np.full(len(m), 0.5), np.zeros(len(m), dtype=bool), np.ones(len(m))
    )
    assert (model.calibration < 1).all(), "no held-out token ever reached 2x → every level shrinks"
    assert (model.survival(probe, [10.0])[:, 0] <= raw + 1e-12).all()
    model.calibration = model._fit_calibration(x, heavy, np.zeros(len(m), dtype=bool), np.ones(len(m)))
    assert (model.calibration > 1).any()
    sf = model.survival(probe, [2.0, 5.0, 10.0, 100.0, 1000.0])
    assert (np.diff(sf, axis=1) <= 1e-12).all() and (sf <= 1).all()


def test_manipulation_guard_only_lowers_trust() -> None:
    from nardis_neural.solana.moonshot.guard import assess_manipulation

    clean = dict.fromkeys(CURRENT_FEATURES, 0.0) | {
        "mint_authority_revoked": 1.0,
        "freeze_authority_revoked": 1.0,
        "lp_burned_fraction": 1.0,
    }
    ok = assess_manipulation(clean, 0.05, 0.2, 0.02, 0.0)
    assert ok.trust > 0.9 and ok.vetoes == []
    worse = [
        {"bot_share_60s": 0.6},
        {"bundle_share": 0.1},
        {"creator_cluster_share": 0.2},
        {"top10_share": 0.6},
        {"dev_sold_fraction": 0.5},
        {"lp_burned_fraction": 0.0},
    ]
    for change in worse:
        v = assess_manipulation(clean | change, 0.05, 0.2, 0.02, 0.0)
        assert v.trust < ok.trust, change
    assert assess_manipulation(clean, 0.05, 3.0, 0.02, 0.0).trust < ok.trust, "out of distribution"
    assert assess_manipulation(clean, 0.05, 0.2, 0.3, 0.0).trust < ok.trust, "ensemble disagreement"
    trap = clean | {"mint_authority_revoked": 0.0, "bundle_share": 0.3, "bot_share_60s": 0.9}
    bad = assess_manipulation(trap, 0.7, 5.0, 0.02, 0.5)
    assert len(bad.vetoes) == 6 and bad.trust < 0.01


def test_censoring_is_not_mistaken_for_the_outcome() -> None:
    x, m, groups, ts = _tail_data(500, 2)
    cap = 3.0  # everything above 3x is only known to be ≥ 3x
    observed = np.minimum(m, cap)
    cens = m > cap
    aware = TailModel(2, members=2, hidden=32)
    aware.fit(x, observed, cens, ts, groups, epochs=60)
    naive = TailModel(2, members=2, hidden=32)
    naive.fit(x, observed, np.zeros(len(m), dtype=bool), ts, groups, epochs=60)
    heavy = np.array([[0.0, 1.0]], dtype=np.float32)
    truth = float((m[x[:, 1] > 0] >= 10).mean())
    p_aware = float(aware.survival(heavy, [10.0])[0, 0])
    p_naive = float(naive.survival(heavy, [10.0])[0, 0])
    assert abs(p_aware - truth) < abs(p_naive - truth), (p_aware, p_naive, truth)


# ---------------------------------------------------------------- simulator + brain
@pytest.mark.slow
def test_runners_and_moonshot_brain_integration(tmp_path: Path) -> None:
    weights = dict(MARKET_PRESETS["degen"]) | {"runner": 0.15}
    hist, arch = simulate_launches(
        LaunchSimSpec(n_tokens=26, seed=4, n_retail=250, duration_seconds=2 * 3600, archetype_weights=weights)
    )
    market = SolanaMarket()
    market.ingest_many(hist.sorted())
    runners = [m for m, a in arch.items() if a == "runner"]
    assert runners, "the degen preset with extra runners produces some"
    for mint in runners:
        r = market.token(mint).reserves
        price = np.asarray(r["sol_reserve"]) / np.asarray(r["token_reserve"])
        assert price.max() / price[0] > 50, "runners compound far beyond ordinary pumps"

    cfg = SolanaConfig(sample_interval_seconds=15.0, graph_top_k=6)
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    SolanaBrain.bootstrap(tmp_path / "ws", hist, cfg, base, device="cpu")
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    assert brain.moonshot is None
    report = brain.fit_moonshot(MoonshotSpec(horizon_seconds=3600), test_fraction=0.3, archetypes=arch)
    assert report["rows"]["train"] > 0 and report["rows"]["test"] > 0
    assert {"calibration", "portfolio", "baselines", "verdict", "threshold_sensitivity", "runners"} <= set(
        report
    )
    assert np.isfinite(report["test_nll"]) and np.isfinite(report["test_nll_marginal"])
    for f in ("tail.json", "tail.pt", "research.json", "REPORT.md"):
        assert (tmp_path / "ws" / "moonshot" / f).exists()
    reloaded = SolanaBrain(tmp_path / "ws", device="cpu")
    assert reloaded.moonshot is not None
    a = reloaded.assess(runners[0])
    assert {"p_ge_2x", "p_ge_1000x", "expected_multiple", "lottery_kelly", "in_entry_window"} <= set(
        a.moonshot
    )
    assert 0 <= a.moonshot["p_ge_1000x"] <= a.moonshot["p_ge_2x"] <= 1
    assert 0 <= a.moonshot["trust"] <= 1 and a.moonshot["chase_rank"] == 1
    assert a.moonshot["chase_score"] == 0 or not a.moonshot["vetoed"]
    ranking = reloaded.moonshot_ranking(max_idle_seconds=1e9, include_vetoed=True)
    scores = [r.moonshot["chase_score"] for r in ranking]
    assert scores == sorted(scores, reverse=True)
    # the neural stack as tail-model inputs: walk-forward OOF forecasts, risk and raw features
    neural = reloaded.fit_moonshot(MoonshotSpec(horizon_seconds=3600), inputs="neural", n_folds=2)
    assert neural["inputs"] == "neural" and neural["rows"]["test"] > 0
    assert reloaded.moonshot is not None and reloaded.moonshot.inputs == "neural"
    assert len(reloaded.moonshot.feature_names) > len(CURRENT_FEATURES)
    b = SolanaBrain(tmp_path / "ws", device="cpu").assess(runners[0])
    assert 0 <= b.moonshot["p_ge_1000x"] <= b.moonshot["p_ge_2x"] <= 1
