"""Edge engine: executable triple-barrier labels, backtester, meta-labeling model, walk-forward
research and its integration into SolanaBrain."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, SolanaMarket, simulate_launches
from nardis_neural.solana.amm import pump_curve, round_trip_cost
from nardis_neural.solana.edge import (
    BarrierSpec,
    Candidates,
    EdgeModel,
    baselines,
    research_markdown,
    simulate,
    trade_stats,
    triple_barrier,
)
from nardis_neural.solana.edge.backtest import select_threshold
from nardis_neural.solana.events import Swap, TokenLaunch
from tests.conftest import make_tiny_config

T0 = 1_750_000_000.0


def _market(path: Sequence[tuple[float, str, bool, float]]) -> SolanaMarket:
    """Launch at T0 then (dt, wallet, is_buy, amount) trades priced on the real curve."""
    m = SolanaMarket()
    m.ingest(TokenLaunch(mint="TOK", t=T0, creator="dev"))
    pool = pump_curve()
    held: dict[str, float] = {}
    for dt, wallet, is_buy, amount in path:
        if is_buy:
            tokens, pool = pool.buy(amount)
            held[wallet] = held.get(wallet, 0.0) + tokens
            m.ingest(Swap("TOK", T0 + dt, wallet, True, amount, tokens, pool.sol, pool.tokens))
        else:
            tokens = held[wallet] * amount
            sol, pool = pool.sell(tokens)
            held[wallet] -= tokens
            m.ingest(Swap("TOK", T0 + dt, wallet, False, sol, tokens, pool.sol, pool.tokens))
    return m


# ---------------------------------------------------------------- barriers
def test_take_profit_is_net_of_costs() -> None:
    m = _market([(1, "a", True, 1.0)] + [(10 + i, f"b{i}", True, 3.0) for i in range(8)])
    out = triple_barrier(
        m.token("TOK"),
        T0 + 2,
        BarrierSpec(take_profit=0.2, stop_loss=0.2, latency_seconds=0.5),
        data_end=T0 + 1000,
    )
    assert out is not None and out.exit_reason == "take_profit"
    assert out.net_return > 0.2, "exit fires at the first mark beyond the barrier (plus latency drift)"
    assert out.win and out.hold_seconds > 0


def test_stop_loss_latency_costs_money() -> None:
    crash = [(1, "whale", True, 20.0), (1.5, "me", True, 0.1)] + [
        (20 + i * 0.5, "whale", False, 0.25) for i in range(8)
    ]
    m = _market(crash)
    fast = triple_barrier(m.token("TOK"), T0 + 2, BarrierSpec(stop_loss=0.1, latency_seconds=0.0), T0 + 1000)
    slow = triple_barrier(m.token("TOK"), T0 + 2, BarrierSpec(stop_loss=0.1, latency_seconds=2.0), T0 + 1000)
    assert fast is not None and slow is not None
    assert fast.exit_reason == slow.exit_reason == "stop_loss"
    assert slow.net_return < fast.net_return, "latency during a crash worsens the exit"


def test_time_exit_pays_round_trip_and_short_history() -> None:
    m = _market([(1, "a", True, 2.0)])
    log = m.token("TOK")
    spec = BarrierSpec(max_hold_seconds=60, latency_seconds=1.0)
    out = triple_barrier(log, T0 + 5, spec, data_end=T0 + 1000)
    assert out is not None and out.exit_reason == "time"
    assert out.net_return == pytest.approx(-round_trip_cost(log.pool(), spec.size_sol), rel=1e-6)
    assert triple_barrier(log, T0 + 5, spec, data_end=T0 + 30) is None
    scaled = BarrierSpec(vol_scaled=True, take_profit=0.25)
    assert triple_barrier(log, T0 + 5, scaled, data_end=T0 + 1000) is not None


# ---------------------------------------------------------------- backtester
def test_simulator_enforces_position_constraints() -> None:
    c = Candidates(
        t=np.array([0.0, 1.0, 2.0, 3.0, 50.0]),
        mint=np.array(["A", "A", "B", "C", "A"]),
        net=np.array([0.1, 0.2, -0.1, 0.3, 0.05]),
        exit_time=np.array([10.0, 11.0, 12.0, 13.0, 60.0]),
    )
    trades = simulate(c, np.ones(5, dtype=bool), max_positions=2)
    assert trades.tolist() == [0, 2, 4], "A re-entry blocked while open; C blocked by max positions"
    st = trade_stats(c, trades)
    assert st["trades"] == 3 and st["hit_rate"] == pytest.approx(2 / 3)
    assert st["total_pnl_sol"] == pytest.approx(0.05) and st["max_drawdown_sol"] == pytest.approx(0.1)
    assert trade_stats(c, np.zeros(0, dtype=np.int64)) == {"trades": 0.0}


def test_threshold_selection_and_baselines() -> None:
    rng = np.random.default_rng(0)
    n = 600
    score = rng.normal(size=n)
    net = 0.05 * score + rng.normal(scale=0.05, size=n) - 0.03
    c = Candidates(
        np.arange(n, dtype=np.float64),
        np.array([f"m{i}" for i in range(n)]),
        net,
        np.arange(n, dtype=np.float64) + 0.5,
    )
    thr, curve = select_threshold(c, score, max_positions=5, min_trades=20)
    assert thr > np.median(score) and curve
    picked = simulate(c, score >= thr, 5)
    assert c.net[picked].mean() > 0 > c.net.mean()
    base = baselines(c, len(picked), momentum=score + rng.normal(size=n) * 5, max_positions=5)
    assert {"all_candidates", "random", "momentum"} <= set(base)
    assert base["random"]["mean_net"] < c.net[picked].mean()


# ---------------------------------------------------------------- meta-labeling model
def test_edge_model_learns_and_roundtrips(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 3000
    x = rng.normal(size=(n, 6)).astype(np.float32)
    net = 0.2 * x[:, 0] - 0.1 * x[:, 1] + rng.normal(scale=0.1, size=n) - 0.05
    ts = np.arange(n, dtype=float)
    model = EdgeModel(6, members=3, feature_names=[f"f{i}" for i in range(6)])
    rep = model.fit(x, net, ts, epochs=60)
    assert rep["win_auc"] > 0.8 and rep["net_ic"] > 0.6
    assert rep["top_decile_realized_net"] > rep["all_realized_net"]
    pred = model.predict(x[:100])
    assert np.allclose(pred.edge_score, pred.expected_net - model.lcb_lambda * pred.uncertainty)
    assert (pred.uncertainty >= 0).all()
    assert ((pred.kelly >= 0) & (pred.kelly <= model.max_kelly)).all()
    model.save(tmp_path / "edge")
    back = EdgeModel.load(tmp_path / "edge")
    np.testing.assert_allclose(back.predict(x[:100]).edge_score, pred.edge_score)
    assert back.feature_names == model.feature_names
    assert set(rep["component_ic"]) == {"mlp", "ridge", "trees"}
    comps = model.predict_components(x[:100])
    assert comps.shape == (3, 100)
    np.testing.assert_allclose(pred.expected_net, comps.mean(axis=0), rtol=1e-6)


# ---------------------------------------------------------------- research + brain
def test_walk_forward_edge_research_and_brain_integration(tmp_path: Path) -> None:
    hist, _ = simulate_launches(LaunchSimSpec(n_tokens=20, seed=13, n_retail=250, duration_seconds=3 * 3600))
    cfg = SolanaConfig(sample_interval_seconds=15.0, graph_top_k=6)
    base = make_tiny_config()
    base.training.epochs = 2
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    SolanaBrain.bootstrap(tmp_path / "ws", hist, cfg, base, device="cpu")
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    assert brain.edge is None
    report = brain.fit_edge(BarrierSpec(max_hold_seconds=120), n_folds=3, max_positions=5)
    rows = report["rows"]
    assert rows["oof"] < rows["total"], "the earliest data is never predicted out-of-fold"
    assert rows["fit"] > rows["tune"] > 0 and rows["test"] > 0
    assert len(report["folds"]) == 3
    assert {"test", "test_baselines", "threshold", "verdict", "edge_model_validation"} <= set(report)
    assert "Edge research report" in research_markdown(report)
    for f in ("edge.json", "members.pt", "research.json", "REPORT.md"):
        assert (tmp_path / "ws" / "edge" / f).exists()
    reloaded = SolanaBrain(tmp_path / "ws", device="cpu")
    assert reloaded.edge is not None
    mint = next(iter(reloaded.market.tokens))
    a = reloaded.assess(mint)
    assert set(a.edge) == {
        "p_win",
        "expected_net",
        "uncertainty",
        "edge_score",
        "kelly_fraction",
        "threshold",
        "above_threshold",
    }
    assert 0 <= a.edge["p_win"] <= 1 and a.edge["kelly_fraction"] >= 0


def test_exported_trees_match_sklearn(tmp_path: Path) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor

    from nardis_neural.solana.edge.trees import TreeEnsemble

    rng = np.random.default_rng(3)
    x = rng.normal(size=(1500, 5))
    y = 2 * x[:, 0] + np.sin(3 * x[:, 1]) + rng.normal(scale=0.1, size=1500)
    x[::13, 2] = np.nan
    model = HistGradientBoostingRegressor(max_depth=3, max_iter=60, random_state=0).fit(x, y)
    trees = TreeEnsemble.export(model)
    np.testing.assert_allclose(trees.predict(x), model.predict(x), atol=1e-12)
    trees.save(tmp_path / "t.npz")
    np.testing.assert_allclose(TreeEnsemble.load(tmp_path / "t.npz").predict(x), model.predict(x), atol=1e-12)
