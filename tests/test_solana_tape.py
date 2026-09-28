"""Tape Transformer: causal tape extraction, discrete-time collapse targets, the network's
masking invariances, learning from wallet identity, persistence and brain integration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from nardis_neural.solana import LaunchSimSpec, SolanaBrain, SolanaConfig, SolanaMarket, simulate_launches
from nardis_neural.solana.config import CURRENT_FEATURES
from nardis_neural.solana.events import Swap
from nardis_neural.solana.moonshot import MoonshotSpec
from nardis_neural.solana.simulator import MARKET_PRESETS
from nardis_neural.solana.tape import (
    TRADE_FEATURES,
    TapeModel,
    TapeNet,
    TapeSpec,
    extract_tape,
    hazard_targets,
    replay_tapes,
    wallet_bucket,
)
from tests.conftest import make_tiny_config
from tests.test_solana_edge import T0, _market

SPEC = TapeSpec(max_trades=8, wallet_buckets=1024)


def test_tape_is_causal_left_padded_and_stable() -> None:
    m = _market([(1, "a", True, 1.0), (3, "b", True, 2.0), (5, "a", False, 0.5)])
    log = m.token("TOK")
    tape = extract_tape(log, m.wallets, T0 + 4, SPEC)
    assert tape.mask.tolist() == [False] * 6 + [True, True], "two trades at or before now, most recent last"
    f = {name: tape.x[:, i] for i, name in enumerate(TRADE_FEATURES)}
    assert f["is_buy"][-2:].tolist() == [1.0, 1.0] and f["price_vs_now"][-1] == pytest.approx(0.0)
    assert f["first_trade_in_token"][-2:].tolist() == [1.0, 1.0]
    assert tape.wallets[-1] == wallet_bucket("b", 1024) and tape.wallets[0] == 0
    assert 1 <= wallet_bucket("b", 1024) < 1024 and wallet_bucket("b", 1024) == wallet_bucket("b", 1024)
    later = extract_tape(log, m.wallets, T0 + 10, SPEC)
    assert later.mask.sum() == 3 and later.x[-1, TRADE_FEATURES.index("is_buy")] == 0.0
    assert later.x[-1, TRADE_FEATURES.index("first_trade_in_token")] == 0.0, "a's second trade"
    assert not extract_tape(log, m.wallets, T0 + 0.5, SPEC).mask.any()


def test_replay_matches_a_past_only_market() -> None:
    store, _ = simulate_launches(LaunchSimSpec(n_tokens=6, seed=3, n_retail=120, duration_seconds=1800))
    events = store.sorted()
    mint = next(e.mint for e in events if isinstance(e, Swap))
    t = events[len(events) // 2].t + 0.5
    x, w, mask = replay_tapes(store, SolanaConfig(), SPEC, [(mint, t)])
    past = SolanaMarket(SolanaConfig())
    past.ingest_many(e for e in events if e.t < t)
    past.advance(t)
    ref = extract_tape(past.token(mint), past.wallets, t, SPEC)
    np.testing.assert_allclose(x[0], ref.x, atol=1e-6)
    assert (w[0] == ref.wallets).all() and (mask[0] == ref.mask).all()


def test_hazard_targets() -> None:
    bins = (60.0, 300.0, 900.0)
    ct = np.array([30.0, 400.0, np.nan, np.nan, 5000.0])
    ob = np.array([7200.0, 7200.0, 7200.0, 350.0, 7200.0])
    ev, sv = hazard_targets(ct, ob, bins)
    assert ev.tolist() == [[1, 0, 0], [0, 0, 1], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    assert sv.tolist() == [[0, 0, 0], [1, 1, 0], [1, 1, 1], [1, 1, 0], [1, 1, 1]]


def test_network_ignores_padding_content() -> None:
    torch.manual_seed(0)
    net = TapeNet(len(TRADE_FEATURES), 5, 64, 10, d=16, heads=2, layers=1).eval()
    x = torch.randn(2, 10, len(TRADE_FEATURES))
    w = torch.randint(1, 64, (2, 10))
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 6:] = True
    cur = torch.randn(2, 5)
    base = net(x, w, mask, cur)
    x2, w2 = x.clone(), w.clone()
    x2[:, :6] = 99.0
    w2[:, :6] = 7
    for a, b in zip(base, net(x2, w2, mask, cur), strict=True):
        assert torch.allclose(a, b, atol=1e-5)
    empty = net(x, w, torch.zeros_like(mask), cur)
    assert all(torch.isfinite(t).all() for t in empty), "an empty tape is well defined"


def _planted(n_tokens: int, seed: int) -> tuple[np.ndarray, ...]:
    """Tapes where a 'smart' wallet's presence means a fat tail and a flagged trade means a crash."""
    rng = np.random.default_rng(seed)
    n = n_tokens * 3
    t, f = SPEC.max_trades, len(TRADE_FEATURES)
    tx = rng.normal(size=(n, t, f)).astype(np.float32)
    tw = rng.integers(2, 1024, size=(n, t))
    tm = np.ones((n, t), dtype=bool)
    smart = rng.random(n) < 0.3
    tw[smart, -1] = 1  # the smart wallet's bucket
    crash = rng.random(n) < 0.4
    tx[crash, -1, TRADE_FEATURES.index("signed_flow")] = -6.0
    s = np.where(smart, 1.3, 0.25)
    u = rng.uniform(1e-6, 1 - 1e-6, n)
    peak = np.exp(s * np.log(u / (1 - u)))
    collapse = np.where(crash, rng.uniform(10, 50, n), np.nan)
    cur = rng.normal(size=(n, 4)).astype(np.float32)
    groups = np.repeat(np.arange(n_tokens), 3)
    return tx, tw, tm, cur, peak, collapse, groups, smart, crash


def test_tape_model_learns_wallets_and_crashes(tmp_path: Path) -> None:
    tx, tw, tm, cur, peak, collapse, groups, smart, crash = _planted(500, 0)
    model = TapeModel(4, MoonshotSpec(), SPEC, members=1, d=32, layers=1, collapse_bins=(60.0, 300.0))
    n = len(peak)
    rep = model.fit(
        tx, tw, tm, cur, peak, np.zeros(n, dtype=bool), collapse, np.full(n, 3600.0),
        np.arange(n, dtype=np.float64), groups, epochs=25,
    )  # fmt: skip
    assert rep["tokens"] == 500 and np.isfinite(rep["validation_loss"])
    pred = model.predict(tx, tw, tm, cur)
    p10 = pred.tail.survival[:, model.spec.levels.index(10.0)]
    assert p10[smart].mean() > 2 * p10[~smart].mean(), "the wallet embedding carries the signal"
    assert pred.collapse[crash, 0].mean() > pred.collapse[~crash, 0].mean() + 0.3
    assert (np.diff(pred.collapse, axis=1) >= -1e-9).all(), "collapse probability grows with the window"
    model.save(tmp_path / "tape")
    back = TapeModel.load(tmp_path / "tape")
    np.testing.assert_allclose(
        back.predict(tx[:20], tw[:20], tm[:20], cur[:20]).collapse, pred.collapse[:20], rtol=1e-5
    )


@pytest.mark.slow
def test_tape_research_and_brain_integration(tmp_path: Path) -> None:
    weights = dict(MARKET_PRESETS["degen"]) | {"runner": 0.15}
    hist, arch = simulate_launches(
        LaunchSimSpec(n_tokens=22, seed=6, n_retail=200, duration_seconds=2 * 3600, archetype_weights=weights)
    )
    cfg = SolanaConfig(sample_interval_seconds=20.0, graph_top_k=6)
    base = make_tiny_config()
    base.training.epochs = 1
    base.ensemble.size = 1
    base.ensemble.mc_dropout_samples = 0
    SolanaBrain.bootstrap(tmp_path / "ws", hist, cfg, base, device="cpu")
    brain = SolanaBrain(tmp_path / "ws", device="cpu")
    report = brain.fit_tape(
        MoonshotSpec(horizon_seconds=3600), TapeSpec(max_trades=16, wallet_buckets=4096),
        test_fraction=0.3, members=1, epochs=2, archetypes=arch,
    )  # fmt: skip
    assert {"test_nll", "calibration", "collapse", "portfolio", "verdict"} <= set(report)
    assert np.isfinite(report["test_nll"]["tape"]) and report["collapse"]
    for f in ("tape.json", "tape.pt", "research.json", "REPORT.md"):
        assert (tmp_path / "ws" / "tape" / f).exists()
    reloaded = SolanaBrain(tmp_path / "ws", device="cpu")
    assert reloaded.tape_model is not None and reloaded.tape_model.n_current == len(CURRENT_FEATURES)
    a = reloaded.assess(next(iter(reloaded.market.tokens)))
    assert {"p_ge_10x", "expected_multiple", "p_collapse_1m", "p_collapse_1h"} <= set(a.tape)
    assert 0 <= a.tape["p_collapse_1m"] <= a.tape["p_collapse_1h"] <= 1
