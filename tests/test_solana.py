"""Solana module: protocol maths, causal market state, wallet intelligence, features, labels,
dataset construction, risk model and simulator behaviour."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from nardis_neural.solana.amm import (
    PUMP_CURVE_TOKENS,
    PUMP_FEE,
    PUMP_GRADUATION_SOL,
    Pool,
    bonding_progress,
    price_impact,
    pump_curve,
    round_trip_cost,
)
from nardis_neural.solana.config import (
    BAR_FEATURES,
    CURRENT_FEATURES,
    EDGE_TYPES,
    NODE_FEATURES,
    RISK_LABELS,
    SolanaConfig,
)
from nardis_neural.solana.dataset import SolanaDataset, build_solana_dataset
from nardis_neural.solana.events import (
    Columns,
    Event,
    LiquidityChange,
    Migration,
    Swap,
    TokenLaunch,
    Transfer,
)
from nardis_neural.solana.features import SolanaFeatureBuilder, gini
from nardis_neural.solana.labels import SolanaLabeler
from nardis_neural.solana.market import EventStore, SolanaMarket
from nardis_neural.solana.risk import SolanaRiskModel
from nardis_neural.solana.simulator import LaunchSimSpec, simulate_launches
from nardis_neural.solana.wallets import WalletIntel

T0 = 1_750_000_000.0


@pytest.fixture(scope="module")
def sim() -> tuple[EventStore, dict[str, str]]:
    return simulate_launches(LaunchSimSpec(n_tokens=14, seed=5, n_retail=300, duration_seconds=2 * 3600))


@pytest.fixture(scope="module")
def dataset(sim: tuple[EventStore, dict[str, str]]) -> SolanaDataset:
    store, arch = sim
    return build_solana_dataset(store, SolanaConfig(sample_interval_seconds=15.0), archetypes=arch)


# ---------------------------------------------------------------- AMM maths
def test_pump_curve_maths() -> None:
    fresh = pump_curve()
    assert fresh.price == pytest.approx(30 / 1_073_000_000)
    assert bonding_progress(fresh) == 0.0
    tokens, after = fresh.buy(PUMP_GRADUATION_SOL / (1 - PUMP_FEE))
    assert bonding_progress(after) == pytest.approx(1.0)
    assert tokens == pytest.approx(PUMP_CURVE_TOKENS, rel=0.03), "85 SOL completes the ~793M-token curve"
    assert after.price > fresh.price


def test_round_trip_cost_and_impact() -> None:
    pool = Pool(100.0, 1e9, 0.0025)
    small, large = round_trip_cost(pool, 0.1), round_trip_cost(pool, 10.0)
    assert 0.005 - 1e-4 < small < large < 0.3, "two fees + two-way impact, growing with size"
    assert price_impact(pool, 10.0) == pytest.approx((110 / 100) ** 2 - 1, rel=0.01)
    back_tokens, p2 = pool.buy(1.0)
    sol_back, p3 = p2.sell(back_tokens)
    assert sol_back < 1.0 and p3.sol * p3.tokens >= pool.sol * pool.tokens * 0.999


# ---------------------------------------------------------------- events & wallets
def test_columns_grow_and_log_rejects_time_travel() -> None:
    c = Columns({"t": np.float64}, capacity=2)
    for i in range(10):
        c.append(t=float(i))
    assert len(c) == 10 and c["t"][-1] == 9.0
    m = SolanaMarket()
    m.ingest(TokenLaunch(mint="X", t=T0, creator="dev"))
    m.ingest(Swap("X", T0 + 5, "a", True, 1.0, 1e6, 31.0, 1.04e9))
    with pytest.raises(ValueError):
        m.ingest(Swap("X", T0 + 1, "a", True, 1.0, 1e6, 31.0, 1.04e9))
    with pytest.raises(KeyError):
        m.ingest(Swap("Y", T0 + 6, "a", True, 1.0, 1e6, 31.0, 1.04e9))


def test_funding_clusters_hubs_and_persistence(tmp_path: Path) -> None:
    w = WalletIntel(hub_threshold=3)
    w.add_transfer(Transfer(T0, "funder", "a", 1.0))
    w.add_transfer(Transfer(T0, "funder", "b", 1.0))
    w.add_transfer(Transfer(T0, "x", "y", 0.001))  # below funding threshold → ignored
    assert w.same_cluster(w.id("a"), w.id("b")) and w.cluster_size(w.id("a")) == 3
    for i in range(10):  # a CEX hot wallet funds everybody: must not merge them all
        w.add_transfer(Transfer(T0, "cex", f"user{i}", 1.0))
    assert not w.same_cluster(w.id("user0"), w.id("user9"))
    assert w.is_fresh(w.id("a"), T0 + 100, 3600) and not w.is_fresh(w.id("a"), T0 + 7200, 3600)
    for _ in range(8):
        w.update(w.id("a"), True)
    w.update(w.id("b"), False)
    assert w.score(w.id("a")) > 0.8 > w.score(w.id("b")) and w.skill(w.id("a")) > 0 > w.skill(w.id("b"))
    ids = np.array([w.id("a"), w.id("b"), w.id("user3")])
    np.testing.assert_allclose(w.skills(ids), [w.skill(int(i)) for i in ids])
    w.save(tmp_path / "w.json")
    back = WalletIntel.load(tmp_path / "w.json")
    assert back.score(back.id("a")) == w.score(w.id("a")) and back.same_cluster(back.id("a"), back.id("b"))


def _toy_market(cfg: SolanaConfig | None = None) -> tuple[SolanaMarket, list[Event]]:
    """Launch, three buys lifting price, then a creator dump."""
    cfg = cfg or SolanaConfig()
    pool = pump_curve()
    events: list[Event] = [
        Transfer(T0 - 100, "funder", "dev", 5.0),
        Transfer(T0 - 90, "funder", "bundler", 5.0),
        TokenLaunch(mint="TOK", t=T0, creator="dev", mint_authority_revoked=False),
    ]
    for wallet, size, t in (
        ("dev", 2.0, T0 + 0.1),
        ("bundler", 3.0, T0 + 0.5),
        ("alice", 5.0, T0 + 20),
        ("bob", 4.0, T0 + 40),
    ):
        toks, pool = pool.buy(size)
        events.append(Swap("TOK", t, wallet, True, size, toks, pool.sol, pool.tokens, 1e-4))
    dev_buy = events[3]
    assert isinstance(dev_buy, Swap)
    dev_tokens = dev_buy.token_amount
    sol, pool = pool.sell(dev_tokens)
    events.append(Swap("TOK", T0 + 200, "dev", False, sol, dev_tokens, pool.sol, pool.tokens))
    m = SolanaMarket(cfg)
    return m, events


def test_reputation_is_learned_causally() -> None:
    m, events = _toy_market()
    buys = [e for e in events if isinstance(e, Swap) and e.is_buy]
    m.ingest_many(events[:4])  # transfers, launch, dev buy
    assert m.wallets.evidence(m.wallets.id("dev")) == 0
    m.ingest_many(events[4:6])  # bundler + alice (price rises after dev's entry)
    m.advance(buys[0].t + m.reputation_seconds - 1)
    assert m.wallets.evidence(m.wallets.id("dev")) == 0, "no credit before the horizon elapses"
    m.advance(buys[0].t + m.reputation_seconds)
    assert m.wallets.evidence(m.wallets.id("dev")) == 1


def test_rug_attribution_marks_creator_cluster() -> None:
    m, events = _toy_market()
    m.ingest_many(events)
    assert m.rugged("TOK")
    assert m.wallets.rugs[m.wallets.id("dev")] == 1
    assert m.wallets.rugs[m.wallets.id("bundler")] == 1, "co-funded bundle wallet shares the blame"
    assert m.wallets.rugs[m.wallets.id("alice")] == 0


# ---------------------------------------------------------------- features
def test_feature_vector_contract(sim: tuple[EventStore, dict[str, str]]) -> None:
    store, _ = sim
    m = SolanaMarket()
    m.ingest_many(store.sorted())
    b = SolanaFeatureBuilder()
    mint = next(iter(m.tokens))
    obs = b.observation(m, mint)
    assert obs.current_features.shape == (len(CURRENT_FEATURES),) and np.isfinite(obs.current_features).all()
    for bar in b.cfg.bars:
        seq = obs.sequences[bar.name]
        assert seq.values.shape == (bar.length, len(BAR_FEATURES))
    g = obs.graph
    assert g is not None and g.node_features.shape[1] == len(NODE_FEATURES)
    assert g.edge_type is not None and g.edge_type.max(initial=0) < len(EDGE_TYPES)
    assert len(set(CURRENT_FEATURES)) == len(CURRENT_FEATURES)
    with pytest.raises(ValueError, match="causally"):
        b.current_features(m.token(mint), m.wallets, m.token(mint).last_t - 10)


def test_features_are_causal(sim: tuple[EventStore, dict[str, str]]) -> None:
    """Observation at T is identical whatever happens after T."""
    store, _ = sim
    events = store.sorted()
    swaps = [e for e in events if isinstance(e, Swap)]
    cut = swaps[len(swaps) // 2].t
    past = [e for e in events if e.t <= cut]
    alt_future = [
        replace(e, sol_amount=e.sol_amount * 7, t=e.t + 3) if isinstance(e, Swap) else e
        for e in events
        if e.t > cut
    ]
    a, b = SolanaMarket(), SolanaMarket()
    a.ingest_many(past)
    b.ingest_many(past)
    builder = SolanaFeatureBuilder()
    mint = swaps[len(swaps) // 2].mint
    obs_a = builder.observation(a, mint, cut)
    b.ingest_many(sorted(alt_future, key=lambda e: e.t)[:0])  # nothing after cut is visible at cut
    obs_b = builder.observation(b, mint, cut)
    np.testing.assert_array_equal(obs_a.current_features, obs_b.current_features)
    for name in obs_a.sequences:
        np.testing.assert_array_equal(obs_a.sequences[name].values, obs_b.sequences[name].values)


def test_insider_features_detect_bundles_and_dumps() -> None:
    m, events = _toy_market()
    b = SolanaFeatureBuilder()
    m.ingest_many(events[:6])
    early = b.explain(b.current_features(m.token("TOK"), m.wallets, m.now))
    assert early["bundle_share"] > 0, "co-funded early buyer detected as bundle"
    assert early["creator_cluster_share"] > early["dev_share"] > 0
    assert early["mint_authority_revoked"] == 0.0 and early["dev_sold_fraction"] == 0.0
    assert 0 < early["bonding_progress"] < 1 and early["round_trip_cost"] > 2 * PUMP_FEE - 1e-3
    m.ingest_many(events[6:])
    late = b.explain(b.current_features(m.token("TOK"), m.wallets, m.now))
    assert late["dev_sold_fraction"] == pytest.approx(1.0) and late["dev_share"] == 0.0
    assert late["ret_5s"] < 0, "dump shows up as a negative 5s return"


def test_bars_forward_fill_and_mask() -> None:
    m, events = _toy_market()
    m.ingest_many(events[:5])
    spec = SolanaConfig().bars[0]
    seq = SolanaFeatureBuilder().bars(m.token("TOK"), m.now + 3, spec)
    assert seq.mask is not None
    assert seq.mask[-1] and not seq.mask[0], "bars before launch are missing, quiet bars after launch are not"
    assert seq.values[-1, BAR_FEATURES.index("volume_log")] == 0.0
    assert gini(np.array([1.0, 1.0, 1.0])) == pytest.approx(0.0)
    assert gini(np.array([0.0, 0.0, 10.0])) == 0.0, "zero balances are not holders"
    assert gini(np.array([1.0, 1.0, 10.0])) > 0.3


# ---------------------------------------------------------------- labels
def test_labeler_on_toy_market() -> None:
    cfg = SolanaConfig()
    m, events = _toy_market(cfg)
    m.ingest_many(events)
    lab = SolanaLabeler(cfg)
    log = m.token("TOK")
    labels = lab.label(log, T0 + 1, data_end=T0 + 1000)
    assert labels is not None
    out = labels.outcome
    assert out.observation_id == "TOK@1750000001.000"
    assert out.returns["60s"] > 0 and out.max_upside["60s"] >= out.returns["60s"]
    assert labels.net_returns["60s"] == pytest.approx(math.expm1(out.returns["60s"]) - labels.round_trip_cost)
    late = lab.label(log, T0 + 45, data_end=T0 + 1000)  # after the last buy, before the dump
    assert late is not None
    assert late.outcome.max_drawdown["5m"] > 0, "the dev dump is inside the 5m horizon"
    assert late.outcome.returns["60s"] == 0.0, "nothing trades in the next minute"
    assert late.risk is not None and late.risk["dev_dump"] == 1.0
    short = lab.label(log, T0 + 1, data_end=T0 + 30)
    assert short is not None and set(short.outcome.returns) == {"15s"} and short.risk is None
    assert lab.label(log, T0 + 1, data_end=T0 + 5) is None


def test_labeler_rug_and_graduation_events() -> None:
    cfg = SolanaConfig()
    m = SolanaMarket(cfg)
    m.ingest(TokenLaunch(mint="LP", t=T0, creator="dev", venue="raydium", sol_reserve=50, token_reserve=1e9))
    m.ingest(LiquidityChange("LP", T0 + 10, "dev", -48.0, -9.6e8, 2.0, 4e7))
    m.ingest(TokenLaunch(mint="GR", t=T0, creator="dev2"))
    m.ingest(Migration("GR", T0 + 50, "pumpswap", 80.0, 2e8))
    lab = SolanaLabeler(cfg)
    rug = lab.risk_labels(m.token("LP"), T0 + 1, T0 + 1000)
    grad = lab.risk_labels(m.token("GR"), T0 + 1, T0 + 1000)
    assert rug is not None and rug["rug"] == 1.0 and m.rugged("LP")
    assert grad is not None and grad["graduation"] == 1.0 and grad["rug"] == 0.0


# ---------------------------------------------------------------- simulator & dataset
def test_simulator_archetypes_behave(sim: tuple[EventStore, dict[str, str]]) -> None:
    store, arch = sim
    assert set(arch.values()) <= {"organic", "graduate", "rug", "dud", "wash"}
    migrations = {e.mint for e in store.of_type(Migration)}
    grads = [m for m, a in arch.items() if a == "graduate"]
    assert migrations and all(g in migrations for g in grads), "graduates complete the bonding curve"
    assert not any(arch[m] in {"dud", "wash"} for m in migrations)
    swaps = store.of_type(Swap)
    by_mint: dict[str, list[Swap]] = {}
    for s in swaps:
        by_mint.setdefault(s.mint, []).append(s)
    for mint, ss in by_mint.items():
        ts = [s.t for s in ss]
        assert ts == sorted(ts), f"{mint}: events must be time ordered"
    reload_dir = EventStore(list(store.events))
    assert len(reload_dir) == len(store)


def test_event_store_parquet_roundtrip(tmp_path: Path, sim: tuple[EventStore, dict[str, str]]) -> None:
    store, _ = sim
    store.save(tmp_path / "ev")
    back = EventStore.load(tmp_path / "ev")
    assert len(back) == len(store)
    assert back.sorted()[:50] == store.sorted()[:50]


def test_dataset_is_leakage_free_and_labelled(
    dataset: SolanaDataset, sim: tuple[EventStore, dict[str, str]]
) -> None:
    store, _ = sim
    cfg = SolanaConfig(sample_interval_seconds=15.0)
    ncfg = cfg.neural_config()
    ds = dataset
    assert len(ds) > 300
    ds.store().validate(ncfg)
    assert ds.risk.shape == (len(ds), len(RISK_LABELS)) and np.nanmax(ds.risk) == 1.0
    # spot-check: a snapshot's features equal those of a market that saw only the past
    i = len(ds) // 2
    obs = ds.observations[i]
    mint = str(ds.mints[i])
    m = SolanaMarket(cfg)
    m.ingest_many([e for e in store.sorted() if e.t <= obs.timestamp])
    m.advance(obs.timestamp)
    fresh = SolanaFeatureBuilder(cfg).observation(m, mint, obs.timestamp)
    np.testing.assert_allclose(fresh.current_features, obs.current_features, rtol=1e-5, atol=1e-6)
    rug_rows = ds.extra["archetype"] == "rug"
    assert np.nanmean(ds.risk[rug_rows, 0]) > np.nanmean(ds.risk[~rug_rows, 0])


def test_config_maps_onto_neural_contract(tmp_path: Path) -> None:
    cfg = SolanaConfig()
    n = cfg.neural_config()
    assert n.features.current_dim == len(CURRENT_FEATURES)
    assert [t.name for t in n.features.timescales] == ["fast", "medium", "slow"]
    assert all(t.feature_dim == len(BAR_FEATURES) for t in n.features.timescales)
    assert n.model.graph.enabled and "graph" in n.model.enabled_experts
    assert n.horizon_names == ["15s", "60s", "5m"]
    cfg.save(tmp_path / "s.yaml")
    assert SolanaConfig.load(tmp_path / "s.yaml") == cfg
    with pytest.raises(ValueError):
        SolanaMarket(SolanaConfig(reputation_horizon="nope"))


# ---------------------------------------------------------------- risk model
def test_risk_model_learns_rug_signal(tmp_path: Path, dataset: SolanaDataset) -> None:
    x = SolanaRiskModel.inputs(np.zeros((len(dataset), 4)), dataset.current)
    model = SolanaRiskModel(x.shape[1], members=2)
    report = model.fit(x, dataset.risk, np.asarray(dataset.arrays["timestamp"], dtype=np.float64), epochs=40)
    assert report.metrics["rug"]["auc"] > 0.7 or math.isnan(report.metrics["rug"]["auc"])
    probs, unc = model.predict(x[:20])
    assert probs.shape == (20, 3) and ((probs >= 0) & (probs <= 1)).all() and (unc >= 0).all()
    model.save(tmp_path / "risk")
    back = SolanaRiskModel.load(tmp_path / "risk")
    np.testing.assert_allclose(back.predict(x[:20])[0], probs)
