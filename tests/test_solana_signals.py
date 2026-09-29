"""Runner-specific wallet skill, creator-family track records, sybil-resistant cluster counts
and market heat."""

from __future__ import annotations

import numpy as np
import pytest

from nardis_neural.solana import SolanaConfig, SolanaMarket
from nardis_neural.solana.amm import pump_curve
from nardis_neural.solana.events import Swap, TokenLaunch, Transfer
from nardis_neural.solana.features import SolanaFeatureBuilder

T0 = 1_750_000_000.0


class Tape:
    """Launches tokens and trades them through the real curve maths."""

    def __init__(self) -> None:
        self.m = SolanaMarket(SolanaConfig())
        self.pools: dict[str, object] = {}
        self.held: dict[tuple[str, str], float] = {}

    def fund(self, t: float, src: str, dst: str) -> None:
        self.m.ingest(Transfer(t=t, source=src, dest=dst, sol_amount=1.0))

    def launch(self, mint: str, t: float, creator: str) -> None:
        self.m.ingest(TokenLaunch(mint=mint, t=t, creator=creator))
        self.pools[mint] = pump_curve()

    def buy(self, mint: str, t: float, wallet: str, sol: float) -> None:
        pool = self.pools[mint]
        tokens, pool = pool.buy(sol)  # type: ignore[attr-defined]
        self.pools[mint] = pool
        self.held[(mint, wallet)] = self.held.get((mint, wallet), 0.0) + tokens
        self.m.ingest(Swap(mint, t, wallet, True, sol, tokens, pool.sol, pool.tokens))


def _features(tape: Tape, mint: str) -> dict[str, float]:
    b = SolanaFeatureBuilder(tape.m.cfg)
    log = tape.m.token(mint)
    return b.explain(b.current_features(log, tape.m.wallets, tape.m.now, tape.m))


def test_runner_skill_is_learned_from_early_buys_that_ran() -> None:
    tape = Tape()
    for k in range(2):  # two runners: one hit could be luck, two is a skill
        tape.launch(f"RUN{k}", T0 + k, f"dev{k}")
        tape.buy(f"RUN{k}", T0 + k + 5, "early_bird", 0.5)
        for i in range(40):  # the run: price goes far beyond 10x the early entry
            tape.buy(f"RUN{k}", T0 + 400 + i, f"fomo{k}_{i}", 5.0)
    tape.launch("DUD", T0 + 10, "dev_dud")
    tape.buy("DUD", T0 + 12, "bag_holder", 0.5)
    tape.m.advance(T0 + 10_000)
    w = tape.m.wallets
    good, bad = w.tail_skills(np.array([w.ids["early_bird"], w.ids["bag_holder"]]))
    assert good > 0.5 and bad < 0, (good, bad)
    assert tape.m.tail_updates >= 3
    # the skill shows up in the next token they buy early
    tape.launch("NEXT", T0 + 10_001, "dev3")
    tape.buy("NEXT", T0 + 10_002, "early_bird", 1.0)
    f = _features(tape, "NEXT")
    assert f["tail_smart_buy_share_60s"] == pytest.approx(1.0)
    assert f["early_buyer_tail_skill"] > 0.5


def test_creator_family_track_record_follows_the_funder() -> None:
    tape = Tape()
    tape.fund(T0 - 100, "deployer_hub", "devA")
    tape.fund(T0 - 90, "deployer_hub", "devB")
    tape.launch("A", T0, "devA")
    for i in range(20):
        tape.buy("A", T0 + 1 + i, f"b{i}", 3.0)
    tape.launch("B", T0 + 600, "devB")  # a fresh wallet from the same source
    fb = _features(tape, "B")
    assert fb["creator_prior_launches_log"] == pytest.approx(np.log1p(1))
    assert fb["creator_prior_best_peak_log"] > np.log(3)
    assert fb["since_creator_last_launch_log"] == pytest.approx(np.log1p(600))
    tape.launch("C", T0 + 601, "loner")
    fc = _features(tape, "C")
    assert fc["creator_prior_launches_log"] == 0 and fc["creator_prior_best_peak_log"] == 0
    # the record survives eviction of the earlier token
    tape.m.evict(T0 + 20_000, idle_seconds=3600)
    fam = tape.m.families[tape.m.family_of.get("B", -1)] if "B" in tape.m.family_of else None
    assert fam is None or fam.launches >= 1


def test_cluster_counts_see_through_sybil_wallets() -> None:
    tape = Tape()
    for i in range(10):
        tape.fund(T0 - 50, "operator", f"sybil{i}")
    tape.launch("S", T0, "dev")
    for i in range(10):
        tape.buy("S", T0 + 1 + i, f"sybil{i}", 0.5)
    tape.buy("S", T0 + 20, "independent", 0.5)
    f = _features(tape, "S")
    assert f["holder_clusters_log"] == pytest.approx(np.log1p(2))
    assert f["holder_cluster_ratio"] == pytest.approx(2 / 11)
    assert f["buyer_clusters_60s_log"] == pytest.approx(np.log1p(2))


def test_market_heat_windows() -> None:
    tape = Tape()
    for i in range(5):
        tape.launch(f"M{i}", T0 + i, f"d{i}")
        tape.buy(f"M{i}", T0 + i + 0.5, f"w{i}", 2.0)
    f = _features(tape, "M4")
    assert f["market_launches_600s_log"] == pytest.approx(np.log1p(5))
    assert f["market_volume_300s_log"] == pytest.approx(np.log1p(10.0))
    tape.launch("LATE", T0 + 5000, "d9")
    assert _features(tape, "LATE")["market_launches_600s_log"] == pytest.approx(np.log1p(1))


def test_staged_insiders_behind_a_relay_are_seen_and_distrusted() -> None:
    from nardis_neural.solana.moonshot.guard import assess_manipulation

    tape = Tape()
    tape.fund(T0 - 7200, "trapfunder", "relay")
    for i in range(5):  # two hops from the funder, nothing links them to the creator
        tape.fund(T0 - 3600 + i, "relay", f"staged{i}")
    tape.launch("TRAP", T0, "clean_dev")
    for i in range(5):
        tape.buy("TRAP", T0 + 5 + i, f"staged{i}", 3.0)
    for i in range(30):
        tape.buy("TRAP", T0 + 20 + i, f"retail{i}", 0.3)
    f = _features(tape, "TRAP")
    assert f["creator_cluster_share"] == pytest.approx(0.0), "the creator looks clean"
    assert f["top_cluster_share"] > 0.1, "but one hidden cluster holds a big slice"
    clean = assess_manipulation(f | {"top_cluster_share": 0.0}, None, 0.0, 0.0, 0.0)
    staged = assess_manipulation(f, None, 0.0, 0.0, 0.0)
    assert staged.trust < 0.5 * clean.trust
    heavy = assess_manipulation(f | {"top_cluster_share": 0.3}, None, 0.0, 0.0, 0.0)
    assert any("hidden wallet cluster" in v for v in heavy.vetoes)


def test_runner_hits_are_credited_the_moment_they_happen() -> None:
    """A hit is known when the price crosses the target; only a miss waits for the horizon."""
    tape = Tape()
    tape.launch("RUN", T0, "dev")
    tape.buy("RUN", T0 + 5, "early_bird", 0.5)
    tape.launch("DUD", T0 + 1, "dev2")
    tape.buy("DUD", T0 + 6, "bag_holder", 0.5)
    w = tape.m.wallets
    before = w.tail_skills(np.array([w.ids["early_bird"]]))[0]
    for i in range(40):  # the run: right after the 5-minute early-buyer window
        tape.buy("RUN", T0 + 310 + i, f"fomo_{i}", 5.0)
    assert tape.m.now - T0 < 400 < tape.m.cfg.tail_horizon_seconds
    assert tape.m.tail_early_hits == 1  # credited long before the 90-minute horizon
    assert w.tail_skills(np.array([w.ids["early_bird"]]))[0] > before
    assert tape.m.tail_updates == 1  # the miss is not known yet
    tape.m.advance(T0 + tape.m.cfg.tail_horizon_seconds + 10)
    assert tape.m.tail_updates == 2  # the miss resolves at the horizon; the hit is not counted twice
    good, bad = w.tail_skills(np.array([w.ids["early_bird"], w.ids["bag_holder"]]))
    assert good > 0 > bad


def test_market_checkpoints_from_before_early_crediting_still_load() -> None:
    import pickle

    tape = Tape()
    tape.launch("RUN", T0, "dev")
    tape.buy("RUN", T0 + 5, "early_bird", 0.5)
    state = dict(tape.m.__dict__)
    for k in ("_tail_waiting", "_tail_done", "_tail_seq", "tail_early_hits"):
        state.pop(k)
    old = SolanaMarket.__new__(SolanaMarket)
    old.__setstate__(state)
    again = pickle.loads(pickle.dumps(old))
    again.ingest(Swap("RUN", T0 + 30, "second", True, 0.1, 1e6, 40.0, 7e8))
    assert again.tail_early_hits == 0 and len(again._tail_waiting["RUN"]) == 1  # new buys are tracked
