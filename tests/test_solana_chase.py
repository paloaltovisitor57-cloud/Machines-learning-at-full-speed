import pytest

from nardis_neural.solana.chase import CHASE_TARGETS, break_even, chase_profile
from nardis_neural.solana.metalabel import MetaLearner, TradeOutcome, TradeProposal
from nardis_neural.solana.moonshot import MoonshotSpec


def test_the_chase_targets_are_fixed() -> None:
    assert CHASE_TARGETS == (2.0, 5.0, 10.0, 100.0, 1000.0)
    spec = MoonshotSpec(levels=[3.0, 10.0])  # a configuration cannot drop the chase
    assert set(CHASE_TARGETS) <= set(spec.levels) and 3.0 in spec.levels
    assert spec.levels == sorted(spec.levels)


def test_break_even_probabilities() -> None:
    # p·k + (1 − p)·L = 1 at the break-even
    for k in CHASE_TARGETS:
        p = break_even(k, 0.7)
        assert p * k + (1 - p) * 0.7 == pytest.approx(1.0)
    assert break_even(10.0) == pytest.approx(0.3 / 9.3)
    with pytest.raises(ValueError):
        break_even(1.0)


def test_chase_target_is_the_craziest_proven_positive_ev_multiple() -> None:
    p = {2.0: 0.40, 5.0: 0.12, 10.0: 0.05, 100.0: 0.002, 1000.0: 0.0001}
    prof = chase_profile(p)
    assert prof["chase_target"] == 10.0  # 100x: 0.2 % < 0.30 % break-even
    assert prof["edge_10x"] == pytest.approx(0.05 / break_even(10.0))
    assert prof["edge_100x"] < 1.0 < prof["edge_10x"]
    # a 1000x edge on paper but with no real hits behind it is not a chase target
    lucky = p | {100.0: 0.01, 1000.0: 0.005}
    assert chase_profile(lucky)["chase_target"] == 1000.0
    assert chase_profile(lucky, hits={2.0: 50, 5.0: 20, 10.0: 9, 100.0: 1, 1000.0: 0})["chase_target"] == 10.0
    # probabilities are made monotone: reaching 10x implies reaching 5x
    assert chase_profile({2.0: 0.1, 5.0: 0.3})["p_5x"] == pytest.approx(0.1)


def test_every_trade_advice_carries_the_chase() -> None:
    m = MetaLearner()
    a = m.advise(TradeProposal("a", "mint", 0.0, {"s": 1.0}))
    for k in CHASE_TARGETS:
        assert f"p_{k:g}x" in a.chase and f"edge_{k:g}x" in a.chase and f"break_even_{k:g}x" in a.chase
    assert a.chase["chase_target"] == 0.0  # nothing is proven before any trade has settled
    m.settle(TradeOutcome("a", 1.0, 12.0, 15.0))
    b = m.advise(TradeProposal("b", "mint", 2.0, {"s": 1.0}), record=False)
    assert b.chase["proven_10x"] == 0.0  # one hit is not proof
