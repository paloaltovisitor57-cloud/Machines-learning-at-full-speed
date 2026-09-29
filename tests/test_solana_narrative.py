import math

import pytest

from nardis_neural.solana import SolanaConfig, SolanaMarket
from nardis_neural.solana.events import Swap, TokenLaunch
from nardis_neural.solana.narrative import NarrativeBook, name_words, normalise

T0 = 1_790_000_000.0


def test_name_words_and_normalisation() -> None:
    assert normalise("$PEPE 2.0") == "pepe20"
    assert name_words("The Official Cat Coin", "CAT") == ("cat",)
    assert name_words("Doge Wif Hat", "DWH") == ("doge", "wif", "hat", "dwh")


def test_heat_credits_the_theme_but_never_the_token_itself() -> None:
    b = NarrativeBook(half_life=1800.0)
    b.on_launch("cat1", T0, "Space Cat", "SCAT")
    b.on_peak("cat1", T0 + 60, 5.5)  # 2x and 5x rungs: 1 + 2 = 3 credits
    assert b.features("cat1", T0 + 61)["narrative_heat_log"] == pytest.approx(0.0, abs=1e-9)
    b.on_launch("cat2", T0 + 120, "Cat Party", "CATP")
    f = b.features("cat2", T0 + 120)
    assert f["narrative_heat_log"] == pytest.approx(math.log1p(3.0 * 0.5 ** (60 / 1800)), rel=1e-6)
    # half-life decay
    later = b.features("cat2", T0 + 60 + 1800)["narrative_heat_log"]
    assert later == pytest.approx(math.log1p(1.5), rel=1e-6)
    # a rung is credited once
    b.on_peak("cat1", T0 + 200, 5.9)
    assert b.heat["cat"] == (3.0, T0 + 60)  # unchanged: no new rung


def test_copycats_and_recent_runner_names() -> None:
    b = NarrativeBook()
    b.on_launch("a", T0, "Moon Frog", "FROG")
    b.on_peak("a", T0 + 100, 6.0)  # a 5x runner
    b.on_launch("b", T0 + 200, "Frog Moon", "FROG")
    b.on_launch("c", T0 + 300, "Frog Moon", "FROG")
    fb = b.features("b", T0 + 300)
    assert fb["name_copycats_3600s_log"] == pytest.approx(math.log1p(2))  # a and c share FROG
    assert fb["copies_recent_runner"] == 1.0
    assert b.features("a", T0 + 300)["copies_recent_runner"] == 0.0  # not a copy of itself
    assert b.features("b", T0 + 200 + 7 * 3600)["name_copycats_3600s_log"] == 0.0


def test_market_feeds_narratives_from_named_launches() -> None:
    m = SolanaMarket(SolanaConfig())
    m.ingest(TokenLaunch("cat1", T0, "dev1", name="Space Cat", symbol="SCAT"))
    price0 = 30.0 / 1.073e9
    m.ingest(Swap("cat1", T0 + 10, "w", True, 50.0, 1e6, 30.0 * 12, 1.073e9 / 1.0))  # price ×12
    assert m.peak["cat1"] > 10
    m.ingest(TokenLaunch("cat2", T0 + 20, "dev2", name="Cat King", symbol="CKING"))
    f = m.narrative.features("cat2", T0 + 20)
    assert f["narrative_heat_log"] > math.log1p(6.0)  # 2x + 5x + 10x rungs = 7 credits
    assert price0 > 0
