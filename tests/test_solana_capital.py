"""Capital engine: overfitting statistics, allocator limits and bankroll arithmetic."""

from __future__ import annotations

import numpy as np
import pytest

from nardis_neural.solana.capital import (
    AllocatorConfig,
    BookState,
    CapitalAllocator,
    Signal,
    TicketRecord,
    bootstrap_book,
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    simulate_book,
)


# ---------------------------------------------------------------- overfitting statistics
def test_deflated_sharpe_punishes_the_best_of_many_noise_trials() -> None:
    rng = np.random.default_rng(0)
    trials = rng.normal(0.0, 1.0, size=(200, 50))  # 50 unskilled strategies, 200 periods
    sharpes = trials.mean(0) / trials.std(0, ddof=1)
    best = trials[:, int(np.argmax(sharpes))]
    lucky = deflated_sharpe(best, n_trials=50, trial_sharpes=sharpes)
    assert lucky["sharpe"] > 0.1 and lucky["dsr"] < 0.5, "the luckiest of 50 noise series is not an edge"
    skilled = rng.normal(0.3, 1.0, 200)
    real = deflated_sharpe(skilled, n_trials=50, trial_sharpes=sharpes)
    assert real["dsr"] > 0.95
    assert expected_max_sharpe(1, 1.0) == 0.0 and expected_max_sharpe(100, 0.01) > expected_max_sharpe(
        10, 0.01
    )


def test_pbo_is_high_for_noise_and_low_for_a_real_edge() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(size=(240, 20))
    assert probability_of_backtest_overfitting(noise, blocks=8)["pbo"] > 0.3
    edge = noise.copy()
    edge[:, 7] += 0.8  # one configuration is genuinely better everywhere
    assert probability_of_backtest_overfitting(edge, blocks=8)["pbo"] < 0.05
    assert np.isnan(probability_of_backtest_overfitting(noise[:, :1])["pbo"])


# ---------------------------------------------------------------- allocator
def _sig(mint: str, kelly: float = 0.04, **kw: float | str | bool) -> Signal:
    base: dict[str, float | str | bool] = {"expected_multiple": 3.0, "lottery_kelly": kelly}
    base.update(kw)
    return Signal(mint, 1_000.0, **base)  # type: ignore[arg-type]


def test_allocator_enforces_every_limit() -> None:
    cfg = AllocatorConfig(
        max_position_fraction=0.03, liquidity_fraction=0.02, max_family_fraction=0.04, max_total_exposure=0.08
    )
    alloc = CapitalAllocator(cfg)
    state = BookState(equity=100.0, peak_equity=100.0)
    out = alloc.allocate(
        [
            _sig("A", family="fam1"),
            _sig("B", family="fam1"),
            _sig("C", liquidity_sol=50.0),
            _sig("D", vetoed=True),
            _sig("E", expected_multiple=0.8),
            _sig("F"),
            _sig("G"),
        ],
        state,
    )
    by = {a.mint: a for a in out}
    assert by["A"].stake_sol == pytest.approx(3.0) and by["A"].reason == "position cap"
    assert by["B"].stake_sol == pytest.approx(1.0) and by["B"].reason == "family cap"
    assert by["C"].stake_sol == pytest.approx(1.0) and by["C"].reason == "liquidity cap"
    assert by["D"].stake_sol == 0 and by["E"].stake_sol == 0
    assert sum(a.stake_sol for a in out) == pytest.approx(8.0), "total exposure cap"
    assert by["G"].stake_sol == 0


def test_uncertainty_trust_track_record_and_governors_shrink_stakes() -> None:
    state = BookState(equity=100.0, peak_equity=100.0)
    base = CapitalAllocator().allocate([_sig("A", kelly=0.02)], state)[0].stake_sol
    assert CapitalAllocator().allocate([_sig("A", kelly=0.02, epistemic=0.15)], state)[0].stake_sol < base
    assert CapitalAllocator().allocate([_sig("A", kelly=0.02, trust=0.5)], state)[
        0
    ].stake_sol == pytest.approx(base / 2)
    assert CapitalAllocator(track_record=0.5).allocate([_sig("A", kelly=0.02)], state)[0].stake_sol < base
    down = BookState(equity=80.0, peak_equity=100.0)  # 20 % drawdown of a 35 % budget
    assert CapitalAllocator().governor(down) == pytest.approx(1 - 0.2 / 0.35)
    ruined = BookState(equity=60.0, peak_equity=100.0)
    assert CapitalAllocator().allocate([_sig("A")], ruined)[0].reason == "drawdown governor"
    day = BookState(equity=80.0, peak_equity=80.0, day=0, day_start_equity=100.0)
    assert CapitalAllocator().allocate([_sig("A")], day)[0].reason == "daily loss stop"


# ---------------------------------------------------------------- bankroll
def test_bankroll_locks_capital_and_compounds_exactly() -> None:
    tickets = [
        TicketRecord(Signal("A", 0.0, 3.0, 0.04), t_exit=10.0, multiple=3.0),
        TicketRecord(Signal("B", 5.0, 3.0, 0.04), t_exit=20.0, multiple=0.0),
        TicketRecord(Signal("C", 15.0, 3.0, 0.04), t_exit=30.0, multiple=2.0),
    ]
    flat = simulate_book(tickets, None, initial_equity=10.0, flat_stake=1.0)
    assert flat["final_equity"] == pytest.approx(10.0 + 2.0 - 1.0 + 1.0)
    assert flat["tickets_taken"] == 3 and flat["max_drawdown"] > 0
    capped = simulate_book(tickets, None, initial_equity=1.5, flat_stake=1.0)
    assert capped["tickets_taken"] == 3 and capped["sol_staked"] == pytest.approx(2.5), (
        "B gets only the 0.5 left"
    )
    sized = simulate_book(tickets, CapitalAllocator(), initial_equity=100.0)
    assert sized["tickets_taken"] == 3 and sized["final_equity"] > 100.0
    risk = bootstrap_book(tickets, None, initial_equity=10.0, flat_stake=1.0, draws=50)
    assert 0 <= risk["prob_ruin"] <= risk["prob_loss"] <= 1


def test_a_stake_cut_below_the_minimum_says_so() -> None:
    book = BookState(equity=10.0, peak_equity=10.0)
    small = CapitalAllocator().allocate([Signal("m", 0.0, 2.0, 0.001)], book)
    assert small[0].stake_sol == 0.0 and small[0].reason.endswith("(below minimum)")
