"""Capital engine: sizing under Kelly, uncertainty, liquidity, correlation and drawdown limits,
event-driven bankroll simulation with a bootstrap risk profile, and multiple-testing-aware
edge statistics (Deflated Sharpe Ratio, Probability of Backtest Overfitting).  Advice only."""

from nardis_neural.solana.capital.allocator import (
    Allocation,
    AllocatorConfig,
    BookState,
    CapitalAllocator,
    Signal,
)
from nardis_neural.solana.capital.bankroll import TicketRecord, bootstrap_book, simulate_book
from nardis_neural.solana.capital.overfit import (
    deflated_sharpe,
    expected_max_sharpe,
    probability_of_backtest_overfitting,
    sharpe,
)

__all__ = [
    "Allocation",
    "AllocatorConfig",
    "BookState",
    "CapitalAllocator",
    "Signal",
    "TicketRecord",
    "bootstrap_book",
    "deflated_sharpe",
    "expected_max_sharpe",
    "probability_of_backtest_overfitting",
    "sharpe",
    "simulate_book",
]
