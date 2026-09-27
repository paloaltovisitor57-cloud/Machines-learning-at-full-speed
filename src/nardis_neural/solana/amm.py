"""Solana DEX pricing maths: pump.fun bonding curve and constant-product AMM pools.

Used to (a) derive price / liquidity / bonding-progress features, (b) build *cost-aware*
labels (returns net of price impact and fees for a realistic trade size) and (c) simulate
launches.  Amounts are in SOL and whole tokens (decimals already applied).
"""

from __future__ import annotations

from dataclasses import dataclass

# pump.fun launch parameters (virtual reserves of a fresh bonding curve)
PUMP_VIRTUAL_SOL = 30.0
PUMP_VIRTUAL_TOKENS = 1_073_000_000.0
PUMP_CURVE_TOKENS = 793_100_000.0
"""Real tokens sold through the curve before graduation."""
PUMP_GRADUATION_SOL = 85.0
"""Approximate real SOL collected when the curve completes and migrates to an AMM."""
PUMP_FEE = 0.01
AMM_FEE = 0.0025
TOTAL_SUPPLY = 1_000_000_000.0


@dataclass(frozen=True)
class Pool:
    """Constant-product pricing reserves (virtual reserves for a bonding curve)."""

    sol: float
    tokens: float
    fee: float
    virtual_sol: float = 0.0

    @property
    def price(self) -> float:
        """SOL per token (spot)."""
        return self.sol / self.tokens if self.tokens > 0 else 0.0

    @property
    def real_sol(self) -> float:
        return max(self.sol - self.virtual_sol, 0.0)

    def buy(self, sol_in: float) -> tuple[float, Pool]:
        """Tokens received for ``sol_in`` SOL and the pool afterwards (fee taken on input)."""
        if sol_in <= 0:
            return 0.0, self
        net = sol_in * (1 - self.fee)
        k = self.sol * self.tokens
        new_sol = self.sol + net
        out = self.tokens - k / new_sol
        return out, Pool(new_sol, self.tokens - out, self.fee, self.virtual_sol)

    def sell(self, tokens_in: float) -> tuple[float, Pool]:
        """SOL received for ``tokens_in`` tokens (fee taken on output)."""
        if tokens_in <= 0:
            return 0.0, self
        k = self.sol * self.tokens
        new_tokens = self.tokens + tokens_in
        gross = self.sol - k / new_tokens
        return gross * (1 - self.fee), Pool(self.sol - gross, new_tokens, self.fee, self.virtual_sol)


def pump_curve(real_sol: float = 0.0) -> Pool:
    """A pump.fun curve after ``real_sol`` SOL of net buying."""
    pool = Pool(PUMP_VIRTUAL_SOL, PUMP_VIRTUAL_TOKENS, PUMP_FEE, PUMP_VIRTUAL_SOL)
    if real_sol > 0:
        _, pool = pool.buy(real_sol / (1 - PUMP_FEE))
    return pool


def bonding_progress(pool: Pool) -> float:
    """Fraction of the curve completed (1.0 = graduation)."""
    return min(pool.real_sol / PUMP_GRADUATION_SOL, 1.0)


def round_trip_cost(pool: Pool, size_sol: float) -> float:
    """Fractional loss of buying ``size_sol`` and immediately selling everything back.

    Captures both fees and two-way price impact — the minimum edge a forecast must exceed
    for a trade of this size to be worthwhile.
    """
    if size_sol <= 0 or pool.tokens <= 0:
        return 0.0
    tokens, after = pool.buy(size_sol)
    back, _ = after.sell(tokens)
    return 1.0 - back / size_sol


def price_impact(pool: Pool, size_sol: float) -> float:
    """Relative spot-price move caused by a buy of ``size_sol``."""
    _, after = pool.buy(size_sol)
    return after.price / pool.price - 1.0 if pool.price > 0 else 0.0


def market_cap_sol(pool: Pool, supply: float = TOTAL_SUPPLY) -> float:
    return pool.price * supply
