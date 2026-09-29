"""The edge the addon always chases: 2x, 5x, 10x, 100x and 1000x.

These targets are fixed by design, not configuration.  Every token assessment, every ranking
and every piece of trade advice is scored against all five, so the trading system always sees
how much of a shot each opportunity has at each multiple, and which is the craziest multiple
still worth chasing.

For a target ``k``, a position that misses it is assumed to end at ``loss_multiple`` (a cut
loser; 0.7 measured on real pump.fun data unless the caller knows better).  Chasing ``k`` has
positive expectancy when

    p · k + (1 − p) · L > 1   ⇔   p > p* = (1 − L) / (k − L)            (break-even probability)

``edge_ratio = p / p*``: above 1 the chase pays on average, and 2 means twice the odds needed.
``chase_target`` is the highest target with ``edge_ratio > 1``, the most ambitious multiple
that is still +EV.  With L = 0.7 the break-evens are 23 %, 7.0 %, 3.2 %, 0.30 % and 0.030 %.

A target only becomes ``chase_target`` once it is *proven*: when observed hit counts are
given, at least ``min_hits`` real hits at that level are required, so a prior on a handful of
trades can never make 1000x look like a +EV chase.

This chases returns without betting the account: sizing stays with the capital engine, whose
limits keep the account alive long enough for the hits to compound.
"""

from __future__ import annotations

from typing import Final

CHASE_TARGETS: Final[tuple[float, ...]] = (2.0, 5.0, 10.0, 100.0, 1000.0)
"""The multiples the addon always chases."""

DEFAULT_LOSS_MULTIPLE: Final[float] = 0.7
"""What a missed chase returns per SOL staked (cut losers averaged about 0.7x on real data)."""


def break_even(target: float, loss_multiple: float = DEFAULT_LOSS_MULTIPLE) -> float:
    """Probability of reaching ``target`` above which chasing it has positive expectancy."""
    if target <= 1.0 or not 0.0 <= loss_multiple < 1.0:
        raise ValueError("target must exceed 1 and the loss multiple must be in [0, 1)")
    return (1.0 - loss_multiple) / (target - loss_multiple)


def chase_profile(
    p_reach: dict[float, float],
    loss_multiple: float = DEFAULT_LOSS_MULTIPLE,
    base_rates: dict[float, float] | None = None,
    hits: dict[float, int] | None = None,
    min_hits: int = 3,
) -> dict[str, float]:
    """Flat dict of the chase for one opportunity.

    ``p_reach`` maps each target to P(reaching it).  Returns, per target ``k``:
    ``p_{k}x``, ``break_even_{k}x``, ``edge_{k}x`` (= p / break-even) and, when ``base_rates``
    are given, ``lift_{k}x`` (= p / base rate: how much better than an average trade).  Also
    ``proven_{k}x`` (1 when enough real hits back it), ``chase_target`` (the highest proven
    target with edge > 1, 0 when none) and ``chase_edge`` (its
    edge ratio), ``tail_ev`` (expected multiple of a ladder that banks each target reached,
    a lower bound on the tail's value) and ``crazy_shot`` (edge of the 100x / 1000x chase).
    """
    out: dict[str, float] = {}
    best, best_edge = 0.0, 0.0
    prev = 1.0
    ordered = sorted(CHASE_TARGETS)
    for k in ordered:
        p = min(max(float(p_reach.get(k, 0.0)), 0.0), prev)  # reaching 10x implies reaching 5x
        prev = p
        be = break_even(k, loss_multiple)
        edge = p / be
        tag = f"{k:g}x"
        out[f"p_{tag}"] = p
        out[f"break_even_{tag}"] = be
        out[f"edge_{tag}"] = edge
        if base_rates is not None and base_rates.get(k, 0.0) > 0:
            out[f"lift_{tag}"] = p / base_rates[k]
        proven = hits is None or hits.get(k, 0) >= min_hits
        out[f"proven_{tag}"] = float(proven)
        if edge > 1.0 and proven:
            best, best_edge = k, edge
    # a position that banks each rung reached: P(end in [k_i, k_{i+1})) · k_i, losers at L
    ps = [out[f"p_{k:g}x"] for k in ordered]
    ev = (1.0 - ps[0]) * loss_multiple
    for i, k in enumerate(ordered):
        nxt = ps[i + 1] if i + 1 < len(ps) else 0.0
        ev += (ps[i] - nxt) * k
    out["tail_ev"] = ev
    out["chase_target"] = best
    out["chase_edge"] = best_edge
    out["crazy_shot"] = max(out["edge_100x"], out["edge_1000x"])
    return out
