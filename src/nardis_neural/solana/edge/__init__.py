"""Edge engine: executable triple-barrier labels, meta-labeling edge model, walk-forward
research and cost-aware backtesting (paper research only — no execution)."""

from nardis_neural.solana.edge.backtest import Candidates, baselines, simulate, trade_stats
from nardis_neural.solana.edge.barriers import BarrierOutcome, BarrierSpec, triple_barrier
from nardis_neural.solana.edge.model import EdgeModel, EdgePrediction
from nardis_neural.solana.edge.research import (
    EdgeResearch,
    edge_features,
    research_markdown,
    run_edge_research,
)

__all__ = [
    "BarrierOutcome",
    "BarrierSpec",
    "Candidates",
    "EdgeModel",
    "EdgePrediction",
    "EdgeResearch",
    "baselines",
    "edge_features",
    "research_markdown",
    "run_edge_research",
    "simulate",
    "trade_stats",
    "triple_barrier",
]
