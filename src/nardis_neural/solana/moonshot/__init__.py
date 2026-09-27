"""Moonshot engine: executable peak-multiple labels, a censored power-law tail model of
P(≥2x … ≥1000x), ladder-payoff expectations and lottery-Kelly sizing hints (paper research
only — no execution)."""

from nardis_neural.solana.moonshot.labels import (
    MoonshotOutcome,
    MoonshotSpec,
    ladder_payoff,
    moonshot_outcome,
)
from nardis_neural.solana.moonshot.research import (
    MoonshotDataset,
    MoonshotResearch,
    build_moonshot_dataset,
    moonshot_markdown,
    run_moonshot_research,
)
from nardis_neural.solana.moonshot.tail import TailModel, TailPrediction

__all__ = [
    "MoonshotDataset",
    "MoonshotOutcome",
    "MoonshotResearch",
    "MoonshotSpec",
    "TailModel",
    "TailPrediction",
    "build_moonshot_dataset",
    "ladder_payoff",
    "moonshot_markdown",
    "moonshot_outcome",
    "run_moonshot_research",
]
