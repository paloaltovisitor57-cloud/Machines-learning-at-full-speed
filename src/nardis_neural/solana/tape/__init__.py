"""Tape Transformer: attention over the raw trade tape with learned wallet embeddings,
predicting fat-tailed peak multiples and the collapse hazard (exit signal)."""

from nardis_neural.solana.tape.dataset import replay_tapes
from nardis_neural.solana.tape.features import TRADE_FEATURES, Tape, TapeSpec, extract_tape, wallet_bucket
from nardis_neural.solana.tape.model import TapeModel, TapeNet, TapePrediction, hazard_targets

__all__ = [
    "TRADE_FEATURES",
    "Tape",
    "TapeModel",
    "TapeNet",
    "TapePrediction",
    "TapeSpec",
    "extract_tape",
    "hazard_targets",
    "replay_tapes",
    "wallet_bucket",
]
