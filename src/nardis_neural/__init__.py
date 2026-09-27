"""nardis_neural — the continually-learning neural market-state brain.

Minimal integration surface::

    from nardis_neural import NeuralEngine, NeuralObservation

    engine = NeuralEngine.load("workspace")          # registry root or model directory
    prediction = engine.predict(observation)

    from nardis_neural import ContinualLearner

    trainer = ContinualLearner("workspace")
    trainer.add_experience(observation, outcome)
    trainer.adapt_if_needed()
    trainer.full_retrain_if_needed()

This package produces probabilistic forecasts only.  It contains no trading logic,
order routing, wallet or RPC functionality.
"""

from nardis_neural.config import NeuralConfig, load_config
from nardis_neural.inference.engine import NeuralEngine
from nardis_neural.schemas import (
    GraphInput,
    NeuralObservation,
    NeuralOutcome,
    NeuralPrediction,
    SequenceInput,
)
from nardis_neural.training.continual import ContinualLearner

NeuralTrainer = ContinualLearner

__all__ = [
    "ContinualLearner",
    "GraphInput",
    "NeuralConfig",
    "NeuralEngine",
    "NeuralObservation",
    "NeuralOutcome",
    "NeuralPrediction",
    "NeuralTrainer",
    "SequenceInput",
    "load_config",
]

__version__ = "0.1.0"
