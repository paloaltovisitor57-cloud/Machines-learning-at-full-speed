"""Reference integration: how a trading system (e.g. Nardis) plugs into the neural brain.

The trading system owns data collection, execution and strategy.  It only has to:

1. turn its market state into a generic :class:`NeuralObservation`,
2. call ``predict`` and use the probabilistic output however its own strategy decides,
3. report the realised future as a :class:`NeuralOutcome` once it is known,
4. periodically call the maintenance hooks (adapt / full retrain / promote).

Run it standalone (bootstraps a workspace from synthetic data)::

    python examples/nardis_integration.py --workspace workspaces/demo --config configs/small.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nardis_neural import (
    ContinualLearner,
    NeuralConfig,
    NeuralObservation,
    NeuralOutcome,
    NeuralPrediction,
    load_config,
)
from nardis_neural.data.sequences import build_bars


def to_observation(
    observation_id: str,
    now: float,
    current_features: npt.NDArray[np.float32],
    event_times: npt.NDArray[np.float64],
    event_features: npt.NDArray[np.float32],
    config: NeuralConfig,
) -> NeuralObservation:
    """Build an observation from a raw, irregular event stream (e.g. trades of one token).

    ``build_bars`` aggregates events causally (only ``time <= now``) into every configured
    resolution, so the observation can never contain future information.
    """
    sequences = {
        ts.name: build_bars(event_times, event_features[:, : ts.feature_dim], now, ts)
        for ts in config.features.timescales
        if event_features.shape[1] >= ts.feature_dim
    }
    return NeuralObservation(
        observation_id=observation_id,
        timestamp=now,
        current_features=current_features,
        sequences=sequences,
    )


class NeuralBrain:
    """Thin adapter the trading system can hold as a singleton."""

    def __init__(self, workspace: str | Path, device: str | None = None) -> None:
        self.learner = ContinualLearner(workspace, device=device)

    def predict(self, observation: NeuralObservation) -> NeuralPrediction:
        # Champion prediction; a challenger (if any) silently runs in shadow mode.
        return self.learner.predict(observation)

    def report_outcome(self, observation: NeuralObservation, outcome: NeuralOutcome) -> None:
        self.learner.add_experience(observation, outcome)

    def maintenance(self) -> dict[str, object]:
        """Call on a timer (e.g. every few minutes) — never on the hot path."""
        adapted = self.learner.adapt_if_needed()
        retrained = self.learner.full_retrain_if_needed()
        decision = self.learner.promote_if_ready()
        self.learner.save()
        return {
            "adapted": None if adapted is None else adapted.status,
            "full_retrain": None if retrained is None else retrained.status,
            "promotion": None if decision is None else decision.promote,
            "champion": self.learner.registry.champion_version,
        }


def bootstrap_workspace(workspace: Path, config: NeuralConfig, n: int = 1500) -> None:
    """Train an initial champion on synthetic data (stand-in for Nardis' historical export)."""
    from nardis_neural.data.loaders import ArrayStore
    from nardis_neural.synthetic import SyntheticSpec, generate_synthetic
    from nardis_neural.training.pipeline import train_engine

    store = ArrayStore(generate_synthetic(config, SyntheticSpec(n_observations=n, seed=0)))
    engine, _ = train_engine(config, store, log=print)
    ContinualLearner.initialize(workspace, engine, config)


def main(
    workspace: Path, config_path: Path | None, n_live: int = 600, device: str | None = None
) -> dict[str, object]:
    config = load_config(config_path)
    if not (workspace / "registry.json").exists():
        bootstrap_workspace(workspace, config)
    brain = NeuralBrain(workspace, device=device)

    # --- simulate the live loop with synthetic "future" data -------------------------
    from nardis_neural.data.sequences import arrays_to_observations, arrays_to_outcomes
    from nardis_neural.synthetic import SyntheticSpec, generate_synthetic

    live = generate_synthetic(
        config, SyntheticSpec(n_observations=n_live, seed=42, start_time=1_700_050_000.0)
    )
    observations = arrays_to_observations(live, config)
    outcomes = arrays_to_outcomes(live, config)
    last: NeuralPrediction | None = None
    for obs, outcome in zip(observations, outcomes, strict=True):
        last = brain.predict(obs)  # hot path: probabilistic forecast only
        brain.report_outcome(obs, outcome)  # later, once the horizons have elapsed
    status = brain.maintenance()

    # --- building an observation straight from raw events ------------------------------
    rng = np.random.default_rng(0)
    times = np.sort(rng.uniform(0, 600, size=400))
    events = rng.normal(size=(400, 7)).astype(np.float32)
    raw_obs = to_observation(
        "raw-demo",
        600.0,
        rng.normal(size=config.features.current_dim).astype(np.float32),
        times,
        events,
        config,
    )
    raw_pred = brain.predict(raw_obs)
    assert last is not None
    print("example prediction:", last.model_dump(exclude={"market_embedding"}))
    print("raw-event prediction confidence:", raw_pred.confidence, "ood:", raw_pred.ood_score)
    print("maintenance:", status)
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("workspaces/demo"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--live", type=int, default=600)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    main(args.workspace, args.config, args.live, args.device)
