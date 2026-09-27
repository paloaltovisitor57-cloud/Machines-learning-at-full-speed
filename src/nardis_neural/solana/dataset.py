"""Leakage-free dataset construction from raw Solana event history.

Two markets are built from the same history:

* a **causal replay** market that ingests events strictly in time order — every snapshot's
  features (including wallet reputations and funding clusters) are computed from it at
  the snapshot time, exactly as a live system would have seen them;
* a **hindsight** market containing the complete history, used *only* to label what
  happened after each snapshot.

Snapshot times are also chosen causally (a token is sampled every
``sample_interval_seconds`` while it is still trading in the replayed market), so the
schedule itself never peeks at a token's future.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from nardis_neural.config import NeuralConfig
from nardis_neural.data.loaders import Array, ArrayStore
from nardis_neural.data.sequences import observations_to_arrays, outcomes_to_arrays
from nardis_neural.schemas import NeuralObservation, NeuralOutcome
from nardis_neural.solana.config import RISK_LABELS, SolanaConfig
from nardis_neural.solana.features import SolanaFeatureBuilder
from nardis_neural.solana.labels import SolanaLabeler
from nardis_neural.solana.market import EventStore, SolanaMarket


@dataclass
class SolanaDataset:
    """Leakage-free snapshots: neural arrays, observations and outcomes plus per-row Solana labels."""

    arrays: dict[str, Array]
    """Canonical neural arrays (features + regression targets)."""
    observations: list[NeuralObservation]
    outcomes: list[NeuralOutcome]
    mints: npt.NDArray[np.str_]
    risk: npt.NDArray[np.float32]
    """(N, len(RISK_LABELS)); NaN where the risk horizon was not complete."""
    net_returns: npt.NDArray[np.float32]
    """(N, H) cost-aware simple returns; NaN for incomplete horizons."""
    round_trip_cost: npt.NDArray[np.float32]
    extra: dict[str, Array] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.observations)

    def store(self) -> ArrayStore:
        """The canonical arrays as an :class:`ArrayStore` (training input)."""
        return ArrayStore(self.arrays)

    @property
    def current(self) -> npt.NDArray[np.float32]:
        """(N, len(CURRENT_FEATURES)) current-state feature matrix."""
        return np.asarray(self.arrays["current"], dtype=np.float32)


def hindsight_market(store: EventStore, cfg: SolanaConfig) -> SolanaMarket:
    """A market that has ingested the whole history — used for labelling only, never for features."""
    market = SolanaMarket(cfg)
    market.ingest_many(store.sorted())
    return market


def build_solana_dataset(
    store: EventStore,
    cfg: SolanaConfig,
    neural_cfg: NeuralConfig | None = None,
    archetypes: dict[str, str] | None = None,
    max_age_seconds: float = 7200.0,
    idle_stop_seconds: float = 300.0,
) -> SolanaDataset:
    """Build leakage-free snapshots: features from a causal replay, labels from hindsight.

    Each token is sampled every ``sample_interval_seconds`` from ``min_token_age_seconds`` after
    launch until it is ``max_age_seconds`` old or idle for ``idle_stop_seconds``; snapshots with
    no complete horizon are dropped.  ``archetypes`` (mint → name) adds an ``archetype`` extra
    column.  Raises ``ValueError`` if no snapshot can be labelled.
    """
    ncfg = neural_cfg or cfg.neural_config()
    builder = SolanaFeatureBuilder(cfg)
    labeler = SolanaLabeler(cfg)
    full = hindsight_market(store, cfg)
    data_end = store.end_time
    replay = SolanaMarket(cfg)
    due: list[tuple[float, str]] = []
    observations: list[NeuralObservation] = []
    outcomes: list[NeuralOutcome] = []
    mints: list[str] = []
    risk_rows: list[list[float]] = []
    net_rows: list[list[float]] = []
    costs: list[float] = []
    horizons = cfg.horizons

    def snapshot(until: float) -> None:
        while due and due[0][0] <= until:
            t, mint = heapq.heappop(due)
            replay.advance(t)
            log = replay.token(mint)
            if t - log.launch.t > max_age_seconds or (len(log.swaps) and t - log.last_t > idle_stop_seconds):
                continue  # token went quiet (known causally) → stop sampling it
            heapq.heappush(due, (t + cfg.sample_interval_seconds, mint))
            labels = labeler.label(full.token(mint), t, data_end)
            if labels is None:
                continue
            observations.append(builder.observation(replay, mint, t))
            outcomes.append(labels.outcome)
            mints.append(mint)
            risk_rows.append(
                [labels.risk[k] for k in RISK_LABELS]
                if labels.risk is not None
                else [np.nan] * len(RISK_LABELS)
            )
            net_rows.append([labels.net_returns.get(h.name, np.nan) for h in horizons])
            costs.append(labels.round_trip_cost)

    from nardis_neural.solana.events import TokenLaunch

    for e in store.sorted():
        snapshot(e.t - 1e-9)
        replay.ingest(e)
        if isinstance(e, TokenLaunch):
            heapq.heappush(due, (e.t + cfg.min_token_age_seconds, e.mint))
    snapshot(data_end)
    if not observations:
        raise ValueError("no labelled snapshots could be built from this history")
    arrays = observations_to_arrays(observations, ncfg) | outcomes_to_arrays(outcomes, ncfg)
    extra: dict[str, Any] = {}
    if archetypes is not None:
        extra["archetype"] = np.asarray([archetypes.get(m, "unknown") for m in mints])
    return SolanaDataset(
        arrays=arrays,
        observations=observations,
        outcomes=outcomes,
        mints=np.asarray(mints),
        risk=np.asarray(risk_rows, dtype=np.float32),
        net_returns=np.asarray(net_rows, dtype=np.float32),
        round_trip_cost=np.asarray(costs, dtype=np.float32),
        extra=extra,
    )
