"""Public input/output contract.

The trading system converts its own market state into a :class:`NeuralObservation`
(generic tensors, no Solana-specific names) and receives a :class:`NeuralPrediction`.
When the future becomes known it reports a :class:`NeuralOutcome`.

Sequence convention: rows are ordered oldest → newest; the last row is the most recent
step.  ``time_deltas[i]`` is the number of seconds between step ``i`` and the observation
timestamp (so it is non-negative and decreasing).  Sequences shorter than the configured
``max_len`` are left-padded internally; longer ones keep the most recent ``max_len`` rows.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


def _as_float_array(value: Any, ndim: int, name: str) -> FloatArray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-D, got shape {arr.shape}")
    return arr


class SequenceInput(BaseModel):
    """One temporal stream for one observation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, use_attribute_docstrings=True)

    values: FloatArray
    """(T, F) feature matrix, oldest row first."""
    time_deltas: FloatArray | None = None
    """(T,) seconds before the observation timestamp; derived from resolution if missing."""
    mask: BoolArray | None = None
    """(T,) True where the step is observed; missing steps may be NaN in ``values``."""

    @field_validator("values", mode="before")
    @classmethod
    def _values(cls, v: Any) -> FloatArray:
        return _as_float_array(v, 2, "values")

    @field_validator("time_deltas", mode="before")
    @classmethod
    def _dts(cls, v: Any) -> FloatArray | None:
        return None if v is None else _as_float_array(v, 1, "time_deltas")

    @field_validator("mask", mode="before")
    @classmethod
    def _mask(cls, v: Any) -> BoolArray | None:
        if v is None:
            return None
        arr = np.asarray(v, dtype=np.bool_)
        if arr.ndim != 1:
            raise ValueError("mask must be 1-D")
        return arr

    @model_validator(mode="after")
    def _lengths(self) -> SequenceInput:
        t = self.values.shape[0]
        if self.time_deltas is not None and self.time_deltas.shape[0] != t:
            raise ValueError("time_deltas length must match values")
        if self.mask is not None and self.mask.shape[0] != t:
            raise ValueError("mask length must match values")
        return self


class GraphInput(BaseModel):
    """Optional relational context (e.g. wallet→token, wallet→wallet, token→token)."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, use_attribute_docstrings=True)

    node_features: FloatArray
    """(N, F_node)."""
    edge_index: IntArray
    """(2, E) source/destination node indices (local to this graph)."""
    edge_type: IntArray | None = None
    """(E,) relation ids in ``[0, num_edge_types)``."""
    target_node: int = 0
    """Index of the node representing the observed asset."""

    @field_validator("node_features", mode="before")
    @classmethod
    def _nf(cls, v: Any) -> FloatArray:
        return _as_float_array(v, 2, "node_features")

    @field_validator("edge_index", mode="before")
    @classmethod
    def _ei(cls, v: Any) -> IntArray:
        arr = np.asarray(v, dtype=np.int64)
        if arr.size == 0:
            arr = arr.reshape(2, 0)
        if arr.ndim != 2 or arr.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, E)")
        return arr

    @field_validator("edge_type", mode="before")
    @classmethod
    def _et(cls, v: Any) -> IntArray | None:
        return None if v is None else np.asarray(v, dtype=np.int64).reshape(-1)

    @model_validator(mode="after")
    def _check(self) -> GraphInput:
        n = self.node_features.shape[0]
        if not 0 <= self.target_node < n:
            raise ValueError("target_node out of range")
        if self.edge_index.size and (self.edge_index.min() < 0 or self.edge_index.max() >= n):
            raise ValueError("edge_index references unknown nodes")
        if self.edge_type is not None and self.edge_type.shape[0] != self.edge_index.shape[1]:
            raise ValueError("edge_type length must equal number of edges")
        return self


class NeuralObservation(BaseModel):
    """Market state at one instant, expressed as generic tensors."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True, use_attribute_docstrings=True)

    observation_id: str
    timestamp: float
    """Unix seconds of the observation (prediction time)."""
    current_features: FloatArray
    """(F_current,) immediate state vector. NaN marks missing values."""
    sequences: dict[str, SequenceInput] = Field(default_factory=dict)
    """Timescale name → stream.  Missing timescales are allowed."""
    graph: GraphInput | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("current_features", mode="before")
    @classmethod
    def _cf(cls, v: Any) -> FloatArray:
        return _as_float_array(v, 1, "current_features")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NeuralObservation:
        """Validate a plain dict (e.g. parsed JSON) into an observation."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict of the observation (arrays become nested lists)."""
        seqs: dict[str, Any] = {}
        for name, s in self.sequences.items():
            seqs[name] = {
                "values": s.values.tolist(),
                "time_deltas": None if s.time_deltas is None else s.time_deltas.tolist(),
                "mask": None if s.mask is None else s.mask.tolist(),
            }
        graph = None
        if self.graph is not None:
            graph = {
                "node_features": self.graph.node_features.tolist(),
                "edge_index": self.graph.edge_index.tolist(),
                "edge_type": None if self.graph.edge_type is None else self.graph.edge_type.tolist(),
                "target_node": self.graph.target_node,
            }
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp,
            "current_features": self.current_features.tolist(),
            "sequences": seqs,
            "graph": graph,
            "metadata": self.metadata,
        }


class NeuralOutcome(BaseModel):
    """Realised future of an observation, keyed by horizon name.

    Horizons that are not yet known may be omitted; they are masked out of training.
    ``max_drawdown`` is a positive magnitude (0.05 = price fell 5% below entry at worst).
    """

    observation_id: str
    returns: dict[str, float] = Field(default_factory=dict)
    max_upside: dict[str, float] = Field(default_factory=dict)
    max_drawdown: dict[str, float] = Field(default_factory=dict)
    volatility: dict[str, float] = Field(default_factory=dict)
    resolved_at: float = Field(default_factory=time.time)

    def task_values(self, task: str) -> dict[str, float]:
        """Horizon name → realised value for regression ``task`` (``KeyError`` for unknown tasks)."""
        mapping = {
            "return": self.returns,
            "max_upside": self.max_upside,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
        }
        return mapping[task]


class NeuralPrediction(BaseModel):
    """Probabilistic forecast.  Contains no trading decision of any kind."""

    observation_id: str
    model_version: str
    timestamp: float
    horizons: list[str]
    expected_returns: dict[str, float]
    return_std: dict[str, float]
    """Total predictive standard deviation of the return (aleatoric + epistemic)."""
    return_quantiles: dict[str, dict[str, float]] = Field(default_factory=dict)
    upside_probabilities: dict[str, float]
    downside_probabilities: dict[str, float]
    maximum_upside: dict[str, float]
    maximum_drawdown: dict[str, float]
    predicted_volatility: dict[str, float]
    epistemic_uncertainty: float
    aleatoric_uncertainty: float
    total_uncertainty: float
    confidence: float
    confidence_components: dict[str, float] = Field(default_factory=dict)
    ood_score: float
    member_disagreement: float
    market_embedding: list[float]
    expert_weights: dict[str, float]
    regime_cluster: int | None = None
