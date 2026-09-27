"""Experience replay for continual learning.

Three pools with different retention policies — old data is never simply discarded when
capacity is reached:

* **recent** — FIFO of the newest experiences.  Items evicted from it are *offered* to
* **historical** — a reservoir sample (Vitter's algorithm R) over everything that ever
  left the recent pool, i.e. an unbiased long-term memory of fixed size.
* **rare** — protected store of rare/extreme outcomes; when full, the lowest-priority
  (then oldest) rare item is evicted, so tail events survive much longer than normal ones.

Sampling strategies: ``uniform``, ``recency`` (exponential half-life), ``prioritized``
(PER with importance weights), ``rare`` and ``regime_balanced``; ``sample_mixture``
combines pools/strategies with configurable fractions.

Persistence stores rows in the canonical array layout (``rows/``, memory-mappable) plus
JSON metadata and an ``embeddings.npy`` matrix.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from nardis_neural.config import REGRESSION_TASKS, NeuralConfig, ReplayConfig
from nardis_neural.data.loaders import (
    KEY_TARGET_MASK,
    Array,
    ArrayStore,
    concat_arrays,
    select_rows,
    target_key,
)
from nardis_neural.data.sequences import observations_to_arrays, outcomes_to_arrays
from nardis_neural.schemas import NeuralObservation, NeuralOutcome

Strategy = Literal["uniform", "recency", "prioritized", "rare", "regime_balanced"]
Pool = Literal["all", "recent", "historical", "rare"]


@dataclass
class Experience:
    """A resolved observation (features + realised targets) with its prediction metadata."""

    observation_id: str
    timestamp: float
    row: dict[str, Array]
    """Canonical arrays with a leading dimension of 1 (features + targets + target_mask)."""
    model_version: str | None = None
    prediction: dict[str, Any] = field(default_factory=dict)
    uncertainty: float | None = None
    embedding: npt.NDArray[np.float32] | None = None
    regime: int | None = None
    priority: float | None = None
    is_rare: bool = False
    seq: int = -1

    def target(self, task: str) -> npt.NDArray[np.float32]:
        """Realised ``task`` values of this row, shape (H,)."""
        return np.asarray(self.row[target_key(task)][0], dtype=np.float32)

    @property
    def magnitude(self) -> float:
        """Largest |return| or max drawdown over valid horizons (0 when none is valid)."""
        mask = np.asarray(self.row[KEY_TARGET_MASK][0], dtype=bool)
        if not mask.any():
            return 0.0
        return float(max(np.abs(self.target("return"))[mask].max(), self.target("max_drawdown")[mask].max()))

    @classmethod
    def create(
        cls,
        observation: NeuralObservation,
        outcome: NeuralOutcome,
        config: NeuralConfig,
        model_version: str | None = None,
        prediction: dict[str, Any] | None = None,
        uncertainty: float | None = None,
        embedding: Sequence[float] | None = None,
        regime: int | None = None,
    ) -> Experience:
        """Build an experience from an observation and its outcome (``ValueError`` if ids differ).

        ``is_rare`` comes from ``replay.rare_return_threshold`` and ``priority`` from the
        standardised error of ``prediction`` (None when unavailable).
        """
        if outcome.observation_id != observation.observation_id:
            raise ValueError("outcome does not belong to this observation")
        row = observations_to_arrays([observation], config) | outcomes_to_arrays([outcome], config)
        exp = cls(
            observation_id=observation.observation_id,
            timestamp=observation.timestamp,
            row=row,
            model_version=model_version,
            prediction=dict(prediction or {}),
            uncertainty=uncertainty,
            embedding=None if embedding is None else np.asarray(embedding, dtype=np.float32),
            regime=regime,
        )
        exp.is_rare = exp.magnitude > config.replay.rare_return_threshold
        exp.priority = prediction_error_priority(exp, prediction or {}, config)
        return exp


def prediction_error_priority(
    exp: Experience, prediction: dict[str, Any], config: NeuralConfig
) -> float | None:
    """Standardised absolute return error of the original prediction (None if unknown)."""
    exp_ret, std = prediction.get("expected_returns"), prediction.get("return_std")
    if not exp_ret or not std:
        return None
    mask = np.asarray(exp.row[KEY_TARGET_MASK][0], dtype=bool)
    y = exp.target("return")
    errs = [
        abs(y[h] - float(exp_ret[name])) / max(float(std[name]), 1e-6)
        for h, name in enumerate(config.horizon_names)
        if mask[h] and name in exp_ret and name in std
    ]
    return float(np.mean(errs)) if errs else None


class ExperienceReplayBuffer:
    """Replay memory with recent, historical (reservoir) and rare pools."""

    def __init__(self, cfg: ReplayConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.recent: deque[Experience] = deque()
        self.historical: list[Experience] = []
        self.rare: list[Experience] = []
        self.seen_historical = 0
        self.next_seq = 0
        self.max_priority = 1.0

    # ------------------------------------------------------------------ insertion
    def add(self, exp: Experience) -> Experience:
        """Insert ``exp`` and return it; assigns its ``seq`` and, if unset, the max priority.

        Overflow of the recent FIFO is offered to the historical reservoir; rare experiences
        are also kept in the rare pool.
        """
        exp.seq = self.next_seq
        self.next_seq += 1
        if exp.priority is None:
            exp.priority = self.max_priority
        self.max_priority = max(self.max_priority, float(exp.priority))
        self.recent.append(exp)
        while len(self.recent) > self.cfg.recent_capacity:
            self._offer_historical(self.recent.popleft())
        if exp.is_rare:
            self._add_rare(exp)
        return exp

    def extend(self, experiences: Iterable[Experience]) -> None:
        """Add each experience in order."""
        for e in experiences:
            self.add(e)

    def _offer_historical(self, exp: Experience) -> None:
        self.seen_historical += 1
        if len(self.historical) < self.cfg.historical_capacity:
            self.historical.append(exp)
            return
        j = int(self.rng.integers(0, self.seen_historical))
        if j < self.cfg.historical_capacity:
            self.historical[j] = exp

    def _add_rare(self, exp: Experience) -> None:
        if len(self.rare) >= self.cfg.rare_capacity:
            worst = min(range(len(self.rare)), key=lambda i: (self.rare[i].priority or 0.0, self.rare[i].seq))
            self.rare.pop(worst)
        self.rare.append(exp)

    # ------------------------------------------------------------------ access
    def all(self) -> list[Experience]:
        """Unique experiences across all pools, sorted by (timestamp, seq)."""
        uniq = {e.seq: e for pool in (self.historical, list(self.recent), self.rare) for e in pool}
        return sorted(uniq.values(), key=lambda e: (e.timestamp, e.seq))

    def __len__(self) -> int:
        return len({e.seq for pool in (self.historical, list(self.recent), self.rare) for e in pool})

    def pool(self, name: Pool) -> list[Experience]:
        """Copy of one pool's items (``all`` → :meth:`all`)."""
        if name == "recent":
            return list(self.recent)
        if name == "historical":
            return list(self.historical)
        if name == "rare":
            return list(self.rare)
        return self.all()

    def update_priorities(self, seqs: Sequence[int], priorities: Sequence[float]) -> None:
        """Set priorities by ``seq`` and update the running max priority."""
        lookup = dict(zip(seqs, priorities, strict=True))
        for e in self.all():
            if e.seq in lookup:
                e.priority = float(lookup[e.seq])
                self.max_priority = max(self.max_priority, e.priority)

    # ------------------------------------------------------------------ sampling
    def probabilities(
        self, items: Sequence[Experience], strategy: Strategy, now: float | None = None
    ) -> npt.NDArray[np.float64]:
        """Sampling probabilities of ``items`` under ``strategy`` (uniform if all weights are 0)."""
        n = len(items)
        if n == 0:
            return np.zeros(0)
        if strategy == "uniform":
            w = np.ones(n)
        elif strategy == "recency":
            ts = np.array([e.timestamp for e in items])
            ref = ts.max() if now is None else now
            w = 0.5 ** (np.maximum(ref - ts, 0.0) / self.cfg.recency_half_life)
        elif strategy == "prioritized":
            pr = np.array([e.priority if e.priority is not None else self.max_priority for e in items])
            w = (pr + self.cfg.priority_epsilon) ** self.cfg.priority_alpha
        elif strategy == "rare":
            w = np.array([e.magnitude for e in items]) + 1e-6
            rare = np.array([e.is_rare for e in items])
            if rare.any():
                w = np.where(rare, w, 0.0)
        elif strategy == "regime_balanced":
            regimes = np.array([-1 if e.regime is None else e.regime for e in items])
            uniq, counts = np.unique(regimes, return_counts=True)
            per = dict(zip(uniq.tolist(), counts.tolist(), strict=True))
            w = np.array([1.0 / per[int(r)] for r in regimes])
        else:
            raise ValueError(f"unknown strategy {strategy}")
        total = w.sum()
        return w / total if total > 0 else np.full(n, 1.0 / n)

    def sample(
        self,
        n: int,
        strategy: Strategy = "uniform",
        pool: Pool = "all",
        now: float | None = None,
        items: Sequence[Experience] | None = None,
        replace: bool = True,
    ) -> tuple[list[Experience], npt.NDArray[np.float64]]:
        """Returns (experiences, importance_weights).  Importance weights are 1 except for
        ``prioritized`` sampling, where they correct the sampling bias (PER, ``beta``)."""
        candidates = list(items) if items is not None else self.pool(pool)
        if not candidates or n <= 0:
            return [], np.zeros(0)
        p = self.probabilities(candidates, strategy, now)
        if not replace:
            n = min(n, int((p > 0).sum()))
        idx = self.rng.choice(len(candidates), size=n, replace=replace, p=p)
        weights = np.ones(n)
        if strategy == "prioritized":
            weights = (len(candidates) * p[idx]) ** (-self.cfg.priority_beta)
            weights = weights / weights.max()
        return [candidates[i] for i in idx], weights

    def sample_mixture(
        self,
        n: int,
        fractions: dict[str, float],
        eligible: set[int] | None = None,
        now: float | None = None,
    ) -> list[Experience]:
        """Mix of pools: keys ``recent`` (recency-weighted), ``historical`` (uniform),
        ``rare`` (rare-weighted), ``difficult`` (prioritised), ``regime`` (balanced)."""
        plan: dict[str, tuple[Pool, Strategy]] = {
            "recent": ("recent", "recency"),
            "historical": ("historical", "uniform"),
            "rare": ("rare", "rare"),
            "difficult": ("all", "prioritized"),
            "regime": ("all", "regime_balanced"),
        }
        total = sum(v for v in fractions.values() if v > 0)
        if total <= 0:
            raise ValueError("replay mixture fractions must sum to a positive value")
        out: list[Experience] = []
        fallback = [e for e in self.all() if eligible is None or e.seq in eligible]
        for key, frac in fractions.items():
            if frac <= 0:
                continue
            pool, strategy = plan[key]
            items = [e for e in self.pool(pool) if eligible is None or e.seq in eligible]
            if not items:
                items = fallback  # empty pool (e.g. no rare events yet) → draw from everything
            k = round(n * frac / total)
            got, _ = self.sample(k, strategy, items=items, now=now)
            out.extend(got)
        return out

    # ------------------------------------------------------------------ conversion
    @staticmethod
    def to_arrays(experiences: Sequence[Experience]) -> dict[str, Array]:
        """Concatenate the experiences' rows into one canonical array dict."""
        if not experiences:
            raise ValueError("no experiences")
        return concat_arrays([e.row for e in experiences])

    def to_store(self, experiences: Sequence[Experience] | None = None) -> ArrayStore:
        """In-memory :class:`ArrayStore` of ``experiences`` (default: every buffered experience)."""
        return ArrayStore(self.to_arrays(self.all() if experiences is None else experiences))

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str | Path) -> Path:
        """Persist rows, metadata, embeddings and pool state to ``directory``."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        items = self.all()
        recent = {e.seq for e in self.recent}
        hist = {e.seq for e in self.historical}
        rare = {e.seq for e in self.rare}
        meta = []
        emb_dim = max((len(e.embedding) for e in items if e.embedding is not None), default=0)
        emb = np.full((len(items), emb_dim), np.nan, dtype=np.float32)
        for i, e in enumerate(items):
            if e.embedding is not None and emb_dim:
                emb[i, : len(e.embedding)] = e.embedding
            meta.append(
                {
                    "observation_id": e.observation_id,
                    "timestamp": e.timestamp,
                    "model_version": e.model_version,
                    "prediction": e.prediction,
                    "uncertainty": e.uncertainty,
                    "has_embedding": e.embedding is not None,
                    "regime": e.regime,
                    "priority": e.priority,
                    "is_rare": e.is_rare,
                    "seq": e.seq,
                    "pools": [
                        p for p, s in (("recent", recent), ("historical", hist), ("rare", rare)) if e.seq in s
                    ],
                }
            )
        if items:
            ArrayStore(self.to_arrays(items)).save(path / "rows")
        np.save(path / "embeddings.npy", emb)
        state = {
            "seen_historical": self.seen_historical,
            "next_seq": self.next_seq,
            "max_priority": self.max_priority,
            "recent_order": [e.seq for e in self.recent],
            "historical_order": [e.seq for e in self.historical],
            "rare_order": [e.seq for e in self.rare],
        }
        (path / "experiences.json").write_text(json.dumps(meta))
        (path / "state.json").write_text(json.dumps(state))
        return path

    @classmethod
    def load(cls, directory: str | Path, cfg: ReplayConfig) -> ExperienceReplayBuffer:
        """Restore a buffer written by :meth:`save` (empty if nothing was saved)."""
        path = Path(directory)
        buf = cls(cfg)
        if not (path / "state.json").exists():
            return buf
        state = json.loads((path / "state.json").read_text())
        meta = json.loads((path / "experiences.json").read_text())
        emb = np.load(path / "embeddings.npy")
        by_seq: dict[int, Experience] = {}
        if meta:
            store = ArrayStore.load(path / "rows", mmap=False)
            for i, m in enumerate(meta):
                row = select_rows(store.arrays, np.array([i]))
                by_seq[int(m["seq"])] = Experience(
                    observation_id=m["observation_id"],
                    timestamp=float(m["timestamp"]),
                    row=row,
                    model_version=m["model_version"],
                    prediction=m["prediction"],
                    uncertainty=m["uncertainty"],
                    embedding=emb[i].copy() if m["has_embedding"] else None,
                    regime=m["regime"],
                    priority=m["priority"],
                    is_rare=bool(m["is_rare"]),
                    seq=int(m["seq"]),
                )
        buf.recent = deque(by_seq[s] for s in state["recent_order"])
        buf.historical = [by_seq[s] for s in state["historical_order"]]
        buf.rare = [by_seq[s] for s in state["rare_order"]]
        buf.seen_historical = int(state["seen_historical"])
        buf.next_seq = int(state["next_seq"])
        buf.max_priority = float(state["max_priority"])
        return buf


def canonical_row(row: dict[str, Array], config: NeuralConfig) -> dict[str, Array]:
    """Keep only the keys an experience row may carry (drops diagnostics like ``regime``)."""
    keep = {"observation_id", "timestamp", "current", KEY_TARGET_MASK}
    keep |= {target_key(t) for t in REGRESSION_TASKS}
    out = {k: v for k, v in row.items() if k in keep or k.startswith("seq.")}
    if config.model.graph.enabled:
        graph = {k: v for k, v in row.items() if k.startswith("graph.")}
        if not graph:  # rows without relational context get an explicit empty graph
            n = len(row["observation_id"])
            fdim = config.features.graph.node_feature_dim
            graph = {
                "graph.node_features": np.zeros((0, fdim), np.float32),
                "graph.edge_index": np.zeros((0, 2), np.int64),
                "graph.edge_type": np.zeros((0,), np.int64),
                "graph.node_offsets": np.zeros(n + 1, np.int64),
                "graph.edge_offsets": np.zeros(n + 1, np.int64),
                "graph.target_node": np.full(n, -1, np.int64),
            }
        out |= graph
    return out


def experiences_from_arrays(
    arrays: dict[str, Array], config: NeuralConfig, model_version: str | None = None
) -> list[Experience]:
    """Split a labelled array dataset into per-row experiences (e.g. bulk ingestion)."""
    n = len(arrays["observation_id"])
    out = []
    for i in range(n):
        row = canonical_row(select_rows(arrays, np.array([i])), config)
        exp = Experience(
            observation_id=str(row["observation_id"][0]),
            timestamp=float(row["timestamp"][0]),
            row=row,
            model_version=model_version,
        )
        if KEY_TARGET_MASK not in row:
            row[KEY_TARGET_MASK] = np.ones((1, len(config.targets.horizons)), dtype=bool)
        missing = [t for t in REGRESSION_TASKS if target_key(t) not in row]
        if missing:
            raise ValueError(f"arrays lack targets {missing}; experiences must be labelled")
        exp.is_rare = exp.magnitude > config.replay.rare_return_threshold
        out.append(exp)
    return out
