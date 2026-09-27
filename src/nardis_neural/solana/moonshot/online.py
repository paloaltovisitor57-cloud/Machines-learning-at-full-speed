"""Online moonshot learning for long streams.

While history (or the live chain) streams through :class:`SolanaBrain`, the tracker:

1. **samples** a ticket candidate for every token inside the entry window at most every
   ``sample_every`` seconds (raw on-chain features at that instant — exactly what the live
   model sees);
2. **resolves** a token's rows once its run is over (quiet for ``resolve_idle_seconds``) or
   its horizon has elapsed, labelling them with :func:`moonshot_outcome` on the data seen
   so far, and moves them into a bounded FIFO training buffer;
3. provides a **causal training set** at any time: resolved rows plus every still-open row
   labelled *now* as right-censored, which is exactly the censored-likelihood setting the
   tail model was built for.

Nothing is kept per token after resolution except the rows themselves, so memory is
bounded by ``capacity`` regardless of how much history streams past.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from nardis_neural.solana.market import SolanaMarket
from nardis_neural.solana.moonshot.labels import MoonshotSpec, moonshot_outcome

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]


@dataclass
class TrainingSet:
    x: F32
    peak: F64
    censored: npt.NDArray[np.bool_]
    t: F64
    mints: npt.NDArray[np.str_]

    def __len__(self) -> int:
        return len(self.peak)

    @property
    def tokens(self) -> int:
        return len(set(self.mints.tolist()))


@dataclass
class MoonshotTracker:
    spec: MoonshotSpec = field(default_factory=MoonshotSpec)
    sample_every: float = 30.0
    capacity: int = 200_000
    pending: dict[str, list[tuple[float, F32]]] = field(default_factory=dict)
    last_sample: dict[str, float] = field(default_factory=dict)
    resolved: deque[tuple[str, float, F32, float, bool]] = field(default_factory=deque)
    resolved_tokens: int = 0

    def observe(self, mints: list[str], current: F32, ages: F64, now: float) -> int:
        """Record a candidate row for every token in the entry window (rate-limited)."""
        added = 0
        lo, hi = self.spec.min_entry_age_seconds, self.spec.max_entry_age_seconds
        for i, mint in enumerate(mints):
            if not lo <= ages[i] <= hi or now - self.last_sample.get(mint, -np.inf) < self.sample_every:
                continue
            self.last_sample[mint] = now
            self.pending.setdefault(mint, []).append((now, np.asarray(current[i], dtype=np.float32).copy()))
            added += 1
        return added

    def _done(self, market: SolanaMarket, mint: str, now: float) -> bool:
        log = market.tokens.get(mint)
        if log is None:
            return True
        rows = self.pending[mint]
        horizon_over = now >= rows[-1][0] + self.spec.horizon_seconds + 2 * self.spec.latency_seconds
        return horizon_over or now - log.last_t >= self.spec.resolve_idle_seconds

    def resolve(self, market: SolanaMarket, now: float, force: set[str] | None = None) -> int:
        """Label and archive the rows of every token whose run is over (or in ``force``)."""
        done = [m for m in self.pending if (force is not None and m in force) or self._done(market, m, now)]
        n = 0
        for mint in done:
            rows = self.pending.pop(mint)
            self.last_sample.pop(mint, None)
            log = market.tokens.get(mint)
            if log is None:
                continue  # evicted before it could be labelled: nothing trustworthy to keep
            for t, x in rows:
                out = moonshot_outcome(log, t, self.spec, now)
                if out is not None:
                    self.resolved.append((mint, t, x, out.peak_multiple, out.censored))
                    n += 1
            self.resolved_tokens += 1
        while len(self.resolved) > self.capacity:
            self.resolved.popleft()
        return n

    def training_set(self, market: SolanaMarket, now: float) -> TrainingSet | None:
        """Resolved rows plus open rows labelled at ``now`` (censored while still running)."""
        rows: list[tuple[str, float, F32, float, bool]] = list(self.resolved)
        for mint, pend in self.pending.items():
            log = market.tokens.get(mint)
            if log is None:
                continue
            for t, x in pend:
                out = moonshot_outcome(log, t, self.spec, now)
                if out is not None:
                    rows.append((mint, t, x, out.peak_multiple, out.censored))
        if not rows:
            return None
        return TrainingSet(
            x=np.stack([r[2] for r in rows]).astype(np.float32),
            peak=np.asarray([r[3] for r in rows], dtype=np.float64),
            censored=np.asarray([r[4] for r in rows], dtype=bool),
            t=np.asarray([r[1] for r in rows], dtype=np.float64),
            mints=np.asarray([r[0] for r in rows]),
        )

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        pend = [(m, t, x) for m, rows in self.pending.items() for t, x in rows]
        res = list(self.resolved)
        d = int(res[0][2].shape[0]) if res else int(pend[0][2].shape[0]) if pend else 0
        np.savez(
            path,
            r_mint=np.asarray([r[0] for r in res], dtype=str),
            r_t=np.asarray([r[1] for r in res], dtype=np.float64),
            r_x=np.stack([r[2] for r in res]) if res else np.zeros((0, d), np.float32),
            r_peak=np.asarray([r[3] for r in res], dtype=np.float64),
            r_cens=np.asarray([r[4] for r in res], dtype=bool),
            p_mint=np.asarray([p[0] for p in pend], dtype=str),
            p_t=np.asarray([p[1] for p in pend], dtype=np.float64),
            p_x=np.stack([p[2] for p in pend]) if pend else np.zeros((0, d), np.float32),
            meta=np.asarray([self.sample_every, self.capacity, self.resolved_tokens], dtype=np.float64),
        )

    @classmethod
    def load(cls, path: str | Path, spec: MoonshotSpec) -> MoonshotTracker:
        with np.load(path) as z:
            every, cap, n_tok = z["meta"].tolist()
            tr = cls(spec, sample_every=float(every), capacity=int(cap))
            tr.resolved_tokens = int(n_tok)
            for m, t, x, p, c in zip(z["r_mint"], z["r_t"], z["r_x"], z["r_peak"], z["r_cens"], strict=True):
                tr.resolved.append((str(m), float(t), x.astype(np.float32), float(p), bool(c)))
            for m, t, x in zip(z["p_mint"], z["p_t"], z["p_x"], strict=True):
                tr.pending.setdefault(str(m), []).append((float(t), x.astype(np.float32)))
                tr.last_sample[str(m)] = max(tr.last_sample.get(str(m), -np.inf), float(t))
        return tr
