"""Wallet intelligence: funding clusters (sybil / bundle detection) and reputations.

* **Funding graph** — SOL transfers above ``funding_min_sol`` link funder and funded
  wallet in a union-find structure, so wallets bankrolled by the same source collapse into
  one *cluster*.  Sources that fund many distinct wallets (exchange hot wallets, faucets)
  are treated as *hubs* and stop merging clusters, otherwise one CEX would glue the whole
  market together.
* **Reputation** — a Beta-Bernoulli posterior per wallet over "an entry by this wallet was
  followed by a gain above the success threshold".  Updates are only ever applied when an
  outcome has *resolved*, so reputations used at time ``t`` contain no future
  information.  A rug counter tracks association with rugged launches.
"""

from __future__ import annotations

import json
from array import array
from pathlib import Path
from typing import Any

import numpy as np

from nardis_neural.solana.events import Transfer


class WalletIntel:
    def __init__(
        self,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        funding_min_sol: float = 0.05,
        hub_threshold: int = 50,
    ) -> None:
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self.funding_min_sol = funding_min_sol
        self.hub_threshold = hub_threshold
        self.ids: dict[str, int] = {}
        self.names: list[str] = []
        # compact typed arrays: millions of wallets stream through a long replay
        self.parent: array[int] = array("q")
        self.size: array[int] = array("q")
        self.alpha: array[float] = array("d")
        self.beta: array[float] = array("d")
        self.trades: array[int] = array("q")
        self.first_seen: array[float] = array("d")
        self.last_seen: array[float] = array("d")
        self.rugs: array[int] = array("q")
        self.funder: dict[int, int] = {}
        self.funded_at: dict[int, float] = {}
        self.funded_count: dict[int, int] = {}

    # ------------------------------------------------------------------ identity
    def id(self, name: str, t: float = 0.0) -> int:
        wid = self.ids.get(name)
        if wid is None:
            wid = len(self.names)
            self.ids[name] = wid
            self.names.append(name)
            self.parent.append(wid)
            self.size.append(1)
            self.alpha.append(self.prior_alpha)
            self.beta.append(self.prior_beta)
            self.trades.append(0)
            self.first_seen.append(t)
            self.last_seen.append(t)
            self.rugs.append(0)
        return wid

    def __len__(self) -> int:
        return len(self.names)

    # ------------------------------------------------------------------ funding graph
    def root(self, wid: int) -> int:
        while self.parent[wid] != wid:
            self.parent[wid] = self.parent[self.parent[wid]]
            wid = self.parent[wid]
        return wid

    def _union(self, a: int, b: int) -> None:
        ra, rb = self.root(a), self.root(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def add_transfer(self, tr: Transfer) -> None:
        if tr.sol_amount < self.funding_min_sol:
            return
        src, dst = self.id(tr.source, tr.t), self.id(tr.dest, tr.t)
        if dst not in self.funder:
            self.funder[dst] = src
            self.funded_at[dst] = tr.t
            self.funded_count[src] = self.funded_count.get(src, 0) + 1
        if self.funded_count.get(src, 0) <= self.hub_threshold:
            self._union(src, dst)

    def cluster_size(self, wid: int) -> int:
        return self.size[self.root(wid)]

    def same_cluster(self, a: int, b: int) -> bool:
        return self.root(a) == self.root(b)

    def is_fresh(self, wid: int, t: float, window: float) -> bool:
        """Wallet first funded less than ``window`` seconds before ``t``."""
        at = self.funded_at.get(wid)
        return at is not None and 0.0 <= t - at <= window

    # ------------------------------------------------------------------ activity & reputation
    def record_trade(self, wid: int, t: float) -> None:
        self.trades[wid] += 1
        self.last_seen[wid] = t

    def update(self, wid: int, success: bool, weight: float = 1.0) -> None:
        if success:
            self.alpha[wid] += weight
        else:
            self.beta[wid] += weight

    def mark_rug(self, wid: int) -> None:
        self.rugs[wid] += 1

    def score(self, wid: int) -> float:
        """Posterior mean success probability."""
        a, b = self.alpha[wid], self.beta[wid]
        return a / (a + b)

    def evidence(self, wid: int) -> float:
        return self.alpha[wid] + self.beta[wid] - self.prior_alpha - self.prior_beta

    def skill(self, wid: int) -> float:
        """Evidence-shrunk edge in [-1, 1]: (score − prior mean) scaled by confidence."""
        prior = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        ev = self.evidence(wid)
        return float(np.clip((self.score(wid) - prior) * 2.0 * ev / (ev + 5.0), -1.0, 1.0))

    def scores(self, wids: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        a = np.frombuffer(self.alpha, dtype=np.float64)[wids]  # zero-copy view, indexed at once
        b = np.frombuffer(self.beta, dtype=np.float64)[wids]
        return np.asarray(a / (a + b), dtype=np.float64)

    def skills(self, wids: np.ndarray[Any, np.dtype[np.int64]]) -> np.ndarray[Any, np.dtype[np.float64]]:
        """Vectorised :meth:`skill`."""
        a = np.frombuffer(self.alpha, dtype=np.float64)[wids]
        b = np.frombuffer(self.beta, dtype=np.float64)[wids]
        prior = self.prior_alpha / (self.prior_alpha + self.prior_beta)
        ev = a + b - self.prior_alpha - self.prior_beta
        return np.asarray(np.clip((a / (a + b) - prior) * 2.0 * ev / (ev + 5.0), -1.0, 1.0), dtype=np.float64)

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict[str, Any]:
        return {
            "params": [self.prior_alpha, self.prior_beta, self.funding_min_sol, self.hub_threshold],
            "names": self.names,
            "parent": self.parent.tolist(),
            "size": self.size.tolist(),
            "alpha": self.alpha.tolist(),
            "beta": self.beta.tolist(),
            "trades": self.trades.tolist(),
            "first_seen": self.first_seen.tolist(),
            "last_seen": self.last_seen.tolist(),
            "rugs": self.rugs.tolist(),
            "funder": [[k, v] for k, v in self.funder.items()],
            "funded_at": [[k, v] for k, v in self.funded_at.items()],
            "funded_count": [[k, v] for k, v in self.funded_count.items()],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> WalletIntel:
        pa, pb, fmin, hub = d["params"]
        w = cls(float(pa), float(pb), float(fmin), int(hub))
        w.names = list(d["names"])
        w.ids = {n: i for i, n in enumerate(w.names)}
        w.parent, w.size = array("q", d["parent"]), array("q", d["size"])
        w.alpha, w.beta = array("d", d["alpha"]), array("d", d["beta"])
        w.trades, w.rugs = array("q", d["trades"]), array("q", d["rugs"])
        w.first_seen, w.last_seen = array("d", d["first_seen"]), array("d", d["last_seen"])
        w.funder = {int(k): int(v) for k, v in d["funder"]}
        w.funded_at = {int(k): float(v) for k, v in d["funded_at"]}
        w.funded_count = {int(k): int(v) for k, v in d["funded_count"]}
        return w

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()))

    @classmethod
    def load(cls, path: str | Path) -> WalletIntel:
        return cls.from_dict(json.loads(Path(path).read_text()))
