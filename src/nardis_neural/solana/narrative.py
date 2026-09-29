"""Narratives: memecoins run in themes.

When a cat token runs, the next cat tokens get bought; when a name is hot, copycats of it
launch within minutes.  This module turns token names and symbols into causal signals:

* **word heat**: every word of a token's name and symbol gets credit each time that token
  climbs a chase rung (2x, 5x, 10x, 100x, 1000x vs its launch price; more for higher rungs).
  The credit decays with a half-life (30 min by default).  A token's ``narrative_heat_log`` is
  the heat of its hottest word, **excluding the credit from its own run**, so a token never
  scores itself;
* **copycats**: how many tokens with the same normalised symbol or name launched in the last
  hour (``name_copycats_3600s_log``);
* **copies a recent runner**: 1 when its symbol or name matches a token (other than itself)
  that reached 5x in the last 6 hours (``copies_recent_runner``).

Everything is updated from events in time order, so a snapshot at ``t`` only sees credit
earned before ``t``.
"""

from __future__ import annotations

import re
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

STOPWORDS = frozenset(
    {
        "the", "and", "for", "coin", "token", "sol", "solana", "pump", "fun", "official", "with", "from",
        "this", "that", "you", "your", "our", "are", "was", "not", "but", "all", "new", "just", "get",
    }
)  # fmt: skip
RUNGS = (2.0, 5.0, 10.0, 100.0, 1000.0)
RUNG_CREDIT = (1.0, 2.0, 4.0, 8.0, 16.0)
_WORD = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    """Lower-case alphanumerics only (``"$PEPE 2.0"`` → ``"pepe20"``)."""
    return "".join(_WORD.findall(text.lower()))


def name_words(name: str, symbol: str) -> tuple[str, ...]:
    """Distinct theme words of a token: name words of 3+ characters (no stopwords) and its symbol."""
    words = [w for w in _WORD.findall(name.lower()) if len(w) >= 3 and w not in STOPWORDS]
    sym = normalise(symbol)
    if len(sym) >= 2 and sym not in STOPWORDS:
        words.append(sym)
    return tuple(dict.fromkeys(words))


@dataclass
class NarrativeBook:
    """Causal word heat, copycat counts and recent-runner names across the market."""

    half_life: float = 1800.0
    copycat_window: float = 3600.0
    runner_memory: float = 6 * 3600.0
    runner_rung: float = 5.0
    heat: dict[str, tuple[float, float]] = field(default_factory=dict)
    """word → (value, time of value)."""
    words: dict[str, tuple[str, ...]] = field(default_factory=dict)
    keys: dict[str, tuple[str, str, float]] = field(default_factory=dict)
    """mint → (normalised symbol, normalised name, launch time)."""
    own: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    """mint → credits (time, amount) its own run added to each of its words."""
    rung: dict[str, int] = field(default_factory=dict)
    launches: deque[tuple[float, str, str]] = field(default_factory=deque)
    counts: Counter[str] = field(default_factory=Counter)
    runners: dict[str, tuple[float, str]] = field(default_factory=dict)
    """normalised symbol / name → (time it reached the runner rung, mint)."""

    def _decay(self, value: float, dt: float) -> float:
        return float(value * 0.5 ** (max(dt, 0.0) / self.half_life))

    def on_launch(self, mint: str, t: float, name: str, symbol: str) -> None:
        """Register a launch (its theme words and copycat keys)."""
        self.words[mint] = name_words(name, symbol)
        sym, nm = normalise(symbol), normalise(name)
        self.keys[mint] = (sym, nm, t)
        self.rung[mint] = 0
        self._expire(t)
        for k in {f"s:{sym}" if sym else "", f"n:{nm}" if nm else ""} - {""}:
            self.launches.append((t, mint, k))
            self.counts[k] += 1

    def _expire(self, now: float) -> None:
        while self.launches and now - self.launches[0][0] > self.copycat_window:
            _, _, k = self.launches.popleft()
            self.counts[k] -= 1
            if self.counts[k] <= 0:
                del self.counts[k]

    def on_peak(self, mint: str, t: float, peak_multiple: float) -> None:
        """Credit the token's words when its running peak (vs launch) climbs a new rung."""
        r = self.rung.get(mint)
        if r is None:
            return
        new = sum(1 for k in RUNGS if peak_multiple >= k)
        if new <= r:
            return
        amount = float(sum(RUNG_CREDIT[r:new]))
        self.rung[mint] = new
        for w in self.words.get(mint, ()):
            v, tv = self.heat.get(w, (0.0, t))
            self.heat[w] = (self._decay(v, t - tv) + amount, t)
        self.own.setdefault(mint, []).append((t, amount))
        if peak_multiple >= self.runner_rung:
            sym, nm, _ = self.keys[mint]
            for key in (f"s:{sym}" if sym else "", f"n:{nm}" if nm else ""):
                if key and key not in self.runners:
                    self.runners[key] = (t, mint)

    def features(self, mint: str, now: float) -> dict[str, float]:
        """``narrative_heat_log``, ``name_copycats_3600s_log`` and ``copies_recent_runner`` for ``mint``."""
        words = self.words.get(mint)
        if words is None:
            return {"narrative_heat_log": 0.0, "name_copycats_3600s_log": 0.0, "copies_recent_runner": 0.0}
        own = sum(self._decay(a, now - t) for t, a in self.own.get(mint, ()))
        best = 0.0
        for w in words:
            v, tv = self.heat.get(w, (0.0, now))
            best = max(best, self._decay(v, now - tv) - own)
        self._expire(now)
        sym, nm, t_launch = self.keys[mint]
        mine = 1 if now - t_launch <= self.copycat_window else 0
        copies = max(
            self.counts.get(f"s:{sym}", 0) - mine if sym else 0,
            self.counts.get(f"n:{nm}", 0) - mine if nm else 0,
        )
        runner = 0.0
        for key in (f"s:{sym}" if sym else "", f"n:{nm}" if nm else ""):
            hit = self.runners.get(key) if key else None
            if hit is not None and hit[1] != mint and now - hit[0] <= self.runner_memory:
                runner = 1.0
        return {
            "narrative_heat_log": float(np.log1p(max(best, 0.0))),
            "name_copycats_3600s_log": float(np.log1p(max(copies, 0))),
            "copies_recent_runner": runner,
        }

    def forget(self, mint: str, now: float) -> None:
        """Drop a token's own records (its credit stays in the word heat); prunes cold words."""
        for d in (self.words, self.keys, self.own, self.rung):
            d.pop(mint, None)
        if len(self.heat) > 20_000:
            self.heat = {w: (v, t) for w, (v, t) in self.heat.items() if self._decay(v, now - t) > 1e-3}
        for key in [k for k, (t, _) in self.runners.items() if now - t > self.runner_memory]:
            del self.runners[key]
