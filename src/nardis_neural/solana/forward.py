"""Forward-test ledger: what the ML signals would have earned, recorded as they fire.

Every assessment round, the ledger:

* **opens** a paper ticket on a token the first time it is inside the moonshot entry window,
  is not vetoed by the manipulation guard, and its expected ladder payoff is at least the
  ticket (with a positive lottery-Kelly fraction);
* **arms the exit alarm** the first time the Tape Transformer's collapse probability for the
  chosen window reaches the chosen threshold (both taken from the tape research report);
* **settles** a ticket once its run is over (horizon elapsed or the token went quiet), by
  simulating the ladder plus the alarm on the real pool path observed since entry — the
  same executable maths as the research, but with nothing chosen in hindsight.

Run inside ``stream-train`` it is a walk-forward backtest of the live system over the
streamed history; run live it is the paper-trading scorecard of the signals.  Paper only —
it never places an order.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nardis_neural.solana.market import SolanaMarket
from nardis_neural.solana.moonshot.labels import MoonshotSpec, moonshot_outcome
from nardis_neural.solana.moonshot.research import ticket_stats


@dataclass
class Ticket:
    """One paper ticket and the signals it was opened on."""

    mint: str
    t_entry: float
    expected_multiple: float
    chase_score: float
    p_ge_10x: float
    trust: float
    exit_at: float | None = None
    ladder_multiple: float | None = None
    peak_multiple: float | None = None
    settled_at: float | None = None


class ForwardLedger:
    """Paper tickets opened and settled from live (or replayed) assessments."""

    def __init__(
        self,
        path: str | Path,
        spec: MoonshotSpec,
        alarm: tuple[str, float] | None = None,
        min_expected_multiple: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.spec = spec
        self.alarm = alarm
        """(tape key such as ``p_collapse_5m``, threshold) or None for the ladder alone."""
        self.min_expected_multiple = min_expected_multiple
        self.open: dict[str, Ticket] = {}
        self.closed: list[Ticket] = []
        self.seen: set[str] = set()

    # ------------------------------------------------------------------ live updates
    def observe(self, reports: list[Any], now: float) -> int:
        """Open tickets and arm exit alarms from a round of ``SolanaAssessment`` objects."""
        opened = 0
        for a in reports:
            m = a.moonshot
            if not m:
                continue
            ticket = self.open.get(a.mint)
            if ticket is not None:
                if self.alarm is not None and ticket.exit_at is None and a.tape:
                    key, theta = self.alarm
                    if a.tape.get(key, 0.0) >= theta:
                        ticket.exit_at = now
                continue
            if a.mint in self.seen or not m.get("in_entry_window"):
                continue
            if (
                m.get("vetoed")
                or m["expected_multiple"] < self.min_expected_multiple
                or m["lottery_kelly"] <= 0
            ):
                continue
            self.seen.add(a.mint)
            self.open[a.mint] = Ticket(
                a.mint,
                now,
                float(m["expected_multiple"]),
                float(m.get("chase_score", 0.0)),
                float(m.get("p_ge_10x", 0.0)),
                float(m.get("trust", 1.0)),
            )
            opened += 1
        return opened

    def settle(self, market: SolanaMarket, now: float, force: bool = False) -> int:
        """Settle tickets whose run is over (or all of them with ``force``)."""
        done = 0
        horizon = self.spec.horizon_seconds + 2 * self.spec.latency_seconds
        for mint, tk in list(self.open.items()):
            log = market.tokens.get(mint)
            if log is None:
                del self.open[mint]  # token forgotten before settlement: nothing honest to record
                continue
            over = now >= tk.t_entry + horizon or now - log.last_t >= self.spec.resolve_idle_seconds
            if not (over or force):
                continue
            out = moonshot_outcome(log, tk.t_entry, self.spec, now, exit_at=tk.exit_at)
            del self.open[mint]
            if out is None:
                continue
            tk.ladder_multiple, tk.peak_multiple, tk.settled_at = out.ladder_multiple, out.peak_multiple, now
            self.closed.append(tk)
            done += 1
        return done

    # ------------------------------------------------------------------ reporting
    def summary(self) -> dict[str, Any]:
        """Realised ticket statistics and how well the entry signals ranked the outcomes."""
        if not self.closed:
            return {"tickets": 0, "open": len(self.open)}
        lad = np.asarray([t.ladder_multiple for t in self.closed], dtype=np.float64)
        peak = np.asarray([t.peak_multiple for t in self.closed], dtype=np.float64)
        p10 = np.asarray([t.p_ge_10x for t in self.closed])
        out: dict[str, Any] = ticket_stats(lad, peak, self.spec.size_sol)
        out |= {
            "open": len(self.open),
            "alarm_exits": int(sum(t.exit_at is not None for t in self.closed)),
            "mean_predicted_p_ge_10x": float(p10.mean()),
            "observed_peak_ge_10x": float((peak >= 10).mean()),
        }
        score = np.asarray([t.chase_score for t in self.closed])
        if len(score) >= 10:
            top = score >= np.quantile(score, 0.8)
            out["top_quintile_mean_multiple"] = float(lad[top].mean())
            out["bottom_quintiles_mean_multiple"] = float(lad[~top].mean()) if (~top).any() else float("nan")
        return out

    def save(self) -> None:
        """Write the ledger (open and closed tickets) and its summary as JSON."""
        self.path.mkdir(parents=True, exist_ok=True)
        state = {
            "alarm": list(self.alarm) if self.alarm else None,
            "min_expected_multiple": self.min_expected_multiple,
            "open": [asdict(t) for t in self.open.values()],
            "closed": [asdict(t) for t in self.closed],
            "seen": sorted(self.seen),
        }
        (self.path / "ledger.json").write_text(json.dumps(state, default=float))
        (self.path / "summary.json").write_text(json.dumps(self.summary(), indent=2, default=float))

    @classmethod
    def load(
        cls, path: str | Path, spec: MoonshotSpec, alarm: tuple[str, float] | None = None
    ) -> ForwardLedger:
        """Restore a saved ledger (or start an empty one); ``alarm`` overrides the saved alarm."""
        p = Path(path)
        f = p / "ledger.json"
        if not f.exists():
            return cls(p, spec, alarm)
        state = json.loads(f.read_text())
        saved = tuple(state["alarm"]) if state.get("alarm") else None
        led = cls(
            p,
            spec,
            alarm or (str(saved[0]), float(saved[1])) if saved else alarm,
            state["min_expected_multiple"],
        )
        led.open = {d["mint"]: Ticket(**d) for d in state["open"]}
        led.closed = [Ticket(**d) for d in state["closed"]]
        led.seen = set(state["seen"])
        return led
