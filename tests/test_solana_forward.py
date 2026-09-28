"""Forward-test ledger: tickets open on live signals, alarms arm, settlement is executable."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from nardis_neural.solana.forward import ForwardLedger
from nardis_neural.solana.moonshot import MoonshotSpec
from tests.test_solana_edge import T0, _market

SPEC = MoonshotSpec(size_sol=0.2, horizon_seconds=600, min_entry_age_seconds=0, trail_activation=50.0)


def _report(
    mint: str, ev: float, collapse: float, vetoed: bool = False, window: bool = True
) -> SimpleNamespace:
    moon = {
        "in_entry_window": float(window),
        "vetoed": float(vetoed),
        "expected_multiple": ev,
        "lottery_kelly": 0.01 if ev >= 1 else 0.0,
        "chase_score": ev,
        "p_ge_10x": 0.1,
        "trust": 0.9,
    }
    return SimpleNamespace(mint=mint, moonshot=moon, tape={"p_collapse_1m": collapse})


def test_ledger_opens_alarms_settles_and_roundtrips(tmp_path: Path) -> None:
    pump = [(1, "a", True, 0.5)] + [(10 + i, f"b{i}", True, 2.0) for i in range(10)]
    dump = [(100 + i, f"b{i}", False, 1.0) for i in range(10)]
    m = _market(pump + dump)
    led = ForwardLedger(tmp_path / "fwd", SPEC, alarm=("p_collapse_1m", 0.5))
    assert led.observe([_report("TOK", 0.5, 0.0)], T0 + 2) == 0, "expected payoff below the ticket"
    assert led.observe([_report("TOK", 3.0, 0.0, vetoed=True)], T0 + 2) == 0, "vetoed"
    assert led.observe([_report("TOK", 3.0, 0.0)], T0 + 2) == 1
    assert led.observe([_report("TOK", 3.0, 0.0)], T0 + 3) == 0, "one ticket per token"
    led.observe([_report("TOK", 3.0, 0.9)], T0 + 30)
    assert led.open["TOK"].exit_at == T0 + 30
    assert led.settle(m, T0 + 60) == 0, "the run is not over yet"
    assert led.settle(m, T0 + 5000) == 1 and not led.open
    tk = led.closed[0]
    assert tk.ladder_multiple is not None and tk.ladder_multiple > 1, (
        "the alarm sold near the top of the pump"
    )
    assert tk.ladder_only_multiple is not None and tk.ladder_only_multiple < tk.ladder_multiple
    s = led.summary()
    assert s["tickets"] == 1 and s["alarm_exits"] == 1
    assert s["alarm_minus_ladder_pnl_sol"] > 0, "both variants are settled, so the alarm can be judged"
    led.save()
    back = ForwardLedger.load(tmp_path / "fwd", SPEC)
    assert back.alarm == ("p_collapse_1m", 0.5) and len(back.closed) == 1 and "TOK" in back.seen
