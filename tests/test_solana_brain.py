"""End-to-end Solana brain: bootstrap from history → stream live events → assess → resolve
→ continual learning → restart; plus the ``nardis-neural solana`` CLI."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from nardis_neural.cli import app
from nardis_neural.config import NeuralConfig
from nardis_neural.solana import EventStore, LaunchSimSpec, SolanaBrain, SolanaConfig, simulate_launches
from nardis_neural.solana.config import CURRENT_FEATURES, RISK_LABELS
from tests.conftest import make_tiny_config

T_HIST = 1_750_000_000.0
T_LIVE = T_HIST + 3 * 3600


def _tiny() -> NeuralConfig:
    cfg = make_tiny_config()
    cfg.training.epochs = 2
    cfg.continual.min_new_samples = 100
    return cfg


def _sol_cfg() -> SolanaConfig:
    return SolanaConfig(sample_interval_seconds=20.0, graph_top_k=8)


@pytest.fixture(scope="module")
def history() -> EventStore:
    return simulate_launches(
        LaunchSimSpec(
            n_tokens=12, seed=3, n_retail=250, duration_seconds=2 * 3600, mint_prefix="H", start_time=T_HIST
        )
    )[0]


@pytest.fixture(scope="module")
def live() -> tuple[EventStore, dict[str, str]]:
    return simulate_launches(
        LaunchSimSpec(
            n_tokens=4, seed=4, n_retail=250, duration_seconds=1200, mint_prefix="L", start_time=T_LIVE
        )
    )


@pytest.fixture(scope="module")
def brain_ws(tmp_path_factory: pytest.TempPathFactory, history: EventStore) -> Path:
    ws = tmp_path_factory.mktemp("sol") / "ws"
    SolanaBrain.bootstrap(ws, history, _sol_cfg(), _tiny(), device="cpu")
    return ws


def test_bootstrap_creates_complete_workspace(brain_ws: Path) -> None:
    for name in ("solana.yaml", "registry.json", "events", "risk/risk.json", "config.yaml"):
        assert (brain_ws / name).exists(), name
    brain = SolanaBrain(brain_ws, device="cpu")
    assert brain.risk is not None and set(brain.risk.report["metrics"]) == set(RISK_LABELS)
    engine = brain.learner.champion
    assert engine.config.features.current_dim == len(CURRENT_FEATURES)
    assert "graph" in engine.expert_names
    assert brain.market.tokens, "market state is rebuilt from the stored history"


def test_live_loop_assess_resolve_learn(
    tmp_path: Path, brain_ws: Path, live: tuple[EventStore, dict[str, str]]
) -> None:
    ws = tmp_path / "ws"
    shutil.copytree(brain_ws, ws)
    brain = SolanaBrain(ws, device="cpu")
    store, _ = live
    events = [e for e in store.sorted() if e.t >= T_LIVE - 3600]
    reports = []
    resolved = 0
    next_round = events[0].t + 10
    for e in events:
        brain.ingest(e)
        if e.t >= next_round:
            next_round = e.t + 10
            reports += brain.assess_active()
            resolved += brain.resolve()
    brain.market.advance(brain.market.now + 400)
    assert len(reports) > 20 and resolved > 0
    r = reports[len(reports) // 2]
    assert set(r.risk) == set(RISK_LABELS) and all(0 <= v <= 1 for v in r.risk.values())
    assert set(r.expected_net_return) == {"15s", "60s", "5m"}
    for h in r.prediction.horizons:
        mu, sd = r.prediction.expected_returns[h], r.prediction.return_std[h]
        assert r.expected_net_return[h] == pytest.approx(math.expm1(mu + 0.5 * sd**2) - r.round_trip_cost)
        assert 0 <= r.prob_net_positive[h] <= 1
    assert set(r.features) == set(CURRENT_FEATURES) and isinstance(r.flags, list)
    forbidden = {"action", "signal", "buy", "sell", "order", "side", "size", "position"}
    assert not forbidden & set(r.model_dump())
    assert len(brain.learner.buffer) == resolved
    assert brain.risk_y, "risk samples are collected from resolved outcomes"
    status = brain.maintenance(risk_refit_min_new=10)
    assert status["champion"] is not None and "adapted" in status
    # restart: market state and assessments are reproduced from the workspace
    again = SolanaBrain(ws, device="cpu")
    assert again.market.now == pytest.approx(max(e.t for e in brain.history.events))
    assert set(again.market.tokens) == set(brain.market.tokens)
    mint = r.mint
    a1 = again.assess(mint)
    a2 = again.assess(mint)
    assert a1.prediction.expected_returns == a2.prediction.expected_returns


def test_flags_surface_red_flags(brain_ws: Path, live: tuple[EventStore, dict[str, str]]) -> None:
    store, arch = live
    brain = SolanaBrain(brain_ws, device="cpu")
    rug_mints = [m for m, a in arch.items() if a == "rug"]
    assert rug_mints, "seed 4 contains a rug launch"
    events = [e for e in store.sorted() if e.t >= T_LIVE - 3600]
    seen_flags: list[str] = []
    for e in events:
        brain.ingest(e)
        if rug_mints and e.t > T_LIVE + 5 and rug_mints[0] in brain.market.tokens:
            log = brain.market.token(rug_mints[0])
            if brain.market.now - log.launch.t > 5 and len(log.swaps) > 5:
                seen_flags = brain.assess(rug_mints[0]).flags
                break
    assert any("bundled" in f or "cluster" in f or "authority" in f or "holders" in f for f in seen_flags)


def test_solana_cli_cycle(tmp_path: Path) -> None:
    runner = CliRunner()

    def run(*args: str | Path) -> str:
        res = runner.invoke(app, [str(a) for a in args], catch_exceptions=False)
        assert res.exit_code == 0, res.output
        return res.output

    base = _tiny()
    base.ensemble.size = 1
    base.save(tmp_path / "tiny.yaml")
    _sol_cfg().save(tmp_path / "sol.yaml")
    run("solana", "init-config", "--out", tmp_path / "default_sol.yaml")
    run(
        "solana",
        "simulate",
        "--out",
        tmp_path / "hist",
        "--tokens",
        "8",
        "--seed",
        "1",
        "--prefix",
        "H",
        "--hours",
        "1.5",
    )
    assert json.loads((tmp_path / "hist" / "archetypes.json").read_text())
    run(
        "solana",
        "simulate",
        "--out",
        tmp_path / "live",
        "--tokens",
        "2",
        "--seed",
        "2",
        "--prefix",
        "L",
        "--start-time",
        str(T_HIST + 2 * 3600),
        "--hours",
        "0.3",
    )
    out = run(
        "solana",
        "build-dataset",
        "--events",
        tmp_path / "hist",
        "--out",
        tmp_path / "ds",
        "--solana-config",
        tmp_path / "sol.yaml",
        "--config",
        tmp_path / "tiny.yaml",
    )
    assert "snapshots" in out and (tmp_path / "ds" / "solana_labels.npz").exists()
    run(
        "solana",
        "bootstrap",
        "--events",
        tmp_path / "hist",
        "--workspace",
        tmp_path / "ws",
        "--solana-config",
        tmp_path / "sol.yaml",
        "--config",
        tmp_path / "tiny.yaml",
        "--epochs",
        "1",
        "--device",
        "cpu",
    )
    out = run(
        "solana",
        "replay",
        "--workspace",
        tmp_path / "ws",
        "--events",
        tmp_path / "live",
        "--out",
        tmp_path / "assessments.jsonl",
        "--device",
        "cpu",
    )
    summary = json.loads(out[out.index("{") :])
    assert summary["assessments"] > 0
    lines = (tmp_path / "assessments.jsonl").read_text().strip().splitlines()
    assert lines and "risk" in json.loads(lines[0])
    out = run("solana", "assess", "--workspace", tmp_path / "ws", "--device", "cpu")
    first = json.loads(out.strip().splitlines()[0])
    assert {"mint", "risk", "expected_net_return", "flags"} <= set(first)
    assert np.isfinite(first["round_trip_cost"])


def _copy(brain_ws: Path, tmp_path: Path) -> Path:
    ws = tmp_path / "ws"
    shutil.copytree(brain_ws, ws)
    return ws


def test_maintenance_survives_a_learner_that_cannot_adapt_and_still_saves(
    tmp_path: Path, brain_ws: Path
) -> None:
    ws = _copy(brain_ws, tmp_path)
    brain = SolanaBrain(ws, device="cpu")
    brain.learner.state.new_since_adapt = 10**6  # due, but the replay buffer holds nothing new
    (ws / "solana_state.json").unlink(missing_ok=True)
    status = brain.maintenance()
    assert status["adapted"] is None and "need at least 10" in status["adapt_error"]
    assert status["full_retrain_error"] is None
    assert (ws / "solana_state.json").exists(), "the save still runs"


def test_state_file_is_strict_json_and_old_infinity_still_loads(tmp_path: Path, brain_ws: Path) -> None:
    ws = _copy(brain_ws, tmp_path)
    brain = SolanaBrain(ws, device="cpu")
    brain.save()

    def strict(name: str) -> float:
        raise ValueError(f"non-standard JSON constant {name}")

    state = json.loads((ws / "solana_state.json").read_text(), parse_constant=strict)
    assert state["moonshot_last_refit"] is None
    assert SolanaBrain(ws, device="cpu").moonshot_last_refit == -np.inf
    state["moonshot_last_refit"] = -np.inf  # written by older versions as -Infinity
    (ws / "solana_state.json").write_text(json.dumps(state))
    assert SolanaBrain(ws, device="cpu").moonshot_last_refit == -np.inf
    assert not list(ws.glob("*.tmp")), "written through a temporary file"


def test_edge_model_without_its_research_report_still_loads(
    tmp_path: Path, brain_ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from nardis_neural.solana.edge.model import EdgeModel

    ws = _copy(brain_ws, tmp_path)
    (ws / "edge").mkdir()
    (ws / "edge" / "edge.json").write_text("{}")
    sentinel = object()
    monkeypatch.setattr(EdgeModel, "load", classmethod(lambda cls, d: sentinel))
    brain = SolanaBrain(ws, device="cpu")
    assert brain.edge is sentinel and brain.edge_meta == {}


def test_bootstrap_refuses_an_existing_workspace_unless_told_to_overwrite(
    tmp_path: Path, brain_ws: Path
) -> None:
    ws = _copy(brain_ws, tmp_path)
    registry = (ws / "registry.json").read_text()
    small = simulate_launches(
        LaunchSimSpec(
            n_tokens=6, seed=5, n_retail=80, duration_seconds=1800, mint_prefix="B", start_time=T_HIST
        )
    )[0]
    with pytest.raises(FileExistsError, match="overwrite"):
        SolanaBrain.bootstrap(ws, small, _sol_cfg(), _tiny(), device="cpu")
    assert (ws / "registry.json").read_text() == registry, "nothing was touched"
    before = sorted(p.name for p in ws.iterdir())
    with pytest.raises(ValueError, match="no labelled snapshots"):  # the rebuild fails before training
        SolanaBrain.bootstrap(ws, EventStore(), _sol_cfg(), _tiny(), device="cpu", overwrite=True)
    assert sorted(p.name for p in ws.iterdir()) == before, "a failed rebuild keeps the old workspace"
    assert (ws / "registry.json").read_text() == registry
    old = json.loads(registry)["champion"]
    (ws / "notes.txt").write_text("mine")
    cfg = _tiny()
    cfg.training.epochs = 1
    cfg.ensemble.size = 1
    brain = SolanaBrain.bootstrap(ws, small, _sol_cfg(), cfg, device="cpu", overwrite=True)
    assert brain.learner.registry.champion_version != old
    assert brain.learner.registry.versions() == [brain.learner.registry.champion_version]
    assert brain.market.tokens and all(m.startswith("B") for m in brain.market.tokens)
    assert (ws / "notes.txt").read_text() == "mine", "only workspace entries are removed"


def test_allocator_book_is_remembered_across_calls_and_restarts(
    tmp_path: Path, brain_ws: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    ws = _copy(brain_ws, tmp_path)
    brain = SolanaBrain(ws, device="cpu")

    def ranking(max_idle_seconds: float = 120.0) -> list[SimpleNamespace]:
        moon = {"expected_multiple": 3.0, "lottery_kelly": 0.05, "epistemic": 0.0, "vetoed": 0.0}
        return [
            SimpleNamespace(
                mint="Y", timestamp=brain.market.now, moonshot=moon, features={"liquidity_sol_log": 20.0}
            )
        ]

    monkeypatch.setattr(brain, "moonshot_ranking", ranking)
    brain.market.family_of |= {"X": 7, "Y": 7}  # one creator family
    first = brain.allocate(100.0)[0]
    assert first.reason == "position cap" and first.stake_sol == pytest.approx(4.0)
    capped = brain.allocate(100.0, {"X": 7.5})[0]
    assert capped.reason == "family cap" and capped.stake_sol == pytest.approx(0.5), "X is in Y's family"
    assert brain.allocate(80.0)[0].reason == "daily loss stop", "down 20 % on the day"
    again = SolanaBrain(ws, device="cpu")
    monkeypatch.setattr(again, "moonshot_ranking", ranking)
    assert again.allocate(80.0)[0].reason == "daily loss stop", "the day's opening equity survives restarts"
    assert again.allocate(80.0, day_start_equity_sol=85.0)[0].reason == "position cap"


def test_pending_assessments_survive_a_restart_and_are_still_labelled(
    tmp_path: Path, brain_ws: Path, live: tuple[EventStore, dict[str, str]]
) -> None:
    ws = _copy(brain_ws, tmp_path)
    brain = SolanaBrain(ws, device="cpu")
    events = [e for e in live[0].sorted() if e.t >= T_LIVE - 3600]
    head, tail = events[: len(events) // 3], events[len(events) // 3 :]
    brain.ingest_many(head)
    brain.assess_active()
    assert brain.pending and brain.resolve() == 0, "their horizons have not elapsed yet"
    brain.save()
    again = SolanaBrain(ws, device="cpu")
    assert [(p.mint, p.t) for p in again.pending] == [(p.mint, p.t) for p in brain.pending]
    assert not again.learner.pending, "the learner's own pending book was not kept"
    again.ingest_many(tail)
    again.market.advance(again.market.now + again.cfg.risk_horizon_seconds + 400)
    start = again.learner.buffer.next_seq
    resolved = again.resolve()
    new = [e for e in again.learner.buffer.all() if e.seq >= start]
    assert resolved > 0 and len(new) == resolved
    assert all(e.model_version for e in new), "each is linked to the prediction made before the restart"
