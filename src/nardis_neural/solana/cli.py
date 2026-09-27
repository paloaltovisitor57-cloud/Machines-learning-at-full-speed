"""``nardis-neural solana …`` commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

app = typer.Typer(
    no_args_is_help=True, help="Solana-specific ML: simulate, build datasets, bootstrap, replay, assess."
)

SolCfg = Annotated[Path | None, typer.Option("--solana-config", help="Solana YAML config")]
BaseCfg = Annotated[Path | None, typer.Option("--config", "-c", help="base neural YAML config")]
DeviceOpt = Annotated[str | None, typer.Option("--device")]


def _sol_cfg(path: Path | None) -> Any:
    from nardis_neural.solana.config import SolanaConfig

    return SolanaConfig.load(path) if path is not None else SolanaConfig()


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=float))


@app.command("init-config")
def init_config(out: Annotated[Path, typer.Option("--out", "-o")] = Path("configs/solana.yaml")) -> None:
    """Write the default Solana configuration."""
    from nardis_neural.solana.config import SolanaConfig

    out.parent.mkdir(parents=True, exist_ok=True)
    SolanaConfig().save(out)
    typer.echo(f"wrote {out}")


@app.command()
def simulate(
    out: Annotated[Path, typer.Option("--out", "-o", help="event directory (Parquet tables)")],
    tokens: Annotated[int, typer.Option("--tokens")] = 40,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    prefix: Annotated[str, typer.Option("--prefix")] = "Mint",
    start_time: Annotated[float, typer.Option("--start-time")] = 1_750_000_000.0,
    hours: Annotated[float, typer.Option("--hours")] = 3.0,
    market: Annotated[
        str, typer.Option("--market", help="archetype mix: default, or degen (mostly duds + runners)")
    ] = "default",
    runners: Annotated[
        float | None, typer.Option("--runners", help="override the share of 100–1000x runner launches")
    ] = None,
) -> None:
    """Simulate memecoin launches (snipers, bundles, rugs, graduations, runners, smart money, bots)."""
    from collections import Counter

    from nardis_neural.solana.simulator import MARKET_PRESETS, LaunchSimSpec, simulate_launches

    if market not in MARKET_PRESETS:
        raise typer.BadParameter(f"unknown market {market!r}; choose from {sorted(MARKET_PRESETS)}")
    weights = dict(MARKET_PRESETS[market])
    if runners is not None:
        weights["runner"] = runners
    store, arch = simulate_launches(
        LaunchSimSpec(
            n_tokens=tokens,
            seed=seed,
            mint_prefix=prefix,
            start_time=start_time,
            duration_seconds=hours * 3600,
            archetype_weights=weights,
        )
    )
    store.save(out)
    (out / "archetypes.json").write_text(json.dumps(arch, indent=2))
    _echo({"events": len(store), "archetypes": dict(Counter(arch.values())), "out": str(out)})


@app.command("build-dataset")
def build_dataset(
    events: Annotated[Path, typer.Option("--events", "-e")],
    out: Annotated[Path, typer.Option("--out", "-o", help="output .npy dataset directory")],
    solana_config: SolCfg = None,
    config: BaseCfg = None,
) -> None:
    """Causal replay + hindsight labelling → canonical neural dataset (+ risk labels)."""
    import numpy as np

    from nardis_neural.config import load_config
    from nardis_neural.solana.dataset import build_solana_dataset
    from nardis_neural.solana.market import EventStore

    cfg = _sol_cfg(solana_config)
    ds = build_solana_dataset(EventStore.load(events), cfg, cfg.neural_config(load_config(config)))
    ds.store().save(out)
    np.savez(
        out / "solana_labels.npz",
        risk=ds.risk,
        net_returns=ds.net_returns,
        cost=ds.round_trip_cost,
        mints=ds.mints.astype(str),
    )
    _echo({"snapshots": len(ds), "risk_rates": np.nanmean(ds.risk, axis=0).tolist(), "out": str(out)})


@app.command()
def bootstrap(
    events: Annotated[Path, typer.Option("--events", "-e")],
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    solana_config: SolCfg = None,
    config: BaseCfg = None,
    epochs: Annotated[int | None, typer.Option("--epochs")] = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="auto | cpu-lite | cpu | gpu | gpu-frontier")
    ] = None,
    device: DeviceOpt = None,
) -> None:
    """Train the neural ensemble + risk model from history and create a Solana workspace."""
    from nardis_neural.config import load_config
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.market import EventStore

    base = load_config(config)
    if profile is not None:
        from nardis_neural.hardware import apply_profile

        base = apply_profile(base, profile)
    if epochs is not None:
        base.training.epochs = epochs
    if device is not None:
        base.training.device = device
    brain = SolanaBrain.bootstrap(
        workspace, EventStore.load(events), _sol_cfg(solana_config), base, device=device, log=typer.echo
    )
    _echo(
        {
            "workspace": str(workspace),
            "champion": brain.learner.registry.champion_version,
            "risk": None if brain.risk is None else brain.risk.report.get("metrics"),
        }
    )


@app.command()
def replay(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    events: Annotated[Path, typer.Option("--events", "-e", help="new events to stream in")],
    every: Annotated[float, typer.Option("--assess-every", help="seconds between assessment rounds")] = 10.0,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="JSONL of assessments")] = None,
    device: DeviceOpt = None,
) -> None:
    """Stream events through the brain as if live: assess, resolve outcomes, then maintain."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.market import EventStore

    brain = SolanaBrain(workspace, device=device)
    stream = EventStore.load(events).sorted()
    fh = out.open("w") if out is not None else None
    n_assess = n_resolved = 0
    next_round = stream[0].t + every if stream else 0.0
    try:
        for e in stream:
            brain.ingest(e)
            if e.t >= next_round:
                next_round = e.t + every
                for rep in brain.assess_active():
                    n_assess += 1
                    if fh is not None:
                        fh.write(rep.model_dump_json(exclude={"features"}) + "\n")
                n_resolved += brain.resolve()
    finally:
        if fh is not None:
            fh.close()
    status = brain.maintenance()
    _echo({"events": len(stream), "assessments": n_assess, "resolved": n_resolved} | status)


@app.command()
def assess(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    mint: Annotated[str | None, typer.Option("--mint", help="default: all recently active tokens")] = None,
    device: DeviceOpt = None,
) -> None:
    """Assess token(s) at the workspace's current market time."""
    from nardis_neural.solana.brain import SolanaBrain

    brain = SolanaBrain(workspace, device=device)
    reports = [brain.assess(mint)] if mint else brain.assess_active(max_idle_seconds=1e9)
    for r in reports:
        typer.echo(r.model_dump_json(exclude={"features", "prediction"}))


RpcOpt = Annotated[
    str | None, typer.Option("--rpc", envvar="SOLANA_RPC_URL", help="read-only RPC endpoint URL")
]


@app.command()
def decode(
    input_file: Annotated[Path, typer.Option("--input", "-i", help="JSONL of getTransaction results")],
    out: Annotated[Path, typer.Option("--out", "-o", help="event directory (Parquet tables)")],
    min_transfer_sol: Annotated[float, typer.Option("--min-transfer-sol")] = 0.05,
) -> None:
    """Decode raw Solana transactions (pump.fun, AMMs, SOL transfers) into market events."""
    from nardis_neural.solana.ingest import TransactionDecoder, decode_transactions
    from nardis_neural.solana.market import EventStore

    txs = [json.loads(line) for line in input_file.read_text().splitlines() if line.strip()]
    decoder = TransactionDecoder(min_transfer_sol=min_transfer_sol)
    store = EventStore(decode_transactions(txs, decoder))
    store.save(out)
    _echo(
        {
            "transactions": decoder.stats.transactions,
            "failed": decoder.stats.failed,
            "events": decoder.stats.events,
            "out": str(out),
        }
    )


@app.command()
def backfill(
    out: Annotated[Path, typer.Option("--out", "-o", help="event directory (Parquet tables)")],
    rpc: RpcOpt = None,
    limit: Annotated[int, typer.Option("--limit", help="max signatures per program")] = 1000,
    raw: Annotated[Path | None, typer.Option("--raw", help="also save raw transactions as JSONL")] = None,
) -> None:
    """Fetch recent pump.fun / PumpSwap history over RPC (read-only) and decode it."""
    from nardis_neural.solana.ingest import ChainStreamer, SolanaRpc
    from nardis_neural.solana.market import EventStore

    if rpc is None:
        raise typer.BadParameter("pass --rpc or set SOLANA_RPC_URL")
    client = SolanaRpc(rpc)
    streamer = ChainStreamer(client, initial_limit=limit)
    raw_fh = raw.open("a") if raw is not None else None
    if raw_fh is not None:
        streamer.raw_sink = lambda tx: raw_fh.write(json.dumps(tx) + "\n")
    try:
        store = EventStore(streamer.poll())
    finally:
        if raw_fh is not None:
            raw_fh.close()
    store.save(out)
    _echo({"events": len(store), "decoded": streamer.decoder.stats.events, "out": str(out)})


@app.command()
def stream(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    rpc: RpcOpt = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="append assessments as JSONL")] = None,
    polls: Annotated[
        int | None, typer.Option("--polls", help="stop after N polls (default: run forever)")
    ] = None,
    poll_interval: Annotated[float, typer.Option("--poll-interval")] = 2.0,
    assess_every: Annotated[float, typer.Option("--assess-every")] = 10.0,
    maintenance_every: Annotated[float, typer.Option("--maintenance-every")] = 600.0,
    device: DeviceOpt = None,
) -> None:
    """Stream live chain activity into a Solana workspace (read-only) and emit assessments."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.ingest import ChainStreamer, SolanaRpc, run_live

    if rpc is None:
        raise typer.BadParameter("pass --rpc or set SOLANA_RPC_URL")
    brain = SolanaBrain(workspace, device=device)
    streamer = ChainStreamer(SolanaRpc(rpc), state_file=workspace / "stream_cursor.json")
    fh = out.open("a") if out is not None else None
    try:
        stats = run_live(brain, streamer, assess_every, maintenance_every, poll_interval, fh, max_polls=polls)
    finally:
        if fh is not None:
            fh.close()
    _echo(stats)


@app.command("edge-research")
def edge_research(
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Solana workspace (history + champion)")
    ],
    folds: Annotated[int, typer.Option("--folds")] = 4,
    take_profit: Annotated[float, typer.Option("--take-profit")] = 0.25,
    stop_loss: Annotated[float, typer.Option("--stop-loss")] = 0.15,
    max_hold: Annotated[float, typer.Option("--max-hold", help="seconds")] = 180.0,
    latency: Annotated[float, typer.Option("--latency", help="entry/exit latency, seconds")] = 1.0,
    max_positions: Annotated[int, typer.Option("--max-positions")] = 5,
    device: DeviceOpt = None,
) -> None:
    """Walk-forward edge research on the workspace history; installs the edge model.

    Paper research: executable triple-barrier outcomes, out-of-fold meta-labeling, threshold
    chosen on a tune period, one report on an untouched test period vs baselines."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.edge import BarrierSpec, research_markdown

    brain = SolanaBrain(workspace, device=device)
    spec = BarrierSpec(
        take_profit=take_profit,
        stop_loss=stop_loss,
        max_hold_seconds=max_hold,
        latency_seconds=latency,
        size_sol=brain.cfg.trade_size_sol,
    )
    report = brain.fit_edge(spec, n_folds=folds, max_positions=max_positions, log=typer.echo)
    typer.echo(research_markdown(report))
    typer.echo(f"edge model installed in {workspace / 'edge'}")


@app.command("moonshot-research")
def moonshot_research(
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Solana workspace (history + champion)")
    ],
    inputs: Annotated[
        str, typer.Option("--inputs", help="raw (on-chain features) or neural (walk-forward OOF stack)")
    ] = "raw",
    size: Annotated[float, typer.Option("--size", help="ticket size, SOL")] = 0.5,
    latency: Annotated[float, typer.Option("--latency", help="entry/exit latency, seconds")] = 1.0,
    horizon_hours: Annotated[float, typer.Option("--horizon-hours")] = 6.0,
    max_entry_age: Annotated[float, typer.Option("--max-entry-age", help="seconds after launch")] = 600.0,
    min_ev: Annotated[
        float, typer.Option("--min-ev", help="ticket when E[ladder payoff] per SOL is at least this")
    ] = 1.0,
    test_fraction: Annotated[float, typer.Option("--test-fraction")] = 0.35,
    folds: Annotated[int, typer.Option("--folds", help="walk-forward folds (neural inputs)")] = 4,
    archetypes: Annotated[
        Path | None, typer.Option("--archetypes", help="simulator archetypes.json for diagnostics")
    ] = None,
    device: DeviceOpt = None,
) -> None:
    """Fat-tail research: P(≥2x … ≥1000x) per token, ladder payoff, lottery-Kelly sizing hints.

    Train on earlier tokens with labels censored at the cutoff, score once on later tokens
    against every-launch, random and momentum tickets; installs the tail model."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.moonshot import MoonshotSpec, moonshot_markdown

    brain = SolanaBrain(workspace, device=device)
    spec = MoonshotSpec(
        size_sol=size,
        latency_seconds=latency,
        horizon_seconds=horizon_hours * 3600,
        max_entry_age_seconds=max_entry_age,
    )
    arch = json.loads(archetypes.read_text()) if archetypes is not None else None
    report = brain.fit_moonshot(
        spec,
        inputs=inputs,
        n_folds=folds,
        test_fraction=test_fraction,
        min_expected_multiple=min_ev,
        archetypes=arch,
        log=typer.echo,
    )
    typer.echo(moonshot_markdown(report))
    typer.echo(f"tail model installed in {workspace / 'moonshot'}")


def _when(value: str) -> float:
    """Unix seconds or an ISO date / datetime (UTC when no zone is given)."""
    from datetime import UTC, datetime

    try:
        return float(value)
    except ValueError:
        dt = datetime.fromisoformat(value)
        return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).timestamp()


@app.command("stream-train")
def stream_train_cmd(
    workspace: Annotated[Path, typer.Option("--workspace", "-w")],
    events: Annotated[
        Path | None, typer.Option("--events", "-e", help="stream a saved event directory instead of RPC")
    ] = None,
    rpc: RpcOpt = None,
    start: Annotated[str | None, typer.Option("--start", help="unix seconds or ISO date (RPC mode)")] = None,
    end: Annotated[str | None, typer.Option("--end", help="unix seconds or ISO date (RPC mode)")] = None,
    segment_minutes: Annotated[float, typer.Option("--segment-minutes")] = 60.0,
    workers: Annotated[int, typer.Option("--workers", help="parallel getTransaction calls")] = 8,
    warmup_hours: Annotated[float, typer.Option("--warmup-hours")] = 6.0,
    evict_idle_hours: Annotated[float, typer.Option("--evict-idle-hours")] = 2.0,
    solana_config: SolCfg = None,
    config: BaseCfg = None,
    profile: Annotated[
        str | None, typer.Option("--profile", help="auto | cpu-lite | cpu | gpu | gpu-frontier")
    ] = "auto",
    device: DeviceOpt = None,
) -> None:
    """Learn by streaming history through the brain — nothing is downloaded to disk.

    RPC mode walks an archival endpoint (e.g. Old Faithful) from --start to --end, fetching
    only pump.fun / PumpSwap transactions. A new workspace bootstraps from the first
    --warmup-hours; an existing one resumes after its last checkpoint."""
    from nardis_neural.config import load_config
    from nardis_neural.solana.ingest.history import HistoryWalker
    from nardis_neural.solana.ingest.rpc import SolanaRpc
    from nardis_neural.solana.market import EventStore
    from nardis_neural.solana.streaming import stream_train

    base = load_config(config)
    if profile is not None:
        from nardis_neural.hardware import apply_profile

        base = apply_profile(base, profile)
    if device is not None:
        base.training.device = device
    source: Any
    decoder = None
    if events is not None:
        source = iter(EventStore.load(events).sorted())
    else:
        if rpc is None:
            raise typer.BadParameter("pass --events, or --rpc (or set SOLANA_RPC_URL)")
        if start is None or end is None:
            raise typer.BadParameter("RPC mode needs --start and --end")
        walker = HistoryWalker(
            SolanaRpc(rpc), _when(start), _when(end), segment_seconds=segment_minutes * 60, workers=workers
        )
        decoder = walker.decoder
        source = walker.events()
    stats = stream_train(
        workspace,
        source,
        _sol_cfg(solana_config),
        base,
        warmup_seconds=warmup_hours * 3600,
        evict_idle_seconds=evict_idle_hours * 3600,
        decoder=decoder,
        device=device,
        log=typer.echo,
    )
    _echo(stats)
