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
) -> None:
    """Simulate memecoin launches (snipers, bundles, rugs, graduations, smart money, bots)."""
    from collections import Counter

    from nardis_neural.solana.simulator import LaunchSimSpec, simulate_launches

    store, arch = simulate_launches(
        LaunchSimSpec(
            n_tokens=tokens,
            seed=seed,
            mint_prefix=prefix,
            start_time=start_time,
            duration_seconds=hours * 3600,
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
    device: DeviceOpt = None,
) -> None:
    """Train the neural ensemble + risk model from history and create a Solana workspace."""
    from nardis_neural.config import load_config
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.market import EventStore

    base = load_config(config)
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
