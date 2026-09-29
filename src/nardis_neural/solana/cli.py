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
DeviceOpt = Annotated[str | None, typer.Option("--device", help="cpu | cuda | cuda:0 | mps (default: auto)")]


def _sol_cfg(path: Path | None) -> Any:
    from nardis_neural.solana.config import SolanaConfig

    return SolanaConfig.load(path) if path is not None else SolanaConfig()


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=float))


@app.command("init-config")
def init_config(
    out: Annotated[Path, typer.Option("--out", "-o", help="YAML file to write")] = Path(
        "configs/solana.yaml"
    ),
) -> None:
    """Write the default Solana configuration."""
    from nardis_neural.solana.config import SolanaConfig

    out.parent.mkdir(parents=True, exist_ok=True)
    SolanaConfig().save(out)
    typer.echo(f"wrote {out}")


@app.command()
def simulate(
    out: Annotated[Path, typer.Option("--out", "-o", help="event directory (Parquet tables)")],
    tokens: Annotated[int, typer.Option("--tokens", help="number of token launches to simulate")] = 40,
    seed: Annotated[int, typer.Option("--seed", help="random seed")] = 0,
    prefix: Annotated[
        str, typer.Option("--prefix", help="mint/wallet name prefix (distinguishes eras)")
    ] = "Mint",
    start_time: Annotated[
        float, typer.Option("--start-time", help="simulation start, unix seconds")
    ] = 1_750_000_000.0,
    hours: Annotated[float, typer.Option("--hours", help="simulated duration, hours")] = 3.0,
    market: Annotated[
        str, typer.Option("--market", help="archetype mix: default, or degen (mostly duds + runners)")
    ] = "default",
    runners: Annotated[
        float | None, typer.Option("--runners", help="override the share of 100–1000x runner launches")
    ] = None,
    herding: Annotated[
        bool, typer.Option("--herding/--no-herding", help="self-exciting (Hawkes) retail demand")
    ] = False,
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
            herding=herding,
        )
    )
    store.save(out)
    (out / "archetypes.json").write_text(json.dumps(arch, indent=2))
    _echo({"events": len(store), "archetypes": dict(Counter(arch.values())), "out": str(out)})


@app.command("build-dataset")
def build_dataset(
    events: Annotated[Path, typer.Option("--events", "-e", help="event directory (Parquet tables)")],
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
    events: Annotated[
        Path, typer.Option("--events", "-e", help="historical event directory (Parquet tables)")
    ],
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Solana workspace directory to create")
    ],
    solana_config: SolCfg = None,
    config: BaseCfg = None,
    epochs: Annotated[
        int | None, typer.Option("--epochs", help="training epochs (default: from config)")
    ] = None,
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
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace directory")],
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
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace directory")],
    mint: Annotated[
        str | None, typer.Option("--mint", help="token mint to assess (default: all recently active tokens)")
    ] = None,
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
    min_transfer_sol: Annotated[
        float, typer.Option("--min-transfer-sol", help="ignore SOL transfers below this amount, SOL")
    ] = 0.05,
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
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace directory")],
    rpc: RpcOpt = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="append assessments as JSONL")] = None,
    polls: Annotated[
        int | None, typer.Option("--polls", help="stop after N polls (default: run forever)")
    ] = None,
    poll_interval: Annotated[float, typer.Option("--poll-interval", help="seconds between RPC polls")] = 2.0,
    assess_every: Annotated[
        float, typer.Option("--assess-every", help="seconds between assessment rounds")
    ] = 10.0,
    maintenance_every: Annotated[
        float, typer.Option("--maintenance-every", help="seconds between maintenance runs")
    ] = 600.0,
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
    folds: Annotated[int, typer.Option("--folds", help="walk-forward folds")] = 4,
    take_profit: Annotated[
        float, typer.Option("--take-profit", help="take-profit barrier, fractional return (0.25 = +25%)")
    ] = 0.25,
    stop_loss: Annotated[
        float, typer.Option("--stop-loss", help="stop-loss barrier, fractional loss (0.15 = -15%)")
    ] = 0.15,
    max_hold: Annotated[
        float, typer.Option("--max-hold", help="max holding time (time barrier), seconds")
    ] = 180.0,
    latency: Annotated[float, typer.Option("--latency", help="entry/exit latency, seconds")] = 1.0,
    max_positions: Annotated[
        int, typer.Option("--max-positions", help="max concurrent open positions in the backtest")
    ] = 5,
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
    horizon_hours: Annotated[
        float, typer.Option("--horizon-hours", help="outcome horizon after entry, hours")
    ] = 6.0,
    max_entry_age: Annotated[
        float, typer.Option("--max-entry-age", help="latest entry after launch, seconds")
    ] = 600.0,
    min_ev: Annotated[
        float, typer.Option("--min-ev", help="ticket when E[ladder payoff] per SOL is at least this")
    ] = 1.0,
    test_fraction: Annotated[
        float, typer.Option("--test-fraction", help="share of later tokens held out for the test")
    ] = 0.35,
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
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Solana workspace (created if new, else resumed)")
    ],
    events: Annotated[
        Path | None, typer.Option("--events", "-e", help="stream a saved event directory instead of RPC")
    ] = None,
    rpc: RpcOpt = None,
    start: Annotated[str | None, typer.Option("--start", help="unix seconds or ISO date (RPC mode)")] = None,
    end: Annotated[str | None, typer.Option("--end", help="unix seconds or ISO date (RPC mode)")] = None,
    segment_minutes: Annotated[
        float, typer.Option("--segment-minutes", help="history segment length, minutes (RPC mode)")
    ] = 60.0,
    workers: Annotated[int, typer.Option("--workers", help="parallel getTransaction calls")] = 8,
    warmup_hours: Annotated[
        float, typer.Option("--warmup-hours", help="hours of stream used to bootstrap a new workspace")
    ] = 6.0,
    evict_idle_hours: Annotated[
        float, typer.Option("--evict-idle-hours", help="forget tokens idle this many hours")
    ] = 2.0,
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


@app.command("tape-research")
def tape_research(
    workspace: Annotated[
        Path, typer.Option("--workspace", "-w", help="Solana workspace (history + champion)")
    ],
    max_trades: Annotated[int, typer.Option("--max-trades", help="trades per tape (most recent kept)")] = 96,
    members: Annotated[int, typer.Option("--members", help="ensemble members")] = 3,
    epochs: Annotated[int, typer.Option("--epochs", help="maximum training epochs per member")] = 40,
    d: Annotated[int, typer.Option("--d", help="Transformer width")] = 64,
    layers: Annotated[int, typer.Option("--layers", help="Transformer layers")] = 2,
    test_fraction: Annotated[
        float,
        typer.Option("--test-fraction", help="share of the latest-launched tokens held out for the test"),
    ] = 0.35,
    archetypes: Annotated[
        Path | None, typer.Option("--archetypes", help="simulator archetypes.json for diagnostics")
    ] = None,
    device: DeviceOpt = None,
) -> None:
    """Train the Tape Transformer (trade tape + wallet embeddings → tail and collapse) and
    score it against the raw-feature tail model on later tokens; installs the model."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.tape.features import TapeSpec
    from nardis_neural.solana.tape.research import tape_markdown

    brain = SolanaBrain(workspace, device=device)
    arch = json.loads(archetypes.read_text()) if archetypes is not None else None
    report = brain.fit_tape(
        tape=TapeSpec(max_trades=max_trades),
        test_fraction=test_fraction,
        members=members,
        epochs=epochs,
        archetypes=arch,
        log=typer.echo,
        d=d,
        layers=layers,
    )
    typer.echo(tape_markdown(report))
    typer.echo(f"tape model installed in {workspace / 'tape'}")


@app.command("stopping-research")
def stopping_research(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace (history)")],
    spacing: Annotated[
        float, typer.Option("--spacing", help="minimum seconds between exit decisions")
    ] = 30.0,
    test_fraction: Annotated[
        float,
        typer.Option("--test-fraction", help="share of the latest-launched tokens held out for the test"),
    ] = 0.35,
    archetypes: Annotated[
        Path | None, typer.Option("--archetypes", help="simulator archetypes.json for diagnostics")
    ] = None,
    utility: Annotated[
        str,
        typer.Option("--utility", help="installed exit objective: log (compounding) or power (runner mode)"),
    ] = "log",
    gamma: Annotated[
        float, typer.Option("--gamma", help="risk aversion of power utility, 0 < gamma < 1")
    ] = 0.5,
) -> None:
    """Fit the optimal-stopping exit model (Longstaff–Schwartz, log utility), score it against
    hold, timers and the ladder on later tokens, and install it."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.stopping import stopping_markdown

    brain = SolanaBrain(workspace)
    arch = json.loads(archetypes.read_text()) if archetypes is not None else None
    report = brain.fit_stopping(
        test_fraction=test_fraction,
        spacing=spacing,
        archetypes=arch,
        log=typer.echo,
        utility=utility,
        gamma=gamma,
    )
    typer.echo(stopping_markdown(report))
    typer.echo(f"stopping model installed in {workspace / 'stopping'}")


@app.command("fetch-history")
def fetch_history_cmd(
    out: Annotated[Path, typer.Option("--out", "-o", help="event directory to write (resumable)")],
    hours: Annotated[float, typer.Option("--hours", help="length of history to fetch, hours")] = 12.0,
    end: Annotated[
        str | None,
        typer.Option("--end", help="end of the window: unix seconds or ISO date (default: 15 min ago)"),
    ] = None,
    workers: Annotated[
        int, typer.Option("--workers", help="parallel getTransaction calls (6 suits most nodes)")
    ] = 6,
    rpc: Annotated[
        str | None,
        typer.Option("--rpc", envvar="SOLANA_RPC_URL", help="read-only (archival) RPC endpoint URL"),
    ] = None,
) -> None:
    """Fetch pump.fun history into an event directory for `bootstrap` and the research commands.

    Resumable: rerun the same command after an interruption and it continues.  The saved
    history keeps only tokens created inside the window, SOL-priced (see clean_history)."""
    import time as _time

    from nardis_neural.solana.ingest import SolanaRpc, fetch_history

    if rpc is None:
        raise typer.BadParameter("pass --rpc or set SOLANA_RPC_URL")
    window = out / "window.json"
    if window.exists():
        w = json.loads(window.read_text())
        start_t, end_t = float(w["start"]), float(w["end"])
    else:
        end_t = _when(end) if end else _time.time() - 900
        start_t = end_t - hours * 3600
        out.mkdir(parents=True, exist_ok=True)
        window.write_text(json.dumps({"start": start_t, "end": end_t}))
    stats = fetch_history(
        SolanaRpc(rpc, retries=8, backoff=1.0, timeout=30.0),
        out,
        start_t,
        end_t,
        workers=workers,
        log=typer.echo,
    )
    _echo(stats)


def _archive_mapping(maps: list[str] | None, features: str | None, percent: bool) -> Any:
    from nardis_neural.solana.archive import ALIASES, ArchiveMapping

    columns = {}
    for m in maps or []:
        if "=" not in m:
            raise typer.BadParameter(f"--map expects field=column, got {m!r}")
        fld, col = m.split("=", 1)
        if fld not in ALIASES:
            raise typer.BadParameter(f"unknown field {fld!r}; one of {', '.join(ALIASES)}")
        columns[fld] = col
    feats = [f.strip() for f in features.split(",") if f.strip()] if features else None
    return ArchiveMapping(columns, feats, percent)


@app.command("meta-train")
def meta_train(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace")],
    archive: Annotated[
        Path, typer.Option("--archive", "-a", help="trade archive: Parquet file/directory or SQLite .db")
    ],
    table: Annotated[str | None, typer.Option("--table", help="SQLite table holding the trades")] = None,
    maps: Annotated[
        list[str] | None, typer.Option("--map", help="field=column, e.g. --map mint=token_ca (repeatable)")
    ] = None,
    features: Annotated[
        str | None, typer.Option("--features", help="comma-separated feature columns (default: all numeric)")
    ] = None,
    return_percent: Annotated[
        bool, typer.Option("--return-percent", help="the return column is in percent (50 = +50 %)")
    ] = False,
) -> None:
    """Train the meta-learner on the trading system's own trade archive (only new trades are added)."""
    from nardis_neural.solana.archive import train_from_archive
    from nardis_neural.solana.metalabel import MetaLearner

    meta_dir = workspace / "meta"
    learner = MetaLearner.load(meta_dir) if (meta_dir / "meta.json").exists() else MetaLearner()
    stats = train_from_archive(
        learner, archive, _archive_mapping(maps, features, return_percent), table=table, log=typer.echo
    )
    learner.save(meta_dir)
    _echo({k: v for k, v in stats.items() if k != "refit"} | {"levels": stats["refit"].get("levels", {})})


@app.command("serve")
def serve(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace")],
    host: Annotated[str, typer.Option("--host", help="bind address (keep 127.0.0.1 unless firewalled)")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option("--port", help="HTTP port")] = 8787,
    rpc: Annotated[
        str | None, typer.Option("--rpc", envvar="SOLANA_RPC_URL", help="read-only RPC endpoint URL")
    ] = None,
    stream: Annotated[
        bool,
        typer.Option("--stream/--no-stream", help="feed the live chain into the brain in the background"),
    ] = True,
    poll_interval: Annotated[float, typer.Option("--poll-interval", help="seconds between RPC polls")] = 2.0,
    workers: Annotated[int, typer.Option("--workers", help="parallel getTransaction calls")] = 6,
    pumpswap: Annotated[
        bool,
        typer.Option("--pumpswap/--no-pumpswap", help="also poll PumpSwap (heavy: hundreds of tx/s)"),
    ] = False,
    archive: Annotated[
        Path | None,
        typer.Option("--archive", "-a", help="trade archive to keep training on (Parquet or SQLite)"),
    ] = None,
    table: Annotated[str | None, typer.Option("--table", help="SQLite table holding the trades")] = None,
    archive_every: Annotated[
        float, typer.Option("--archive-every", help="seconds between archive rescans")
    ] = 600.0,
    maps: Annotated[list[str] | None, typer.Option("--map", help="field=column (repeatable)")] = None,
    features: Annotated[
        str | None, typer.Option("--features", help="comma-separated feature columns")
    ] = None,
    return_percent: Annotated[
        bool, typer.Option("--return-percent", help="return column in percent")
    ] = False,
    device: DeviceOpt = None,
) -> None:
    """Run the addon as a local HTTP/JSON sidecar for the trading system (advice only).

    Endpoints: GET /health /tokens /moonshots /ranking /assess; POST /advise_trade /settle_trade
    /hold_advice /allocate /ingest /save.  --alert-url pushes new moonshot candidates.
    See docs/INTEGRATION.md."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.ingest import ChainStreamer, SolanaRpc
    from nardis_neural.solana.service import AddonService, make_server

    brain = SolanaBrain(workspace, device=device)
    service = AddonService(brain)
    if stream:
        if rpc is None:
            raise typer.BadParameter("pass --rpc or set SOLANA_RPC_URL, or use --no-stream")
        from nardis_neural.solana.ingest.pumpfun import PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM

        programs = [PUMP_FUN_PROGRAM, PUMP_SWAP_PROGRAM] if pumpswap else [PUMP_FUN_PROGRAM]
        streamer = ChainStreamer(
            SolanaRpc(rpc, retries=6, backoff=1.0),
            programs=programs,
            state_file=workspace / "stream_cursor.json",
            workers=workers,
        )
        service.stream(streamer, poll_interval=poll_interval)
    if archive is not None:
        service.train_on_archive(
            archive, _archive_mapping(maps, features, return_percent), table, archive_every
        )
    server = make_server(service, host, port)
    typer.echo(f"addon listening on http://{host}:{port} (stream {'on' if stream else 'off'})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.stop()


@app.command("runner-research")
def runner_research(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace (history)")],
    horizon_minutes: Annotated[
        float,
        typer.Option("--horizon-minutes", help="minutes after entry to hit the target"),
    ] = 30.0,
    test_fraction: Annotated[
        float,
        typer.Option("--test-fraction", help="share of the latest-launched tokens held out for the test"),
    ] = 0.35,
) -> None:
    """Train the runner detector (P(reach 2x / 5x / 10x / 100x / 1000x)), score it against the tail
    model on later tokens, show which signals identify runners, and install it."""
    from nardis_neural.solana.brain import SolanaBrain
    from nardis_neural.solana.moonshot import MoonshotSpec
    from nardis_neural.solana.runners import runner_markdown

    brain = SolanaBrain(workspace)
    report = brain.fit_runners(
        MoonshotSpec(horizon_seconds=horizon_minutes * 60), test_fraction=test_fraction, log=typer.echo
    )
    typer.echo(runner_markdown(report))
    typer.echo(f"runner detector installed in {workspace / 'runners'}")


@app.command("forward-report")
def forward_report(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace")],
) -> None:
    """Paper-ticket scorecard of the ML signals (forward test recorded during stream / stream-train)."""
    from nardis_neural.solana.forward import ForwardLedger
    from nardis_neural.solana.moonshot.labels import MoonshotSpec

    ledger = ForwardLedger.load(workspace / "forward", MoonshotSpec())
    _echo({"alarm": ledger.alarm} | ledger.summary())


@app.command("research-suite")
def research_suite(
    seeds: Annotated[str, typer.Option("--seeds", help="comma-separated simulator seeds")] = "7,19,23",
    market: Annotated[str, typer.Option("--market", help="archetype mix: default or degen")] = "degen",
    tokens: Annotated[int, typer.Option("--tokens", help="launches per simulated market")] = 150,
    hours: Annotated[float, typer.Option("--hours", help="simulated hours per market")] = 8.0,
    tape: Annotated[
        bool, typer.Option("--tape/--no-tape", help="also run the (slower) tape research")
    ] = False,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="write the JSON result here")] = None,
) -> None:
    """Run the moonshot (and optionally tape) research on several independent simulated markets
    and report every metric as mean ± sd across seeds."""
    from nardis_neural.solana.suite import run_suite, suite_markdown

    result = run_suite(
        [int(s) for s in seeds.split(",") if s.strip()], market, tokens, hours, tape=tape, log=typer.echo
    )
    if out is not None:
        out.write_text(json.dumps(result, indent=2, default=float))
    typer.echo(suite_markdown(result))


@app.command()
def allocate(
    workspace: Annotated[Path, typer.Option("--workspace", "-w", help="Solana workspace")],
    equity: Annotated[float, typer.Option("--equity", help="current bankroll in SOL")],
    peak: Annotated[
        float | None, typer.Option("--peak", help="peak bankroll in SOL (drawdown governor)")
    ] = None,
    device: DeviceOpt = None,
) -> None:
    """Recommended stakes for the current moonshot opportunities (advice only, never orders)."""
    from nardis_neural.solana.brain import SolanaBrain

    brain = SolanaBrain(workspace, device=device)
    allocations = brain.allocate(equity, peak_equity_sol=peak)
    _echo(
        {
            "track_record": brain.forward.track_record(),
            "allocations": [
                {"mint": a.mint, "stake_sol": a.stake_sol, "fraction": a.fraction, "reason": a.reason}
                for a in allocations
            ],
        }
    )
