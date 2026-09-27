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
