"""``nardis-neural`` command-line interface.

A *workspace* is a model-registry directory (champion/challenger/retired models, replay
buffer, shadow records, reports).  ``--model`` accepts a workspace (uses its champion)
or a single model directory.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from nardis_neural.config import NeuralConfig, load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Neural market-state brain (ML only).")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file")]
DeviceOpt = Annotated[str | None, typer.Option("--device", help="cpu | cuda | cuda:0 | mps (default: auto)")]
ModelOpt = Annotated[Path, typer.Option("--model", "-m", help="workspace or model directory")]
WorkspaceOpt = Annotated[Path, typer.Option("--workspace", "-w", help="registry workspace directory")]
DataOpt = Annotated[Path, typer.Option("--data", "-d", help="dataset (.npy dir, .npz, .pt, .parquet)")]


def _echo(obj: Any) -> None:
    typer.echo(json.dumps(obj, indent=2, default=float))


def _write(path: Path | None, obj: Any) -> None:
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, default=float))
        typer.echo(f"wrote {path}")


def _load_engine(model: Path, device: str | None) -> Any:
    from nardis_neural.inference.engine import NeuralEngine

    return NeuralEngine.load(model, device=device)


def _dataset(path: Path, config: NeuralConfig) -> Any:
    from nardis_neural.data.datasets import MarketDataset
    from nardis_neural.data.loaders import load_store

    cache = path.with_name(path.stem + "_cache") if path.suffix == ".parquet" else None
    return MarketDataset(load_store(path, config, cache_dir=cache), config)


@app.command("init-config")
def init_config(out: Annotated[Path, typer.Option("--out", "-o")] = Path("configs/default.yaml")) -> None:
    """Write the default configuration to a YAML file."""
    out.parent.mkdir(parents=True, exist_ok=True)
    NeuralConfig().save(out)
    typer.echo(f"wrote {out}")


@app.command("generate-synthetic")
def generate_synthetic_cmd(
    out: Annotated[Path, typer.Option("--out", "-o")],
    n: Annotated[int, typer.Option("--n")] = 4000,
    seed: Annotated[int, typer.Option("--seed")] = 0,
    fmt: Annotated[str, typer.Option("--format", help="npy | npz | pt | parquet")] = "npy",
    graph: Annotated[bool, typer.Option("--graph/--no-graph")] = False,
    drift_shift: Annotated[float, typer.Option("--drift-shift")] = 0.0,
    start_time: Annotated[float, typer.Option("--start-time")] = 1_700_000_000.0,
    config: ConfigOpt = None,
) -> None:
    """Generate a synthetic multi-regime dataset (testing only, not a market model)."""
    from nardis_neural.data.loaders import ArrayStore
    from nardis_neural.synthetic import SyntheticSpec, generate_synthetic

    cfg = load_config(config)
    arrays = generate_synthetic(
        cfg,
        SyntheticSpec(
            n_observations=n, seed=seed, graph=graph, drift_shift=drift_shift, start_time=start_time
        ),
    )
    store = ArrayStore(arrays)
    if fmt == "npy":
        store.save(out)
    elif fmt == "npz":
        store.save_npz(out)
    elif fmt == "pt":
        store.save_torch(out)
    elif fmt == "parquet":
        store.to_parquet(out, cfg)
    else:
        raise typer.BadParameter(f"unknown format {fmt}")
    typer.echo(f"wrote {n} observations to {out}")


@app.command()
def train(
    data: DataOpt,
    config: ConfigOpt = None,
    workspace: Annotated[
        Path | None, typer.Option("--workspace", "-w", help="register as champion here")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="save model directory here")] = None,
    epochs: Annotated[int | None, typer.Option("--epochs")] = None,
    ensemble_size: Annotated[int | None, typer.Option("--ensemble-size")] = None,
    pretrained: Annotated[
        Path | None, typer.Option("--pretrained", help="encoder weights from `pretrain`")
    ] = None,
    run_dir: Annotated[Path | None, typer.Option("--run-dir", help="per-epoch checkpoints / metrics")] = None,
    resume: Annotated[bool, typer.Option("--resume")] = False,
    device: DeviceOpt = None,
) -> None:
    """Train a calibrated deep ensemble on a labelled dataset."""
    import torch

    from nardis_neural.training.continual import ContinualLearner
    from nardis_neural.training.pipeline import train_engine
    from nardis_neural.training.pretraining import load_pretrained

    if workspace is None and out is None:
        raise typer.BadParameter("pass --workspace and/or --out")
    cfg = load_config(config)
    if ensemble_size is not None:
        cfg.ensemble.size = ensemble_size
        cfg.ensemble.embedding_member = min(cfg.ensemble.embedding_member, ensemble_size - 1)
    if device is not None:
        cfg.training.device = device
    ds = _dataset(data, cfg)
    state = load_pretrained(pretrained) if pretrained is not None else None
    engine, report = train_engine(
        cfg,
        ds.store,
        epochs=epochs,
        pretrained_state=state,
        run_dir=run_dir,
        resume=resume,
        device=torch.device(device) if device else None,
        log=typer.echo,
    )
    if out is not None:
        engine.save(out)
        typer.echo(f"saved model {engine.version} to {out}")
    if workspace is not None:
        from nardis_neural.lifecycle.champion import ModelRegistry

        if (workspace / "registry.json").exists() and ModelRegistry(workspace).champion_version is not None:
            registry = ModelRegistry(workspace)
            registry.register(engine, "candidate", "trained via CLI")
            registry.set_status(engine.version, "challenger", "trained via CLI; entering shadow mode")
            typer.echo(f"workspace already has a champion; registered {engine.version} as challenger")
        else:
            ContinualLearner.initialize(workspace, engine, cfg, device=engine.device)
            typer.echo(f"registered {engine.version} as champion in {workspace}")
    _echo({k: v for k, v in report.validation_metrics.items() if k.count(".") <= 1})


@app.command()
def pretrain(
    data: DataOpt,
    out: Annotated[Path, typer.Option("--out", "-o")],
    config: ConfigOpt = None,
    epochs: Annotated[int | None, typer.Option("--epochs")] = None,
    device: DeviceOpt = None,
) -> None:
    """Self-supervised pretraining of the temporal encoders on (unlabelled) sequences."""
    import torch

    from nardis_neural.data.normalization import FeatureNormalizer
    from nardis_neural.training.pretraining import pretrain_encoders, save_pretrained

    cfg = load_config(config)
    ds = _dataset(data, cfg)
    norm = FeatureNormalizer.fit(ds.store, ds.indices, cfg)
    model, result = pretrain_encoders(
        cfg, ds, norm, epochs=epochs, device=torch.device(device) if device else None, log=typer.echo
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    save_pretrained(out, model, result, cfg)
    typer.echo(f"saved pretrained encoders to {out}")


@app.command()
def evaluate(
    model: ModelOpt,
    data: DataOpt,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    device: DeviceOpt = None,
) -> None:
    """Evaluate a model on a labelled dataset (regression, calibration, ranking, tails)."""
    from nardis_neural.training.pipeline import evaluate_engine

    engine = _load_engine(model, device)
    metrics, _ = evaluate_engine(engine, _dataset(data, engine.config))
    _write(out, metrics)
    _echo({k: v for k, v in metrics.items() if k.count(".") <= 1})


@app.command()
def predict(
    model: ModelOpt,
    input_file: Annotated[Path | None, typer.Option("--input", "-i", help="JSON observation(s)")] = None,
    data: Annotated[Path | None, typer.Option("--data", "-d", help="dataset to predict")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="JSONL output")] = None,
    limit: Annotated[int, typer.Option("--limit")] = 10_000,
    device: DeviceOpt = None,
) -> None:
    """Predict from JSON observations or a dataset; prints / writes NeuralPrediction JSON."""
    from nardis_neural.data.sequences import arrays_to_observations
    from nardis_neural.schemas import NeuralObservation

    engine = _load_engine(model, device)
    if input_file is not None:
        raw = json.loads(input_file.read_text())
        obs = [NeuralObservation.from_dict(o) for o in (raw if isinstance(raw, list) else [raw])]
    elif data is not None:
        ds = _dataset(data, engine.config)
        idx = np.arange(min(limit, len(ds)))
        obs = arrays_to_observations(ds.store.select(ds.indices[idx]), engine.config)
    else:
        raise typer.BadParameter("pass --input or --data")
    preds = engine.predict_batch(obs)
    lines = [p.model_dump_json() for p in preds]
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")
        typer.echo(f"wrote {len(lines)} predictions to {out}")
    else:
        for line in lines[:20]:
            typer.echo(line)


@app.command("extract-embeddings")
def extract_embeddings_cmd(
    model: ModelOpt,
    data: DataOpt,
    out: Annotated[Path, typer.Option("--out", "-o", help=".parquet or .npz")],
    device: DeviceOpt = None,
) -> None:
    """Save MarketStateEmbeddings + expert weights + predictions for a dataset."""
    from nardis_neural.regimes.embeddings import extract_embeddings

    engine = _load_engine(model, device)
    table = extract_embeddings(engine, _dataset(data, engine.config))
    table.save(out)
    typer.echo(f"wrote {len(table)} embeddings ({table.embeddings.shape[1]}-d) to {out}")


@app.command("cluster-regimes")
def cluster_regimes(
    model: ModelOpt,
    data: DataOpt,
    method: Annotated[str, typer.Option("--method", help="kmeans | gmm | hdbscan")] = "kmeans",
    k: Annotated[int | None, typer.Option("--k")] = None,
    auto_k: Annotated[bool, typer.Option("--auto-k")] = False,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    device: DeviceOpt = None,
) -> None:
    """Discover latent regimes from embeddings and report per-cluster statistics."""
    from nardis_neural.regimes.clustering import RegimeClusterer, cluster_statistics
    from nardis_neural.regimes.embeddings import extract_embeddings

    engine = _load_engine(model, device)
    table = extract_embeddings(engine, _dataset(data, engine.config))
    rc = engine.config.regimes.model_copy(update={"method": method, "auto_select": auto_k})
    if k is not None:
        rc = rc.model_copy(update={"n_clusters": k})
    clusterer, labels = RegimeClusterer.fit(table.embeddings, rc)
    stats = cluster_statistics(
        labels,
        table.expert_weights,
        table.expert_names,
        table.horizons,
        table.predictions.get("return.mean"),
        table.targets.get("return"),
        table.timestamps,
        extra={k2: v for k2, v in table.predictions.items() if np.asarray(v).ndim == 1 and k2 != "regime"},
    )
    result = {
        "method": clusterer.method,
        "n_clusters": clusterer.n_clusters,
        "selection": clusterer.selection_scores,
        "clusters": stats,
    }
    _write(out, result)
    _echo(result)


@app.command()
def ingest(workspace: WorkspaceOpt, data: DataOpt, device: DeviceOpt = None) -> None:
    """Add labelled outcomes to the replay buffer (and shadow-compare if a challenger exists)."""
    from nardis_neural.training.continual import ContinualLearner

    learner = ContinualLearner(workspace, device=device)
    ds = _dataset(data, learner.config)
    n = learner.add_labelled_arrays(ds.store.select(ds.indices))
    _echo({"ingested": n} | learner.status())


@app.command()
def adapt(
    workspace: WorkspaceOpt,
    force: Annotated[bool, typer.Option("--force", help="adapt even below the sample threshold")] = False,
    device: DeviceOpt = None,
) -> None:
    """Continual adaptation: fine-tune a champion clone on the replay mixture."""
    from nardis_neural.training.continual import ContinualLearner

    learner = ContinualLearner(workspace, device=device)
    rep = learner.adapt() if force else learner.adapt_if_needed()
    _echo(rep.model_dump() if rep is not None else {"adapted": False, "reason": "not enough new samples"})


@app.command("full-retrain")
def full_retrain(
    workspace: WorkspaceOpt,
    data: Annotated[Path | None, typer.Option("--data", "-d", help="extra external dataset")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    device: DeviceOpt = None,
) -> None:
    """Train a fresh ensemble on weighted recent/historical/rare/difficult data."""
    from nardis_neural.training.continual import ContinualLearner

    learner = ContinualLearner(workspace, device=device)
    external = _dataset(data, learner.config).store if data is not None else None
    rep = learner.full_retrain(external) if force else learner.full_retrain_if_needed(external)
    _echo(rep.model_dump() if rep is not None else {"retrained": False, "reason": "not due"})


@app.command("shadow-evaluate")
def shadow_evaluate(
    workspace: WorkspaceOpt,
    data: Annotated[
        Path | None, typer.Option("--data", "-d", help="labelled data to replay in shadow")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    device: DeviceOpt = None,
) -> None:
    """Compare challenger vs champion on shadow predictions (optionally replaying a dataset)."""
    from nardis_neural.training.continual import ContinualLearner

    learner = ContinualLearner(workspace, device=device)
    challenger = learner.challenger
    if challenger is None:
        raise typer.BadParameter("workspace has no challenger")
    report = (
        learner.shadow.evaluate_dataset(learner.champion, challenger, _dataset(data, learner.config))
        if data is not None
        else learner.shadow_report()
    )
    assert report is not None
    payload = report.to_dict()
    _write(out, payload)
    keys = ("return.rmse", "upside.log_loss", "downside.log_loss", "upside.ece", "return.rank_corr")
    _echo(
        {
            "n": report.n,
            "champion": {k: report.champion.get(k) for k in keys},
            "challenger": {k: report.challenger.get(k) for k in keys},
        }
    )


@app.command()
def promote(
    workspace: WorkspaceOpt,
    version: Annotated[str | None, typer.Option("--version", help="force-promote this version")] = None,
    device: DeviceOpt = None,
) -> None:
    """Evaluate promotion gates for the challenger and promote if they pass."""
    from nardis_neural.training.continual import ContinualLearner

    learner = ContinualLearner(workspace, device=device)
    if version is not None:
        learner.promote(version)
        _echo({"promoted": version, "forced": True})
        return
    decision = learner.promote_if_ready()
    if decision is None:
        _echo({"promoted": False, "reason": "no challenger"})
        return
    typer.echo(decision.to_markdown())


@app.command("rollback")
def rollback_cmd(
    workspace: WorkspaceOpt,
    to: Annotated[
        str | None, typer.Option("--to", help="version to restore (default: previous champion)")
    ] = None,
    reason: Annotated[str, typer.Option("--reason")] = "manual rollback",
) -> None:
    """Restore a previous champion (weights, normaliser, calibration, config, ensemble)."""
    from nardis_neural.training.continual import ContinualLearner

    learner = ContinualLearner(workspace)
    restored = learner.rollback(to, reason)
    _echo({"champion": restored})


@app.command("drift-report")
def drift_report_cmd(
    model: ModelOpt,
    reference: Annotated[Path, typer.Option("--reference", "-r")],
    current: Annotated[Path, typer.Option("--current", "-k")],
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    device: DeviceOpt = None,
) -> None:
    """Input, embedding, prediction and error drift between two datasets."""
    from nardis_neural.monitoring.drift import drift_report

    engine = _load_engine(model, device)
    rep = drift_report(engine, _dataset(reference, engine.config), _dataset(current, engine.config))
    _write(out, rep.model_dump())
    summary: dict[str, Any] = rep.summary()
    summary["input_fraction_drifted"] = rep.input.fraction_drifted
    if rep.embedding is not None:
        summary["embedding"] = rep.embedding.stats
    if rep.error is not None:
        summary["error"] = rep.error.stats
    _echo(summary)


@app.command("inspect-model")
def inspect_model(model: ModelOpt, device: DeviceOpt = None) -> None:
    """Show model metadata, architecture summary and reference statistics."""
    engine = _load_engine(model, device)
    info = engine.describe()
    info.pop("training_stats", None)
    _echo(info)


@app.command()
def status(workspace: WorkspaceOpt) -> None:
    """Show workspace lifecycle status (champion, challenger, replay, counters)."""
    from nardis_neural.training.continual import ContinualLearner

    _echo(ContinualLearner(workspace, device="cpu").status())


@app.command()
def benchmark(
    model: ModelOpt,
    data: DataOpt,
    batch_sizes: Annotated[str, typer.Option("--batch-sizes")] = "1,32,256",
    mc_samples: Annotated[int | None, typer.Option("--mc-samples", help="override MC-dropout passes")] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o")] = None,
    device: DeviceOpt = None,
) -> None:
    """Measure inference latency, throughput and memory."""
    from nardis_neural.benchmark import benchmark_engine
    from nardis_neural.data.sequences import arrays_to_observations

    engine = _load_engine(model, device)
    ds = _dataset(data, engine.config)
    obs = arrays_to_observations(ds.store.select(ds.indices[: min(256, len(ds))]), engine.config)
    if mc_samples is not None:
        engine.config.ensemble.mc_dropout_samples = mc_samples
    res = benchmark_engine(engine, obs, tuple(int(b) for b in batch_sizes.split(",")))
    _write(out, res)
    _echo(res)


if __name__ == "__main__":  # pragma: no cover
    app()
