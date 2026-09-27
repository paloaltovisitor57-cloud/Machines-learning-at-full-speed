"""Drives every CLI command through a complete train → adapt → shadow → promote → rollback cycle."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from nardis_neural.cli import app
from tests.conftest import make_tiny_config

runner = CliRunner()


def run(*args: str | Path) -> str:
    result = runner.invoke(app, [str(a) for a in args], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


def _last_json(output: str) -> dict[str, object]:
    start = output.index("{")
    parsed: dict[str, object] = json.loads(output[start:])
    return parsed


@pytest.fixture(scope="module")
def cli_env(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("cli")
    cfg = make_tiny_config()
    cfg.training.epochs = 2
    cfg.pretrain.epochs = 1
    cfg.pretrain.batch_size = 256
    cfg.save(root / "tiny.yaml")
    paths = {"root": root, "cfg": root / "tiny.yaml", "ws": root / "ws", "model": root / "model"}
    run("generate-synthetic", "--out", root / "base", "--n", "900", "--seed", "1", "--config", paths["cfg"])
    run(
        "generate-synthetic",
        "--out",
        root / "new",
        "--n",
        "500",
        "--seed",
        "2",
        "--config",
        paths["cfg"],
        "--start-time",
        "1700100000",
    )
    run(
        "generate-synthetic",
        "--out",
        root / "shadow.parquet",
        "--n",
        "300",
        "--seed",
        "3",
        "--format",
        "parquet",
        "--config",
        paths["cfg"],
        "--start-time",
        "1700200000",
    )
    run(
        "generate-synthetic",
        "--out",
        root / "shifted.npz",
        "--n",
        "300",
        "--seed",
        "4",
        "--format",
        "npz",
        "--drift-shift",
        "4",
        "--config",
        paths["cfg"],
    )
    out = run(
        "train",
        "--data",
        root / "base",
        "--config",
        paths["cfg"],
        "--workspace",
        paths["ws"],
        "--out",
        paths["model"],
        "--device",
        "cpu",
    )
    assert "registered" in out and "return.rmse" in out
    return paths


def test_init_config(tmp_path: Path) -> None:
    run("init-config", "--out", tmp_path / "c.yaml")
    assert "horizons" in (tmp_path / "c.yaml").read_text()


def test_inspect_evaluate_predict(cli_env: dict[str, Path]) -> None:
    info = _last_json(run("inspect-model", "--model", cli_env["model"], "--device", "cpu"))
    assert info["ensemble_size"] == 2 and info["experts"] == [
        "transformer",
        "recurrent",
        "tcn",
        "ssm",
        "tabular",
    ]
    metrics = _last_json(
        run(
            "evaluate",
            "--model",
            cli_env["ws"],
            "--data",
            cli_env["root"] / "new",
            "--device",
            "cpu",
            "--out",
            cli_env["root"] / "eval.json",
        )
    )
    assert "upside.auc" in metrics and (cli_env["root"] / "eval.json").exists()
    out_file = cli_env["root"] / "preds.jsonl"
    run(
        "predict",
        "--model",
        cli_env["ws"],
        "--data",
        cli_env["root"] / "new",
        "--limit",
        "5",
        "--out",
        out_file,
        "--device",
        "cpu",
    )
    lines = out_file.read_text().strip().splitlines()
    first = json.loads(lines[0])
    assert len(lines) == 5 and {"expected_returns", "confidence", "expert_weights"} <= set(first)
    from nardis_neural.data.loaders import ArrayStore, select_rows
    from nardis_neural.data.sequences import arrays_to_observations

    store = ArrayStore.load(cli_env["root"] / "new")
    obs = arrays_to_observations(select_rows(store.arrays, [0, 1]), make_tiny_config())  # type: ignore[arg-type]
    (cli_env["root"] / "obs.json").write_text(json.dumps([o.to_dict() for o in obs]))
    out = run(
        "predict", "--model", cli_env["model"], "--input", cli_env["root"] / "obs.json", "--device", "cpu"
    )
    assert json.loads(out.strip().splitlines()[0])["observation_id"] == obs[0].observation_id


def test_embeddings_regimes_drift_benchmark(cli_env: dict[str, Path]) -> None:
    root = cli_env["root"]
    run(
        "extract-embeddings",
        "--model",
        cli_env["model"],
        "--data",
        root / "new",
        "--out",
        root / "emb.parquet",
        "--device",
        "cpu",
    )
    assert (root / "emb.parquet").exists()
    res = _last_json(
        run(
            "cluster-regimes",
            "--model",
            cli_env["model"],
            "--data",
            root / "new",
            "--method",
            "gmm",
            "--k",
            "3",
            "--device",
            "cpu",
            "--out",
            root / "regimes.json",
        )
    )
    assert res["n_clusters"] == 3 and len(res["clusters"]) == 3  # type: ignore[arg-type]
    drift = _last_json(
        run(
            "drift-report",
            "--model",
            cli_env["model"],
            "--reference",
            root / "base",
            "--current",
            root / "shifted.npz",
            "--device",
            "cpu",
            "--out",
            root / "drift.json",
        )
    )
    assert drift["input"] is True and drift["any_drift"] is True
    bench = _last_json(
        run(
            "benchmark",
            "--model",
            cli_env["model"],
            "--data",
            root / "new",
            "--batch-sizes",
            "1,16",
            "--device",
            "cpu",
        )
    )
    assert bench["batches"]["16"]["throughput_obs_per_s"] > 0  # type: ignore[index]


def test_continual_cycle(cli_env: dict[str, Path]) -> None:
    ws, root = cli_env["ws"], cli_env["root"]
    ingest = _last_json(run("ingest", "--workspace", ws, "--data", root / "new", "--device", "cpu"))
    assert ingest["ingested"] == 500
    adapt = _last_json(run("adapt", "--workspace", ws, "--device", "cpu"))
    assert adapt["status"] in {"challenger", "failed"}
    status = _last_json(run("status", "--workspace", ws))
    assert status["replay_size"] == 500
    if adapt["status"] == "failed":
        pytest.skip("synthetic candidate failed offline validation; lifecycle covered elsewhere")
    challenger = str(adapt["candidate_version"])
    shadow = _last_json(
        run(
            "shadow-evaluate",
            "--workspace",
            ws,
            "--data",
            root / "shadow.parquet",
            "--device",
            "cpu",
            "--out",
            root / "shadow.json",
        )
    )
    assert shadow["n"] == 300
    out = run("promote", "--workspace", ws, "--device", "cpu")
    assert "Promotion decision" in out
    if "REJECT" in out:
        run("promote", "--workspace", ws, "--version", challenger)
    assert _last_json(run("status", "--workspace", ws))["champion"] == challenger
    restored = _last_json(run("rollback", "--workspace", ws, "--reason", "cli test"))
    assert restored["champion"] != challenger
    full = _last_json(run("full-retrain", "--workspace", ws, "--force", "--device", "cpu"))
    assert full["kind"] == "full_retrain"


def test_pretrain_cli(cli_env: dict[str, Path]) -> None:
    out = cli_env["root"] / "pre.pt"
    run(
        "pretrain",
        "--data",
        cli_env["root"] / "base",
        "--config",
        cli_env["cfg"],
        "--out",
        out,
        "--epochs",
        "1",
        "--device",
        "cpu",
    )
    assert out.exists()
    run(
        "train",
        "--data",
        cli_env["root"] / "base",
        "--config",
        cli_env["cfg"],
        "--out",
        cli_env["root"] / "m2",
        "--pretrained",
        out,
        "--epochs",
        "1",
        "--ensemble-size",
        "1",
        "--device",
        "cpu",
    )
    assert (cli_env["root"] / "m2" / "manifest.json").exists()


def test_cli_errors(cli_env: dict[str, Path]) -> None:
    res = runner.invoke(app, ["train", "--data", str(cli_env["root"] / "base")])
    assert res.exit_code != 0
    res = runner.invoke(app, ["predict", "--model", str(cli_env["model"])])
    assert res.exit_code != 0


def test_train_into_existing_workspace_registers_challenger(cli_env: dict[str, Path], tmp_path: Path) -> None:
    import shutil

    ws = tmp_path / "ws"
    shutil.copytree(cli_env["ws"], ws)
    before = _last_json(run("status", "--workspace", ws))
    out = run(
        "train",
        "--data",
        cli_env["root"] / "base",
        "--config",
        cli_env["cfg"],
        "--workspace",
        ws,
        "--epochs",
        "1",
        "--ensemble-size",
        "1",
        "--device",
        "cpu",
    )
    assert "challenger" in out
    after = _last_json(run("status", "--workspace", ws))
    assert after["champion"] == before["champion"] and after["challenger"] is not None
