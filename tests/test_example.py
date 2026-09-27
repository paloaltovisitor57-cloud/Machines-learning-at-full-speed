"""The documented integration example must keep working."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from tests.conftest import make_tiny_config


def test_integration_example_runs(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "nardis_integration.py"
    spec = importlib.util.spec_from_file_location("nardis_integration", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["nardis_integration"] = module
    spec.loader.exec_module(module)
    cfg = make_tiny_config()
    cfg.continual.min_new_samples = 100
    cfg_path = tmp_path / "cfg.yaml"
    cfg.save(cfg_path)
    status = module.main(tmp_path / "ws", cfg_path, n_live=150, device="cpu")
    assert status["champion"] is not None
    assert status["adapted"] in {"challenger", "failed"}
