"""Self-contained, reproducible model checkpoints.

Layout of a model directory (written atomically, never modified afterwards)::

    <version>/
      manifest.json      metadata: version, parent, timestamps, git commit, data fingerprint,
                         ensemble metadata, training statistics, target definitions,
                         reference statistics, format version, library versions
      config.yaml        full NeuralConfig (architecture, horizons, everything)
      members/member_<i>.pt   network weights (state_dict, loaded with weights_only=True)
      normalizer.json    feature / target normalisation state
      calibration.json   probability calibrators + calibration report
      ood.json           OOD detector state
      regimes.json       optional regime clusterer
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, Field

from nardis_neural.config import CLASSIFICATION_TASKS, REGRESSION_TASKS, NeuralConfig
from nardis_neural.data.normalization import FeatureNormalizer
from nardis_neural.inference.calibration import CalibrationSet
from nardis_neural.inference.ood import OODDetector
from nardis_neural.models.ensemble import DeepEnsemble
from nardis_neural.models.main import NardisNeuralNetwork
from nardis_neural.regimes.clustering import RegimeClusterer

FORMAT_VERSION = 1


def git_commit() -> str | None:
    """HEAD commit of the source checkout, or None when git or the repository is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=Path(__file__).resolve().parent,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def new_version_id(prefix: str = "m") -> str:
    """New unique version id ``<prefix>-<UTC %Y%m%dT%H%M%S>-<6 hex chars>``."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


def target_definitions(config: NeuralConfig) -> dict[str, Any]:
    """Human-readable definitions of every target, event label and horizon (in seconds)."""
    return {
        "regression": {
            "return": "log(price[t+h] / price[t])",
            "max_upside": "max(0, max_{t<s<=t+h} log(price[s]/price[t]))",
            "max_drawdown": "max(0, -min_{t<s<=t+h} log(price[s]/price[t]))  (positive magnitude)",
            "volatility": "realised volatility over (t, t+h]",
        },
        "classification": {
            "upside": {h.name: f"return > {h.upside_threshold}" for h in config.targets.horizons},
            "downside": {h.name: f"max_drawdown > {h.downside_threshold}" for h in config.targets.horizons},
        },
        "horizons": {h.name: h.seconds for h in config.targets.horizons},
        "tasks": list(REGRESSION_TASKS) + list(CLASSIFICATION_TASKS),
    }


class ModelMetadata(BaseModel):
    """Checkpoint manifest (``manifest.json``): lineage, provenance, statistics and versions."""

    version: str
    parent_version: str | None = None
    created_at: float = Field(default_factory=time.time)
    git_commit: str | None = None
    data_fingerprint: str | None = None
    training_stats: dict[str, Any] = Field(default_factory=dict)
    ensemble: dict[str, Any] = Field(default_factory=dict)
    reference: dict[str, float] = Field(default_factory=dict)
    """Reference statistics from validation data (e.g. median epistemic uncertainty)."""
    expert_names: list[str] = Field(default_factory=list)
    horizons: list[str] = Field(default_factory=list)
    target_definitions: dict[str, Any] = Field(default_factory=dict)
    origin: str = "train"
    """train | adapt | full_retrain | manual"""
    format_version: int = FORMAT_VERSION
    torch_version: str = torch.__version__
    package_version: str = "0.1.0"

    @property
    def created_at_iso(self) -> str:
        """``created_at`` as an ISO-8601 UTC string."""
        return datetime.fromtimestamp(self.created_at, UTC).isoformat()


@dataclass
class CheckpointContents:
    """Everything a checkpoint directory holds, as loaded objects."""

    config: NeuralConfig
    ensemble: DeepEnsemble
    normalizer: FeatureNormalizer
    calibration: CalibrationSet
    ood: OODDetector | None
    regimes: RegimeClusterer | None
    metadata: ModelMetadata


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=float))


def save_checkpoint(path: str | Path, contents: CheckpointContents, overwrite: bool = False) -> Path:
    """Write ``contents`` atomically to ``path`` via a temporary sibling directory; returns ``path``.

    Fills in the metadata's ensemble size, expert names, horizons, target definitions and
    (if unset) git commit, mutating ``contents.metadata``.  A non-empty existing directory
    raises FileExistsError unless ``overwrite``, which deletes it first.
    """
    target = Path(path)
    if target.exists() and any(target.iterdir()):
        if not overwrite:
            raise FileExistsError(f"refusing to overwrite existing checkpoint {target}")
        shutil.rmtree(target)
    tmp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex[:6]}")
    (tmp / "members").mkdir(parents=True)
    try:
        md = contents.metadata
        md.ensemble = md.ensemble | {"size": contents.ensemble.size}
        md.expert_names = list(contents.ensemble.member(0).expert_names)
        md.horizons = contents.config.horizon_names
        md.target_definitions = target_definitions(contents.config)
        if md.git_commit is None:
            md.git_commit = git_commit()
        _write_json(tmp / "manifest.json", md.model_dump(mode="json"))
        contents.config.save(tmp / "config.yaml")
        for i in range(contents.ensemble.size):
            state = {k: v.detach().cpu() for k, v in contents.ensemble.member(i).state_dict().items()}
            torch.save(state, tmp / "members" / f"member_{i}.pt")
        _write_json(tmp / "normalizer.json", contents.normalizer.to_dict())
        _write_json(tmp / "calibration.json", contents.calibration.to_dict())
        if contents.ood is not None:
            _write_json(tmp / "ood.json", contents.ood.to_dict())
        if contents.regimes is not None:
            _write_json(tmp / "regimes.json", contents.regimes.to_dict())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.rmdir()
        tmp.replace(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> CheckpointContents:
    """Load a checkpoint directory; ensemble members are put in eval mode on ``device``.

    Raises FileNotFoundError without ``manifest.json`` and ValueError for a newer format.
    """
    root = Path(path)
    if not (root / "manifest.json").exists():
        raise FileNotFoundError(f"{root} is not a model checkpoint (manifest.json missing)")
    metadata = ModelMetadata.model_validate(json.loads((root / "manifest.json").read_text()))
    if metadata.format_version > FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {metadata.format_version} is newer than supported {FORMAT_VERSION}"
        )
    config = NeuralConfig.load(root / "config.yaml")
    members = []
    for i in range(int(metadata.ensemble.get("size", config.ensemble.size))):
        net = NardisNeuralNetwork(config)
        state = torch.load(root / "members" / f"member_{i}.pt", map_location="cpu", weights_only=True)
        net.load_state_dict(state)
        net.eval()
        members.append(net)
    ensemble = DeepEnsemble(members).to(torch.device(device))
    normalizer = FeatureNormalizer.from_dict(json.loads((root / "normalizer.json").read_text()))
    calibration = CalibrationSet.from_dict(json.loads((root / "calibration.json").read_text()))
    ood = (
        OODDetector.from_dict(json.loads((root / "ood.json").read_text()))
        if (root / "ood.json").exists()
        else None
    )
    regimes = (
        RegimeClusterer.from_dict(json.loads((root / "regimes.json").read_text()))
        if (root / "regimes.json").exists()
        else None
    )
    return CheckpointContents(config, ensemble, normalizer, calibration, ood, regimes, metadata)
