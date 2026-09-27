"""Model registry: champion / candidate / challenger / retired / failed lifecycle.

Every model version lives in its own immutable directory under ``<root>/models``.  The
registry (``registry.json``, written atomically) only moves *status pointers*; weights are
never overwritten, so the champion can never be mutated in place and any previous
champion can be restored exactly.  All transitions are appended to an audit log.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from nardis_neural.inference.engine import NeuralEngine

Status = Literal["champion", "candidate", "challenger", "retired", "failed"]


class StatusChange(BaseModel):
    status: Status
    at: float = Field(default_factory=time.time)
    reason: str = ""


class RegistryEntry(BaseModel):
    version: str
    status: Status
    path: str
    parent_version: str | None = None
    origin: str = "train"
    created_at: float = Field(default_factory=time.time)
    history: list[StatusChange] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    deleted: bool = False


class RegistryState(BaseModel):
    entries: dict[str, RegistryEntry] = Field(default_factory=dict)
    champion: str | None = None
    champion_history: list[str] = Field(default_factory=list)
    events: list[dict[str, Any]] = Field(default_factory=list)


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.file = self.root / "registry.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        self.state = (
            RegistryState.model_validate(json.loads(self.file.read_text()))
            if self.file.exists()
            else RegistryState()
        )

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        tmp = self.file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.state.model_dump(mode="json"), indent=2))
        tmp.replace(self.file)

    def _event(self, kind: str, **data: Any) -> None:
        self.state.events.append({"event": kind, "at": time.time(), **data})

    # ------------------------------------------------------------------ queries
    def entry(self, version: str) -> RegistryEntry:
        if version not in self.state.entries:
            raise KeyError(f"unknown model version {version}")
        return self.state.entries[version]

    def versions(self, status: Status | None = None) -> list[str]:
        items = sorted(self.state.entries.values(), key=lambda e: e.created_at)
        return [e.version for e in items if status is None or e.status == status]

    @property
    def champion_version(self) -> str | None:
        return self.state.champion

    @property
    def challenger_version(self) -> str | None:
        ch = self.versions("challenger")
        return ch[-1] if ch else None

    def path(self, version: str) -> Path:
        return self.root / self.entry(version).path

    def champion_path(self) -> Path:
        if self.state.champion is None:
            raise RuntimeError("registry has no champion")
        return self.path(self.state.champion)

    def load_engine(self, version: str, device: torch.device | str | None = None) -> NeuralEngine:
        from nardis_neural.inference.engine import NeuralEngine

        entry = self.entry(version)
        if entry.deleted:
            raise FileNotFoundError(f"weights of {version} were pruned")
        return NeuralEngine.load(self.path(version), device=device)

    # ------------------------------------------------------------------ transitions
    def register(self, engine: NeuralEngine, status: Status = "candidate", reason: str = "") -> RegistryEntry:
        version = engine.version
        if version in self.state.entries:
            raise ValueError(f"version {version} already registered (models are immutable)")
        rel = Path("models") / version
        engine.save(self.root / rel)
        entry = RegistryEntry(
            version=version,
            status=status,
            path=str(rel),
            parent_version=engine.metadata.parent_version,
            origin=engine.metadata.origin,
            history=[StatusChange(status=status, reason=reason or "registered")],
            metrics=dict(engine.metadata.training_stats.get("validation_metrics", {})),
        )
        self.state.entries[version] = entry
        self._event("register", version=version, status=status, reason=reason)
        if status == "champion":
            self._set_champion(version, reason or "initial champion")
        self.save()
        return entry

    def set_status(self, version: str, status: Status, reason: str = "") -> None:
        if status == "champion":
            raise ValueError("use promote() to make a model champion")
        entry = self.entry(version)
        if entry.status == "champion":
            raise ValueError("the champion's status can only change through promote() or rollback()")
        if status == "challenger":
            for other in self.versions("challenger"):
                if other != version:
                    self._transition(other, "retired", f"replaced by challenger {version}")
        self._transition(version, status, reason)
        self.save()

    def _transition(self, version: str, status: Status, reason: str) -> None:
        entry = self.entry(version)
        entry.status = status
        entry.history.append(StatusChange(status=status, reason=reason))
        self._event("status", version=version, status=status, reason=reason)

    def _set_champion(self, version: str, reason: str) -> None:
        old = self.state.champion
        if old is not None and old != version:
            self._transition(old, "retired", f"superseded by {version}")
        self._transition(version, "champion", reason)
        self.state.champion = version
        self.state.champion_history.append(version)

    def promote(self, version: str, reason: str = "", report: dict[str, Any] | None = None) -> None:
        entry = self.entry(version)
        if entry.deleted:
            raise FileNotFoundError(f"cannot promote pruned model {version}")
        if entry.status not in ("candidate", "challenger", "retired"):
            raise ValueError(f"cannot promote a model with status {entry.status}")
        previous = self.state.champion
        self._set_champion(version, reason or "promoted")
        self._event("promote", version=version, previous=previous, reason=reason, report=report or {})
        self.save()

    def restore(self, version: str, reason: str, demote_current_to: Status = "retired") -> str | None:
        """Make an earlier model champion again (rollback). Returns the demoted version."""
        entry = self.entry(version)
        if entry.deleted:
            raise FileNotFoundError(f"weights of {version} were pruned; cannot restore")
        current = self.state.champion
        if current == version:
            raise ValueError(f"{version} is already champion")
        if current is not None:
            self._transition(current, demote_current_to, f"rolled back: {reason}")
        self.state.champion = None
        self._set_champion(version, f"rollback: {reason}")
        self._event("rollback", version=version, previous=current, reason=reason)
        self.save()
        return current

    def previous_champion(self) -> str | None:
        """Most recent former champion that still has weights on disk."""
        current = self.state.champion
        for v in reversed(self.state.champion_history):
            if v != current and not self.entry(v).deleted:
                return v
        return None

    def prune(self, keep_champions: int) -> list[str]:
        """Delete weights of failed models and of retired models beyond the newest
        ``keep_champions`` former champions.  Registry entries stay for auditability."""
        protected = {self.state.champion}
        former = [v for v in reversed(self.state.champion_history) if v != self.state.champion]
        seen: list[str] = []
        for v in former:
            if v not in seen:
                seen.append(v)
        protected |= set(seen[:keep_champions])
        protected |= set(self.versions("challenger")) | set(self.versions("candidate"))
        removed = []
        for v, e in self.state.entries.items():
            if v in protected or e.deleted or e.status not in ("retired", "failed"):
                continue
            shutil.rmtree(self.root / e.path, ignore_errors=True)
            e.deleted = True
            removed.append(v)
        if removed:
            self._event("prune", versions=removed)
            self.save()
        return removed
