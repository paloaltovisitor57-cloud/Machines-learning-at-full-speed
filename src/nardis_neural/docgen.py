"""Build README.md: the hand-written docs plus a reference generated from the code.

    python -m nardis_neural.docgen            # rewrite README.md
    python -m nardis_neural.docgen --check    # exit 1 if README.md is stale

Part I is ``docs/OVERVIEW.md``.  Part II is every document in ``docs/`` with its headings
nested one level.  Part III is generated from the live code, so it cannot drift: every
CLI command and option, every configuration field with its default and description,
every feature, every output field, the public Python API, and the test inventory.  A test
fails whenever README.md differs from what this module produces.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
import re
import sys
import types
import typing
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
PARTS = (
    ("ARCHITECTURE", "Architecture"),
    ("CONTINUAL_LEARNING", "Continual learning"),
    ("INTEGRATION", "Integration"),
    ("SOLANA", "Solana intelligence layer"),
    ("EDGE", "Edge engine"),
    ("MOONSHOT", "Moonshot engine"),
    ("TAPE", "Tape Transformer"),
)
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _cell(text: object) -> str:
    """One markdown table cell."""
    s = " ".join(str(text).split())
    return s.replace("|", "\\|") if s else " "


def _links(text: str) -> str:
    """Links written relative to docs/ → relative to the repository root."""
    text = re.sub(r"\]\(([A-Z_]+\.md)", r"](docs/\1", text)
    return text.replace("](../", "](")


def _demote(text: str) -> str:
    """Nest every heading one level deeper (code blocks untouched)."""
    out, fence = [], False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        out.append("#" + line if not fence and line.startswith("#") else line)
    return "\n".join(out)


# ------------------------------------------------------------------------------ CLI
def _cli() -> Iterator[str]:
    import typer.main

    from nardis_neural.cli import app

    def walk(cmd: Any, path: str) -> Iterator[str]:
        subs = getattr(cmd, "commands", None)
        if subs:
            for name in subs:
                yield from walk(subs[name], f"{path} {name}")
            return
        yield f"### `{path}`"
        yield ""
        doc = inspect.cleandoc(cmd.help or "")
        if doc:
            yield doc
            yield ""
        rows = []
        for p in cmd.params:
            if p.name == "help":
                continue
            names = ", ".join(f"`{o}`" for o in [*p.opts, *getattr(p, "secondary_opts", [])])
            kind = "flag" if getattr(p, "is_flag", False) else getattr(p.type, "name", str(p.type))
            default = "required" if p.required else _cell(json.dumps(p.default, default=str))
            env = f" (env `{p.envvar}`)" if getattr(p, "envvar", None) else ""
            rows.append(f"| {names} | {kind} | {default} | {_cell(getattr(p, 'help', '') or '')}{env} |")
        if rows:
            yield "| option | type | default | description |"
            yield "|---|---|---|---|"
            yield from rows
            yield ""

    yield "## Command line"
    yield ""
    yield "Every command of the `nardis-neural` entry point (`--help` on any command prints the same)."
    yield ""
    yield from walk(typer.main.get_command(app), "nardis-neural")


# ------------------------------------------------------------------------------ configuration
def _type_name(ann: Any) -> str:
    if isinstance(ann, type):
        return ann.__name__
    return str(ann).replace("typing.", "").replace("nardis_neural.", "")


def _model_type(ann: Any) -> type[BaseModel] | None:
    if isinstance(ann, type) and issubclass(ann, BaseModel):
        return ann
    for arg in typing.get_args(ann):
        found = _model_type(arg)
        if found is not None:
            return found
    return None


def _model_rows(model: type[BaseModel], prefix: str = "") -> Iterator[str]:
    nested: list[tuple[str, type[BaseModel]]] = []
    for name, field in model.model_fields.items():
        key = f"{prefix}{name}"
        sub = _model_type(field.annotation)
        default = field.get_default(call_default_factory=True)
        if isinstance(default, BaseModel):
            default_s = "(section below)"
        elif isinstance(default, list) and default and isinstance(default[0], BaseModel):
            default_s = _cell(json.dumps([d.model_dump(mode="json") for d in default]))
        else:
            default_s = _cell(json.dumps(default, default=str))
        yield (
            f"| `{key}` | {_cell(_type_name(field.annotation))} | {default_s} | "
            f"{_cell(field.description or '')} |"
        )
        if sub is not None and isinstance(default, BaseModel):
            nested.append((f"{key}.", sub))
    for p, sub in nested:
        yield from _model_rows(sub, p)


def _config() -> Iterator[str]:
    from nardis_neural.config import HorizonConfig, NeuralConfig
    from nardis_neural.solana.config import BarSpec, SolanaConfig
    from nardis_neural.solana.edge.barriers import BarrierSpec
    from nardis_neural.solana.moonshot.guard import GuardConfig
    from nardis_neural.solana.moonshot.labels import MoonshotSpec
    from nardis_neural.solana.tape.features import TapeSpec

    yield "## Configuration"
    yield ""
    yield (
        "Every field, its type, its default and its description. Nested keys use dots, as in "
        "the YAML files (`nardis-neural init-config`, `nardis-neural solana init-config`)."
    )
    sections: list[tuple[str, type[BaseModel]]] = [
        ("Neural engine — `NeuralConfig` (YAML root)", NeuralConfig),
        ("Solana layer — `SolanaConfig`", SolanaConfig),
        ("Forecast horizon entries — `HorizonConfig`", HorizonConfig),
        ("Solana bar streams — `BarSpec`", BarSpec),
        ("Edge labels — `BarrierSpec`", BarrierSpec),
        ("Moonshot labels and ladder — `MoonshotSpec`", MoonshotSpec),
        ("Manipulation guard — `GuardConfig`", GuardConfig),
        ("Trade tape — `TapeSpec`", TapeSpec),
    ]
    for title, model in sections:
        yield ""
        yield f"### {title}"
        yield ""
        yield "| key | type | default | description |"
        yield "|---|---|---|---|"
        yield from _model_rows(model)


# ------------------------------------------------------------------------------ features & outputs
def _features() -> Iterator[str]:
    from nardis_neural.solana.config import (
        BAR_FEATURE_DOCS,
        CURRENT_FEATURES,
        EDGE_TYPES,
        FEATURE_DOCS,
        NODE_FEATURE_DOCS,
        RISK_LABELS,
        SolanaConfig,
    )

    yield "## Solana features"
    yield ""
    yield f"### Current-state vector ({len(CURRENT_FEATURES)} features, in model input order)"
    yield ""
    yield "| # | feature | meaning |"
    yield "|---|---|---|"
    for i, name in enumerate(CURRENT_FEATURES):
        yield f"| {i} | `{name}` | {_cell(FEATURE_DOCS[name])} |"
    cfg = SolanaConfig()
    bars = ", ".join(f"`{b.name}` {b.resolution_seconds:g} s × {b.length}" for b in cfg.bars)
    yield ""
    yield f"### Trade bars ({bars})"
    yield ""
    yield "| # | bar feature | meaning |"
    yield "|---|---|---|"
    for i, (name, doc) in enumerate(BAR_FEATURE_DOCS.items()):
        yield f"| {i} | `{name}` | {_cell(doc)} |"
    yield ""
    yield "### Wallet graph"
    yield ""
    yield "| # | node feature | meaning |"
    yield "|---|---|---|"
    for i, (name, doc) in enumerate(NODE_FEATURE_DOCS.items()):
        yield f"| {i} | `{name}` | {_cell(doc)} |"
    yield ""
    yield "Edge types: " + ", ".join(f"`{e}`" for e in EDGE_TYPES) + "."
    from nardis_neural.solana.tape.features import TRADE_FEATURE_DOCS, TapeSpec

    yield ""
    yield f"### Trade tape (last {TapeSpec().max_trades} trades, one vector per trade + a hashed wallet id)"
    yield ""
    yield "| # | trade feature | meaning |"
    yield "|---|---|---|"
    for i, (name, doc) in enumerate(TRADE_FEATURE_DOCS.items()):
        yield f"| {i} | `{name}` | {_cell(doc)} |"
    yield ""
    yield "Launch-risk labels: " + ", ".join(f"`{r}`" for r in RISK_LABELS) + "."
    horizons = ", ".join(
        f"`{h.name}` ({h.seconds:g} s, up ≥ {h.upside_threshold:g}, down ≥ {h.downside_threshold:g})"
        for h in cfg.horizons
    )
    yield ""
    yield f"Forecast horizons: {horizons}."


def _fields(model: type[BaseModel]) -> Iterator[str]:
    yield "| field | type | description |"
    yield "|---|---|---|"
    for name, field in model.model_fields.items():
        yield f"| `{name}` | {_cell(_type_name(field.annotation))} | {_cell(field.description or '')} |"


def _outputs() -> Iterator[str]:
    from nardis_neural.schemas import NeuralObservation, NeuralOutcome, NeuralPrediction
    from nardis_neural.solana.brain import SolanaAssessment
    from nardis_neural.solana.config import FEATURE_DOCS
    from nardis_neural.solana.moonshot.guard import assess_manipulation
    from nardis_neural.solana.moonshot.labels import MoonshotSpec

    yield "## Inputs and outputs"
    for title, model in (
        ("`NeuralObservation` (input)", NeuralObservation),
        ("`NeuralOutcome` (label reported later)", NeuralOutcome),
        ("`NeuralPrediction` (output)", NeuralPrediction),
        ("`SolanaAssessment` (Solana output)", SolanaAssessment),
    ):
        yield ""
        yield f"### {title}"
        yield ""
        yield from _fields(model)
    edge_keys = {
        "p_win": "calibrated P(the executable round trip beats zero)",
        "expected_net": "expected net return of the round trip (stacked learners)",
        "uncertainty": "disagreement between the stacked learners",
        "edge_score": "expected_net − λ · uncertainty (lower confidence bound)",
        "kelly_fraction": "capped fractional-Kelly sizing hint",
        "threshold": "entry threshold chosen on the research tune period",
        "above_threshold": "1.0 when edge_score ≥ threshold",
    }
    levels = MoonshotSpec().levels
    moon_keys = {f"p_ge_{k:g}x": f"calibrated P(peak multiple ≥ {k:g}x)" for k in levels} | {
        "median_multiple": "median of the predicted peak multiple",
        "expected_multiple": "expected ladder payoff per SOL under the predicted distribution",
        "lottery_kelly": "lottery-Kelly bankroll fraction × trust (0 when vetoed)",
        "tail_index": "power-law exponent of the predicted far tail (smaller = heavier)",
        "epistemic": "largest ensemble spread of P(peak ≥ k)",
        "out_of_range_share": "share of inputs outside the training range (clamped)",
        "in_entry_window": "1.0 inside the moonshot entry window",
        "trust": "manipulation-guard trust in [0, 1]",
        "vetoed": "1.0 when a hard veto fired",
        "chase_score": "trust × expected_multiple (0 when vetoed)",
        "chase_rank": "rank of chase_score within the assessed batch (1 = best)",
    }
    clean = dict.fromkeys(FEATURE_DOCS, 0.0) | {
        "mint_authority_revoked": 1.0,
        "freeze_authority_revoked": 1.0,
    }
    for factor in assess_manipulation(clean, None, 0.0, 0.0, 0.0).factors:
        moon_keys[f"guard.{factor}"] = f"guard factor `{factor}` (1 = no concern; the product is trust)"
    from nardis_neural.solana.tape.model import DEFAULT_BINS, window_label

    tape_keys = {f"p_ge_{k:g}x": f"calibrated P(peak multiple ≥ {k:g}x) from the tape" for k in levels} | {
        "expected_multiple": "expected ladder payoff per SOL",
        "median_multiple": "median predicted peak multiple",
        "lottery_kelly": "lottery-Kelly bankroll fraction (not trust-adjusted)",
        "tail_index": "power-law exponent of the predicted far tail",
        "epistemic": "largest ensemble spread of P(peak ≥ k)",
    }
    for b in DEFAULT_BINS:
        tape_keys[f"p_collapse_{window_label(b)}"] = f"P(the ticket's value halves within {window_label(b)})"
    for title, keys in (
        ("`SolanaAssessment.edge` keys", edge_keys),
        ("`SolanaAssessment.moonshot` keys", moon_keys),
        ("`SolanaAssessment.tape` keys", tape_keys),
    ):
        yield ""
        yield f"### {title}"
        yield ""
        yield "| key | meaning |"
        yield "|---|---|"
        for k, v in keys.items():
            yield f"| `{k}` | {_cell(v)} |"


# ------------------------------------------------------------------------------ API
def _first_line(obj: Any) -> str:
    """First paragraph of the object's *own* docstring (never inherited or auto-generated)."""
    if inspect.isclass(obj):
        doc = obj.__dict__.get("__doc__") or ""
        if doc.startswith(f"{obj.__name__}("):  # dataclass placeholder signature
            doc = ""
        doc = inspect.cleandoc(doc)
    else:
        doc = inspect.getdoc(obj) or ""
    return " ".join(doc.strip().split("\n\n")[0].split()) if doc else ""


def _summary(obj: Any) -> str:
    text = _cell(_first_line(obj)).strip()
    return f" — {text}" if text else ""


def _signature(fn: Any) -> str:
    try:
        sig = str(inspect.signature(fn))
    except (TypeError, ValueError):
        return "(…)"
    return _ADDR.sub("", sig.replace("nardis_neural.", ""))


def _api() -> Iterator[str]:
    import nardis_neural

    yield "## Python API"
    yield ""
    yield "Every public module, class, method and function, with its signature and summary."
    names = sorted(m.name for m in pkgutil.walk_packages(nardis_neural.__path__, "nardis_neural."))
    for modname in ["nardis_neural", *names]:
        mod = importlib.import_module(modname)
        members = [
            (n, o)
            for n, o in vars(mod).items()
            if not n.startswith("_")
            and (inspect.isclass(o) or inspect.isfunction(o))
            and getattr(o, "__module__", None) == modname
        ]
        yield ""
        yield f"### `{modname}`"
        yield ""
        summary = _first_line(mod)
        if summary:
            yield summary
            yield ""
        for n, o in sorted(members, key=lambda kv: kv[0].lower()):
            if inspect.isclass(o):
                yield f"- **class `{n}`**{_summary(o)}"
                for mn, mo in sorted(vars(o).items()):
                    fn = mo.__func__ if isinstance(mo, classmethod | staticmethod) else mo
                    if mn.startswith("_") or not isinstance(fn, types.FunctionType):
                        continue
                    yield f"  - `{mn}{_signature(fn)}`{_summary(fn)}"
            else:
                yield f"- `{n}{_signature(o)}`{_summary(o)}"


# ------------------------------------------------------------------------------ tests
def _test_functions() -> list[tuple[str, str, str]]:
    out = []
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test_"):
                out.append((path.name, node.name, (ast.get_docstring(node) or "").split("\n")[0]))
    return out


def _tests() -> Iterator[str]:
    tests = _test_functions()
    yield "## Test inventory"
    yield ""
    yield f"{len(tests)} test functions (some are parametrised over devices, experts or formats)."
    current = ""
    for file, name, doc in tests:
        if file != current:
            current = file
            module_doc = (ast.get_docstring(ast.parse((ROOT / "tests" / file).read_text())) or "").split(
                "\n\n"
            )[0]
            yield ""
            yield f"### `tests/{file}`"
            yield ""
            if module_doc:
                yield " ".join(module_doc.split())
                yield ""
        yield f"- `{name}`" + (f" — {doc}" if doc else "")


# ------------------------------------------------------------------------------ assembly
def build() -> str:
    """Assemble the full README text: overview, every ``docs/`` part and the generated reference."""
    overview = (ROOT / "docs" / "OVERVIEW.md").read_text()
    overview = overview.replace("{{TESTS}}", str(len(_test_functions())))
    chunks = [
        "<!-- Generated by `python -m nardis_neural.docgen` from docs/*.md and the code. "
        "Edit the sources, not this file. -->",
        "",
        _links(overview).rstrip(),
        "",
        "# Part II — In depth",
        "",
        "The complete contents of every document in `docs/`.",
    ]
    for stem, _ in PARTS:
        chunks += ["", _links(_demote((ROOT / "docs" / f"{stem}.md").read_text())).rstrip()]
    chunks += ["", "# Part III — Generated reference", "", "Generated from the code; it cannot drift."]
    for section in (_cli, _config, _features, _outputs, _api, _tests):
        chunks += ["", *section()]
    return "\n".join(chunks).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    """Rewrite README.md, or with ``--check`` only compare it; returns the exit code (1 when stale)."""
    args = sys.argv[1:] if argv is None else argv
    text = build()
    target = ROOT / "README.md"
    if "--check" in args:
        if not target.exists() or target.read_text() != text:
            print("README.md is stale: run python -m nardis_neural.docgen")
            return 1
        return 0
    target.write_text(text)
    print(f"wrote {target} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
