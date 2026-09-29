"""Packaging metadata: every file pattern shipped with a wheel matches real files."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_data_and_yaml_presets_match_files() -> None:
    tool = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["setuptools"]
    for package, patterns in tool["package-data"].items():
        pkg_dir = ROOT / "src" / package.replace(".", "/")
        for pattern in patterns:
            assert list(pkg_dir.glob(pattern)), f"package-data {package}:{pattern} matches nothing"
    shipped = [f for patterns in tool["data-files"].values() for p in patterns for f in ROOT.glob(p)]
    assert {f.name for f in shipped} >= {"default.yaml", "small.yaml"}, "YAML presets are installed"
