"""The README is generated from docs/ and the code; these tests keep it complete and in sync."""

from __future__ import annotations

from pathlib import Path

from nardis_neural import docgen
from nardis_neural.solana.config import (
    BAR_FEATURE_DOCS,
    BAR_FEATURES,
    CURRENT_FEATURES,
    FEATURE_DOCS,
    NODE_FEATURE_DOCS,
    NODE_FEATURES,
)

README = Path(__file__).resolve().parents[1] / "README.md"


def test_every_feature_is_documented() -> None:
    assert list(FEATURE_DOCS) == list(CURRENT_FEATURES), "FEATURE_DOCS must follow model input order"
    assert list(BAR_FEATURE_DOCS) == list(BAR_FEATURES)
    assert list(NODE_FEATURE_DOCS) == list(NODE_FEATURES)
    assert all(doc.strip() for doc in (*FEATURE_DOCS.values(), *BAR_FEATURE_DOCS.values()))
    from nardis_neural.solana.tape.features import TRADE_FEATURE_DOCS, TRADE_FEATURES

    assert list(TRADE_FEATURE_DOCS) == list(TRADE_FEATURES)


def test_readme_is_generated_and_up_to_date() -> None:
    text = docgen.build()
    assert README.read_text() == text, "README.md is stale: run `python -m nardis_neural.docgen`"
    for name in CURRENT_FEATURES:
        assert f"`{name}`" in text
    for part in (
        "# Part II — In depth",
        "# Part III — Generated reference",
        "## Command line",
        "## Python API",
    ):
        assert part in text
    assert "solana stream-train" in text and "solana moonshot-research" in text


def test_every_doc_is_in_the_readme() -> None:
    docs = {p.stem for p in (README.parent / "docs").glob("*.md")} - {"OVERVIEW"}
    assert docs == {stem for stem, _ in docgen.PARTS}, "add every docs/*.md to docgen.PARTS"
