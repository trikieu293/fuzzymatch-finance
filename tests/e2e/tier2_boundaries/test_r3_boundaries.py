"""Tier 2: Boundary & Corner Case Tests for Requirement R3.

AutoMatcher & CLI Boundaries:
- Empty tables and 1x1 single-row tables
- Unicode, emoji, and non-ASCII character handling
- Non-existent and malformed file handling in CLI
- All-identical and all-distinct candidate distributions
- CLI boundary arguments
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ..conftest import require_feature


@pytest.fixture
def auto_matcher_cls() -> Any:
    """Fixture providing AutoMatcher gated by Milestone M3."""
    return require_feature("fuzzymatch.api.auto", "AutoMatcher", milestone="M3")


@pytest.fixture
def fuzzymatch_cli_main() -> Any:
    """Fixture providing fuzzymatch.cli.main gated by Milestone M3."""
    return require_feature("fuzzymatch.cli", "main", milestone="M3")


def test_r3_automatcher_empty_inputs(auto_matcher_cls: Any) -> Any:
    """R3 Boundary: 0-row DataFrames passed to AutoMatcher return empty result without error."""
    df_empty_l = pd.DataFrame(columns=["id", "name", "amount"])
    df_empty_r = pd.DataFrame(columns=["id", "name", "amount"])

    matcher = auto_matcher_cls()
    results = matcher.fit_predict(df_empty_l, df_empty_r)
    assert results is not None


def test_r3_automatcher_single_row(auto_matcher_cls: Any) -> Any:
    """R3 Boundary: 1x1 DataFrames execute cleanly."""
    df_l = pd.DataFrame({"name": ["Acme Corp"]})
    df_r = pd.DataFrame({"name": ["Acme Corp"]})

    matcher = auto_matcher_cls()
    results = matcher.fit_predict(df_l, df_r)
    assert results is not None


def test_r3_automatcher_unicode_and_emojis(auto_matcher_cls: Any) -> Any:
    """R3 Boundary: Unicode, German umlauts, French accents, and emojis in text fields."""
    df_l = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["München Präzision GmbH 🚀", "Société Générale d'Électronique S.A."],
        }
    )
    df_r = pd.DataFrame(
        {
            "id": [10, 20],
            "name": ["Munchen Prazision", "Societe Generale d'Electronique"],
        }
    )

    matcher = auto_matcher_cls()
    results = matcher.fit_predict(df_l, df_r)
    assert results is not None


def test_r3_cli_missing_file_error(
    fuzzymatch_cli_main: Any, capsys: pytest.CaptureFixture[str]
) -> Any:
    """R3 Boundary: CLI cleanly exits with non-zero code on non-existent input files."""
    exit_code = fuzzymatch_cli_main(
        ["match", "non_existent_file_left.csv", "non_existent_file_right.csv"]
    )
    assert exit_code != 0
    captured = capsys.readouterr()
    assert len(captured.err) > 0 or len(captured.out) > 0


def test_r3_cli_malformed_input_file(
    fuzzymatch_cli_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> Any:
    """R3 Boundary: Malformed or unparseable input file exits cleanly with non-zero exit code."""
    bad_file = tmp_path / "corrupted.csv"
    bad_file.write_bytes(b"\x00\xff\xfe\x01\x02\x03\x04\x05")

    good_file = tmp_path / "valid.csv"
    pd.DataFrame({"id": [1], "name": ["Test"]}).to_csv(good_file, index=False)

    exit_code = fuzzymatch_cli_main(["match", str(bad_file), str(good_file)])
    assert exit_code != 0


def test_r3_automatcher_all_identical_names(auto_matcher_cls: Any) -> Any:
    """R3 Boundary: Tables where every single row has the identical name."""
    df_l = pd.DataFrame({"id": range(10), "name": ["Identical Name"] * 10})
    df_r = pd.DataFrame({"id": range(10), "name": ["Identical Name"] * 10})

    matcher = auto_matcher_cls(budget_cap=50)
    results = matcher.fit_predict(df_l, df_r)
    assert results is not None
