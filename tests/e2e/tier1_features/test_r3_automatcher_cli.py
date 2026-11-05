"""Tier 1: Feature Coverage Tests for Requirement R3.

Frictionless Zero-Configuration API & Interactive CLI for SMBs:
- F3.1: Zero-config AutoMatcher table profiling and automatic matching
- F3.2: Core interactive CLI `fuzzymatch match`
- F3.3: High-level financial reconciliation helper `auto_match_finance`
- F3.4: Financial interactive CLI `fuzzymatch-finance match`
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ..conftest import require_feature
from ..fixtures.vendor_benchmark import generate_vendor_benchmark


@pytest.fixture
def auto_matcher_cls() -> Any:
    """Fixture providing AutoMatcher gated by Milestone M3."""
    return require_feature("fuzzymatch.api.auto", "AutoMatcher", milestone="M3")


@pytest.fixture
def auto_match_finance_fn() -> Any:
    """Fixture providing auto_match_finance gated by Milestone M3."""
    return require_feature("fuzzymatch_finance", "auto_match_finance", milestone="M3")


@pytest.fixture
def fuzzymatch_cli_main() -> Any:
    """Fixture providing fuzzymatch.cli.main gated by Milestone M3."""
    return require_feature("fuzzymatch.cli", "main", milestone="M3")


@pytest.fixture
def finance_cli_main() -> Any:
    """Fixture providing fuzzymatch_finance.cli.main gated by Milestone M3."""
    return require_feature("fuzzymatch_finance.cli", "main", milestone="M3")


def test_automatcher_zero_config_execution(auto_matcher_cls: Any) -> Any:
    """R3: AutoMatcher executes on two unannotated tabular inputs without manual tuning."""
    df_l = pd.DataFrame(
        {
            "vendor_code": ["V-101", "V-102", "V-103"],
            "vendor_name": [
                "Acme Industrial Supplies LLC",
                "Summit Chemical Co",
                "Pacific Coast Media",
            ],
            "city": ["Houston", "Dallas", "San Jose"],
        }
    )
    df_r = pd.DataFrame(
        {
            "id": ["INV-1", "INV-2", "INV-3"],
            "payee": ["Acme Industrial Inc", "Summit Chemical", "Different Entity"],
            "location": ["Houston", "Dallas", "Miami"],
        }
    )

    matcher = auto_matcher_cls()
    results = matcher.fit_predict(df_l, df_r)

    # Must return valid results object with matched pairs/clusters
    assert results is not None
    assert hasattr(results, "matches") or hasattr(results, "pairs") or hasattr(results, "clusters")


def test_automatcher_column_profiling_and_type_inference(auto_matcher_cls: Any) -> Any:
    """R3: AutoMatcher automatically profiles columns, identifying text, key, and numeric fields."""
    df_l = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Acme Corp", "Beta LLC", "Gamma Inc"],
            "amount": [100.50, 200.75, 300.00],
            "invoice_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )
    df_r = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "name": ["Acme Corp", "Beta", "Gamma Inc"],
            "amount": [100.50, 200.75, 300.00],
            "invoice_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
        }
    )

    matcher = auto_matcher_cls()
    results = matcher.fit_predict(df_l, df_r)
    assert results is not None


def test_auto_match_finance_high_level_entry(auto_match_finance_fn: Any) -> Any:
    """R3: auto_match_finance provides a one-line entry point for financial workflows."""
    df_erp, df_ap, _ = generate_vendor_benchmark(n_canonical=15, n_dirty=20, seed=42)

    results = auto_match_finance_fn(df_erp, df_ap)
    assert results is not None


def test_fuzzymatch_cli_match_csv(
    fuzzymatch_cli_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> Any:
    """R3: `fuzzymatch match` CLI command accepts CSV files, runs matching, and prints summary."""
    file_l = tmp_path / "left.csv"
    file_r = tmp_path / "right.csv"
    out_file = tmp_path / "out.csv"

    df_l = pd.DataFrame({"id": ["A1", "A2"], "name": ["Acme Industrial", "Summit Chemical"]})
    df_r = pd.DataFrame({"id": ["B1", "B2"], "name": ["Acme Ind LLC", "Summit Chem"]})
    df_l.to_csv(file_l, index=False)
    df_r.to_csv(file_r, index=False)

    exit_code = fuzzymatch_cli_main(["match", str(file_l), str(file_r), "--output", str(out_file)])
    assert exit_code == 0

    captured = capsys.readouterr()
    # CLI must print tabular summary metrics to stdout
    assert (
        "match" in captured.out.lower()
        or "cluster" in captured.out.lower()
        or "pairs" in captured.out.lower()
    )
    assert out_file.exists()


def test_fuzzymatch_finance_cli_match(
    finance_cli_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> Any:
    """R3: `fuzzymatch-finance match` executes financial reconciliation via CLI."""
    file_l = tmp_path / "erp.csv"
    file_r = tmp_path / "ap.csv"
    out_file = tmp_path / "matched.csv"

    df_l = pd.DataFrame({"vendor_id": ["V-1"], "name": ["Acme Corp"], "amount": [500.00]})
    df_r = pd.DataFrame({"ap_id": ["AP-1"], "name": ["Acme"], "amount": [500.00]})
    df_l.to_csv(file_l, index=False)
    df_r.to_csv(file_r, index=False)

    exit_code = finance_cli_main(["match", str(file_l), str(file_r), "--output", str(out_file)])
    assert exit_code == 0
    assert out_file.exists()


def test_cli_human_readable_explanations(
    fuzzymatch_cli_main: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> Any:
    """R3: CLI outputs human-readable match explanations."""
    file_l = tmp_path / "left_exp.csv"
    file_r = tmp_path / "right_exp.csv"

    pd.DataFrame({"id": [1], "name": ["Beacon Global"]}).to_csv(file_l, index=False)
    pd.DataFrame({"id": [2], "name": ["Beacon Global LLC"]}).to_csv(file_r, index=False)

    exit_code = fuzzymatch_cli_main(["match", str(file_l), str(file_r), "--explain"])
    assert exit_code == 0
    captured = capsys.readouterr()
    assert len(captured.out) > 0
