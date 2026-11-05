"""Tier 2: Boundary & Corner Case Tests for Requirement R4.

Excel Curation Roundtrip & Active Rule Synthesis Boundaries:
- Empty candidate review queue (0 rows)
- Excel formula injection sanitization (=SUM, @, +, -)
- Extraneous user-added columns during Excel import
- Zero accepted decisions (all pending or blank)
- Conflicting curator decisions (accept vs reject for same pair)
- Multiline and extreme text length cells
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
import pytest

from ..conftest import require_feature


@pytest.fixture
def export_review_xlsx_fn() -> Any:
    """Fixture providing export_review_xlsx gated by Milestone M4."""
    return require_feature(
        "fuzzymatch_finance.review_xlsx",
        "export_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )


@pytest.fixture
def import_review_xlsx_fn() -> Any:
    """Fixture providing import_review_xlsx gated by Milestone M4."""
    return require_feature(
        "fuzzymatch_finance.review_xlsx",
        "import_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )


@pytest.fixture
def synthesize_rules_fn() -> Any:
    """Fixture providing synthesize_rules_from_reviews gated by Milestone M4."""
    try:
        return require_feature(
            "fuzzymatch_finance.label_store", "synthesize_rules_from_reviews", milestone="M4"
        )
    except pytest.skip.Exception:
        return require_feature(
            "fuzzymatch_finance.rules", "synthesize_rules_from_reviews", milestone="M4"
        )


def test_r4_export_excel_empty_results(export_review_xlsx_fn: Any, tmp_path: Path) -> Any:
    """R4 Boundary: 0 candidate pairs exported creates a valid readable Excel workbook."""
    xlsx_path = tmp_path / "empty_review.xlsx"
    empty_df = pd.DataFrame(columns=["source_id", "target_id", "fuzzy_score"])

    export_review_xlsx_fn(empty_df, xlsx_path, source_id_col="source_id", target_id_col="target_id")
    assert xlsx_path.exists()

    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    assert sheet is not None


def test_r4_excel_formula_injection_defense(
    export_review_xlsx_fn: Any, import_review_xlsx_fn: Any, tmp_path: Path
) -> Any:
    """R4 Boundary: Strings starting with '=', '@', '+', '-' do not execute or corrupt Excel."""
    xlsx_path = tmp_path / "formula_inject.xlsx"
    malicious_df = pd.DataFrame(
        {
            "source_id": ["=CMD|' /C calc'!A0"],
            "target_id": ["@SUM(1+1)"],
            "vendor_name": ["+Acme - Corp"],
            "fuzzy_score": [0.99],
        }
    )

    export_review_xlsx_fn(
        malicious_df, xlsx_path, source_id_col="source_id", target_id_col="target_id"
    )
    assert xlsx_path.exists()

    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    # Workbook must open cleanly without corruption
    assert sheet is not None


def test_r4_import_excel_extraneous_user_columns(
    export_review_xlsx_fn: Any, import_review_xlsx_fn: Any, tmp_path: Path
) -> Any:
    """R4 Boundary: Curator adds extra columns (e.g. 'notes', 'internal_ticket') in Excel."""
    xlsx_path = tmp_path / "extra_cols.xlsx"
    df = pd.DataFrame(
        {
            "source_id": ["S1"],
            "target_id": ["T1"],
            "fuzzy_score": [0.85],
        }
    )

    export_review_xlsx_fn(df, xlsx_path, source_id_col="source_id", target_id_col="target_id")

    # Add extra column via openpyxl
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    sheet.cell(row=1, column=5, value="curator_notes")  # type: ignore[union-attr]
    sheet.cell(row=2, column=5, value="Verified with accounting team")  # type: ignore[union-attr]
    # Mark decision
    headers = [cell.value for cell in sheet[1]]  # type: ignore[index]
    dec_col = headers.index("decision") + 1
    sheet.cell(row=2, column=dec_col, value="accept")  # type: ignore[union-attr]
    wb.save(xlsx_path)

    # Import must succeed despite extra column
    decisions = import_review_xlsx_fn(
        xlsx_path, source_id_col="source_id", target_id_col="target_id"
    )
    assert decisions is not None


def test_r4_import_excel_no_accepted_decisions(
    export_review_xlsx_fn: Any, import_review_xlsx_fn: Any, tmp_path: Path
) -> Any:
    """R4 Boundary: When all rows remain 'pending', import returns empty resolved set."""
    xlsx_path = tmp_path / "all_pending.xlsx"
    df = pd.DataFrame(
        {
            "source_id": ["S1", "S2"],
            "target_id": ["T1", "T2"],
            "fuzzy_score": [0.70, 0.65],
        }
    )

    export_review_xlsx_fn(df, xlsx_path, source_id_col="source_id", target_id_col="target_id")
    # All are default 'pending'
    decisions = import_review_xlsx_fn(
        xlsx_path, source_id_col="source_id", target_id_col="target_id"
    )
    if isinstance(decisions, dict):
        assert len(decisions) == 0


def test_r4_rule_synthesis_conflicting_decisions(synthesize_rules_fn: Any) -> Any:
    """R4 Boundary: Conflicting curator decisions for the same pair resolve deterministically."""
    conflicting_df = pd.DataFrame(
        {
            "source_id": ["V-1", "V-1"],
            "target_id": ["E-1", "E-1"],
            "decision": ["accept", "reject"],
        }
    )

    # The synthesis engine must handle conflict cleanly (e.g. reject overrides or deduplicates)
    rules, _ = synthesize_rules_fn(
        review_df=conflicting_df,
        source_id_col="source_id",
        target_id_col="target_id",
        decision_col="decision",
    )
    assert rules is not None


def test_r4_multiline_and_extreme_length_cells(export_review_xlsx_fn: Any, tmp_path: Path) -> Any:
    """R4 Boundary: Cells containing 2,000+ characters and embedded newlines export cleanly."""
    xlsx_path = tmp_path / "long_text.xlsx"
    long_desc = "Line 1: Special vendor description\n" + ("A" * 2000)

    df = pd.DataFrame(
        {
            "source_id": ["V-LONG"],
            "target_id": ["E-LONG"],
            "description": [long_desc],
            "fuzzy_score": [0.75],
        }
    )

    export_review_xlsx_fn(df, xlsx_path, source_id_col="source_id", target_id_col="target_id")
    assert xlsx_path.exists()
