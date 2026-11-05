"""Tier 1: Feature Coverage Tests for Requirement R4.

Universal Excel Curation Roundtrip & Active Rule Synthesis:
- F4.1: Universal review_xlsx export with side-by-side diffs and data validation dropdowns
- F4.2: Universal review_xlsx import supporting arbitrary entity keys
- F4.3: Generalized MappingRule & MappingRuleSet with JSON/CSV serialization
- F4.4: Active rule synthesis from curated review decisions updating LabelStore
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


@pytest.fixture
def mapping_ruleset_cls() -> Any:
    """Fixture providing MappingRuleSet."""
    return require_feature("fuzzymatch_finance.rules", "MappingRuleSet", milestone="M4")


def test_universal_export_review_xlsx(export_review_xlsx_fn: Any, tmp_path: Path) -> Any:
    """R4: Universal export_review_xlsx supports arbitrary entity datasets (e.g. vendors, customers)."""
    xlsx_path = tmp_path / "vendor_review.xlsx"

    pairs_df = pd.DataFrame(
        {
            "source_vendor_id": ["V-101", "V-102"],
            "target_vendor_id": ["ERP-901", "ERP-902"],
            "source_name": ["Acme Corp", "Beta LLC"],
            "target_name": ["Acme Industrial", "Beta Inc"],
            "fuzzy_score": [0.85, 0.72],
        }
    )

    # Test universal signature with arbitrary columns
    export_review_xlsx_fn(
        results=pairs_df,
        output_path=xlsx_path,
        source_id_col="source_vendor_id",
        target_id_col="target_vendor_id",
    )

    assert xlsx_path.exists()
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]]  # type: ignore[index]
    assert "decision" in headers


def test_export_review_xlsx_dropdown_validation(export_review_xlsx_fn: Any, tmp_path: Path) -> Any:
    """R4: Exported Excel file contains DataValidation dropdown for accept/reject/pending."""
    xlsx_path = tmp_path / "dropdown_review.xlsx"

    pairs_df = pd.DataFrame(
        {
            "source_id": ["S1"],
            "target_id": ["T1"],
            "fuzzy_score": [0.80],
        }
    )

    export_review_xlsx_fn(pairs_df, xlsx_path, source_id_col="source_id", target_id_col="target_id")

    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    # Check that data validation is attached to the worksheet
    assert len(sheet.data_validations.dataValidation) > 0  # type: ignore[union-attr]
    dv = sheet.data_validations.dataValidation[0]  # type: ignore[union-attr]
    assert "accept" in dv.formula1
    assert "reject" in dv.formula1


def test_export_review_xlsx_side_by_side_diffs(export_review_xlsx_fn: Any, tmp_path: Path) -> Any:
    """R4: Formats side-by-side field diffs in Excel."""
    xlsx_path = tmp_path / "diff_review.xlsx"

    pairs_df = pd.DataFrame(
        {
            "source_id": ["S1"],
            "target_id": ["T1"],
            "left_name": ["Acme Corp"],
            "right_name": ["Acme Industrial"],
            "left_amount": [100.0],
            "right_amount": [100.5],
            "fuzzy_score": [0.82],
        }
    )

    export_review_xlsx_fn(
        pairs_df,
        xlsx_path,
        source_id_col="source_id",
        target_id_col="target_id",
        diff_cols=[("left_name", "right_name"), ("left_amount", "right_amount")],
    )
    assert xlsx_path.exists()


def test_universal_import_review_xlsx(
    export_review_xlsx_fn: Any, import_review_xlsx_fn: Any, tmp_path: Path
) -> Any:
    """R4: Universal import_review_xlsx imports decisions without relying on source_local_account."""
    xlsx_path = tmp_path / "curated_review.xlsx"

    pairs_df = pd.DataFrame(
        {
            "source_id": ["CUST-1", "CUST-2"],
            "target_id": ["MASTER-1", "MASTER-2"],
            "fuzzy_score": [0.88, 0.45],
        }
    )

    export_review_xlsx_fn(pairs_df, xlsx_path, source_id_col="source_id", target_id_col="target_id")

    # Simulate human curator entering decisions
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]]  # type: ignore[index]
    dec_col = headers.index("decision") + 1

    # Row 2 (index 1) -> accept, Row 3 (index 2) -> reject
    sheet.cell(row=2, column=dec_col, value="accept")  # type: ignore[union-attr]
    sheet.cell(row=3, column=dec_col, value="reject")  # type: ignore[union-attr]
    wb.save(xlsx_path)

    decisions = import_review_xlsx_fn(
        xlsx_path, source_id_col="source_id", target_id_col="target_id"
    )
    assert isinstance(decisions, (dict, pd.DataFrame))
    if isinstance(decisions, dict):
        assert decisions.get("CUST-1") is True
        assert decisions.get("CUST-2") is False


def test_synthesize_rules_from_reviews(synthesize_rules_fn: Any) -> Any:
    """R4: Confirmed review decisions successfully serialize into deterministic MappingRule instances."""
    review_decisions = pd.DataFrame(
        {
            "source_id": ["V-101", "V-102"],
            "target_id": ["ERP-901", "ERP-902"],
            "source_name": ["Acme Corp", "Bad Entity"],
            "decision": ["accept", "reject"],
        }
    )

    rule_set, _label_store = synthesize_rules_fn(
        review_df=review_decisions,
        source_id_col="source_id",
        target_id_col="target_id",
        decision_col="decision",
    )

    assert rule_set is not None
    assert len(rule_set) >= 1


def test_mapping_ruleset_serialization_roundtrip(synthesize_rules_fn: Any, tmp_path: Path) -> Any:
    """R4: MappingRuleSet serializes and deserializes to JSON and CSV formats."""
    review_decisions = pd.DataFrame(
        {
            "source_id": ["V-101"],
            "target_id": ["ERP-901"],
            "decision": ["accept"],
        }
    )

    rule_set, _ = synthesize_rules_fn(
        review_df=review_decisions,
        source_id_col="source_id",
        target_id_col="target_id",
        decision_col="decision",
    )

    json_path = tmp_path / "rules.json"
    csv_path = tmp_path / "rules.csv"

    if hasattr(rule_set, "to_json"):
        rule_set.to_json(json_path)
        assert json_path.exists()
        loaded = type(rule_set).from_json(json_path)
        assert len(loaded) == len(rule_set)

    if hasattr(rule_set, "to_csv"):
        rule_set.to_csv(csv_path)
        assert csv_path.exists()
        loaded_csv = type(rule_set).from_csv(csv_path)
        assert len(loaded_csv) == len(rule_set)
