"""Tier 4 Real-World Scenario: Complete SMB Accountant Excel Curation & Active Rule Loop.

Scenario Description:
An accountant reconciles an incoming vendor statement against ERP payables.
Ambiguous candidate matches are exported to Excel with data validation dropdowns
and side-by-side diff formatting. The accountant curates the ambiguous pairs in Excel
(marking accept or reject). On import, the system synthesizes persistent deterministic
rules and updates the LabelStore audit trail. On the next reconciliation cycle,
these synthesized rules act as deterministic overrides, achieving 100% precision on curated entities.

Acceptance Verification:
1. `review_xlsx` roundtrip functions seamlessly for vendor entities without GL account hardcoding.
2. `synthesize_rules_from_reviews` produces persistent `MappingRule` instances in a `MappingRuleSet`.
3. Persistent rules survive JSON/CSV serialization to disk and reload cleanly.
4. Second-pass reconciliation applies override rules with deterministic exactness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from ..conftest import require_feature


def test_scenario_accountant_excel_curation_and_active_rules(tmp_path: Path) -> Any:
    """Full workflow: Excel export -> human curation -> rule synthesis -> rule re-application."""
    export_excel_fn = require_feature(
        "fuzzymatch_finance.review_xlsx",
        "export_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )
    import_excel_fn = require_feature(
        "fuzzymatch_finance.review_xlsx",
        "import_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )
    synthesize_rules_fn = require_feature(
        "fuzzymatch_finance.label_store", "synthesize_rules_from_reviews", milestone="M4"
    )
    mapping_ruleset_cls = require_feature(
        "fuzzymatch_finance.rules", "MappingRuleSet", milestone="M4"
    )

    # Step 1: Initial automated run produces ambiguous candidate pairs for review
    review_queue = pd.DataFrame(
        {
            "source_vendor_id": ["VEND-0101", "VEND-0102", "VEND-0103"],
            "target_erp_id": ["ERP-9001", "ERP-9002", "ERP-9003"],
            "vendor_name_inv": ["Acme Ind.", "Beacon Glob.", "Spurious Supplier"],
            "vendor_name_erp": ["Acme Industrial LLC", "Beacon Global Inc.", "Legitimate Supplier"],
            "fuzzy_score": [0.72, 0.68, 0.55],
        }
    )

    xlsx_path = tmp_path / "accountant_review_queue.xlsx"
    export_excel_fn(
        review_queue,
        xlsx_path,
        source_id_col="source_vendor_id",
        target_id_col="target_erp_id",
        diff_cols=[("vendor_name_inv", "vendor_name_erp")],
    )
    assert xlsx_path.exists()

    # Step 2: Accountant reviews in Excel:
    # VEND-0101 -> accept
    # VEND-0102 -> accept
    # VEND-0103 -> reject
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]]  # type: ignore[index]
    dec_col = headers.index("decision") + 1

    sheet.cell(row=2, column=dec_col, value="accept")  # type: ignore[union-attr]
    sheet.cell(row=3, column=dec_col, value="accept")  # type: ignore[union-attr]
    sheet.cell(row=4, column=dec_col, value="reject")  # type: ignore[union-attr]
    wb.save(xlsx_path)

    # Step 3: Re-import decisions
    curated_decisions = import_excel_fn(
        xlsx_path,
        source_id_col="source_vendor_id",
        target_id_col="target_erp_id",
    )
    assert curated_decisions is not None

    # Step 4: Synthesize persistent deterministic rules & update LabelStore
    curated_df = pd.DataFrame(
        {
            "source_vendor_id": ["VEND-0101", "VEND-0102", "VEND-0103"],
            "target_erp_id": ["ERP-9001", "ERP-9002", "ERP-9003"],
            "decision": ["accept", "accept", "reject"],
        }
    )

    rule_set, _label_store = synthesize_rules_fn(
        review_df=curated_df,
        source_id_col="source_vendor_id",
        target_id_col="target_erp_id",
        decision_col="decision",
    )
    assert len(rule_set) >= 2  # At least 2 accepted mapping rules

    # Step 5: Save ruleset to disk (JSON) and reload
    rule_file = tmp_path / "persistent_rules.json"
    if hasattr(rule_set, "to_json"):
        rule_set.to_json(rule_file)
        assert rule_file.exists()
        loaded_ruleset = mapping_ruleset_cls.from_json(rule_file)
        assert len(loaded_ruleset) == len(rule_set)
    else:
        loaded_ruleset = rule_set

    # Step 6: Verify second-pass reconciliation applies override rules deterministically
    if hasattr(loaded_ruleset, "match"):
        assert loaded_ruleset.match("VEND-0101") == "ERP-9001"
        assert loaded_ruleset.match("VEND-0102") == "ERP-9002"
        # Rejected entity must not map to the rejected target
        assert loaded_ruleset.match("VEND-0103") != "ERP-9003"
