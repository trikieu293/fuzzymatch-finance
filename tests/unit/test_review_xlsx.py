"""v1.2 Phase B contracts: XLSX review export/import round-trip."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch_finance.policy import MappingPolicy
from fuzzymatch_finance.results import MappingResult
from fuzzymatch_finance.review_xlsx import export_review_xlsx, import_review_xlsx

POLICY = MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.6)


def _result(n: int = 6) -> MappingResult:
    queue = pd.DataFrame(
        {
            "source_local_account": [f"s{i}" for i in range(n)],
            "fuzzy_score": [i / n for i in range(n)],
            "suggested_target": [f"t{i}" for i in range(n)],
        }
    )
    return MappingResult(
        approved_mappings=pd.DataFrame(),
        review_queue=queue,
        evidence=pd.DataFrame(),
        audit_log="",
    )


class TestExportReviewXlsx:
    def test_creates_file_with_decision_dropdown(self, tmp_path: Path) -> None:
        p = tmp_path / "review.xlsx"
        export_review_xlsx(_result(), POLICY, p)
        from openpyxl import load_workbook

        wb = load_workbook(p)
        ws = wb["review"]
        header = [c.value for c in ws[1]]
        assert "source_local_account" in header
        assert "decision" in header
        dec_idx = header.index("decision") + 1
        decisions = [
            row[0]
            for row in ws.iter_rows(min_col=dec_idx, max_col=dec_idx, min_row=2, values_only=True)
        ]
        assert all(v == "pending" for v in decisions)
        validations = ws.data_validations.dataValidation
        assert validations, "expected a data-validation dropdown"
        assert any('"accept,reject,pending"' in (dv.formula1 or "") for dv in validations)
        assert ws.freeze_panes == "A2"

    def test_export_is_deterministic_with_seed(self, tmp_path: Path) -> None:
        from openpyxl import load_workbook

        pa = tmp_path / "a.xlsx"
        pb = tmp_path / "b.xlsx"
        export_review_xlsx(_result(), POLICY, pa, seed=7)
        export_review_xlsx(_result(), POLICY, pb, seed=7)
        rows_a = list(load_workbook(pa)["review"].iter_rows(min_row=2, values_only=True))
        rows_b = list(load_workbook(pb)["review"].iter_rows(min_row=2, values_only=True))
        assert rows_a == rows_b


class TestImportReviewXlsx:
    def test_roundtrip_outcomes(self, tmp_path: Path) -> None:
        from openpyxl import load_workbook

        p = tmp_path / "review.xlsx"
        export_review_xlsx(_result(4), POLICY, p)

        wb = load_workbook(p)
        ws = wb["review"]
        header = [c.value for c in ws[1]]
        dec_col = header.index("decision") + 1
        ws.cell(row=2, column=dec_col, value="accept")
        ws.cell(row=3, column=dec_col, value="reject")
        wb.save(p)

        outcomes = import_review_xlsx(p)
        # queue is boundary-prioritized: s2 (0.5) and s3 (0.75) sit nearest
        # the 0.6 boundary, so they occupy the first two data rows
        assert outcomes == {"s2": True, "s3": False}

    def test_missing_columns_rejected(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.xlsx"
        pd.DataFrame({"foo": [1]}).to_excel(p, sheet_name="review", index=False)
        with pytest.raises(ValueError, match="missing columns"):
            import_review_xlsx(p)
