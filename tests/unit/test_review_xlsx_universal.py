"""Comprehensive unit tests for universal review_xlsx export and import."""

from __future__ import annotations

import uuid
from pathlib import Path

import openpyxl
import pandas as pd
import pytest

from fuzzymatch_finance.results import MappingResult
from fuzzymatch_finance.review_xlsx import (
    ReviewDecisions,
    _desanitize_value,
    _sanitize_value,
    export_review_xlsx,
    import_review_xlsx,
)


def test_review_decisions_container() -> None:
    df = pd.DataFrame({"source_id": ["S1"], "target_id": ["T1"], "decision": ["accept"]})
    outcomes = {"S1": True}
    dec = ReviewDecisions(outcomes, df)

    assert isinstance(dec, dict)
    assert dec == {"S1": True}
    assert dec["S1"] is True
    assert dec.dataframe is df
    df_copy = dec.to_dataframe()
    assert df_copy.equals(df)
    assert df_copy is not df


def test_sanitize_and_desanitize_helpers() -> None:
    # Non-strings and missing
    assert _sanitize_value(123) == 123
    assert _sanitize_value(None) is None
    assert _sanitize_value(float("nan")) is None
    assert _sanitize_value(pd.NA) is None
    assert _sanitize_value([1, 2]) == "[1, 2]"
    u = uuid.uuid4()
    assert _sanitize_value(u) == str(u)

    # Strings starting with formula prefixes
    assert _sanitize_value("=1+1") == "'=1+1"
    assert _sanitize_value("@SUM(A1:A5)") == "'@SUM(A1:A5)"
    assert _sanitize_value("+447123456") == "'+447123456"
    assert _sanitize_value("-100") == "'-100"
    assert _sanitize_value("Normal Text") == "Normal Text"
    assert _sanitize_value("CRLF\r\nNewline") == "CRLF\nNewline"

    # Desanitization
    assert _desanitize_value(123) == 123
    assert _desanitize_value("'") == "'"
    assert _desanitize_value("'=1+1") == "=1+1"
    assert _desanitize_value("'@SUM") == "@SUM"
    assert _desanitize_value("'+44") == "+44"
    assert _desanitize_value("'-100") == "-100"
    assert _desanitize_value("'SingleQuote") == "'SingleQuote"
    assert _desanitize_value("Normal") == "Normal"


def test_export_review_xlsx_various_input_types(tmp_path: Path) -> None:
    # 1. MappingResult without policy
    q = pd.DataFrame(
        {
            "source_id": ["S1", "S2"],
            "target_id": ["T1", "T2"],
            "fuzzy_score": [0.8, 0.6],
        }
    )
    mr = MappingResult(
        approved_mappings=pd.DataFrame(),
        review_queue=q,
        evidence=pd.DataFrame(),
        audit_log="",
    )
    p1 = tmp_path / "mr.xlsx"
    out1 = export_review_xlsx(mr, p1)
    assert out1.exists()

    # 2. Object with .review_candidates
    class FakeResultCandidates:
        def __init__(self, df: pd.DataFrame) -> None:
            self.review_candidates = df

    p2 = tmp_path / "cand.xlsx"
    export_review_xlsx(FakeResultCandidates(q), p2)
    assert p2.exists()

    # 3. Object with .matches
    class FakeResultMatches:
        def __init__(self, df: pd.DataFrame) -> None:
            self.matches = df

    p3 = tmp_path / "matches.xlsx"
    export_review_xlsx(FakeResultMatches(q), p3)
    assert p3.exists()

    # 4. Raw list of dicts
    p4 = tmp_path / "list_dicts.xlsx"
    export_review_xlsx([{"source_id": "S1", "target_id": "T1", "decision": None}], p4)
    assert p4.exists()

    # 5. Unsupported type
    with pytest.raises(TypeError, match="Unsupported results type"):
        export_review_xlsx(12345, tmp_path / "err.xlsx")


def test_export_review_xlsx_path_resolution(tmp_path: Path) -> None:
    df = pd.DataFrame({"source_id": ["S1"], "target_id": ["T1"]})

    # output_path provided
    p1 = export_review_xlsx(df, output_path=tmp_path / "p1.xlsx")
    assert p1.exists()

    # path provided
    p2 = export_review_xlsx(df, path=tmp_path / "p2.xlsx")
    assert p2.exists()

    # No path provided raises ValueError
    with pytest.raises(ValueError, match="output_path or path must be specified"):
        export_review_xlsx(df)


def test_export_review_xlsx_empty_dataframe_variations(tmp_path: Path) -> None:
    # Completely empty frame without columns
    empty_no_cols = pd.DataFrame()
    p1 = tmp_path / "empty_no_cols.xlsx"
    export_review_xlsx(empty_no_cols, p1)
    assert p1.exists()
    wb = openpyxl.load_workbook(p1)
    assert "decision" in [c.value for c in wb["review"][1]]

    # Empty frame with columns
    empty_with_cols = pd.DataFrame(columns=["source_id", "target_id", "fuzzy_score"])
    p2 = tmp_path / "empty_with_cols.xlsx"
    export_review_xlsx(empty_with_cols, p2)
    assert p2.exists()


def test_export_review_xlsx_diff_cols_highlighting(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "source_id": ["S1", "S2", "S3"],
            "target_id": ["T1", "T2", "T3"],
            "left_name": ["Acme Corp", "Beta LLC", None],
            "right_name": ["Acme Corp", "Beta Inc", "Gamma LLC"],
            "left_val": [10.0, 20.0, None],
            "right_val": [10.0000001, 25.5, None],
            "extra_info": ["E1", "E2", "E3"],
            "fuzzy_score": [0.95, 0.70, 0.50],
        }
    )
    p = tmp_path / "diffs.xlsx"
    export_review_xlsx(
        df,
        p,
        diff_cols=[("left_name", "right_name"), ("left_val", "right_val")],
    )
    assert p.exists()

    wb = openpyxl.load_workbook(p)
    ws = wb["review"]
    headers = [c.value for c in ws[1]]
    # Ensure ordered columns
    assert headers[:3] == ["source_id", "target_id", "decision"]
    assert "left_name" in headers
    assert "right_name" in headers

    # S1: identical names (no diff fill), identical numbers within 1e-6 (no diff fill)
    l_name_col = headers.index("left_name") + 1
    r_name_col = headers.index("right_name") + 1
    l_val_col = headers.index("left_val") + 1
    r_val_col = headers.index("right_val") + 1

    # Row 2 (S1) -> identical
    assert ws.cell(row=2, column=l_name_col).fill.fill_type is None
    assert ws.cell(row=2, column=r_name_col).fill.fill_type is None

    # Row 3 (S2) -> "Beta LLC" vs "Beta Inc" (diff fill)
    fill_l = ws.cell(row=3, column=l_name_col).fill
    fill_r = ws.cell(row=3, column=r_name_col).fill
    assert fill_l.fill_type == "solid"
    assert fill_r.fill_type == "solid"

    # Row 3 (S2) -> 20.0 vs 25.5 (diff fill)
    assert ws.cell(row=3, column=l_val_col).fill.fill_type == "solid"
    assert ws.cell(row=3, column=r_val_col).fill.fill_type == "solid"

    # Row 4 (S3) -> None vs "Gamma LLC" (diff fill)
    assert ws.cell(row=4, column=l_name_col).fill.fill_type == "solid"
    # Row 4 (S3) -> None vs None (no diff fill)
    assert ws.cell(row=4, column=l_val_col).fill.fill_type is None


def test_import_review_xlsx_error_handling(tmp_path: Path) -> None:
    # 1. No path
    with pytest.raises(ValueError, match="path or input_path must be provided"):
        import_review_xlsx()

    # 2. File not found
    with pytest.raises(FileNotFoundError, match="Review file not found"):
        import_review_xlsx(tmp_path / "nonexistent.xlsx")

    # 3. Missing decision column
    p_bad = tmp_path / "no_decision.xlsx"
    pd.DataFrame({"source_id": ["S1"], "target_id": ["T1"]}).to_excel(
        p_bad, sheet_name="review", index=False
    )
    with pytest.raises(ValueError, match="missing columns: .*decision"):
        import_review_xlsx(p_bad)

    # 4. Missing source ID column
    p_no_src = tmp_path / "no_src.xlsx"
    pd.DataFrame({"unknown_col": ["S1"], "decision": ["accept"]}).to_excel(
        p_no_src, sheet_name="review", index=False
    )
    with pytest.raises(ValueError, match="missing columns"):
        import_review_xlsx(p_no_src)


def test_import_review_xlsx_empty_frame_and_all_pending(tmp_path: Path) -> None:
    # Empty file
    p_empty = tmp_path / "empty.xlsx"
    pd.DataFrame(columns=["source_id", "target_id", "decision"]).to_excel(
        p_empty, sheet_name="review", index=False
    )

    dec_dict = import_review_xlsx(p_empty)
    assert len(dec_dict) == 0

    dec_df = import_review_xlsx(p_empty, as_dataframe=True)
    assert isinstance(dec_df, pd.DataFrame)
    assert dec_df.empty

    # File with only pending rows
    p_pending = tmp_path / "pending.xlsx"
    pd.DataFrame(
        {
            "source_id": ["S1", "S2"],
            "target_id": ["T1", "T2"],
            "decision": ["pending", "investigate"],
        }
    ).to_excel(p_pending, sheet_name="review", index=False)

    dec2 = import_review_xlsx(p_pending)
    assert len(dec2) == 0

    dec2_df = import_review_xlsx(p_pending, as_dataframe=True)
    assert isinstance(dec2_df, pd.DataFrame)
    assert len(dec2_df) == 0


def test_import_review_xlsx_desanitizes_formula_prefixes(tmp_path: Path) -> None:
    p = tmp_path / "sanitized.xlsx"
    df = pd.DataFrame(
        {
            "source_id": ["=CMD|calc", "@SUM(1+1)", "+12345", "-500", "Normal"],
            "target_id": ["T1", "T2", "T3", "T4", "T5"],
            "decision": ["accept", "accept", "reject", "reject", "accept"],
        }
    )
    export_review_xlsx(df, p)

    # Re-import
    dec = import_review_xlsx(p)
    assert dec.get("=CMD|calc") is True
    assert dec.get("@SUM(1+1)") is True
    assert dec.get("+12345") is False
    assert dec.get("-500") is False
    assert dec.get("Normal") is True

    # Re-import as dataframe
    df_imp = import_review_xlsx(p, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert "=CMD|calc" in df_imp["source_id"].values
    assert "@SUM(1+1)" in df_imp["source_id"].values
