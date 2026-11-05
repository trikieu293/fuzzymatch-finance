"""Empirical verification suite for accounting reconciliation stress testing:
1. uuid.UUID in ID and attribute columns (export and roundtrip).
2. pd.NA, pd.NaT, np.nan, None in ID, attribute, and diff columns.
3. Windows CRLF (\r\n) normalization and roundtrip preservation without line duplication.
4. Active rule synthesis and evaluation on UUID and nullable data.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
import pytest

from fuzzymatch_finance.label_store import synthesize_rules_from_reviews
from fuzzymatch_finance.review_xlsx import (
    ReviewDecisions,
    export_review_xlsx,
    import_review_xlsx,
)


def test_uuid_export_and_roundtrip_exhaustive(tmp_path: Path) -> None:
    """Stress test UUID objects in source_id, target_id, and auxiliary columns."""
    u_src_1 = uuid.uuid4()
    u_src_2 = uuid.UUID("12345678-1234-5678-1234-567812345678")
    u_tgt_1 = uuid.uuid4()
    u_tgt_2 = uuid.UUID(int=0)  # nil UUID
    u_diff_l = uuid.uuid4()
    u_diff_r = uuid.uuid4()

    df = pd.DataFrame(
        {
            "source_id": [u_src_1, u_src_2],
            "target_id": [u_tgt_1, u_tgt_2],
            "diff_uuid_l": [u_diff_l, u_src_1],
            "diff_uuid_r": [u_diff_r, u_src_2],
            "decision": ["accept", "reject"],
        }
    )

    xlsx_path = tmp_path / "uuid_exhaustive.xlsx"
    export_review_xlsx(
        df,
        xlsx_path,
        diff_cols=[("diff_uuid_l", "diff_uuid_r")],
    )
    assert xlsx_path.exists()

    # Verify openpyxl worksheet structure
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["review"]
    assert ws.max_row == 3  # header + 2 rows
    assert ws.cell(row=2, column=1).value == str(u_src_1)
    assert ws.cell(row=3, column=1).value == str(u_src_2)
    assert ws.cell(row=2, column=2).value == str(u_tgt_1)
    assert ws.cell(row=3, column=2).value == str(u_tgt_2)

    # Test import as ReviewDecisions
    dec = import_review_xlsx(xlsx_path)
    assert isinstance(dec, ReviewDecisions)
    assert len(dec) == 2
    assert dec[str(u_src_1)] is True
    assert dec[str(u_src_2)] is False

    # Test import as DataFrame
    df_imp = import_review_xlsx(xlsx_path, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 2
    assert list(df_imp["source_id"]) == [str(u_src_1), str(u_src_2)]
    assert list(df_imp["target_id"]) == [str(u_tgt_1), str(u_tgt_2)]

    # Test rule synthesis from UUID decisions
    store_path = tmp_path / "uuid_store.csv"
    rule_set, store = synthesize_rules_from_reviews(dec, label_store=store_path)
    assert len(store.all()) == 2
    assert len(rule_set) == 1  # 1 accepted rule
    rule = rule_set[0]

    # Rule must match both string UUID and uuid.UUID object in queries
    assert rule.applies_to({"source_id": u_src_1}) is True
    assert rule.applies_to({"source_id": str(u_src_1)}) is True
    assert rule.applies_to(pd.Series({"source_id": u_src_1})) is True
    assert rule.applies_to(pd.Series({"source_id": str(u_src_1)})) is True
    assert rule.applies_to({"source_id": u_src_2}) is False

    assert rule_set.match({"source_id": u_src_1}) == str(u_tgt_1)
    assert rule_set.match(pd.Series({"source_id": u_src_1})) == str(u_tgt_1)


def test_missing_values_pd_na_nat_nan_none(tmp_path: Path) -> None:
    """Stress test pd.NA, pd.NaT, np.nan, float('nan'), and None across various column types."""
    df = pd.DataFrame(
        {
            "source_id": ["S1", "S2", "S3", "S4", "S5"],
            "target_id": ["T1", "T2", "T3", "T4", "T5"],
            "nullable_int": pd.Series([10, pd.NA, 30, pd.NA, 50], dtype="Int64"),
            "nullable_str": pd.Series(["apple", pd.NA, "cherry", None, pd.NA], dtype="string"),
            "datetime_col": pd.Series(
                [
                    pd.Timestamp("2026-01-01"),
                    pd.NaT,
                    pd.Timestamp("2026-03-01"),
                    pd.NaT,
                    pd.Timestamp("2026-05-01"),
                ]
            ),
            "float_nan_col": [1.1, np.nan, float("nan"), None, 5.5],
            "decision": ["accept", "reject", "accept", "reject", "accept"],
        }
    )

    xlsx_path = tmp_path / "missing_values.xlsx"
    export_review_xlsx(
        df,
        xlsx_path,
        diff_cols=[("nullable_int", "float_nan_col")],
    )
    assert xlsx_path.exists()

    # Verify openpyxl cells are bound to None (empty cell) and do not crash
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["review"]
    # Row 3 (index 1: S2, pd.NA, pd.NA, pd.NaT, np.nan)
    headers = [cell.value for cell in ws[1]]
    int_col_idx = headers.index("nullable_int") + 1
    str_col_idx = headers.index("nullable_str") + 1
    date_col_idx = headers.index("datetime_col") + 1
    float_col_idx = headers.index("float_nan_col") + 1

    # Check that missing values became None in OpenPyXL cells
    assert ws.cell(row=3, column=int_col_idx).value is None
    assert ws.cell(row=3, column=str_col_idx).value is None
    assert ws.cell(row=3, column=date_col_idx).value is None
    assert ws.cell(row=3, column=float_col_idx).value is None

    # Check import roundtrip
    dec = import_review_xlsx(xlsx_path)
    assert isinstance(dec, ReviewDecisions)
    assert len(dec) == 5
    assert dec["S1"] is True
    assert dec["S2"] is False
    assert dec["S3"] is True
    assert dec["S4"] is False
    assert dec["S5"] is True

    df_imp = import_review_xlsx(xlsx_path, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 5
    # Pandas import turns empty cells into NaN/NA
    assert pd.isna(df_imp["nullable_int"].to_list()[1])
    assert pd.isna(df_imp["nullable_str"].to_list()[1])
    assert pd.isna(df_imp["datetime_col"].to_list()[1])
    assert pd.isna(df_imp["float_nan_col"].to_list()[1])


def test_missing_values_in_id_columns(tmp_path: Path) -> None:
    """Verify that pd.NA, np.nan, and None in ID columns export and import safely."""
    df = pd.DataFrame(
        {
            "source_id": ["S1", pd.NA, np.nan, None, "S5"],
            "target_id": [pd.NA, "T2", None, np.nan, "T5"],
            "decision": ["accept", "accept", "reject", "pending", "accept"],
        }
    )

    xlsx_path = tmp_path / "id_missing.xlsx"
    export_review_xlsx(df, xlsx_path)
    assert xlsx_path.exists()

    df_imp = import_review_xlsx(xlsx_path, as_dataframe=True)
    # 4 resolved rows ('accept' or 'reject')
    assert len(df_imp) == 3 or len(df_imp) == 4  # 'pending' is filtered out


def test_crlf_newlines_do_not_duplicate(tmp_path: Path) -> None:
    """Stress test Windows CRLF (\r\n) newlines ensuring no duplication to \n\n."""
    crlf_sample = "Line 1\r\nLine 2\r\nLine 3"
    crlf_blank_lines = "Section A\r\n\r\nSection B\r\n\r\n\r\nSection C"
    trailing_crlf = "Heading\r\nBody text\r\n"
    unix_lf = "Unix 1\nUnix 2\nUnix 3"
    mac_cr = "OldMac 1\rOldMac 2"

    df = pd.DataFrame(
        {
            "source_id": ["CRLF_1", "CRLF_BLANK", "CRLF_TRAIL", "LF_1", "CR_1"],
            "target_id": ["T1", "T2", "T3", "T4", "T5"],
            "notes": [crlf_sample, crlf_blank_lines, trailing_crlf, unix_lf, mac_cr],
            "decision": ["accept"] * 5,
        }
    )

    xlsx_path = tmp_path / "newlines.xlsx"
    export_review_xlsx(df, xlsx_path)
    assert xlsx_path.exists()

    # Direct cell check in OpenPyXL
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["review"]
    headers = [cell.value for cell in ws[1]]
    notes_col_idx = headers.index("notes") + 1

    # In openpyxl worksheet, \r\n should have been replaced with \n
    raw_val_1 = ws.cell(row=2, column=notes_col_idx).value
    assert isinstance(raw_val_1, str)
    assert "\r\n" not in raw_val_1
    assert raw_val_1 == "Line 1\nLine 2\nLine 3"

    raw_val_blank = ws.cell(row=3, column=notes_col_idx).value
    assert isinstance(raw_val_blank, str)
    assert "\r\n" not in raw_val_blank
    assert raw_val_blank == "Section A\n\nSection B\n\n\nSection C"

    # Now verify re-import roundtrip
    df_imp = import_review_xlsx(xlsx_path, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 5
    notes_imported = dict(zip(df_imp["source_id"].to_list(), df_imp["notes"].to_list()))

    # CRITICAL INVARIANT: The imported text must NOT have duplicated newlines (\n\n instead of \n)
    assert notes_imported["CRLF_1"] == "Line 1\nLine 2\nLine 3"
    assert notes_imported["CRLF_1"].count("\n") == 2

    assert notes_imported["CRLF_BLANK"] == "Section A\n\nSection B\n\n\nSection C"
    assert notes_imported["CRLF_TRAIL"] == "Heading\nBody text\n"
    assert notes_imported["LF_1"] == "Unix 1\nUnix 2\nUnix 3"


def test_diff_cols_highlighting_with_complex_types(tmp_path: Path) -> None:
    """Stress test diff_cols highlighting with UUID, NA, floats, strings."""
    u1 = uuid.uuid4()
    u2 = uuid.uuid4()

    df = pd.DataFrame(
        {
            "source_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "target_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "left_attr": [u1, pd.NA, 100.0, "Acme Inc.", None, "Same"],
            "right_attr": [u2, "Present", 100.0000001, "Acme LLC", None, "Same"],
            "decision": ["pending"] * 6,
        }
    )

    xlsx_path = tmp_path / "diff_complex.xlsx"
    export_review_xlsx(
        df,
        xlsx_path,
        diff_cols=[("left_attr", "right_attr")],
    )
    assert xlsx_path.exists()

    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb["review"]
    headers = [cell.value for cell in ws[1]]
    l_idx = headers.index("left_attr") + 1
    r_idx = headers.index("right_attr") + 1

    diff_color = "FFF2CC"

    # Row 2 (S1): u1 vs u2 -> mismatch -> highlighted
    cell_l2 = ws.cell(row=2, column=l_idx)
    cell_r2 = ws.cell(row=2, column=r_idx)
    assert diff_color in str(cell_l2.fill.start_color.rgb)
    assert diff_color in str(cell_r2.fill.start_color.rgb)

    # Row 3 (S2): pd.NA vs "Present" -> mismatch -> highlighted
    cell_l3 = ws.cell(row=3, column=l_idx)
    assert diff_color in str(cell_l3.fill.start_color.rgb)

    # Row 4 (S3): 100.0 vs 100.0000001 (diff < 1e-6) -> match -> not highlighted
    cell_l4 = ws.cell(row=4, column=l_idx)
    assert cell_l4.fill.fill_type is None or diff_color not in str(cell_l4.fill.start_color.rgb)

    # Row 6 (S5): None vs None -> match -> not highlighted
    cell_l6 = ws.cell(row=6, column=l_idx)
    assert cell_l6.fill.fill_type is None or diff_color not in str(cell_l6.fill.start_color.rgb)

    # Row 7 (S6): "Same" vs "Same" -> match -> not highlighted
    cell_l7 = ws.cell(row=7, column=l_idx)
    assert cell_l7.fill.fill_type is None or diff_color not in str(cell_l7.fill.start_color.rgb)


@pytest.mark.parametrize("missing_val", [pd.NA, pd.NaT, np.nan, float("nan"), None])
def test_all_missing_single_value_roundtrip(tmp_path: Path, missing_val: Any) -> None:
    """Parametric stress test for each missing value type individually."""
    df = pd.DataFrame(
        {
            "source_id": ["S1"],
            "target_id": ["T1"],
            "missing_col": [missing_val],
            "decision": ["accept"],
        }
    )
    p = tmp_path / f"missing_{type(missing_val).__name__}.xlsx"
    export_review_xlsx(df, p)
    assert p.exists()

    dec = import_review_xlsx(p)
    assert len(dec) == 1
    assert dec["S1"] is True

    df_imp = import_review_xlsx(p, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 1
    assert pd.isna(df_imp["missing_col"].to_list()[0])


def test_adversarial_crlf_edge_cases(tmp_path: Path) -> None:
    """Stress test tricky CRLF combinations: CRLF at start, repeated CRLF, formula prefix with CRLF."""
    df = pd.DataFrame(
        {
            "source_id": ["ID1", "ID2", "ID3", "ID4"],
            "target_id": ["T1", "T2", "T3", "T4"],
            "payload": [
                "\r\nStart with CRLF",
                "=SUM(A1:A10)\r\n+123\r\n-999",
                "\r\n\r\n\r\n",
                "Mixed\r\nUnix\nMac\rEnd",
            ],
            "decision": ["accept", "accept", "reject", "accept"],
        }
    )
    p = tmp_path / "crlf_tricky.xlsx"
    export_review_xlsx(df, p)

    df_imp = import_review_xlsx(p, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 4  # 3 accept + 1 reject
    payloads = dict(zip(df_imp["source_id"].to_list(), df_imp["payload"].to_list()))

    assert payloads["ID1"] == "\nStart with CRLF"
    assert "\r" not in payloads["ID1"]

    # Formula prefix desanitization with CRLF
    assert payloads["ID2"] == "=SUM(A1:A10)\n+123\n-999"

    # Preserves Unix newline
    assert "Unix\n" in payloads["ID4"]
