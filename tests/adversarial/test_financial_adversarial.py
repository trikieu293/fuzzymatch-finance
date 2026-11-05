"""Adversarial stress and invariant verification harness for financial entity resolution.

Empirically tests:
1. Positional vs Flag CLI syntax and argument combinations for fuzzymatch and fuzzymatch-finance.
2. openpyxl review export workbook integrity, dropdown data validation, and conditional formatting.
3. Financial column resolution heuristics on messy real-world schemas.
4. Robustness against invalid, empty, corrupt files and invalid arguments.
5. Dollar volume reconciliation and variance calculation invariants (cardinality, non-negativity, double-counting).
"""

from __future__ import annotations

from pathlib import Path

import fuzzymatch.cli as core_cli
import openpyxl
import pandas as pd
import pytest

import fuzzymatch_finance.cli as fin_cli
from fuzzymatch_finance.cli import (
    FinanceSummaryMetrics,
    _resolve_column_name,
    auto_match_finance,
)

# ===========================================================================
# 1. CLI Syntax: Positional vs Flag Combinations & Robustness
# ===========================================================================


class TestCLISyntaxAndInvocation:
    """Empirical verification of CLI argument combinations and exit codes."""

    @pytest.fixture
    def sample_csvs(self, tmp_path: Path) -> tuple[Path, Path]:
        left = tmp_path / "left.csv"
        right = tmp_path / "right.csv"
        df_l = pd.DataFrame(
            {
                "bank_id": ["B1", "B2"],
                "payee_name": ["Acme Corp LLC", "Globex Inc"],
                "amount": [150.00, 250.50],
                "txn_date": ["2026-01-10", "2026-01-12"],
            }
        )
        df_r = pd.DataFrame(
            {
                "ledger_id": ["L1", "L2"],
                "vendor_name": ["Acme Corp", "Globex Corp"],
                "total": [150.00, 250.00],
                "invoice_date": ["2026-01-10", "2026-01-13"],
            }
        )
        df_l.to_csv(left, index=False)
        df_r.to_csv(right, index=False)
        return left, right

    def test_core_cli_positional_syntax(
        self, sample_csvs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        left, right = sample_csvs
        out = tmp_path / "out_core_pos.csv"
        code = core_cli.main(["match", str(left), str(right), "-o", str(out)])
        assert code == 0
        assert out.exists()
        df_res = pd.read_csv(out)
        assert len(df_res) >= 1

    def test_core_cli_flag_syntax(self, sample_csvs: tuple[Path, Path], tmp_path: Path) -> None:
        left, right = sample_csvs
        out = tmp_path / "out_core_flag.csv"
        code = core_cli.main(
            ["match", "--left", str(left), "--right", str(right), "--output", str(out)]
        )
        assert code == 0
        assert out.exists()

    def test_core_cli_short_flag_syntax(
        self, sample_csvs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        left, right = sample_csvs
        out = tmp_path / "out_core_short.csv"
        code = core_cli.main(["match", "-l", str(left), "-r", str(right), "-o", str(out)])
        assert code == 0
        assert out.exists()

    def test_core_cli_mixed_syntax(self, sample_csvs: tuple[Path, Path], tmp_path: Path) -> None:
        left, right = sample_csvs
        # positional left, flag right works
        code1 = core_cli.main(["match", str(left), "-r", str(right)])
        assert code1 == 0
        # flag left, positional right works with exit code 0
        code2 = core_cli.main(["match", "-l", str(left), str(right)])
        assert code2 == 0

    def test_core_cli_excel_output(self, sample_csvs: tuple[Path, Path], tmp_path: Path) -> None:
        left, right = sample_csvs
        out_xlsx = tmp_path / "out_core.xlsx"
        code = core_cli.main(["match", str(left), str(right), "-o", str(out_xlsx)])
        assert code == 0
        assert out_xlsx.exists()
        wb = openpyxl.load_workbook(out_xlsx)
        assert len(wb.sheetnames) >= 1

    def test_core_cli_missing_arguments(self, sample_csvs: tuple[Path, Path]) -> None:
        left, _ = sample_csvs
        # missing both
        assert core_cli.main(["match"]) == 2
        # missing right
        assert core_cli.main(["match", str(left)]) == 2
        # missing left via flag
        assert core_cli.main(["match", "-r", str(left)]) == 2

    def test_finance_cli_positional_syntax(
        self, sample_csvs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        left, right = sample_csvs
        out = tmp_path / "out_fin_pos.csv"
        rev = tmp_path / "rev_fin_pos.xlsx"
        code = fin_cli.main(
            ["match", str(left), str(right), "-o", str(out), "--review-output", str(rev)]
        )
        assert code == 0
        assert out.exists()
        assert rev.exists()

    def test_finance_cli_flag_syntax(self, sample_csvs: tuple[Path, Path], tmp_path: Path) -> None:
        left, right = sample_csvs
        out = tmp_path / "out_fin_flag.csv"
        rev = tmp_path / "rev_fin_flag.xlsx"
        code = fin_cli.main(
            [
                "match",
                "--left",
                str(left),
                "--right",
                str(right),
                "--output",
                str(out),
                "--review-out",
                str(rev),
            ]
        )
        assert code == 0
        assert out.exists()
        assert rev.exists()

    def test_finance_cli_short_flag_syntax(
        self, sample_csvs: tuple[Path, Path], tmp_path: Path
    ) -> None:
        left, right = sample_csvs
        out = tmp_path / "out_fin_short.csv"
        code = fin_cli.main(["match", "-l", str(left), "-r", str(right), "-o", str(out)])
        assert code == 0
        assert out.exists()

    def test_finance_cli_mixed_syntax(self, sample_csvs: tuple[Path, Path], tmp_path: Path) -> None:
        left, right = sample_csvs
        # positional left, flag right works
        code1 = fin_cli.main(["match", str(left), "-r", str(right)])
        assert code1 == 0
        # flag left, positional right works with exit code 0
        code2 = fin_cli.main(["match", "-l", str(left), str(right)])
        assert code2 == 0

    def test_finance_cli_excel_output(self, sample_csvs: tuple[Path, Path], tmp_path: Path) -> None:
        left, right = sample_csvs
        out_xlsx = tmp_path / "out_fin.xlsx"
        code = fin_cli.main(["match", str(left), str(right), "-o", str(out_xlsx)])
        assert code == 0
        assert out_xlsx.exists()
        wb = openpyxl.load_workbook(out_xlsx)
        assert len(wb.sheetnames) >= 1

    def test_finance_cli_missing_arguments(self, sample_csvs: tuple[Path, Path]) -> None:
        left, _ = sample_csvs
        # missing both
        assert fin_cli.main(["match"]) == 2
        # missing right
        assert fin_cli.main(["match", str(left)]) == 2
        # missing left via flag
        assert fin_cli.main(["match", "-r", str(left)]) == 2


# ===========================================================================
# 2. openpyxl Review Export: Dropdowns, Formatting, and Boundaries
# ===========================================================================


class TestOpenpyxlReviewExport:
    """Empirical verification of Excel review export structure and validation."""

    def test_review_xlsx_structure_and_validation(self, tmp_path: Path) -> None:
        df_l = pd.DataFrame(
            {
                "bank_id": ["B101", "B102"],
                "vendor_name": ["Apex Logistics Inc", "Bravo Consulting LLC"],
                "amount": [1250.00, 3400.00],
                "date": ["2026-03-01", "2026-03-05"],
            }
        )
        df_r = pd.DataFrame(
            {
                "ledger_id": ["L201", "L202"],
                "vendor_name": ["Apex Logistics", "Bravo Consulting"],
                "amount": [1250.00, 3400.00],
                "date": ["2026-03-01", "2026-03-06"],
            }
        )
        results = auto_match_finance(df_l, df_r)
        out_xlsx = tmp_path / "nested" / "deep" / "review.xlsx"
        saved_path = results.export_review_xlsx(out_xlsx)
        assert saved_path.exists()

        wb = openpyxl.load_workbook(saved_path)
        assert "review" in wb.sheetnames
        ws = wb["review"]

        # Check headers
        expected_headers = [
            "pair_id",
            "decision",
            "source_id",
            "target_id",
            "source_name",
            "target_name",
            "source_amount",
            "target_amount",
            "amount_diff",
            "source_date",
            "target_date",
            "date_diff_days",
            "fuzzy_score",
            "explanation",
        ]
        actual_headers = [
            ws.cell(row=1, column=c).value for c in range(1, len(expected_headers) + 1)
        ]
        assert actual_headers == expected_headers

        # Check freeze panes
        assert ws.freeze_panes == "A2"

        # Check header fill
        for col_idx in range(1, len(expected_headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            assert (
                cell.fill.start_color.rgb == "001F4E79" or cell.fill.start_color.value == "1F4E79"
            )
            assert cell.font.bold is True

        # Check DataValidation
        dvs = ws.data_validations.dataValidation
        assert len(dvs) >= 1
        dv = dvs[0]
        assert dv.type == "list"
        assert dv.formula1 == '"accept,reject,pending"'

        # Check conditional formatting exists on fuzzy_score
        assert len(ws.conditional_formatting) >= 1

    def test_review_xlsx_zero_records(self, tmp_path: Path) -> None:
        """Export review xlsx when no records exist or inputs are empty."""
        df_l = pd.DataFrame(columns=["id", "name", "amount", "date"])
        df_r = pd.DataFrame(columns=["id", "name", "amount", "date"])
        results = auto_match_finance(df_l, df_r)
        out_xlsx = tmp_path / "empty_review.xlsx"
        saved_path = results.export_review_xlsx(out_xlsx)
        assert saved_path.exists()

        wb = openpyxl.load_workbook(saved_path)
        ws = wb["review"]
        assert ws.cell(row=1, column=1).value == "pair_id"
        # Row 2 should be empty
        assert ws.cell(row=2, column=1).value is None


# ===========================================================================
# 3. Financial Column Resolution Heuristics on Messy Real-World Schemas
# ===========================================================================


class TestColumnResolutionHeuristics:
    """Stress testing column resolution heuristics against irregular schemas."""

    def test_candidate_exact_and_substring_matches(self) -> None:
        df = pd.DataFrame(
            {
                "  AP_ID  ": [1],
                "Posting_DT": ["2026-01-01"],
                "Net_Total_Amt": [100.50],
                "Supplier_Firm_Name": ["Test Corp"],
            }
        )
        id_col = _resolve_column_name(df, None, ("bank_id", "ap_id"), "id")
        dt_col = _resolve_column_name(df, None, ("date",), "date")
        amt_col = _resolve_column_name(df, None, ("amount",), "amount")
        txt_col = _resolve_column_name(df, None, ("name",), "text")

        assert id_col == "  AP_ID  "
        assert dt_col == "Posting_DT"
        assert amt_col == "Net_Total_Amt"
        assert txt_col == "Supplier_Firm_Name"

    def test_explicit_column_missing_raises_value_error(self) -> None:
        df = pd.DataFrame({"col1": [1], "col2": [2]})
        with pytest.raises(ValueError, match="Specified amount column 'missing' not found"):
            _resolve_column_name(df, "missing", ("amount",), "amount")

    def test_integer_column_names_handling(self) -> None:
        """Test schemas with integer column headers (common from raw Excel imports)."""
        df_l = pd.DataFrame({0: ["B1"], 1: ["Acme LLC"], 2: [500.0], 3: ["2026-01-01"]})
        df_r = pd.DataFrame({0: ["B1"], 1: ["Acme"], 2: [500.0], 3: ["2026-01-01"]})
        # If headers are integers, resolution should return None gracefully without unhandled crash
        # and auto_match_finance should fall back safely
        results = auto_match_finance(df_l, df_r)
        assert results is not None
        assert isinstance(results.summary, FinanceSummaryMetrics)


# ===========================================================================
# 4. Invalid Files & Robust Error Handling
# ===========================================================================


class TestInvalidFilesAndArguments:
    """Empirical verification that invalid inputs return clean non-zero exit codes."""

    def test_nonexistent_file_handling(self, tmp_path: Path) -> None:
        fake = tmp_path / "nonexistent.csv"
        code1 = core_cli.main(["match", str(fake), str(fake)])
        assert code1 == 1
        code2 = fin_cli.main(["match", str(fake), str(fake)])
        assert code2 == 1

    def test_zero_byte_empty_file_handling(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty.csv"
        empty.write_text("", encoding="utf-8")
        code1 = core_cli.main(["match", str(empty), str(empty)])
        assert code1 == 1
        code2 = fin_cli.main(["match", str(empty), str(empty)])
        assert code2 == 1

    def test_corrupt_binary_file_handling(self, tmp_path: Path) -> None:
        corrupt = tmp_path / "corrupt.xlsx"
        corrupt.write_bytes(b"\x00\xff\xfe\x00NotAZipFile")
        code1 = core_cli.main(["match", str(corrupt), str(corrupt)])
        assert code1 == 1
        code2 = fin_cli.main(["match", str(corrupt), str(corrupt)])
        assert code2 == 1

    def test_directory_passed_as_file(self, tmp_path: Path) -> None:
        d = tmp_path / "somedir"
        d.mkdir()
        code1 = core_cli.main(["match", str(d), str(d)])
        assert code1 == 1
        code2 = fin_cli.main(["match", str(d), str(d)])
        assert code2 == 1


# ===========================================================================
# 5. Volume Reconciliation & Variance Invariants
# ===========================================================================


class TestVolumeAndVarianceInvariants:
    """Stress tests on mathematical volume reconciliation and 1-to-1 matching invariants."""

    def test_volume_calculation_non_negativity_and_sum(self) -> None:
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Alpha Inc", "Beta Corp"],
                "amount": [-150.00, 250.00],  # Negative amount (credit / refund)
                "date": ["2026-01-01", "2026-01-02"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Alpha Inc", "Gamma Ltd"],
                "amount": [150.00, 500.00],
                "date": ["2026-01-01", "2026-01-03"],
            }
        )
        res = auto_match_finance(df_l, df_r)
        s = res.summary
        assert s.left_dollar_volume == 400.00  # |-150| + |250|
        assert s.right_dollar_volume == 650.00  # |150| + |500|
        assert s.total_dollar_volume == 1050.00
        assert s.reconciled_dollar_volume == 150.00
        assert s.unmatched_dollar_volume == 750.00  # 250 left + 500 right
        assert s.left_count == 2
        assert s.right_count == 2
        assert s.reconciled_count == 1
        assert s.unmatched_left_count == 1
        assert s.unmatched_right_count == 1

    def test_cardinality_1_to_1_double_matching_vulnerability(self) -> None:
        """CRITICAL CHALLENGE: Multiple left records matching the SAME right record.

        In 1-to-1 bank reconciliation, 1 bank transaction cannot be matched to 2
        distinct ledger transactions without splitting.
        If right table has 1 record of $1,000, and left table has 2 records of $1,000:
        Reconciled count must be <= min(|L|, |R|) = 1.
        Reconciled volume must be <= right volume = $1,000.
        """
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Acme Corp", "Acme Corp"],
                "amount": [1000.00, 1000.00],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1"],
                "name": ["Acme Corp"],
                "amount": [1000.00],
                "date": ["2026-01-01"],
            }
        )
        res = auto_match_finance(df_l, df_r)

        # Test whether right_id is deduplicated
        matched_rights = res.matches["right_id"].tolist()
        has_duplicate_rights = len(matched_rights) > len(set(matched_rights))

        # In a strict 1-to-1 match:
        # 1. Right transactions must not be assigned to multiple left transactions
        # 2. Reconciled count must not exceed right table size
        # 3. Reconciled dollar volume must not exceed right dollar volume ($1000)
        reconciled_count = res.summary.reconciled_count
        reconciled_vol = res.summary.reconciled_dollar_volume
        right_vol = res.summary.right_dollar_volume

        print("\nEmpirical 1-to-1 Invariant Check:")
        print(f"  Left table size: {len(df_l)}, Right table size: {len(df_r)}")
        print(f"  Reconciled count: {reconciled_count}")
        print(f"  Reconciled volume: {reconciled_vol}, Right volume: {right_vol}")
        print(f"  Matched pairs: {res.pairs}")

        # Record whether this invariant is violated in current implementation
        # (This assertion reveals whether double-matching occurs)
        assert not has_duplicate_rights, (
            f"VULNERABILITY CONFIRMED: Right record matched {len(matched_rights)} times! "
            f"Reconciled count {reconciled_count} > min(2, 1)=1. "
            f"Reconciled volume ${reconciled_vol} > Right volume ${right_vol}."
        )

    def test_unmatched_right_volume_discrepancy_vulnerability(self) -> None:
        """CRITICAL CHALLENGE: Unmatched right dollar volume completely omitted from variance volume.

        If Left table has $100 and Right table has $100 and $900:
        Left has 0 unmatched records ($0.00).
        Right has 1 unmatched record ($900.00).
        Total unmatched variance volume must be $900.00!
        Currently, auto_match_finance calculates:
            unm_vol = max(0.0, l_vol - rec_vol)
        which results in $0.00 variance volume despite $900 of unmatched transactions.
        """
        df_l = pd.DataFrame(
            {
                "id": ["L1"],
                "name": ["Acme Corp"],
                "amount": [100.00],
                "date": ["2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Acme Corp", "Zeta Logistics"],
                "amount": [100.00, 900.00],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )
        res = auto_match_finance(df_l, df_r)
        s = res.summary
        assert s.unmatched_right_count == 1
        # The expected unmatched variance dollar volume is $900.00:
        # If the code incorrectly reports $0.00, this assertion documents the bug.
        assert s.unmatched_dollar_volume == 900.00, (
            f"VULNERABILITY CONFIRMED: Unmatched right record of $900.00 ignored! "
            f"unmatched_dollar_volume reported ${s.unmatched_dollar_volume:,.2f} instead of $900.00."
        )
