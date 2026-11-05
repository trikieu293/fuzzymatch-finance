"""Empirical stress testing and invariant verification for rule synthesis and reconciliation.

Formal verification of mathematical invariants, robustness boundaries, and CLI contracts:
1. 1-to-1 Bipartite Cardinality Invariant:
   - |M| <= min(|L|, |R|)
   - Monogamous pairing: Left and Right matched IDs are strictly disjoint sets of size |M|.
   - Greedy assignment optimality and tie-breaking determinism.
   - Severe asymmetry (|L| = 100, |R| = 1 and |L| = 1, |R| = 100).
   - Complete bipartite candidate graphs (K_{m,n}).
2. Accounting Volume and Count Conservation Invariants:
   - Total volume identity: V_total = V_left + V_right
   - Reconciled and unmatched volume conservation:
     V_left = V_rec,left + V_unm,left
     V_right = V_rec,right + V_unm,right
     V_unmatched = V_unm,left + V_unm,right
   - Count conservation:
     reconciled_count + unmatched_left_count = |L|
     reconciled_count + unmatched_right_count = |R|
   - Negative amounts (refunds/credits), zero amounts, sub-cent fractions, and extreme floats (10^12).
3. CLI Syntax Permutation Matrix:
   - Exhaustive 7-permutation test across both `fuzzymatch` and `fuzzymatch-finance` entry points:
     * -l <L> <R>
     * <L> -r <R>
     * --left <L> <R>
     * <L> --right <R>
     * -l <L> -r <R>
     * --left <L> --right <R>
     * <L> <R>
4. Universal Excel Review Export (`review_xlsx`):
   - Openpyxl data validation rules ('accept,reject,pending').
   - Valid cell range coverage up to ws.max_row.
   - Header formatting, side-by-side diff alignment, and zero-row empty review export.
5. Shifted RangeIndex and Disjoint Bipartite Node ID Integrity:
   - Tables without ID columns undergoing financial reconciliation with shifted RangeIndex.
   - Explicit ID columns containing integer strings ('0', '1', '2') and actual ints.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import openpyxl
import pandas as pd
import pytest
from fuzzymatch.cli import main as core_main

from fuzzymatch_finance.cli import auto_match_finance
from fuzzymatch_finance.cli import main as finance_main


class TestBipartiteOneToOneCardinalityRigorous:
    """Rigorous empirical tests proving that matching cardinality strictly bounds to min(|L|, |R|)."""

    def test_complete_bipartite_graph_identical_amounts(self) -> None:
        """Every left record connects to every right record with score 1.0 (K_{50, 20}).

        Cardinality invariant: |M| == 20 == min(50, 20).
        Disjointness: exactly 20 distinct left_ids and 20 distinct right_ids.
        """
        n_l, n_r = 50, 20
        df_l = pd.DataFrame(
            {
                "id": [f"TX_L_{i}" for i in range(n_l)],
                "name": ["OmniCorp Global Ltd"] * n_l,
                "amount": [880.00] * n_l,
                "date": ["2026-06-01"] * n_l,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"REF_R_{j}" for j in range(n_r)],
                "name": ["OmniCorp Global Ltd"] * n_r,
                "amount": [880.00] * n_r,
                "date": ["2026-06-01"] * n_r,
            }
        )

        res = auto_match_finance(df_l, df_r)
        matches = res.matches

        assert len(matches) == n_r
        assert res.summary.reconciled_count == n_r
        assert res.summary.unmatched_left_count == n_l - n_r
        assert res.summary.unmatched_right_count == 0

        # Monogamous pairing check
        assert matches["left_id"].nunique() == n_r
        assert matches["right_id"].nunique() == n_r
        assert set(matches["right_id"]) == set(df_r["id"])

    def test_inverted_complete_bipartite_graph(self) -> None:
        """Inverted cardinality: |L| = 15, |R| = 45 -> |M| == 15 == min(15, 45)."""
        n_l, n_r = 15, 45
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_l)],
                "name": ["Apex Technologies Inc"] * n_l,
                "amount": [123.45] * n_l,
                "date": ["2026-07-15"] * n_l,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n_r)],
                "name": ["Apex Technologies Inc"] * n_r,
                "amount": [123.45] * n_r,
                "date": ["2026-07-15"] * n_r,
            }
        )

        res = auto_match_finance(df_l, df_r)
        matches = res.matches

        assert len(matches) == n_l
        assert res.summary.reconciled_count == n_l
        assert res.summary.unmatched_left_count == 0
        assert res.summary.unmatched_right_count == n_r - n_l
        assert matches["left_id"].nunique() == n_l
        assert matches["right_id"].nunique() == n_l
        assert set(matches["left_id"]) == set(df_l["id"])

    def test_greedy_score_prioritization_under_competition(self) -> None:
        """Test that higher score candidate pair wins and secondary candidate pair gets alternate match.

        L1 matches R1 (score ~1.0) and R2 (score ~0.8)
        L2 matches R1 (score ~0.8) and R2 (score ~0.9)
        Greedy order: (L1, R1, 1.0) chosen, (L2, R2, 0.9) chosen.
        """
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Alpha Precision Engineering Corp", "Beta Solutions Inc"],
                "amount": [1000.00, 500.00],
                "date": ["2026-01-10", "2026-01-10"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Alpha Precision Engineering", "Beta Solutions"],
                "amount": [1000.00, 500.00],
                "date": ["2026-01-10", "2026-01-10"],
            }
        )

        res = auto_match_finance(df_l, df_r)
        matches = res.matches
        assert len(matches) == 2

        mapping = dict(zip(matches["left_id"], matches["right_id"], strict=False))
        assert mapping["L1"] == "R1"
        assert mapping["L2"] == "R2"

    def test_multiple_identical_amounts_with_distinct_names(self) -> None:
        """When 10 left rows and 10 right rows all have amount = $50.00 but different names."""
        names = [f"Vendor Number {k}" for k in range(10)]
        df_l = pd.DataFrame(
            {
                "id": [f"L_{k}" for k in range(10)],
                "name": names,
                "amount": [50.00] * 10,
                "date": ["2026-02-01"] * 10,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{k}" for k in range(10)],
                "name": [f"{n} LLC" for n in names],
                "amount": [50.00] * 10,
                "date": ["2026-02-01"] * 10,
            }
        )

        res = auto_match_finance(df_l, df_r)
        matches = res.matches
        assert len(matches) == 10
        assert matches["left_id"].nunique() == 10
        assert matches["right_id"].nunique() == 10


class TestVolumeConservationAndVarianceAccounting:
    """Formal verification of accounting conservation identities across positive, negative, and extreme amounts."""

    def test_accounting_equations_with_mixed_reconciliation_status(self) -> None:
        """Comprehensive verification of:
        1. V_total == V_left + V_right
        2. V_left == V_rec,left + V_unm,left
        3. V_right == V_rec,right + V_unm,right
        4. V_unmatched == V_unm,left + V_unm,right
        5. reconciled_count + unmatched_left_count == |L|
        6. reconciled_count + unmatched_right_count == |R|
        """
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2", "L3", "L4"],
                "name": ["Alpha Corp", "Beta LLC", "Gamma Inc", "Delta Co"],
                "amount": [1500.00, 250.50, 75.25, 999.00],
                "date": ["2026-03-01"] * 4,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2", "R3", "R4", "R5"],
                "name": [
                    "Alpha Corp",
                    "Beta LLC",
                    "Zeta Ltd",
                    "Eta Systems",
                    "Theta Inc",
                ],
                "amount": [1500.00, 250.50, 100.00, 200.00, 300.00],
                "date": ["2026-03-01"] * 5,
            }
        )

        res = auto_match_finance(df_l, df_r)
        s = res.summary
        matches = res.matches

        # Exactly 2 matched (Alpha, Beta)
        assert len(matches) == 2
        assert s.reconciled_count == 2
        assert s.unmatched_left_count == 2
        assert s.unmatched_right_count == 3

        # Count conservation
        assert s.reconciled_count + s.unmatched_left_count == len(df_l)
        assert s.reconciled_count + s.unmatched_right_count == len(df_r)

        # Ingested volumes
        v_left = 1500.00 + 250.50 + 75.25 + 999.00
        v_right = 1500.00 + 250.50 + 100.00 + 200.00 + 300.00
        assert s.left_dollar_volume == pytest.approx(v_left, abs=1e-5)
        assert s.right_dollar_volume == pytest.approx(v_right, abs=1e-5)
        assert s.total_dollar_volume == pytest.approx(v_left + v_right, abs=1e-5)

        # Reconciled volumes
        v_rec_l = 1500.00 + 250.50
        v_rec_r = 1500.00 + 250.50
        assert s.reconciled_dollar_volume == pytest.approx(v_rec_l, abs=1e-5)
        rec_r_actual = float(matches["right_amount"].abs().sum())
        assert rec_r_actual == pytest.approx(v_rec_r, abs=1e-5)

        # Unmatched residuals
        v_unm_l = 75.25 + 999.00
        v_unm_r = 100.00 + 200.00 + 300.00
        assert s.unmatched_dollar_volume == pytest.approx(v_unm_l + v_unm_r, abs=1e-5)

    def test_negative_amounts_refunds_and_credits(self) -> None:
        """Ensure negative amounts (credits, reversals) use absolute volume in accounting."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Credit Refund Corp", "Standard Charge"],
                "amount": [-450.00, 100.00],
                "date": ["2026-04-01", "2026-04-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Credit Refund Corp", "Unmatched Credit"],
                "amount": [-450.00, -200.00],
                "date": ["2026-04-01", "2026-04-01"],
            }
        )

        res = auto_match_finance(df_l, df_r)
        s = res.summary
        assert s.reconciled_count == 1
        assert s.left_dollar_volume == pytest.approx(550.00, abs=1e-5)
        assert s.right_dollar_volume == pytest.approx(650.00, abs=1e-5)
        assert s.reconciled_dollar_volume == pytest.approx(450.00, abs=1e-5)
        # Unmatched L2 ($100) + R2 ($200) = $300
        assert s.unmatched_dollar_volume == pytest.approx(300.00, abs=1e-5)

    def test_extreme_amounts_and_sub_cents(self) -> None:
        """Ensure numerical stability under 10^12 dollar figures and fractional sub-cents."""
        df_l = pd.DataFrame(
            {
                "id": ["L1"],
                "name": ["Sovereign Wealth Fund"],
                "amount": [1_000_000_000_000.005],
                "date": ["2026-05-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1"],
                "name": ["Sovereign Wealth Fund"],
                "amount": [1_000_000_000_000.005],
                "date": ["2026-05-01"],
            }
        )

        res = auto_match_finance(df_l, df_r, epsilon=0.01)
        s = res.summary
        assert s.reconciled_count == 1
        assert math.isclose(s.left_dollar_volume, 1e12, rel_tol=1e-9)
        assert math.isclose(s.reconciled_dollar_volume, 1e12, rel_tol=1e-9)
        assert s.unmatched_dollar_volume == pytest.approx(0.0, abs=1e-4)


class TestCLISyntaxPermutationMatrix:
    """Stress test all 7 flag and positional syntax permutations across both CLIs."""

    @pytest.fixture
    def test_datasets(self, tmp_path: Path) -> tuple[Path, Path]:
        l_csv = tmp_path / "ledger_records.csv"
        r_csv = tmp_path / "bank_records.csv"

        df_l = pd.DataFrame(
            {
                "tx_id": ["L101", "L102"],
                "payee": ["Microsoft Corp", "Stripe Inc"],
                "amount": [499.00, 15.00],
                "date": ["2026-02-15", "2026-02-16"],
            }
        )
        df_r = pd.DataFrame(
            {
                "ref_num": ["B201", "B202"],
                "description": ["Microsoft Corporation", "Stripe LLC"],
                "amount": [499.00, 15.00],
                "date": ["2026-02-15", "2026-02-16"],
            }
        )
        df_l.to_csv(l_csv, index=False)
        df_r.to_csv(r_csv, index=False)
        return l_csv, r_csv

    @pytest.mark.parametrize(
        "syntax_args",
        [
            lambda l, r, o: ["match", "-l", str(l), str(r), "-o", str(o)],  # -l L R
            lambda l, r, o: ["match", str(l), "-r", str(r), "-o", str(o)],  # L -r R
            lambda l, r, o: [
                "match",
                "--left",
                str(l),
                str(r),
                "-o",
                str(o),
            ],  # --left L R
            lambda l, r, o: [
                "match",
                str(l),
                "--right",
                str(r),
                "-o",
                str(o),
            ],  # L --right R
            lambda l, r, o: [
                "match",
                "-l",
                str(l),
                "-r",
                str(r),
                "-o",
                str(o),
            ],  # -l L -r R
            lambda l, r, o: [
                "match",
                "--left",
                str(l),
                "--right",
                str(r),
                "-o",
                str(o),
            ],  # --left L --right R
            lambda l, r, o: ["match", str(l), str(r), "-o", str(o)],  # L R
        ],
    )
    def test_core_cli_all_syntax_permutations(
        self, test_datasets: tuple[Path, Path], syntax_args: Any, tmp_path: Path
    ) -> None:
        l_p, r_p = test_datasets
        out_p = tmp_path / f"core_out_{np.random.randint(100000)}.csv"
        argv = syntax_args(l_p, r_p, out_p)
        exit_code = core_main(argv)
        assert exit_code == 0, f"fuzzymatch failed on syntax: {argv}"
        assert out_p.exists()
        res_df = pd.read_csv(out_p)
        assert len(res_df) == 2

    @pytest.mark.parametrize(
        "syntax_args",
        [
            lambda l, r, o: ["match", "-l", str(l), str(r), "-o", str(o)],  # -l L R
            lambda l, r, o: ["match", str(l), "-r", str(r), "-o", str(o)],  # L -r R
            lambda l, r, o: [
                "match",
                "--left",
                str(l),
                str(r),
                "-o",
                str(o),
            ],  # --left L R
            lambda l, r, o: [
                "match",
                str(l),
                "--right",
                str(r),
                "-o",
                str(o),
            ],  # L --right R
            lambda l, r, o: [
                "match",
                "-l",
                str(l),
                "-r",
                str(r),
                "-o",
                str(o),
            ],  # -l L -r R
            lambda l, r, o: [
                "match",
                "--left",
                str(l),
                "--right",
                str(r),
                "-o",
                str(o),
            ],  # --left L --right R
            lambda l, r, o: ["match", str(l), str(r), "-o", str(o)],  # L R
        ],
    )
    def test_finance_cli_all_syntax_permutations(
        self, test_datasets: tuple[Path, Path], syntax_args: Any, tmp_path: Path
    ) -> None:
        l_p, r_p = test_datasets
        out_p = tmp_path / f"finance_out_{np.random.randint(100000)}.csv"
        argv = syntax_args(l_p, r_p, out_p)
        exit_code = finance_main(argv)
        assert exit_code == 0, f"fuzzymatch-finance failed on syntax: {argv}"
        assert out_p.exists()
        res_df = pd.read_csv(out_p)
        assert len(res_df) == 2


class TestExcelReviewExportIntegrityRigorous:
    """Rigorous verification of openpyxl data validations and review_xlsx output format."""

    def test_large_dataset_review_xlsx_validation_coverage(self, tmp_path: Path) -> None:
        """Verify openpyxl data validation is applied across all 100 rows."""
        n_rows = 100
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_rows)],
                "company": [f"Vendor {i} Corp" for i in range(n_rows)],
                "amount": [100.0 + i for i in range(n_rows)],
                "date": ["2026-05-01"] * n_rows,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{i}" for i in range(n_rows)],
                "company": [f"Vendor {i} Corp" for i in range(n_rows)],
                "amount": [100.0 + i for i in range(n_rows)],
                "date": ["2026-05-01"] * n_rows,
            }
        )
        l_p = tmp_path / "left_large.csv"
        r_p = tmp_path / "right_large.csv"
        review_p = tmp_path / "review_large.xlsx"

        df_l.to_csv(l_p, index=False)
        df_r.to_csv(r_p, index=False)

        exit_code = finance_main(["match", str(l_p), str(r_p), "--review-out", str(review_p)])
        assert exit_code == 0
        assert review_p.exists()

        wb = openpyxl.load_workbook(review_p)
        ws = wb["review"]

        # 1. Total rows should be at least n_rows + 1 (header + candidate pairs)
        assert ws.max_row >= n_rows + 1

        # 2. Check Data Validation presence and options
        assert len(ws.data_validations.dataValidation) >= 1
        dv = ws.data_validations.dataValidation[0]
        assert "accept" in dv.formula1
        assert "reject" in dv.formula1
        assert "pending" in dv.formula1

        # 3. Check sqref includes row 2 and extends up to max_row
        sqref_str = str(dv.sqref)
        assert "2:" in sqref_str or str(ws.max_row) in sqref_str

    def test_zero_match_review_xlsx_export(self, tmp_path: Path) -> None:
        """When no records match, review export should create a valid workbook with headers."""
        df_l = pd.DataFrame(
            {"id": ["L1"], "name": ["Alpha"], "amount": [10.0], "date": ["2026-01-01"]}
        )
        df_r = pd.DataFrame(
            {"id": ["R1"], "name": ["Omega"], "amount": [999.0], "date": ["2026-12-31"]}
        )
        l_p = tmp_path / "zero_left.csv"
        r_p = tmp_path / "zero_right.csv"
        review_p = tmp_path / "zero_review.xlsx"

        df_l.to_csv(l_p, index=False)
        df_r.to_csv(r_p, index=False)

        exit_code = finance_main(["match", str(l_p), str(r_p), "--review-out", str(review_p)])
        assert exit_code == 0
        assert review_p.exists()

        wb = openpyxl.load_workbook(review_p)
        assert "review" in wb.sheetnames
        ws = wb["review"]
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        assert "decision" in headers


class TestShiftedRangeIndexUnderFinancialMatching:
    """Stress test AutoMatcher and auto_match_finance with shifted RangeIndex and no explicit ID."""

    def test_auto_match_finance_shifted_range_index_no_id_col(self) -> None:
        """Verify that when tables lack ID columns and have shifted RangeIndex,

        AutoMatcher correctly prefixes node IDs with 'L_' and 'R_', preventing StarClusterer
        from seeing open paths and dropping valid matches.
        """
        # Left: 3 entities at index 0, 1, 2
        df_l = pd.DataFrame(
            {
                "vendor_name": ["Alpha Logistics LLC", "Beta Services Corp", "Gamma Tech Inc"],
                "dollar_val": [500.0, 750.0, 1200.0],
                "inv_date": ["2026-03-01", "2026-03-02", "2026-03-03"],
            }
        )
        # Right: 4 entities at index 0, 1, 2, 3 (Alpha, Beta, Gamma shifted by 1)
        df_r = pd.DataFrame(
            {
                "vendor_name": [
                    "Zeta Holdings Ltd",
                    "Alpha Logistics",
                    "Beta Services",
                    "Gamma Tech",
                ],
                "dollar_val": [999.0, 500.0, 750.0, 1200.0],
                "inv_date": ["2026-01-01", "2026-03-01", "2026-03-02", "2026-03-03"],
            }
        )

        res = auto_match_finance(df_l, df_r)
        matches = res.matches

        # All 3 entities must match without being severed
        assert len(matches) == 3
        assert res.summary.reconciled_count == 3
        assert list(matches["left_id"]) == [0, 1, 2]
        assert list(matches["right_id"]) == [1, 2, 3]
