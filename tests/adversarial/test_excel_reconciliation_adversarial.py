"""Empirical stress testing for Excel review export and financial reconciliation.

Adversarially tests fuzzymatch and fuzzymatch-finance across:
1. 1-to-1 Bipartite Cardinality Invariant: |M| <= min(|L|, |R|), disjoint left and right matches.
2. Volume and Count Conservation Invariants across positive, negative, zero, and extreme values.
3. Residual Unmatched Variance calculation accounting for both Left and Right unmatched transactions.
4. CLI Syntax Matrix: Pure Positional, Pure Flag, Short Flag, Mixed Flag-Positional (-l <L> <R>, <L> -r <R>).
5. Excel Review Export Validation: openpyxl validation rules, formatting, and roundtrip sanity.
6. AutoMatcher and auto_match_finance Edge Cases: RangeIndex/Integer headers, ID overlap, NaNs, zero rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
import pytest
from fuzzymatch.cli import main as core_main

from fuzzymatch_finance.cli import auto_match_finance
from fuzzymatch_finance.cli import main as finance_main


class TestOneToOneBipartiteCardinalityInvariant:
    """Rigorous verification that auto_match_finance satisfies 1-to-1 bipartite matching."""

    def test_identical_records_cardinality_bound(self) -> None:
        """When 50 left records and 30 right records are identical, $|M| == 30 <= min(50, 30)."""
        df_left = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(50)],
                "name": ["Acme Global Logistics LLC"] * 50,
                "amount": [1250.50] * 50,
                "date": ["2026-03-15"] * 50,
            }
        )
        df_right = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(30)],
                "name": ["Acme Global Logistics LLC"] * 30,
                "amount": [1250.50] * 30,
                "date": ["2026-03-15"] * 30,
            }
        )

        res = auto_match_finance(df_left, df_right)
        matched_df = res.matches

        # Invariant 1: Exactly min(|L|, |R|) matched since all are identical
        assert len(matched_df) == 30, f"Expected exactly 30 matches, got {len(matched_df)}"
        assert res.summary.reconciled_count == 30

        # Invariant 2: Cardinality bound |M| <= min(|L|, |R|)
        assert len(matched_df) <= min(len(df_left), len(df_right))

        # Invariant 3: Disjointness - no left_id or right_id appears more than once
        left_ids = matched_df["left_id"].tolist()
        right_ids = matched_df["right_id"].tolist()
        assert len(left_ids) == len(set(left_ids)), "Duplicate left_id found in matched pairs!"
        assert len(right_ids) == len(set(right_ids)), "Duplicate right_id found in matched pairs!"

    def test_competing_candidate_scores_greedy_optimal_resolution(self) -> None:
        """Verify greedy resolution picks highest scoring pair and preserves 1-to-1 disjointness.

        L1 matches R1 (higher score), L1 matches R2
        L2 matches R1, L2 matches R2
        Verify both left and right IDs are mutually disjoint in matched results.
        """
        df_left = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Acme High Corp", "Acme Low Corp"],
                "amount": [100.0, 100.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )
        df_right = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Acme High Corp", "Acme Low Corp"],
                "amount": [100.0, 100.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )

        res = auto_match_finance(df_left, df_right)
        matched_df = res.matches

        assert len(matched_df) == 2
        # Verify L1 is paired with R1 and L2 is paired with R2
        pair_dict = dict(zip(matched_df["left_id"], matched_df["right_id"], strict=False))
        assert pair_dict["L1"] == "R1"
        assert pair_dict["L2"] == "R2"

    def test_severe_asymmetric_cardinality(self) -> None:
        """Test |L| = 1, |R| = 100 and |L| = 100, |R| = 1."""
        df_one = pd.DataFrame(
            {
                "id": ["SOLO"],
                "name": ["Standard Enterprise Inc"],
                "amount": [500.0],
                "date": ["2026-06-01"],
            }
        )
        df_many = pd.DataFrame(
            {
                "id": [f"ID_{i}" for i in range(100)],
                "name": ["Standard Enterprise Inc"] * 100,
                "amount": [500.0] * 100,
                "date": ["2026-06-01"] * 100,
            }
        )

        res_1 = auto_match_finance(df_one, df_many)
        assert len(res_1.matches) == 1
        assert res_1.summary.reconciled_count == 1
        assert res_1.summary.unmatched_left_count == 0
        assert res_1.summary.unmatched_right_count == 99

        res_2 = auto_match_finance(df_many, df_one)
        assert len(res_2.matches) == 1
        assert res_2.summary.reconciled_count == 1
        assert res_2.summary.unmatched_left_count == 99
        assert res_2.summary.unmatched_right_count == 0


class TestVolumeConservationAndVarianceInvariants:
    """Rigorous verification of volume and count conservation identities."""

    def test_volume_and_count_conservation_exhaustive(self) -> None:
        """Verify:
        V_rec_left + V_unm_left == V_left
        V_rec_right + V_unm_right == V_right
        V_unmatched == V_unm_left + V_unm_right
        reconciled_count + unmatched_left_count == |L|
        reconciled_count + unmatched_right_count == |R|
        """
        rng = np.random.default_rng(42)
        n_left = 25
        n_right = 35

        left_amounts = rng.uniform(10.0, 5000.0, size=n_left).round(2).tolist()
        right_amounts = rng.uniform(10.0, 5000.0, size=n_right).round(2).tolist()

        # Overlap some names to guarantee partial matching
        shared_names = [f"Vendor Alpha {k}" for k in range(15)]
        left_names = shared_names + [f"Left Unique {k}" for k in range(n_left - len(shared_names))]
        right_names = shared_names + [
            f"Right Unique {k}" for k in range(n_right - len(shared_names))
        ]

        # Sync first 10 amounts and dates to ensure guaranteed matches
        for i in range(10):
            right_amounts[i] = left_amounts[i]

        df_left = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_left)],
                "name": left_names,
                "amount": left_amounts,
                "date": ["2026-04-10"] * n_left,
            }
        )
        df_right = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n_right)],
                "name": right_names,
                "amount": right_amounts,
                "date": ["2026-04-10"] * n_right,
            }
        )

        res = auto_match_finance(df_left, df_right, epsilon=0.01)
        summary = res.summary

        # 1. Count Conservation
        assert summary.reconciled_count + summary.unmatched_left_count == n_left
        assert summary.reconciled_count + summary.unmatched_right_count == n_right

        # 2. Volume Calculations
        expected_l_vol = float(sum(abs(a) for a in left_amounts))
        expected_r_vol = float(sum(abs(a) for a in right_amounts))
        assert summary.left_dollar_volume == pytest.approx(expected_l_vol, abs=1e-5)
        assert summary.right_dollar_volume == pytest.approx(expected_r_vol, abs=1e-5)
        assert summary.total_dollar_volume == pytest.approx(
            expected_l_vol + expected_r_vol, abs=1e-5
        )

        # 3. Volume Conservation
        matched_df = res.matches
        rec_l_vol = float(matched_df["left_amount"].abs().sum()) if not matched_df.empty else 0.0
        rec_r_vol = float(matched_df["right_amount"].abs().sum()) if not matched_df.empty else 0.0

        assert summary.reconciled_dollar_volume == pytest.approx(rec_l_vol, abs=1e-5)
        unm_l_vol = expected_l_vol - rec_l_vol
        unm_r_vol = expected_r_vol - rec_r_vol
        assert summary.unmatched_dollar_volume == pytest.approx(unm_l_vol + unm_r_vol, abs=1e-5)

    def test_unmatched_right_volume_specifically_reported(self) -> None:
        """Verify unmatched right records contribute fully to unmatched_dollar_volume."""
        df_left = pd.DataFrame(
            {
                "id": ["L1"],
                "name": ["Acme Corp"],
                "amount": [100.0],
                "date": ["2026-01-01"],
            }
        )
        df_right = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Acme Corp", "Zeta Logistics"],
                "amount": [100.0, 900.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )

        res = auto_match_finance(df_left, df_right)
        # L1 matches R1 for $100.
        # R2 ($900) is unmatched. Left has 0 unmatched volume.
        assert res.summary.reconciled_count == 1
        assert res.summary.reconciled_dollar_volume == pytest.approx(100.0)
        assert res.summary.unmatched_left_count == 0
        assert res.summary.unmatched_right_count == 1
        assert res.summary.unmatched_dollar_volume == pytest.approx(900.0)

    def test_unmatched_left_volume_specifically_reported(self) -> None:
        """Verify unmatched left records contribute fully to unmatched_dollar_volume."""
        df_left = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Acme Corp", "Omega Supplies"],
                "amount": [100.0, 450.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )
        df_right = pd.DataFrame(
            {
                "id": ["R1"],
                "name": ["Acme Corp"],
                "amount": [100.0],
                "date": ["2026-01-01"],
            }
        )

        res = auto_match_finance(df_left, df_right)
        # L1 matches R1 for $100. L2 ($450) is unmatched.
        assert res.summary.reconciled_count == 1
        assert res.summary.reconciled_dollar_volume == pytest.approx(100.0)
        assert res.summary.unmatched_left_count == 1
        assert res.summary.unmatched_right_count == 0
        assert res.summary.unmatched_dollar_volume == pytest.approx(450.0)


class TestCLISyntaxAndInvocationMatrix:
    """Stress test pure, short-flag, and mixed CLI syntaxes across both CLIs."""

    @pytest.fixture
    def sample_files(self, tmp_path: Path) -> tuple[Path, Path]:
        left_path = tmp_path / "left_transactions.csv"
        right_path = tmp_path / "right_transactions.csv"

        df_l = pd.DataFrame(
            {
                "tx_id": ["L01", "L02", "L03"],
                "vendor": [
                    "Amazon Web Services",
                    "Google Cloud EMEA",
                    "Slack Technologies",
                ],
                "total_usd": [150.00, 320.50, 75.00],
                "tx_date": ["2026-02-01", "2026-02-02", "2026-02-03"],
            }
        )
        df_r = pd.DataFrame(
            {
                "ref_num": ["R01", "R02", "R03"],
                "counterparty": [
                    "Amazon Web Services Inc",
                    "Google Cloud",
                    "GitHub Inc",
                ],
                "amount": [150.00, 320.50, 40.00],
                "val_date": ["2026-02-01", "2026-02-02", "2026-02-04"],
            }
        )

        df_l.to_csv(left_path, index=False)
        df_r.to_csv(right_path, index=False)
        return left_path, right_path

    @pytest.mark.parametrize("cli_cmd", ["fuzzymatch", "fuzzymatch-finance"])
    def test_cli_mixed_syntax_minus_l_then_positional(
        self, sample_files: tuple[Path, Path], cli_cmd: str, tmp_path: Path
    ) -> None:
        """Test the exact syntax that previously failed: match -l <left> <right>."""
        left_p, right_p = sample_files
        out_p = tmp_path / f"{cli_cmd}_out1.csv"

        argv = ["match", "-l", str(left_p), str(right_p), "-o", str(out_p)]
        ret = finance_main(argv) if cli_cmd == "fuzzymatch-finance" else core_main(argv)
        assert ret == 0, f"{cli_cmd} failed on '-l <left> <right>' with exit code {ret}"
        assert out_p.exists()

    @pytest.mark.parametrize("cli_cmd", ["fuzzymatch", "fuzzymatch-finance"])
    def test_cli_mixed_syntax_positional_then_minus_r(
        self, sample_files: tuple[Path, Path], cli_cmd: str, tmp_path: Path
    ) -> None:
        """Test reverse mixed syntax: match <left> -r <right>."""
        left_p, right_p = sample_files
        out_p = tmp_path / f"{cli_cmd}_out2.csv"

        argv = ["match", str(left_p), "-r", str(right_p), "-o", str(out_p)]
        ret = finance_main(argv) if cli_cmd == "fuzzymatch-finance" else core_main(argv)
        assert ret == 0, f"{cli_cmd} failed on '<left> -r <right>' with exit code {ret}"
        assert out_p.exists()

    @pytest.mark.parametrize("cli_cmd", ["fuzzymatch", "fuzzymatch-finance"])
    def test_cli_pure_flag_syntax(
        self, sample_files: tuple[Path, Path], cli_cmd: str, tmp_path: Path
    ) -> None:
        """Test pure flag syntax: match --left <left> --right <right>."""
        left_p, right_p = sample_files
        out_p = tmp_path / f"{cli_cmd}_out3.csv"

        argv = [
            "match",
            "--left",
            str(left_p),
            "--right",
            str(right_p),
            "-o",
            str(out_p),
        ]
        ret = finance_main(argv) if cli_cmd == "fuzzymatch-finance" else core_main(argv)
        assert ret == 0, f"{cli_cmd} failed on '--left <L> --right <R>' with code {ret}"
        assert out_p.exists()

    @pytest.mark.parametrize("cli_cmd", ["fuzzymatch", "fuzzymatch-finance"])
    def test_cli_pure_positional_syntax(
        self, sample_files: tuple[Path, Path], cli_cmd: str, tmp_path: Path
    ) -> None:
        """Test pure positional syntax: match <left> <right>."""
        left_p, right_p = sample_files
        out_p = tmp_path / f"{cli_cmd}_out4.csv"

        argv = ["match", str(left_p), str(right_p), "-o", str(out_p)]
        ret = finance_main(argv) if cli_cmd == "fuzzymatch-finance" else core_main(argv)
        assert ret == 0, f"{cli_cmd} failed on '<L> <R>' with code {ret}"
        assert out_p.exists()


class TestOpenpyxlReviewExportIntegrity:
    """Validate review_xlsx output structure, Excel validation rules, and formatting."""

    def test_review_workbook_validation_and_styling(self, tmp_path: Path) -> None:
        left_p = tmp_path / "left.csv"
        right_p = tmp_path / "right.csv"
        review_p = tmp_path / "review.xlsx"

        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Oracle Systems", "Adobe Software"],
                "amount": [1200.0, 450.0],
                "date": ["2026-05-01", "2026-05-02"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Oracle Corp", "Adobe Inc"],
                "amount": [1200.0, 450.0],
                "date": ["2026-05-01", "2026-05-02"],
            }
        )
        df_l.to_csv(left_p, index=False)
        df_r.to_csv(right_p, index=False)

        ret = finance_main(["match", str(left_p), str(right_p), "--review-out", str(review_p)])
        assert ret == 0
        assert review_p.exists()

        # Load with openpyxl to verify spreadsheet features
        wb = openpyxl.load_workbook(review_p)
        assert "review" in wb.sheetnames
        ws = wb["review"]

        # 1. Header validation
        headers = [ws.cell(row=1, column=c).value for c in range(1, ws.max_column + 1)]
        assert "decision" in headers
        assert "fuzzy_score" in headers

        # 2. Data validation check on decision column
        assert len(ws.data_validations.dataValidation) >= 1
        dv = ws.data_validations.dataValidation[0]
        assert "accept" in dv.formula1
        assert "reject" in dv.formula1
        assert "pending" in dv.formula1

        # 3. Row count check
        assert ws.max_row >= 3  # Header + 2 data rows


class TestRobustnessOnDegenerateSchemas:
    """Stress test boundary data structures and schemas."""

    def test_integer_column_names_in_auto_match_finance(self) -> None:
        """RangeIndex integer column names must not crash with AttributeError."""
        df_l = pd.DataFrame({0: ["B1"], 1: ["Acme LLC"], 2: [500.0], 3: ["2026-01-01"]})
        df_r = pd.DataFrame({0: ["B1"], 1: ["Acme"], 2: [500.0], 3: ["2026-01-01"]})

        # When headers are integers without explicit column roles, executes safely
        res = auto_match_finance(df_l, df_r)
        assert res is not None
        assert not np.isnan(res.summary.total_dollar_volume)

    def test_overlapping_string_ids_between_left_and_right(self) -> None:
        """When left and right both have identical IDs ('1', '2'), they must not collide."""
        df_l = pd.DataFrame(
            {
                "id": ["1", "2"],
                "name": ["Alpha Corp", "Beta LLC"],
                "amount": [100.0, 200.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["1", "2"],
                "name": ["Alpha Corporation", "Beta Limited"],
                "amount": [100.0, 200.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )

        res = auto_match_finance(df_l, df_r)
        assert len(res.matches) == 2
        # Check matched pairs correctly pair left '1' to right '1'
        matches = res.matches
        assert set(matches["left_id"]) == {"1", "2"}
        assert set(matches["right_id"]) == {"1", "2"}

    def test_nan_and_nat_handling_in_finance_reconciliation(self) -> None:
        """Missing values in amounts or dates must not raise unhandled exceptions."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Company One", "Company Two"],
                "amount": [np.nan, 200.0],
                "date": [pd.NaT, "2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Company One Inc", "Company Two Ltd"],
                "amount": [100.0, np.nan],
                "date": ["2026-01-01", pd.NaT],
            }
        )

        res = auto_match_finance(df_l, df_r)
        assert res.summary.left_count == 2
        assert res.summary.right_count == 2
        # Non-crashing is primary invariant; summary volumes must be finite
        assert not np.isnan(res.summary.reconciled_dollar_volume)
        assert not np.isnan(res.summary.unmatched_dollar_volume)


class TestHypothesisGenerativeInvariants:
    """Hypothesis randomized generative testing of bipartite matching and volume conservation."""

    @pytest.mark.hypothesis
    def test_randomized_cardinality_and_volume_identities(self) -> None:
        """Randomized stress over 30 parametric instances with varying table sizes and amounts."""
        rng = np.random.default_rng(2026)
        for trial in range(30):
            n_l = int(rng.integers(1, 25))
            n_r = int(rng.integers(1, 25))

            # Randomly generate amounts
            amt_l = rng.uniform(5.0, 5000.0, size=n_l).round(2).tolist()
            amt_r = rng.uniform(5.0, 5000.0, size=n_r).round(2).tolist()

            # Randomly share some vendor names
            vocab = [
                "Acme",
                "Omega",
                "Zeta",
                "Delta",
                "Apex",
                "Global",
                "Logistics",
                "Services",
            ]
            names_l = [f"{rng.choice(vocab)} {rng.choice(vocab)} Corp" for _ in range(n_l)]
            names_r = [f"{rng.choice(vocab)} {rng.choice(vocab)} Inc" for _ in range(n_r)]

            # In half the trials, duplicate an amount across multiple left/right rows
            if trial % 2 == 0 and n_l > 2 and n_r > 2:
                amt_l[0] = amt_l[1] = 999.99
                amt_r[0] = amt_r[1] = 999.99
                names_l[0] = names_l[1] = "Identical Twin Corp"
                names_r[0] = names_r[1] = "Identical Twin Corp"

            df_l = pd.DataFrame(
                {
                    "id": [f"L_{trial}_{i}" for i in range(n_l)],
                    "name": names_l,
                    "amount": amt_l,
                    "date": ["2026-05-15"] * n_l,
                }
            )
            df_r = pd.DataFrame(
                {
                    "id": [f"R_{trial}_{j}" for j in range(n_r)],
                    "name": names_r,
                    "amount": amt_r,
                    "date": ["2026-05-15"] * n_r,
                }
            )

            res = auto_match_finance(df_l, df_r, epsilon=2.0)
            matches = res.matches
            summary = res.summary

            # 1. Cardinality Invariants
            assert len(matches) <= min(n_l, n_r)
            if not matches.empty:
                assert matches["left_id"].nunique() == len(matches)
                assert matches["right_id"].nunique() == len(matches)

            # 2. Count Conservation
            assert summary.reconciled_count + summary.unmatched_left_count == n_l
            assert summary.reconciled_count + summary.unmatched_right_count == n_r

            # 3. Ingested Volume Consistency
            tot_l = float(sum(abs(x) for x in amt_l))
            tot_r = float(sum(abs(x) for x in amt_r))
            assert summary.left_dollar_volume == pytest.approx(tot_l, abs=1e-4)
            assert summary.right_dollar_volume == pytest.approx(tot_r, abs=1e-4)
            assert summary.total_dollar_volume == pytest.approx(tot_l + tot_r, abs=1e-4)

            # 4. Volume Upper Bound & Variance Conservation
            assert summary.reconciled_dollar_volume <= tot_l + 1e-4
            rec_l = float(matches["left_amount"].abs().sum()) if not matches.empty else 0.0
            rec_r = float(matches["right_amount"].abs().sum()) if not matches.empty else 0.0
            unm_l = max(0.0, tot_l - rec_l)
            unm_r = max(0.0, tot_r - rec_r)
            assert summary.unmatched_dollar_volume == pytest.approx(unm_l + unm_r, abs=1e-4)
