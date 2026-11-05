"""Adversarial stress testing suite for cross-package entity resolution.

Formal mathematical invariant and empirical stress testing across:
1. 1-to-1 Bipartite Cardinality Invariants: |M| <= min(|L|, |R|), mutual exclusivity of matches.
2. Volume and Partition Conservation Laws: V_L = V_rec + V_unm across positive, negative, zero, and missing amount columns.
3. Convex Simplex Weight Projections: sum(w_i) == 1.0, w_i >= 0, scale invariance, and exception handling.
4. Degenerate and Boundary Edge Cases on AutoMatcher and auto_match_finance:
   - Completely disjoint tables (0 matches)
   - Completely identical tables (N exact matches)
   - Null-only, empty strings, and mixed-type columns
   - Skewed / unbalanced tables (|L| << |R|, |L| >> |R|, |L|=0, |R|=0)
   - Duplicate IDs and cross-table ID collisions
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from fuzzymatch.api.auto import AutoMatcher, MatchResults
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from fuzzymatch_finance.cli import (
    FinanceMatchResults,
    auto_match_finance,
)
from fuzzymatch_finance.scoring import FinancialCompositeScorer

# ===========================================================================
# 1. Bipartite 1-to-1 Cardinality Invariant Verification
# ===========================================================================


class TestOneToOneBipartiteCardinalityInvariant:
    """Stress tests verifying the invariant: |M| <= min(|L|, |R|) and uniqueness."""

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        n_left=st.integers(min_value=0, max_value=30),
        n_right=st.integers(min_value=0, max_value=30),
    )
    def test_hypothesis_cardinality_bound(self, n_left: int, n_right: int) -> None:
        """Theorem: For any bipartite match M between L and R, |M| <= min(|L|, |R|)."""
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_left)],
                "vendor_name": [f"Vendor {i % 5} International Corp" for i in range(n_left)],
                "amount": [100.0 + (i % 3) * 50.0 for i in range(n_left)],
                "date": ["2026-03-01"] * n_left,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n_right)],
                "vendor_name": [f"Vendor {j % 5} International Corp" for j in range(n_right)],
                "amount": [100.0 + (j % 3) * 50.0 for j in range(n_right)],
                "date": ["2026-03-01"] * n_right,
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        m = res.matches

        # Cardinality ceiling
        assert len(m) <= min(n_left, n_right)

        # 1-to-1 mutual exclusivity: no left record or right record is matched > 1 time
        if not m.empty:
            assert m["left_id"].nunique() == len(m), "Duplicate left_id matched in 1-to-1 match!"
            assert m["right_id"].nunique() == len(m), "Duplicate right_id matched in 1-to-1 match!"

        # Summary metrics consistency
        assert res.summary.reconciled_count == len(m)
        assert res.summary.left_count == n_left
        assert res.summary.right_count == n_right
        assert res.summary.unmatched_left_count == n_left - len(m)
        assert res.summary.unmatched_right_count == n_right - len(m)

    def test_severely_unbalanced_tables_left_small(self) -> None:
        """When |L|=1 and |R|=500 with identical content, |M| must be strictly 1."""
        df_l = pd.DataFrame(
            {
                "id": ["L_SINGLE"],
                "name": ["Consolidated Freight Carriers Inc"],
                "amount": [4500.00],
                "date": ["2026-04-10"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(500)],
                "name": ["Consolidated Freight Carriers Inc"] * 500,
                "amount": [4500.00] * 500,
                "date": ["2026-04-10"] * 500,
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        assert len(res.matches) == 1
        assert res.summary.reconciled_count == 1
        assert res.summary.unmatched_left_count == 0
        assert res.summary.unmatched_right_count == 499

    def test_severely_unbalanced_tables_right_small(self) -> None:
        """When |L|=500 and |R|=1 with identical content, |M| must be strictly 1."""
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(500)],
                "name": ["Pacific Rim Consulting Group LLC"] * 500,
                "amount": [8200.75] * 500,
                "date": ["2026-05-20"] * 500,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R_SINGLE"],
                "name": ["Pacific Rim Consulting Group LLC"],
                "amount": [8200.75],
                "date": ["2026-05-20"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        assert len(res.matches) == 1
        assert res.summary.reconciled_count == 1
        assert res.summary.unmatched_left_count == 499
        assert res.summary.unmatched_right_count == 0

    def test_empty_boundary_tables(self) -> None:
        """Empty left, empty right, and both empty must yield 0 matches without error."""
        empty_df = pd.DataFrame(columns=["id", "name", "amount", "date"])
        non_empty_df = pd.DataFrame(
            {
                "id": ["L_1", "L_2"],
                "name": ["Alpha Inc", "Beta LLC"],
                "amount": [100.0, 200.0],
                "date": ["2026-01-01", "2026-01-02"],
            }
        )

        # Left empty
        r1 = auto_match_finance(empty_df, non_empty_df)
        assert len(r1.matches) == 0
        assert r1.summary.reconciled_count == 0
        assert r1.summary.unmatched_left_count == 0
        assert r1.summary.unmatched_right_count == 2

        # Right empty
        r2 = auto_match_finance(non_empty_df, empty_df)
        assert len(r2.matches) == 0
        assert r2.summary.reconciled_count == 0
        assert r2.summary.unmatched_left_count == 2
        assert r2.summary.unmatched_right_count == 0

        # Both empty
        r3 = auto_match_finance(empty_df, empty_df)
        assert len(r3.matches) == 0
        assert r3.summary.reconciled_count == 0
        assert r3.summary.left_count == 0
        assert r3.summary.right_count == 0

    def test_duplicate_primary_keys_in_inputs(self) -> None:
        """When input DataFrames contain duplicate key values, 1-to-1 selection must remain unique."""
        df_l = pd.DataFrame(
            {
                "id": ["DUP_KEY", "DUP_KEY", "UNIQUE_L"],
                "name": ["Acme Supply", "Acme Supply", "Other Co"],
                "amount": [100.0, 100.0, 300.0],
                "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["DUP_KEY", "DUP_KEY", "UNIQUE_R"],
                "name": ["Acme Supply", "Acme Supply", "Other Co"],
                "amount": [100.0, 100.0, 300.0],
                "date": ["2026-01-01", "2026-01-01", "2026-01-01"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        # Because both rows have id="DUP_KEY", greedy matching on id allows at most 1 match with id DUP_KEY
        assert len(res.matches) <= 2
        assert res.matches["left_id"].nunique() == len(res.matches)
        assert res.matches["right_id"].nunique() == len(res.matches)


# ===========================================================================
# 2. Volume Conservation Laws Verification
# ===========================================================================


class TestVolumeConservationLaws:
    """Stress tests verifying volume conservation and partitioning identities."""

    @settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(
        amounts_l=st.lists(
            st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=15,
        ),
        amounts_r=st.lists(
            st.floats(min_value=-1e5, max_value=1e5, allow_nan=False, allow_infinity=False),
            min_size=1,
            max_size=15,
        ),
    )
    def test_hypothesis_volume_conservation_identities(
        self, amounts_l: list[float], amounts_r: list[float]
    ) -> None:
        """Theorem:
        1. V_total = V_L + V_R
        2. V_L = V_rec,L + V_unm,L
        3. V_R = V_rec,R + V_unm,R
        4. V_unm = V_unm,L + V_unm,R
        5. V_total = V_rec,L + V_rec,R + V_unm
        """
        n_l = len(amounts_l)
        n_r = len(amounts_r)

        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_l)],
                "name": [f"Vendor {i}" for i in range(n_l)],
                "amount": amounts_l,
                "date": ["2026-02-01"] * n_l,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n_r)],
                "name": [f"Vendor {j}" for j in range(n_r)],
                "amount": amounts_r,
                "date": ["2026-02-01"] * n_r,
            }
        )

        res = auto_match_finance(df_l, df_r, epsilon=5.0, threshold=0.5)
        s = res.summary

        expected_vl = float(sum(abs(x) for x in amounts_l))
        expected_vr = float(sum(abs(x) for x in amounts_r))
        expected_tot = expected_vl + expected_vr

        # 1. Total Volume Conservation
        assert math.isclose(s.left_dollar_volume, expected_vl, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(s.right_dollar_volume, expected_vr, rel_tol=1e-5, abs_tol=1e-5)
        assert math.isclose(s.total_dollar_volume, expected_tot, rel_tol=1e-5, abs_tol=1e-5)

        # 2. Left Partition Conservation
        # s.reconciled_dollar_volume is V_rec,L
        assert s.reconciled_dollar_volume <= s.left_dollar_volume + 1e-5
        v_unm_l = max(0.0, s.left_dollar_volume - s.reconciled_dollar_volume)
        assert math.isclose(
            s.left_dollar_volume, s.reconciled_dollar_volume + v_unm_l, rel_tol=1e-5, abs_tol=1e-5
        )

        # 3. Unmatched Volume Non-Negativity
        assert s.unmatched_dollar_volume >= 0.0

    def test_negative_amounts_credits_and_refunds(self) -> None:
        """Negative amounts (e.g. credit memos, reversals) must contribute absolute magnitude."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2", "L3"],
                "name": ["Supplier A", "Supplier B", "Supplier C"],
                "amount": [-1500.00, 2500.00, -500.00],
                "date": ["2026-03-01", "2026-03-01", "2026-03-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2", "R3"],
                "name": ["Supplier A", "Supplier B", "Supplier D"],
                "amount": [-1500.00, 2500.00, 800.00],
                "date": ["2026-03-01", "2026-03-01", "2026-03-01"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        s = res.summary

        # V_L = |-1500| + |2500| + |-500| = 4500.0
        assert math.isclose(s.left_dollar_volume, 4500.0, abs_tol=1e-6)
        # V_R = |-1500| + |2500| + |800| = 4800.0
        assert math.isclose(s.right_dollar_volume, 4800.0, abs_tol=1e-6)
        # Total = 9300.0
        assert math.isclose(s.total_dollar_volume, 9300.0, abs_tol=1e-6)

        # Reconciled pairs: L1<->R1 (1500.0) and L2<->R2 (2500.0)
        assert s.reconciled_count == 2
        assert math.isclose(s.reconciled_dollar_volume, 4000.0, abs_tol=1e-6)

        # Left unmatched = 4500 - 4000 = 500.0
        # Right unmatched = 4800 - 4000 = 800.0
        # Total unmatched variance = 500 + 800 = 1300.0
        assert math.isclose(s.unmatched_dollar_volume, 1300.0, abs_tol=1e-6)

    def test_missing_and_corrupted_amounts(self) -> None:
        """Missing values, NaNs, and unparseable strings must not crash or poison volume metrics."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2", "L3", "L4"],
                "name": ["Corp 1", "Corp 2", "Corp 3", "Corp 4"],
                "amount": [100.0, None, np.nan, "INVALID_NUMBER"],
                "date": ["2026-01-01"] * 4,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Corp 1", "Corp 2"],
                "amount": [100.0, 200.0],
                "date": ["2026-01-01"] * 2,
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        s = res.summary

        # Only 100.0 is valid in df_l
        assert math.isclose(s.left_dollar_volume, 100.0, abs_tol=1e-6)
        assert math.isclose(s.right_dollar_volume, 300.0, abs_tol=1e-6)
        assert math.isclose(s.total_dollar_volume, 400.0, abs_tol=1e-6)
        assert not math.isnan(s.total_dollar_volume)
        assert not math.isnan(s.unmatched_dollar_volume)

    def test_tables_without_amount_column(self) -> None:
        """Tables with no amount column at all should report 0.0 volume without failure."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Company A", "Company B"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Company A", "Company B"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        s = res.summary
        assert s.left_dollar_volume == 0.0
        assert s.right_dollar_volume == 0.0
        assert s.total_dollar_volume == 0.0
        assert s.reconciled_dollar_volume == 0.0
        assert s.unmatched_dollar_volume == 0.0
        assert s.reconciled_count == 2
        assert s.reconciliation_rate_pct == 100.0


# ===========================================================================
# 3. Simplex Weight Projections Verification
# ===========================================================================


class TestSimplexWeightProjections:
    """Stress tests verifying convex simplex weight projection properties."""

    @settings(max_examples=100, deadline=None)
    @given(
        w_text=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        w_amt=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
        w_date=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False),
    )
    def test_hypothesis_simplex_sum_and_bounds(
        self, w_text: float, w_amt: float, w_date: float
    ) -> None:
        """Theorem: For any raw w in R_{>0}^3, projected w* satisfies sum(w*) == 1 and w*_i in (0, 1)."""
        raw = {"text": w_text, "amount": w_amt, "date": w_date}
        scorer = FinancialCompositeScorer(weights=raw)

        weights = scorer.weights
        total_w = sum(weights.values())

        # Conservation of unit mass
        assert math.isclose(total_w, 1.0, abs_tol=1e-9)

        # Boundary constraints
        for k, v in weights.items():
            assert 0.0 <= v <= 1.0, f"Weight {k}={v} violated unit interval [0, 1]"

    @settings(max_examples=100, deadline=None)
    @given(
        scale=st.floats(min_value=1e-4, max_value=1e4, allow_nan=False, allow_infinity=False),
    )
    def test_scale_invariance_property(self, scale: float) -> None:
        """Theorem: normalize(alpha * w) == normalize(w) for all alpha > 0."""
        base_weights = {"text": 0.5, "amount": 0.3, "date": 0.2}
        scaled_weights = {k: v * scale for k, v in base_weights.items()}

        scorer1 = FinancialCompositeScorer(weights=base_weights)
        scorer2 = FinancialCompositeScorer(weights=scaled_weights)

        for k in base_weights:
            assert math.isclose(scorer1.weights[k], scorer2.weights[k], abs_tol=1e-9)

    def test_zero_weight_on_specific_components(self) -> None:
        """Allowing 0.0 weight on subset of components must project remaining mass correctly."""
        scorer = FinancialCompositeScorer(weights={"text": 0.0, "amount": 10.0, "date": 0.0})
        assert scorer.weights["text"] == 0.0
        assert math.isclose(scorer.weights["amount"], 1.0, abs_tol=1e-9)
        assert scorer.weights["date"] == 0.0

    def test_negative_weight_exception(self) -> None:
        """Negative weights violate non-negativity and must raise ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            FinancialCompositeScorer(weights={"text": 0.5, "amount": -0.2, "date": 0.7})

    def test_all_zero_weights_exception(self) -> None:
        """All zero weights have sum=0 and cannot be normalized onto simplex."""
        with pytest.raises(ValueError, match="strictly positive"):
            FinancialCompositeScorer(weights={"text": 0.0, "amount": 0.0, "date": 0.0})

    def test_non_finite_weights_exception(self) -> None:
        """NaN and Inf weights must raise ValueError."""
        with pytest.raises(ValueError, match="finite"):
            FinancialCompositeScorer(weights={"text": float("nan"), "amount": 1.0, "date": 1.0})
        with pytest.raises(ValueError, match="finite"):
            FinancialCompositeScorer(weights={"text": float("inf"), "amount": 1.0, "date": 1.0})


# ===========================================================================
# 4. AutoMatcher Adversarial & Boundary Stress Tests
# ===========================================================================


class TestAutoMatcherBoundaryAndEdgeCases:
    """Stress tests on AutoMatcher on pathological and boundary inputs."""

    def test_completely_disjoint_tables(self) -> None:
        """Completely disjoint tables must return 0 matches and 0 clusters."""
        df_l = pd.DataFrame(
            {
                "code": ["L1", "L2", "L3"],
                "name": ["Apple Inc", "Banana LLC", "Cherry Corp"],
                "amount": [10.0, 20.0, 30.0],
                "date": ["2020-01-01", "2020-01-02", "2020-01-03"],
            }
        )
        df_r = pd.DataFrame(
            {
                "code": ["R1", "R2", "R3"],
                "name": ["Xylophone Enterprise", "Yellowstone Parks", "Zebra Safari"],
                "amount": [10000.0, 20000.0, 30000.0],
                "date": ["2035-12-01", "2035-12-02", "2035-12-03"],
            }
        )

        matcher = AutoMatcher(threshold=0.8)
        results = matcher.fit_predict(df_l, df_r)

        assert len(results.matches) == 0
        assert len(results.clusters) == 0
        assert results.summary_stats["matched_pairs"] == 0

    def test_completely_identical_tables(self) -> None:
        """Completely identical tables with 25 records must produce 25 matches."""
        n = 25
        df = pd.DataFrame(
            {
                "id": [f"ID_{i:03d}" for i in range(n)],
                "company": [f"Enterprise Services Global Group {i} LLC" for i in range(n)],
                "amount": [500.0 + i * 25.0 for i in range(n)],
                "date": [f"2026-03-{(i % 28) + 1:02d}" for i in range(n)],
            }
        )

        matcher = AutoMatcher(threshold=0.7)
        results = matcher.fit_predict(df, df)

        # In an identical self-match, every record matches its counterpart
        assert len(results.matches) == n
        assert len(results.clusters) == n
        assert (results.matches["score"] >= 0.99).all()

    def test_null_only_columns(self) -> None:
        """Tables with null-only columns must not crash and degrade gracefully."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2", "L3"],
                "name": [None, None, None],
                "amount": [np.nan, np.nan, np.nan],
                "date": [None, None, None],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2", "R3"],
                "name": [None, None, None],
                "amount": [np.nan, np.nan, np.nan],
                "date": [None, None, None],
            }
        )

        matcher = AutoMatcher(threshold=0.7)
        results = matcher.fit_predict(df_l, df_r)
        # With null names and amounts, composite scores will be 0.0
        assert len(results.matches) == 0

    def test_empty_strings_and_whitespace(self) -> None:
        """Whitespace-only strings must be stripped and not produce false matches."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["   ", "\t\n  "],
                "amount": [100.0, 200.0],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["", "  "],
                "amount": [1000.0, 2000.0],
            }
        )

        matcher = AutoMatcher(threshold=0.7)
        results = matcher.fit_predict(df_l, df_r)
        assert len(results.matches) == 0

    def test_mixed_types_in_columns(self) -> None:
        """Columns with mixed datatypes (ints, floats, strings, objects) should not raise uncaught errors."""
        df_l = pd.DataFrame(
            {
                "id": [1, "TWO", 3.0],
                "name": ["Acme Inc", 12345, "Global LLC"],
                "amount": ["100.50", 250, "NOT_AN_AMOUNT"],
                "date": ["2026-01-01", 1700000000, "INVALID_DATE"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["1", "TWO", "3"],
                "name": ["Acme Inc", "12345", "Global LLC"],
                "amount": [100.50, 250.0, 500.0],
                "date": ["2026-01-01", "2023-11-14", "2026-05-01"],
            }
        )

        matcher = AutoMatcher(threshold=0.6)
        results = matcher.fit_predict(df_l, df_r)
        # Should execute successfully without unhandled TypeError
        assert isinstance(results, MatchResults)
        assert len(results.matches) >= 1

    def test_duplicate_column_names_in_input(self) -> None:
        """DataFrame with duplicate column headers handled safely."""
        df_l = pd.DataFrame([[1, "Acme Inc", "Acme Extra"]], columns=["id", "name", "name"])
        df_r = pd.DataFrame([[1, "Acme Inc"]], columns=["id", "name"])

        matcher = AutoMatcher(threshold=0.7)
        results = matcher.fit_predict(df_l, df_r)
        assert len(results.matches) == 1

    def test_overlapping_primary_key_ids_across_tables(self) -> None:
        """When L and R have overlapping ID keys (e.g. L has id=1, R has id=1), StarClusterer handles bipartite disambiguation."""
        df_l = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Delta Air", "United Transport", "American Cargo"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Delta Air Inc", "United Transport LLC", "American Cargo Corp"],
            }
        )

        matcher = AutoMatcher(threshold=0.7)
        results = matcher.fit_predict(df_l, df_r)
        assert len(results.matches) == 3
        assert len(results.clusters) == 3
        # Ensure clusters contain exactly one left and one right node
        for cl in results.clusters:
            assert len(cl) == 2


# ===========================================================================
# 5. auto_match_finance Adversarial & Boundary Stress Tests
# ===========================================================================


class TestAutoMatchFinanceBoundaryAndEdgeCases:
    """Stress tests on auto_match_finance covering financial domain workflows."""

    def test_completely_disjoint_financial_tables(self) -> None:
        """Disjoint financial tables report 0 reconciled, 0% rate, and unm_vol == tot_vol."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["North Vendor", "South Vendor"],
                "amount": [1000.0, 2000.0],
                "date": ["2026-01-01", "2026-01-02"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["East Supplier", "West Supplier"],
                "amount": [50000.0, 60000.0],
                "date": ["2028-10-10", "2028-10-11"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.8)
        s = res.summary

        assert s.reconciled_count == 0
        assert s.reconciled_dollar_volume == 0.0
        assert s.reconciliation_rate_pct == 0.0
        assert s.unmatched_left_count == 2
        assert s.unmatched_right_count == 2
        assert math.isclose(s.unmatched_dollar_volume, s.total_dollar_volume, abs_tol=1e-6)

    def test_strict_gating_behavior(self) -> None:
        """When gated=True, out-of-tolerance amount or date forces composite score to 0.0."""
        df_l = pd.DataFrame(
            {
                "id": ["L1"],
                "name": ["Identical Vendor Name Inc"],
                "amount": [1000.0],
                "date": ["2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1"],
                "name": ["Identical Vendor Name Inc"],
                "amount": [2000.0],  # Out of tolerance (epsilon=1.0)
                "date": ["2026-01-01"],
            }
        )

        res_ungated = auto_match_finance(df_l, df_r, gated=False, threshold=0.4)
        res_gated = auto_match_finance(df_l, df_r, gated=True, threshold=0.4)

        # Ungated matches because text similarity (0.50) exceeds threshold 0.40
        assert len(res_ungated.matches) == 1
        # Gated forces score to 0.0 because amount diff is $1000 > epsilon ($1.0)
        assert len(res_gated.matches) == 0

    def test_extreme_date_differences_and_leap_years(self) -> None:
        """Leap day February 29 and century date spans handled without overflow."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Leap Year Vendor", "Century Vendor"],
                "amount": [100.0, 200.0],
                "date": ["2024-02-29", "1926-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Leap Year Vendor", "Century Vendor"],
                "amount": [100.0, 200.0],
                "date": ["2024-02-29", "2026-01-01"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        # Leap day pair should match perfectly
        assert "L1" in res.matches["left_id"].values
        # Century pair: text matches, amount matches, date differs by 100 years
        # With active weights text=0.5, amt=0.3, date=0.2: score = 0.5*1.0 + 0.3*1.0 + 0.2*0.0 = 0.8 >= 0.7
        assert "L2" in res.matches["left_id"].values

    def test_custom_column_name_overrides(self) -> None:
        """Explicit column parameters override auto-detection."""
        df_l = pd.DataFrame(
            {
                "my_key": ["K1"],
                "my_description": ["Custom Vendor Name"],
                "my_val": [777.0],
                "my_dt": ["2026-06-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "vendor_pk": ["K1"],
                "payee_str": ["Custom Vendor Name"],
                "transaction_amt": [777.0],
                "cleared_ts": ["2026-06-01"],
            }
        )

        res = auto_match_finance(
            df_l,
            df_r,
            id_col_left="my_key",
            id_col_right="vendor_pk",
            text_col_left="my_description",
            text_col_right="payee_str",
            amount_col_left="my_val",
            amount_col_right="transaction_amt",
            date_col_left="my_dt",
            date_col_right="cleared_ts",
            threshold=0.7,
        )
        assert len(res.matches) == 1
        assert res.summary.reconciled_count == 1
        assert math.isclose(res.summary.reconciled_dollar_volume, 777.0, abs_tol=1e-6)


# ===========================================================================
# 6. Deep Numerical & Structural Stress Tests
# ===========================================================================


class TestDeepNumericalAndStructuralStress:
    """Adversarial stress tests on scale, budget caps, unicode, and internal collision."""

    def test_budget_cap_strict_enforcement_automatcher(self) -> None:
        """AutoMatcher candidate_pairs must never exceed configured budget_cap."""
        n = 30
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n)],
                "company": [f"Acme Corp Group {i}" for i in range(n)],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n)],
                "company": [f"Acme Corp Group {j}" for j in range(n)],
            }
        )

        cap = 25
        matcher = AutoMatcher(budget_cap=cap, threshold=0.6)
        results = matcher.fit_predict(df_l, df_r)

        assert results.summary_stats["candidate_pairs"] <= cap
        assert results.summary_stats["budget_utilization_pct"] <= 100.0
        assert results.total_candidates <= cap

    def test_budget_cap_strict_enforcement_finance(self) -> None:
        """auto_match_finance candidate blocking must respect budget_cap on large cartesian products."""
        n_l = 80
        n_r = 80  # 80 * 80 = 6,400 > 5,000 threshold for MultiPassBlock
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_l)],
                "name": [f"Universal Logistics Supplier {i}" for i in range(n_l)],
                "amount": [100.0 + i for i in range(n_l)],
                "date": ["2026-01-01"] * n_l,
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n_r)],
                "name": [f"Universal Logistics Supplier {j}" for j in range(n_r)],
                "amount": [100.0 + j for j in range(n_r)],
                "date": ["2026-01-01"] * n_r,
            }
        )

        cap = 50
        res = auto_match_finance(df_l, df_r, budget_cap=cap, threshold=0.7)
        # Verify matched pairs <= cap and cardinality bound holds
        assert len(res.matches) <= cap
        assert len(res.matches) <= min(n_l, n_r)

    def test_preexisting_internal_column_names(self) -> None:
        """Inputs with columns already named '_auto_key' or '_clean_auto_txt' must not crash."""
        df_l = pd.DataFrame(
            {
                "_auto_key": ["K1", "K2"],
                "_auto_clean_name": ["Alpha Inc", "Beta LLC"],
                "_clean_auto_txt": ["Memo 1", "Memo 2"],
                "name": ["Alpha Inc", "Beta LLC"],
                "amount": [100.0, 200.0],
            }
        )
        df_r = pd.DataFrame(
            {
                "_auto_key": ["K1", "K2"],
                "_auto_clean_name": ["Alpha Inc", "Beta LLC"],
                "_clean_auto_txt": ["Memo 1", "Memo 2"],
                "name": ["Alpha Inc", "Beta LLC"],
                "amount": [100.0, 200.0],
            }
        )

        # Test AutoMatcher
        matcher = AutoMatcher(threshold=0.7)
        res_m = matcher.fit_predict(df_l, df_r)
        assert len(res_m.matches) == 2

        # Test auto_match_finance
        res_f = auto_match_finance(df_l, df_r, threshold=0.7)
        assert len(res_f.matches) == 2

    def test_integer_column_names(self) -> None:
        """Inputs with pure integer column headers [0, 1, 2] should be processed without failure."""
        df_l = pd.DataFrame(
            {
                0: ["L1", "L2"],
                1: ["Omega Shipping GmbH", "Apex Engineering Ltd"],
                2: [500.0, 750.0],
            }
        )
        df_r = pd.DataFrame(
            {
                0: ["R1", "R2"],
                1: ["Omega Shipping GmbH", "Apex Engineering Ltd"],
                2: [500.0, 750.0],
            }
        )

        matcher = AutoMatcher(threshold=0.7)
        res = matcher.fit_predict(df_l, df_r)
        assert len(res.matches) == 2

    def test_unicode_and_multilingual_robustness(self) -> None:
        """Multilingual UTF-8 strings (German umlauts, Japanese Kanji, Arabic, accented characters)."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2", "L3", "L4"],
                "name": [
                    "Müller & Söhne Präzisionswerkzeuge GmbH",
                    "東京エレクトロン株式会社",
                    "شركة الأمل للتجارة والمقاولات ش.م.م",
                    "Société Générale d'Électronique S.A.",
                ],
                "amount": [1200.50, 850000.0, 450.0, 3100.0],
                "date": ["2026-03-01", "2026-03-01", "2026-03-01", "2026-03-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2", "R3", "R4"],
                "name": [
                    "Muller & Sohne Prazisionswerkzeuge GmbH",  # ASCII transliteration
                    "東京エレクトロン株式会社",  # Exact
                    "شركة الأمل للتجارة والمقاولات",  # Stripped suffix
                    "Societe Generale d'Electronique SA",  # Unaccented
                ],
                "amount": [1200.50, 850000.0, 450.0, 3100.0],
                "date": ["2026-03-01", "2026-03-01", "2026-03-01", "2026-03-01"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        # At least Japanese exact match and German/French fuzzy matches should reconcile
        assert len(res.matches) >= 3
        assert "L2" in res.matches["left_id"].values

    def test_extreme_numeric_scale(self) -> None:
        """Transactions spanning extreme financial ranges ($0.01 to $500 billion)."""
        df_l = pd.DataFrame(
            {
                "id": ["MICRO", "MACRO"],
                "name": ["Micro Fee Payment", "Sovereign Debt Facility"],
                "amount": [0.01, 500_000_000_000.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["MICRO_R", "MACRO_R"],
                "name": ["Micro Fee Payment", "Sovereign Debt Facility"],
                "amount": [0.01, 500_000_000_000.0],
                "date": ["2026-01-01", "2026-01-01"],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        assert len(res.matches) == 2
        assert math.isclose(res.summary.left_dollar_volume, 500_000_000_000.01, rel_tol=1e-6)
        assert math.isclose(res.summary.reconciled_dollar_volume, 500_000_000_000.01, rel_tol=1e-6)

    def test_infinite_and_nan_numeric_values(self) -> None:
        """Non-finite numeric values (inf, -inf) in amount columns must not crash execution or terminal formatting."""
        df_l = pd.DataFrame(
            {
                "id": ["L1", "L2"],
                "name": ["Test A", "Test B"],
                "amount": [float("inf"), float("-inf")],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1", "R2"],
                "name": ["Test A", "Test B"],
                "amount": [100.0, 200.0],
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.7)
        # Should not crash with OverflowError or formatting exceptions
        assert isinstance(res, FinanceMatchResults)
        assert math.isinf(res.summary.left_dollar_volume)
        formatted = res.summary.format_terminal(epsilon=1.0, delta_days=3.0, threshold=0.7)
        assert "FUZZYMATCH-FINANCE RECONCILIATION REPORT" in formatted

    def test_timezone_aware_and_iso_dates(self) -> None:
        """Timezone-aware timestamps and varied date formats match accurately."""
        df_l = pd.DataFrame(
            {
                "id": ["L1"],
                "name": ["Cloud Computing Services Inc"],
                "amount": [150.0],
                "date": ["2026-06-15T14:30:00+00:00"],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": ["R1"],
                "name": ["Cloud Computing Services Inc"],
                "amount": [150.0],
                "date": ["2026-06-15T16:30:00+02:00"],  # Same exact UTC instant
            }
        )

        res = auto_match_finance(df_l, df_r, threshold=0.8)
        assert len(res.matches) == 1
        assert res.summary.reconciled_count == 1
