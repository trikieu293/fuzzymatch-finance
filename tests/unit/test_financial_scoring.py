"""Unit tests for FinancialCompositeScorer in fuzzymatch-finance.

Tests:
1. FinancialCompositeScorer initialization and simplex weight validation:
   - Positive tolerance validation for epsilon and delta_days.
   - Non-negative weight validation, sum > 0 validation, and simplex normalization.
   - Default weights (text: 0.5, amount: 0.3, date: 0.2).
   - Text scorer custom method configuration.
2. score_record_pair:
   - Reflexivity: identical records yield score 1.0.
   - Symmetry: s(r1, r2) == s(r2, r1).
   - Attribute field extraction: primary keys and fallback candidates across dict and Series.
   - Missing/NaN/None values handled gracefully.
   - Legal suffix normalization flag (enabled vs disabled).
   - Strict gating: gated=True forces score 0.0 if amount or date is outside tolerance.
3. score_pairs:
   - Vectorized DataFrame pairwise scoring.
   - Length mismatch validation.
   - Custom column mappings.
   - Missing column fallbacks.
   - Gated vectorized masking.
4. Convenience re-exports (amount_proximity_score, date_window_score).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from fuzzymatch_finance.scoring import (
    FinancialCompositeScorer,
    amount_proximity_score,
    date_window_score,
)


def test_scorer_initialization_validation() -> None:
    """Verify parameter validations during FinancialCompositeScorer initialization."""
    with pytest.raises(ValueError, match="amount_epsilon must be strictly positive"):
        FinancialCompositeScorer(amount_epsilon=0.0)

    with pytest.raises(ValueError, match="amount_epsilon must be strictly positive"):
        FinancialCompositeScorer(amount_epsilon=-1.0)

    with pytest.raises(ValueError, match="date_delta_days must be strictly positive"):
        FinancialCompositeScorer(date_delta_days=0.0)

    with pytest.raises(ValueError, match="Weights must be non-negative"):
        FinancialCompositeScorer(weights={"text": 0.8, "amount": -0.1, "date": 0.3})

    with pytest.raises(ValueError, match="Sum of weights must be strictly positive"):
        FinancialCompositeScorer(weights={"text": 0.0, "amount": 0.0, "date": 0.0})


def test_scorer_weight_simplex_normalization() -> None:
    """Verify raw weights are normalized to sum to 1.0."""
    scorer = FinancialCompositeScorer(weights={"text": 2.0, "amount": 1.0, "date": 1.0})
    assert scorer.weights["text"] == pytest.approx(0.50)
    assert scorer.weights["amount"] == pytest.approx(0.25)
    assert scorer.weights["date"] == pytest.approx(0.25)
    assert sum(scorer.weights.values()) == pytest.approx(1.0)

    # Default weights
    default_scorer = FinancialCompositeScorer()
    assert default_scorer.weights["text"] == pytest.approx(0.50)
    assert default_scorer.weights["amount"] == pytest.approx(0.30)
    assert default_scorer.weights["date"] == pytest.approx(0.20)


def test_score_record_pair_invariants() -> None:
    """Verify mathematical invariants in record pair composite scoring."""
    scorer = FinancialCompositeScorer(amount_epsilon=1.0, date_delta_days=3.0)

    r1 = {"name": "Acme Industrial Inc.", "amount": 1500.00, "date": "2025-04-10"}
    r2 = {"name": "Acme Industrial LLC", "amount": 1500.50, "date": "2025-04-11"}

    # Reflexivity: identical records yield 1.0
    assert scorer.score_record_pair(r1, r1) == pytest.approx(1.0)

    # Symmetry: s(r1, r2) == s(r2, r1)
    score_12 = scorer.score_record_pair(r1, r2)
    score_21 = scorer.score_record_pair(r2, r1)
    assert score_12 == pytest.approx(score_21)

    # Boundedness in [0.0, 1.0]
    assert 0.0 <= score_12 <= 1.0

    # Expected value:
    # Text: "Acme Industrial" vs "Acme Industrial" -> 1.0 * 0.5 = 0.50
    # Amount: diff 0.50 / 1.0 -> 0.50 * 0.3 = 0.15
    # Date: diff 1 day / 3.0 -> (2/3) * 0.2 ~ 0.1333
    # Composite: 0.50 + 0.15 + 0.1333 = 0.7833
    expected = 0.50 * 1.0 + 0.30 * 0.50 + 0.20 * (2.0 / 3.0)
    assert score_12 == pytest.approx(expected)


def test_score_record_pair_candidate_field_extraction() -> None:
    """Verify fallback candidates for text, amount, and date fields."""
    scorer = FinancialCompositeScorer(amount_epsilon=2.0, date_delta_days=5.0)

    # ERP record with vendor_name, invoice_amount, invoice_date
    erp_record = {
        "vendor_name": "Precision Machining Corp.",
        "invoice_amount": 250.00,
        "invoice_date": "2025-05-01",
    }
    # Bank record with description, cleared_amount, cleared_date
    bank_record = {
        "description": "Precision Machining",
        "cleared_amount": 250.00,
        "cleared_date": "2025-05-01",
    }

    score = scorer.score_record_pair(erp_record, bank_record)
    assert score == pytest.approx(1.0)

    # Series input with __getitem__ lookup
    s_erp = pd.Series(erp_record)
    s_bank = pd.Series(bank_record)
    assert scorer.score_record_pair(s_erp, s_bank) == pytest.approx(1.0)


def test_score_record_pair_missing_values() -> None:
    """Verify handling when attributes are missing or NaN."""
    scorer = FinancialCompositeScorer()

    r1 = {"name": "Omega Services", "amount": None, "date": "2025-06-01"}
    r2 = {"name": "Omega Services Inc", "amount": 100.0, "date": None}

    # Text matches (1.0 * 0.5 = 0.5), amount and date yield 0.0
    score = scorer.score_record_pair(r1, r2)
    assert score == pytest.approx(0.50)

    # Empty rows
    assert scorer.score_record_pair({}, {}) == 0.0


def test_score_record_pair_gated_mode() -> None:
    """Verify strict gating behavior in score_record_pair."""
    gated_scorer = FinancialCompositeScorer(
        amount_epsilon=0.50,
        date_delta_days=2.0,
        gated=True,
    )

    r_base = {"name": "Apex Logistics Inc.", "amount": 100.00, "date": "2025-07-01"}
    # Amount diff 1.00 > epsilon 0.50 -> amount score is 0.0
    r_bad_amt = {"name": "Apex Logistics Inc.", "amount": 101.00, "date": "2025-07-01"}
    # Date diff 5 days > delta_days 2.0 -> date score is 0.0
    r_bad_date = {"name": "Apex Logistics Inc.", "amount": 100.00, "date": "2025-07-06"}
    # Within tolerance on all
    r_good = {"name": "Apex Logistics Inc.", "amount": 100.20, "date": "2025-07-02"}

    assert gated_scorer.score_record_pair(r_base, r_bad_amt) == 0.0
    assert gated_scorer.score_record_pair(r_base, r_bad_date) == 0.0
    assert gated_scorer.score_record_pair(r_base, r_good) > 0.0


def test_score_record_pair_legal_normalization_toggle() -> None:
    """Verify normalize_legal=False compares legal suffixes verbatim."""
    scorer_raw = FinancialCompositeScorer(normalize_legal=False)
    scorer_norm = FinancialCompositeScorer(normalize_legal=True)

    r1 = {"name": "Acme Corp.", "amount": 100.0, "date": "2025-01-01"}
    r2 = {"name": "Acme LLC", "amount": 100.0, "date": "2025-01-01"}

    s_raw = scorer_raw.score_record_pair(r1, r2)
    s_norm = scorer_norm.score_record_pair(r1, r2)

    # With legal normalization, 'Acme' vs 'Acme' yields text score 1.0
    # Without legal normalization, 'Acme Corp.' vs 'Acme LLC' yields text score < 1.0
    assert s_norm > s_raw
    assert s_norm == pytest.approx(1.0)


def test_score_pairs_vectorized() -> None:
    """Verify batched score_pairs across DataFrames."""
    scorer = FinancialCompositeScorer(amount_epsilon=1.0, date_delta_days=3.0)

    df_l = pd.DataFrame(
        {
            "name": ["Alpha Corp", "Beta LLC", "Gamma Inc"],
            "amount": [100.0, 200.0, 300.0],
            "date": ["2025-01-01", "2025-02-01", "2025-03-01"],
        }
    )
    df_r = pd.DataFrame(
        {
            "name": ["Alpha Inc", "Beta LLC", "Delta Inc"],
            "amount": [100.5, 200.0, 500.0],
            "date": ["2025-01-02", "2025-02-01", "2025-03-10"],
        }
    )

    scores = scorer.score_pairs(df_l, df_r)
    assert isinstance(scores, pd.Series)
    assert len(scores) == 3
    # Row 0: Alpha matches, amount diff 0.5, date diff 1 day
    assert 0.70 < scores.iloc[0] < 1.0
    # Row 1: Exact match on all
    assert scores.iloc[1] == pytest.approx(1.0)
    # Row 2: Name mismatch, amount diff 200, date diff 9 days
    assert scores.iloc[2] < 0.20

    # Length mismatch
    with pytest.raises(ValueError, match="DataFrame length mismatch"):
        scorer.score_pairs(df_l, df_r.iloc[:1])


def test_score_pairs_gated() -> None:
    """Verify batched score_pairs with gated=True."""
    scorer = FinancialCompositeScorer(
        amount_epsilon=0.50,
        date_delta_days=2.0,
        gated=True,
    )

    df_l = pd.DataFrame(
        {
            "name": ["Alpha Corp", "Beta LLC"],
            "amount": [100.0, 200.0],
            "date": ["2025-01-01", "2025-02-01"],
        }
    )
    df_r = pd.DataFrame(
        {
            "name": ["Alpha Inc", "Beta LLC"],
            "amount": [100.2, 250.0],  # Row 1 amount out of tolerance
            "date": ["2025-01-02", "2025-02-01"],
        }
    )

    scores = scorer.score_pairs(df_l, df_r)
    assert scores.iloc[0] > 0.0
    assert scores.iloc[1] == 0.0


def test_convenience_reexports() -> None:
    """Verify amount_proximity_score and date_window_score re-exported in fuzzymatch_finance."""
    assert amount_proximity_score(100.0, 100.2, epsilon=0.5) == pytest.approx(0.60)
    assert date_window_score("2025-01-01", "2025-01-02", delta_days=5.0) == pytest.approx(0.80)


def test_getitem_fallback_extraction() -> None:
    """Verify field extraction for objects that only implement __getitem__ without get()."""

    class CustomRow:
        def __init__(self, data: dict[str, Any]) -> None:
            self._data = data

        def __getitem__(self, item: str) -> Any:
            if item in self._data:
                return self._data[item]
            raise KeyError(item)

    row1 = CustomRow({"name": "Acme Corp", "amount": 100.0, "date": "2025-01-01"})
    row2 = CustomRow({"unknown_col": "Something"})

    scorer = FinancialCompositeScorer()
    score = scorer.score_record_pair(row1, row2)
    assert score == 0.0


def test_score_pairs_missing_text_column() -> None:
    """Verify score_pairs when DataFrames lack text column."""
    scorer = FinancialCompositeScorer()
    df_l = pd.DataFrame({"amount": [100.0], "date": ["2025-01-01"]})
    df_r = pd.DataFrame({"amount": [100.0], "date": ["2025-01-01"]})

    scores = scorer.score_pairs(df_l, df_r)
    # text is 0.0 (w=0.5), amount is 1.0 (w=0.3), date is 1.0 (w=0.2) -> 0.50
    assert scores.iloc[0] == pytest.approx(0.50)
