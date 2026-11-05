"""Tier 2: Boundary & Corner Case Tests for Requirement R2.

Legal Entity Normalization & Continuous Scorer Boundaries:
- Empty strings, whitespace, and isolated legal suffixes
- Exact distance cutoffs (|a1 - a2| = epsilon, |t1 - t2| = delta)
- Missing (NaN, None, NaT) inputs to numeric and date scorers
- Negative amounts and leap year date transitions
- Degenerate zero or negative epsilon / delta tolerances
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ..conftest import require_feature


@pytest.fixture
def strip_legal_suffixes_fn() -> Any:
    """Fixture providing strip_legal_suffixes gated by Milestone M2."""
    try:
        return require_feature("fuzzymatch.normalize.legal", "strip_legal_suffixes", milestone="M2")
    except pytest.skip.Exception:
        return require_feature("fuzzymatch_finance.lexicon", "strip_legal_suffixes", milestone="M2")


@pytest.fixture
def numeric_scorer_cls() -> Any:
    """Fixture providing NumericProximityScorer gated by Milestone M2."""
    return require_feature("fuzzymatch.scoring.numeric", "NumericProximityScorer", milestone="M2")


@pytest.fixture
def date_scorer_cls() -> Any:
    """Fixture providing DateWindowScorer gated by Milestone M2."""
    return require_feature("fuzzymatch.scoring.date", "DateWindowScorer", milestone="M2")


def test_r2_legal_suffix_empty_or_whitespace(strip_legal_suffixes_fn: Any) -> Any:
    """R2 Boundary: Empty strings, whitespace, and punctuation do not raise exceptions."""
    assert strip_legal_suffixes_fn("") == ""
    assert strip_legal_suffixes_fn("   ").strip() == ""
    assert strip_legal_suffixes_fn(", .").strip() in [", .", ""]


def test_r2_legal_suffix_only_string_preservation(strip_legal_suffixes_fn: Any) -> Any:
    """R2 Boundary: If a string consists solely of a legal suffix, it must NOT strip to empty."""
    suffix_only = ["Inc", "LLC", "Ltd", "Corp", "GmbH"]
    for s in suffix_only:
        res = strip_legal_suffixes_fn(s).strip()
        # Must retain original token to avoid producing an empty entity name
        assert len(res) > 0, f"Stripping solely suffix '{s}' resulted in an empty string!"


def test_r2_numeric_scorer_exact_boundary(numeric_scorer_cls: Any) -> Any:
    """R2 Boundary: Exact |a1 - a2| = epsilon yields exactly 0.0."""
    scorer = numeric_scorer_cls(epsilon=5.0)

    # |10.0 - 15.0| = 5.0 -> score = 0.0
    res = scorer.score_pair(pd.Series([10.0]), pd.Series([15.0]))
    assert math.isclose(float(res.iloc[0]), 0.0, abs_tol=1e-9)

    # Negative numbers: |-50.0 - (-52.0)| = 2.0 -> score = 1 - 2/5 = 0.6
    res_neg = scorer.score_pair(pd.Series([-50.0]), pd.Series([-52.0]))
    assert math.isclose(float(res_neg.iloc[0]), 0.6, abs_tol=1e-6)


def test_r2_numeric_scorer_nan_and_none(numeric_scorer_cls: Any) -> Any:
    """R2 Boundary: Missing/NaN/None values produce 0.0 without throwing exceptions."""
    scorer = numeric_scorer_cls(epsilon=10.0)

    res = scorer.score_pair(pd.Series([np.nan, 100.0, None]), pd.Series([100.0, np.nan, None]))
    assert len(res) == 3
    assert float(res.iloc[0]) == 0.0
    assert float(res.iloc[1]) == 0.0
    assert float(res.iloc[2]) == 0.0


def test_r2_date_scorer_exact_boundary_and_nat(date_scorer_cls: Any) -> Any:
    """R2 Boundary: Exact |t1 - t2| = delta yields 0.0, and NaT/None yields 0.0."""
    scorer = date_scorer_cls(delta_days=3.0)

    # Exactly 3 days difference
    d1 = pd.Series([pd.Timestamp("2024-02-28")])  # Leap year
    d2 = pd.Series([pd.Timestamp("2024-03-02")])  # Feb 29 was leap day: diff = 3 days
    res_leap = scorer.score_pair(d1, d2)
    assert math.isclose(float(res_leap.iloc[0]), 0.0, abs_tol=1e-6)

    # NaT input
    res_nat = scorer.score_pair(pd.Series([pd.NaT]), pd.Series([pd.Timestamp("2025-01-01")]))
    assert float(res_nat.iloc[0]) == 0.0


def test_r2_scorer_zero_or_negative_tolerance_rejection(
    numeric_scorer_cls: Any, date_scorer_cls: Any
) -> Any:
    """R2 Boundary: Degenerate epsilon <= 0 or delta_days <= 0 raises ValueError."""
    with pytest.raises((ValueError, ZeroDivisionError)):
        scorer = numeric_scorer_cls(epsilon=0.0)
        scorer.score_pair(pd.Series([10.0]), pd.Series([10.0]))

    with pytest.raises((ValueError, ZeroDivisionError)):
        scorer_d = date_scorer_cls(delta_days=0.0)
        scorer_d.score_pair(
            pd.Series([pd.Timestamp("2025-01-01")]), pd.Series([pd.Timestamp("2025-01-01")])
        )
