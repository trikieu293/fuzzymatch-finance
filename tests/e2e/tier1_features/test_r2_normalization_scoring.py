"""Tier 1: Feature Coverage Tests for Requirement R2.

Composite Domain Scoring & Legal Entity Normalization:
- F2.1: Legal entity suffix normalization across international jurisdictions
- F2.2: Continuous numeric proximity scorer with linear decay
- F2.3: Continuous date window proximity scorer
- F2.4: Financial composite multi-attribute scorer
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import pytest

from ..conftest import require_feature


@pytest.fixture
def strip_legal_suffixes_fn() -> Any:
    """Fixture providing strip_legal_suffixes gated by Milestone M2."""
    try:
        return require_feature("fuzzymatch.normalize.legal", "strip_legal_suffixes", milestone="M2")
    except pytest.skip.Exception:
        # Fallback check in lexicon or normalize
        return require_feature("fuzzymatch_finance.lexicon", "strip_legal_suffixes", milestone="M2")


@pytest.fixture
def numeric_scorer_cls() -> Any:
    """Fixture providing NumericProximityScorer gated by Milestone M2."""
    return require_feature("fuzzymatch.scoring.numeric", "NumericProximityScorer", milestone="M2")


@pytest.fixture
def date_scorer_cls() -> Any:
    """Fixture providing DateWindowScorer gated by Milestone M2."""
    return require_feature("fuzzymatch.scoring.date", "DateWindowScorer", milestone="M2")


@pytest.fixture
def financial_composite_scorer_cls() -> Any:
    """Fixture providing FinancialCompositeScorer gated by Milestone M2."""
    return require_feature("fuzzymatch_finance.scoring", "FinancialCompositeScorer", milestone="M2")


def test_legal_suffix_normalization_international(strip_legal_suffixes_fn: Any) -> Any:
    """R2: Normalizes corporate legal suffixes across international jurisdictions."""
    cases = [
        ("Acme Corp.", "Acme"),
        ("Acme Corporation", "Acme"),
        ("BlueSky Technologies Inc", "BlueSky Technologies"),
        ("BlueSky Technologies, Inc.", "BlueSky Technologies"),
        ("Cascade Supply LLC", "Cascade Supply"),
        ("Summit Chemical GmbH", "Summit Chemical"),
        ("Global Logistics Ltd.", "Global Logistics"),
        ("Iberian Rail S.A.", "Iberian Rail"),
        ("Precision Fasteners Co.", "Precision Fasteners"),
    ]
    for raw, expected in cases:
        normalized = strip_legal_suffixes_fn(raw).strip()
        assert normalized == expected, f"Expected '{expected}' for '{raw}', got '{normalized}'"


def test_legal_suffix_preserves_brand_tokens_and_non_suffixes(strip_legal_suffixes_fn: Any) -> Any:
    """R2: Preserves distinctive brand tokens and avoids stripping interior words."""
    # "Incorporated" as part of brand or non-suffix position
    text = "General Industrial Solutions"
    assert strip_legal_suffixes_fn(text).strip() == "General Industrial Solutions"

    # Words like "Incorporated" if not recognized as terminal suffix or brand
    brand = "Acme Tools and Company Supplies"
    # Should not strip internal "Company"
    assert "Company" in strip_legal_suffixes_fn(brand)


def test_legal_suffix_idempotence(strip_legal_suffixes_fn: Any) -> Any:
    """R2 & Invariant: Legal entity suffix normalization is strictly idempotent: f(f(s)) = f(s)."""
    samples = [
        "Apex Logistics LLC",
        "Apex Logistics",
        "Vanguard Electronics GmbH & Co. KG",
        "Summit Chemical Ltd.",
    ]
    for s in samples:
        f1 = strip_legal_suffixes_fn(s)
        f2 = strip_legal_suffixes_fn(f1)
        assert f1 == f2, f"Idempotence violation on '{s}': '{f1}' != '{f2}'"


def test_numeric_proximity_continuous_score(numeric_scorer_cls: Any) -> Any:
    """R2: NumericProximityScorer computes max(0.0, 1.0 - |a1 - a2| / epsilon)."""
    scorer = numeric_scorer_cls(epsilon=10.0)

    # Perfect match: |a1 - a2| = 0 -> 1.0
    s_perf = scorer.score_pair(pd.Series([100.0]), pd.Series([100.0]))
    assert math.isclose(float(s_perf.iloc[0]), 1.0, abs_tol=1e-6)

    # Half distance: |100 - 105| = 5 -> 1 - 5/10 = 0.5
    s_half = scorer.score_pair(pd.Series([100.0]), pd.Series([105.0]))
    assert math.isclose(float(s_half.iloc[0]), 0.5, abs_tol=1e-6)

    # Exact boundary: |100 - 110| = 10 -> 0.0
    s_bound = scorer.score_pair(pd.Series([100.0]), pd.Series([110.0]))
    assert math.isclose(float(s_bound.iloc[0]), 0.0, abs_tol=1e-6)

    # Outside boundary: |100 - 120| = 20 -> max(0, 1 - 20/10) = 0.0
    s_out = scorer.score_pair(pd.Series([100.0]), pd.Series([120.0]))
    assert float(s_out.iloc[0]) == 0.0


def test_date_window_continuous_score(date_scorer_cls: Any) -> Any:
    """R2: DateWindowScorer computes max(0.0, 1.0 - |t1 - t2| / delta_days)."""
    scorer = date_scorer_cls(delta_days=5.0)

    d1 = pd.Series([pd.Timestamp("2025-05-10")])
    # Exact same day -> 1.0
    s_same = scorer.score_pair(d1, pd.Series([pd.Timestamp("2025-05-10")]))
    assert math.isclose(float(s_same.iloc[0]), 1.0, abs_tol=1e-6)

    # 2 days lag -> 1 - 2/5 = 0.6
    s_lag = scorer.score_pair(d1, pd.Series([pd.Timestamp("2025-05-12")]))
    assert math.isclose(float(s_lag.iloc[0]), 0.6, abs_tol=1e-6)

    # 5 days lag (boundary) -> 0.0
    s_bound = scorer.score_pair(d1, pd.Series([pd.Timestamp("2025-05-15")]))
    assert math.isclose(float(s_bound.iloc[0]), 0.0, abs_tol=1e-6)

    # 10 days lag (outside) -> 0.0
    s_out = scorer.score_pair(d1, pd.Series([pd.Timestamp("2025-05-20")]))
    assert float(s_out.iloc[0]) == 0.0


def test_financial_composite_scorer_combination(financial_composite_scorer_cls: Any) -> Any:
    """R2: FinancialCompositeScorer combines string, numeric amount, and date window."""
    # Composite scorer weights: text=0.5, amount=0.3, date=0.2 (sums to 1.0)
    scorer = financial_composite_scorer_cls(
        amount_epsilon=2.0,
        date_delta_days=5.0,
        weights={"text": 0.5, "amount": 0.3, "date": 0.2},
    )

    df_l = pd.DataFrame(
        {
            "name": ["Acme Supplies"],
            "amount": [100.00],
            "date": ["2025-06-01"],
        }
    )
    df_r = pd.DataFrame(
        {
            "name": ["Acme Supplies LLC"],
            "amount": [101.00],  # diff 1.0 / 2.0 -> amount_score = 0.5
            "date": ["2025-06-02"],  # diff 1 day / 5.0 -> date_score = 0.8
        }
    )

    score = scorer.score_record_pair(df_l.iloc[0], df_r.iloc[0])
    assert 0.0 <= score <= 1.0
    # Score must be positive since text, amount, and date are all in close proximity
    assert score >= 0.50
