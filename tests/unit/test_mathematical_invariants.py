"""Mathematical and property-based invariant verification for fuzzymatch-finance.

Verifies normalization idempotence, inter-rater agreement bounds and symmetries,
rule deterministic prioritization, and ISO 7064 / Luhn check-digit corruption sensitivity.
"""

from __future__ import annotations

import math
from typing import ClassVar

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from fuzzymatch_finance import (
    MappingPolicy,
    MappingRule,
    MappingRuleSet,
    accounting_normalizer,
    validate_cusip,
    validate_cusip_check_digit,
    validate_iban,
    validate_lei,
)
from fuzzymatch_finance.agreement import cohen_kappa
from fuzzymatch_finance.cli import auto_match_finance
from fuzzymatch_finance.scoring import FinancialCompositeScorer

# ---------------------------------------------------------------------------
# 1. Normalization Idempotence
# ---------------------------------------------------------------------------


class TestNormalizationInvariants:
    """Rigorous verification of normalizer idempotence: f(f(s)) == f(s)."""

    @given(st.text(max_size=100))
    @settings(max_examples=100)
    def test_accounting_normalizer_idempotence(self, text: str) -> None:
        """Applying accounting_normalizer twice produces the identical output."""
        normalizer = accounting_normalizer()
        df1 = pd.DataFrame({"name": [text]})
        once = normalizer.transform(df1, ["name"])
        twice = normalizer.transform(once.copy(), ["name"])
        assert once["name"].iloc[0] == twice["name"].iloc[0]


# ---------------------------------------------------------------------------
# 2. Inter-Rater Agreement Invariants (cohen_kappa)
# ---------------------------------------------------------------------------


class TestAgreementInvariants:
    """Verification of Cohen's kappa identity, symmetry, and metric properties."""

    @given(st.lists(st.sampled_from(["match", "possible", "non_match"]), min_size=5, max_size=50))
    @settings(max_examples=100)
    def test_cohen_kappa_identity(self, ratings: list[str]) -> None:
        """Self-agreement is always identically 1.0."""
        assert math.isclose(cohen_kappa(ratings, list(ratings)), 1.0, abs_tol=1e-12)

    @given(
        st.lists(st.sampled_from(["match", "possible", "non_match"]), min_size=5, max_size=50),
        st.lists(st.sampled_from(["match", "possible", "non_match"]), min_size=5, max_size=50),
    )
    @settings(max_examples=100)
    def test_cohen_kappa_symmetry_and_bounds(self, rater_a: list[str], rater_b: list[str]) -> None:
        """kappa(a, b) == kappa(b, a) and kappa in [-1.0, 1.0]."""
        # Truncate to equal length
        min_len = min(len(rater_a), len(rater_b))
        a = rater_a[:min_len]
        b = rater_b[:min_len]
        k_ab = cohen_kappa(a, b)
        k_ba = cohen_kappa(b, a)
        assert math.isclose(k_ab, k_ba, abs_tol=1e-12)
        assert -1.0 - 1e-9 <= k_ab <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 3. Policy and Rule Monotonicity
# ---------------------------------------------------------------------------


class TestPolicyAndRuleInvariants:
    """Verification of policy monotonicity and rule priority order."""

    @given(
        st.floats(min_value=0.0, max_value=1.0),
        st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(max_examples=100)
    def test_mapping_policy_threshold_monotonicity(self, s1: float, s2: float) -> None:
        """Higher scores must monotonically receive non-decreasing decision status."""
        policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)
        p_low = min(s1, s2)
        p_high = max(s1, s2)

        order = {"unmapped": 0, "needs_review": 1, "auto_mapped": 2}
        status_low = policy.classify_score(p_low)
        status_high = policy.classify_score(p_high)
        assert order[status_high] >= order[status_low]

    def test_mapping_rule_priority_determinism(self) -> None:
        """Lower priority numbers strictly override higher priority numbers."""
        rule_low_prio = MappingRule(
            rule_id="R10",
            description="Lower priority rule",
            priority=10,
            condition={"company_code": "1000"},
            target={"target_account": "ACC_TEN"},
        )
        rule_high_prio = MappingRule(
            rule_id="R1",
            description="High priority rule",
            priority=1,
            condition={"company_code": "1000"},
            target={"target_account": "ACC_ONE"},
        )
        # Pass in unordered list
        rule_set = MappingRuleSet([rule_low_prio, rule_high_prio])
        row = pd.Series({"company_code": "1000", "local_account": "12345"})
        res = rule_set.apply(row)
        assert res.matched is True
        assert res.rule_id == "R1"
        assert res.target == {"target_account": "ACC_ONE"}


# ---------------------------------------------------------------------------
# 4. Financial Check-Digit Corruption Sensitivity (ISO 7064 & Luhn)
# ---------------------------------------------------------------------------


class TestCheckDigitCorruptionSensitivity:
    """100% single-character substitution error detection verification."""

    # Valid real-world IBAN benchmarks
    VALID_IBANS: ClassVar[list[str]] = [
        "GB82WEST12345698765432",
        "DE89370400440532013000",
        "FR1420041010050500013M02606",
    ]

    # Valid 20-character LEI benchmarks (ISO 7064 Mod 97-10)
    VALID_LEIS: ClassVar[list[str]] = [
        "5493006MHB84DD0ZWV18",
        "2138001W1ED2V2WQLG48",
        "549300O540N0427EP077",
    ]

    # Valid 9-character CUSIP benchmarks
    VALID_CUSIPS: ClassVar[list[str]] = [
        "037833100",  # Apple Inc
        "594918104",  # Microsoft Corp
        "38259P508",  # Google / Alphabet
    ]

    def test_valid_identifier_baselines(self) -> None:
        """Baseline valid identifiers must all evaluate to True."""
        for iban in self.VALID_IBANS:
            assert validate_iban(iban) is True
        for lei in self.VALID_LEIS:
            assert validate_lei(lei) is True
        for cusip in self.VALID_CUSIPS:
            assert validate_cusip(cusip) is True

    @given(st.sampled_from(VALID_IBANS), st.data())
    @settings(max_examples=100)
    def test_iban_single_character_corruption_sensitivity(
        self, iban: str, data: st.DataObject
    ) -> None:
        """Any single character substitution in a valid IBAN must invalidate it."""
        clean = iban.replace(" ", "")
        idx = data.draw(st.integers(min_value=0, max_value=len(clean) - 1))
        original_char = clean[idx]

        # Draw a different alphanumeric char
        if original_char.isdigit():
            replacement = data.draw(
                st.sampled_from([str(d) for d in range(10) if str(d) != original_char])
            )
        else:
            replacement = data.draw(
                st.sampled_from(
                    [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) != original_char]
                )
            )

        corrupted = clean[:idx] + replacement + clean[idx + 1 :]
        assert validate_iban(corrupted) is False

    @given(st.sampled_from(VALID_LEIS), st.data())
    @settings(max_examples=100)
    def test_lei_single_character_corruption_sensitivity(
        self, lei: str, data: st.DataObject
    ) -> None:
        """Any single character substitution in a valid LEI must invalidate it."""
        idx = data.draw(st.integers(min_value=0, max_value=19))
        original_char = lei[idx]
        if original_char.isdigit():
            replacement = data.draw(
                st.sampled_from([str(d) for d in range(10) if str(d) != original_char])
            )
        else:
            replacement = data.draw(
                st.sampled_from(
                    [chr(c) for c in range(ord("A"), ord("Z") + 1) if chr(c) != original_char]
                )
            )
        corrupted = lei[:idx] + replacement + lei[idx + 1 :]
        assert validate_lei(corrupted) is False

    @given(st.sampled_from(VALID_CUSIPS), st.data())
    @settings(max_examples=100)
    def test_cusip_check_digit_corruption_sensitivity(
        self, cusip: str, data: st.DataObject
    ) -> None:
        """Any single-digit substitution in the 9th check digit flips validation from True to False."""
        original_check = cusip[8]
        new_check = data.draw(
            st.sampled_from([str(d) for d in range(10) if str(d) != original_check])
        )
        corrupted = cusip[:8] + new_check
        assert validate_cusip(corrupted) is False

    def test_invalid_identifier_formats(self) -> None:
        """Malformed or out-of-spec strings must evaluate to False."""
        assert validate_iban("") is False
        assert validate_iban("GB12") is False
        assert validate_iban("1234567890") is False
        assert validate_lei("") is False
        assert validate_lei("SHORT") is False
        assert validate_lei("5493006MHB84DD0ZWV18EXTRA") is False
        assert validate_cusip("") is False
        assert validate_cusip("12345") is False
        assert validate_cusip("03783310A") is False  # Non-digit check character

    def test_special_characters_and_edge_cases(self) -> None:
        """Validate CUSIP special characters *, @, # and ISO7064 non-alphanumeric rejection."""
        from fuzzymatch_finance.identifiers import _iso7064_mod97_10

        assert _iso7064_mod97_10("GB!12") is False
        assert validate_cusip("037833*02") is True
        assert validate_cusip("037833*01") is False  # Wrong check digit
        assert validate_cusip("037833@01") is True
        assert validate_cusip("037833#00") is True
        assert validate_cusip("037833!01") is False  # Invalid character in CUSIP
        assert validate_cusip_check_digit is validate_cusip
        assert validate_cusip_check_digit("037833100") is True


# ---------------------------------------------------------------------------
# 5. Financial Scoring and Conservation Invariants
# ---------------------------------------------------------------------------


class TestFinancialScoringAndConservationInvariants:
    """Hypothesis property tests for composite financial scoring and conservation laws."""

    @settings(max_examples=100)
    @given(
        w_text=st.floats(min_value=0.1, max_value=10.0),
        w_amount=st.floats(min_value=0.1, max_value=10.0),
        w_date=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_financial_scorer_simplex_weight_conservation(
        self, w_text: float, w_amount: float, w_date: float
    ) -> None:
        """Raw positive weights are projected onto the probability simplex: sum(w) == 1.0."""
        scorer = FinancialCompositeScorer(
            weights={"text": w_text, "amount": w_amount, "date": w_date}
        )
        assert math.isclose(sum(scorer.weights.values()), 1.0, abs_tol=1e-9)
        for w in scorer.weights.values():
            assert w >= 0.0

    @settings(max_examples=100, deadline=None)
    @given(
        n_left=st.integers(min_value=1, max_value=20),
        n_right=st.integers(min_value=1, max_value=20),
    )
    def test_reconciliation_cardinality_and_volume_conservation(
        self, n_left: int, n_right: int
    ) -> None:
        """1-to-1 cardinality conservation and total volume conservation in financial matching."""
        df_l = pd.DataFrame(
            {
                "id": [f"L_{i}" for i in range(n_left)],
                "name": [f"Vendor {i} International Inc" for i in range(n_left)],
                "amount": [100.0 + i * 10 for i in range(n_left)],
            }
        )
        df_r = pd.DataFrame(
            {
                "id": [f"R_{j}" for j in range(n_right)],
                "name": [f"Vendor {j} International Inc" for j in range(n_right)],
                "amount": [100.0 + j * 10 for j in range(n_right)],
            }
        )
        results = auto_match_finance(df_l, df_r, threshold=0.7)
        m = results.matches
        s = results.summary

        # 1. 1-to-1 Cardinality Invariant: |M| <= min(|L|, |R|)
        assert len(m) <= min(n_left, n_right)
        if not m.empty:
            assert m["left_id"].nunique() == len(m)
            assert m["right_id"].nunique() == len(m)

        # 2. Total Volume Conservation: Total = Left + Right
        assert math.isclose(
            s.total_dollar_volume,
            s.left_dollar_volume + s.right_dollar_volume,
            abs_tol=1e-6,
        )
        # Left Volume Partition: Left = Reconciled + Unmatched Left
        assert s.left_dollar_volume >= s.reconciled_dollar_volume - 1e-6
