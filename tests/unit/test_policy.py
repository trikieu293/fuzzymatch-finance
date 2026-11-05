from __future__ import annotations

import pytest

from fuzzymatch_finance.policy import MappingPolicy


def test_policy_classifies_scores_into_expected_bands():
    policy = MappingPolicy(auto_accept_threshold=0.92, review_threshold=0.70)

    assert policy.classify_score(0.95) == "auto_mapped"
    assert policy.classify_score(0.80) == "needs_review"
    assert policy.classify_score(0.50) == "unmapped"


def test_policy_rejects_invalid_threshold_order():
    with pytest.raises(ValueError, match="auto_accept_threshold"):
        MappingPolicy(auto_accept_threshold=0.50, review_threshold=0.70)


def test_policy_rejects_negative_evidence_floor():
    with pytest.raises(ValueError, match="min_auto_accept_evidence"):
        MappingPolicy(
            auto_accept_threshold=0.90,
            review_threshold=0.70,
            min_auto_accept_evidence=-1,
        )


def test_policy_route_bands():
    policy = MappingPolicy(auto_accept_threshold=0.90, review_threshold=0.70)
    # When band is needs_review, returns band without reason codes
    assert policy.route(0.80, "needs_review", n_candidates=50, max_score=0.85) == (
        "needs_review",
        None,
    )
    # When band is auto_mapped but evidence is thin
    assert policy.route(0.95, "auto_mapped", n_candidates=5, max_score=0.95) == (
        "needs_review",
        ["thin_evidence"],
    )
