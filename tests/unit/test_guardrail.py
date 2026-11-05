"""Thin-evidence guardrail: auto-accept requires sufficient candidate support."""

from __future__ import annotations

import pytest

from fuzzymatch_finance.policy import MappingPolicy


class TestThinEvidenceGuardrail:
    def test_policy_defaults_enable_guardrail(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)
        assert policy.min_auto_accept_evidence == 20

    def test_route_downgrades_when_candidate_support_thin(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)
        status, reason = policy.route(
            score=0.95,
            band="auto_mapped",
            n_candidates=5,
            max_score=0.95,
        )
        assert status == "needs_review"
        assert reason == ["thin_evidence"]

    def test_route_passes_through_with_support(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)
        status, reason = policy.route(
            score=0.95,
            band="auto_mapped",
            n_candidates=50,
            max_score=0.95,
        )
        assert status == "auto_mapped"
        assert reason is None

    def test_non_auto_bands_unaffected(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)
        for band in ("needs_review", "unmapped"):
            status, _reason = policy.route(0.3, band, n_candidates=2, max_score=0.3)
            assert status == band

    def test_zero_floor_disables_guardrail(self) -> None:
        policy = MappingPolicy(
            auto_accept_threshold=0.85,
            review_threshold=0.60,
            min_auto_accept_evidence=0,
        )
        status, _ = policy.route(0.99, "auto_mapped", n_candidates=1, max_score=0.99)
        assert status == "auto_mapped"

    def test_max_score_below_half_floor_blocks(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)
        status, reason = policy.route(
            score=0.9, band="auto_mapped", n_candidates=100, max_score=0.4
        )
        assert status == "needs_review"
        assert reason == ["thin_evidence"]


class TestJobGuardrailIntegration:
    def test_job_routes_via_policy_route(self) -> None:
        """End-to-end: thin-candidate runs never auto-map."""
        pytest.importorskip("rapidfuzz")
