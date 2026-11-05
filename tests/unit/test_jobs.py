from __future__ import annotations

import pandas as pd
from fuzzymatch import ExactBlock, FSClassifier, Matcher
from fuzzymatch.decisions import Explanation, FieldScore, MatchDecision

from fuzzymatch_finance.jobs import FinanceMappingJob
from fuzzymatch_finance.policy import MappingPolicy
from fuzzymatch_finance.rules import MappingRule, MappingRuleSet


class _StubMatcher:
    """Minimal duck-typed stand-in for ``fuzzymatch.Matcher``."""

    def __init__(
        self,
        decisions: list[MatchDecision] | None = None,
        link_type: str = "link",
        scoring_fields: list[str] | None = None,
    ) -> None:
        self.link_type = link_type
        self.scoring_fields = ["name"] if scoring_fields is None else scoring_fields
        self.scoring_method = "token_sort_ratio"
        self._decisions = decisions if decisions is not None else []
        self.fitted_with: tuple[pd.DataFrame, pd.DataFrame] | None = None

    def fit(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame | None = None,
        labels: pd.Series | None = None,
    ) -> _StubMatcher:
        self.fitted_with = (df_left, df_right if df_right is not None else df_left)
        return self

    def decisions(self) -> list[MatchDecision]:
        return self._decisions


def _decision(left_id: int, right_id: int, score: float, band: str = "match") -> MatchDecision:
    field_score = FieldScore(
        field="name", left_value="src value", right_value="tgt value", score=score
    )
    explanation = Explanation(
        field_scores=(field_score,),
        threshold_band=band,
        blocking_rules=("ExactBlock",),
    )
    return MatchDecision(
        left_id=left_id,
        right_id=right_id,
        status=band,
        score=score,
        explanation=explanation,
    )


def _single_unmatched_source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_code": ["TH01"],
            "local_account": ["999999"],
            "name": ["Misc Expense"],
        }
    )


def _two_row_target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_account": ["999000", "998000"],
            "name": ["Misc Expense", "Other Expense"],
        }
    )


def _fuzzy_job(matcher, **policy_kwargs) -> FinanceMappingJob:
    return FinanceMappingJob(
        rule_set=MappingRuleSet([]),
        policy=MappingPolicy(
            auto_accept_threshold=0.92,
            review_threshold=0.70,
            **policy_kwargs,
        ),
        fuzzy_matcher=matcher,
    )


def test_job_applies_rules_before_fuzzy_matching():
    source = pd.DataFrame(
        {
            "company_code": ["TH01", "TH01"],
            "local_account": ["610010", "999999"],
            "name": ["Office Supplies", "Misc Expense"],
        }
    )
    target = pd.DataFrame(
        {
            "target_account": ["610000", "999000"],
            "name": ["Office Supplies Expense", "Misc Expense"],
        }
    )
    rules = MappingRuleSet(
        [
            MappingRule(
                rule_id="coa_prefix_opex_v1",
                description="Map OPEX prefix",
                priority=1,
                condition={"company_code": "TH01", "local_account_prefix": "6"},
                target={"target_account": "610000"},
            )
        ]
    )
    policy = MappingPolicy(auto_accept_threshold=0.92, review_threshold=0.70)
    job = FinanceMappingJob(rule_set=rules, policy=policy)

    result = job.run(source, target)

    assert result.approved_mappings.shape[0] >= 1
    assert "decision_status" in result.review_queue.columns
    assert result.audit_log is not None
    assert result.audit_log.splitlines()[0].startswith("{")


def test_job_without_fuzzy_matcher_keeps_placeholder_reason():
    job = FinanceMappingJob(
        rule_set=MappingRuleSet([]),
        policy=MappingPolicy(auto_accept_threshold=0.92, review_threshold=0.70),
    )

    result = job.run(_single_unmatched_source(), _two_row_target())

    reasons = list(result.review_queue["reason_codes"].iloc[0])
    assert reasons == ["requires_fuzzy_fallback"]
    assert result.evidence.empty


def test_fuzzy_matcher_requires_link_mode():
    job = _fuzzy_job(_StubMatcher(link_type="dedupe"))

    try:
        job.run(_single_unmatched_source(), _two_row_target())
    except ValueError as exc:
        assert "link_type='link'" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-link matcher")


def test_fuzzy_fallback_auto_maps_high_score():
    matcher = _StubMatcher([_decision(0, 0, 0.95)])
    job = _fuzzy_job(matcher, min_auto_accept_evidence=0)

    result = job.run(_single_unmatched_source(), _two_row_target())

    approved = result.approved_mappings
    assert len(approved) == 1
    assert approved["decision_status"].iloc[0] == "fuzzy_auto_mapped"
    assert approved["target_account"].iloc[0] == "999000"
    assert approved["reason_codes"].iloc[0] == ["fuzzy_fallback_auto_accept"]

    evidence = result.evidence
    assert len(evidence) == 1
    assert evidence["policy_status"].iloc[0] == "auto_mapped"
    assert evidence["scoring_method"].iloc[0] == "token_sort_ratio"
    assert evidence["blocking_rules"].iloc[0] == "ExactBlock"

    assert '"event_type": "fuzzy_mapping_applied"' in result.audit_log


def test_fuzzy_fallback_thin_evidence_downgrades_to_review():
    """Default policy: a single candidate must never auto-map (B6 guardrail)."""
    matcher = _StubMatcher([_decision(0, 0, 0.99)])
    job = _fuzzy_job(matcher)

    result = job.run(_single_unmatched_source(), _two_row_target())

    assert result.approved_mappings.empty
    review = result.review_queue
    assert len(review) == 1
    assert review["reason_codes"].iloc[0] == [
        "fuzzy_fallback_needs_review",
        "thin_evidence",
    ]


def test_fuzzy_fallback_routes_mid_score_to_review():
    matcher = _StubMatcher([_decision(0, 0, 0.80)])
    job = _fuzzy_job(matcher)

    result = job.run(_single_unmatched_source(), _two_row_target())

    assert result.approved_mappings.empty
    review = result.review_queue
    assert len(review) == 1
    assert review["reason_codes"].iloc[0] == ["fuzzy_fallback_needs_review"]
    assert result.evidence["policy_status"].iloc[0] == "needs_review"
    assert '"event_type": "review_queue_item_created"' in result.audit_log


def test_fuzzy_fallback_low_confidence_goes_to_review():
    matcher = _StubMatcher([_decision(0, 0, 0.30)])
    job = _fuzzy_job(matcher)

    result = job.run(_single_unmatched_source(), _two_row_target())

    assert result.approved_mappings.empty
    assert result.review_queue["reason_codes"].iloc[0] == ["fuzzy_fallback_low_confidence"]
    assert result.evidence["policy_status"].iloc[0] == "unmapped"


def test_fuzzy_picks_best_candidate_per_source_row():
    source = pd.DataFrame(
        {
            "company_code": ["TH01", "TH01"],
            "local_account": ["888001", "888002"],
            "name": ["Alpha Costs", "Beta Costs"],
        }
    )
    matcher = _StubMatcher(
        [
            _decision(0, 0, 0.75),
            _decision(0, 1, 0.93),
            _decision(1, 0, 0.50),
        ]
    )
    job = _fuzzy_job(matcher, min_auto_accept_evidence=0)

    result = job.run(source, _two_row_target())

    approved = result.approved_mappings
    assert len(approved) == 1
    assert approved["source_local_account"].iloc[0] == "888001"
    assert approved["target_account"].iloc[0] == "998000"

    review = result.review_queue
    assert len(review) == 1
    assert review["source_local_account"].iloc[0] == "888002"


def test_fuzzy_no_candidate_keeps_review_entry():
    matcher = _StubMatcher([])
    job = _fuzzy_job(matcher)

    result = job.run(_single_unmatched_source(), _two_row_target())

    assert result.approved_mappings.empty
    assert result.evidence.empty
    assert result.review_queue["reason_codes"].iloc[0] == ["fuzzy_no_candidate"]


def test_fuzzy_scoring_field_default_applied_when_missing():
    matcher = _StubMatcher([], scoring_fields=[])
    job = _fuzzy_job(matcher)

    job.run(_single_unmatched_source(), _two_row_target())

    assert matcher.scoring_fields == ["name"]


def test_fuzzy_stage_skipped_when_all_rows_rule_matched():
    matcher = _StubMatcher()
    rules = MappingRuleSet(
        [
            MappingRule(
                rule_id="catch_all_v1",
                description="Map everything",
                priority=1,
                condition={"company_code": "TH01"},
                target={"target_account": "900000"},
            )
        ]
    )
    job = FinanceMappingJob(
        rule_set=rules,
        policy=MappingPolicy(auto_accept_threshold=0.92, review_threshold=0.70),
        fuzzy_matcher=matcher,
    )

    result = job.run(_single_unmatched_source(), _two_row_target())

    assert matcher.fitted_with is None
    assert result.review_queue.empty
    assert result.evidence.empty
    assert len(result.approved_mappings) == 1


def test_real_core_matcher_end_to_end():
    matcher = Matcher(
        link_type="link",
        blocking=ExactBlock(["name"]),
        classifier=FSClassifier(),
        scoring_fields=["name"],
    )
    job = FinanceMappingJob(
        rule_set=MappingRuleSet([]),
        policy=MappingPolicy(auto_accept_threshold=0.90, review_threshold=0.70),
        fuzzy_matcher=matcher,
    )
    source = pd.DataFrame(
        {
            "company_code": ["TH01"],
            "local_account": ["610010"],
            "name": ["Office Supplies"],
        }
    )
    target = pd.DataFrame(
        {
            "target_account": ["610000"],
            "name": ["Office Supplies"],
        }
    )

    result = job.run(source, target)

    # Real matcher produces a confident score, but the default thin-evidence
    # guardrail (B6) must downgrade single-candidate runs to review.
    assert result.approved_mappings.empty
    review = result.review_queue
    assert len(review) == 1
    assert review["reason_codes"].iloc[0] == [
        "fuzzy_fallback_needs_review",
        "thin_evidence",
    ]
    assert result.evidence["threshold_band"].iloc[0] == "match"
    assert result.evidence["fuzzy_score"].iloc[0] > 0.9
