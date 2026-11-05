"""Tests for review-queue label capture and prioritization (learning loop)."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pandas as pd

from fuzzymatch_finance.jobs import FinanceMappingJob
from fuzzymatch_finance.label_store import LabeledPairStore, capture_labels
from fuzzymatch_finance.policy import MappingPolicy
from fuzzymatch_finance.results import MappingResult
from fuzzymatch_finance.rules import MappingRule, MappingRuleSet

# ---------------------------------------------------------------------------
# Fixtures: one deterministic rule approval + fuzzy-less review queue
# ---------------------------------------------------------------------------


def _job() -> FinanceMappingJob:
    return FinanceMappingJob(
        rule_set=MappingRuleSet(
            rules=[
                MappingRule(
                    rule_id="coa_prefix_opex_v1",
                    description="Map OPEX prefix",
                    priority=1,
                    condition={"company_code": "TH01", "local_account_prefix": "6"},
                    target={"target_account": "610000"},
                )
            ]
        ),
        policy=MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.7),
    )


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "company_code": ["TH01", "TH01", "TH02"],
            "local_account": ["6000", "6100", "6200"],
        },
        index=[0, 1, 2],
    )


def _target() -> pd.DataFrame:
    return pd.DataFrame({"target_account": ["OPEX", "PAYROLL", "UTIL"]})


def _result() -> MappingResult:
    return _job().run(source=_source(), target=_target())


class TestCaptureLabels:
    def test_approved_rows_captured_as_positives(self, tmp_path: Path) -> None:
        store = LabeledPairStore(tmp_path / "labels.csv")
        from fuzzymatch_finance.label_store import capture_labels

        written = capture_labels(store, _result(), dataset_tag="coa", include_approved=True)
        pairs = store.for_dataset("coa")
        assert written == len(pairs) > 0
        assert all(p.label == 1 for p in pairs)

    def test_review_outcomes_recorded_against_suggested_target(self, tmp_path: Path) -> None:
        store = LabeledPairStore(tmp_path / "labels.csv")
        from fuzzymatch.decisions import Explanation, FieldScore, MatchDecision

        class _StubMatcher:
            link_type = "link"
            scoring_fields: ClassVar[list[str]] = []
            scoring_method = "token_sort_ratio"

            def fit(self, left, right, labels=None):
                self._left = left.reset_index(drop=True)

            def decisions(self):
                out = []
                for i in range(len(self._left)):
                    fs = FieldScore(field="name", left_value="a", right_value="b", score=0.72)
                    out.append(
                        MatchDecision(
                            left_id=i,
                            right_id=0,
                            status="possible",
                            score=0.72,
                            explanation=Explanation(field_scores=(fs,), threshold_band="possible"),
                        )
                    )
                return out

        job = FinanceMappingJob(
            rule_set=MappingRuleSet(rules=[]),
            policy=MappingPolicy(auto_accept_threshold=0.95, review_threshold=0.90),
            fuzzy_matcher=_StubMatcher(),  # type: ignore[arg-type]
        )
        result = job.run(source=_source(), target=_target())
        review = result.review_queue_dataframe()
        scored_sources = list(review[review["fuzzy_score"].notna()]["source_local_account"])
        assert len(scored_sources) >= 2

        store_result = capture_labels(
            store,
            result,
            outcomes={scored_sources[0]: True, scored_sources[1]: False},
            dataset_tag="coa",
            labeler="review:alice",
        )
        assert store_result == 2
        pairs = {p.left_key: p for p in store.all() if p.labeler == "review:alice"}
        assert pairs[scored_sources[0]].label == 1
        assert pairs[scored_sources[1]].label == 0
        assert all(p.right_key == "OPEX" for p in pairs.values())

    def test_unknown_sources_are_skipped(self, tmp_path: Path) -> None:
        store = LabeledPairStore(tmp_path / "labels.csv")
        from fuzzymatch_finance.label_store import capture_labels

        written = capture_labels(
            store, _result(), outcomes={"not-a-source": True}, dataset_tag="coa"
        )
        assert written >= 0  # no crash; unknown keys ignored

    def test_include_approved_false_skips_rules(self, tmp_path: Path) -> None:
        store = LabeledPairStore(tmp_path / "labels.csv")
        from fuzzymatch_finance.label_store import capture_labels

        written = capture_labels(store, _result(), dataset_tag="coa", include_approved=False)
        assert written == 0


# ---------------------------------------------------------------------------
# Prioritization
# ---------------------------------------------------------------------------


class TestPrioritizedReview:
    def test_orders_by_distance_to_nearest_boundary(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.7)
        queue = pd.DataFrame(
            [
                {"source_local_account": "far", "fuzzy_score": 0.10},
                {"source_local_account": "edge", "fuzzy_score": 0.69},
                {"source_local_account": "mid", "fuzzy_score": 0.80},
            ]
        )
        result = MappingResult(
            approved_mappings=pd.DataFrame(),
            review_queue=queue,
            evidence=pd.DataFrame(),
            audit_log="",
        )
        ordered = result.prioritized_review(policy)
        assert list(ordered["source_local_account"]) == ["edge", "mid", "far"]
        # original untouched
        assert list(result.review_queue["source_local_account"]) == [
            "far",
            "edge",
            "mid",
        ]

    def test_rows_without_score_go_last(self) -> None:
        policy = MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.7)
        queue = pd.DataFrame(
            [
                {"source_local_account": "noscore", "reason_codes": ["x"]},
                {"source_local_account": "scored", "fuzzy_score": 0.71},
            ]
        )
        result = MappingResult(
            approved_mappings=pd.DataFrame(),
            review_queue=queue,
            evidence=pd.DataFrame(),
            audit_log="",
        )
        ordered = result.prioritized_review(policy)
        assert ordered.iloc[-1]["source_local_account"] == "noscore"

    def test_fuzzy_review_rows_carry_score_and_target(self) -> None:
        """Fuzzy-stage review rows expose fuzzy_score + suggested_target."""
        from fuzzymatch.decisions import Explanation, FieldScore, MatchDecision

        class _StubMatcher:
            link_type = "link"
            scoring_fields: ClassVar[list[str]] = []
            scoring_method = "token_sort_ratio"

            def fit(self, left, right, labels=None):
                self._left = left.reset_index(drop=True)

            def decisions(self):
                out = []
                for i in range(len(self._left)):
                    if i == 0:
                        continue  # no candidate for first row
                    fs = FieldScore(field="name", left_value="a", right_value="b", score=0.72)
                    out.append(
                        MatchDecision(
                            left_id=i,
                            right_id=0,
                            status="possible",
                            score=0.72,
                            explanation=Explanation(field_scores=(fs,), threshold_band="possible"),
                        )
                    )
                return out

        job = FinanceMappingJob(
            rule_set=MappingRuleSet(rules=[]),
            policy=MappingPolicy(auto_accept_threshold=0.95, review_threshold=0.90),
            fuzzy_matcher=_StubMatcher(),  # type: ignore[arg-type]
        )
        result = job.run(source=_source(), target=_target())
        review = result.review_queue_dataframe()
        assert "suggested_target" in review.columns
        assert "fuzzy_score" in review.columns
        scored = review[review["fuzzy_score"].notna()]
        assert len(scored) == 2
