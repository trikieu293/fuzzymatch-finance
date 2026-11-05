"""Run manifest and drift-audit verification contracts."""

from __future__ import annotations

from typing import ClassVar

import pandas as pd

from fuzzymatch_finance.jobs import FinanceMappingJob
from fuzzymatch_finance.policy import MappingPolicy
from fuzzymatch_finance.rules import MappingRule, MappingRuleSet


def _job(policy: MappingPolicy | None = None) -> FinanceMappingJob:
    return FinanceMappingJob(
        rule_set=MappingRuleSet(
            rules=[
                MappingRule(
                    rule_id="r1",
                    description="d",
                    priority=1,
                    condition={"company_code": "TH01", "local_account_prefix": "6"},
                    target={"target_account": "610000"},
                )
            ]
        ),
        policy=policy or MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.7),
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


class TestRunManifest:
    def test_manifest_present_after_run(self) -> None:
        result = _job().run(_source(), _target())
        assert result.manifest["config_hash"]
        assert result.manifest["finance_version"]
        assert result.manifest["fuzzymatch_version"]
        assert result.manifest["rows_source"] == 3

    def test_config_hash_deterministic_per_job(self) -> None:
        first = _job().run(_source(), _target())
        second = _job().run(_source(), _target())
        assert first.manifest["config_hash"] == second.manifest["config_hash"]

    def test_policy_change_changes_hash(self) -> None:
        strict = _job(MappingPolicy(auto_accept_threshold=0.95, review_threshold=0.9))
        loose = _job(MappingPolicy(auto_accept_threshold=0.5, review_threshold=0.3))
        assert (
            strict.run(_source(), _target()).manifest["config_hash"]
            != loose.run(_source(), _target()).manifest["config_hash"]
        )

    def test_default_manifest_empty_for_backcompat(self) -> None:
        from fuzzymatch_finance.results import MappingResult

        bare = MappingResult(
            approved_mappings=pd.DataFrame(),
            review_queue=pd.DataFrame(),
            evidence=pd.DataFrame(),
            audit_log="",
        )
        assert bare.manifest == {}


class TestDriftAuditWiring:
    def test_reference_scores_emit_drift_event(self) -> None:
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
        result = job.run(_source(), _target(), reference_scores=[0.1] * 20)
        events = [line for line in result.audit_log.splitlines() if "score_drift_check" in line]
        assert len(events) == 1
        assert '"psi"' in events[0]

    def test_no_reference_no_event(self) -> None:
        result = _job().run(_source(), _target())
        assert "score_drift_check" not in result.audit_log
