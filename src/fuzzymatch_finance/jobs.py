"""Finance mapping job orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Hashable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version as _pkg_version
from typing import cast

import fuzzymatch
import pandas as pd
from fuzzymatch import Matcher
from fuzzymatch.decisions import MatchDecision
from fuzzymatch.monitoring import drift_report
from fuzzymatch.normalize.steps import replace_words

from .lexicon import DEFAULT_ACCOUNTING_SYNONYMS
from .logging import AuditEvent, AuditLog
from .policy import MappingPolicy
from .results import MappingResult
from .rules import MappingRuleSet


@dataclass(frozen=True)
class FinanceMappingJob:
    """Run a rule-first finance mapping job with an optional fuzzy fallback.

    Workflow per source row:

    1. Deterministic rules (:class:`MappingRuleSet`) are evaluated first.
    2. Rows that no rule resolves fall back to fuzzy matching against the
       target table via the injected core :class:`~fuzzymatch.Matcher`
       (link mode). The best candidate per source row is scored, then routed
       by :meth:`MappingPolicy.classify_score`.
    3. Every decision is recorded in the structured JSONL audit log; fuzzy
       decisions additionally populate the evidence table.
    """

    rule_set: MappingRuleSet
    policy: MappingPolicy
    fuzzy_matcher: Matcher | None = None
    fuzzy_scoring_field: str = "name"
    lexicon_overrides: dict[str, str] | None = None

    def run(
        self,
        source: pd.DataFrame,
        target: pd.DataFrame,
        reference_scores: list[float] | None = None,
    ) -> MappingResult:
        """Execute the rule-first mapping workflow.

        Args:
            source: Source finance records to map.
            target: Target finance records to match against.
            reference_scores: Optional score sample from a previous good run;
                when provided, a PSI drift check is appended to the audit log.

        Returns:
            MappingResult with approved mappings, review queue, evidence for
            every fuzzy decision (empty when no fuzzy fallback ran), and
            structured JSONL audit log plus a reproducibility manifest.
        """
        if self.fuzzy_matcher is not None and self.fuzzy_matcher.link_type != "link":
            raise ValueError("fuzzy_matcher must be configured with link_type='link'")

        approved_rows: list[dict[str, object]] = []
        review_rows: list[dict[str, object]] = []
        evidence_rows: list[dict[str, object]] = []
        events: list[AuditEvent] = []

        unmatched: list[tuple[Hashable, str]] = []

        for label, row in source.iterrows():
            rule_result = self.rule_set.apply(row)
            if rule_result.matched and rule_result.target is not None:
                approved_rows.append(
                    {
                        "source_local_account": row.get("local_account"),
                        "target_account": rule_result.target["target_account"],
                        "decision_status": rule_result.decision_status,
                        "reason_codes": rule_result.reason_codes,
                    }
                )
                events.append(
                    AuditEvent(
                        event_type="rule_mapping_applied",
                        payload={
                            "source_local_account": row.get("local_account"),
                            "target_account": rule_result.target["target_account"],
                            "rule_id": rule_result.rule_id,
                            "reason_codes": rule_result.reason_codes,
                        },
                    )
                )
                continue

            unmatched.append((label, str(row.get("local_account"))))

        if self.fuzzy_matcher is not None and unmatched:
            fuzzy_approved, fuzzy_review, fuzzy_evidence, fuzzy_events = self._run_fuzzy_fallback(
                self.fuzzy_matcher, source, target, unmatched
            )
            approved_rows.extend(fuzzy_approved)
            review_rows.extend(fuzzy_review)
            evidence_rows.extend(fuzzy_evidence)
            events.extend(fuzzy_events)
        else:
            for _, source_account in unmatched:
                review_rows.append(
                    {
                        "source_local_account": source_account,
                        "decision_status": "needs_review",
                        "reason_codes": ["requires_fuzzy_fallback"],
                    }
                )
                events.append(
                    AuditEvent(
                        event_type="review_queue_item_created",
                        payload={
                            "source_local_account": source_account,
                            "decision_status": "needs_review",
                            "reason_codes": ["requires_fuzzy_fallback"],
                        },
                    )
                )

        if reference_scores:
            fuzzy_scores = [
                float(cast(float, e["fuzzy_score"])) for e in evidence_rows if "fuzzy_score" in e
            ]
            if fuzzy_scores:
                drift = drift_report(reference_scores, fuzzy_scores)
                events.append(
                    AuditEvent(
                        event_type="score_drift_check",
                        payload={
                            "psi": drift["psi"],
                            "warning": drift["warning"],
                            "n_current": len(fuzzy_scores),
                            "n_reference": len(reference_scores),
                        },
                    )
                )

        audit_log = AuditLog(events).to_jsonl()
        manifest = self._manifest(source, target)
        return MappingResult(
            approved_mappings=pd.DataFrame(approved_rows),
            review_queue=pd.DataFrame(review_rows),
            evidence=pd.DataFrame(evidence_rows),
            audit_log=audit_log,
            manifest=manifest,
        )

    def _manifest(self, source: pd.DataFrame, target: pd.DataFrame) -> dict[str, object]:
        """Reproducibility manifest: config hash + versions + row counts."""
        config_blob = json.dumps(
            {
                "rules": [
                    (r.rule_id, r.priority, r.condition, r.target) for r in self.rule_set.rules
                ],
                "policy": (
                    self.policy.auto_accept_threshold,
                    self.policy.review_threshold,
                ),
                "matcher": type(self.fuzzy_matcher).__name__,
            },
            sort_keys=True,
        )
        effective_lexicon: dict[str, str] = dict(DEFAULT_ACCOUNTING_SYNONYMS)
        if self.lexicon_overrides:
            effective_lexicon.update(self.lexicon_overrides)
        lexicon_blob = json.dumps(effective_lexicon, sort_keys=True)
        data_blob = hashlib.sha256(
            pd.util.hash_pandas_object(source, index=False).to_numpy().tobytes()
            + pd.util.hash_pandas_object(target, index=False).to_numpy().tobytes()
        ).hexdigest()
        deps_blob = json.dumps(
            {
                "pandas": pd.__version__,
                "finance": _pkg_version("fuzzymatch-finance"),
                "core": fuzzymatch.__version__,
            },
            sort_keys=True,
        )
        return {
            "config_hash": hashlib.sha256(config_blob.encode()).hexdigest()[:16],
            "lexicon_hash": hashlib.sha256(lexicon_blob.encode()).hexdigest()[:16],
            "data_hash": data_blob[:16],
            "code_version": (
                f"fuzzymatch-finance@{_pkg_version('fuzzymatch-finance')}"
                + f"+fuzzymatch@{fuzzymatch.__version__}"
            ),
            "deps_fingerprint": hashlib.sha256(deps_blob.encode()).hexdigest()[:16],
            "finance_version": _pkg_version("fuzzymatch-finance"),
            "fuzzymatch_version": fuzzymatch.__version__,
            "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            "rows_source": len(source),
        }

    def _run_fuzzy_fallback(
        self,
        matcher: Matcher,
        source: pd.DataFrame,
        target: pd.DataFrame,
        unmatched: list[tuple[Hashable, str]],
    ) -> tuple[
        list[dict[str, object]],
        list[dict[str, object]],
        list[dict[str, object]],
        list[AuditEvent],
    ]:
        """Fuzzy-match unresolved rows against target and route by policy.

        Returns four lists: approved rows, review rows, evidence rows, and
        audit events produced by the fuzzy stage.
        """
        if not matcher.scoring_fields:
            matcher.scoring_fields = [self.fuzzy_scoring_field]

        candidates = source.loc[[label for label, _ in unmatched]].reset_index(drop=True)

        # Audit F2: curated overrides must influence scoring, not just the
        # manifest hash. Effective map = defaults + overrides, applied to the
        # scoring columns of the fit copies only (inputs never mutated).
        if self.lexicon_overrides:
            effective: dict[str, str] = dict(DEFAULT_ACCOUNTING_SYNONYMS)
            effective.update(self.lexicon_overrides)
            synonym_step = replace_words(effective)
            candidates = candidates.copy()
            target = target.copy()
            for field in matcher.scoring_fields:
                if field in candidates.columns:
                    candidates[field] = synonym_step(candidates[field].astype(str))
                if field in target.columns:
                    target[field] = synonym_step(target[field].astype(str))

        matcher.fit(candidates, target)
        decisions = matcher.decisions()

        best_by_left: dict[object, MatchDecision] = {}
        support: dict[object, int] = {}
        for decision in decisions:
            incumbent = best_by_left.get(decision.left_id)
            if incumbent is None or decision.score > incumbent.score:
                best_by_left[decision.left_id] = decision
            support[decision.left_id] = support.get(decision.left_id, 0) + 1

        approved: list[dict[str, object]] = []
        review: list[dict[str, object]] = []
        evidence: list[dict[str, object]] = []
        events: list[AuditEvent] = []

        for offset in range(len(candidates)):
            source_account = unmatched[offset][1]
            candidate_decision = best_by_left.get(offset)

            if candidate_decision is None:
                review.append(
                    {
                        "source_local_account": source_account,
                        "decision_status": "needs_review",
                        "reason_codes": ["fuzzy_no_candidate"],
                    }
                )
                events.append(
                    AuditEvent(
                        event_type="review_queue_item_created",
                        payload={
                            "source_local_account": source_account,
                            "decision_status": "needs_review",
                            "reason_codes": ["fuzzy_no_candidate"],
                        },
                    )
                )
                continue

            right_position = cast(int, candidate_decision.right_id)
            target_account = str(target.iloc[right_position]["target_account"])
            score_f = float(candidate_decision.score)
            band = self.policy.classify_score(score_f)
            band, guard_reasons = self.policy.route(
                score=score_f,
                band=band,
                n_candidates=support.get(candidate_decision.left_id, 0),
                max_score=score_f,
            )
            threshold_band = candidate_decision.explanation.threshold_band

            evidence.append(
                {
                    "source_local_account": source_account,
                    "target_account": target_account,
                    "fuzzy_score": float(candidate_decision.score),
                    "threshold_band": threshold_band,
                    "policy_status": band,
                    "scoring_method": matcher.scoring_method,
                    "blocking_rules": "|".join(candidate_decision.explanation.blocking_rules),
                }
            )

            if band == "auto_mapped":
                approved.append(
                    {
                        "source_local_account": source_account,
                        "target_account": target_account,
                        "decision_status": "fuzzy_auto_mapped",
                        "reason_codes": ["fuzzy_fallback_auto_accept"],
                    }
                )
                events.append(
                    AuditEvent(
                        event_type="fuzzy_mapping_applied",
                        payload={
                            "source_local_account": source_account,
                            "target_account": target_account,
                            "fuzzy_score": float(candidate_decision.score),
                            "threshold_band": threshold_band,
                        },
                    )
                )
            else:
                reason_code = (
                    "fuzzy_fallback_needs_review"
                    if band == "needs_review"
                    else "fuzzy_fallback_low_confidence"
                )
                review.append(
                    {
                        "source_local_account": source_account,
                        "decision_status": "needs_review",
                        "reason_codes": [reason_code, *(guard_reasons or [])],
                        "fuzzy_score": score_f,
                        "suggested_target": target_account,
                    }
                )
                events.append(
                    AuditEvent(
                        event_type="review_queue_item_created",
                        payload={
                            "source_local_account": source_account,
                            "target_account": target_account,
                            "fuzzy_score": float(candidate_decision.score),
                            "policy_status": band,
                            "reason_codes": [reason_code],
                        },
                    )
                )

        return approved, review, evidence, events
