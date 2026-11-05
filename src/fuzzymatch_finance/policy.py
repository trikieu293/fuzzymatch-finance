"""Finance mapping policy objects."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MappingPolicy:
    """Controls threshold-based auto-accept, review, and reject behaviour.

    The thin-evidence guardrail requires at least ``min_auto_accept_evidence``
    candidate pairs for a source row (and a best score of at least half the
    auto-accept threshold) before an ``auto_mapped`` decision is allowed.
    Below that floor, decisions downgrade to review: small or noisy candidate
    pools must not produce confident automatic mappings.
    """

    auto_accept_threshold: float
    review_threshold: float
    min_auto_accept_evidence: int = 20

    def __post_init__(self) -> None:
        if self.auto_accept_threshold < self.review_threshold:
            raise ValueError("auto_accept_threshold must be >= review_threshold")
        if self.min_auto_accept_evidence < 0:
            raise ValueError("min_auto_accept_evidence must be >= 0")

    def classify_score(self, score: float) -> str:
        """Map a confidence score to a status label."""
        if score >= self.auto_accept_threshold:
            return "auto_mapped"
        if score >= self.review_threshold:
            return "needs_review"
        return "unmapped"

    def route(
        self,
        score: float,
        band: str,
        n_candidates: int,
        max_score: float,
    ) -> tuple[str, list[str] | None]:
        """Route one decision, applying the thin-evidence guardrail.

        Returns ``(status, reason_codes)``; reason codes are None unless the
        guardrail downgraded an auto-mapping.
        """
        if band != "auto_mapped":
            return band, None
        evidence_ok = (
            n_candidates >= self.min_auto_accept_evidence
            and max_score >= self.auto_accept_threshold / 2
        )
        if not evidence_ok and self.min_auto_accept_evidence > 0:
            return "needs_review", ["thin_evidence"]
        return "auto_mapped", None
