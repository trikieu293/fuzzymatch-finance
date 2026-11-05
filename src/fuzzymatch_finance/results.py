"""Finance mapping result containers and exports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .policy import MappingPolicy


@dataclass(frozen=True)
class MappingResult:
    """Result of a finance mapping job."""

    approved_mappings: pd.DataFrame
    review_queue: pd.DataFrame
    evidence: pd.DataFrame
    audit_log: str
    manifest: dict[str, Any] = field(default_factory=dict)

    def prioritized_review(self, policy: MappingPolicy) -> pd.DataFrame:
        """Review queue ordered by closeness to the policy boundaries.

        Rows whose ``fuzzy_score`` sits nearest a decision boundary
        (``auto_accept_threshold`` or ``review_threshold``) come first —
        resolving them yields the most training signal. Rows without a
        score keep their original relative order and go last.
        """
        queue = self.review_queue.copy()
        if queue.empty or "fuzzy_score" not in queue.columns:
            return queue
        boundaries = (policy.auto_accept_threshold, policy.review_threshold)

        def _distance(score: float) -> float:
            return min(abs(score - b) for b in boundaries)

        scored_mask = queue["fuzzy_score"].notna()
        scored = queue[scored_mask].copy()
        scored["_boundary_distance"] = scored["fuzzy_score"].map(_distance)
        scored = scored.sort_values("_boundary_distance", kind="stable")
        scored = scored.drop(columns=["_boundary_distance"])
        unscored = queue[~scored_mask]
        return pd.concat([scored, unscored], ignore_index=True)

    def export_review_queue(
        self,
        policy: MappingPolicy,
        path: str,
        exploration_rate: float = 0.10,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Write the prioritized review queue with an exploration slice flagged.

        A deterministic random fraction (``exploration_rate``) of scored rows
        is flagged ``exploration=True`` so discovery is not bounded by current
        model beliefs. Returns the exported frame.
        """
        import numpy as np

        exported = self.prioritized_review(policy).copy()
        n = len(exported)
        flags = np.zeros(n, dtype=bool)
        if n and exploration_rate > 0:
            rng = np.random.default_rng(seed)
            n_explore = round(n * exploration_rate)
            if n_explore:
                flags[rng.choice(n, size=n_explore, replace=False)] = True
        exported["exploration"] = flags
        exported.to_csv(path, index=False)
        return exported

    def approved_mappings_dataframe(self) -> pd.DataFrame:
        """Return a copy of the approved mappings."""
        return self.approved_mappings.copy()

    def review_queue_dataframe(self) -> pd.DataFrame:
        """Return a copy of the review queue."""
        return self.review_queue.copy()

    def evidence_dataframe(self) -> pd.DataFrame:
        """Return a copy of the evidence table."""
        return self.evidence.copy()

    def audit_log_jsonl(self) -> str:
        """Return the structured JSONL audit log."""
        return self.audit_log

    def to_approved_mappings_csv(self, path: str) -> None:
        """Write approved mappings to CSV."""
        self.approved_mappings.to_csv(path, index=False)

    def to_review_queue_csv(self, path: str) -> None:
        """Write review queue to CSV."""
        self.review_queue.to_csv(path, index=False)

    def to_evidence_csv(self, path: str) -> None:
        """Write evidence table to CSV."""
        self.evidence.to_csv(path, index=False)

    def to_audit_jsonl(self, path: str) -> None:
        """Write audit log to a JSONL file."""
        with open(path, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(self.audit_log)
