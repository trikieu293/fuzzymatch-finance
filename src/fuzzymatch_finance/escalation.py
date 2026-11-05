"""Optional LLM/SLM suggestion schema for ambiguous finance mappings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMSuggestion:
    """An AI-assisted suggestion for an unresolved mapping."""

    suggested_target_id: str
    confidence_label: str
    rationale: str
    fields_used: list[str]
    caveats: list[str]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation."""
        return {
            "suggested_target_id": self.suggested_target_id,
            "confidence_label": self.confidence_label,
            "rationale": self.rationale,
            "fields_used": list(self.fields_used),
            "caveats": list(self.caveats),
        }
