from __future__ import annotations

from fuzzymatch_finance.escalation import LLMSuggestion


def test_llm_suggestion_serializes():
    suggestion = LLMSuggestion(
        suggested_target_id="group_gl_610000",
        confidence_label="low",
        rationale="Account name and cost-center context suggest OPEX mapping.",
        fields_used=["name", "company_code"],
        caveats=["manual approval required"],
    )

    assert suggestion.to_dict()["suggested_target_id"] == "group_gl_610000"
    assert suggestion.to_dict()["confidence_label"] == "low"
    assert suggestion.to_dict()["rationale"].startswith("Account name")
