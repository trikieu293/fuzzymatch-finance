"""fuzzymatch-finance public API."""

from __future__ import annotations

__version__ = "1.0.0"

from .cli import (
    FinanceMatchResults,
    FinanceSummaryMetrics,
    auto_match_finance,
)
from .escalation import LLMSuggestion
from .identifiers import (
    validate_cusip,
    validate_cusip_check_digit,
    validate_iban,
    validate_lei,
)
from .jobs import FinanceMappingJob
from .label_store import (
    LabeledPair,
    LabeledPairStore,
    LabelStore,
    capture_labels,
    synthesize_rules_from_reviews,
)
from .lexicon import (
    DEFAULT_ACCOUNTING_SYNONYMS,
    DEFAULT_LEGAL_SUFFIXES,
    accounting_normalizer,
    legal_entity_normalizer,
    strip_legal_suffixes,
)
from .logging import AuditEvent, AuditLog
from .policy import MappingPolicy
from .results import MappingResult
from .review_xlsx import (
    ReviewDecisions,
    export_review_xlsx,
    import_review_xlsx,
)
from .rules import MappingRule, MappingRuleSet
from .scoring import (
    FinancialCompositeScorer,
    amount_proximity_score,
    date_window_score,
)

__all__ = [
    "DEFAULT_ACCOUNTING_SYNONYMS",
    "DEFAULT_LEGAL_SUFFIXES",
    "AuditEvent",
    "AuditLog",
    "FinanceMappingJob",
    "FinanceMatchResults",
    "FinanceSummaryMetrics",
    "FinancialCompositeScorer",
    "LLMSuggestion",
    "LabelStore",
    "LabeledPair",
    "LabeledPairStore",
    "MappingPolicy",
    "MappingResult",
    "MappingRule",
    "MappingRuleSet",
    "ReviewDecisions",
    "accounting_normalizer",
    "amount_proximity_score",
    "auto_match_finance",
    "capture_labels",
    "date_window_score",
    "export_review_xlsx",
    "import_review_xlsx",
    "legal_entity_normalizer",
    "strip_legal_suffixes",
    "synthesize_rules_from_reviews",
    "validate_cusip",
    "validate_cusip_check_digit",
    "validate_iban",
    "validate_lei",
]
