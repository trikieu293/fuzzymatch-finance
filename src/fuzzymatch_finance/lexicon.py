"""Accounting-domain synonym lexicon for chart-of-account matching.

Regional accounting vocabularies describe the same concepts with different
words (UK "Debtors" vs US "Receivables"). Feeding this map through
:func:`fuzzymatch.normalize.steps.replace_words` inside a Normalizer aligns
the vocabulary BEFORE similarity scoring, lifting match quality without any
labels.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from fuzzymatch.normalize import Normalizer
from fuzzymatch.normalize.legal import DEFAULT_LEGAL_SUFFIXES, strip_legal_suffixes
from fuzzymatch.normalize.steps import (
    casefold,
    replace_words,
    strip_whitespace,
    unicode_normalize,
)

__all__ = [
    "DEFAULT_ACCOUNTING_SYNONYMS",
    "DEFAULT_LEGAL_SUFFIXES",
    "accounting_normalizer",
    "legal_entity_normalizer",
    "load_lexicon_overrides",
    "strip_legal_suffixes",
    "validate_lexicon_map",
]

DEFAULT_ACCOUNTING_SYNONYMS: dict[str, str] = {
    # receivables family
    "debtors": "receivables",
    "sundry debtors": "trade receivables",
    "debtor": "receivable",
    # payables family
    "creditors": "payables",
    "sundry creditors": "trade payables",
    "creditor": "payable",
    # inventory family
    "stock in hand": "inventory",
    "stock": "inventory",
    "stocks": "inventory",
    # revenue family
    "turnover": "revenue",
    "sales account": "revenue",
    # expenses / assets misc
    "p and l": "profit loss",
    "profit and loss": "profit loss",
    "capital account": "equity",
    "fixed assets": "property plant equipment",
    "prepaid expenses": "prepayments",
}


def accounting_normalizer() -> Normalizer:
    """Normalizer tuned for chart-of-account text (English charts)."""
    return Normalizer(
        [
            unicode_normalize("NFC"),
            casefold(),
            strip_whitespace(),
            replace_words(DEFAULT_ACCOUNTING_SYNONYMS),
        ]
    )


def legal_entity_normalizer() -> Normalizer:
    """Normalizer tuned for corporate legal entity names across global jurisdictions."""
    return Normalizer(
        [
            unicode_normalize("NFC"),
            casefold(),
            strip_whitespace(),
            strip_legal_suffixes(),
            strip_whitespace(),
        ]
    )


def validate_lexicon_map(
    mapping: dict[str, str],
    defaults: dict[str, str] | None = None,
) -> None:
    """Reject lexicon maps that chain or collide with default keys.

    Raises:
        ValueError: If any value is itself a key in ``mapping`` (chaining),
            or if a key collides with a *default* mapping's key.
    """
    keys = set(mapping)
    for src, tgt in mapping.items():
        if tgt in keys:
            raise ValueError(f"chain detected: {src!r} -> {tgt!r} is also a key")
    if defaults:
        for src, tgt in mapping.items():
            if src in defaults and defaults[src] != tgt:
                raise ValueError(f"override {src!r} conflicts with a default mapping")


def load_lexicon_overrides(
    path: Path | str,
    defaults: dict[str, str] | None = None,
) -> dict[str, str]:
    """Load approved synonym overrides from a curation CSV.

    CSV columns: ``source,target,status`` where status ∈
    {approved, pending, rejected}. Only approved rows are returned;
    pending/rejected never reach the runtime lexicon. The result map is
    validated against :func:`validate_lexicon_map` before returning.
    """
    frame = pd.read_csv(Path(path))
    required = {"source", "target", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"curation CSV missing columns: {sorted(missing)}")
    approved = frame[frame["status"] == "approved"]
    mapping = {str(r.source): str(r.target) for r in approved.itertuples(index=False)}
    validate_lexicon_map(mapping, defaults=defaults)
    return mapping
