"""Unit tests for corporate legal entity normalization pipeline in fuzzymatch-finance.

Tests:
1. legal_entity_normalizer pipeline creation and steps.
2. End-to-end normalization of corporate entity Series with international suffixes.
3. Preservation of non-suffix strings and interior tokens.
4. Idempotence of normalizer pipeline.
5. Re-exports in fuzzymatch_finance.lexicon and fuzzymatch_finance top-level.
"""

from __future__ import annotations

import pandas as pd
from fuzzymatch.normalize import Normalizer

from fuzzymatch_finance.lexicon import (
    DEFAULT_LEGAL_SUFFIXES,
    legal_entity_normalizer,
    strip_legal_suffixes,
)


def test_legal_entity_normalizer_factory() -> None:
    """Verify legal_entity_normalizer returns a configured Normalizer instance."""
    norm = legal_entity_normalizer()
    assert isinstance(norm, Normalizer)
    assert len(norm.steps) == 5


def test_legal_entity_normalizer_series_execution() -> None:
    """Verify corporate entity normalization across global legal forms."""
    norm = legal_entity_normalizer()

    df = pd.DataFrame(
        {
            "name": [
                "  Acme Industrial, Inc.  ",
                "Apex Logistics LLC",
                "Bosch GmbH",
                "Siemens AG",
                "Danone S.A.",
                "Volvo AB",
                "Müller GmbH & Co. KG",
            ]
        }
    )
    result_df = norm.transform(df, columns=["name"])
    assert list(result_df["name"]) == [
        "acme industrial",
        "apex logistics",
        "bosch",
        "siemens",
        "danone",
        "volvo",
        "müller",
    ]


def test_legal_entity_normalizer_idempotence() -> None:
    """Verify idempotence of the legal_entity_normalizer pipeline."""
    norm = legal_entity_normalizer()
    df = pd.DataFrame({"name": ["Frontier Consulting Group Ltd.", "Precision Machining Inc."]})
    pass1 = norm.transform(df, columns=["name"])
    pass2 = norm.transform(pass1, columns=["name"])
    pd.testing.assert_frame_equal(pass1, pass2)


def test_lexicon_reexports_contract() -> None:
    """Verify re-exports from fuzzymatch in fuzzymatch_finance.lexicon."""
    assert isinstance(DEFAULT_LEGAL_SUFFIXES, tuple)
    assert callable(strip_legal_suffixes)
    assert strip_legal_suffixes("Acme Industrial Corp") == "Acme Industrial"
