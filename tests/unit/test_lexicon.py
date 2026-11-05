"""Accounting synonym lexicon contracts (B2)."""

from __future__ import annotations

import pandas as pd

from fuzzymatch_finance import DEFAULT_ACCOUNTING_SYNONYMS, accounting_normalizer


class TestLexicon:
    def test_core_pairs_present(self) -> None:
        assert DEFAULT_ACCOUNTING_SYNONYMS["debtors"] == "receivables"
        assert DEFAULT_ACCOUNTING_SYNONYMS["stock"] == "inventory"
        assert DEFAULT_ACCOUNTING_SYNONYMS["turnover"] == "revenue"

    def test_normalizer_maps_vocabulary(self) -> None:
        normalizer = accounting_normalizer()
        frame = pd.DataFrame({"name": ["Sundry Debtors", "Stock In Hand", "Turnover Account"]})
        out = normalizer.transform(frame, ["name"])
        assert out["name"].tolist() == [
            "trade receivables",
            "inventory",  # 'stock in hand' phrase-mapped wholesale
            "revenue account",
        ]

    def test_normalizer_still_cleans_whitespace(self) -> None:
        normalizer = accounting_normalizer()
        frame = pd.DataFrame({"name": ["  Cash   at   Bank  "]})
        out = normalizer.transform(frame, ["name"])
        assert out["name"].iloc[0] == "cash at bank"
