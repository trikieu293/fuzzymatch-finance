"""v1.2 Phase B contracts: Cohen's kappa reviewer-agreement tracking."""

from __future__ import annotations

from typing import ClassVar

import pytest

from fuzzymatch_finance.agreement import cohen_kappa, gold_set_report


class TestCohenKappa:
    def test_perfect_agreement(self) -> None:
        assert cohen_kappa(["accept"] * 5, ["accept"] * 5) == 1.0

    def test_known_value(self) -> None:
        # a=[acc,acc,rej,rej], b=[acc,rej,acc,rej]:
        # Po=0.5, Pe=0.5 -> kappa = 0 (chance-level agreement)
        a = ["accept", "accept", "reject", "reject"]
        b = ["accept", "reject", "accept", "reject"]
        assert cohen_kappa(a, b) == pytest.approx(0.0)

    def test_positive_kappa_example(self) -> None:
        # Balanced marginals (10/10 each), Po = 0.8, Pe = 0.5 -> kappa = 0.6
        a = ["accept"] * 10 + ["reject"] * 10
        b = ["accept"] * 8 + ["reject"] * 2 + ["accept"] * 2 + ["reject"] * 8
        assert cohen_kappa(a, b) == pytest.approx(0.6)

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            cohen_kappa(["accept"], [])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cohen_kappa([], [])


class TestGoldSetReport:
    GOLD: ClassVar[dict[str, str]] = {
        "acct_a": "accept",
        "acct_b": "reject",
        "acct_c": "accept",
    }

    def test_full_overlap(self) -> None:
        report = gold_set_report(self.GOLD, dict(self.GOLD))
        assert report["kappa"] == 1.0
        assert report["n"] == 3
        assert report["agree_rate"] == 1.0

    def test_partial_overlap_counts_intersection(self) -> None:
        decisions = {"acct_a": "accept", "acct_b": "accept", "zzz": "accept"}
        report = gold_set_report(self.GOLD, decisions)
        assert report["n"] == 2
        assert report["agree_rate"] == pytest.approx(0.5)
        assert -1.0 <= report["kappa"] <= 1.0

    def test_no_overlap_raises(self) -> None:
        with pytest.raises(ValueError, match="overlap"):
            gold_set_report(self.GOLD, {"zzz": "accept"})
