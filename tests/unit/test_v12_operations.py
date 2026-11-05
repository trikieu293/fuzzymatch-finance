"""Financial operations contracts: exploration export, curation loader, lexicon_hash."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch_finance import DEFAULT_ACCOUNTING_SYNONYMS
from fuzzymatch_finance.jobs import FinanceMappingJob
from fuzzymatch_finance.lexicon import load_lexicon_overrides
from fuzzymatch_finance.policy import MappingPolicy
from fuzzymatch_finance.results import MappingResult
from fuzzymatch_finance.rules import MappingRuleSet


def _result(n: int = 20) -> MappingResult:
    queue = pd.DataFrame(
        {
            "source_local_account": [f"s{i}" for i in range(n)],
            "fuzzy_score": [i / n for i in range(n)],
        }
    )
    return MappingResult(
        approved_mappings=pd.DataFrame(),
        review_queue=queue,
        evidence=pd.DataFrame(),
        audit_log="",
    )


POLICY = MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.6)


class TestExplorationExport:
    def test_flags_exploration_slice(self, tmp_path: Path) -> None:
        result = _result()
        out = result.export_review_queue(POLICY, tmp_path / "q.csv", exploration_rate=0.25, seed=7)
        assert "exploration" in out.columns
        flagged = int(out["exploration"].sum())
        assert flagged == pytest.approx(len(out) * 0.25, abs=1)

    def test_deterministic_with_seed(self, tmp_path: Path) -> None:
        a = _result().export_review_queue(POLICY, tmp_path / "a.csv", seed=42)
        b = _result().export_review_queue(POLICY, tmp_path / "b.csv", seed=42)
        assert a["source_local_account"].tolist() == b["source_local_account"].tolist()

    def test_writes_csv(self, tmp_path: Path) -> None:
        p = tmp_path / "queue.csv"
        _result().export_review_queue(POLICY, p)
        assert p.exists()


class TestCurationLoader:
    def _write(self, tmp_path, rows):
        p = tmp_path / "overrides.csv"
        pd.DataFrame(rows, columns=["source", "target", "status"]).to_csv(p, index=False)
        return p

    def test_approved_only_returned(self, tmp_path) -> None:
        p = self._write(
            tmp_path,
            [
                ("debtors", "receivables", "approved"),
                ("stock", "inventory", "pending"),
            ],
        )
        assert load_lexicon_overrides(p) == {"debtors": "receivables"}

    def test_rejected_never_applied(self, tmp_path) -> None:
        p = self._write(tmp_path, [("debtors", "payables", "rejected")])
        assert load_lexicon_overrides(p) == {}

    def test_chained_key_rejected(self, tmp_path) -> None:
        p = self._write(
            tmp_path,
            [
                ("debtors", "receivables", "approved"),
                ("receivables", "trade receivables", "approved"),
            ],
        )
        with pytest.raises(ValueError, match="chain"):
            load_lexicon_overrides(p, defaults=DEFAULT_ACCOUNTING_SYNONYMS)

    def test_conflicting_default_key_rejected(self, tmp_path) -> None:
        # [coverage guard] audit F3: defaults-conflict branch (lexicon.py:75-77)
        p = self._write(tmp_path, [("stock", "payables", "approved")])
        with pytest.raises(ValueError, match="conflicts with a default"):
            load_lexicon_overrides(p, defaults=DEFAULT_ACCOUNTING_SYNONYMS)

    def test_value_equal_default_key_allowed(self, tmp_path) -> None:
        # [coverage guard] audit F3: same key+value as default is not a conflict
        p = self._write(tmp_path, [("stock", "inventory", "approved")])
        assert load_lexicon_overrides(p, defaults=DEFAULT_ACCOUNTING_SYNONYMS) == {
            "stock": "inventory"
        }

    def test_missing_columns_rejected(self, tmp_path) -> None:
        # [coverage guard] audit F3: missing-columns ValueError (lexicon.py:97)
        p = tmp_path / "bad.csv"
        pd.DataFrame({"source": ["a"], "target": ["b"]}).to_csv(p, index=False)
        with pytest.raises(ValueError, match="missing columns"):
            load_lexicon_overrides(p)


class TestLexiconHashInManifest:
    def _job(self, overrides=None):
        return FinanceMappingJob(
            rule_set=MappingRuleSet([]),
            policy=MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.6),
            fuzzy_matcher=None,
            lexicon_overrides=overrides,
        )

    def _src_tgt(self):
        src = pd.DataFrame({"local_account": ["a"], "name": ["x"]})
        tgt = pd.DataFrame({"target_account": ["t"], "name": ["y"]})
        return src, tgt

    def test_manifest_contains_five_keys(self) -> None:
        result = self._job().run(*self._src_tgt())
        for key in ("config_hash", "lexicon_hash", "data_hash", "code_version", "deps_fingerprint"):
            assert key in result.manifest

    def test_lexicon_hash_changes_with_overrides(self) -> None:
        r1 = self._job().run(*self._src_tgt())
        r2 = self._job(overrides={"novel token": "unique target"}).run(*self._src_tgt())
        assert r1.manifest["lexicon_hash"] != r2.manifest["lexicon_hash"]


class TestOverridesAffectScoring:
    """Audit F2: lexicon_overrides must influence fuzzy scores, not just hashes."""

    SRC = pd.DataFrame(
        {
            "local_account": [
                "HP Loans",
                "Office Rent",
                "Bank Charges",
                "Travel",
                "Salaries Wages",
            ]
        }
    )
    # "local_account" duplicated in target: blocking needs a shared field,
    # while job output displays "target_account".
    TGT = pd.DataFrame(
        {
            "target_account": [
                "Hire Purchase Loans",
                "Trade Creditors",
                "Payroll Costs",
                "Rent Expense",
            ],
            "local_account": [
                "Hire Purchase Loans",
                "Trade Creditors",
                "Payroll Costs",
                "Rent Expense",
            ],
        }
    )

    def _run(self, overrides: dict[str, str] | None) -> MappingResult:
        from fuzzymatch import FSClassifier, Matcher
        from fuzzymatch.blocking import SortedNeighborhood

        matcher = Matcher(
            link_type="link",
            blocking=SortedNeighborhood(["local_account"], window=5),
            classifier=FSClassifier(),
            scoring_fields=["local_account"],
            term_frequency=True,
        )
        job = FinanceMappingJob(
            rule_set=MappingRuleSet(rules=[]),
            policy=MappingPolicy(auto_accept_threshold=0.9, review_threshold=0.6),
            fuzzy_matcher=matcher,
            fuzzy_scoring_field="local_account",
            lexicon_overrides=overrides,
        )
        return job.run(self.SRC.copy(), self.TGT.copy())

    def test_override_improves_score_for_mapped_synonym(self) -> None:
        base = self._run(None)
        boosted = self._run({"hp": "hire purchase"})
        base_score = float(base.evidence["fuzzy_score"].max())
        boost_score = float(boosted.evidence["fuzzy_score"].max())
        assert boost_score > base_score
        base_hp = base.evidence[base.evidence["source_local_account"] == "HP Loans"]
        boost_hp = boosted.evidence[boosted.evidence["source_local_account"] == "HP Loans"]
        assert float(boost_hp.iloc[0]["fuzzy_score"]) > float(base_hp.iloc[0]["fuzzy_score"])

    def test_override_flips_best_candidate_target(self) -> None:
        base = self._run(None)
        boosted = self._run({"salaries wages": "payroll costs"})

        base_row = base.evidence[base.evidence["source_local_account"] == "Salaries Wages"]
        assert str(base_row.iloc[0]["target_account"]) == "Trade Creditors"

        boosted_row = boosted.evidence[boosted.evidence["source_local_account"] == "Salaries Wages"]
        assert str(boosted_row.iloc[0]["target_account"]) == "Payroll Costs"
