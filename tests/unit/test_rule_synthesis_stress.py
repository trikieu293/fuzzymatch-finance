"""Adversarial stress-testing suite for Milestone 4: MappingRule, MappingRuleSet, serialization, and rule synthesis."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch_finance.label_store import (
    LabeledPairStore,
    synthesize_rules_from_reviews,
)
from fuzzymatch_finance.rules import (
    MappingRule,
    MappingRuleSet,
)

# =====================================================================
# 1. CONFLICTING DECISIONS STRESS TESTS
# =====================================================================


class TestConflictingDecisions:
    """Stress test conflict resolution in synthesize_rules_from_reviews."""

    def test_intra_batch_accept_then_reject(self) -> None:
        """When same pair is accepted then rejected in same batch, reject must win (keep='last')."""
        df = pd.DataFrame(
            {
                "source_id": ["S100", "S100"],
                "target_id": ["T200", "T200"],
                "decision": ["accept", "reject"],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        assert len(rules) == 0, "Rejected pair should not synthesize any rules"
        assert len(store) == 1, "Store should record exactly one pair"
        pair = store.all()[0]
        assert pair.left_key == "S100"
        assert pair.right_key == "T200"
        assert pair.label == 0, "Store label must be 0 (rejected)"

    def test_intra_batch_reject_then_accept(self) -> None:
        """When same pair is rejected then accepted in same batch, accept must win (keep='last')."""
        df = pd.DataFrame(
            {
                "source_id": ["S100", "S100"],
                "target_id": ["T200", "T200"],
                "decision": ["reject", "accept"],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        assert len(rules) == 1, "Accepted pair must synthesize exactly one rule"
        assert len(store) == 1, "Store should record exactly one pair"
        assert rules.match("S100") == "T200"
        assert store.all()[0].label == 1

    def test_one_source_multiple_targets_accepted(self) -> None:
        """When one source is accepted against two targets, latest accepted target wins for rule synthesis."""
        df = pd.DataFrame(
            {
                "source_id": ["VEND-1", "VEND-1"],
                "target_id": ["ERP-A", "ERP-B"],
                "decision": ["accept", "accept"],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        # Both pairs recorded in store as valid audit evidence
        assert len(store) == 2
        # But deterministic rule maps source to the latest accepted target
        assert len(rules) == 1
        assert rules.match("VEND-1") == "ERP-B"

    def test_interleaved_conflicts_deterministic(self) -> None:
        """Interleaved multi-entity decisions resolve deterministically without crash."""
        df = pd.DataFrame(
            {
                "source_id": ["S1", "S2", "S1", "S3", "S2", "S1"],
                "target_id": ["T1", "T2", "T1", "T3", "T2", "T1"],
                "decision": ["accept", "reject", "reject", "accept", "accept", "accept"],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        # S1->T1 ended with 'accept'
        # S2->T2 ended with 'accept'
        # S3->T3 ended with 'accept'
        assert len(rules) == 3
        assert rules.match("S1") == "T1"
        assert rules.match("S2") == "T2"
        assert rules.match("S3") == "T3"
        assert len(store) == 3
        assert all(p.label == 1 for p in store.all())

    def test_cross_batch_conflict_resolution(self, tmp_path: Path) -> None:
        """Overriding a decision in a later batch updates LabelStore in place."""
        store_path = tmp_path / "persistent_labels.csv"
        store = LabeledPairStore(store_path)

        # Batch 1: S1 -> T1 accepted
        b1 = pd.DataFrame({"source_id": ["S1"], "target_id": ["T1"], "decision": ["accept"]})
        r1, store = synthesize_rules_from_reviews(b1, label_store=store)
        assert len(r1) == 1
        assert len(store) == 1
        assert store.all()[0].label == 1

        # Batch 2: Human auditor reverses decision: S1 -> T1 rejected
        b2 = pd.DataFrame({"source_id": ["S1"], "target_id": ["T1"], "decision": ["reject"]})
        r2, store = synthesize_rules_from_reviews(b2, label_store=store)
        assert len(r2) == 0
        assert len(store) == 1, "Store should not duplicate the pair; should overwrite"
        assert store.all()[0].label == 0, "Store label should now be 0"

        # Batch 3: Auditor changes mind again: S1 -> T1 accepted
        b3 = pd.DataFrame({"source_id": ["S1"], "target_id": ["T1"], "decision": ["accept"]})
        r3, store = synthesize_rules_from_reviews(b3, label_store=store)
        assert len(r3) == 1
        assert len(store) == 1
        assert store.all()[0].label == 1

    @pytest.mark.parametrize(
        "token,expected_label",
        [
            ("accept", 1),
            ("ACCEPT", 1),
            ("  accepted  ", 1),
            ("Approve", 1),
            ("APPROVED", 1),
            ("1", 1),
            ("true", 1),
            ("TRUE", 1),
            ("reject", 0),
            ("REJECT", 0),
            ("  rejected  ", 0),
            ("declined", 0),
            ("0", 0),
            ("false", 0),
            ("FALSE", 0),
        ],
    )
    def test_decision_token_normalization(self, token: str, expected_label: int) -> None:
        df = pd.DataFrame(
            {
                "source_id": ["S1"],
                "target_id": ["T1"],
                "decision": [token],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        assert len(store) == 1
        assert store.all()[0].label == expected_label
        if expected_label == 1:
            assert len(rules) == 1
        else:
            assert len(rules) == 0

    def test_ignorable_and_nan_decisions(self) -> None:
        df = pd.DataFrame(
            {
                "source_id": ["S1", "S2", "S3", "S4", "S5", None, "   "],
                "target_id": ["T1", "T2", "T3", "T4", "T5", "T6", "T7"],
                "decision": [
                    "pending",
                    "investigate",
                    "skip",
                    None,
                    "unknown_token",
                    "accept",
                    "accept",
                ],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        assert len(rules) == 0
        assert len(store) == 0

    def test_special_characters_in_identifiers(self) -> None:
        """Identifiers with commas, quotes, and newlines."""
        df = pd.DataFrame(
            {
                "source_id": ["V,100", 'V"200"', "V\n300"],
                "target_id": ["T,100", 'T"200"', "T\n300"],
                "decision": ["accept", "accept", "accept"],
            }
        )
        rules, store = synthesize_rules_from_reviews(df)
        assert len(rules) == 3
        assert len(store) == 3

        # Serialization roundtrip with these special identifiers
        json_str = rules.to_json()
        loaded_j = MappingRuleSet.from_json(json_str)
        assert len(loaded_j) == 3

        csv_str = rules.to_csv()
        loaded_c = MappingRuleSet.from_csv(csv_str)
        assert len(loaded_c) == 3


# =====================================================================
# 2. SERIALIZATION ROUNDTRIP STRESS TESTS
# =====================================================================


class TestSerializationRoundtrips:
    """Stress test JSON and CSV roundtrip fidelity across edge-case data."""

    def test_json_roundtrip_with_complex_and_unicode_data(self, tmp_path: Path) -> None:
        r1 = MappingRule(
            rule_id="rule_ünicöde_🎉",
            description='Rule with "quotes", \nnewlines, and \\backslashes',
            priority=1,
            condition={"vendor": "Bayerische Motoren Werke AG", "code": 100, "active": True},
            target={"target_erp": "BMW-DE", "currency": "EUR", "tax_rate": 0.19},
        )
        r2 = MappingRule(
            rule_id="rule_asian_文字",
            description="Japanese supplier mapping",
            priority=5,
            condition={"supplier": "ソニー株式会社", "prefecture": "東京都"},
            target={"target_id": "SONY-JP", "tags": ["electronics", "japan"]},
        )
        ruleset = MappingRuleSet([r2, r1])

        # String roundtrip
        json_str = ruleset.to_json()
        restored_str = MappingRuleSet.from_json(json_str)
        assert len(restored_str) == 2
        # Priority order preserved
        assert restored_str[0].rule_id == "rule_ünicöde_🎉"
        assert restored_str[0].condition == r1.condition
        assert restored_str[0].target == r1.target
        assert restored_str[0].description == r1.description
        assert restored_str[1].rule_id == "rule_asian_文字"
        assert restored_str[1].condition == r2.condition
        assert restored_str[1].target == r2.target

        # File roundtrip
        file_path = tmp_path / "nested" / "rules.json"
        ruleset.to_json(file_path)
        assert file_path.exists()
        restored_file = MappingRuleSet.from_json(file_path)
        assert len(restored_file) == 2
        assert restored_file[0].rule_id == restored_str[0].rule_id

    def test_csv_roundtrip_with_commas_and_quotes(self, tmp_path: Path) -> None:
        r1 = MappingRule(
            rule_id="r,with,commas",
            description='Description with "nested quotes" and ,commas,',
            priority=10,
            condition={"address": "123 Main St, Suite 400", "country": "US"},
            target={"erp_id": "VEND,100", "target_id": "T1"},
        )
        r2 = MappingRule(
            rule_id="r_normal",
            description="Simple rule",
            priority=20,
            condition={"vendor_id": "V2"},
            target={"target_id": "T2"},
        )
        ruleset = MappingRuleSet([r1, r2])

        # String roundtrip
        csv_str = ruleset.to_csv()
        restored_str = MappingRuleSet.from_csv(csv_str)
        assert len(restored_str) == 2
        assert restored_str[0].rule_id == "r,with,commas"
        assert restored_str[0].condition == r1.condition
        assert restored_str[0].target == r1.target
        assert restored_str[0].description == r1.description

        # File roundtrip
        file_path = tmp_path / "nested" / "rules.csv"
        ruleset.to_csv(file_path)
        assert file_path.exists()
        restored_file = MappingRuleSet.from_csv(file_path)
        assert len(restored_file) == 2
        assert restored_file[0].condition == r1.condition

    def test_empty_ruleset_serialization(self, tmp_path: Path) -> None:
        empty_set = MappingRuleSet([])
        assert len(empty_set) == 0

        # JSON
        json_str = empty_set.to_json()
        loaded_json = MappingRuleSet.from_json(json_str)
        assert len(loaded_json) == 0

        # CSV
        csv_str = empty_set.to_csv()
        loaded_csv = MappingRuleSet.from_csv(csv_str)
        assert len(loaded_csv) == 0

    def test_malformed_json_handling(self) -> None:
        # Invalid syntax
        with pytest.raises(json.JSONDecodeError):
            MappingRuleSet.from_json("{broken json")

        # Missing rules key in dict
        with pytest.raises(ValueError, match="Invalid JSON format"):
            MappingRuleSet.from_json('{"other_key": 123}')

        # Primitive JSON types
        with pytest.raises(ValueError, match="Invalid JSON format"):
            MappingRuleSet.from_json("42")

    def test_malformed_csv_handling(self) -> None:
        # Missing required column
        with pytest.raises(ValueError, match="CSV missing required columns"):
            MappingRuleSet.from_csv("rule_id,priority\nr1,10\n")

        # Missing target column
        with pytest.raises(ValueError, match="CSV missing required columns"):
            MappingRuleSet.from_csv("rule_id,condition\nr1,{}\n")


# =====================================================================
# 3. SCALAR & SERIES MATCHING ACROSS ENTITY TYPES
# =====================================================================


class TestScalarAndSeriesMatching:
    """Stress test scalar match() and Series applies_to() lookups."""

    def test_vendor_matching_arbitrary_keys(self) -> None:
        rule = MappingRule(
            rule_id="vend_rule_1",
            description="Vendor rule",
            priority=10,
            condition={"vendor_id": "V-999", "country": "DE"},
            target={"target_vendor_id": "ERP-VEND-999"},
        )
        ruleset = MappingRuleSet([rule])

        # Scalar matching directly on condition values
        assert ruleset.match("V-999") == "ERP-VEND-999"
        assert ruleset.match("DE") == "ERP-VEND-999"
        assert ruleset.match("NONEXISTENT") is None

        # Series matching with exact condition subset
        row_full = pd.Series(
            {
                "vendor_id": "V-999",
                "country": "DE",
                "name": "Bayer AG",
                "spend": 50000.0,
            }
        )
        assert ruleset.match(row_full) == "ERP-VEND-999"

        # Series matching with mismatched country
        row_mismatch = pd.Series({"vendor_id": "V-999", "country": "FR"})
        assert ruleset.match(row_mismatch) is None

        # Series matching with missing condition key
        row_missing = pd.Series({"vendor_id": "V-999"})
        assert ruleset.match(row_missing) is None

    def test_customer_matching(self) -> None:
        rule = MappingRule(
            rule_id="cust_rule",
            description="Customer rule",
            priority=5,
            condition={"customer_num": "C-1234"},
            target={"target_account": "CUST-REC-01"},
        )
        ruleset = MappingRuleSet([rule])
        assert ruleset.match("C-1234") == "CUST-REC-01"
        assert (
            ruleset.match(pd.Series({"customer_num": "C-1234", "city": "Berlin"})) == "CUST-REC-01"
        )

    def test_product_sku_fallback_target(self) -> None:
        """When target does not have target_id or target_erp_id, match returns first value in target."""
        rule = MappingRule(
            rule_id="prod_rule",
            description="Product rule",
            priority=1,
            condition={"sku": "SKU-XYZ"},
            target={"custom_catalog_code": "CAT-8888"},
        )
        ruleset = MappingRuleSet([rule])
        assert ruleset.match("SKU-XYZ") == "CAT-8888"

    def test_synthesized_rule_condition_series_lookup_bug(self) -> None:
        """Verify that synthesize_rules_from_reviews condition supports Series matching on domain rows.

        When input has 'source_vendor_id', the synthesized rule condition is:
        {'source_vendor_id': 'V-100'} without extraneous 'source_id' key.
        MappingRule.applies_to against a Series with 'source_vendor_id' succeeds.
        """
        review_df = pd.DataFrame(
            {
                "source_vendor_id": ["V-100"],
                "target_erp_id": ["E-200"],
                "decision": ["accept"],
            }
        )
        rules, _ = synthesize_rules_from_reviews(
            review_df,
            source_id_col="source_vendor_id",
            target_id_col="target_erp_id",
        )
        assert len(rules) == 1
        assert rules[0].condition == {"source_vendor_id": "V-100"}

        # Scalar match succeeds because it checks any(str(val) == query) across condition values
        assert rules.match("V-100") == "E-200"

        # Series matching on the vendor row succeeds
        source_row = pd.Series({"source_vendor_id": "V-100", "vendor_name": "Test Co"})
        assert rules.match(source_row) == "E-200"

        # Also matches via standard alias 'source_id'
        assert rules.match(pd.Series({"source_id": "V-100"})) == "E-200"

        # Contrast with standard 'source_id' where condition has only one key
        review_df_standard = pd.DataFrame(
            {
                "source_id": ["V-100"],
                "target_id": ["E-200"],
                "decision": ["accept"],
            }
        )
        rules_standard, _ = synthesize_rules_from_reviews(review_df_standard)
        assert rules_standard.match(pd.Series({"source_id": "V-100"})) == "E-200"

    def test_target_none_string_bug(self) -> None:
        """Verify that MappingRuleSet.match returns None (not 'None') when target_id is None."""
        r = MappingRule(
            rule_id="r_none",
            description="rule with None target",
            priority=1,
            condition={"code": "XYZ"},
            target={"target_id": None},
        )
        rs = MappingRuleSet([r])
        val = rs.match("XYZ")
        assert val is None


# =====================================================================
# 4. CONTAINER PROTOCOL INVARIANTS
# =====================================================================


class TestContainerProtocol:
    """Stress test MappingRuleSet container invariants."""

    def test_len_iter_getitem_and_sorting(self) -> None:
        r1 = MappingRule("r100", "low priority", 100, {"k": 1}, {"t": 1})
        r2 = MappingRule("r1", "high priority", 1, {"k": 2}, {"t": 2})
        r3 = MappingRule("r50", "medium priority", 50, {"k": 3}, {"t": 3})

        ruleset = MappingRuleSet([r1, r2, r3])

        # len()
        assert len(ruleset) == 3

        # Iteration yields sorted by priority
        priorities = [r.priority for r in ruleset]
        assert priorities == [1, 50, 100]

        # __getitem__ positive and negative indexing
        assert ruleset[0].rule_id == "r1"
        assert ruleset[1].rule_id == "r50"
        assert ruleset[2].rule_id == "r100"
        assert ruleset[-1].rule_id == "r100"

        # IndexError on out of bounds
        with pytest.raises(IndexError):
            _ = ruleset[10]

    def test_immutability(self) -> None:
        rule = MappingRule("r1", "test", 10, {"a": 1}, {"b": 2})
        ruleset = MappingRuleSet([rule])

        with pytest.raises(FrozenInstanceError):
            ruleset.rules = []  # type: ignore[misc]

        with pytest.raises(FrozenInstanceError):
            rule.priority = 20  # type: ignore[misc]


# =====================================================================
# 5. LABEL STORE ATOMIC UPDATES & PERSISTENCE
# =====================================================================


class TestLabelStorePersistence:
    """Stress test LabelStore across multiple batches and atomic file writes."""

    def test_multi_batch_active_learning_simulation(self, tmp_path: Path) -> None:
        store_path = tmp_path / "audit_store.csv"
        store = LabeledPairStore(store_path)

        # Batch 1: 30 pairs
        batch1 = pd.DataFrame(
            {
                "source_id": [f"S_{i}" for i in range(30)],
                "target_id": [f"T_{i}" for i in range(30)],
                "decision": ["accept" if i % 2 == 0 else "reject" for i in range(30)],
            }
        )
        rules1, store = synthesize_rules_from_reviews(batch1, label_store=store)
        assert len(rules1) == 15
        assert len(store) == 30
        assert store_path.exists()

        # Batch 2: 30 new pairs
        batch2 = pd.DataFrame(
            {
                "source_id": [f"S_{i}" for i in range(30, 60)],
                "target_id": [f"T_{i}" for i in range(30, 60)],
                "decision": ["accept" for _ in range(30)],
            }
        )
        rules2, store = synthesize_rules_from_reviews(batch2, label_store=store)
        assert len(rules2) == 30
        assert len(store) == 60

        # Batch 3: Correct 10 pairs from Batch 1 (change reject -> accept)
        batch3 = pd.DataFrame(
            {
                "source_id": [f"S_{i}" for i in range(1, 20, 2)],  # odd indices were reject
                "target_id": [f"T_{i}" for i in range(1, 20, 2)],
                "decision": ["accept" for _ in range(10)],
            }
        )
        rules3, store = synthesize_rules_from_reviews(batch3, label_store=store)
        assert len(rules3) == 10
        # Store should NOT increase in length; it should update in place
        assert len(store) == 60, f"Expected 60 total records, found {len(store)}"

        # Verify no .tmp files left in directory
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0, f"Dangling temporary files found: {tmp_files}"

    def test_dataset_tags_partitioning(self, tmp_path: Path) -> None:
        store_path = tmp_path / "tagged_store.csv"
        store = LabeledPairStore(store_path)

        # Record same entity keys under different dataset tags
        store.record("E1", "E2", 1, dataset_tag="vendors")
        store.record("E1", "E2", 0, dataset_tag="customers")

        assert len(store) == 2
        vendor_pairs = store.for_dataset("vendors")
        customer_pairs = store.for_dataset("customers")
        assert len(vendor_pairs) == 1
        assert vendor_pairs[0].label == 1
        assert len(customer_pairs) == 1
        assert customer_pairs[0].label == 0
