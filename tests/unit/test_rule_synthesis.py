"""Comprehensive unit tests for rule synthesis, MappingRule, and MappingRuleSet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch_finance.label_store import (
    LabeledPairStore,
    LabelStore,
)
from fuzzymatch_finance.label_store import (
    synthesize_rules_from_reviews as synth_from_store,
)
from fuzzymatch_finance.review_xlsx import ReviewDecisions
from fuzzymatch_finance.rules import (
    MappingRule,
    MappingRuleSet,
    RuleResult,
)
from fuzzymatch_finance.rules import (
    synthesize_rules_from_reviews as synth_from_rules,
)


def test_label_store_alias() -> None:
    assert LabelStore is LabeledPairStore


def test_mapping_rule_scalar_and_dict_matching() -> None:
    rule = MappingRule(
        rule_id="r1",
        description="Vendor rule",
        priority=10,
        condition={"source_vendor_id": "V-100", "country": "US", "local_account_prefix": "4"},
        target={"target_erp_id": "ERP-999", "target_id": "ERP-999"},
    )

    # Scalar matching
    assert rule.applies_to("V-100") is True
    assert rule.applies_to(100) is False
    assert rule.applies_to("US") is True
    assert rule.applies_to("4100") is True  # prefix match
    assert rule.applies_to("5100") is False

    # Series matching
    s_match = pd.Series({"source_vendor_id": "V-100", "country": "US", "local_account": "4200"})
    assert rule.applies_to(s_match) is True

    s_mismatch = pd.Series({"source_vendor_id": "V-100", "country": "UK", "local_account": "4200"})
    assert rule.applies_to(s_mismatch) is False

    s_missing = pd.Series({"source_vendor_id": "V-100"})
    assert rule.applies_to(s_missing) is False

    rule_source_alias = MappingRule(
        rule_id="r_alias",
        description="Alias rule",
        priority=10,
        condition={"source_id": "S-1"},
        target={"target_id": "T-1"},
    )
    assert rule_source_alias.applies_to(pd.Series({"unrelated": "XYZ"})) is False

    # Rule without local_account_prefix: test missing condition key in row
    rule_general = MappingRule(
        rule_id="r_gen",
        description="General rule",
        priority=10,
        condition={"vendor_id": "V-1", "dept": "Finance"},
        target={"target_id": "T-1"},
    )
    assert rule_general.applies_to(pd.Series({"vendor_id": "V-1"})) is False

    # Evaluation
    res = rule.evaluate(s_match)
    assert isinstance(res, RuleResult)
    assert res.matched is True
    assert res.target == {"target_erp_id": "ERP-999", "target_id": "ERP-999"}
    assert res.rule_id == "r1"

    res_fail = rule.evaluate(s_mismatch)
    assert res_fail.matched is False
    assert res_fail.target is None
    assert res_fail.rule_id is None


def test_mapping_ruleset_container_and_match() -> None:
    r1 = MappingRule("r1", "Rule 1", 10, {"source_id": "S1"}, {"target_id": "T1"})
    r2 = MappingRule("r2", "Rule 2", 20, {"source_id": "S2"}, {"target_erp_id": "ERP-2"})
    r3 = MappingRule("r3", "Rule 3", 30, {"source_id": "S3"}, {"target_account": "ACC-3"})
    r4 = MappingRule("r4", "Rule 4", 40, {"source_id": "S4"}, {"custom_key": "CUST-4"})
    r5 = MappingRule("r5", "Rule 5", 50, {"source_id": "S5"}, {})

    ruleset = MappingRuleSet([r3, r1, r2, r4, r5])  # Unsorted input

    # __len__
    assert len(ruleset) == 5

    # Priority sorting on post-init
    assert ruleset[0].rule_id == "r1"
    assert ruleset[1].rule_id == "r2"
    assert ruleset[2].rule_id == "r3"

    # __iter__
    ids = [r.rule_id for r in ruleset]
    assert ids == ["r1", "r2", "r3", "r4", "r5"]

    # match() with scalar queries
    assert ruleset.match("S1") == "T1"
    assert ruleset.match("S2") == "ERP-2"
    assert ruleset.match("S3") == "ACC-3"
    assert ruleset.match("S4") == "CUST-4"
    assert ruleset.match("S5") is None  # empty target
    assert ruleset.match("NONEXISTENT") is None


def test_mapping_ruleset_json_serialization(tmp_path: Path) -> None:
    r1 = MappingRule("r1", "Rule 1", 10, {"src": "A"}, {"tgt": "B"})
    r2 = MappingRule("r2", "Rule 2", 20, {"src": "C"}, {"tgt": "D"})
    ruleset = MappingRuleSet([r1, r2])

    # to_json string and file
    json_path = tmp_path / "rules.json"
    json_str = ruleset.to_json(json_path)
    assert json_path.exists()
    assert '"r1"' in json_str

    # from_json file
    loaded_file = MappingRuleSet.from_json(json_path)
    assert len(loaded_file) == 2
    assert loaded_file[0].rule_id == "r1"

    # from_json raw JSON string
    loaded_str = MappingRuleSet.from_json(json_str)
    assert len(loaded_str) == 2

    # from_json bare list format
    bare_list = json.dumps(
        [
            {"rule_id": "rx", "condition": {"x": 1}, "target": {"y": 2}},
        ]
    )
    loaded_bare = MappingRuleSet.from_json(bare_list)
    assert len(loaded_bare) == 1
    assert loaded_bare[0].rule_id == "rx"

    # Invalid JSON format
    with pytest.raises(ValueError, match="Invalid JSON format"):
        MappingRuleSet.from_json('"just a string"')


def test_mapping_ruleset_csv_serialization(tmp_path: Path) -> None:
    r1 = MappingRule("r1", "Rule 1", 10, {"src": "A"}, {"tgt": "B"})
    ruleset = MappingRuleSet([r1])

    csv_path = tmp_path / "rules.csv"
    csv_str = ruleset.to_csv(csv_path)
    assert csv_path.exists()
    assert "r1" in csv_str

    # from_csv file
    loaded_file = MappingRuleSet.from_csv(csv_path)
    assert len(loaded_file) == 1
    assert loaded_file[0].rule_id == "r1"
    assert loaded_file[0].condition == {"src": "A"}

    # from_csv string
    loaded_str = MappingRuleSet.from_csv(csv_str)
    assert len(loaded_str) == 1

    # Missing columns error
    bad_csv = "rule_id,priority\nr1,10\n"
    with pytest.raises(ValueError, match="CSV missing required columns"):
        MappingRuleSet.from_csv(bad_csv)


def test_synthesize_rules_from_reviews_comprehensive(tmp_path: Path) -> None:
    # 1. ReviewDecisions input
    rev_df = pd.DataFrame(
        {
            "source_id": ["S1", "S2"],
            "target_id": ["T1", "T2"],
            "decision": ["accept", "reject"],
        }
    )
    decisions = ReviewDecisions({"S1": True, "S2": False}, rev_df)

    rule_set, store = synth_from_rules(decisions)
    assert len(rule_set) == 1
    assert rule_set[0].target["target_id"] == "T1"
    assert len(store) == 2

    # 2. Unsupported type
    with pytest.raises(TypeError, match="Unsupported review_df type"):
        synth_from_store(12345)

    # 3. Empty input
    empty_rules, _ = synth_from_store(pd.DataFrame())
    assert len(empty_rules) == 0

    # 4. Custom label_store path (str and Path)
    store_file = tmp_path / "custom_store.csv"
    _, st_path = synth_from_store(rev_df, label_store=store_file)
    assert store_file.exists()
    assert len(st_path) == 2

    _, st_str = synth_from_store(rev_df, label_store=str(store_file))
    assert len(st_str) == 2

    inst_store = LabeledPairStore(tmp_path / "inst.csv")
    _, st_inst = synth_from_store(rev_df, label_store=inst_store)
    assert st_inst is inst_store

    # 5. Missing required columns
    bad_df = pd.DataFrame({"source_id": ["S1"], "unknown": ["T1"]})
    with pytest.raises(ValueError, match="review DataFrame missing required columns"):
        synth_from_store(bad_df)

    bad_df_no_src = pd.DataFrame({"unrelated_col": [1]})
    with pytest.raises(ValueError, match="review DataFrame missing required columns"):
        synth_from_store(bad_df_no_src)

    # 6. Column fallbacks (source_vendor_id, target_erp_id, status)
    fallback_df = pd.DataFrame(
        {
            "source_vendor_id": ["V1"],
            "target_erp_id": ["E1"],
            "status": ["approve"],
        }
    )
    fb_rules, fb_store = synth_from_store(fallback_df)
    assert len(fb_rules) == 1
    assert len(fb_store) == 1
    assert fb_rules.match("V1") == "E1"

    # 7. Various positive and negative decision tokens, and pending ignore
    token_df = pd.DataFrame(
        {
            "source_id": ["S1", "S2", "S3", "S4", "S5", "S6"],
            "target_id": ["T1", "T2", "T3", "T4", "T5", "T6"],
            "decision": ["accepted", "1", "true", "declined", "0", "pending"],
        }
    )
    tok_rules, tok_store = synth_from_store(token_df)
    assert len(tok_rules) == 3
    assert len(tok_store) == 5  # 3 positive + 2 negative (pending ignored)

    # 8. Deduplication with conflicting decisions (keep='last')
    conflict_df = pd.DataFrame(
        {
            "source_id": ["S1", "S1"],
            "target_id": ["T1", "T1"],
            "decision": ["accept", "reject"],
        }
    )
    conf_rules, _ = synth_from_store(conflict_df)
    # Latest decision was reject -> no rules synthesized
    assert len(conf_rules) == 0

    # 9. Multiple targets accepted for same source_id (keep='last')
    multi_df = pd.DataFrame(
        {
            "source_id": ["S1", "S1"],
            "target_id": ["T1", "T2"],
            "decision": ["accept", "accept"],
        }
    )
    multi_rules, _ = synth_from_store(multi_df)
    assert len(multi_rules) == 1
    assert multi_rules.match("S1") == "T2"

    # 10. Only rejected rows -> empty ruleset
    only_reject = pd.DataFrame(
        {
            "source_id": ["S1"],
            "target_id": ["T1"],
            "decision": ["reject"],
        }
    )
    rej_rules, rej_store = synth_from_store(only_reject)
    assert len(rej_rules) == 0
    assert len(rej_store) == 1
