from __future__ import annotations

import pandas as pd

from fuzzymatch_finance.rules import MappingRule, MappingRuleSet


def test_rule_applies_before_fuzzy_matching():
    rules = MappingRuleSet(
        [
            MappingRule(
                rule_id="coa_prefix_opex_v1",
                description="Map local OPEX accounts by prefix",
                priority=1,
                condition={"company_code": "TH01", "local_account_prefix": "6"},
                target={"target_account": "610000"},
            )
        ]
    )
    row = pd.Series({"company_code": "TH01", "local_account": "610010", "name": "Office Supplies"})

    result = rules.apply(row)

    assert result.matched is True
    assert result.decision_status == "rule_mapped"
    assert result.target["target_account"] == "610000"
    assert result.reason_codes == ["known_company_mapping_rule"]


def test_rule_set_keeps_unmatched_rows_unresolved():
    rules = MappingRuleSet([])
    row = pd.Series({"company_code": "TH01", "local_account": "999999", "name": "Misc"})

    result = rules.apply(row)

    assert result.matched is False
    assert result.decision_status == "unresolved"
    assert result.target is None


def test_rule_set_respects_priority_order():
    rule_high = MappingRule(
        rule_id="high",
        description="high-priority override",
        priority=1,
        condition={"company_code": "TH01", "local_account_prefix": "6"},
        target={"target_account": "610000"},
    )
    rule_low = MappingRule(
        rule_id="low",
        description="low-priority fallback",
        priority=99,
        condition={"company_code": "TH01", "local_account_prefix": "6"},
        target={"target_account": "999000"},
    )
    rules = MappingRuleSet([rule_low, rule_high])
    row = pd.Series({"company_code": "TH01", "local_account": "610010", "name": "Office Supplies"})

    result = rules.apply(row)

    assert result.rule_id == "high"
    assert result.target["target_account"] == "610000"


def test_rule_does_not_apply_when_company_code_mismatches():
    rule = MappingRule(
        rule_id="th01_only",
        description="Only applies to TH01",
        priority=1,
        condition={"company_code": "TH01", "local_account_prefix": "6"},
        target={"target_account": "610000"},
    )
    row = pd.Series({"company_code": "TH02", "local_account": "610010", "name": "Office Supplies"})

    assert rule.applies_to(row) is False

    # Verify through the rule set as well
    rules = MappingRuleSet([rule])
    result = rules.apply(row)
    assert result.matched is False
    assert result.decision_status == "unresolved"


def test_from_json_and_csv_missing_file_paths(tmp_path):
    import pytest

    # Path objects that do not exist must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        MappingRuleSet.from_json(tmp_path / "does_not_exist.json")

    with pytest.raises(FileNotFoundError):
        MappingRuleSet.from_csv(tmp_path / "does_not_exist.csv")

    # Filename strings that end in .json or .csv and do not exist must raise FileNotFoundError
    with pytest.raises(FileNotFoundError):
        MappingRuleSet.from_json(str(tmp_path / "missing_file.json"))

    with pytest.raises(FileNotFoundError):
        MappingRuleSet.from_csv(str(tmp_path / "missing_file.csv"))

    # Existing file paths as string
    existing_json = tmp_path / "rules_str.json"
    existing_json.write_text('{"rules": []}', encoding="utf-8")
    assert len(MappingRuleSet.from_json(str(existing_json))) == 0

    existing_csv = tmp_path / "rules_str.csv"
    existing_csv.write_text("rule_id,condition,target\n", encoding="utf-8")
    assert len(MappingRuleSet.from_csv(str(existing_csv))) == 0


def test_single_line_payloads_without_newlines():
    # Single-line JSON without newlines must parse without checking filesystem
    loaded_json = MappingRuleSet.from_json('{"rules": []}')
    assert len(loaded_json) == 0

    # Single-line CSV without newlines
    loaded_csv = MappingRuleSet.from_csv("rule_id,condition,target")
    assert len(loaded_csv) == 0
