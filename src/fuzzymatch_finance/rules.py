"""Deterministic finance mapping rules."""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from .label_store import LabeledPairStore


@dataclass(frozen=True)
class RuleResult:
    """The outcome of evaluating a mapping rule against one row."""

    matched: bool
    decision_status: str
    target: dict[str, Any] | None
    reason_codes: list[str]
    rule_id: str | None = None


_SOURCE_ID_ALIASES = (
    "source_id",
    "source_vendor_id",
    "source_customer_id",
    "source_local_account",
    "left_id",
    "source_key",
)


@dataclass(frozen=True)
class MappingRule:
    """One deterministic company mapping rule."""

    rule_id: str
    description: str
    priority: int
    condition: dict[str, Any]
    target: dict[str, Any]

    def applies_to(self, row: pd.Series | dict[str, Any] | str | int) -> bool:
        """Check whether this rule matches the given row or scalar identifier."""
        # 1. Scalar query support (e.g. ruleset.match("VEND-0101"))
        if isinstance(row, (str, int)):
            query = str(row)
            # Match directly in condition values
            if any(str(val) == query for val in self.condition.values()):
                return True
            # Legacy prefix match
            prefix = str(self.condition.get("local_account_prefix", ""))
            return bool(prefix and query.startswith(prefix))

        # 2. Tabular row / dict mapping evaluation
        # Legacy GL prefix & company code support
        company_code = str(row.get("company_code", ""))
        local_account = str(row.get("local_account", ""))
        prefix = str(self.condition.get("local_account_prefix", ""))
        required_company = str(self.condition.get("company_code", ""))

        if required_company and company_code != required_company:
            return False
        if prefix and not local_account.startswith(prefix):
            return False

        # General entity dictionary condition matching
        for cond_k, cond_v in self.condition.items():
            if cond_k in ("company_code", "local_account_prefix"):
                continue
            if cond_k in row:
                row_val = row.get(cond_k)
            elif cond_k in _SOURCE_ID_ALIASES:
                row_val = None
                matched_alias = False
                for alias in _SOURCE_ID_ALIASES:
                    if alias in row:
                        row_val = row.get(alias)
                        matched_alias = True
                        break
                if not matched_alias:
                    return False
            else:
                return False

            if str(row_val) != str(cond_v):
                return False

        return True

    def evaluate(self, row: pd.Series | dict[str, Any] | str | int) -> RuleResult:
        """Evaluate this rule against a single row or scalar query."""
        if self.applies_to(row):
            return RuleResult(
                matched=True,
                decision_status="rule_mapped",
                target=dict(self.target),
                reason_codes=["known_company_mapping_rule"],
                rule_id=self.rule_id,
            )
        return RuleResult(
            matched=False,
            decision_status="unresolved",
            target=None,
            reason_codes=[],
            rule_id=None,
        )


@dataclass(frozen=True)
class MappingRuleSet:
    """An ordered collection of mapping rules evaluated by priority."""

    rules: list[MappingRule] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sort on frozen dataclass via object.__setattr__
        object.__setattr__(self, "rules", sorted(self.rules, key=lambda r: r.priority))

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterator[MappingRule]:
        return iter(self.rules)

    def __getitem__(self, index: int) -> MappingRule:
        return self.rules[index]

    def apply(self, row: pd.Series | dict[str, Any] | str | int) -> RuleResult:
        """Evaluate all rules by priority; return the first match."""
        for rule in self.rules:
            result = rule.evaluate(row)
            if result.matched:
                return result
        return RuleResult(
            matched=False,
            decision_status="unresolved",
            target=None,
            reason_codes=[],
            rule_id=None,
        )

    def match(self, key_or_row: str | int | pd.Series | dict[str, Any]) -> str | None:
        """Convenience method returning the matched target identifier or None."""
        result = self.apply(key_or_row)
        if not result.matched or result.target is None:
            return None

        # Look for standard target identifier keys
        for candidate_key in ("target_id", "target_erp_id", "target_account", "target_vendor_id"):
            if candidate_key in result.target and result.target[candidate_key] is not None:
                return str(result.target[candidate_key])

        for val in result.target.values():
            if val is not None:
                return str(val)
        return None

    def to_json(self, path: Path | str | None = None, indent: int = 2) -> str:
        """Serialize rule set to JSON format."""
        payload = {
            "version": "1.0",
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "description": r.description,
                    "priority": r.priority,
                    "condition": r.condition,
                    "target": r.target,
                }
                for r in self.rules
            ],
        }
        text = json.dumps(payload, indent=indent, sort_keys=True)
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        return text

    @classmethod
    def from_json(cls, source: Path | str) -> MappingRuleSet:
        """Load MappingRuleSet from a JSON file path or JSON string."""
        if isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        else:
            raw_str = str(source).strip()
            if raw_str.startswith(("{", "[")):
                text = str(source)
            elif str(source).endswith(".json"):
                p = Path(source)
                if not p.is_file():
                    raise FileNotFoundError(f"JSON file not found: {source}")
                text = p.read_text(encoding="utf-8")
            else:
                is_file = False
                if "\n" not in str(source) and "\r" not in str(source):
                    try:
                        p = Path(source)
                        if p.is_file():
                            is_file = True
                    except (OSError, ValueError):
                        is_file = False

                if is_file:
                    text = Path(source).read_text(encoding="utf-8")
                else:
                    text = str(source)

        data = json.loads(text)
        if isinstance(data, dict) and "rules" in data:
            raw_rules = data["rules"]
        elif isinstance(data, list):
            raw_rules = data
        else:
            raise ValueError(
                "Invalid JSON format for MappingRuleSet: expected dict with 'rules' or list of rules"
            )

        rules = [
            MappingRule(
                rule_id=str(r["rule_id"]),
                description=str(r.get("description", "")),
                priority=int(r.get("priority", 100)),
                condition=dict(r["condition"]),
                target=dict(r["target"]),
            )
            for r in raw_rules
        ]
        return cls(rules)

    def to_csv(self, path: Path | str | None = None) -> str:
        """Serialize rule set to CSV format."""
        rows = [
            {
                "rule_id": r.rule_id,
                "description": r.description,
                "priority": r.priority,
                "condition": json.dumps(r.condition, sort_keys=True),
                "target": json.dumps(r.target, sort_keys=True),
            }
            for r in self.rules
        ]
        df = pd.DataFrame(
            rows, columns=["rule_id", "description", "priority", "condition", "target"]
        )
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
        return df.to_csv(index=False)

    @classmethod
    def from_csv(cls, source: Path | str) -> MappingRuleSet:
        """Load MappingRuleSet from a CSV file path or CSV string."""
        if isinstance(source, Path):
            df = pd.read_csv(source)
        else:
            raw_str = str(source).strip()
            if "\n" in raw_str or "\r" in raw_str:
                df = pd.read_csv(io.StringIO(str(source)))
            elif str(source).endswith(".csv"):
                p = Path(source)
                if not p.is_file():
                    raise FileNotFoundError(f"CSV file not found: {source}")
                df = pd.read_csv(p)
            else:
                is_file = False
                try:
                    p = Path(source)
                    if p.is_file():
                        is_file = True
                except (OSError, ValueError):
                    is_file = False

                if is_file:
                    df = pd.read_csv(Path(source))
                else:
                    df = pd.read_csv(io.StringIO(str(source)))

        required = {"rule_id", "condition", "target"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV missing required columns: {sorted(required - set(df.columns))}")

        rules: list[MappingRule] = []
        for _, row in df.iterrows():
            cond_val = row["condition"]
            tgt_val = row["target"]
            cond = json.loads(cond_val) if isinstance(cond_val, str) else dict(cond_val)
            tgt = json.loads(tgt_val) if isinstance(tgt_val, str) else dict(tgt_val)
            rules.append(
                MappingRule(
                    rule_id=str(row["rule_id"]),
                    description=str(row.get("description", "")),
                    priority=int(row.get("priority", 100)),
                    condition=cond,
                    target=tgt,
                )
            )
        return cls(rules)


# Re-export active learning synthesizer and store alias from label_store
def synthesize_rules_from_reviews(
    review_df: pd.DataFrame | Any,
    label_store: LabeledPairStore | Path | str | None = None,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
    decision_col: str = "decision",
    dataset_tag: str = "default",
    labeler: str = "review",
    priority_start: int = 100,
) -> tuple[MappingRuleSet, LabeledPairStore]:
    """Re-export of synthesize_rules_from_reviews from label_store."""
    from .label_store import synthesize_rules_from_reviews as _synth

    return _synth(
        review_df=review_df,
        label_store=label_store,
        source_id_col=source_id_col,
        target_id_col=target_id_col,
        decision_col=decision_col,
        dataset_tag=dataset_tag,
        labeler=labeler,
        priority_start=priority_start,
    )
