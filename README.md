# fuzzymatch-finance

[![CI](https://github.com/trikieu293/fuzzymatch-finance/actions/workflows/ci.yml/badge.svg)](https://github.com/trikieu293/fuzzymatch-finance/actions)
[![Type Checking](https://img.shields.io/badge/types-mypy%20--strict-blue.svg)](https://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen.svg)](https://github.com/trikieu293/fuzzymatch-finance/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)

Finance and FP&A domain workflow package built on top of [`fuzzymatch`](https://github.com/trikieu293/fuzzymatch) (>=1.0.0,<2.0.0).

Provides deterministic rule-first mapping, fuzzy fallback with structured evidence, policy-based threshold routing, Excel review workflows, financial identifier integrity checks, and audited active learning loops.

## Highlights

- **Rule-First Workflow** — evaluates deterministic company and account rules before falling back to probabilistic fuzzy matching.
- **Financial Identifier Integrity** — ISO 7064 Mod 97-10 (`validate_iban`, `validate_lei`) and Luhn Mod 10 (`validate_cusip`) validators with 100% single-character corruption sensitivity.
- **Accounting Synonym Lexicons** — built-in default accounting synonyms with runtime overrides and curation loaders that hash into the reproducibility manifest.
- **Audited Human-in-the-Loop Review** — boundary-prioritized Excel export/import round-trips (`review_xlsx`) with exploration slices for discovering new synonyms.
- **Inter-Rater Agreement** — Cohen's kappa evaluation against curated gold standards (`gold_set_report`).
- **5-Key Reproducibility Manifest** — every run signs an audit manifest containing `config_hash`, `lexicon_hash`, `data_hash`, `code_version`, and `deps_fingerprint`.
- **Strict Quality Assurance** — 404 tests (unit, adversarial, e2e), `mypy --strict` clean, `ruff` clean, PEP 561 typed (`py.typed`).

## Installation

```bash
pip install fuzzymatch-finance
```

For development (editable install with dev extras):

```bash
git clone https://github.com/trikieu293/fuzzymatch-finance
cd fuzzymatch-finance
pip install -e ".[dev]"
```

Requires Python 3.11+ and `fuzzymatch>=1.0.0,<2.0.0`.

## Core Workflow: Rule-First with Fuzzy Fallback

```text
Deterministic Rules → Policy Route → [Auto-Accept | Needs Review | Low Confidence]
         ↓ (unmatched)
   Fuzzy Fallback → Policy Route → [Auto-Accept | Needs Review | Low Confidence]
         ↓
  Structured Audit Log (JSONL) + 5-Key Manifest
```

### Basic Usage

```python
import pandas as pd
import fuzzymatch as fm
from fuzzymatch_finance import (
    FinanceMappingJob,
    MappingPolicy,
    MappingRule,
    MappingRuleSet,
)

# 1. Define deterministic rules
rules = MappingRuleSet(
    [
        MappingRule(
            rule_id="R001",
            description="Exact match on local account prefix",
            priority=1,
            condition={"company_code": "1000", "local_account_prefix": "100"},
            target={"target_account": "1000-CASH"},
        ),
    ]
)

# 2. Configure policy thresholds
policy = MappingPolicy(auto_accept_threshold=0.85, review_threshold=0.60)

# 3. Configure core fuzzy matcher fallback
fuzzy_matcher = fm.Matcher(
    link_type="link",
    blocking=fm.ExactBlock(["currency"]),
    scorer=fm.RapidFuzzScorer(),
    classifier=fm.FSClassifier(),
    scoring_fields=["account_name"],
    scoring_method="token_set_ratio",
)

job = FinanceMappingJob(
    rule_set=rules,
    policy=policy,
    fuzzy_matcher=fuzzy_matcher,
    fuzzy_scoring_field="account_name",
)

result = job.run(source_df, target_df)
print("Approved:", len(result.approved_mappings))
print("Review Queue:", len(result.review_queue))
print("Evidence:", len(result.evidence))
```

## Domain Workflows with `FinanceMappingJob`

`FinanceMappingJob` orchestrates deterministic rule execution followed by probabilistic fuzzy matching across corporate finance scenarios:

### 1. General Ledger & Chart of Accounts Mapping
Reconciles disparate local charts of accounts to a unified global corporate taxonomy:
```python
job = FinanceMappingJob(rule_set=gl_rules, policy=policy, fuzzy_matcher=matcher)
result = job.run(local_gl_df, corporate_coa_df)
approved = result.approved_mappings_dataframe()
```

### 2. Vendor Deduplication & Master Data Cleanup
Deduplicates vendor records across business units using rule-based tax ID / LEI matching combined with fuzzy vendor name matching:
```python
from fuzzymatch_finance.identifiers import validate_lei

# Match on valid LEI first, fall back to fuzzy vendor name
valid_vendors = vendor_df[vendor_df["lei"].apply(validate_lei)]
```

### 3. Financial Hierarchy & Cost Center Rollup
Maps departmental cost centers and legal entities into consolidation trees using prefix rules and fuzzy descriptions.

### 4. Securities & Instrument Matching
Matches trade tickets and positions using CUSIP, LEI, and IBAN check-digit validators:
```python
from fuzzymatch_finance.identifiers import validate_cusip, validate_lei


def clean_security(row):
    if validate_cusip(row["cusip"]):
        return "valid_cusip"
    elif validate_lei(row["lei"]):
        return "valid_lei"
    return "requires_fuzzy_match"
```

## Financial Identifier Validation

Provides high-integrity check-digit validation for critical financial identifiers:

| Identifier | Standard | Function | Error Detection |
| :--- | :--- | :--- | :--- |
| **IBAN** | ISO 13616 / ISO 7064 Mod 97-10 | `validate_iban(iban: str) -> bool` | 100% single-character substitution & transposition |
| **LEI** | ISO 17442 / ISO 7064 Mod 97-10 | `validate_lei(lei: str) -> bool` | 100% single-character substitution & transposition |
| **CUSIP** | ANSI X9.6 / Luhn Mod 10 | `validate_cusip(cusip: str) -> bool` | 100% single-digit check-digit error detection |

```python
from fuzzymatch_finance.identifiers import validate_iban, validate_lei, validate_cusip

assert validate_iban("GB82WEST12345698765432") is True
assert validate_lei("5493001KJTIIGC8Y1R12") is True
assert validate_cusip("037833100") is True
```

## Review Workflow & Active Learning Loop

```python
from fuzzymatch_finance.review_xlsx import export_review_xlsx, import_review_xlsx
from fuzzymatch_finance.label_store import LabeledPairStore, capture_labels
from fuzzymatch_finance.lexicon import load_lexicon_overrides

# 1. Export prioritized review queue to Excel (with exploration slice)
export_review_xlsx(result, policy, "review_queue.xlsx", exploration_rate=0.10)

# 2. Curators annotate decisions in Excel (accept / reject)

# 3. Import human labels and persist to label store
outcomes = import_review_xlsx("review_queue.xlsx")
store = LabeledPairStore("curated_labels.csv")
capture_labels(store, result, outcomes=outcomes, dataset_tag="coa-q3")

# 4. Retrain with mined synonym overrides
from dataclasses import replace

overrides = load_lexicon_overrides("curated_synonyms.csv")
job = replace(job, lexicon_overrides=overrides)
```

### Reviewer Quality Tracking

Track inter-annotator agreement against a known gold standard:

```python
from fuzzymatch_finance.agreement import gold_set_report

report = gold_set_report(gold_pairs, curator_decisions)
# {'kappa': 0.88, 'n': 100, 'agree_rate': 0.94}
```

## Reproducibility Manifest

Every `MappingResult` embeds a cryptographic manifest verifying 5 pillars of execution:
```json
{
  "config_hash": "a1b2c3...",
  "lexicon_hash": "d4e5f6...",
  "data_hash": "789abc...",
  "code_version": "1.0.0",
  "deps_fingerprint": "fuzzymatch==1.0.0|pandas==2.2.3|..."
}
```

## Development & Testing

```bash
# Run tests with 95% coverage gate
python -m pytest tests/unit/ tests/adversarial/ tests/e2e/ -v --cov=src/fuzzymatch_finance --cov-fail-under=95

# Type checking
python -m mypy

# Linting
python -m ruff check src tests
```

## Citation

```bibtex
@software{fuzzymatch_finance2026,
  author = {Tri Kieu},
  title = {fuzzymatch-finance: FP&A & Financial Entity Resolution Workflows},
  version = {1.0.0},
  year = {2026},
  url = {https://github.com/trikieu293/fuzzymatch-finance}
}
```

## License

MIT. See [LICENSE](LICENSE).
