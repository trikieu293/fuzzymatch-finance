# Changelog — fuzzymatch-finance

All notable changes to **fuzzymatch-finance** are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [1.0.0] - 2026-11-05

Initial production release of `fuzzymatch-finance` — corporate finance, accounting reconciliation, and FP&A workflow extension built on `fuzzymatch`.

### Added
- **Rule-First Mapping Orchestration** (`fuzzymatch_finance.jobs.FinanceMappingJob`):
  Evaluates deterministic accounting and company rules with prioritized ordering before falling back to probabilistic fuzzy matching.
- **Universal Excel Round-Trip Review** (`fuzzymatch_finance.review_xlsx`):
  Boundary-prioritized Excel export/import round-trips with status-coded rows (Auto-Accept, Review, Reject), drop-down validation, exploration slices for discovering new synonyms, and automated label extraction.
- **Active Rule Synthesis** (`fuzzymatch_finance.label_store.synthesize_rules_from_reviews`):
  Automated induction of deterministic `MappingRule` instances from verified review decisions, accelerating chart of accounts migration.
- **Financial Composite Scoring** (`fuzzymatch_finance.scoring.FinancialCompositeScorer`):
  Specialized composite scoring integrating fuzzy text similarity, continuous amount proximity (`amount_proximity_score`), and invoice date tolerance windows (`date_window_score`).
- **Accounting & Legal Normalizers** (`fuzzymatch_finance.lexicon`):
  - `accounting_normalizer`: Built-in accounting synonym dictionary (`DEFAULT_ACCOUNTING_SYNONYMS`) aligning US, UK, and international chart of accounts terminology.
  - `legal_entity_normalizer` and `strip_legal_suffixes`: Punctuation-tolerant canonical normalization of corporate suffixes.
- **High-Integrity Financial Identifier Validation** (`fuzzymatch_finance.identifiers`):
  - ISO 13616 / ISO 7064 Mod 97-10 IBAN validator (`validate_iban`).
  - ISO 17442 / ISO 7064 Mod 97-10 LEI validator (`validate_lei`).
  - ANSI X9.6 / Luhn Mod 10 CUSIP validator (`validate_cusip`, `validate_cusip_check_digit`).
- **Policy Routing & Guardrails** (`fuzzymatch_finance.policy.MappingPolicy`):
  Configurable auto-accept and review thresholds with thin-evidence candidate guards.
- **Audit Logging & Agreement Metrics**:
  - `AuditEvent` and `AuditLog`: Structured JSONL audit trails with execution metadata.
  - `agreement.gold_set_report`: Cohen's kappa and consensus tracking against curated gold standards.
  - 5-key cryptographic reproducibility manifest (`config_hash`, `lexicon_hash`, `data_hash`, `code_version`, `deps_fingerprint`).
- **CLI & Automated Mapping**:
  - `fuzzymatch-finance` command-line interface and `auto_match_finance` workflow pipeline.