# Contributing to fuzzymatch-finance

Thank you for your interest in contributing to `fuzzymatch-finance`! We welcome contributions to financial entity resolution algorithms, accounting reconciliation logic, active learning loops, and documentation.

---

## Code of Conduct & Principles

1. **Accounting Conservation Laws**: Reconciled and unmatched transaction volumes and counts must strictly conserve balance sheet identities.
2. **Deterministic Precedence**: Explicit financial rules must strictly take precedence over probabilistic fuzzy matching.
3. **Reproducibility Mandate**: Every matching job and active learning session must generate deterministic cryptographic manifests (`config_hash`, `lexicon_hash`, `data_hash`).

---

## Development Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/trikieu293/fuzzymatch-finance.git
cd fuzzymatch-finance
pip install -e ".[dev]"
```

Ensure core `fuzzymatch` is also available locally or installed from PyPI.

---

## Code Quality Standards

Before submitting a Pull Request, verify that all quality gates pass:

### 1. Code Formatting and Linting
```bash
ruff check src tests
ruff format --check src tests
```

### 2. Static Type Checking
```bash
mypy
```

### 3. Unit, Adversarial, and E2E Tests
```bash
pytest tests/unit tests/adversarial tests/e2e --cov=src/fuzzymatch_finance --cov-report=term-missing --cov-fail-under=95
```

---

## Commit Guidelines

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` A new feature or reconciliation enhancement
- `fix:` A bug fix or volume conservation resolution
- `docs:` Documentation improvements
- `test:` Adding or updating property-based or adversarial tests
- `refactor:` Code refactoring without behavior change
- `chore:` Release tasks, dependency updates, packaging
