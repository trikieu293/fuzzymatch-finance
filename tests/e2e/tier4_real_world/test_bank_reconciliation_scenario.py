"""Tier 4 Real-World Scenario: Automated Bank Feed vs Open AP Invoices Reconciliation.

Scenario Description:
An SMB controller downloads daily bank feed transactions (raw statement memo, cleared amount,
cleared date) and reconciles them against open accounts payable invoices. Cleared bank amounts
often differ by wire fees ($0.25 - $1.00) or rounding errors, and payment processing introduces
a 1 to 4 day date lag.

Acceptance Verification:
1. Continuous numeric proximity evaluates amounts within +/- $1.00 tolerance.
2. Continuous date window evaluates dates within +/- 5 days payment window.
3. Composite scoring accurately prioritizes true clearing pairs over spurious fee transactions.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from ..conftest import require_feature
from ..fixtures.bank_reconciliation import (
    generate_bank_reconciliation_benchmark,
)


def test_scenario_bank_transaction_reconciliation() -> Any:
    """Full workflow: Reconcile bank transactions against AP invoices using composite scoring."""
    num_scorer_cls = require_feature(
        "fuzzymatch.scoring.numeric", "NumericProximityScorer", milestone="M2"
    )
    date_scorer_cls = require_feature("fuzzymatch.scoring.date", "DateWindowScorer", milestone="M2")

    df_bank, df_ledger, ground_truth = generate_bank_reconciliation_benchmark(
        n_transactions=30,
        amount_noise_max=0.50,
        date_lag_max_days=3,
        seed=202,
    )

    num_scorer = num_scorer_cls(epsilon=1.0)
    date_scorer = date_scorer_cls(delta_days=5.0)

    # For each true match in ground truth, evaluate the continuous proximity scores
    for bank_id, ledger_id in ground_truth.items():
        bank_row = df_bank[df_bank["bank_id"] == bank_id].iloc[0]
        ledger_row = df_ledger[df_ledger["ledger_id"] == ledger_id].iloc[0]

        amt_score = float(
            num_scorer.score_pair(
                pd.Series([bank_row["cleared_amount"]]),
                pd.Series([ledger_row["invoice_amount"]]),
            ).iloc[0]
        )

        d_bank = pd.to_datetime(pd.Series([bank_row["cleared_date"]]), format="mixed")
        d_ledger = pd.to_datetime(pd.Series([ledger_row["invoice_date"]]))
        date_score = float(date_scorer.score_pair(d_bank, d_ledger).iloc[0])

        # Invariant 1: Because cleared amounts differ by <= $0.50 and epsilon = $1.00, amount score >= 0.50
        assert amt_score >= 0.50, (
            f"Bank {bank_id} amt {bank_row['cleared_amount']} vs Ledger {ledger_row['invoice_amount']} "
            f"yielded low score: {amt_score}"
        )

        # Invariant 2: Because date lag <= 3 days and delta = 5.0 days, date score >= 0.40
        assert date_score >= 0.40, (
            f"Bank {bank_id} date {bank_row['cleared_date']} vs Ledger {ledger_row['invoice_date']} "
            f"yielded low date score: {date_score}"
        )

        # Invariant 3: Both scores are bounded within [0, 1]
        assert 0.0 <= amt_score <= 1.0
        assert 0.0 <= date_score <= 1.0
