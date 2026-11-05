"""Synthetic benchmark fixture: Bank feed transactions vs Accounts Payable invoices."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .smb_generators import (
    SMBDataGenerator,
    corrupt_amount,
    corrupt_date,
    format_date_randomly,
)

BANK_NARRATIVE_TEMPLATES: list[str] = [
    "ACH DEBIT {NAME} REF {REF} WEB PMT",
    "WIRE TFR TO {NAME} / INV {REF}",
    "CHECK {REF} PAID TO {NAME}",
    "DIRECT DEP {NAME} *TXN {REF}*",
    "POS DEBIT PURCHASE {NAME} LOC {REF}",
]


def generate_bank_reconciliation_benchmark(
    n_transactions: int = 50,
    amount_noise_max: float = 0.50,
    date_lag_max_days: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Generate Bank Statement feed vs Accounts Payable invoice register benchmark.

    Returns:
        df_bank: Bank statement transactions with messy descriptions and clearing dates/amounts.
        df_ledger: AP invoice ledger with clean vendor names and invoice dates/amounts.
        ground_truth: Mapping from bank_id to ledger_id.
    """
    gen = SMBDataGenerator(seed=seed)
    canonical = gen.generate_canonical_entities(count=n_transactions)
    rng = gen.rng

    bank_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    ground_truth: dict[str, str] = {}

    for i, ent in enumerate(canonical):
        ledger_id = f"INV-{i:04d}"
        bank_id = f"BNK-{i:04d}"

        # 80% have matching bank transactions, 20% are un-reconciled
        is_matched = rng.random() > 0.20

        # Ledger record (canonical AP invoice)
        base_amt = float(ent["base_amount"])
        base_dt = ent["base_date"]  # type: ignore[assignment]
        payee = str(ent["full_name"])

        ledger_rows.append(
            {
                "ledger_id": ledger_id,
                "payee_name": payee,
                "invoice_amount": base_amt,
                "invoice_date": base_dt.strftime("%Y-%m-%d"),
            }
        )

        if is_matched:
            ground_truth[bank_id] = ledger_id
            # Bank cleared amount has small rounding or wire fee difference
            cleared_amt = corrupt_amount(base_amt, rng, max_delta=amount_noise_max)
            # Bank date has 1-4 days payment processing lag
            cleared_dt = corrupt_date(base_dt, rng, max_days=date_lag_max_days)

            # Bank description embeds messy base name and reference number
            template = rng.choice(BANK_NARRATIVE_TEMPLATES)
            ref_num = rng.randint(10000, 99999)
            bank_desc = template.format(
                NAME=str(ent["base_name"]).upper(),
                REF=ref_num,
            )

            bank_rows.append(
                {
                    "bank_id": bank_id,
                    "statement_description": bank_desc,
                    "cleared_amount": cleared_amt,
                    "cleared_date": format_date_randomly(cleared_dt, rng),
                }
            )
        else:
            # Unmatched bank line (e.g. bank fee, interest charge)
            unmatched_id = f"BNK-UNM-{i:04d}"
            bank_rows.append(
                {
                    "bank_id": unmatched_id,
                    "statement_description": f"MONTHLY ACCOUNT MAINTENANCE FEE {rng.randint(100, 999)}",
                    "cleared_amount": round(rng.uniform(15.0, 50.0), 2),
                    "cleared_date": format_date_randomly(base_dt, rng),
                }
            )

    return pd.DataFrame(bank_rows), pd.DataFrame(ledger_rows), ground_truth
