"""Synthetic benchmark fixture: Dirty Accounts Payable ledger vs clean ERP Vendor Master."""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd

from .smb_generators import (
    SMBDataGenerator,
    add_legal_suffix,
    corrupt_amount,
    corrupt_date,
    corrupt_typo,
    format_date_randomly,
)


def generate_vendor_benchmark(
    n_canonical: int = 50,
    n_dirty: int = 80,
    noise_rate: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Generate ERP Vendor Master vs AP Invoice register benchmark.

    Returns:
        df_erp: Canonical ERP vendor master DataFrame (left table)
        df_ap: Dirty AP invoice register DataFrame (right table)
        ground_truth: Mapping from ap_id to canonical erp_id
    """
    gen = SMBDataGenerator(seed=seed)
    canonical = gen.generate_canonical_entities(count=n_canonical)

    erp_rows: list[dict[str, Any]] = []
    for ent in canonical:
        erp_rows.append(
            {
                "vendor_id": ent["canonical_id"],
                "vendor_name": ent["full_name"],
                "tax_id": ent["tax_id"],
                "city": ent["city"],
                "state": ent["state"],
                "zip_code": ent["zip_code"],
            }
        )
    df_erp = pd.DataFrame(erp_rows)

    ap_rows: list[dict[str, Any]] = []
    ground_truth: dict[str, str] = {}

    rng = gen.rng
    for i in range(n_dirty):
        ap_id = f"AP-{i:04d}"
        if i < len(canonical):
            ent = canonical[i]
            matched_canonical = True
        else:
            ent = rng.choice(canonical)
            matched_canonical = rng.random() > 0.15  # 85% matchable, 15% novel/unmatchable

        if matched_canonical:
            ground_truth[ap_id] = str(ent["canonical_id"])
            raw_name = str(ent["base_name"])
            # Apply suffix changes
            name_with_suffix = add_legal_suffix(raw_name, rng)
            # Maybe apply typos
            if rng.random() < noise_rate:
                final_name = corrupt_typo(name_with_suffix, rng, num_edits=1)
            else:
                final_name = name_with_suffix

            # Corrupt tax ID or drop it (simulating missing blocking key)
            if rng.random() < 0.25:
                tax_id_val = None  # Missing key
            elif rng.random() < noise_rate:
                tax_id_val = corrupt_typo(str(ent["tax_id"]), rng, num_edits=1)
            else:
                tax_id_val = str(ent["tax_id"])

            city_val = str(ent["city"]) if rng.random() > 0.10 else None
            state_val = str(ent["state"])
            zip_val = (
                str(ent["zip_code"])
                if rng.random() > 0.15
                else corrupt_typo(str(ent["zip_code"]), rng)
            )

            amt = corrupt_amount(float(ent["base_amount"]), rng, max_delta=0.75)
            dt = corrupt_date(ent["base_date"], rng, max_days=3)  # type: ignore[arg-type]
        else:
            # Novel unmatchable vendor
            novel_base = f"Unique Novel Vendor {i}"
            final_name = add_legal_suffix(novel_base, rng)
            tax_id_val = f"99-{rng.randint(1000000, 9999999)}"
            city_val = "Chicago"
            state_val = "IL"
            zip_val = "60601"
            amt = round(rng.uniform(200.0, 5000.0), 2)
            dt = datetime.date(2025, 6, 1)

        ap_rows.append(
            {
                "ap_id": ap_id,
                "vendor_name": final_name,
                "tax_id": tax_id_val,
                "city": city_val,
                "state": state_val,
                "zip_code": zip_val,
                "invoice_amount": amt,
                "invoice_date": format_date_randomly(dt, rng),
            }
        )

    df_ap = pd.DataFrame(ap_rows)
    return df_erp, df_ap, ground_truth


def evaluate_candidate_recall(
    candidate_pairs: set[tuple[Any, Any]],
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_id_col: str,
    right_id_col: str,
    ground_truth: dict[str, str],
) -> dict[str, float]:
    """Calculate candidate recall over ground truth pairs."""
    # Convert index pairs to ID pairs if needed
    left_ids = list(df_left[left_id_col])
    right_ids = list(df_right[right_id_col])

    id_pairs: set[tuple[str, str]] = set()
    for l_idx, r_idx in candidate_pairs:
        # If pairs are integers (indices)
        if isinstance(l_idx, int) and isinstance(r_idx, int):
            id_pairs.add((str(left_ids[l_idx]), str(right_ids[r_idx])))
        else:
            id_pairs.add((str(l_idx), str(r_idx)))

    true_pairs = {(target_erp, ap_id) for ap_id, target_erp in ground_truth.items()}
    hits = 0
    for erp_id, ap_id in true_pairs:
        if (erp_id, ap_id) in id_pairs or (ap_id, erp_id) in id_pairs:
            hits += 1

    recall = hits / len(true_pairs) if true_pairs else 1.0
    return {
        "candidate_recall": recall,
        "hits": hits,
        "true_matches": len(true_pairs),
        "candidate_count": len(candidate_pairs),
    }
