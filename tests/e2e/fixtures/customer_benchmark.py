"""Synthetic benchmark fixture: Customer master deduplication and intransitive triad tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .smb_generators import (
    SMBDataGenerator,
    add_legal_suffix,
    corrupt_typo,
)


def generate_intransitive_triad(
    tau: float = 0.70,
) -> tuple[pd.DataFrame, list[tuple[str, str, float]]]:
    """Generate a canonical 3-node intransitive triad (A, B, C).

    Records:
        A: "Apex Logistics LLC", City: "Austin", State: "TX", Phone: "512-555-0101"
        B: "Apex Logistics",     City: "Austin", State: "TX", Phone: None (missing phone/suffix)
        C: "Apex Supply LLC",    City: "Austin", State: "TX", Phone: "512-555-0199"

    Pairwise Affinities:
        (A, B): 0.88 >= tau (Strong match on name prefix and city)
        (B, C): 0.75 >= tau (Moderate match on shared prefix "Apex" and city)
        (A, C): 0.40 < tau  (Weak match: conflicting brand tokens "Logistics" vs "Supply")

    Under standard Connected Components, A, B, and C merge into a single cluster {A, B, C},
    causing false transitive linking of Apex Logistics and Apex Supply.
    Under Star/Pivot Clustering, A and C must NEVER be merged together.
    """
    records = [
        {
            "customer_id": "CUST-A",
            "company_name": "Apex Logistics LLC",
            "city": "Austin",
            "state": "TX",
            "phone": "512-555-0101",
        },
        {
            "customer_id": "CUST-B",
            "company_name": "Apex Logistics",
            "city": "Austin",
            "state": "TX",
            "phone": None,
        },
        {
            "customer_id": "CUST-C",
            "company_name": "Apex Supply LLC",
            "city": "Austin",
            "state": "TX",
            "phone": "512-555-0199",
        },
    ]
    edges = [
        ("CUST-A", "CUST-B", 0.88),
        ("CUST-B", "CUST-C", 0.75),
        ("CUST-A", "CUST-C", 0.40),
    ]
    return pd.DataFrame(records), edges


def generate_customer_benchmark(
    n_canonical: int = 40,
    duplicate_ratio: float = 0.50,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Generate messy customer master dataset for single-table entity resolution / deduplication.

    Returns:
        df_customers: Tabular customer records with duplicates and noise.
        ground_truth_clusters: Map from canonical_id to set of customer_ids belonging to that entity.
    """
    gen = SMBDataGenerator(seed=seed)
    canonical = gen.generate_canonical_entities(count=n_canonical)
    rng = gen.rng

    records: list[dict[str, Any]] = []
    ground_truth_clusters: dict[str, set[str]] = {}

    for i, ent in enumerate(canonical):
        c_id = f"CUST-{i:04d}-0"
        canon_id = str(ent["canonical_id"])
        ground_truth_clusters[canon_id] = {c_id}

        records.append(
            {
                "customer_id": c_id,
                "company_name": ent["full_name"],
                "contact_email": f"billing@{str(ent['base_name']).lower().replace(' ', '')}.com",
                "city": ent["city"],
                "state": ent["state"],
                "zip_code": ent["zip_code"],
            }
        )

        # Add duplicates if selected
        if rng.random() < duplicate_ratio:
            n_dups = rng.randint(1, 2)
            for d in range(n_dups):
                dup_id = f"CUST-{i:04d}-{d + 1}"
                ground_truth_clusters[canon_id].add(dup_id)
                dup_name = add_legal_suffix(str(ent["base_name"]), rng)
                if rng.random() < 0.5:
                    dup_name = corrupt_typo(dup_name, rng, num_edits=1)

                records.append(
                    {
                        "customer_id": dup_id,
                        "company_name": dup_name,
                        "contact_email": f"accounts@{str(ent['base_name']).lower().replace(' ', '')}.com"
                        if rng.random() > 0.3
                        else None,
                        "city": str(ent["city"]) if rng.random() > 0.2 else None,
                        "state": ent["state"],
                        "zip_code": corrupt_typo(str(ent["zip_code"]), rng)
                        if rng.random() < 0.2
                        else ent["zip_code"],
                    }
                )

    return pd.DataFrame(records), ground_truth_clusters
