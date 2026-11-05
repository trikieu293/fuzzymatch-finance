"""Tier 4 Real-World Scenario: Dirty Accounts Payable Ledger vs ERP Vendor Master.

Scenario Description:
An SMB operates an ERP system (e.g., NetSuite, QuickBooks Enterprise) with a clean
canonical vendor master. The accounts payable team receives dirty supplier invoices
from email, OCR scans, and decentralized purchasing cards. Vendor names suffer from
corporate suffix variations (Inc vs LLC vs Ltd), typographical errors, and 25% of invoices
have missing or corrupted tax identification numbers.

Acceptance Verification:
1. Multi-pass candidate generation achieves >= 98% recall across all true matchable invoices.
2. Candidate budget cap prevents N*M combinatorial explosion on single-workstation hardware.
3. Intransitive vendor triads are correctly split without false transitive chaining.
"""

from __future__ import annotations

from typing import Any

from ..conftest import require_feature
from ..fixtures.vendor_benchmark import (
    evaluate_candidate_recall,
    generate_vendor_benchmark,
)


def test_scenario_dirty_vendor_master_reconciliation() -> Any:
    """Full workflow: Reconcile 150 dirty AP invoice records against 100 clean ERP vendors."""
    multi_pass_cls = require_feature(
        "fuzzymatch.blocking.multi_pass", "MultiPassBlock", milestone="M1"
    )
    star_cluster_cls = require_feature(
        "fuzzymatch.clustering.star_cluster", "StarClusterer", milestone="M1"
    )
    strip_legal_fn = require_feature(
        "fuzzymatch.normalize.legal", "strip_legal_suffixes", milestone="M2"
    )
    from fuzzymatch.blocking.exact import ExactBlock
    from fuzzymatch.blocking.sorted_neighborhood import SortedNeighborhood

    # Generate benchmark dataset
    df_erp, df_ap, ground_truth = generate_vendor_benchmark(
        n_canonical=100,
        n_dirty=150,
        noise_rate=0.40,
        seed=101,
    )

    # Pre-process: add normalized legal names
    df_erp["norm_name"] = df_erp["vendor_name"].apply(strip_legal_fn)
    df_ap["norm_name"] = df_ap["vendor_name"].apply(strip_legal_fn)

    # Multi-pass blocking configuration:
    # Pass 1: Exact match on Tax ID (fast path)
    # Pass 2: Exact match on normalized legal name
    # Pass 3: Sorted neighborhood on normalized name (window=8)
    pass1 = ExactBlock(["tax_id"])
    pass2 = ExactBlock(["norm_name"])
    pass3 = SortedNeighborhood(["norm_name"], window=8)

    budget_cap = 25_000
    blocker = multi_pass_cls(passes=[pass1, pass2, pass3], budget_cap=budget_cap)
    candidate_index = blocker.block(df_erp, df_ap)

    # 1. Verify candidate recall >= 98%
    metrics = evaluate_candidate_recall(
        candidate_pairs=set(candidate_index.pairs),
        df_left=df_erp,
        df_right=df_ap,
        left_id_col="vendor_id",
        right_id_col="ap_id",
        ground_truth=ground_truth,
    )
    assert metrics["candidate_recall"] >= 0.98, (
        f"Vendor reconciliation recall {metrics['candidate_recall']:.2%} < 98%"
    )

    # 2. Verify budget cap containment
    assert metrics["candidate_count"] <= budget_cap

    # 3. Form graph and verify StarClusterer resolves components cleanly
    pairs = [
        (df_erp.iloc[l]["vendor_id"], df_ap.iloc[r]["ap_id"]) for l, r in candidate_index.pairs
    ]
    # Simple Jaccard/Levenshtein similarity placeholder weights for demonstration
    weights = [0.85] * len(pairs)

    clusterer = star_cluster_cls(cutoff=0.75, verify_clique=True)
    clusters = clusterer.cluster(pairs, edge_weights=weights)

    assert len(clusters) > 0
    # Ensure no cluster collapses distinct ERP vendors together
    for cl in clusters:
        erp_nodes_in_cluster = [node for node in cl if str(node).startswith("ENT-")]
        assert len(erp_nodes_in_cluster) <= 1, (
            f"Transitive chaining merged multiple distinct ERP vendors: {erp_nodes_in_cluster}"
        )
