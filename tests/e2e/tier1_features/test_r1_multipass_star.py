"""Tier 1: Feature Coverage Tests for Requirement R1.

Multi-Pass Disjunctive Candidate Generation & Transitive Cluster Protection:
- F1.1: MultiPassBlock candidate generation
- F1.2: Configurable candidate budget cap
- F1.3: Star/Pivot clustering eliminating false transitive chaining
- F1.4: String identifier support in clustering
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from ..conftest import require_feature
from ..fixtures.customer_benchmark import generate_intransitive_triad
from ..fixtures.vendor_benchmark import (
    evaluate_candidate_recall,
    generate_vendor_benchmark,
)


@pytest.fixture
def multi_pass_blocker_cls() -> Any:
    """Fixture providing MultiPassBlock class gated by Milestone M1."""
    return require_feature("fuzzymatch.blocking.multi_pass", "MultiPassBlock", milestone="M1")


@pytest.fixture
def star_clusterer_cls() -> Any:
    """Fixture providing StarClusterer class gated by Milestone M1."""
    return require_feature("fuzzymatch.clustering.star_cluster", "StarClusterer", milestone="M1")


def test_multipass_disjunctive_accumulation(multi_pass_blocker_cls: Any) -> Any:
    """R1: MultiPassBlock sequentially executes passes and accumulates unique candidate pairs."""
    from fuzzymatch.blocking.exact import ExactBlock

    df_l = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "key1": ["A", "B", "C"],
            "key2": ["X", "Y", "Z"],
        }
    )
    df_r = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "key1": ["A", "MISMATCH", "C"],
            "key2": ["MISMATCH", "Y", "MISMATCH"],
        }
    )

    pass1 = ExactBlock(["key1"])
    pass2 = ExactBlock(["key2"])

    blocker = multi_pass_blocker_cls(passes=[pass1, pass2], budget_cap=1000)
    candidate_index = blocker.block(df_l, df_r)

    # Pass 1 catches (0, 0) and (2, 2)
    # Pass 2 catches (1, 1)
    # Total accumulated unique pairs should be 3
    pairs = set(candidate_index.pairs)
    assert len(pairs) == 3
    assert (0, 0) in pairs
    assert (1, 1) in pairs
    assert (2, 2) in pairs


def test_multipass_budget_cap_enforcement(multi_pass_blocker_cls: Any) -> Any:
    """R1: Configurable candidate budget cap B_max strictly bounds generated pairs."""
    from fuzzymatch.blocking.exact import ExactBlock

    # Generate Cartesian product scenario where all rows share the same key
    n_left, n_right = 20, 30
    df_l = pd.DataFrame({"key": ["COMMON"] * n_left})
    df_r = pd.DataFrame({"key": ["COMMON"] * n_right})

    # Total possible pairs = 20 * 30 = 600
    budget_cap = 50
    blocker = multi_pass_blocker_cls(passes=[ExactBlock(["key"])], budget_cap=budget_cap)
    candidate_index = blocker.block(df_l, df_r)

    assert len(candidate_index.pairs) <= budget_cap
    assert len(candidate_index.pairs) == budget_cap


def test_multipass_candidate_recall_dirty_benchmark(multi_pass_blocker_cls: Any) -> Any:
    """R1: Multi-pass candidate generation achieves >= 98% recall on dirty benchmark dataset."""
    from fuzzymatch.blocking.exact import ExactBlock
    from fuzzymatch.blocking.sorted_neighborhood import SortedNeighborhood

    df_erp, df_ap, ground_truth = generate_vendor_benchmark(
        n_canonical=40, n_dirty=60, noise_rate=0.30, seed=42
    )

    # Multi-pass strategy:
    # Pass 1: Exact match on tax_id (recovers clean rows)
    # Pass 2: Sorted neighborhood on vendor_name window=5 (recovers typos/suffixes)
    pass1 = ExactBlock(["tax_id"])
    pass2 = SortedNeighborhood(["vendor_name"], window=5)

    blocker = multi_pass_blocker_cls(passes=[pass1, pass2], budget_cap=10_000)
    candidate_index = blocker.block(df_erp, df_ap)

    metrics = evaluate_candidate_recall(
        candidate_pairs=set(candidate_index.pairs),
        df_left=df_erp,
        df_right=df_ap,
        left_id_col="vendor_id",
        right_id_col="ap_id",
        ground_truth=ground_truth,
    )

    # Acceptance Criteria: Candidate Recall >= 98%
    assert metrics["candidate_recall"] >= 0.98, (
        f"MultiPass recall {metrics['candidate_recall']:.2%} is below 98% threshold"
    )
    assert metrics["candidate_count"] <= 10_000


def test_star_clustering_intransitive_triad_elimination(star_clusterer_cls: Any) -> Any:
    """R1: Star/Pivot clustering eliminates false transitive links on a 3-node intransitive triad.

    Graph:
        w(A, B) = 0.88 >= tau (0.70)
        w(B, C) = 0.75 >= tau (0.70)
        w(A, C) = 0.40 < tau  (0.70)
    Invariant: A and C must NEVER be placed in the same cluster.
    """
    _df_triad, edges = generate_intransitive_triad(tau=0.70)

    pairs = [(u, v) for u, v, _ in edges]
    weights = [w for _, _, w in edges]

    clusterer = star_clusterer_cls(cutoff=0.70, verify_clique=True)
    clusters = clusterer.cluster(pairs, edge_weights=weights)

    # Find clusters containing CUST-A and CUST-C
    cluster_a = next(c for c in clusters if "CUST-A" in c)
    cluster_c = next(c for c in clusters if "CUST-C" in c)

    # Invariant: A and C must be in disjoint clusters
    assert cluster_a != cluster_c, "Transitive false chaining occurred: A and C were merged!"
    assert "CUST-C" not in cluster_a
    assert "CUST-A" not in cluster_c


def test_star_clustering_string_and_uuid_identifiers(star_clusterer_cls: Any) -> Any:
    """R1 & Survey: Clustering handles arbitrary string and UUID node IDs without integer cast bugs."""
    pairs = [
        ("uuid-1111-aaaa", "uuid-2222-bbbb"),
        ("uuid-2222-bbbb", "uuid-3333-cccc"),
        ("non_numeric_id_alpha", "non_numeric_id_beta"),
    ]
    weights = [0.95, 0.90, 0.85]

    clusterer = star_clusterer_cls(cutoff=0.80, verify_clique=False)
    clusters = clusterer.cluster(pairs, edge_weights=weights)

    # Verify all string IDs are present in the output clusters
    flattened_nodes = {node for cl in clusters for node in cl}
    assert "uuid-1111-aaaa" in flattened_nodes
    assert "non_numeric_id_alpha" in flattened_nodes
    assert len(clusters) >= 2


def test_star_clustering_complete_clique_cohesion(star_clusterer_cls: Any) -> Any:
    """R1: Nodes forming a complete clique where all pairwise weights >= cutoff stay together."""
    nodes = ["N1", "N2", "N3", "N4"]
    pairs: list[tuple[str, str]] = []
    weights: list[float] = []

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            pairs.append((nodes[i], nodes[j]))
            weights.append(0.92)

    clusterer = star_clusterer_cls(cutoff=0.75, verify_clique=True)
    clusters = clusterer.cluster(pairs, edge_weights=weights)

    # A complete clique above cutoff must form a single cohesive cluster
    assert len(clusters) == 1
    assert clusters[0] == set(nodes)
