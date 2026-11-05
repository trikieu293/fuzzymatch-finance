"""Tier 2: Boundary & Corner Case Tests for Requirement R1.

Multi-Pass Blocking & Star/Pivot Clustering Boundaries:
- Empty DataFrames
- All-null / all-missing blocking keys
- Tight candidate budget caps (B=0, B=1)
- Degenerate graphs (0 edges, singleton vertices, complete bipartite)
- Cross-pass pair deduplication
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from ..conftest import require_feature


@pytest.fixture
def multi_pass_blocker_cls() -> Any:
    """Fixture providing MultiPassBlock class gated by Milestone M1."""
    return require_feature("fuzzymatch.blocking.multi_pass", "MultiPassBlock", milestone="M1")


@pytest.fixture
def star_clusterer_cls() -> Any:
    """Fixture providing StarClusterer class gated by Milestone M1."""
    return require_feature("fuzzymatch.clustering.star_cluster", "StarClusterer", milestone="M1")


def test_r1_empty_dataframes(multi_pass_blocker_cls: Any) -> Any:
    """R1 Boundary: 0-row DataFrames passed to MultiPassBlock return an empty CandidateIndex."""
    from fuzzymatch.blocking.exact import ExactBlock

    df_empty_l = pd.DataFrame(columns=["id", "key"])
    df_empty_r = pd.DataFrame(columns=["id", "key"])

    blocker = multi_pass_blocker_cls(passes=[ExactBlock(["key"])], budget_cap=100)
    ci = blocker.block(df_empty_l, df_empty_r)

    assert len(ci.pairs) == 0


def test_r1_all_null_blocking_keys(multi_pass_blocker_cls: Any) -> Any:
    """R1 Boundary: DataFrames where blocking keys are 100% NaN/None produce zero candidate pairs."""
    from fuzzymatch.blocking.exact import ExactBlock

    df_l = pd.DataFrame({"id": [1, 2, 3], "key": [None, None, None]})
    df_r = pd.DataFrame({"id": [10, 20, 30], "key": [None, None, None]})

    blocker = multi_pass_blocker_cls(passes=[ExactBlock(["key"])], budget_cap=50)
    ci = blocker.block(df_l, df_r)

    # Missing values must never match each other as valid candidate pairs
    assert len(ci.pairs) == 0


def test_r1_tight_budget_cap_limits(multi_pass_blocker_cls: Any) -> Any:
    """R1 Boundary: Candidate budget caps of 0 and 1 strictly enforce ceiling."""
    from fuzzymatch.blocking.exact import ExactBlock

    df_l = pd.DataFrame({"key": ["SAME"] * 10})
    df_r = pd.DataFrame({"key": ["SAME"] * 10})

    # Budget cap = 1
    b1 = multi_pass_blocker_cls(passes=[ExactBlock(["key"])], budget_cap=1)
    ci1 = b1.block(df_l, df_r)
    assert len(ci1.pairs) <= 1

    # Budget cap <= 0 must be rejected as invalid
    with pytest.raises(ValueError):
        multi_pass_blocker_cls(passes=[ExactBlock(["key"])], budget_cap=0)


def test_r1_disconnected_graph_star_cluster(star_clusterer_cls: Any) -> Any:
    """R1 Boundary: Disconnected graph with 0 edges produces singleton clusters for isolated nodes."""
    pairs: list[tuple[str, str]] = []
    weights: list[float] = []

    clusterer = star_clusterer_cls(cutoff=0.50, verify_clique=True)
    clusters = clusterer.cluster(pairs, edge_weights=weights)

    # 0 edges input yields 0 multi-node clusters
    assert len(clusters) == 0


def test_r1_duplicate_pairs_deduplication(multi_pass_blocker_cls: Any) -> Any:
    """R1 Boundary: Multiple passes generating the same candidate pair deduplicate without inflation."""
    from fuzzymatch.blocking.exact import ExactBlock

    df_l = pd.DataFrame({"key1": ["A", "B"], "key2": ["A", "B"]})
    df_r = pd.DataFrame({"key1": ["A", "B"], "key2": ["A", "B"]})

    # Pass 1 and Pass 2 will both find (0, 0) and (1, 1)
    pass1 = ExactBlock(["key1"])
    pass2 = ExactBlock(["key2"])

    blocker = multi_pass_blocker_cls(passes=[pass1, pass2], budget_cap=100)
    ci = blocker.block(df_l, df_r)

    # Candidate pairs must be set-unique
    assert len(ci.pairs) == 2
    assert len(ci.pairs) == len(set(ci.pairs))


def test_r1_star_clustering_threshold_extremes(star_clusterer_cls: Any) -> Any:
    """R1 Boundary: Threshold extremes (cutoff=0.0 and cutoff=1.0) behave deterministically."""
    pairs = [("A", "B"), ("B", "C")]
    weights = [0.80, 0.60]

    # cutoff = 1.0 (strict): nothing connects unless weight == 1.0
    c_strict = star_clusterer_cls(cutoff=1.0)
    clusters_strict = c_strict.cluster(pairs, edge_weights=weights)
    # No multi-node clusters should form if all weights < 1.0
    assert all(len(c) <= 1 for c in clusters_strict) or len(clusters_strict) == 0

    # cutoff = 0.0 (all connected)
    c_loose = star_clusterer_cls(cutoff=0.0)
    clusters_loose = c_loose.cluster(pairs, edge_weights=weights)
    assert len(clusters_loose) >= 1
