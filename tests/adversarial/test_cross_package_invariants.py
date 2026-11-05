"""Cross-package Empirical Stress Testing & Adversarial Invariant Harness.

Adversarially tests mathematical invariants under Hypothesis (>=100 iterations per property):
1. NumericProximityScorer: metric axioms (symmetry, reflexivity, bounds), extreme values
   (inf, -inf, nan, 1e15, subnormals, negative offsets), and strict monotonicity.
2. DateWindowScorer: metric axioms, extreme years, timezone awareness, NaT/None robustness,
   and monotonic decay.
3. StarClusterer: intra-cluster clique invariant, non-transitivity on arbitrary intransitive
   triads, cycle graphs C_n (n >= 4), and random topologies under verify_clique=True.
4. FinancialCompositeScorer: probability simplex projection, strict gating, and score_pairs
   vectorized equivalence.
5. Financial Identifiers: ISO 7064 Mod 97-10, LEI, and CUSIP adjacent transposition sensitivity.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, ClassVar

import pandas as pd
from fuzzymatch.clustering.star_cluster import StarClusterer
from fuzzymatch.scoring.date import DateWindowScorer
from fuzzymatch.scoring.numeric import NumericProximityScorer
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from fuzzymatch_finance import (
    validate_cusip,
    validate_iban,
    validate_lei,
)
from fuzzymatch_finance.scoring import FinancialCompositeScorer

# ---------------------------------------------------------------------------
# 1. Adversarial Numeric Proximity Scorer Invariants
# ---------------------------------------------------------------------------


class TestAdversarialNumericProximityInvariants:
    """Rigorous adversarial stress-testing of NumericProximityScorer."""

    EXTREME_FLOATS: ClassVar[list[float]] = [
        float("inf"),
        -float("inf"),
        float("nan"),
        1e15,
        -1e15,
        1e-15,
        -1e-15,
        0.0,
        -0.0,
        1e308,
        -1e308,
        1e-308,
    ]

    @settings(max_examples=150)
    @given(
        x=st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.sampled_from(EXTREME_FLOATS),
        ),
        y=st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.sampled_from(EXTREME_FLOATS),
        ),
        eps=st.floats(min_value=1e-5, max_value=1e15),
    )
    def test_numeric_metric_axioms_and_extreme_values(self, x: float, y: float, eps: float) -> None:
        """Verify boundedness in [0.0, 1.0], symmetry, and reflexivity across extreme floats."""
        scorer = NumericProximityScorer(epsilon=eps)
        s_xy = scorer.score(x, y)
        s_yx = scorer.score(y, x)

        # 1. Boundedness: strictly in [0.0, 1.0]
        assert 0.0 <= s_xy <= 1.0, f"Bounds violation: {s_xy} for ({x}, {y})"
        assert 0.0 <= s_yx <= 1.0

        # 2. Symmetry: s(x, y) == s(y, x)
        assert math.isclose(s_xy, s_yx, abs_tol=1e-12), f"Symmetry broken: {s_xy} vs {s_yx}"

        # 3. Reflexivity: s(x, x) == 1.0 iff x is finite
        s_xx = scorer.score(x, x)
        if math.isfinite(x):
            assert math.isclose(s_xx, 1.0, abs_tol=1e-12), (
                f"Reflexivity failed for finite x={x}: {s_xx}"
            )
        else:
            assert s_xx == 0.0, f"Non-finite float must yield 0.0: got {s_xx} for x={x}"

    @settings(max_examples=150)
    @given(
        x=st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False),
        y1=st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False),
        y2=st.floats(min_value=-1e12, max_value=1e12, allow_nan=False, allow_infinity=False),
        eps=st.floats(min_value=1e-3, max_value=1e12, allow_nan=False, allow_infinity=False),
    )
    def test_numeric_proximity_strict_monotonicity(
        self, x: float, y1: float, y2: float, eps: float
    ) -> None:
        """As physical distance |x - y| increases, proximity score must monotonically decrease."""
        scorer = NumericProximityScorer(epsilon=eps)
        d1 = abs(x - y1)
        d2 = abs(x - y2)

        s1 = scorer.score(x, y1)
        s2 = scorer.score(x, y2)

        # Monotonicity with respect to true Euclidean / metric distance:
        # |x - y1| <= |x - y2| ==> s(x, y1) >= s(x, y2)
        if d1 <= d2:
            assert s1 >= s2 - 1e-12, (
                f"Monotonicity violated: d1={d1} (s1={s1}) <= d2={d2} (s2={s2})"
            )
        else:
            assert s2 >= s1 - 1e-12, (
                f"Monotonicity violated: d2={d2} (s2={s2}) <= d1={d1} (s1={s1})"
            )

        # Exact truncation invariant: if distance is strictly >= epsilon, score is 0.0
        if d1 >= eps:
            assert s1 == 0.0
        if d2 >= eps:
            assert s2 == 0.0

    @settings(max_examples=100)
    @given(
        values_l=st.lists(
            st.one_of(
                st.floats(allow_nan=True, allow_infinity=True),
                st.sampled_from(EXTREME_FLOATS),
                st.just(None),  # type: ignore[arg-type]
            ),
            min_size=1,
            max_size=30,
        ),
        eps=st.floats(min_value=1e-3, max_value=1e6),
    )
    def test_numeric_score_pair_matches_scalar_score(self, values_l: list[Any], eps: float) -> None:
        """score_pair matches scalar score elementwise for arbitrary mixed values."""
        # Reverse values_l for right side
        values_r = list(reversed(values_l))
        s_left = pd.Series(values_l)
        s_right = pd.Series(values_r)

        scorer = NumericProximityScorer(epsilon=eps)
        pair_scores = scorer.score_pair(s_left, s_right)

        for i in range(len(values_l)):
            scalar_s = scorer.score(values_l[i], values_r[i])
            vec_s = pair_scores.iloc[i]
            assert math.isclose(scalar_s, vec_s, abs_tol=1e-9), (
                f"Mismatch at index {i}: scalar={scalar_s} vs vec={vec_s} for ({values_l[i]}, {values_r[i]})"
            )


# ---------------------------------------------------------------------------
# 2. Adversarial Date Window Scorer Invariants
# ---------------------------------------------------------------------------


class TestAdversarialDateWindowInvariants:
    """Rigorous adversarial stress-testing of DateWindowScorer."""

    @settings(max_examples=150)
    @given(
        days1=st.integers(min_value=-36500, max_value=36500),
        days2=st.integers(min_value=-36500, max_value=36500),
        delta=st.floats(min_value=0.1, max_value=365.0),
    )
    def test_date_metric_axioms_across_broad_epochs(
        self, days1: int, days2: int, delta: float
    ) -> None:
        """Verify bounds [0.0, 1.0], symmetry, and reflexivity across arbitrary date ranges."""
        base = datetime.date(2025, 1, 1)
        d1 = base + datetime.timedelta(days=days1)
        d2 = base + datetime.timedelta(days=days2)

        scorer = DateWindowScorer(delta_days=delta)
        s12 = scorer.score(d1, d2)
        s21 = scorer.score(d2, d1)

        assert 0.0 <= s12 <= 1.0
        assert 0.0 <= s21 <= 1.0
        assert math.isclose(s12, s21, abs_tol=1e-12)

        # Reflexivity
        assert scorer.score(d1, d1) == 1.0
        assert scorer.score(d2, d2) == 1.0

    @settings(max_examples=150)
    @given(
        day_offset1=st.integers(min_value=0, max_value=200),
        day_offset2=st.integers(min_value=0, max_value=200),
        delta=st.floats(min_value=1.0, max_value=100.0),
    )
    def test_date_monotonicity_unambiguous_days(
        self, day_offset1: int, day_offset2: int, delta: float
    ) -> None:
        """For dates with day > 12 (no day/month swap ambiguity), score is strictly monotonic in calendar distance."""
        # Use base date with day=15
        base = datetime.date(2025, 6, 15)
        near_off = min(day_offset1, day_offset2)
        far_off = max(day_offset1, day_offset2)

        scorer = DateWindowScorer(delta_days=delta)
        s_near = scorer.score(base, base + datetime.timedelta(days=near_off))
        s_far = scorer.score(base, base + datetime.timedelta(days=far_off))

        # Because day/month transposition can only decrease effective distance,
        # we verify that if both target dates have day > 12, standard calendar monotonicity holds
        target_near = base + datetime.timedelta(days=near_off)
        target_far = base + datetime.timedelta(days=far_off)
        if target_near.day > 12 and target_far.day > 12:
            assert s_near >= s_far - 1e-12, (
                f"Date monotonicity violated: near={near_off} (score={s_near}) < far={far_off} (score={s_far})"
            )

    @settings(max_examples=100)
    @given(
        raw_val=st.sampled_from(
            [
                None,
                pd.NaT,
                "not-a-date",
                "",
                "   ",
                "9999-99-99",
                12345678901234567890,
                float("nan"),
                float("inf"),
            ]
        ),
        delta=st.floats(min_value=1.0, max_value=30.0),
    )
    def test_date_scorer_corrupted_inputs_neutrality(self, raw_val: Any, delta: float) -> None:
        """Corrupted, unparseable, or missing date values evaluate safely to 0.0 without exception."""
        scorer = DateWindowScorer(delta_days=delta)
        valid_date = datetime.date(2025, 1, 1)

        assert scorer.score(raw_val, valid_date) == 0.0
        assert scorer.score(valid_date, raw_val) == 0.0
        assert scorer.score(raw_val, raw_val) == 0.0

    @settings(max_examples=100)
    @given(
        dates=st.lists(
            st.sampled_from(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-10",
                    "2025-05-15",
                    None,
                    pd.NaT,
                    "corrupted",
                ]
            ),
            min_size=1,
            max_size=20,
        ),
        delta=st.floats(min_value=1.0, max_value=10.0),
    )
    def test_date_score_pair_matches_scalar_score(self, dates: list[Any], delta: float) -> None:
        """score_pair matches scalar score elementwise for arbitrary mixed dates."""
        s1 = pd.Series(dates)
        s2 = pd.Series(list(reversed(dates)))
        scorer = DateWindowScorer(delta_days=delta)
        pair_scores = scorer.score_pair(s1, s2)

        for i in range(len(dates)):
            sc = scorer.score(s1.iloc[i], s2.iloc[i])
            assert math.isclose(sc, pair_scores.iloc[i], abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 3. Adversarial Star Clustering Invariants
# ---------------------------------------------------------------------------


class TestAdversarialStarClusteringInvariants:
    """Rigorous adversarial stress-testing of StarClusterer."""

    @settings(max_examples=150)
    @given(
        w_ab=st.floats(min_value=0.5, max_value=1.0),
        w_bc=st.floats(min_value=0.5, max_value=1.0),
        w_ac=st.floats(min_value=0.0, max_value=0.499),
        tau=st.floats(min_value=0.5, max_value=0.9),
    )
    def test_star_clusterer_intransitive_triad_adversarial(
        self, w_ab: float, w_bc: float, w_ac: float, tau: float
    ) -> None:
        """Under verify_clique=True, no cluster can contain both A and C when w(A, C) < tau."""
        assume(w_ab >= tau and w_bc >= tau and w_ac < tau)
        pairs = [("A", "B"), ("B", "C"), ("A", "C")]
        weights = [w_ab, w_bc, w_ac]

        sc_strict = StarClusterer(cutoff=tau, verify_clique=True)
        clusters = sc_strict.cluster(pairs, weights)

        for clu in clusters:
            assert not ({"A", "C"}.issubset(clu)), (
                f"Transitive chaining occurred under verify_clique=True: cluster={clu}"
            )

    @settings(max_examples=100)
    @given(
        n=st.integers(min_value=4, max_value=12),
        edge_weight=st.floats(min_value=0.6, max_value=1.0),
        tau=st.floats(min_value=0.5, max_value=0.59),
    )
    def test_star_clusterer_cycle_graphs_no_false_transitivity(
        self, n: int, edge_weight: float, tau: float
    ) -> None:
        """On a cycle graph C_n (n >= 4) with all cycle edges >= tau and all chords < tau,
        every cluster under verify_clique=True must be a certified clique (size <= 2 for triangle-free cycles).
        """
        nodes = [f"v_{i}" for i in range(n)]
        pairs: list[tuple[str, str]] = []
        weights: list[float] = []

        # Only cycle edges exist
        for i in range(n):
            u = nodes[i]
            v = nodes[(i + 1) % n]
            pairs.append((u, v))
            weights.append(edge_weight)

        sc = StarClusterer(cutoff=tau, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        # 1. Verification of clique size bound for triangle-free graph
        for clu in clusters:
            assert len(clu) <= 2, f"Cluster {clu} on cycle C_{n} has size > 2 without chords!"
            if len(clu) == 2:
                u, v = list(clu)
                # Check they are actually adjacent on the cycle
                idx_u = int(u.split("_")[1])
                idx_v = int(v.split("_")[1])
                diff = abs(idx_u - idx_v)
                assert diff == 1 or diff == n - 1, (
                    f"Non-adjacent cycle nodes clustered together: {clu}"
                )

        # 2. Clusters are mutually disjoint
        assigned: set[str] = set()
        for clu in clusters:
            assert clu.isdisjoint(assigned), f"Clusters overlap on cycle C_{n}: {clu} & {assigned}"
            assigned.update(clu)

        # 3. All vertices are partitioned
        assert assigned == set(nodes)

    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow], deadline=None)
    @given(
        n_nodes=st.integers(min_value=3, max_value=12),
        edge_data=st.lists(
            st.tuples(
                st.integers(min_value=0, max_value=11),
                st.integers(min_value=0, max_value=11),
                st.floats(min_value=0.0, max_value=1.0),
            ),
            min_size=1,
            max_size=30,
        ),
        tau=st.floats(min_value=0.2, max_value=0.8),
    )
    def test_star_clusterer_arbitrary_random_graph_clique_invariant(
        self, n_nodes: int, edge_data: list[tuple[int, int, float]], tau: float
    ) -> None:
        """Fundamental Clique Invariant:
        For ANY graph and ANY cluster C under verify_clique=True, every pair u, v in C (u != v)
        must satisfy w(u, v) >= tau in the graph.
        """
        # Map indices to node strings
        node_names = [f"N_{i}" for i in range(n_nodes)]
        pairs: list[tuple[str, str]] = []
        weights: list[float] = []

        # Track max weight for each undirected pair
        edge_map: dict[frozenset[str], float] = {}

        for u_idx, v_idx, w in edge_data:
            u = node_names[u_idx % n_nodes]
            v = node_names[v_idx % n_nodes]
            pairs.append((u, v))
            weights.append(w)
            if u != v:
                fs = frozenset([u, v])
                edge_map[fs] = max(edge_map.get(fs, -1.0), w)

        sc = StarClusterer(cutoff=tau, verify_clique=True)
        clusters = sc.cluster(pairs, weights)

        # Verify Clique Invariant on all formed clusters
        for clu in clusters:
            clu_list = list(clu)
            for i in range(len(clu_list)):
                for j in range(i + 1, len(clu_list)):
                    u = clu_list[i]
                    v = clu_list[j]
                    max_w = edge_map.get(frozenset([u, v]), 0.0)
                    assert max_w >= tau, (
                        f"Clique invariant broken in cluster {clu}: edge ({u}, {v}) has weight {max_w} < cutoff {tau}"
                    )

    def test_star_clusterer_degenerate_structures(self) -> None:
        """Verify degenerate inputs: empty, singletons, self-loops, and star hub-and-spoke graphs."""
        sc = StarClusterer(cutoff=0.5, verify_clique=True)

        # 1. Empty pairs
        assert sc.cluster([]) == []

        # 2. Self loops only
        assert sc.cluster([("A", "A")], [1.0]) == [{"A"}]
        assert sc.cluster([("A", "A")], [0.2]) == []

        # 3. All edges below cutoff
        assert sc.cluster([("A", "B"), ("B", "C")], [0.4, 0.4]) == []

        # 4. Pure Star Graph with 1 Hub and 5 Spokes (No spoke-to-spoke connections)
        # Hub = H, Spokes = S0..S4.
        # With verify_clique=True, H can cluster with AT MOST ONE spoke (because spokes have no edges between them).
        spoke_pairs = [(f"S_{i}", "H") for i in range(5)]
        spoke_weights = [1.0] * 5
        clusters = sc.cluster(spoke_pairs, spoke_weights)

        # Hub is clustered with at most one spoke
        hub_cluster = next(c for c in clusters if "H" in c)
        assert len(hub_cluster) <= 2, (
            f"Hub clustered with multiple non-adjacent spokes: {hub_cluster}"
        )

        # All remaining spokes must be singleton clusters
        for clu in clusters:
            if "H" not in clu:
                assert len(clu) == 1


# ---------------------------------------------------------------------------
# 4. Adversarial Financial Composite Scorer Invariants
# ---------------------------------------------------------------------------


class TestAdversarialFinancialScorerInvariants:
    """Rigorous adversarial stress-testing of FinancialCompositeScorer."""

    @settings(max_examples=100)
    @given(
        w_text=st.floats(min_value=1e-3, max_value=1e3),
        w_amt=st.floats(min_value=1e-3, max_value=1e3),
        w_date=st.floats(min_value=1e-3, max_value=1e3),
    )
    def test_simplex_weight_projection_invariants(
        self, w_text: float, w_amt: float, w_date: float
    ) -> None:
        """Any strictly positive raw weights project exactly onto the probability simplex."""
        scorer = FinancialCompositeScorer(weights={"text": w_text, "amount": w_amt, "date": w_date})
        weights = scorer.weights
        assert math.isclose(sum(weights.values()), 1.0, abs_tol=1e-12)
        assert all(w >= 0.0 for w in weights.values())

    @settings(max_examples=100)
    @given(
        name1=st.sampled_from(["Acme Corp", "Beta LLC", "Gamma GmbH", "", None]),
        amt1=st.sampled_from([100.0, 500.25, 0.0, -50.0, None, float("nan")]),
        date1=st.sampled_from(["2025-01-10", "2025-06-15", None, "corrupted"]),
    )
    def test_gated_mode_strict_zeroing(self, name1: Any, amt1: Any, date1: Any) -> None:
        """In gated mode, if amount proximity or date window proximity is 0.0, composite is strictly 0.0."""
        scorer_gated = FinancialCompositeScorer(gated=True, amount_epsilon=5.0, date_delta_days=3.0)
        row1 = {"name": name1, "amount": amt1, "date": date1}
        # Discrepant second row: large amount difference (1000.0) ensuring s_amount == 0.0
        row2 = {"name": name1, "amount": 999999.0, "date": date1}

        score = scorer_gated.score_record_pair(row1, row2)
        assert score == 0.0, f"Gating failed to zero score: {score}"

    @settings(max_examples=100, deadline=None)
    @given(
        names=st.lists(
            st.sampled_from(["Alpha Inc", "Beta LLC", "Gamma Corp", "Delta GmbH"]),
            min_size=5,
            max_size=15,
        ),
        amounts=st.lists(st.floats(min_value=10.0, max_value=1000.0), min_size=5, max_size=15),
        dates=st.lists(
            st.sampled_from(["2025-01-01", "2025-01-05", "2025-01-12"]),
            min_size=5,
            max_size=15,
        ),
    )
    def test_financial_score_pairs_matches_scalar_records(
        self, names: list[str], amounts: list[float], dates: list[str]
    ) -> None:
        """Vectorized score_pairs matches scalar score_record_pair elementwise."""
        n = min(len(names), len(amounts), len(dates))
        df_l = pd.DataFrame(
            {
                "name": names[:n],
                "amount": amounts[:n],
                "date": dates[:n],
            }
        )
        # Right side has slightly perturbed amounts
        df_r = pd.DataFrame(
            {
                "name": list(reversed(names[:n])),
                "amount": [a + 0.5 for a in reversed(amounts[:n])],
                "date": list(reversed(dates[:n])),
            }
        )

        scorer = FinancialCompositeScorer(amount_epsilon=2.0, date_delta_days=5.0)
        vec_scores = scorer.score_pairs(df_l, df_r)

        for i in range(n):
            scalar_s = scorer.score_record_pair(df_l.iloc[i], df_r.iloc[i])
            # RapidFuzz cpdist uses float32 array backend, giving ~1e-7 precision relative to float64 scalar
            assert math.isclose(scalar_s, vec_scores.iloc[i], abs_tol=1e-6)


# ---------------------------------------------------------------------------
# 5. Adversarial Financial Check-Digit Transposition Invariants
# ---------------------------------------------------------------------------


class TestAdversarialCheckDigitInvariants:
    """Rigorous verification of adjacent transposition sensitivity across financial identifiers."""

    VALID_IBANS: ClassVar[list[str]] = [
        "GB82WEST12345698765432",
        "DE89370400440532013000",
        "FR1420041010050500013M02606",
    ]

    VALID_LEIS: ClassVar[list[str]] = [
        "5493006MHB84DD0ZWV18",
        "2138001W1ED2V2WQLG48",
        "549300O540N0427EP077",
    ]

    VALID_CUSIPS: ClassVar[list[str]] = [
        "037833100",  # Apple Inc
        "594918104",  # Microsoft Corp
        "38259P508",  # Google / Alphabet
    ]

    @settings(max_examples=100)
    @given(
        iban=st.sampled_from(VALID_IBANS),
        idx=st.integers(min_value=0, max_value=20),
    )
    def test_iban_adjacent_transposition_sensitivity(self, iban: str, idx: int) -> None:
        """Swapping two distinct adjacent characters must invalidate the IBAN (ISO 7064 error detection guarantee)."""
        clean = iban.replace(" ", "")
        assume(idx < len(clean) - 1)
        assume(clean[idx] != clean[idx + 1])

        chars = list(clean)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        corrupted = "".join(chars)

        assert validate_iban(corrupted) is False, (
            f"Transposition error not caught in IBAN: {clean} -> {corrupted}"
        )

    @settings(max_examples=100)
    @given(
        lei=st.sampled_from(VALID_LEIS),
        idx=st.integers(min_value=0, max_value=18),
    )
    def test_lei_adjacent_transposition_sensitivity(self, lei: str, idx: int) -> None:
        """Swapping two distinct adjacent characters in a 20-character LEI must invalidate it."""
        assume(lei[idx] != lei[idx + 1])
        chars = list(lei)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        corrupted = "".join(chars)

        assert validate_lei(corrupted) is False, (
            f"Transposition error not caught in LEI: {lei} -> {corrupted}"
        )

    @settings(max_examples=100)
    @given(
        cusip=st.sampled_from(VALID_CUSIPS),
        idx=st.integers(min_value=0, max_value=7),
    )
    def test_cusip_adjacent_transposition_sensitivity(self, cusip: str, idx: int) -> None:
        """Swapping two distinct adjacent characters in a CUSIP must invalidate it."""
        assume(cusip[idx] != cusip[idx + 1])
        chars = list(cusip)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        corrupted = "".join(chars)

        # Mod 10 Luhn-variant detects transposition of distinct adjacent characters
        assert validate_cusip(corrupted) is False, (
            f"Transposition error not caught in CUSIP: {cusip} -> {corrupted}"
        )
