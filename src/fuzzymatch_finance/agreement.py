"""Reviewer-agreement tracking against a fixed gold set (v1.2 Phase B).

Cohen's kappa measures reviewer consistency beyond chance. Tracking it per
curation cycle catches reviewer drift: falling kappa means accept/reject
judgements are diverging from the calibrated gold standard, which silently
poisons the labels feeding the learning loop.
"""

from __future__ import annotations


def cohen_kappa(rater_a: list[str], rater_b: list[str]) -> float:
    """Cohen's kappa between two aligned rating sequences.

    Returns 1.0 when both raters show zero marginal variance (degenerate
    perfect agreement), the standard kappa otherwise.

    Raises:
        ValueError: On empty inputs or length mismatch.
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(f"length mismatch: {len(rater_a)} vs {len(rater_b)}")
    if not rater_a or not rater_b:
        raise ValueError("cohen_kappa requires non-empty ratings")

    n = len(rater_a)
    categories = sorted(set(rater_a) | set(rater_b))
    po = sum(1 for a, b in zip(rater_a, rater_b) if a == b) / n
    pa = {c: rater_a.count(c) / n for c in categories}
    pb = {c: rater_b.count(c) / n for c in categories}
    pe = sum(pa[c] * pb[c] for c in categories)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def gold_set_report(
    gold: dict[str, str],
    decisions: dict[str, str],
) -> dict[str, object]:
    """Agreement of curated decisions against the gold standard.

    Only sources present in both maps are scored.

    Returns ``{"kappa": float, "n": int, "agree_rate": float}``.

    Raises:
        ValueError: If there is no overlap between gold and decisions.
    """
    common = sorted(set(gold) & set(decisions))
    if not common:
        raise ValueError("no overlap between gold set and decisions")
    a = [gold[k] for k in common]
    b = [decisions[k] for k in common]
    agree_rate = sum(1 for x, y in zip(a, b) if x == y) / len(common)
    return {
        "kappa": cohen_kappa(a, b),
        "n": len(common),
        "agree_rate": agree_rate,
    }
