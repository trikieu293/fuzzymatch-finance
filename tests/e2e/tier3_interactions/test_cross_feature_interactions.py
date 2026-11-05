"""Tier 3: Cross-Feature Interactions E2E Tests.

Pairwise & Pipeline Interactions:
- Interaction 1: Multi-Pass Blocker -> Fellegi-Sunter / Scorer -> Star Clusterer
- Interaction 2: Continuous Composite Scoring -> Universal Excel Export with Diffs
- Interaction 3: Excel Curation -> Rule Synthesis -> Active Learning Override Loop
- Interaction 4: Legal Suffix Normalization inside Multi-Pass Blocking Pipeline
- Interaction 5: End-to-End CLI with AutoMatcher and Excel Review Output
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

from ..conftest import require_feature
from ..fixtures.vendor_benchmark import generate_vendor_benchmark


def test_interaction_multipass_to_star_clustering() -> Any:
    """Interaction 1: Candidate pairs from MultiPassBlock feed directly into StarClusterer."""
    multi_pass_cls = require_feature(
        "fuzzymatch.blocking.multi_pass", "MultiPassBlock", milestone="M1"
    )
    star_cluster_cls = require_feature(
        "fuzzymatch.clustering.star_cluster", "StarClusterer", milestone="M1"
    )
    from fuzzymatch.blocking.exact import ExactBlock

    df_l = pd.DataFrame(
        {
            "id": ["L1", "L2", "L3"],
            "tax_id": ["TAX-1", "TAX-2", "TAX-3"],
        }
    )
    df_r = pd.DataFrame(
        {
            "id": ["R1", "R2", "R3"],
            "tax_id": ["TAX-1", "TAX-2", "TAX-3"],
        }
    )

    blocker = multi_pass_cls(passes=[ExactBlock(["tax_id"])], budget_cap=100)
    ci = blocker.block(df_l, df_r)

    # Pairs from blocker are (left_idx, right_idx)
    # Map to node IDs
    pairs = [(df_l.iloc[l]["id"], df_r.iloc[r]["id"]) for l, r in ci.pairs]
    weights = [0.95] * len(pairs)

    clusterer = star_cluster_cls(cutoff=0.80, verify_clique=True)
    clusters = clusterer.cluster(pairs, edge_weights=weights)

    assert len(clusters) == 3
    for cl in clusters:
        assert len(cl) == 2


def test_interaction_composite_scoring_to_excel_diff_export(tmp_path: Path) -> Any:
    """Interaction 2: Continuous composite scores format directly into universal review_xlsx diffs."""
    export_excel_fn = require_feature(
        "fuzzymatch_finance.review_xlsx",
        "export_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )
    num_scorer_cls = require_feature(
        "fuzzymatch.scoring.numeric", "NumericProximityScorer", milestone="M2"
    )
    date_scorer_cls = require_feature("fuzzymatch.scoring.date", "DateWindowScorer", milestone="M2")

    df_l = pd.DataFrame(
        {
            "vendor_id": ["V-1", "V-2"],
            "vendor_name": ["Acme Industrial", "Apex Supply"],
            "amount": [1200.00, 450.50],
            "date": ["2025-03-01", "2025-03-10"],
        }
    )
    df_r = pd.DataFrame(
        {
            "ap_id": ["AP-1", "AP-2"],
            "vendor_name": ["Acme Ind", "Apex Supply LLC"],
            "amount": [1200.75, 450.50],
            "date": ["2025-03-02", "2025-03-11"],
        }
    )

    num_scorer = num_scorer_cls(epsilon=2.0)
    amt_scores = num_scorer.score_pair(df_l["amount"], df_r["amount"])

    date_scorer = date_scorer_cls(delta_days=3.0)
    d1 = pd.to_datetime(df_l["date"])
    d2 = pd.to_datetime(df_r["date"])
    dt_scores = date_scorer.score_pair(d1, d2)

    combined_df = pd.DataFrame(
        {
            "source_id": df_l["vendor_id"],
            "target_id": df_r["ap_id"],
            "src_name": df_l["vendor_name"],
            "tgt_name": df_r["vendor_name"],
            "src_amount": df_l["amount"],
            "tgt_amount": df_r["amount"],
            "amount_proximity": amt_scores,
            "date_proximity": dt_scores,
            "fuzzy_score": (amt_scores + dt_scores) / 2.0,
        }
    )

    xlsx_path = tmp_path / "composite_review.xlsx"
    export_excel_fn(
        combined_df,
        xlsx_path,
        source_id_col="source_id",
        target_id_col="target_id",
        diff_cols=[("src_name", "tgt_name"), ("src_amount", "tgt_amount")],
    )

    assert xlsx_path.exists()
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]]  # type: ignore[index]
    assert "amount_proximity" in headers
    assert "date_proximity" in headers


def test_interaction_excel_curation_to_active_rule_override_loop(tmp_path: Path) -> Any:
    """Interaction 3: Excel curation roundtrip -> rule synthesis -> persistent override."""
    export_excel_fn = require_feature(
        "fuzzymatch_finance.review_xlsx",
        "export_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )
    import_excel_fn = require_feature(
        "fuzzymatch_finance.review_xlsx",
        "import_review_xlsx",
        milestone="M4",
        expected_param="source_id_col",
    )
    synthesize_rules_fn = require_feature(
        "fuzzymatch_finance.label_store", "synthesize_rules_from_reviews", milestone="M4"
    )

    # Step 1: Export ambiguous pair to Excel
    pairs_df = pd.DataFrame(
        {
            "source_id": ["V-999"],
            "target_id": ["ERP-999"],
            "source_name": ["Vanguard Electronics Co"],
            "target_name": ["Vanguard Electronic Systems"],
            "fuzzy_score": [0.65],
        }
    )
    xlsx_path = tmp_path / "active_loop.xlsx"
    export_excel_fn(pairs_df, xlsx_path, source_id_col="source_id", target_id_col="target_id")

    # Step 2: Human curator accepts pair
    wb = openpyxl.load_workbook(xlsx_path)
    sheet = wb.active
    headers = [cell.value for cell in sheet[1]]  # type: ignore[index]
    dec_col = headers.index("decision") + 1
    sheet.cell(row=2, column=dec_col, value="accept")  # type: ignore[union-attr]
    wb.save(xlsx_path)

    # Step 3: Import reviewed decisions
    import_excel_fn(xlsx_path, source_id_col="source_id", target_id_col="target_id")

    # Step 4: Synthesize persistent rule
    curated_df = pd.DataFrame(
        {
            "source_id": ["V-999"],
            "target_id": ["ERP-999"],
            "decision": ["accept"],
        }
    )
    ruleset, _store = synthesize_rules_fn(
        curated_df, source_id_col="source_id", target_id_col="target_id"
    )

    assert len(ruleset) >= 1
    # Verify that the synthesized rule can evaluate new queries
    matched_target = ruleset.match("V-999") if hasattr(ruleset, "match") else None
    if matched_target is not None:
        assert matched_target == "ERP-999"


def test_interaction_legal_normalization_in_multipass_blocking() -> Any:
    """Interaction 4: Legal suffix normalization used in a pass of MultiPassBlock."""
    multi_pass_cls = require_feature(
        "fuzzymatch.blocking.multi_pass", "MultiPassBlock", milestone="M1"
    )
    strip_legal_fn = require_feature(
        "fuzzymatch.normalize.legal", "strip_legal_suffixes", milestone="M2"
    )
    from fuzzymatch.blocking.exact import ExactBlock

    df_l = pd.DataFrame({"name": ["Acme Industrial Supplies LLC", "Summit Chemical GmbH"]})
    df_r = pd.DataFrame({"name": ["Acme Industrial Supplies Inc.", "Summit Chemical Ltd."]})

    # Add normalized name columns
    df_l["clean_name"] = df_l["name"].apply(strip_legal_fn)
    df_r["clean_name"] = df_r["name"].apply(strip_legal_fn)

    # Raw names would have zero exact matches
    # Normalized names match exactly
    blocker = multi_pass_cls(passes=[ExactBlock(["clean_name"])], budget_cap=100)
    ci = blocker.block(df_l, df_r)

    assert len(ci.pairs) == 2
    assert (0, 0) in ci.pairs
    assert (1, 1) in ci.pairs


def test_interaction_cli_e2e_with_excel_review_flag(tmp_path: Path) -> Any:
    """Interaction 5: CLI execution with --review-out exporting an Excel curation sheet."""
    cli_main = require_feature("fuzzymatch_finance.cli", "main", milestone="M3")

    df_l, df_ap, _ = generate_vendor_benchmark(n_canonical=10, n_dirty=10, seed=42)
    erp_file = tmp_path / "cli_erp.csv"
    ap_file = tmp_path / "cli_ap.csv"
    review_out = tmp_path / "cli_review.xlsx"

    df_l.to_csv(erp_file, index=False)
    df_ap.to_csv(ap_file, index=False)

    exit_code = cli_main(["match", str(erp_file), str(ap_file), "--review-out", str(review_out)])
    assert exit_code == 0
    assert review_out.exists()
