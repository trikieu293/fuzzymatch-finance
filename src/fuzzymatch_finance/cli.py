"""Financial companion command line interface: `fuzzymatch-finance match`.

Provides zero-configuration financial entity resolution, bank transaction reconciliation,
financial summary metrics, and Excel curation export.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd
from fuzzymatch.blocking.exact import ExactBlock
from fuzzymatch.blocking.multi_pass import MultiPassBlock
from fuzzymatch.blocking.sorted_neighborhood import SortedNeighborhood
from fuzzymatch.clustering.star_cluster import StarClusterer
from fuzzymatch.normalize.legal import strip_legal_suffixes
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

from . import __version__
from .scoring import (
    _DEFAULT_AMOUNT_CANDIDATES,
    _DEFAULT_DATE_CANDIDATES,
    _DEFAULT_TEXT_CANDIDATES,
    FinancialCompositeScorer,
)

__all__ = [
    "FinanceMatchResults",
    "FinanceSummaryMetrics",
    "auto_match_finance",
    "main",
]

_ID_CANDIDATES = (
    "bank_id",
    "ledger_id",
    "vendor_id",
    "ap_id",
    "invoice_id",
    "transaction_id",
    "txn_id",
    "account_id",
    "id",
    "canonical_id",
    "source_id",
    "target_id",
)

_DECISIONS = '"accept,reject,pending"'


@dataclass(frozen=True)
class FinanceSummaryMetrics:
    """Financial reconciliation summary metrics."""

    left_count: int
    right_count: int
    left_dollar_volume: float
    right_dollar_volume: float
    total_dollar_volume: float
    reconciled_count: int
    reconciled_dollar_volume: float
    reconciliation_rate_pct: float
    unmatched_left_count: int
    unmatched_right_count: int
    unmatched_dollar_volume: float
    avg_confidence: float

    def format_terminal(self, epsilon: float, delta_days: float, threshold: float) -> str:
        """Format metrics into a clean, professional ASCII summary table."""
        has_volume = self.left_dollar_volume > 0.0 or self.right_dollar_volume > 0.0

        l_vol_str = f"$ {self.left_dollar_volume:,.2f}" if has_volume else "N/A"
        r_vol_str = f"$ {self.right_dollar_volume:,.2f}" if has_volume else "N/A"
        tot_vol_str = f"$ {self.total_dollar_volume:,.2f}" if has_volume else "N/A"
        rec_vol_str = f"$ {self.reconciled_dollar_volume:,.2f}" if has_volume else "N/A"
        unm_vol_str = f"$ {self.unmatched_dollar_volume:,.2f}" if has_volume else "N/A"

        sep = "=" * 80
        thin_sep = "-" * 80

        lines = [
            sep,
            "                    FUZZYMATCH-FINANCE RECONCILIATION REPORT                    ",
            sep,
            f"  Confidence Threshold: {threshold:.2f}  |  Amount Epsilon: ${epsilon:.2f}  |  Date Delta: {delta_days:.1f} days",
            thin_sep,
            "1. Total Ingested Transactions:",
            f"   • Left Table:                  {self.left_count:>6} records   |   Volume: {l_vol_str}",
            f"   • Right Table:                 {self.right_count:>6} records   |   Volume: {r_vol_str}",
            f"   • Total Ingested:              {self.left_count + self.right_count:>6} records   |   Volume: {tot_vol_str}",
            "",
            "2. Reconciled Transactions:",
            f"   • Reconciled Transactions:     {self.reconciled_count:>6} matched pairs ({self.reconciliation_rate_pct:.1f}% rate)",
            f"   • Reconciled Dollar Volume:    {rec_vol_str}",
            "",
            "3. Unmatched Variance:",
            f"   • Unmatched Left Records:      {self.unmatched_left_count:>6} records",
            f"   • Unmatched Right Records:     {self.unmatched_right_count:>6} records",
            f"   • Unmatched Variance Volume:   {unm_vol_str}",
            "",
            "4. Match Quality & Confidence:",
            f"   • Average Match Confidence:    {self.avg_confidence:.3f} ({self.avg_confidence * 100:.1f}%)",
            sep,
        ]
        return "\n".join(lines)


@dataclass
class FinanceMatchResults:
    """Container for financial match results and curation outputs."""

    matches: pd.DataFrame
    pairs: list[tuple[Any, Any]]
    clusters: list[set[Any]]
    summary: FinanceSummaryMetrics
    review_candidates: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return DataFrame of reconciled matches."""
        return self.matches.copy()

    def export_review_xlsx(self, path: str | Path) -> Path:
        """Export candidate pairs to an Excel sheet with dropdown curation validation."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "review"

        review_df = self.review_candidates.copy()
        if review_df.empty:
            review_df = pd.DataFrame(
                columns=[
                    "pair_id",
                    "decision",
                    "source_id",
                    "target_id",
                    "source_name",
                    "target_name",
                    "source_amount",
                    "target_amount",
                    "amount_diff",
                    "source_date",
                    "target_date",
                    "date_diff_days",
                    "fuzzy_score",
                    "explanation",
                ]
            )

        # Write headers
        headers = list(review_df.columns)
        ws.append(headers)

        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
        for col_idx in range(1, len(headers) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Write data rows
        for row in review_df.itertuples(index=False):
            ws.append(list(row))

        n_rows = len(review_df)
        ws.freeze_panes = "A2"

        # Setup dropdown DataValidation on 'decision'
        if "decision" in headers:
            dec_idx = headers.index("decision") + 1
            col_letter = ws.cell(row=1, column=dec_idx).column_letter
            dv = DataValidation(type="list", formula1=_DECISIONS, allow_blank=False)
            dv.error = "Please select accept, reject, or pending"
            dv.errorTitle = "Invalid Review Decision"
            ws.add_data_validation(dv)
            dv.add(f"{col_letter}2:{col_letter}{max(n_rows + 1, 2)}")

        # Setup color scale on 'fuzzy_score'
        if "fuzzy_score" in headers and n_rows > 0:
            score_idx = headers.index("fuzzy_score") + 1
            col_letter = ws.cell(row=1, column=score_idx).column_letter
            ws.conditional_formatting.add(
                f"{col_letter}2:{col_letter}{n_rows + 1}",
                ColorScaleRule(
                    start_type="min",
                    start_color="FFF5F5",
                    end_type="max",
                    end_color="B7E1CD",
                ),
            )

        # Auto-adjust column widths
        for col in ws.columns:
            col_letter = col[0].column_letter
            max_len = max(len(str(c.value or "")) for c in col)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

        wb.save(out_path)
        return out_path


def _resolve_column_name(
    df: pd.DataFrame,
    explicit_col: str | None,
    candidates: tuple[str, ...],
    semantic_name: str,
) -> str | None:
    """Resolve column name by explicit request, candidate list, or substring match."""
    if explicit_col is not None:
        if explicit_col not in df.columns:
            raise ValueError(
                f"Specified {semantic_name} column '{explicit_col}' not found in table columns: {list(df.columns)}"
            )
        return explicit_col

    lower_to_orig = {str(c).lower().strip(): str(c) for c in df.columns}
    # 1. Exact candidate match
    for cand in candidates:
        if cand.lower() in lower_to_orig:
            return lower_to_orig[cand.lower()]

    # 2. Substring heuristic match
    for col_lower, col_orig in lower_to_orig.items():
        if semantic_name == "amount" and (
            "amount" in col_lower or "total" in col_lower or col_lower.endswith("_amt")
        ):
            return col_orig
        if semantic_name == "date" and (
            "date" in col_lower or "time" in col_lower or col_lower.endswith("_dt")
        ):
            return col_orig
        if semantic_name == "text" and (
            "name" in col_lower
            or "desc" in col_lower
            or "memo" in col_lower
            or "payee" in col_lower
        ):
            return col_orig
        if semantic_name == "id" and ("_id" in col_lower or col_lower.endswith("id")):
            return col_orig

    return None


def auto_match_finance(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    amount_col_left: str | None = None,
    amount_col_right: str | None = None,
    date_col_left: str | None = None,
    date_col_right: str | None = None,
    text_col_left: str | None = None,
    text_col_right: str | None = None,
    id_col_left: str | None = None,
    id_col_right: str | None = None,
    epsilon: float = 1.0,
    delta_days: float = 3.0,
    threshold: float = 0.7,
    budget_cap: int = 50_000,
    weights: dict[str, float] | None = None,
    gated: bool = False,
    normalize_legal: bool = True,
) -> FinanceMatchResults:
    """Zero-configuration automated financial entity and transaction reconciliation.

    Profiles tables, auto-detects financial columns, executes budget-capped multi-pass
    blocking, scores via FinancialCompositeScorer, and eliminates transitive false links
    using StarClusterer.
    """
    id_l = _resolve_column_name(df_left, id_col_left, _ID_CANDIDATES, "id")
    id_r = _resolve_column_name(df_right, id_col_right, _ID_CANDIDATES, "id")

    txt_l = _resolve_column_name(df_left, text_col_left, _DEFAULT_TEXT_CANDIDATES, "text")
    txt_r = _resolve_column_name(df_right, text_col_right, _DEFAULT_TEXT_CANDIDATES, "text")

    amt_l = _resolve_column_name(df_left, amount_col_left, _DEFAULT_AMOUNT_CANDIDATES, "amount")
    amt_r = _resolve_column_name(df_right, amount_col_right, _DEFAULT_AMOUNT_CANDIDATES, "amount")

    dt_l = _resolve_column_name(df_left, date_col_left, _DEFAULT_DATE_CANDIDATES, "date")
    dt_r = _resolve_column_name(df_right, date_col_right, _DEFAULT_DATE_CANDIDATES, "date")

    # Ingested volume metrics
    l_vol = (
        float(pd.to_numeric(df_left[amt_l], errors="coerce").fillna(0.0).abs().sum())
        if amt_l
        else 0.0
    )
    r_vol = (
        float(pd.to_numeric(df_right[amt_r], errors="coerce").fillna(0.0).abs().sum())
        if amt_r
        else 0.0
    )
    tot_vol = l_vol + r_vol

    # Handle empty tables
    if df_left.empty or df_right.empty:
        summary = FinanceSummaryMetrics(
            left_count=len(df_left),
            right_count=len(df_right),
            left_dollar_volume=l_vol,
            right_dollar_volume=r_vol,
            total_dollar_volume=tot_vol,
            reconciled_count=0,
            reconciled_dollar_volume=0.0,
            reconciliation_rate_pct=0.0,
            unmatched_left_count=len(df_left),
            unmatched_right_count=len(df_right),
            unmatched_dollar_volume=tot_vol,
            avg_confidence=0.0,
        )
        empty_matches = pd.DataFrame(
            columns=[
                "cluster_id",
                "left_id",
                "right_id",
                "score",
                "left_name",
                "right_name",
                "left_amount",
                "right_amount",
                "left_date",
                "right_date",
            ]
        )
        return FinanceMatchResults(
            matches=empty_matches,
            pairs=[],
            clusters=[],
            summary=summary,
            review_candidates=pd.DataFrame(),
        )

    # Prepare normalized text keys for blocking
    df_l_work = df_left.reset_index(drop=True)
    df_r_work = df_right.reset_index(drop=True)

    clean_col = "_clean_auto_txt"
    df_l_work[clean_col] = (
        df_l_work[txt_l].fillna("").astype(str).map(strip_legal_suffixes).str.lower().str.strip()
        if txt_l
        else ""
    )
    df_r_work[clean_col] = (
        df_r_work[txt_r].fillna("").astype(str).map(strip_legal_suffixes).str.lower().str.strip()
        if txt_r
        else ""
    )

    # Candidate generation: multi-pass or cartesian product
    n_prod = len(df_left) * len(df_right)
    if n_prod <= 5000:
        candidate_pairs = [(i, j) for i in range(len(df_left)) for j in range(len(df_right))]
    else:
        passes: list[Any] = []
        if txt_l and txt_r:
            passes.append(ExactBlock([clean_col]))
            passes.append(SortedNeighborhood([clean_col], window=5))
        else:
            passes.append(SortedNeighborhood([clean_col], window=5))

        blocker = MultiPassBlock(passes=passes, budget_cap=budget_cap)
        ci = blocker.block(df_l_work, df_r_work)
        candidate_pairs = list(ci.pairs)
        if not candidate_pairs:
            candidate_pairs = [
                (i, j) for i in range(min(len(df_left), 50)) for j in range(min(len(df_right), 50))
            ]

    # Setup Scorer with adaptive weights
    if weights is not None:
        active_weights = weights
    else:
        if amt_l and amt_r and dt_l and dt_r:
            active_weights = {"text": 0.5, "amount": 0.3, "date": 0.2}
        elif amt_l and amt_r:
            active_weights = {"text": 0.7, "amount": 0.3, "date": 0.0}
        elif dt_l and dt_r:
            active_weights = {"text": 0.8, "amount": 0.0, "date": 0.2}
        else:
            active_weights = {"text": 1.0, "amount": 0.0, "date": 0.0}

    primary_text_col = txt_l or "name"
    primary_amount_col = amt_l or "amount"
    primary_date_col = dt_l or "date"

    scorer = FinancialCompositeScorer(
        amount_epsilon=epsilon,
        date_delta_days=delta_days,
        weights=active_weights,
        text_col=primary_text_col,
        amount_col=primary_amount_col,
        date_col=primary_date_col,
        gated=gated,
        normalize_legal=normalize_legal,
    )

    # Check for ID overlap
    all_l_ids = {
        str(df_left.iloc[i][id_l]) if id_l else str(df_left.index[i]) for i in range(len(df_left))
    }
    all_r_ids = {
        str(df_right.iloc[j][id_r]) if id_r else str(df_right.index[j])
        for j in range(len(df_right))
    }
    has_id_overlap = bool(all_l_ids & all_r_ids)

    scored_records: list[dict[str, Any]] = []
    for l_idx, r_idx in candidate_pairs:
        row_l = df_left.iloc[l_idx]
        row_r = df_right.iloc[r_idx]

        l_val_id = row_l[id_l] if id_l else df_left.index[l_idx]
        r_val_id = row_r[id_r] if id_r else df_right.index[r_idx]
        l_val_txt = row_l[txt_l] if txt_l else None
        r_val_txt = row_r[txt_r] if txt_r else None
        l_val_amt = row_l[amt_l] if amt_l else None
        r_val_amt = row_r[amt_r] if amt_r else None
        l_val_dt = row_l[dt_l] if dt_l else None
        r_val_dt = row_r[dt_r] if dt_r else None

        # Build normalized candidate row maps for scorer
        score_left = {
            primary_text_col: l_val_txt,
            primary_amount_col: l_val_amt,
            primary_date_col: l_val_dt,
        }
        score_right = {
            primary_text_col: r_val_txt,
            primary_amount_col: r_val_amt,
            primary_date_col: r_val_dt,
        }
        score = scorer.score_record_pair(score_left, score_right)

        # Amount diff calculation
        amt_diff = None
        if l_val_amt is not None and r_val_amt is not None:
            try:
                amt_diff = abs(float(l_val_amt) - float(r_val_amt))
            except (ValueError, TypeError):
                pass

        # Date diff calculation
        date_diff = None
        if l_val_dt is not None and r_val_dt is not None:
            try:
                d1 = pd.to_datetime(l_val_dt)
                d2 = pd.to_datetime(r_val_dt)
                date_diff = abs((d1 - d2).total_seconds()) / 86400.0
            except (ValueError, TypeError, OverflowError):
                date_diff = None

        scored_records.append(
            {
                "left_idx": l_idx,
                "right_idx": r_idx,
                "left_id": l_val_id,
                "right_id": r_val_id,
                "left_name": l_val_txt,
                "right_name": r_val_txt,
                "left_amount": l_val_amt,
                "right_amount": r_val_amt,
                "amount_diff": amt_diff,
                "left_date": l_val_dt,
                "right_date": r_val_dt,
                "date_diff_days": date_diff,
                "score": score,
                "fuzzy_score": score,
                "explanation": f"Score {score:.2f} | Amt diff: {amt_diff} | Date diff: {date_diff}",
            }
        )

    scored_df = pd.DataFrame(scored_records)
    passing_pairs = scored_df[scored_df["score"] >= threshold].copy()
    if passing_pairs.empty:
        matched_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "left_id",
                "right_id",
                "score",
                "left_name",
                "right_name",
                "left_amount",
                "right_amount",
                "left_date",
                "right_date",
            ]
        )
        clusters: list[set[Any]] = []
    else:
        edge_pairs = [
            (
                f"L_{r['left_id']}" if has_id_overlap else str(r["left_id"]),
                f"R_{r['right_id']}" if has_id_overlap else str(r["right_id"]),
            )
            for _, r in passing_pairs.iterrows()
        ]
        weights_list = passing_pairs["score"].tolist()
        clusterer = StarClusterer(cutoff=threshold, verify_clique=True)
        clusters = clusterer.cluster(edge_pairs, edge_weights=weights_list)

        node_to_cluster: dict[Any, str] = {}
        for clu_idx, clu_set in enumerate(clusters):
            clu_name = f"CLU-{clu_idx + 1:04d}"
            for node in clu_set:
                node_to_cluster[node] = clu_name

        # Map cluster_id ensuring both endpoints belong to the same cluster
        assigned_clusters: list[str] = []
        for _, r in passing_pairs.iterrows():
            nl = f"L_{r['left_id']}" if has_id_overlap else str(r["left_id"])
            nr = f"R_{r['right_id']}" if has_id_overlap else str(r["right_id"])
            cl_l = node_to_cluster.get(nl)
            cl_r = node_to_cluster.get(nr)
            if cl_l is not None and cl_l == cl_r:
                assigned_clusters.append(cl_l)
            else:
                assigned_clusters.append("CLU-UNASSIGNED")
        passing_pairs["cluster_id"] = assigned_clusters

        # Filter out pairs that do not share the same cluster
        valid_cluster_pairs = passing_pairs[passing_pairs["cluster_id"] != "CLU-UNASSIGNED"]

        # Greedy 1-to-1 bipartite match resolution:
        # Sort by score descending and iteratively select mutually disjoint left and right records
        sorted_pairs = valid_cluster_pairs.sort_values("score", ascending=False)
        selected_indices: list[Any] = []
        seen_left: set[Any] = set()
        seen_right: set[Any] = set()

        for idx, r in sorted_pairs.iterrows():
            lid = r["left_id"]
            rid = r["right_id"]
            if lid not in seen_left and rid not in seen_right:
                selected_indices.append(idx)
                seen_left.add(lid)
                seen_right.add(rid)

        matched_df = sorted_pairs.loc[selected_indices].reset_index(drop=True)

    # Compute Reconciled & Variance Metrics
    rec_count = len(matched_df)
    rec_vol = 0.0
    if amt_l and not matched_df.empty and "left_amount" in matched_df.columns:
        rec_vol = float(
            pd.to_numeric(matched_df["left_amount"], errors="coerce").fillna(0.0).abs().sum()
        )

    rec_vol_right = 0.0
    if amt_r and not matched_df.empty and "right_amount" in matched_df.columns:
        rec_vol_right = float(
            pd.to_numeric(matched_df["right_amount"], errors="coerce").fillna(0.0).abs().sum()
        )

    rec_rate_pct = (
        (rec_vol / l_vol * 100.0)
        if l_vol > 0.0
        else ((rec_count / len(df_left) * 100.0) if len(df_left) > 0 else 0.0)
    )
    unm_l_count = len(df_left) - (matched_df["left_id"].nunique() if not matched_df.empty else 0)
    unm_r_count = len(df_right) - (matched_df["right_id"].nunique() if not matched_df.empty else 0)

    unm_l_vol = max(0.0, l_vol - rec_vol) if amt_l else 0.0
    unm_r_vol = max(0.0, r_vol - rec_vol_right) if amt_r else 0.0
    unm_vol = unm_l_vol + unm_r_vol

    avg_conf = float(matched_df["score"].mean()) if not matched_df.empty else 0.0

    summary = FinanceSummaryMetrics(
        left_count=len(df_left),
        right_count=len(df_right),
        left_dollar_volume=l_vol,
        right_dollar_volume=r_vol,
        total_dollar_volume=tot_vol,
        reconciled_count=rec_count,
        reconciled_dollar_volume=rec_vol,
        reconciliation_rate_pct=rec_rate_pct,
        unmatched_left_count=unm_l_count,
        unmatched_right_count=unm_r_count,
        unmatched_dollar_volume=unm_vol,
        avg_confidence=avg_conf,
    )

    # Prepare review candidates
    review_slice = scored_df.sort_values("score", ascending=False).reset_index(drop=True)
    review_candidates = pd.DataFrame(
        {
            "pair_id": [f"REV-{k + 1:04d}" for k in range(len(review_slice))],
            "decision": ["pending"] * len(review_slice),
            "source_id": review_slice["left_id"],
            "target_id": review_slice["right_id"],
            "source_name": review_slice["left_name"],
            "target_name": review_slice["right_name"],
            "source_amount": review_slice["left_amount"],
            "target_amount": review_slice["right_amount"],
            "amount_diff": review_slice["amount_diff"],
            "source_date": review_slice["left_date"],
            "target_date": review_slice["right_date"],
            "date_diff_days": review_slice["date_diff_days"],
            "fuzzy_score": review_slice["score"],
            "explanation": review_slice["explanation"],
        }
    )

    pairs_list = (
        [(row["left_id"], row["right_id"]) for _, row in matched_df.iterrows()]
        if not matched_df.empty
        else []
    )

    return FinanceMatchResults(
        matches=matched_df,
        pairs=pairs_list,
        clusters=clusters,
        summary=summary,
        review_candidates=review_candidates,
    )


def _load_tabular_data(path_str: str) -> pd.DataFrame:
    """Load CSV or Excel tabular file safely."""
    p = Path(path_str)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path_str}")
    suffix = p.suffix.lower()
    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix == ".tsv" else ","
        try:
            return pd.read_csv(p, sep=sep)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV file '{path_str}': {e}") from e
    elif suffix in (".xlsx", ".xls"):
        try:
            return pd.read_excel(p)
        except Exception as e:
            raise ValueError(f"Failed to parse Excel file '{path_str}': {e}") from e
    else:
        try:
            return pd.read_csv(p)
        except Exception as e:
            raise ValueError(f"Unsupported file format '{path_str}': {e}") from e


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for `fuzzymatch-finance`."""
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="fuzzymatch-finance",
        description="Financial entity resolution and transaction reconciliation CLI.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available commands")

    match_p = subparsers.add_parser("match", help="Match and reconcile two financial datasets")
    match_p.add_argument("pos_left", nargs="?", default=None, help="Path to left table (CSV/Excel)")
    match_p.add_argument(
        "pos_right", nargs="?", default=None, help="Path to right table (CSV/Excel)"
    )
    match_p.add_argument("--left", "-l", dest="flag_left", default=None, help="Path to left table")
    match_p.add_argument(
        "--right", "-r", dest="flag_right", default=None, help="Path to right table"
    )
    match_p.add_argument(
        "--output", "-o", default=None, help="Output destination path for reconciled pairs"
    )
    match_p.add_argument(
        "--review-output",
        "--review-out",
        dest="review_output",
        default=None,
        help="Output destination for manual curation Excel sheet (.xlsx)",
    )

    match_p.add_argument("--amount-col-left", default=None, help="Left amount column name")
    match_p.add_argument("--amount-col-right", default=None, help="Right amount column name")
    match_p.add_argument("--date-col-left", default=None, help="Left date column name")
    match_p.add_argument("--date-col-right", default=None, help="Right date column name")
    match_p.add_argument("--text-col-left", default=None, help="Left text/name column name")
    match_p.add_argument("--text-col-right", default=None, help="Right text/name column name")
    match_p.add_argument("--id-col-left", default=None, help="Left ID column name")
    match_p.add_argument("--id-col-right", default=None, help="Right ID column name")

    match_p.add_argument(
        "--epsilon",
        "--amount-epsilon",
        type=float,
        default=1.0,
        dest="epsilon",
        help="Numeric amount tolerance (+/- $) [default: 1.0]",
    )
    match_p.add_argument(
        "--delta-days",
        "--date-delta-days",
        type=float,
        default=3.0,
        dest="delta_days",
        help="Date window tolerance (+/- days) [default: 3.0]",
    )
    match_p.add_argument(
        "--threshold", type=float, default=0.7, help="Match confidence threshold [default: 0.7]"
    )
    match_p.add_argument(
        "--budget-cap", type=int, default=50_000, help="Candidate budget cap [default: 50000]"
    )
    match_p.add_argument("--gated", action="store_true", default=False, help="Enable strict gating")
    match_p.add_argument(
        "--explain", action="store_true", default=False, help="Display match score explanations"
    )

    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 2

    if args.subcommand != "match":
        parser.print_help()
        return 2

    left_path = args.flag_left or args.pos_left
    right_path = args.flag_right or (args.pos_right if args.flag_left is None else args.pos_left)

    if not left_path or not right_path:
        sys.stderr.write("Error: Both left and right input files must be specified.\n")
        match_p.print_help()
        return 2

    try:
        df_left = _load_tabular_data(left_path)
        df_right = _load_tabular_data(right_path)
    except (ValueError, OSError, RuntimeError) as e:
        sys.stderr.write(f"Error loading inputs: {e}\n")
        return 1

    try:
        results = auto_match_finance(
            df_left=df_left,
            df_right=df_right,
            amount_col_left=args.amount_col_left,
            amount_col_right=args.amount_col_right,
            date_col_left=args.date_col_left,
            date_col_right=args.date_col_right,
            text_col_left=args.text_col_left,
            text_col_right=args.text_col_right,
            id_col_left=args.id_col_left,
            id_col_right=args.id_col_right,
            epsilon=args.epsilon,
            delta_days=args.delta_days,
            threshold=args.threshold,
            budget_cap=args.budget_cap,
            gated=args.gated,
        )
    except (ValueError, TypeError, KeyError, RuntimeError) as e:
        sys.stderr.write(f"Matching error: {e}\n")
        return 1

    # Print summary metrics to stdout
    report_text = results.summary.format_terminal(
        epsilon=args.epsilon,
        delta_days=args.delta_days,
        threshold=args.threshold,
    )
    sys.stdout.write(report_text + "\n")

    # Print explanation if requested
    if args.explain and not results.matches.empty:
        sys.stdout.write("\nMatch Explanations (Top 10):\n")
        for _idx, row in results.matches.head(10).iterrows():
            sys.stdout.write(
                f"  [{row.get('cluster_id', 'N/A')}] {row.get('left_id', '')} <-> {row.get('right_id', '')} | "
                f"Score: {row.get('score', 0.0):.3f} | {row.get('left_name', '')} vs {row.get('right_name', '')}\n"
            )

    # Save output matches if requested
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix in (".xlsx", ".xls"):
            results.matches.to_excel(out_path, index=False)
        else:
            results.matches.to_csv(out_path, index=False)
        sys.stdout.write(f"Saved {len(results.matches)} matched records to: {out_path}\n")

    # Save review Excel workbook if requested
    if args.review_output:
        rev_path = results.export_review_xlsx(args.review_output)
        sys.stdout.write(
            f"Exported review sheet with {len(results.review_candidates)} candidate pairs to: {rev_path}\n"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
