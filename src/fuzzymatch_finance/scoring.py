"""Financial composite multi-attribute scoring combining text, amount, and date.

Provides composite similarity evaluation for financial entity resolution (AP/AR
invoices, vendor master records, bank feed transaction lines).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from fuzzymatch.normalize.legal import strip_legal_suffixes
from fuzzymatch.scoring.date import DateWindowScorer, date_window_score
from fuzzymatch.scoring.numeric import NumericProximityScorer, amount_proximity_score
from fuzzymatch.scoring.rapidfuzz_wrapper import RapidFuzzScorer

__all__ = [
    "FinancialCompositeScorer",
    "amount_proximity_score",
    "date_window_score",
]

_DEFAULT_TEXT_CANDIDATES = (
    "name",
    "vendor_name",
    "customer_name",
    "account_name",
    "description",
    "src_name",
    "tgt_name",
)

_DEFAULT_AMOUNT_CANDIDATES = (
    "amount",
    "cleared_amount",
    "invoice_amount",
    "total",
    "total_amount",
    "src_amount",
    "tgt_amount",
)

_DEFAULT_DATE_CANDIDATES = (
    "date",
    "cleared_date",
    "invoice_date",
    "transaction_date",
    "src_date",
    "tgt_date",
)


def _extract_field(
    row: Mapping[str, Any] | pd.Series,
    primary_key: str,
    candidates: tuple[str, ...],
) -> Any:
    """Extract a field value by primary name or fallback candidates."""
    if hasattr(row, "get"):
        val = row.get(primary_key)
        if val is not None:
            return val
        for cand in candidates:
            val = row.get(cand)
            if val is not None:
                return val
    elif hasattr(row, "__getitem__"):
        try:
            return row[primary_key]
        except (KeyError, IndexError):
            for cand in candidates:
                try:
                    return row[cand]
                except (KeyError, IndexError):
                    continue
    return None


def _resolve_column(
    df: pd.DataFrame,
    primary_col: str,
    candidates: tuple[str, ...],
) -> str | None:
    """Resolve column name by primary_col first, then fallback candidates."""
    return (
        primary_col
        if primary_col in df.columns
        else next((c for c in candidates if c in df.columns), None)
    )


class FinancialCompositeScorer:
    """Multi-attribute composite scorer for financial records.

    Combines:
    1. String similarity on entity names (with legal entity suffix normalization).
    2. Continuous numeric amount proximity within +/- amount_epsilon.
    3. Continuous date window proximity within +/- date_delta_days.

    Supports convex simplex weighting and optional strict gating.
    """

    def __init__(
        self,
        amount_epsilon: float = 1.0,
        date_delta_days: float = 3.0,
        weights: dict[str, float] | None = None,
        text_scorer: str | RapidFuzzScorer | None = None,
        text_col: str = "name",
        amount_col: str = "amount",
        date_col: str = "date",
        gated: bool = False,
        normalize_legal: bool = True,
    ) -> None:
        if amount_epsilon <= 0.0:
            raise ValueError(f"amount_epsilon must be strictly positive, got {amount_epsilon}")
        if date_delta_days <= 0.0:
            raise ValueError(f"date_delta_days must be strictly positive, got {date_delta_days}")

        self.amount_epsilon = float(amount_epsilon)
        self.date_delta_days = float(date_delta_days)
        self.text_col = text_col
        self.amount_col = amount_col
        self.date_col = date_col
        self.gated = gated
        self.normalize_legal = normalize_legal

        # Setup weights (must be finite, non-negative, and sum to > 0)
        raw_weights = weights or {"text": 0.5, "amount": 0.3, "date": 0.2}
        for w in raw_weights.values():
            if not math.isfinite(w) or w < 0.0:
                msg = (
                    f"Weights must be finite: {raw_weights}"
                    if not math.isfinite(w)
                    else f"Weights must be non-negative: {raw_weights}"
                )
                raise ValueError(msg)
        total_w = sum(raw_weights.values())
        if total_w <= 0.0:
            raise ValueError(f"Sum of weights must be strictly positive, got {total_w}")
        self.weights = {k: v / total_w for k, v in raw_weights.items()}

        self.numeric_scorer = NumericProximityScorer(epsilon=self.amount_epsilon)
        self.date_scorer = DateWindowScorer(delta_days=self.date_delta_days)
        self.rapid_fuzz = RapidFuzzScorer()
        self.text_method = text_scorer if isinstance(text_scorer, str) else "token_sort_ratio"

    def score_record_pair(
        self,
        left_row: Mapping[str, Any] | pd.Series,
        right_row: Mapping[str, Any] | pd.Series,
    ) -> float:
        """Compute composite similarity score for a pair of records.

        Returns:
            float in [0.0, 1.0].
        """
        # 1. Text similarity
        raw_left_text = _extract_field(left_row, self.text_col, _DEFAULT_TEXT_CANDIDATES)
        raw_right_text = _extract_field(right_row, self.text_col, _DEFAULT_TEXT_CANDIDATES)

        left_str = "" if raw_left_text is None or pd.isna(raw_left_text) else str(raw_left_text)
        right_str = "" if raw_right_text is None or pd.isna(raw_right_text) else str(raw_right_text)

        if self.normalize_legal:
            left_str = strip_legal_suffixes(left_str).strip()
            right_str = strip_legal_suffixes(right_str).strip()

        score_text_fn = getattr(self.rapid_fuzz, self.text_method)
        s_text = float(score_text_fn(left_str, right_str))

        # 2. Amount proximity
        raw_left_amt = _extract_field(left_row, self.amount_col, _DEFAULT_AMOUNT_CANDIDATES)
        raw_right_amt = _extract_field(right_row, self.amount_col, _DEFAULT_AMOUNT_CANDIDATES)
        s_amount = self.numeric_scorer.score(raw_left_amt, raw_right_amt)

        # 3. Date window proximity
        raw_left_date = _extract_field(left_row, self.date_col, _DEFAULT_DATE_CANDIDATES)
        raw_right_date = _extract_field(right_row, self.date_col, _DEFAULT_DATE_CANDIDATES)
        s_date = self.date_scorer.score(raw_left_date, raw_right_date)

        # Strict gating check: if gated and amount or date is out of tolerance, return 0.0
        if self.gated and (s_amount == 0.0 or s_date == 0.0):
            return 0.0

        w_text = self.weights.get("text", 0.0)
        w_amount = self.weights.get("amount", 0.0)
        w_date = self.weights.get("date", 0.0)

        composite = w_text * s_text + w_amount * s_amount + w_date * s_date
        return float(np.clip(composite, 0.0, 1.0))

    def score_pairs(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
    ) -> pd.Series:
        """Vectorized/batched pairwise scoring across two DataFrames of equal length."""
        if len(df_left) != len(df_right):
            raise ValueError(f"DataFrame length mismatch: {len(df_left)} vs {len(df_right)}")

        # 1. Amounts
        l_amt_col = _resolve_column(df_left, self.amount_col, _DEFAULT_AMOUNT_CANDIDATES)
        r_amt_col = _resolve_column(df_right, self.amount_col, _DEFAULT_AMOUNT_CANDIDATES)
        s_amount = (
            self.numeric_scorer.score_pair(
                df_left[l_amt_col],
                pd.Series(df_right[r_amt_col].to_numpy(), index=df_left.index),
            )
            if l_amt_col is not None and r_amt_col is not None
            else pd.Series(0.0, index=df_left.index)
        )

        # 2. Dates
        l_dt_col = _resolve_column(df_left, self.date_col, _DEFAULT_DATE_CANDIDATES)
        r_dt_col = _resolve_column(df_right, self.date_col, _DEFAULT_DATE_CANDIDATES)
        s_date = (
            self.date_scorer.score_pair(
                df_left[l_dt_col],
                pd.Series(df_right[r_dt_col].to_numpy(), index=df_left.index),
            )
            if l_dt_col is not None and r_dt_col is not None
            else pd.Series(0.0, index=df_left.index)
        )

        # 3. Texts
        l_txt_col = _resolve_column(df_left, self.text_col, _DEFAULT_TEXT_CANDIDATES)
        r_txt_col = _resolve_column(df_right, self.text_col, _DEFAULT_TEXT_CANDIDATES)
        if l_txt_col is not None and r_txt_col is not None:
            l_series = df_left[l_txt_col].apply(lambda x: "" if x is None or pd.isna(x) else str(x))
            r_series = pd.Series(
                ["" if x is None or pd.isna(x) else str(x) for x in df_right[r_txt_col].tolist()],
                index=df_left.index,
            )
            if self.normalize_legal:
                l_series = l_series.map(lambda s: strip_legal_suffixes(s).strip())
                r_series = r_series.map(lambda s: strip_legal_suffixes(s).strip())
            s_text = self.rapid_fuzz.score_pair(self.text_method, l_series, r_series)
        else:
            s_text = pd.Series(0.0, index=df_left.index)

        w_text = self.weights.get("text", 0.0)
        w_amount = self.weights.get("amount", 0.0)
        w_date = self.weights.get("date", 0.0)

        composite = w_text * s_text + w_amount * s_amount + w_date * s_date
        if self.gated:
            gate_mask = (s_amount > 0.0) & (s_date > 0.0)
            composite = composite.where(gate_mask, 0.0)

        return composite.clip(lower=0.0, upper=1.0).astype(float)
