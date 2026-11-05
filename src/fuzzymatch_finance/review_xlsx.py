"""Universal XLSX review export/import: curator-friendly round-trip.

Supports arbitrary entity datasets (vendors, customers, products, GL accounts)
with side-by-side field diff formatting, OpenPyXL data validation dropdowns,
formula injection defense, and dual-contract decision re-import.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

from .policy import MappingPolicy
from .results import MappingResult

_DECISIONS = '"accept,reject,pending"'
_SHEET = "review"
_INJECTION_PREFIXES = ("=", "@", "+", "-")


class ReviewDecisions(dict[str, bool]):
    """Structured container for imported curation decisions.

    Subclasses ``dict[str, bool]`` for backward compatibility with existing
    consumers expecting ``{source_id: is_accepted}``, while attaching the
    underlying resolved DataFrame via ``.dataframe``.
    """

    def __init__(
        self,
        outcomes: dict[str, bool],
        resolved_df: pd.DataFrame,
    ) -> None:
        super().__init__(outcomes)
        self.dataframe = resolved_df

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the resolved decisions DataFrame."""
        return self.dataframe.copy()


def _sanitize_value(val: Any) -> Any:
    """Prepend single quote to strings starting with formula prefixes."""
    if val is None:
        return None
    try:
        if bool(pd.isna(val)):
            return None
    except ValueError:
        pass
    if isinstance(val, (int, np.integer, float, np.floating, bool)):
        return val
    s_val = str(val).replace("\r\n", "\n")
    if s_val.startswith(_INJECTION_PREFIXES):
        return f"'{s_val}"
    return s_val


def _desanitize_value(val: Any) -> Any:
    """Strip leading single quote if prepended for formula injection defense."""
    if (
        isinstance(val, str)
        and val.startswith("'")
        and len(val) > 1
        and val[1] in _INJECTION_PREFIXES
    ):
        return val[1:]
    return val


def _prioritized_with_exploration(
    result: MappingResult,
    policy: MappingPolicy,
    exploration_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Boundary-prioritized queue with a seeded exploration flag column."""
    exported = result.prioritized_review(policy).copy()
    n = len(exported)
    flags = np.zeros(n, dtype=bool)
    if n and exploration_rate > 0:
        rng = np.random.default_rng(seed)
        n_explore = round(n * exploration_rate)
        if n_explore:
            flags[rng.choice(n, size=n_explore, replace=False)] = True
    exported["exploration"] = flags
    return exported


def export_review_xlsx(
    results: MappingResult | pd.DataFrame | Any,
    output_path: str | Path | MappingPolicy | None = None,
    path: str | Path | None = None,
    policy: MappingPolicy | None = None,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
    diff_cols: Sequence[tuple[str, str]] | None = None,
    exploration_rate: float = 0.10,
    seed: int = 42,
    dropdown_options: str = _DECISIONS,
) -> Path:
    """Write candidate pairs or review queue to an Excel (.xlsx) file for curation.

    Adds a ``decision`` column pre-filled with ``pending`` and constrained
    by an Excel dropdown data validation. Highlights side-by-side field diffs,
    colors ``fuzzy_score`` bands, and freezes header row. Deterministic under ``seed``.

    Parameters:
        results: Input pairs DataFrame, MatchResults, or legacy MappingResult.
        output_path: Target .xlsx output path (or MappingPolicy in legacy call).
        path: Alternative path parameter for legacy calls.
        policy: MappingPolicy for legacy boundary prioritization.
        source_id_col: Name of source identifier column.
        target_id_col: Name of target identifier column.
        diff_cols: Sequence of (left_col, right_col) pairs for side-by-side diff styling.
        exploration_rate: Fraction of scored rows flagged for exploratory review.
        seed: Random seed for exploration selection.
        dropdown_options: Data validation formula string for dropdown.

    Returns:
        Path to the written workbook.
    """
    # 1. Resolve path and policy across universal and legacy positional signatures
    actual_policy: MappingPolicy | None = policy
    target_path: Path | str | None = None

    if isinstance(output_path, MappingPolicy):
        actual_policy = output_path
        target_path = path
    elif output_path is not None:
        target_path = output_path
    elif path is not None:
        target_path = path

    if target_path is None:
        raise ValueError("output_path or path must be specified")

    out_path = Path(target_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 2. Extract DataFrame from results container
    if isinstance(results, MappingResult):
        if actual_policy is not None:
            df = _prioritized_with_exploration(results, actual_policy, exploration_rate, seed)
        else:
            df = results.review_queue_dataframe().copy()
    elif hasattr(results, "review_candidates"):
        df = pd.DataFrame(results.review_candidates).copy()
    elif hasattr(results, "matches"):
        df = pd.DataFrame(results.matches).copy()
    elif isinstance(results, pd.DataFrame):
        df = results.copy()
    else:
        try:
            df = pd.DataFrame(results)
        except Exception as exc:
            raise TypeError(f"Unsupported results type: {type(results)}") from exc

    # If empty DataFrame and no columns, ensure standard columns exist
    if df.empty and len(df.columns) == 0:
        df = pd.DataFrame(columns=[source_id_col, target_id_col, "decision"])
    elif "decision" not in df.columns:
        df["decision"] = "pending"
    else:
        df["decision"] = df["decision"].fillna("pending")

    # Reorder columns to place diff pairs side-by-side if specified
    if diff_cols:
        ordered_cols: list[str] = []
        for col_name in [source_id_col, target_id_col, "decision"]:
            if col_name in df.columns and col_name not in ordered_cols:
                ordered_cols.append(col_name)
        for left_c, right_c in diff_cols:
            if left_c in df.columns and left_c not in ordered_cols:
                ordered_cols.append(left_c)
            if right_c in df.columns and right_c not in ordered_cols:
                ordered_cols.append(right_c)
        for c in df.columns:
            if c not in ordered_cols:
                ordered_cols.append(c)
        df = df[ordered_cols]

    # 3. Build Excel Workbook via OpenPyXL
    wb = Workbook()
    ws = wb.active
    ws.title = _SHEET

    headers = [str(c) for c in df.columns]
    ws.append(headers)

    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    decision_header_fill = PatternFill(start_color="2E75B6", end_color="2E75B6", fill_type="solid")

    for col_idx, h_name in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col_idx)
        cell.font = header_font
        cell.fill = decision_header_fill if h_name == "decision" else header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")

    diff_pair_indices: list[tuple[int, int]] = []
    if diff_cols:
        for left_c, right_c in diff_cols:
            if left_c in headers and right_c in headers:
                diff_pair_indices.append((headers.index(left_c) + 1, headers.index(right_c) + 1))

    diff_fill = PatternFill(start_color="FFF2CC", end_color="FFF2CC", fill_type="solid")

    for row_values in df.itertuples(index=False):
        sanitized_row = [_sanitize_value(v) for v in row_values]
        ws.append(sanitized_row)

    n_rows = len(df)

    # Highlight divergent fields for side-by-side diff pairs
    if diff_pair_indices and n_rows > 0:
        for r_idx in range(2, n_rows + 2):
            for l_col, r_col in diff_pair_indices:
                v_l = ws.cell(row=r_idx, column=l_col).value
                v_r = ws.cell(row=r_idx, column=r_col).value
                mismatch = False
                if v_l is None and v_r is None:
                    mismatch = False
                elif v_l is None or v_r is None:
                    mismatch = True
                else:
                    try:
                        fl_l = float(v_l)
                        fl_r = float(v_r)
                        mismatch = abs(fl_l - fl_r) > 1e-6
                    except (ValueError, TypeError):
                        mismatch = str(v_l).strip() != str(v_r).strip()
                if mismatch:
                    ws.cell(row=r_idx, column=l_col).fill = diff_fill
                    ws.cell(row=r_idx, column=r_col).fill = diff_fill

    # Freeze header row
    ws.freeze_panes = "A2"

    # Attach DataValidation dropdown for decisions
    if "decision" in headers:
        dec_col_idx = headers.index("decision") + 1
        col_letter = get_column_letter(dec_col_idx)
        dv = DataValidation(type="list", formula1=dropdown_options, allow_blank=False)
        dv.error = "Please select accept, reject, or pending"
        dv.errorTitle = "Invalid Review Decision"
        ws.add_data_validation(dv)
        dv.add(f"{col_letter}2:{col_letter}{max(n_rows + 1, 2)}")

    # Conditional color scale formatting on fuzzy_score
    if "fuzzy_score" in headers and n_rows > 0:
        score_col_idx = headers.index("fuzzy_score") + 1
        col_letter = get_column_letter(score_col_idx)
        ws.conditional_formatting.add(
            f"{col_letter}2:{col_letter}{n_rows + 1}",
            ColorScaleRule(
                start_type="min",
                start_color="FFF5F5",
                end_type="max",
                end_color="B7E1CD",
            ),
        )

    # Column auto-sizing
    for col in ws.columns:
        first_cell = col[0]
        col_letter = first_cell.column_letter
        max_len = max(len(str(c.value or "")) for c in col)
        ws.column_dimensions[col_letter].width = min(max(max_len + 3, 12), 45)

    wb.save(out_path)
    return out_path


def import_review_xlsx(
    path: str | Path | None = None,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
    input_path: str | Path | None = None,
    as_dataframe: bool = False,
) -> ReviewDecisions | pd.DataFrame:
    """Read curated review decisions from an Excel (.xlsx) file.

    Extracts rows whose ``decision`` is ``accept`` or ``reject`` (ignoring
    ``pending`` or blank rows). Supports arbitrary entity identifier columns
    with automatic fallback to legacy names, tolerates extraneous user-added
    columns, and desanitizes formula-injected values.

    Parameters:
        path: Path to the curated .xlsx workbook.
        source_id_col: Name of the source identifier column.
        target_id_col: Name of the target identifier column.
        input_path: Alternative path argument.
        as_dataframe: If True, returns the resolved pd.DataFrame directly.
                      If False (default), returns a ReviewDecisions mapping
                      which behaves as dict[str, bool] with .dataframe access.

    Returns:
        ReviewDecisions (dict[str, bool] subclass) or pd.DataFrame.
    """
    resolved_path = path or input_path
    if resolved_path is None:
        raise ValueError("path or input_path must be provided")

    in_path = Path(resolved_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Review file not found: {in_path}")

    # Read sheet 'review' or first sheet
    with pd.ExcelFile(in_path) as excel_file:
        target_sheet = _SHEET if _SHEET in excel_file.sheet_names else excel_file.sheet_names[0]
        frame = pd.read_excel(excel_file, sheet_name=target_sheet)

    if "decision" not in frame.columns:
        raise ValueError("review sheet missing columns: ['decision']")

    # Resolve source identifier column with automatic fallbacks
    actual_source_id_col: str | None = None
    if source_id_col in frame.columns:
        actual_source_id_col = source_id_col
    else:
        fallbacks = [
            "source_id",
            "source_local_account",
            "source_vendor_id",
            "source_customer_id",
            "left_id",
            "source_key",
        ]
        for cand in fallbacks:
            if cand in frame.columns:
                actual_source_id_col = cand
                break

    if actual_source_id_col is None:
        missing = sorted({source_id_col, "decision"} - set(frame.columns))
        raise ValueError(f"review sheet missing columns: {missing}")

    if frame.empty:
        empty_df = pd.DataFrame(columns=frame.columns)
        return empty_df if as_dataframe else ReviewDecisions({}, empty_df)

    norm_decisions = frame["decision"].astype(str).str.strip().str.lower()
    resolved_mask = norm_decisions.isin(["accept", "reject"])
    resolved = frame[resolved_mask].copy()

    # Desanitize values that were sanitized during export
    for col in resolved.columns:
        resolved[col] = resolved[col].map(_desanitize_value)

    if as_dataframe:
        return resolved

    outcomes: dict[str, bool] = {
        str(row[actual_source_id_col]): str(row["decision"]).strip().lower() == "accept"
        for _, row in resolved.iterrows()
    }
    return ReviewDecisions(outcomes, resolved)
