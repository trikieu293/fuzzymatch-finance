"""CSV-backed store for labeled pairs captured from the review workflow."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .results import MappingResult
from .rules import MappingRule, MappingRuleSet

_COLUMNS = ["left_key", "right_key", "label", "labeler", "timestamp", "dataset_tag"]
_KEY_COLUMNS = ["left_key", "right_key", "dataset_tag"]


@dataclass(frozen=True)
class LabeledPair:
    """One human- or rule-confirmed pair label.

    Attributes:
        left_key: Stable identifier of the left record.
        right_key: Stable identifier of the right record.
        label: 1 for a confirmed match, 0 for a confirmed non-match.
        labeler: Who produced the label (e.g. ``"review:alice"``, ``"rules"``).
        timestamp: When the label was recorded (UTC).
        dataset_tag: Logical dataset name so one store can serve many jobs.
    """

    left_key: str
    right_key: str
    label: int
    labeler: str
    timestamp: datetime
    dataset_tag: str

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {self.label!r}")


class LabeledPairStore:
    """Append/upsert labeled pairs to a CSV file.

    The unit of uniqueness is ``(left_key, right_key, dataset_tag)``:
    recording the same triple again replaces the earlier label (latest wins),
    enabling correction of mistakes without history bloat.
    Writes are atomic (temp file + os.replace).
    """

    def __init__(self, path: Path | str):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def record(
        self,
        left_key: str,
        right_key: str,
        label: int,
        labeler: str = "human",
        dataset_tag: str = "default",
        timestamp: datetime | None = None,
    ) -> None:
        """Record (or overwrite) one pair label.

        Validation lives on :class:`LabeledPair`; the explicit check here just
        gives a friendlier error at the call site.
        """
        if label not in (0, 1):
            raise ValueError(f"label must be 0 or 1, got {label!r}")
        ts = timestamp or datetime.now(UTC)
        self.extend(
            [
                LabeledPair(
                    left_key=str(left_key),
                    right_key=str(right_key),
                    label=int(label),
                    labeler=labeler,
                    timestamp=ts,
                    dataset_tag=dataset_tag,
                )
            ]
        )

    def extend(self, pairs: Iterable[LabeledPair]) -> None:
        """Upsert several :class:`LabeledPair` rows in one write."""
        incoming = list(pairs)
        frame = self.frame()
        new_rows = pd.DataFrame(
            [
                {
                    "left_key": p.left_key,
                    "right_key": p.right_key,
                    "label": int(p.label),
                    "labeler": p.labeler,
                    "timestamp": _isoformat(p.timestamp),
                    "dataset_tag": p.dataset_tag,
                }
                for p in incoming
            ],
            columns=_COLUMNS,
        )
        merged = pd.concat([frame, new_rows], ignore_index=True)
        merged = merged.drop_duplicates(subset=_KEY_COLUMNS, keep="last")
        merged = merged.sort_values(_KEY_COLUMNS, kind="stable").reset_index(drop=True)
        tmp_path = self._path.with_suffix(self._path.suffix + ".tmp")
        merged.to_csv(tmp_path, index=False)
        os.replace(tmp_path, self._path)

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def frame(self) -> pd.DataFrame:
        """All stored labels as a DataFrame with the contract columns."""
        if not self._path.exists() or self._path.stat().st_size == 0:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.read_csv(self._path)

    def all(self) -> list[LabeledPair]:
        """All stored labels as typed records, ordered by key columns."""
        frame = self.frame()
        return [
            LabeledPair(
                left_key=str(row.left_key),
                right_key=str(row.right_key),
                label=int(cast(int, row.label)),
                labeler=str(row.labeler),
                timestamp=_parse_ts(row.timestamp),
                dataset_tag=str(row.dataset_tag),
            )
            for row in frame.itertuples(index=False)
        ]

    def for_dataset(self, dataset_tag: str) -> list[LabeledPair]:
        """Labels belonging to one logical dataset."""
        return [p for p in self.all() if p.dataset_tag == dataset_tag]

    def __len__(self) -> int:
        return len(self.all())


LabelStore = LabeledPairStore  # Canonical alias


def capture_labels(
    store: LabeledPairStore,
    result: MappingResult,
    *,
    outcomes: dict[str, bool] | None = None,
    dataset_tag: str = "default",
    labeler: str = "review",
    include_approved: bool = True,
) -> int:
    """Persist training labels from a mapping run.

    Approved rows are recorded as positives (labeler ``"rules"``). Review rows
    whose source account appears in ``outcomes`` are recorded against their
    suggested target with the reviewer's accept/reject decision; sources
    without an outcome or without a suggested target are skipped.

    Args:
        store: Destination store.
        result: A :class:`~fuzzymatch_finance.results.MappingResult`.
        outcomes: ``{source_local_account: accepted?}`` for resolved reviews.
        dataset_tag: Logical dataset name for the labels.
        labeler: Attribution string for review-sourced labels.
        include_approved: Whether rule/fuzzy-approved pairs become positives.

    Returns:
        Number of labels written.
    """
    written = 0
    approved = result.approved_mappings_dataframe()
    if include_approved and not approved.empty:
        for row in approved.itertuples(index=False):
            store.record(
                str(row.source_local_account),
                str(row.target_account),
                1,
                labeler="rules",
                dataset_tag=dataset_tag,
            )
            written += 1

    if not outcomes:
        return written

    review = result.review_queue_dataframe()
    evidence_target = {}
    evidence = result.evidence_dataframe()
    if not evidence.empty and {"source_local_account", "target_account"} <= set(evidence.columns):
        evidence_target = dict(zip(evidence["source_local_account"], evidence["target_account"]))

    review_sources = set()
    if not review.empty and "source_local_account" in review.columns:
        review_sources = set(review["source_local_account"])

    for source, accepted in outcomes.items():
        if source not in review_sources:
            continue
        target = evidence_target.get(source)
        row_mask = review["source_local_account"] == source
        if (
            target is None
            and not review.empty
            and "suggested_target" in review.columns
            and row_mask.any()
        ):
            candidate = review.loc[row_mask, "suggested_target"].iloc[0]
            if pd.notna(candidate):
                target = candidate
        if target is None:
            continue
        store.record(
            str(source),
            str(target),
            int(bool(accepted)),
            labeler=labeler,
            dataset_tag=dataset_tag,
        )
        written += 1
    return written


def synthesize_rules_from_reviews(
    review_df: pd.DataFrame | Any,
    label_store: LabeledPairStore | Path | str | None = None,
    source_id_col: str = "source_id",
    target_id_col: str = "target_id",
    decision_col: str = "decision",
    dataset_tag: str = "default",
    labeler: str = "review",
    priority_start: int = 100,
) -> tuple[MappingRuleSet, LabeledPairStore]:
    """Converts accepted/rejected pairs into persistent MappingRule instances while updating LabelStore.

    Args:
        review_df: Curated review queue DataFrame or ReviewDecisions container.
        label_store: Optional destination store (instance, path, or None). If None, provisions
            an isolated temporary LabeledPairStore.
        source_id_col: Column name identifying source entities.
        target_id_col: Column name identifying target entities.
        decision_col: Column name containing review decisions ('accept', 'reject', 'pending').
        dataset_tag: Tag for partitioning labeled pairs in LabelStore.
        labeler: Attribution identifier for curated decisions.
        priority_start: Initial priority number for synthesized rules.

    Returns:
        tuple[MappingRuleSet, LabeledPairStore]: Synthesized ruleset and updated audit store.
    """
    # 1. Resolve store destination
    if label_store is None:
        with tempfile.NamedTemporaryFile(suffix="_labels.csv", delete=False) as tmp_file:
            tmp_name = tmp_file.name
        store = LabeledPairStore(tmp_name)
    elif isinstance(label_store, (str, Path)):
        store = LabeledPairStore(label_store)
    else:
        store = label_store

    # 2. Extract DataFrame
    if hasattr(review_df, "dataframe") and isinstance(review_df.dataframe, pd.DataFrame):
        work_input = review_df.dataframe.copy()
    elif isinstance(review_df, pd.DataFrame):
        work_input = review_df.copy()
    else:
        try:
            work_input = pd.DataFrame(review_df)
        except Exception as exc:
            raise TypeError(f"Unsupported review_df type: {type(review_df)}") from exc

    if work_input.empty:
        return MappingRuleSet([]), store

    # 3. Resolve column names with intelligent fallback
    src_col = source_id_col if source_id_col in work_input.columns else None
    if src_col is None:
        for fallback in (
            "source_id",
            "source_vendor_id",
            "source_local_account",
            "source_customer_id",
            "left_id",
        ):
            if fallback in work_input.columns:
                src_col = fallback
                break

    tgt_col = target_id_col if target_id_col in work_input.columns else None
    if tgt_col is None:
        for fallback in (
            "target_id",
            "target_erp_id",
            "target_account",
            "suggested_target",
            "right_id",
        ):
            if fallback in work_input.columns:
                tgt_col = fallback
                break

    dec_col = decision_col if decision_col in work_input.columns else None
    if dec_col is None:
        for fallback in ("decision", "status", "action"):
            if fallback in work_input.columns:
                dec_col = fallback
                break

    if src_col is None or tgt_col is None or dec_col is None:
        missing = []
        if src_col is None:
            missing.append(source_id_col)
        if tgt_col is None:
            missing.append(target_id_col)
        if dec_col is None:
            missing.append(decision_col)
        raise ValueError(f"review DataFrame missing required columns: {missing}")

    # 4. Clean and normalize decisions
    work_df = (
        work_input[[src_col, tgt_col, dec_col]].copy().dropna(subset=[src_col, tgt_col, dec_col])
    )
    work_df[src_col] = work_df[src_col].astype(str).str.strip()
    work_df[tgt_col] = work_df[tgt_col].astype(str).str.strip()
    work_df["clean_decision"] = work_df[dec_col].astype(str).str.strip().str.lower()

    # Filter out empty keys
    work_df = work_df[(work_df[src_col] != "") & (work_df[tgt_col] != "")]

    accept_tokens = {"accept", "accepted", "approve", "approved", "1", "true"}
    reject_tokens = {"reject", "rejected", "declined", "0", "false"}

    pairs_to_record: list[LabeledPair] = []
    resolved_rows: list[dict[str, Any]] = []

    # Deduplicate within batch preserving latest decision (keep='last')
    deduped = work_df.drop_duplicates(subset=[src_col, tgt_col], keep="last")

    now = datetime.now(UTC)
    for row in deduped.itertuples(index=False):
        s_val = getattr(row, src_col)
        t_val = getattr(row, tgt_col)
        d_val = row.clean_decision

        if d_val in accept_tokens:
            label = 1
        elif d_val in reject_tokens:
            label = 0
        else:
            continue

        pairs_to_record.append(
            LabeledPair(
                left_key=s_val,
                right_key=t_val,
                label=label,
                labeler=labeler,
                timestamp=now,
                dataset_tag=dataset_tag,
            )
        )
        if label == 1:
            resolved_rows.append(
                {
                    "source_id": s_val,
                    "target_id": t_val,
                    "orig_src_col": src_col,
                    "orig_tgt_col": tgt_col,
                }
            )

    # 5. Atomic upsert to LabelStore
    if pairs_to_record:
        store.extend(pairs_to_record)

    if not resolved_rows:
        return MappingRuleSet([]), store

    # 6. Deduplicate accepted source mappings (keep latest accepted per source_id)
    acc_df = pd.DataFrame(resolved_rows).drop_duplicates(subset=["source_id"], keep="last")

    # 7. Synthesize deterministic MappingRule instances
    synthesized_rules: list[MappingRule] = []
    for idx, r in enumerate(acc_df.itertuples(index=False)):
        s_id = str(r.source_id)
        t_id = str(r.target_id)
        orig_s = str(r.orig_src_col)
        orig_t = str(r.orig_tgt_col)

        rule = MappingRule(
            rule_id=f"rule_{s_id}_{t_id}",
            description=f"Curated review mapping: {s_id} -> {t_id}",
            priority=priority_start + idx,
            condition={
                orig_s: s_id,
            },
            target={
                orig_t: t_id,
            },
        )
        synthesized_rules.append(rule)

    return MappingRuleSet(synthesized_rules), store


def _isoformat(ts: datetime) -> str:
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.astimezone(UTC).isoformat()


def _parse_ts(value: object) -> datetime:
    text = str(value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:  # pragma: no cover - python < 3.11 fallback
        parsed = datetime.strptime(text[:19], "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)
