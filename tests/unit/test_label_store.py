"""Tests for the CSV-backed labeled-pair store (learning-loop capture)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch_finance.label_store import (
    LabeledPair,
    LabeledPairStore,
    _isoformat,
    _parse_ts,
)


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "labels" / "pairs.csv"


def _pair(label: int = 1, tag: str = "coa", left: str = "L1", right: str = "R1") -> LabeledPair:
    return LabeledPair(
        left_key=left,
        right_key=right,
        label=label,
        labeler="tester",
        timestamp=datetime(2026, 8, 23, tzinfo=UTC),
        dataset_tag=tag,
    )


class TestRecordAndRead:
    def test_record_then_all_roundtrips(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        store.record("L1", "R1", 1, labeler="t", dataset_tag="coa")
        pairs = LabeledPairStore(store_path).all()
        assert len(pairs) == 1
        assert pairs[0].left_key == "L1"
        assert pairs[0].label == 1

    def test_persists_across_instances(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        store.record("A", "B", 0)
        again = LabeledPairStore(store_path)
        again.record("C", "D", 1)
        assert len(LabeledPairStore(store_path)) == 2


class TestUpsertSemantics:
    def test_same_pair_and_tag_is_overwritten(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        store.record("L1", "R1", 1, dataset_tag="coa")
        store.record("L1", "R1", 0, dataset_tag="coa")
        pairs = store.all()
        assert len(pairs) == 1
        assert pairs[0].label == 0

    def test_same_keys_different_tags_both_kept(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        store.record("L1", "R1", 1, dataset_tag="coa")
        store.record("L1", "R1", 1, dataset_tag="vendors")
        assert len(store) == 2


class TestValidation:
    @pytest.mark.parametrize("bad_label", [2, -1, 5])
    def test_label_must_be_zero_or_one(self, store_path: Path, bad_label: int) -> None:
        store = LabeledPairStore(store_path)
        with pytest.raises(ValueError, match="label"):
            store.record("L", "R", bad_label)

    def test_extend_rejects_invalid_labels(self, store_path: Path) -> None:
        """Validation lives on the record type, so extend() cannot bypass it."""
        with pytest.raises(ValueError, match="label"):
            _pair(label=7)
        store = LabeledPairStore(store_path)
        with pytest.raises(ValueError, match="label"):
            store.extend([_pair(label=1), _pair(label=3)])


class TestQueries:
    def test_for_dataset_filters_by_tag(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        store.extend([_pair(tag="coa"), _pair(tag="vendors"), _pair(tag="coa", right="R9")])
        assert len(store.for_dataset("coa")) == 2
        assert len(store.for_dataset("vendors")) == 1

    def test_frame_has_contract_columns(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        store.record("L", "R", 1)
        frame = store.frame()
        assert list(frame.columns) == [
            "left_key",
            "right_key",
            "label",
            "labeler",
            "timestamp",
            "dataset_tag",
        ]
        assert set(frame["label"].unique()) <= {0, 1}

    def test_empty_store_contract(self, store_path: Path) -> None:
        store = LabeledPairStore(store_path)
        assert store.all() == []
        assert len(store) == 0
        assert store.frame().empty

    def test_capture_labels_suggested_target_fallback(self, store_path: Path) -> None:
        from fuzzymatch_finance.label_store import capture_labels
        from fuzzymatch_finance.results import MappingResult

        store = LabeledPairStore(store_path)
        # Empty evidence, but review queue has suggested_target
        review = pd.DataFrame(
            [
                {
                    "source_local_account": "SRC1",
                    "suggested_target": "TGT1",
                    "decision": "approved",
                },
                {"source_local_account": "SRC2", "suggested_target": None, "decision": "approved"},
            ]
        )
        result = MappingResult(
            approved_mappings=pd.DataFrame([]),
            review_queue=review,
            evidence=pd.DataFrame([]),
            audit_log="",
        )
        written = capture_labels(
            store, result, outcomes={"SRC1": True, "SRC2": True, "UNKNOWN": True}
        )
        assert written == 1
        records = store.all()
        assert len(records) == 1
        assert records[0].left_key == "SRC1"
        assert records[0].right_key == "TGT1"

    def test_timestamp_helpers(self) -> None:
        # Naive datetime
        naive = datetime(2026, 10, 17, 12, 0, 0)
        iso_str = _isoformat(naive)
        assert "+00:00" in iso_str

        # Non-ISO format fallback
        parsed = _parse_ts("2026-10-17 14:30:00")
        assert parsed.hour == 14
        assert parsed.tzinfo is not None
