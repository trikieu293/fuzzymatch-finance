from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from fuzzymatch_finance.results import MappingResult


@pytest.fixture
def sample_result():
    return MappingResult(
        approved_mappings=pd.DataFrame(
            [{"source_local_account": "610010", "target_account": "610000"}]
        ),
        review_queue=pd.DataFrame(
            [{"source_local_account": "999999", "decision_status": "needs_review"}]
        ),
        evidence=pd.DataFrame([{"source_local_account": "610010", "reason": "rule"}]),
        audit_log='{"event_type":"mapping_decision_created"}\n',
    )


def test_result_exports_dataframes_and_jsonl_strings(sample_result):
    assert sample_result.approved_mappings_dataframe().shape == (1, 2)
    assert sample_result.review_queue_dataframe().shape == (1, 2)
    assert sample_result.evidence_dataframe().shape == (1, 2)
    assert sample_result.audit_log_jsonl().startswith('{"event_type"')


def test_to_approved_mappings_csv(sample_result, tmp_path: Path):
    out = tmp_path / "approved.csv"
    sample_result.to_approved_mappings_csv(str(out))
    df = pd.read_csv(out)
    assert df.shape == (1, 2)
    assert list(df.columns) == ["source_local_account", "target_account"]


def test_to_review_queue_csv(sample_result, tmp_path: Path):
    out = tmp_path / "review.csv"
    sample_result.to_review_queue_csv(str(out))
    df = pd.read_csv(out)
    assert df.shape == (1, 2)
    assert list(df.columns) == ["source_local_account", "decision_status"]


def test_to_evidence_csv(sample_result, tmp_path: Path):
    out = tmp_path / "evidence.csv"
    sample_result.to_evidence_csv(str(out))
    df = pd.read_csv(out)
    assert df.shape == (1, 2)
    assert list(df.columns) == ["source_local_account", "reason"]


def test_to_audit_jsonl(sample_result, tmp_path: Path):
    out = tmp_path / "audit.jsonl"
    sample_result.to_audit_jsonl(str(out))
    content = out.read_text(encoding="utf-8")
    assert content == sample_result.audit_log
    # Validate it's parseable JSON
    record = json.loads(content.strip())
    assert record["event_type"] == "mapping_decision_created"


def test_prioritized_review_without_fuzzy_score(sample_result):
    from fuzzymatch_finance.policy import MappingPolicy

    policy = MappingPolicy(auto_accept_threshold=0.90, review_threshold=0.70)
    # sample_result.review_queue does not contain "fuzzy_score"
    prioritized = sample_result.prioritized_review(policy)
    assert "fuzzy_score" not in prioritized.columns
    assert len(prioritized) == 1
