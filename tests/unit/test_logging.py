from __future__ import annotations

from fuzzymatch_finance.logging import AuditEvent, AuditLog


def test_audit_log_writes_jsonl():
    log = AuditLog(
        [
            AuditEvent(
                event_type="mapping_decision_created",
                payload={"job_id": "job_1"},
            )
        ]
    )
    content = log.to_jsonl()

    assert content.endswith("\n")
    assert '"event_type": "mapping_decision_created"' in content
    assert '"job_id": "job_1"' in content


def test_audit_log_multiple_events():
    events = [
        AuditEvent("rule_mapped", {"source": "610010", "target": "610000"}),
        AuditEvent("needs_review", {"source": "999999"}),
    ]
    log = AuditLog(events)
    lines = log.to_jsonl().strip().split("\n")

    assert len(lines) == 2
    assert '"event_type": "rule_mapped"' in lines[0]
    assert '"event_type": "needs_review"' in lines[1]
