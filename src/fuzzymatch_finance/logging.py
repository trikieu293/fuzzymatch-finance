"""Structured JSONL audit logging for finance mapping jobs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """One structured audit event."""

    event_type: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AuditLog:
    """A sequence of audit events with JSONL serialisation."""

    events: list[AuditEvent]

    def to_jsonl(self) -> str:
        """Return one JSON line per event, trailing newline."""
        return "".join(
            json.dumps({"event_type": event.event_type, **event.payload}, sort_keys=True) + "\n"
            for event in self.events
        )
