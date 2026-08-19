"""The governance trail: what happened, who did it, and what was checked.

M3 logged a single `[audit]` line at the end of a graph run. That answered *what
was booked* and, once the roster landed, *who acted* — but it was a string in a
list, written only on the orchestrator path, and the chat path where guests
actually book wrote nothing at all.

This is the structured version, and it is deliberately small. An audit trail
earns its keep by being complete and boring: append-only, no updates, one record
per decision, each carrying its own timestamp and actor. Nothing in here
interprets anything — reading the trail is a separate job from writing it, and a
writer that also judged would be the thing you least want to trust.

Records are held per-session in memory and optionally mirrored to a JSONL file
(`TF4_AUDIT_LOG`), which is what makes the trail outlive the process for a demo
or a grader. Writing is best-effort: a governance log that could take down a
guest's booking by filling a disk would be a worse bargain than losing a line.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Where a durable copy goes, if anywhere. Unset in tests so they never touch disk.
AUDIT_LOG_PATH = os.getenv("TF4_AUDIT_LOG", "").strip()

_write_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Record:
    """One line of the trail. Frozen, because an audit record is never edited."""

    event: str                       # "tool_call" | "grounding" | "approval" | "booking"
    actor: str | None                # the roster unit responsible, when there is one
    member_id: str | None            # whose journey this belongs to
    detail: dict[str, Any] = field(default_factory=dict)
    at: str = field(default_factory=_now)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def emit(
    event: str,
    *,
    actor: str | None = None,
    member_id: str | None = None,
    **detail: Any,
) -> dict[str, Any]:
    """Write one record and hand it back as a plain dict.

    The graph carries its trail inside LangGraph state, which is checkpointed and
    has to stay JSON-shaped, so it collects dicts rather than `Record` objects.
    Same record, same mirror, two homes.
    """
    entry = Record(event=event, actor=actor, member_id=member_id, detail=detail)
    _mirror(entry)
    return entry.as_dict()


@dataclass
class Trail:
    """An append-only sequence of records for one guest's session."""

    member_id: str | None = None
    records: list[Record] = field(default_factory=list)

    def record(self, event: str, *, actor: str | None = None, **detail: Any) -> Record:
        entry = Record(event=event, actor=actor, member_id=self.member_id, detail=detail)
        self.records.append(entry)
        _mirror(entry)
        return entry

    def of(self, event: str) -> list[Record]:
        return [r for r in self.records if r.event == event]

    def as_dicts(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in self.records]


def _mirror(entry: Record) -> None:
    """Best-effort copy to the JSONL log, if one is configured.

    Swallows its own failures on purpose. The trail is evidence about a booking;
    it must never become a reason a guest doesn't get one.
    """
    if not AUDIT_LOG_PATH:
        return
    try:
        with _write_lock:
            path = Path(AUDIT_LOG_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry.as_dict(), ensure_ascii=False) + "\n")
    except OSError:
        pass
