"""Does the reply only state what a tool actually told us?

Invariant 1 says the model never invents — "not a restaurant, a time, an email, a
dish, or a confirmation id". For most of that list the guarantee is structural:
`book_table` refuses a restaurant Scout never surfaced and a slot the backend
never offered, so an invented one cannot reach the ledger. But refusing to *act*
on a made-up detail is not the same as refusing to *say* it, and the reply text
has never been checked at all.

This module checks it, and does so in code rather than with a second model. Three
of the four things it looks for have exact answers already sitting in session
state, and the week 4 reasoning decision makes the argument for that case:
a claim with a deterministic checker should be checked, not estimated — cheaper,
exact, auditable, and it cannot itself hallucinate. It also costs no API key,
which keeps the offline demo whole.

**What is covered:** clock times, ISO dates, confirmation ids, email addresses.
Each has a regular shape and an exact allowed set, so a false positive is close
to impossible — which matters, because a finding here removes a sentence.

**What is deliberately not covered:** dish and restaurant *names*, which would
need entity extraction to find and a judgement to verify; and phone numbers,
where a street address ("104-0061 Chuo City") reads enough like one to strip a
perfectly good sentence. Those stay a job for a model judge, if one is ever worth
its cost. The gap is real and named rather than papered over.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Times the guest or Dino may write: "7pm", "7:30 pm", "19:00". A bare number
# ("7", "4 people", "24 hours") is deliberately not a time — too ambiguous to
# enforce on, and "free to cancel up to 24 hours before" must never be flagged.
_TIME_TOKEN = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", re.I)
_ISO_DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_CONFIRMATION = re.compile(r"\bTF4-\d+\b", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# A sentence ends at ., !, ? or … followed by space — applied within a line, so a
# markdown bullet or a line break stays its own unit and a stripped bullet
# doesn't take the one after it with it.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _readings(match: re.Match[str]) -> set[str]:
    """The `HH:MM` values one written time could mean.

    "7:30" with no am/pm is genuinely ambiguous, so both readings come back and
    the caller keeps whichever is a real slot. Being generous here is the safe
    direction: the cost of an extra reading is a claim we let through, and the
    cost of missing one is a good sentence deleted.
    """
    hour = int(match.group(1))
    minute = int(match.group(2)) if match.group(2) else 0
    has_colon = match.group(2) is not None
    ampm = (match.group(3) or "").lower().replace(".", "")
    if (not ampm and not has_colon) or minute > 59:
        return set()
    if ampm:
        h = hour % 12 + (12 if ampm == "pm" else 0)
        return {f"{h:02d}:{minute:02d}"} if 0 <= h <= 23 else set()
    out = set()
    if 0 <= hour <= 23:
        out.add(f"{hour:02d}:{minute:02d}")
    if hour < 12:
        out.add(f"{hour + 12:02d}:{minute:02d}")
    return out


def clock_times(text: str) -> set[str]:
    """Every `HH:MM` reading of every explicit time in `text`.

    Shared with the chat path, where `_requested_times` uses it to hold a booking
    to the time the guest actually asked for. It lives here because governance is
    the lower layer — the concierge imports it, never the other way round.
    """
    out: set[str] = set()
    for match in _TIME_TOKEN.finditer(text or ""):
        out |= _readings(match)
    return out


@dataclass(frozen=True)
class GroundedFacts:
    """What a reply is allowed to state, gathered from actual tool results."""

    times: frozenset[str] = frozenset()
    dates: frozenset[str] = frozenset()
    confirmation_ids: frozenset[str] = frozenset()
    emails: frozenset[str] = frozenset()

    @classmethod
    def gather(cls, session: Any) -> "GroundedFacts":
        """Read the facts off a live concierge session.

        Duck-typed on purpose: governance must not import the chat module it is
        called from. Times and dates accumulate across the whole session rather
        than tracking only the latest lookup, so recapping an earlier restaurant's
        slots stays legitimate.
        """
        profile = getattr(session, "profile", None) or {}
        bookings = list((getattr(session, "bookings", None) or {}).values())

        ids = {b.get("confirmation_id") for b in bookings if b.get("confirmation_id")}
        # A returning guest's own history is quotable too — `recall_guest_profile`
        # hands those ids back, so saying one is grounded, not invented.
        for past in profile.get("past_bookings") or []:
            if isinstance(past, dict) and past.get("confirmation_id"):
                ids.add(past["confirmation_id"])

        times = set(getattr(session, "offered_times", None) or ())
        dates = set(getattr(session, "offered_dates", None) or ())
        for b in bookings:
            if b.get("time"):
                times.add(b["time"])
            if b.get("date"):
                dates.add(b["date"])

        return cls(
            times=frozenset(times),
            dates=frozenset(dates),
            confirmation_ids=frozenset(ids),
            emails=frozenset(e for e in [profile.get("email")] if e),
        )


@dataclass(frozen=True)
class Finding:
    """One thing the reply stated that no tool result supports."""

    kind: str      # "time" | "date" | "confirmation_id" | "email"
    value: str     # as written in the reply
    sentence: str  # the sentence it appeared in


@dataclass(frozen=True)
class Verdict:
    reply: str                              # what should actually be sent
    findings: tuple[Finding, ...] = ()
    removed: tuple[str, ...] = ()
    rewritten: bool = False

    @property
    def grounded(self) -> bool:
        return not self.findings

    def as_audit(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "rewritten": self.rewritten,
            "findings": [{"kind": f.kind, "value": f.value} for f in self.findings],
            "removed": list(self.removed),
        }


def _findings_in(sentence: str, facts: GroundedFacts) -> list[Finding]:
    """Every unsupported claim in one sentence."""
    out: list[Finding] = []

    for match in _TIME_TOKEN.finditer(sentence):
        readings = _readings(match)
        # No reading at all means it wasn't a time (a bare "4 people"), not that
        # it was an ungrounded one.
        if readings and not (readings & facts.times):
            out.append(Finding("time", match.group(0).strip(), sentence))

    for match in _ISO_DATE.finditer(sentence):
        if match.group(0) not in facts.dates:
            out.append(Finding("date", match.group(0), sentence))

    for match in _CONFIRMATION.finditer(sentence):
        if match.group(0).upper() not in {c.upper() for c in facts.confirmation_ids}:
            out.append(Finding("confirmation_id", match.group(0), sentence))

    for match in _EMAIL.finditer(sentence):
        if match.group(0).lower() not in {e.lower() for e in facts.emails}:
            out.append(Finding("email", match.group(0), sentence))

    return out


def check(reply: str, facts: GroundedFacts) -> Verdict:
    """Check a reply against what tools returned, removing what they didn't say.

    The sentence carrying an unsupported claim is dropped rather than the whole
    reply: a warm answer with one invented time in it is mostly good, and binning
    it would cost the guest more than the claim did.

    One deliberate fail-safe — if every sentence is unsupported, the reply goes
    out unchanged and the findings are still recorded. Sending a guest nothing at
    all is a worse failure than sending them something to argue with, and a reply
    that trips every check is far more likely to be a bug in here than a model
    that invented every single word.
    """
    if not (reply or "").strip():
        return Verdict(reply=reply)

    kept_lines: list[str] = []
    findings: list[Finding] = []
    removed: list[str] = []
    kept_any = False

    for line in reply.split("\n"):
        if not line.strip():
            kept_lines.append(line)
            continue
        kept: list[str] = []
        for sentence in _SENTENCE_END.split(line):
            bad = _findings_in(sentence, facts)
            if bad:
                findings.extend(bad)
                removed.append(sentence.strip())
            else:
                kept.append(sentence)
                kept_any = True
        # A line that lost every sentence is dropped whole, so a stripped bullet
        # doesn't leave its marker behind.
        if kept:
            kept_lines.append(" ".join(s.strip() for s in kept))

    if not findings:
        return Verdict(reply=reply)
    if not kept_any:
        return Verdict(reply=reply, findings=tuple(findings), removed=(), rewritten=False)

    rewritten = "\n".join(kept_lines).strip()
    return Verdict(
        reply=rewritten,
        findings=tuple(findings),
        removed=tuple(removed),
        rewritten=True,
    )
