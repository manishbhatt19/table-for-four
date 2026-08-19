"""M4 — the human gate, the grounding check, and the trail that records both.

Everything here is offline: the gate is answered by a callable rather than a
person, and the grounding check never calls a model by design.
"""

import json
from datetime import date, timedelta

from table_for_four.governance import audit, grounding
from table_for_four.governance.grounding import GroundedFacts

FUTURE_DATE = (date.today() + timedelta(days=7)).isoformat()


def _facts(**kw) -> GroundedFacts:
    return GroundedFacts(
        times=frozenset(kw.get("times", {"19:00", "20:00"})),
        dates=frozenset(kw.get("dates", {FUTURE_DATE})),
        confirmation_ids=frozenset(kw.get("ids", {"TF4-0001"})),
        emails=frozenset(kw.get("emails", {"sam@example.com"})),
    )


# --- What the grounding check catches ----------------------------------------

def test_a_time_no_tool_ever_offered_is_removed():
    # The gap this closes: eleven refusals stop a booking at an invented time,
    # and not one of them stops Dino saying "how does 7pm sound?" first.
    verdict = grounding.check(
        "Great news — they have a table. How does 6pm sound?", _facts()
    )

    assert not verdict.grounded
    assert [f.kind for f in verdict.findings] == ["time"]
    assert "6pm" not in verdict.reply
    assert "Great news" in verdict.reply, "the good sentence must survive"


def test_a_confirmation_id_the_ledger_never_issued_is_removed():
    verdict = grounding.check(
        "You're all set! Your confirmation is TF4-9999. See you then.", _facts()
    )

    assert [f.value for f in verdict.findings] == ["TF4-9999"]
    assert "TF4-9999" not in verdict.reply
    assert "See you then." in verdict.reply


def test_an_email_the_guest_never_gave_is_removed():
    verdict = grounding.check(
        "I'll send it to sam@exampl.com just to confirm.", _facts()
    )

    assert [f.kind for f in verdict.findings] == ["email"]
    assert verdict.rewritten is False, "nothing survived, so nothing was rewritten"


def test_the_real_confirmation_and_time_pass_untouched():
    reply = ("All booked — TF4-0001, 19:00, and I'll email sam@example.com. "
             f"That's {FUTURE_DATE}.")
    verdict = grounding.check(reply, _facts())

    assert verdict.grounded
    assert verdict.reply == reply
    assert verdict.removed == ()


# --- What it must NOT catch, because a false positive deletes a good sentence --

def test_the_cancellation_policy_is_not_a_time():
    # "up to 24 hours before" is the policy Dino is required to state. A checker
    # that reads bare numbers as times would delete it every single booking.
    verdict = grounding.check(
        "Free to cancel up to 24 hours before, and there are 4 of you.", _facts()
    )
    assert verdict.grounded, f"false positive: {verdict.findings}"


def test_prices_addresses_and_ratings_are_not_times():
    verdict = grounding.check(
        "It's about $45 a head, rated 4.6, at 1-2-3 Ginza, Chuo City 104-0061.",
        _facts(),
    )
    assert verdict.grounded, f"false positive: {verdict.findings}"


def test_both_readings_of_an_ambiguous_time_are_allowed():
    # "7:30" could be 07:30 or 19:30. If either is a real slot, the guest is
    # being told something true, and deleting it would be the worse error.
    verdict = grounding.check("7:30 works!", _facts(times={"19:30"}))
    assert verdict.grounded


# --- How it rewrites ----------------------------------------------------------

def test_a_stripped_bullet_does_not_leave_its_marker_behind():
    verdict = grounding.check(
        "Two options:\n- Osteria at 19:00\n- Trattoria at 6am", _facts()
    )

    assert "6am" not in verdict.reply
    assert "- Osteria at 19:00" in verdict.reply
    assert "- Trattoria" not in verdict.reply, "the whole bullet should go"


def test_a_reply_that_fails_everywhere_is_sent_anyway_and_recorded():
    # The fail-safe. A reply where every sentence trips the check is far more
    # likely to be a bug in here than a model that invented every word, and
    # sending the guest nothing at all is the worse failure.
    reply = "Booked at 6am. Confirmation TF4-9999."
    verdict = grounding.check(reply, _facts())

    assert verdict.reply == reply       # unchanged
    assert verdict.rewritten is False
    assert len(verdict.findings) == 2   # but nothing is hidden
    assert not verdict.grounded


def test_an_empty_reply_is_left_alone():
    assert grounding.check("", _facts()).reply == ""


# --- Facts come from what tools actually returned -----------------------------

def test_facts_accumulate_across_the_session_not_just_the_last_lookup():
    # Recapping the times of a restaurant discussed ten turns ago is legitimate,
    # so the allowed set must not reset every time availability is re-checked.
    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    session.offered_times.update({"19:00", "20:00"})
    session.offered_dates.add(FUTURE_DATE)
    session.bookings["p1|x|21:00"] = {"confirmation_id": "TF4-0007", "time": "21:00",
                                      "date": FUTURE_DATE}

    facts = GroundedFacts.gather(session)

    assert {"19:00", "20:00", "21:00"} <= facts.times
    assert "TF4-0007" in facts.confirmation_ids
    assert facts.emails == frozenset({"g@x.com"})


def test_a_returning_guests_own_history_is_quotable():
    # recall_guest_profile hands these ids back, so repeating one is grounded.
    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={
        "email": "g@x.com",
        "past_bookings": [{"confirmation_id": "TF4-0003", "restaurant": "Osteria"}],
    })
    assert "TF4-0003" in GroundedFacts.gather(session).confirmation_ids


# --- The trail ----------------------------------------------------------------

def test_the_trail_is_append_only_and_names_an_actor():
    trail = audit.Trail(member_id="g@x.com")
    trail.record("tool_call", actor="scout", tool="recommend_restaurants", outcome="ok")
    trail.record("tool_call", actor="booker", tool="book_table", outcome="booked")

    assert [r.actor for r in trail.records] == ["scout", "booker"]
    assert all(r.member_id == "g@x.com" for r in trail.records)
    assert all(r.at for r in trail.records)
    assert len(trail.of("tool_call")) == 2


def test_a_record_cannot_be_edited_after_the_fact():
    import dataclasses

    import pytest

    record = audit.Trail().record("approval", actor=None, approved=True)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.approved = False  # type: ignore[misc]


def test_every_tool_call_in_the_chat_path_lands_in_the_trail(monkeypatch):
    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    session.recommendations = {"p1": {"place_id": "p1", "name": "Osteria"}}
    session.pending["party_size"] = 2

    cc._dispatch(session, "check_availability_times", {"place_id": "p1", "date": FUTURE_DATE})

    call = session.trail.of("tool_call")[-1]
    assert call.actor == "scout"          # the unit that owns the handler
    assert call.detail["tool"] == "check_availability_times"
    assert call.detail["outcome"] in {"ok", "no_availability"}


# --- The human gate -----------------------------------------------------------

def _gate_run(approve, thread):
    from table_for_four.agent.graph import run_concierge
    from table_for_four.mcp_servers.booking.backend.app import reset_store

    reset_store()
    return run_concierge("Italian for 2 on Friday", use_llm=False,
                         thread_id=thread, approve=approve)


def test_nothing_is_booked_until_a_human_says_yes():
    # The milestone, in one assertion: create_booking is the only irreversible
    # step, so it is the one step the agent may not take on its own say-so.
    final = _gate_run("no", "declined")

    assert final["approved"] is False
    assert final.get("booking") is None
    assert any("declined" in line for line in final["log"])


def test_no_approver_at_all_declines_rather_than_defaulting_to_yes():
    # A convenient default would quietly undo the whole gate.
    final = _gate_run(None, "unattended")

    assert final["approved"] is False
    assert final.get("booking") is None


def test_a_resume_that_is_not_clearly_a_yes_declines():
    # Malformed input must not be read as consent.
    for i, decision in enumerate([{"lol": 1}, "maybe", "", 0]):
        final = _gate_run(decision, f"garbled-{i}")
        assert final["approved"] is False, f"{decision!r} was treated as approval"


def test_saying_yes_books_and_the_trail_records_who_was_shown_what():
    shown = {}
    final = _gate_run(lambda payload: shown.update(payload) or "yes", "approved")

    assert final["booking"]["booked"] is True
    # The human saw the irreversible details before answering.
    assert shown["restaurant"] and shown["time"] and shown["irreversible"] is True
    approval = next(r for r in final["audit"] if r["event"] == "approval")
    assert approval["detail"]["approved"] is True
    assert approval["detail"]["shown"]["time"] == shown["time"]


def test_the_closing_audit_record_states_whether_a_human_approved():
    final = _gate_run(True, "closing")
    summary = json.loads(
        next(line for line in final["log"] if line.startswith("[audit]"))[len("[audit] "):]
    )

    assert summary["approved_by_human"] is True
    assert summary["actors"] == ["scout", "booker"]
    assert [r["event"] for r in final["audit"]] == ["approval", "run_complete"]
