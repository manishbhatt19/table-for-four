"""Offline tests for the Streamlit chat surface.

Streamlit's own `AppTest` runs the app headlessly, so these drive the real
script — no browser, no model, no network. The session is injected directly
because the identity form and the greeting turn both need a model; everything
after that point is what we care about here.
"""

from datetime import date, timedelta

import pytest

from table_for_four.agent.concierge_chat import ConciergeSession

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = "src/table_for_four/ui/chat_app.py"
FUTURE_DATE = (date.today() + timedelta(days=7)).isoformat()
SLOTS = ["18:00", "18:30", "19:00", "19:30", "20:00", "20:30"]


def _app(session: ConciergeSession) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=30)
    at.session_state["llm"] = object()  # non-None: only used when a turn is sent
    at.session_state["session"] = session
    at.session_state["display"] = [{"role": "assistant", "content": "Here are the times."}]
    return at


def _chips(at) -> list[str]:
    """Just the time buttons — the sidebar has its own."""
    return [b.label for b in at.button if (b.key or "").startswith("slot-")]


def _session_with_times(slots=SLOTS) -> ConciergeSession:
    session = ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    session.recommendations = {"p1": {"place_id": "p1", "name": "Osteria"}}
    session.availability = {"place_id": "p1", "date": FUTURE_DATE,
                            "party_size": 2, "slots": list(slots)}
    session.pending.update({"place_id": "p1", "restaurant": "Osteria", "date": FUTURE_DATE})
    return session


def test_open_times_are_offered_as_buttons():
    at = _app(_session_with_times()).run()

    for slot in SLOTS:
        assert slot in _chips(at), f"{slot} was returned by the tool but not offered"


def test_only_times_a_tool_returned_are_ever_shown():
    # Invariant 1 reaches the UI too: the chips are the slots from the last
    # availability check, never a guess at what a restaurant might manage.
    at = _app(_session_with_times(["19:00"])).run()

    assert _chips(at) == ["19:00"]


def test_tapping_a_time_speaks_as_the_guest(monkeypatch):
    # The whole point of routing a tap back through the message path: the choice
    # has to land in the transcript as the guest's own words, because that is what
    # the pass reads before it will book a time. A chip that quietly set state
    # would walk straight around `_requested_times`.
    from langchain_core.messages import HumanMessage

    import table_for_four.agent.concierge_chat as cc

    # The app re-imports this by name on every run, so patching the source module
    # before the run is what the fresh import picks up.
    monkeypatch.setattr(cc, "_run_turn", lambda *_a, **_k: "Lovely — booking that now.")

    session = _session_with_times()
    at = _app(session)
    at.run()

    at.button(key=f"slot-p1-{FUTURE_DATE}-19:00").click().run()

    said = [m.content for m in session.messages if isinstance(m, HumanMessage)]
    assert "19:00" in said, "the tap never reached the transcript"
    assert "19:00" in cc._requested_times(session), "the pass would not see the choice"


def test_no_chips_before_any_times_have_been_looked_up():
    session = ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    at = _app(session).run()

    assert _chips(at) == []


def test_the_offer_is_spent_once_the_table_is_booked():
    # Times already taken up are history, not a standing invitation to rebook.
    session = _session_with_times()
    session.bookings[f"p1|{FUTURE_DATE}|19:00"] = {"confirmation_id": "TF4-0001"}
    at = _app(session).run()

    assert _chips(at) == []
