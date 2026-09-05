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

    # Labelled the way a guest reads a time, not the way the ledger stores one.
    for slot, label in zip(SLOTS, ["6 PM", "6:30 PM", "7 PM", "7:30 PM", "8 PM", "8:30 PM"]):
        assert label in _chips(at), f"{slot} was returned by the tool but not offered"


def test_only_times_a_tool_returned_are_ever_shown():
    # Invariant 1 reaches the UI too: the chips are the slots from the last
    # availability check, never a guess at what a restaurant might manage.
    at = _app(_session_with_times(["19:00"])).run()

    assert _chips(at) == ["7 PM"]


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
    assert "7 PM" in said, "the tap never reached the transcript"
    # And it still resolves to the slot the booking guard compares against, which
    # is the whole reason a tap speaks instead of setting state.
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


# --- The reserve gate --------------------------------------------------------

def _awaiting(session: ConciergeSession) -> ConciergeSession:
    session.confirm_in_ui = True
    session.pending_reservation = {
        "key": f"p1|{FUTURE_DATE}|19:00",
        "restaurant": "Osteria", "address": "12 Greenwich Ave",
        "date": FUTURE_DATE, "time": "19:00", "time_display": "7 PM",
        "party_size": 2, "guest_name": "Sam", "email": "g@x.com",
        "perk": "Free dessert", "perk_sample": False, "special_requests": None,
    }
    return session


def test_the_guest_is_shown_what_they_are_about_to_book():
    at = _app(_awaiting(_session_with_times())).run()

    labels = [b.label for b in at.button]
    assert "✅ Reserve" in labels
    assert "↩️ Change my mind" in labels
    # The details are on screen, not just in the model's prose.
    page = " ".join(m.value for m in at.markdown)
    for detail in ("Osteria", "7 PM", "Free dessert", "g@x.com"):
        assert detail in page, f"{detail} was not shown before the irreversible step"
    assert "Nothing is booked until you press Reserve" in page


def test_the_place_is_pictured_beside_the_question():
    # The gate is where a guest decides, and a table of eight rows is a poor
    # thing to decide over. Whatever picture the handler found goes next to it —
    # a photograph cropped to its frame, the generated card at its own width.
    session = _awaiting(_session_with_times())
    session.pending_reservation["photo"] = {
        "url": "https://img/osteria.jpg", "description": "Photo from Google Places",
        "source": "google places",
    }
    at = _app(session).run()

    page = " ".join(m.value for m in at.markdown)
    assert "https://img/osteria.jpg" in page
    assert "Photo from Google Places" in page, "an uncredited photograph of someone's room"


def test_the_gate_still_stands_without_a_picture():
    # Every photo source can come up empty, and a missing image is not a reason
    # to lose the one screen where the booking can be refused.
    at = _app(_awaiting(_session_with_times())).run()

    assert "✅ Reserve" in [b.label for b in at.button]


def test_the_open_times_step_aside_while_a_reservation_is_waiting():
    # While a booking is waiting on a decision, that is the only thing being
    # asked; time chips underneath would confuse what the buttons apply to.
    at = _app(_awaiting(_session_with_times())).run()

    assert _chips(at) == []


def test_pressing_reserve_speaks_as_the_guest(monkeypatch):
    import table_for_four.agent.concierge_chat as cc
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(cc, "_run_turn", lambda *_a, **_k: "Booked — see you Friday.")

    session = _awaiting(_session_with_times())
    at = _app(session)
    at.run()
    at.button(key="gate-reserve").click().run()

    said = [m.content for m in session.messages if isinstance(m, HumanMessage)]
    assert any("reserve it" in s.lower() for s in said)
    assert session.pending_reservation is None
    assert f"p1|{FUTURE_DATE}|19:00" in session.reserved


def test_changing_your_mind_puts_the_card_away(monkeypatch):
    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "_run_turn", lambda *_a, **_k: "Of course — what shall we change?")

    session = _awaiting(_session_with_times())
    at = _app(session)
    at.run()
    at.button(key="gate-cancel").click().run()

    assert session.pending_reservation is None
    assert session.reserved == set()


def test_the_other_times_stay_tappable_after_the_guest_names_one():
    # Demo feedback: a guest who said "7pm" stopped being shown buttons, so the
    # alternatives came back as prose — a list to read where a moment before there
    # were times to tap. They stay, relabelled: their time stands, the rest are
    # there to look over.
    from langchain_core.messages import HumanMessage

    session = _session_with_times()
    session.messages = [HumanMessage(content="7pm please")]
    at = _app(session).run()

    assert "7 PM" in _chips(at)
    assert any("only if you'd like to change it" in c.value for c in at.caption)


def test_the_picker_stays_while_the_guest_has_not_answered():
    from langchain_core.messages import HumanMessage

    session = _session_with_times()
    session.messages = [HumanMessage(content="somewhere nice for dinner")]
    at = _app(session).run()

    assert "7 PM" in _chips(at)


def test_a_time_that_is_not_open_does_not_dismiss_the_picker():
    # They asked for 9pm and it isn't free — that is exactly when they still need
    # to see what is.
    from langchain_core.messages import HumanMessage

    session = _session_with_times(["18:00", "19:00"])
    session.messages = [HumanMessage(content="9pm if you have it")]
    at = _app(session).run()

    assert _chips(at) == ["6 PM", "7 PM"]


def test_after_reserve_the_guest_gets_the_confirmation_not_the_picker(monkeypatch):
    # The other half of the same report: pressing Reserve handed back a time
    # picker instead of the booking, its photos and the perk.
    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "create_booking",
                        lambda **k: {"booked": True, "confirmation_id": "TF4-0001", "booking": {}})
    monkeypatch.setattr(cc.profile_memory, "remember", lambda *a, **k: {"email": "g@x.com"})
    monkeypatch.setattr(cc, "place_photos", lambda refs: [])
    monkeypatch.setattr(cc, "lookup_dining_highlights", lambda **k: {
        "source": "live", "citations": [],
        "highlights": [{"title": "Osteria", "snippet": "The cacio e pepe.",
                        "url": "https://guide.example/o"}],
        "images": [{"url": "https://osteria.example/room.jpg", "source": "osteria.example"}],
    })
    monkeypatch.setattr(cc, "_run_turn", lambda *_a, **_k: "All set — see you Friday.")

    session = _awaiting(_session_with_times())
    session.recommendations = {"p1": {"place_id": "p1", "name": "Osteria",
                                      "cuisine": "italian", "perk_title": "Free dessert"}}
    session.pending_reservation["args"] = {
        "place_id": "p1", "date": FUTURE_DATE, "time": "19:00", "party_size": 2,
    }
    at = _app(session)
    at.run()
    at.button(key="gate-reserve").click().run()

    assert session.bookings, "Reserve did not produce a booking"
    booked = next(iter(session.bookings.values()))
    assert booked["confirmation_id"] == "TF4-0001"
    assert booked["perk_applied"] == "Free dessert"
    assert session.media, "the photos should be waiting with the confirmation"

    # And they have to actually reach the reply. The media watermark used to be
    # taken after the gate had already run, so the photos a booking collected
    # counted as old and were dropped — a confirmed table with no pictures.
    last = at.session_state["display"][-1]
    assert last["role"] == "assistant"
    assert last["media"], "the confirmation went out without its photos"
    assert last["media"][0]["images"][0]["url"] == "https://osteria.example/room.jpg"
    assert _chips(at) == [], "a booked outing must not be offered times again"
