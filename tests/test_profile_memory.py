"""Offline tests for the long-term member-profile store.

These drive an **ephemeral** Chroma collection (no persistent store, no network
beyond the one-time local embedding-model download Chroma caches) through the
pure functions that take a `Collection`, so no LLM or chat loop is involved.
"""

from datetime import date, timedelta

import chromadb
import pytest

from table_for_four.agent.profile_memory import (
    build_collection,
    load_profile,
    looks_like_email,
    normalize_id,
    profile_summary,
    resolve_key,
    save_profile,
    search_profiles,
    set_booking_status,
    set_email,
    sticky_conflicts,
    update_profile,
)

# `_handle_book` refuses a date in the past, so booking tests must aim at a date
# that stays in the future as the calendar moves — never a hardcoded one.
FUTURE_DATE = (date.today() + timedelta(days=7)).isoformat()


@pytest.fixture()
def collection():
    # Fresh per test so accumulation assertions start clean.
    return build_collection(chromadb.EphemeralClient())


def test_normalize_id_is_stable_and_slug_like():
    assert normalize_id("  Manish Bhatt ") == "manish-bhatt"
    assert normalize_id("Manish!!") == "manish"
    assert normalize_id("") == "guest"


def test_unknown_member_returns_none(collection):
    assert load_profile(collection, "nobody") is None


def test_search_profiles_empty_store_returns_nothing(collection):
    assert search_profiles(collection, "anyone at all") == []


def test_search_profiles_recalls_member_by_meaning(collection):
    # Three members with distinct tastes, embedded via their profile summaries.
    update_profile(collection, "wine@x.com", {
        "name": "Giulia", "cuisines": ["Italian"],
        "interests": ["Sicilian wine", "sommelier pairings"], "dining_atmosphere": "romantic",
    })
    update_profile(collection, "tacos@x.com", {
        "name": "Diego", "cuisines": ["Mexican"], "interests": ["street tacos", "margaritas"],
    })
    update_profile(collection, "sushi@x.com", {
        "name": "Aiko", "cuisines": ["Japanese"], "interests": ["omakase", "sake"],
    })

    hits = search_profiles(collection, "the guest who loves Sicilian wine")
    assert hits, "expected a semantic match"
    assert hits[0]["member_id"] == "wine@x.com"
    assert hits[0]["name"] == "Giulia"
    assert 0.0 <= hits[0]["similarity"] <= 1.0
    # Ranked, best first.
    assert all(hits[i]["similarity"] >= hits[i + 1]["similarity"] for i in range(len(hits) - 1))


def test_save_and_load_roundtrip(collection):
    saved = save_profile(collection, "Manish", {"name": "Manish", "pronouns": "he/him"})
    assert saved["member_id"] == "manish"
    assert "updated_at" in saved

    loaded = load_profile(collection, "manish")
    assert loaded["name"] == "Manish"
    assert loaded["pronouns"] == "he/him"


def test_update_merges_scalars_and_unions_lists(collection):
    update_profile(collection, "Manish", {"cuisines": ["italian"], "pronouns": "he/him"})
    update_profile(collection, "Manish", {"cuisines": ["japanese", "italian"]})
    update_profile(collection, "Manish", {"home_location": "Midtown"})

    p = load_profile(collection, "manish")
    # Cuisines dedupe and are ordered by recency, newest last — mentioning italian
    # again in the second update moves it after japanese.
    assert p["cuisines"] == ["japanese", "italian"]
    assert p["pronouns"] == "he/him"
    assert p["home_location"] == "Midtown"


def test_a_list_value_is_not_duplicated_by_its_own_casing(collection):
    # Seen in the demo store: ['Chinese', 'Italian', 'italian'] — one favourite
    # counted twice, which also burns a slot in the three-cuisine window.
    update_profile(collection, "Casey", {"cuisines": ["Italian"], "dietary": ["Gluten-Free"]})
    update_profile(collection, "Casey", {"cuisines": ["italian"], "dietary": ["gluten-free"]})

    p = load_profile(collection, "casey")
    assert p["cuisines"] == ["italian"]      # re-mentioned, so the newer spelling wins
    assert p["dietary"] == ["Gluten-Free"]   # kept as first recorded, not doubled


def test_cuisines_keep_only_the_three_most_recent(collection):
    # Taste drifts. Cuisines are a rolling window of what the guest has actually
    # been eating lately, not a permanent list that grows forever.
    for cuisine in ["thai", "italian", "japanese", "mexican"]:
        update_profile(collection, "Manish", {"cuisines": [cuisine]})

    p = load_profile(collection, "manish")
    assert p["cuisines"] == ["italian", "japanese", "mexican"]
    assert "thai" not in p["cuisines"], "the oldest preference should have aged out"


def test_revisiting_a_cuisine_keeps_it_alive(collection):
    update_profile(collection, "Manish", {"cuisines": ["thai"]})
    update_profile(collection, "Manish", {"cuisines": ["italian"]})
    update_profile(collection, "Manish", {"cuisines": ["japanese"]})
    update_profile(collection, "Manish", {"cuisines": ["thai"]})      # back to thai
    update_profile(collection, "Manish", {"cuisines": ["mexican"]})

    p = load_profile(collection, "manish")
    # Italian was pushed out, but thai survived because it was ordered again.
    assert p["cuisines"] == ["japanese", "thai", "mexican"]


def test_dietary_needs_are_never_aged_out(collection):
    # The distinction that matters: a preference can expire, an allergy cannot.
    for need in ["gluten-free", "nut allergy", "shellfish allergy", "no dairy"]:
        update_profile(collection, "Manish", {"dietary": [need]})

    p = load_profile(collection, "manish")
    assert p["dietary"] == ["gluten-free", "nut allergy", "shellfish allergy", "no dairy"]


def test_update_ignores_none_and_preserves_existing(collection):
    save_profile(collection, "Ada", {"name": "Ada", "dietary": ["vegan"]})
    update_profile(collection, "Ada", {"name": None, "cuisines": ["thai"]})

    p = load_profile(collection, "ada")
    assert p["name"] == "Ada"          # not blanked by the None update
    assert p["dietary"] == ["vegan"]   # untouched
    assert p["cuisines"] == ["thai"]


def test_kids_and_high_chair_survive_roundtrip(collection):
    save_profile(collection, "Sam", {
        "name": "Sam",
        "kids": [{"age": 3, "needs_high_chair": True}, {"age": 8, "needs_high_chair": False}],
    })
    p = load_profile(collection, "sam")
    assert p["kids"][0]["needs_high_chair"] is True
    assert p["kids"][1]["age"] == 8


def test_resolve_key_distinguishes_email_from_name():
    assert resolve_key("Manish") == "manish"
    assert resolve_key("Manish Bhatt") == "manish-bhatt"
    assert resolve_key("Manish@Example.com") == "manish@example.com"
    assert looks_like_email("a@b.co") and not looks_like_email("just-a-name")


def test_set_email_becomes_unique_key_and_migrates_provisional(collection):
    # A name-keyed provisional profile built up during intake...
    save_profile(collection, "Manish", {"name": "Manish", "cuisines": ["italian"]})
    key, prof, returning = set_email(collection, "Manish", "Manish@Example.com")

    assert key == "manish@example.com"      # email normalized + canonical
    assert returning is False
    assert prof["email"] == "manish@example.com"
    assert prof["cuisines"] == ["italian"]  # carried over from the provisional doc
    assert load_profile(collection, "Manish") is None          # provisional removed
    assert load_profile(collection, "manish@example.com")["name"] == "Manish"


def test_returning_member_recognized_by_email_and_details_merge(collection):
    # First visit: member established under their email with a standing dietary need.
    save_profile(collection, "Manish", {"name": "Manish", "dietary": ["gluten-free"]})
    set_email(collection, "Manish", "m@x.com")

    # Second visit: fresh name-keyed session, a new cuisine, SAME email.
    save_profile(collection, "Manish", {"name": "Manish", "cuisines": ["japanese"]})
    key, prof, returning = set_email(collection, "Manish", "m@x.com")

    assert returning is True                 # recognized as a repeat guest
    assert prof["dietary"] == ["gluten-free"]  # remembered from the first visit
    assert prof["cuisines"] == ["japanese"]    # added this visit


def test_email_must_come_from_a_guest_message(monkeypatch):
    # The model cannot invent an email: set_confirmation_email rejects any address
    # that doesn't appear in the guest's own messages.
    import json as _json

    import chromadb

    import table_for_four.agent.concierge_chat as cc
    from table_for_four.agent import profile_memory as pm
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(pm, "_collection", pm.build_collection(chromadb.EphemeralClient()))

    session = cc.ConciergeSession(member_id="manish")
    session.messages = [HumanMessage(content="Hi, I'd like an Italian table for four.")]

    invented = _json.loads(cc._handle_email(session, {"email": "made-up@example.com"}))
    assert invented["status"] == "rejected"

    session.messages.append(HumanMessage(content="Sure, my email is real.guest@example.com"))
    accepted = _json.loads(cc._handle_email(session, {"email": "real.guest@example.com"}))
    assert accepted["status"] == "saved"
    assert accepted["email"] == "real.guest@example.com"


def test_returning_member_recognized_when_email_given(monkeypatch):
    # Demo-feedback regression: giving a previously-seen email must flag the guest
    # as returning and hand the model a name + last booking to welcome them back.
    import json as _json

    import chromadb

    import table_for_four.agent.concierge_chat as cc
    from table_for_four.agent import profile_memory as pm
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(pm, "_collection", pm.build_collection(chromadb.EphemeralClient()))

    # A member who booked on a previous visit, keyed by email.
    pm.update_profile(pm._collection, "repeat@x.com", {
        "name": "Giulia", "cuisines": ["Italian"],
        "past_bookings": [{"restaurant": "Osteria Midtown", "confirmation_id": "TF4-0001",
                           "date": "2026-08-01", "status": "confirmed"}],
    })

    # New session (started under a fresh handle); the guest types the same email.
    session = cc.ConciergeSession(member_id="a-guest")
    session.messages = [HumanMessage(content="my email is repeat@x.com")]
    out = _json.loads(cc._handle_email(session, {"email": "repeat@x.com"}))

    assert out["returning_member"] is True
    assert out["saved_preferences"]["name"] == "Giulia"
    assert out["last_booking"]["restaurant"] == "Osteria Midtown"
    assert "welcome" in out["note"].lower()


def _listed(session, place_id="p1", name="Osteria", perk_id=None):
    session.recommendations = {place_id: {"place_id": place_id, "name": name, "perk_id": perk_id}}


def test_book_is_idempotent_per_request(monkeypatch):
    # An identical booking must not create a second reservation.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    calls = {"n": 0}

    def fake_create(**_kw):
        calls["n"] += 1
        return {"booked": True, "confirmation_id": f"TF4-000{calls['n']}", "booking": {}}

    monkeypatch.setattr(cc, "create_booking", fake_create)
    monkeypatch.setattr(cc.profile_memory, "remember", lambda *a, **k: {"email": "g@x.com", "party_size": 4})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com", "party_size": 4})
    _listed(session)
    args = {"place_id": "p1", "date": FUTURE_DATE, "time": "19:00", "party_size": 4}

    first = _json.loads(cc._handle_book(session, args))
    second = _json.loads(cc._handle_book(session, args))

    assert first["status"] == "booked" and first["confirmation_id"] == "TF4-0001"
    assert second["status"] == "already_booked"
    assert second["confirmation_id"] == "TF4-0001"
    assert calls["n"] == 1  # booking backend hit exactly once


def test_book_is_gated_on_email(monkeypatch):
    # book_table must refuse until a confirmation email is on file — and must NOT
    # hit the booking backend when it refuses.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    called = {"n": 0}
    monkeypatch.setattr(cc, "create_booking", lambda **k: called.__setitem__("n", called["n"] + 1) or {})

    session = cc.ConciergeSession(member_id="manish", profile={"name": "Manish"})
    _listed(session)
    out = _json.loads(cc._handle_book(session, {"place_id": "p1", "date": "Friday", "time": "19:00"}))

    assert out["status"] == "email_required"
    assert called["n"] == 0


def test_book_rejects_a_restaurant_not_in_recommendations():
    # The model cannot book a place that was never surfaced to the guest.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    _listed(session)  # only p1 is listed
    out = _json.loads(cc._handle_book(session, {"place_id": "ghost", "date": "Friday", "time": "19:00"}))
    assert out["status"] == "unknown_restaurant"


def test_book_requires_a_party_size(monkeypatch):
    # book_table must refuse — never silently default — when no party size is known,
    # and must NOT hit the booking backend when it refuses.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    called = {"n": 0}
    monkeypatch.setattr(cc, "create_booking", lambda **k: called.__setitem__("n", called["n"] + 1) or {})

    # Email on file (passes the email gate) but no party size anywhere.
    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    _listed(session)
    out = _json.loads(cc._handle_book(session, {"place_id": "p1", "date": FUTURE_DATE, "time": "19:00"}))

    assert out["status"] == "need_party_size"
    assert called["n"] == 0  # booking backend never hit without a party size


def test_every_tool_result_restates_what_we_already_know():
    # Guests notice being asked twice. Rather than trusting the model to remember
    # across a long chat, each tool result carries the known facts back to it.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(
        member_id="g@x.com", profile={"email": "g@x.com", "name": "Sam"}
    )
    session.pending["party_size"] = 4
    _listed(session)

    out = _json.loads(cc._dispatch(session, "check_availability_times", {"place_id": "p1"}))

    assert out["known_so_far"]["email_on_file"] == "g@x.com"
    assert out["known_so_far"]["guest_name"] == "Sam"
    assert out["known_so_far"]["party_size"] == 4
    assert "Do NOT ask for it again" in out["do_not_ask"]


def test_no_email_reminder_when_there_is_no_email():
    # The reminder must not appear when the email genuinely still needs asking for.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="sam", profile={"name": "Sam"})
    _listed(session)
    out = _json.loads(cc._dispatch(session, "check_availability_times", {"place_id": "p1"}))

    assert "do_not_ask" not in out
    assert "email_on_file" not in out.get("known_so_far", {})


def test_booking_records_the_cuisine_as_a_recent_preference(monkeypatch):
    # An actual booking is stronger evidence of taste than anything said in passing.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    monkeypatch.setattr(cc, "create_booking",
                        lambda **k: {"booked": True, "confirmation_id": "TF4-0001", "booking": {}})
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {"email": "g@x.com"})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    session.recommendations = {"p1": {"place_id": "p1", "name": "Osteria", "cuisine": "italian"}}
    out = _json.loads(cc._handle_book(session, {
        "place_id": "p1", "date": FUTURE_DATE, "time": "19:00", "party_size": 4,
    }))

    assert out["status"] == "booked"
    assert remembered["cuisines"] == ["italian"]


def test_sticky_conflicts_only_flags_values_already_on_file():
    # A first value is learning, not a change — nothing to confirm.
    assert sticky_conflicts({}, {"home_location": "Midtown", "party_size": 2}) == {}

    profile = {"home_location": "Midtown", "party_size": 2, "cuisines": ["italian"]}
    # Same value (or a different casing of it) is not a change either.
    assert sticky_conflicts(profile, {"home_location": "midtown ", "party_size": 2}) == {}
    # A genuinely different value is flagged, with both sides for the question.
    flagged = sticky_conflicts(profile, {"home_location": "Brooklyn", "party_size": 6})
    assert flagged["home_location"] == {"saved": "Midtown", "proposed": "Brooklyn"}
    assert flagged["party_size"] == {"saved": 2, "proposed": 6}
    # Below the cap a new cuisine just gets learned; at the cap it would displace
    # one of the guest's three, so it needs asking about.
    assert "cuisines" not in sticky_conflicts(profile, {"cuisines": ["thai"]})
    full = {"cuisines": ["italian", "japanese", "mexican"]}
    assert sticky_conflicts(full, {"cuisines": ["thai"]})["cuisines"]["proposed"] == ["thai"]
    assert sticky_conflicts(full, {"cuisines": ["Italian"]}) == {}  # already a favourite


def _booking_session(profile):
    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com", **profile})
    session.recommendations = {"p1": {"place_id": "p1", "name": "Osteria", "cuisine": "thai"}}
    session.pending["location"] = "Brooklyn"
    return session


def _patch_booking(monkeypatch, remembered):
    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "create_booking",
                        lambda **k: {"booked": True, "confirmation_id": "TF4-0001", "booking": {}})
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {"email": "g@x.com"})


def test_booking_never_rewrites_a_standing_preference_on_its_own(monkeypatch):
    # The bug this guards: a night out in Brooklyn for six quietly became the
    # guest's home area, usual party size, and taste. Now it only gets *offered*.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    _patch_booking(monkeypatch, remembered)
    session = _booking_session({
        "home_location": "Midtown", "party_size": 2,
        "cuisines": ["italian", "japanese", "mexican"],
    })

    out = _json.loads(cc._handle_book(session, {
        "place_id": "p1", "date": FUTURE_DATE, "time": "19:00", "party_size": 6,
    }))

    assert out["status"] == "booked"
    assert set(remembered) == {"past_bookings"}          # nothing standing was touched
    assert set(out["preference_check"]["proposals"]) == {
        "home_location", "party_size", "cuisines",
    }
    assert session.pref_offer["proposals"]               # the offer is on the record

def test_booking_still_learns_preferences_it_has_never_had(monkeypatch):
    # Consent is needed to *change* a saved preference, not to learn a first one.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    _patch_booking(monkeypatch, remembered)
    session = _booking_session({})

    out = _json.loads(cc._handle_book(session, {
        "place_id": "p1", "date": FUTURE_DATE, "time": "19:00", "party_size": 6,
    }))

    assert "preference_check" not in out
    assert remembered["home_location"] == "Brooklyn"
    assert remembered["party_size"] == 6
    assert remembered["cuisines"] == ["thai"]


def test_remember_tool_cannot_overwrite_a_standing_preference(monkeypatch):
    # The model saving "home_location: Brooklyn" mid-search must not land either.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {"email": "g@x.com"})

    session = cc.ConciergeSession(
        member_id="g@x.com", profile={"email": "g@x.com", "home_location": "Midtown"}
    )
    out = _json.loads(cc._handle_remember(
        session, {"home_location": "Brooklyn", "dietary": ["gluten-free"]}
    ))

    assert out["status"] == "saved_with_confirmation_needed"
    assert remembered == {"dietary": ["gluten-free"]}     # the free field still saved
    assert out["not_saved"]["home_location"]["saved"] == "Midtown"


def test_a_standing_preference_changes_only_when_the_guest_says_so(monkeypatch):
    # Silence after the offer leaves the saved value alone; a plain yes applies it.
    import json as _json

    from langchain_core.messages import HumanMessage

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {"email": "g@x.com"})

    session = cc.ConciergeSession(
        member_id="g@x.com", profile={"email": "g@x.com", "home_location": "Midtown"}
    )
    session.messages = [HumanMessage(content="book me somewhere in Brooklyn")]
    cc._offer_preference_changes(
        session, {"home_location": {"saved": "Midtown", "proposed": "Brooklyn"}}
    )

    # The guest hasn't answered yet — an eager confirm must change nothing.
    out = _json.loads(cc._handle_confirm_prefs(session, {"home_location": "Brooklyn"}))
    assert out["status"] == "not_authorized"
    assert remembered == {}

    # ...and once they say yes, it lands.
    session.messages.append(HumanMessage(content="yes please, that'd be great"))
    out = _json.loads(cc._handle_confirm_prefs(session, {"home_location": "Brooklyn"}))
    assert out["status"] == "updated"
    assert remembered == {"home_location": "Brooklyn"}
    assert session.pref_offer == {}                      # one offer, one answer


def test_guest_can_ask_for_a_preference_change_unprompted(monkeypatch):
    # No offer needed when the guest raises it themselves ("I've moved").
    import json as _json

    from langchain_core.messages import HumanMessage

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {"email": "g@x.com"})

    session = cc.ConciergeSession(
        member_id="g@x.com", profile={"email": "g@x.com", "home_location": "Midtown"}
    )
    session.messages = [HumanMessage(content="I've moved — make Brooklyn my home area")]
    out = _json.loads(cc._handle_confirm_prefs(session, {"home_location": "Brooklyn"}))

    assert out["status"] == "updated"
    assert remembered == {"home_location": "Brooklyn"}


def test_merely_naming_an_area_is_not_permission_to_change_it(monkeypatch):
    # Mentioning where tonight's dinner is — the exact source of the old drift.
    import json as _json

    from langchain_core.messages import HumanMessage

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {"email": "g@x.com"})

    session = cc.ConciergeSession(
        member_id="g@x.com", profile={"email": "g@x.com", "home_location": "Midtown"}
    )
    session.messages = [HumanMessage(content="somewhere in Brooklyn tonight, 4 of us")]
    out = _json.loads(cc._handle_confirm_prefs(
        session, {"home_location": "Brooklyn", "party_size": 4}
    ))

    assert out["status"] == "not_authorized"
    assert remembered == {}
    assert sorted(out["unchanged"]) == ["home_location", "party_size"]


def test_parse_time_tokens_reads_clock_times():
    from table_for_four.agent.concierge_chat import _parse_time_tokens

    assert "19:00" in _parse_time_tokens("7pm works")
    assert "19:30" in _parse_time_tokens("how about 7:30 pm")
    assert "19:00" in _parse_time_tokens("book 19:00")
    # A bare number (party size, guest count) is NOT a time.
    assert _parse_time_tokens("a table for 4 people") == set()
    # A colon time without am/pm keeps both readings so a real slot can match.
    both = _parse_time_tokens("7:30")
    assert {"07:30", "19:30"} <= both


def test_book_enforces_the_time_the_guest_requested(monkeypatch):
    # Regression: the guest asked for an available time; the model must book THAT
    # time, not a different open slot.
    import json as _json

    import table_for_four.agent.concierge_chat as cc
    from langchain_core.messages import HumanMessage

    called = {"n": 0}
    monkeypatch.setattr(cc, "create_booking", lambda **k: called.__setitem__("n", called["n"] + 1) or {})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com", "party_size": 2})
    _listed(session)
    session.availability = {"place_id": "p1", "date": "2026-09-04", "party_size": 2,
                            "slots": ["19:00", "20:00"]}
    session.messages = [HumanMessage(content="7pm works great")]

    out = _json.loads(cc._handle_book(session, {
        "place_id": "p1", "date": "2026-09-04", "time": "20:00", "party_size": 2,
    }))
    assert out["status"] == "time_mismatch"
    assert out["requested_times"] == ["19:00"]
    assert called["n"] == 0  # never booked the wrong time


def test_book_accepts_the_requested_time(monkeypatch):
    import json as _json

    import table_for_four.agent.concierge_chat as cc
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(cc, "create_booking",
                        lambda **k: {"booked": True, "confirmation_id": "TF4-0009", "booking": {}})
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda *a, **k: {"email": "g@x.com", "party_size": 2})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com", "party_size": 2})
    _listed(session)
    session.availability = {"place_id": "p1", "date": "2026-09-04", "party_size": 2,
                            "slots": ["19:00", "20:00"]}
    session.messages = [HumanMessage(content="let's do 7pm")]

    out = _json.loads(cc._handle_book(session, {
        "place_id": "p1", "date": "2026-09-04", "time": "19:00", "party_size": 2,
    }))
    assert out["status"] == "booked"
    assert out["time"] == "19:00"


def test_set_booking_status_updates_in_place(collection):
    # Flipping a booking's status must edit the row, not append a duplicate (the
    # list-union merge would otherwise leave both the old and new versions).
    update_profile(collection, "g@x.com", {
        "email": "g@x.com",
        "past_bookings": [{"restaurant": "Osteria", "confirmation_id": "TF4-0001",
                           "status": "confirmed"}],
    })
    set_booking_status(collection, "g@x.com", "TF4-0001", "cancelled")
    bookings = load_profile(collection, "g@x.com")["past_bookings"]
    assert len(bookings) == 1
    assert bookings[0]["status"] == "cancelled"


def _open_far_future_slot(place: str, party_size: int = 2) -> tuple[str, str]:
    from table_for_four.mcp_servers.booking.backend.app import available_slots
    for day in range(1, 29):
        date = f"2099-12-{day:02d}"
        slots = available_slots(place, date, party_size)
        if slots:
            return date, slots[0]
    raise AssertionError("no far-future open slot found")


def test_cancel_reservation_marks_history_cancelled(monkeypatch):
    # End-to-end concierge cancel: a far-future booking cancels under the real
    # clock, and long-term memory is updated to match the ledger.
    import json as _json

    import chromadb

    import table_for_four.agent.concierge_chat as cc
    from table_for_four.agent import profile_memory as pm

    monkeypatch.setattr(pm, "_collection", pm.build_collection(chromadb.EphemeralClient()))

    place = "fixture-osteria-1"
    date, slot = _open_far_future_slot(place)
    booked = cc.create_booking(
        place_id=place, restaurant_name="Osteria Midtown", date=date, time=slot,
        party_size=2, guest_name="Manish", restaurant_phone="(212) 555-0142",
        website="https://example.com/osteria-midtown", guest_email="g@x.com",
    )
    conf = booked["confirmation_id"]
    pm.update_profile(pm._collection, "g@x.com", {
        "email": "g@x.com",
        "past_bookings": [{"restaurant": "Osteria Midtown", "confirmation_id": conf,
                           "date": date, "time": slot, "party_size": 2, "status": "confirmed"}],
    })
    session = cc.ConciergeSession(
        member_id="g@x.com", profile=pm.load_profile(pm._collection, "g@x.com")
    )

    out = _json.loads(cc._handle_cancel(session, {"confirmation_id": conf}))
    assert out["status"] == "cancelled"
    stored = pm.load_profile(pm._collection, "g@x.com")["past_bookings"][0]
    assert stored["status"] == "cancelled"


def test_cancel_too_late_is_relayed_not_claimed(monkeypatch):
    # When the backend says too_late, the handler must surface the restaurant's
    # contact and the "don't claim cancellation" instruction — never a success.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "cancel_booking", lambda *a, **k: {
        "status": "too_late", "message": "within 24h",
        "restaurant_name": "Osteria Midtown", "restaurant_phone": "(212) 555-0142",
        "website": "https://example.com/osteria-midtown",
    })
    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    out = _json.loads(cc._handle_cancel(session, {"confirmation_id": "TF4-0001"}))
    assert out["status"] == "too_late"
    assert out["restaurant_phone"] == "(212) 555-0142"
    assert "instruction" in out


def test_cancel_requires_a_confirmation_id():
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    out = _json.loads(cc._handle_cancel(session, {}))
    assert out["status"] == "need_confirmation_id"


def test_recommend_filters_cuisine_and_flags_perks():
    # Uses the real (offline) search + perks fixtures.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    # A guest whose identity is settled: the "have we met?" gate doesn't apply.
    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    out = _json.loads(cc._handle_recommend(
        session, {"cuisine": "italian", "party_size": 4, "date": "Friday", "keywords": "family dinner"}
    ))
    assert out["status"] == "ok"
    names = {r["name"] for r in out["recommendations"]}
    assert names and "Le Petit Bistro" not in names          # cuisine is a hard filter
    assert session.recommendations                            # stored for the next step
    assert isinstance(out["restaurants_with_perks"], list)


def test_recommend_attaches_sample_perks_on_live_data(monkeypatch):
    # On live data, real Google ids don't match fixture-keyed perks — so 1-2 top
    # recommendations should get a clearly-labeled SAMPLE offer instead.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "search_restaurants", lambda **_k: {"source": "live", "results": [
        {"place_id": "ChIJ_a", "name": "Real Italian A", "primary_type": "italian_restaurant", "rating": 4.8},
        {"place_id": "ChIJ_b", "name": "Real Italian B", "primary_type": "italian_restaurant", "rating": 4.6},
        {"place_id": "ChIJ_c", "name": "Real Italian C", "primary_type": "italian_restaurant", "rating": 4.5},
    ]})

    def fake_find_perks(query, place_ids=None, **_k):
        if place_ids:  # nothing matches the live ids
            return {"results": []}
        return {"results": [
            {"perk_id": "perk-1", "place_id": "fixture-x", "title": "Aperitivo Welcome", "similarity": 0.9},
            {"perk_id": "perk-2", "place_id": "fixture-y", "title": "Weekend Family Feast", "similarity": 0.8},
        ]}

    monkeypatch.setattr(cc, "find_perks", fake_find_perks)

    session = cc.ConciergeSession(member_id="p@x.com", profile={"email": "p@x.com"})
    out = _json.loads(cc._handle_recommend(session, {"cuisine": "italian", "keywords": "italian dinner"}))

    perked = [r for r in out["recommendations"] if r["has_perk"]]
    assert 1 <= len(perked) <= 2
    assert all(r["perk_sample"] for r in perked)     # labeled as illustrative
    assert "perk_note" in out                         # guest-facing sample disclaimer
    assert out["restaurants_with_perks"]


def test_check_times_requires_a_listed_restaurant():
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g")
    out = _json.loads(cc._handle_times(session, {"place_id": "ghost", "date": "Friday"}))
    assert out["status"] == "unknown_restaurant"


def test_profile_summary_mentions_key_facts():
    summary = profile_summary({
        "name": "Manish",
        "pronouns": "he/him",
        "cuisines": ["italian"],
        "dietary": ["gluten-free"],
    })
    assert "Manish" in summary
    assert "italian" in summary
    assert "gluten-free" in summary


# --- Cuisine hygiene ---------------------------------------------------------

def test_a_place_category_is_never_read_as_a_cuisine():
    # Demo-feedback regression: Google's primary_type is as often a generic
    # category as a cuisine, and "restaurant" was landing in guests' favourites.
    import table_for_four.agent.concierge_chat as cc

    for category in ("restaurant", "bar", "cafe", "food", "fine_dining_restaurant", ""):
        assert cc._cuisine_from_place_type(category) is None, category

    assert cc._cuisine_from_place_type("italian_restaurant") == "italian"
    assert cc._cuisine_from_place_type("vegan_restaurant") == "vegan"
    assert cc._cuisine_from_place_type("steak_house") == "steakhouse"


def test_free_text_cuisine_is_cleaned_or_rejected():
    import table_for_four.agent.concierge_chat as cc

    assert cc._clean_cuisine("Italian restaurant") == "italian"
    assert cc._clean_cuisine("  THAI food ") == "thai"
    assert cc._clean_cuisine("restaurant") is None
    assert cc._clean_cuisine("fast food") is None
    assert cc._clean_cuisine("") is None


def test_remember_refuses_a_venue_name_or_category_as_a_cuisine(monkeypatch):
    # The model sometimes offers the restaurant itself as the guest's taste. A name
    # is not a cuisine, and a guest must never be greeted with "you love Osteria!".
    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    monkeypatch.setattr(cc.profile_memory, "remember",
                        lambda _id, updates: remembered.update(updates) or {})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    _listed(session, name="Osteria Morini")

    assert cc._handle_remember(session, {"cuisines": ["restaurant"]}) == "Nothing to save."
    assert cc._handle_remember(session, {"cuisines": ["Osteria Morini"]}) == "Nothing to save."
    assert remembered == {}  # neither reached long-term memory

    cc._handle_remember(session, {"cuisines": ["Italian restaurant", "food"]})
    assert remembered["cuisines"] == ["italian"]


def test_booking_a_generically_typed_place_falls_back_to_the_searched_cuisine(monkeypatch):
    # Live Google results are often typed plain "restaurant". What the guest asked
    # for is then the honest taste signal — and nothing at all is better than junk.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    remembered: dict = {}
    _patch_booking(monkeypatch, remembered)

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    session.recommendations = {"p1": {"place_id": "p1", "name": "Corner Table", "cuisine": None}}
    session.pending["cuisine"] = "italian"  # what they searched for
    args = {"place_id": "p1", "date": FUTURE_DATE, "time": "19:00", "party_size": 4}
    assert _json.loads(cc._handle_book(session, args))["status"] == "booked"
    assert remembered["cuisines"] == ["italian"]

    remembered.clear()
    session.pending.pop("cuisine")
    session.bookings.clear()  # same booking again, this time with nothing to infer
    assert _json.loads(cc._handle_book(session, args))["status"] == "booked"
    assert "cuisines" not in remembered


# --- "Have we met before?" ---------------------------------------------------

def _patch_search(monkeypatch, calls):
    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "search_restaurants", lambda **k: calls.append(k) or {
        "source": "fixture",
        "results": [{"place_id": "p1", "name": "Osteria", "primary_type": "italian_restaurant"}],
    })
    monkeypatch.setattr(cc, "find_perks", lambda **k: {"results": []})


def test_first_search_asks_whether_the_guest_has_dined_with_us_before(monkeypatch):
    # Demo feedback: a returning guest's saved preferences are only reachable by
    # email, so ask for it BEFORE searching — too late once a shortlist is on screen.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    calls: list = []
    _patch_search(monkeypatch, calls)

    from langchain_core.messages import HumanMessage

    session = cc.ConciergeSession(member_id="sam")
    session.messages = [HumanMessage(content="I'd like an Italian table on Friday")]
    gate = _json.loads(cc._handle_recommend(session, {"cuisine": "italian"}))

    assert gate["status"] == "ask_if_returning"
    assert "set_confirmation_email" in gate["message"]
    assert calls == []  # nothing was searched behind the guest's back

    # Asking then answering itself doesn't count: the guest has to actually reply.
    assert _json.loads(
        cc._handle_recommend(session, {"cuisine": "italian"})
    )["status"] == "awaiting_answer"
    assert calls == []

    # Whatever they answer, the question is asked once and the search then runs.
    session.messages.append(HumanMessage(content="Nope, first time!"))
    out = _json.loads(cc._handle_recommend(session, {"cuisine": "italian"}))
    assert out["status"] == "ok"
    assert len(calls) == 1


def test_a_recognized_guest_is_never_asked_if_they_have_dined_before(monkeypatch):
    # We already have their file — asking would be the opposite of remembering.
    import json as _json

    import table_for_four.agent.concierge_chat as cc

    calls: list = []
    _patch_search(monkeypatch, calls)

    session = cc.ConciergeSession(
        member_id="g@x.com", profile={"email": "g@x.com", "name": "Sam"}
    )
    out = _json.loads(cc._handle_recommend(session, {"cuisine": "italian"}))

    assert out["status"] == "ok"
    assert len(calls) == 1


def test_giving_an_email_settles_identity_so_the_search_is_not_gated(monkeypatch):
    import chromadb

    import table_for_four.agent.concierge_chat as cc
    from table_for_four.agent import profile_memory as pm
    from langchain_core.messages import HumanMessage

    monkeypatch.setattr(pm, "_collection", pm.build_collection(chromadb.EphemeralClient()))

    session = cc.ConciergeSession(member_id="sam")
    session.messages = [HumanMessage(content="it's sam@x.com")]
    cc._handle_email(session, {"email": "sam@x.com"})

    assert session.asked_returning is True
