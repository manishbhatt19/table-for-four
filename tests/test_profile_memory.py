"""Offline tests for the long-term member-profile store.

These drive an **ephemeral** Chroma collection (no persistent store, no network
beyond the one-time local embedding-model download Chroma caches) through the
pure functions that take a `Collection`, so no LLM or chat loop is involved.
"""

import chromadb
import pytest

from agent.profile_memory import (
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
    update_profile,
)


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
    # list field unions and dedupes; scalar fields overwrite/persist
    assert p["cuisines"] == ["italian", "japanese"]
    assert p["pronouns"] == "he/him"
    assert p["home_location"] == "Midtown"


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

    import agent.concierge_chat as cc
    from agent import profile_memory as pm
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


def _listed(session, place_id="p1", name="Osteria", perk_id=None):
    session.recommendations = {place_id: {"place_id": place_id, "name": name, "perk_id": perk_id}}


def test_book_is_idempotent_per_request(monkeypatch):
    # An identical booking must not create a second reservation.
    import json as _json

    import agent.concierge_chat as cc

    calls = {"n": 0}

    def fake_create(**_kw):
        calls["n"] += 1
        return {"booked": True, "confirmation_id": f"TF4-000{calls['n']}", "booking": {}}

    monkeypatch.setattr(cc, "create_booking", fake_create)
    monkeypatch.setattr(cc.profile_memory, "remember", lambda *a, **k: {"email": "g@x.com", "party_size": 4})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com", "party_size": 4})
    _listed(session)
    args = {"place_id": "p1", "date": "2026-08-07", "time": "19:00", "party_size": 4}

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

    import agent.concierge_chat as cc

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

    import agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    _listed(session)  # only p1 is listed
    out = _json.loads(cc._handle_book(session, {"place_id": "ghost", "date": "Friday", "time": "19:00"}))
    assert out["status"] == "unknown_restaurant"


def test_book_requires_a_party_size(monkeypatch):
    # book_table must refuse — never silently default — when no party size is known,
    # and must NOT hit the booking backend when it refuses.
    import json as _json

    import agent.concierge_chat as cc

    called = {"n": 0}
    monkeypatch.setattr(cc, "create_booking", lambda **k: called.__setitem__("n", called["n"] + 1) or {})

    # Email on file (passes the email gate) but no party size anywhere.
    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    _listed(session)
    out = _json.loads(cc._handle_book(session, {"place_id": "p1", "date": "2026-08-07", "time": "19:00"}))

    assert out["status"] == "need_party_size"
    assert called["n"] == 0  # booking backend never hit without a party size


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
    from mock_booking_api.app import available_slots
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

    import agent.concierge_chat as cc
    from agent import profile_memory as pm

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

    import agent.concierge_chat as cc

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

    import agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    out = _json.loads(cc._handle_cancel(session, {}))
    assert out["status"] == "need_confirmation_id"


def test_recommend_filters_cuisine_and_flags_perks():
    # Uses the real (offline) search + perks fixtures.
    import json as _json

    import agent.concierge_chat as cc

    session = cc.ConciergeSession(member_id="g")
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

    import agent.concierge_chat as cc

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

    session = cc.ConciergeSession(member_id="p")
    out = _json.loads(cc._handle_recommend(session, {"cuisine": "italian", "keywords": "italian dinner"}))

    perked = [r for r in out["recommendations"] if r["has_perk"]]
    assert 1 <= len(perked) <= 2
    assert all(r["perk_sample"] for r in perked)     # labeled as illustrative
    assert "perk_note" in out                         # guest-facing sample disclaimer
    assert out["restaurants_with_perks"]


def test_check_times_requires_a_listed_restaurant():
    import json as _json

    import agent.concierge_chat as cc

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
