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


def test_book_is_idempotent_per_request(monkeypatch):
    # An identical request must not create a second reservation.
    import json as _json

    import agent.concierge_chat as cc

    calls = {"n": 0}

    def fake_run(request, **_kw):
        calls["n"] += 1
        return {
            "narrative": "Booked!",
            "booking": {"confirmation_id": f"TF4-000{calls['n']}"},
            "chosen": {"restaurant": {"name": "Osteria"}},
        }

    monkeypatch.setattr(cc, "run_concierge", fake_run)
    # Real remember() returns the merged profile (still carrying the email); model it.
    monkeypatch.setattr(cc.profile_memory, "remember", lambda *a, **k: {"email": "g@x.com"})

    session = cc.ConciergeSession(member_id="g@x.com", profile={"email": "g@x.com"})
    args = {"cuisine": "italian", "party_size": 4, "date": "Friday", "time": "7pm"}

    first = _json.loads(cc._handle_book(session, args))
    second = _json.loads(cc._handle_book(session, args))

    assert first["confirmation_id"] == "TF4-0001"
    assert second["status"] == "already_booked"
    assert second["confirmation_id"] == "TF4-0001"
    assert calls["n"] == 1  # pipeline ran exactly once


def test_book_is_gated_on_email(monkeypatch):
    # book_table must refuse until a confirmation email is on file — and must NOT
    # invoke the booking pipeline when it refuses.
    import json as _json

    import agent.concierge_chat as cc

    called = {"n": 0}
    monkeypatch.setattr(cc, "run_concierge", lambda *a, **k: called.__setitem__("n", called["n"] + 1) or {})

    session = cc.ConciergeSession(member_id="manish", profile={"name": "Manish"})
    out = _json.loads(cc._handle_book(session, {"cuisine": "italian", "party_size": 4}))

    assert out["status"] == "email_required"
    assert called["n"] == 0  # pipeline never ran


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
