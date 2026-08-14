"""The harness: declared units, enforced grants, and a provably free refactor.

Two kinds of test live here.

The **cost tests** exist because moving instructions into files is only worth doing
if it is free. The golden file is a literal proof that the system prompt did not
change by a single byte when it left `concierge_chat.py`, and the schema budget
stops the roster from quietly becoming a second place to write prompt.

The **grant tests** are the ones that make the roster more than a folder of
markdown. All of them pass by construction today — no code path calls
`create_booking` while acting as the Curator. The point is that they would now keep
passing: the boundary is enforced at the tool registry rather than remembered.

Everything here is offline and calls no model.
"""

import json
from pathlib import Path

import pytest

from table_for_four.agent import roster
from table_for_four.agent.tools import TOOLS

GOLDEN = Path(__file__).parent / "golden" / "dino_system_prompt.txt"

# What the model was handed before the harness existed, in characters of serialized
# JSON. The roster may source that text; it may not add to it.
SCHEMA_CHAR_BUDGET = 6624


# --- The cost invariant ------------------------------------------------------

def test_dinos_brief_is_byte_for_byte_what_it_always_was():
    # The risk in moving a 13k-character prompt out of a string literal is an
    # invisible edit — a re-wrapped line, a lost trailing space — that changes the
    # model's behaviour and can't be traced back. The golden file is the proof that
    # Phase 1 was a move and not an edit.
    assert roster.build_system_prompt() == GOLDEN.read_text(encoding="utf-8")


def test_the_tool_schemas_did_not_grow():
    # Sourcing tool descriptions from the roster makes the .md files model-facing
    # through text that was already being sent. If that ever stops being true, the
    # harness has started costing the guest tokens on every single turn.
    from table_for_four.agent.concierge_chat import TOOL_SCHEMAS

    assert len(json.dumps(TOOL_SCHEMAS, ensure_ascii=False)) <= SCHEMA_CHAR_BUDGET


def test_every_tool_the_model_sees_is_described_by_its_owning_unit():
    from table_for_four.agent.concierge_chat import TOOL_SCHEMAS

    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        assert schema["function"]["description"] == roster.tool_description(name)


# --- The roster is coherent --------------------------------------------------

def test_every_handler_belongs_to_exactly_one_unit():
    # A handler owned by nobody has no grant to check and no actor to audit; a
    # handler owned by two has no answer to "who did this?".
    from table_for_four.agent.concierge_chat import _HANDLERS

    declared = {h for unit in roster.UNITS.values() for h in unit.handlers}
    assert declared == set(_HANDLERS)


def test_a_units_never_list_is_genuinely_denied():
    # `never` is documentation, and documentation drifts. Hold it to the whitelist
    # so a capability quietly added to `tools` can't sit beside a promise not to use it.
    for unit in roster.UNITS.values():
        for capability in unit.never:
            assert not unit.grants(capability), f"{unit.name} both grants and forswears {capability}"


def test_dino_holds_no_capability_at_all():
    # The host talks; everything else is somebody's job. If Dino ever acquires a
    # grant, the whole front-of-house/back-of-house separation is decorative.
    assert roster.UNITS["dino"].tools == frozenset()
    assert roster.UNITS["dino"].handlers == ()


# --- The grants are enforced -------------------------------------------------

def test_the_curator_cannot_book_a_table():
    # Curator is the only unit that reaches the open web. It must never be the unit
    # that commits a guest to anything.
    with roster.acting_as("curator"):
        with pytest.raises(roster.NotGranted):
            TOOLS["create_booking"](place_id="p1")


def test_the_curator_cannot_write_to_the_member_book():
    from table_for_four.agent import profile_memory

    with roster.acting_as("curator"):
        with pytest.raises(roster.NotGranted):
            profile_memory.remember("g@x.com", {"cuisines": ["Italian"]})


def test_the_scout_cannot_write_a_preference():
    # What a guest browses is not evidence of what a guest wants remembered.
    from table_for_four.agent import profile_memory

    with roster.acting_as("scout"):
        with pytest.raises(roster.NotGranted):
            profile_memory.remember("g@x.com", {"home_location": "Brooklyn"})


def test_the_booker_cannot_search():
    # A booking may only name a restaurant the Scout already surfaced. Denying the
    # capability means it cannot conjure one to book even if the shortlist is empty.
    with roster.acting_as("booker"):
        with pytest.raises(roster.NotGranted):
            TOOLS["search_restaurants"](query="italian")


def test_the_booker_cannot_adopt_an_email_identity():
    # Who the guest *is* stays Steward's to establish, from the guest's own typing.
    from table_for_four.agent import profile_memory

    with roster.acting_as("booker"):
        with pytest.raises(roster.NotGranted):
            profile_memory.adopt_email("someone", "guest@example.com")


def test_the_steward_cannot_book_or_search():
    # Memory exists to offer, never to decide.
    with roster.acting_as("steward"):
        with pytest.raises(roster.NotGranted):
            TOOLS["search_restaurants"](query="italian")
        with pytest.raises(roster.NotGranted):
            TOOLS["create_booking"](place_id="p1")


def test_nothing_is_checked_outside_a_unit():
    # The harness constrains the units it declares; it is not a sandbox around the
    # process. A test or the perks eval calling a tool directly must still work —
    # claiming otherwise would be the decorative version of this.
    assert roster.acting_unit() is None
    out = TOOLS["search_restaurants"](query="italian")
    assert "results" in out


# --- Dispatch tags the acting unit -------------------------------------------

def test_dispatch_runs_each_handler_as_the_unit_that_owns_it(monkeypatch):
    import table_for_four.agent.concierge_chat as cc

    seen: dict[str, str | None] = {}

    def spy(name):
        def handler(_session, _args):
            seen[name] = roster.acting_unit()
            return "{}"
        return handler

    monkeypatch.setattr(cc, "_HANDLERS", {n: spy(n) for n in cc._HANDLERS})
    session = cc.ConciergeSession(member_id="g@x.com")
    for name in ("recommend_restaurants", "show_dining_highlights",
                 "set_confirmation_email", "book_table"):
        cc._dispatch(session, name, {})

    assert seen == {
        "recommend_restaurants": "scout",
        "show_dining_highlights": "curator",
        "set_confirmation_email": "steward",
        "book_table": "booker",
    }


def test_the_acting_unit_is_put_back_after_a_tool_call(monkeypatch):
    # Dino owns the turn; a handler borrows the floor and must hand it back, or the
    # next tool call in the same turn runs under the wrong grant.
    import table_for_four.agent.concierge_chat as cc

    monkeypatch.setattr(cc, "_HANDLERS", {"recall_guest_profile": lambda *_: "{}"})
    session = cc.ConciergeSession(member_id="g@x.com")
    with roster.acting_as("dino"):
        cc._dispatch(session, "recall_guest_profile", {})
        assert roster.acting_unit() == "dino"
    assert roster.acting_unit() is None


def test_a_grant_breach_is_not_swallowed_into_a_chat_message(monkeypatch):
    # _dispatch turns a failing tool into a polite string so the chat survives. A
    # unit reaching past its grant is a bug in us, not a tool having a bad day —
    # papering over it would hide exactly what the harness exists to catch.
    import table_for_four.agent.concierge_chat as cc

    def curator_tries_to_book(_session, _args):
        return TOOLS["create_booking"](place_id="p1")

    monkeypatch.setattr(cc, "_HANDLERS", {"show_dining_highlights": curator_tries_to_book})
    session = cc.ConciergeSession(member_id="g@x.com")
    with pytest.raises(roster.NotGranted):
        cc._dispatch(session, "show_dining_highlights", {})


def test_an_ordinary_tool_failure_still_comes_back_as_a_message(monkeypatch):
    import table_for_four.agent.concierge_chat as cc

    def broken(_session, _args):
        raise RuntimeError("the perks store fell over")

    monkeypatch.setattr(cc, "_HANDLERS", {"recall_guest_profile": broken})
    session = cc.ConciergeSession(member_id="g@x.com")
    assert "fell over" in cc._dispatch(session, "recall_guest_profile", {})


# --- The audit trail names an actor ------------------------------------------

def test_the_audit_line_says_who_acted():
    # The join into M4: an audit line that cannot name the actor is a diary.
    from table_for_four.agent.graph import run_concierge
    from table_for_four.mcp_servers.booking.backend.app import reset_store

    reset_store()
    final = run_concierge("Italian dinner for 2 on Friday", use_llm=False)

    assert final["actors"] == ["scout", "booker"]
    audit = next(line for line in final["log"] if line.startswith("[audit]"))
    assert json.loads(audit[len("[audit] "):])["actors"] == ["scout", "booker"]
