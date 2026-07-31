"""Ava — the interpersonal Table for Four concierge (conversational front-end).

A warm, human-facing layer that guides a guest through a full booking *journey*
rather than one-shot auto-booking:

    welcome -> understand intent -> gather missing details (incl. email)
      -> recommend restaurants (flagging which carry a perk)
      -> guest picks one -> show available times (mock data)
      -> book it (or refine and search again) -> share dining tips
      -> offer another booking -> close, remembering the guest for next time.

The guest stays in the loop at the choice points (which restaurant, which time).
The model orchestrates the journey through tools; it never invents a restaurant,
a time, an email, or a confirmation.

What it demonstrates:

* **Long-term memory (Chroma).** Name, email, cuisine/location preferences,
  party size, dietary needs, kids/high-chair info, interests, and past bookings
  are saved to `agent.profile_memory`, keyed by email, so a returning guest is
  recognized and their usuals can be reused.
* **Tool use.** The model reaches the world only through the tools below.
* **Deterministic guardrails.** Dining-only scope; booking gated on a real email;
  emails/restaurants/times must be ones actually surfaced (no fabrication);
  duplicate bookings are idempotent.

Run it:
    uv run python -m agent chat
    uv run python -m agent chat --name "Manish"

Requires an OpenAI (or OpenRouter) key in `.env`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import profile_memory, reasoning
from agent.config import get_chat_llm
from agent.tools import check_availability, create_booking, find_perks, search_restaurants

MAX_TOOL_HOPS = 6  # guard: bound tool-call chaining within a single guest turn
MAX_RECOMMENDATIONS = 4


# --- Persona & guardrails ----------------------------------------------------

SYSTEM_PROMPT = """\
You are **Ava**, a warm, gracious concierge for "Table for Four", a boutique
restaurant-reservation service. You guide each guest through booking a wonderful
table and make them feel genuinely looked after.

## Your one job (hard guardrail)
You ONLY help with dining: recommending restaurants and booking tables. If a guest
asks for anything outside dining/hospitality — coding, medical/legal/financial
advice, general knowledge, homework — warmly decline in one sentence and steer
back to the table. Never answer the off-topic question, even partially.

## How you talk
- Warm, personable, concise. Ask ONE or at most TWO things at a time.
- Always use the pronouns the guest gave you; NEVER guess pronouns from a name —
  ask politely and early if you don't know them.
- A light, friendly touch is welcome (a favorite cuisine memory, an occasion), but
  the table always comes first.

## The journey — follow this order
1. **Welcome** the guest warmly and introduce yourself. If they're a returning
   member, welcome them back by name and offer to reuse their usual preferences.
2. **Understand their intent** — what kind of outing is this?
3. **Gather what's missing** (only what wasn't already said): their email, the
   date/time, party size, location/area, cuisine, and any dietary needs or kids
   (with ages / high-chair needs). Save durable details with
   `remember_guest_details` as they arrive. Ask for the email as part of this,
   framed as where to send the confirmation.
4. **Recommend** — call `recommend_restaurants` and present a short shortlist.
   Clearly mention which one or two carry a **special perk** (offer), by name.
   Only ever mention restaurants the tool returned.
5. **Guest picks one** → call `check_availability_times` and tell them the open
   times for that restaurant and date. Only offer times the tool returned.
6. **Book** — once they choose an available time, call `book_table`. If there's no
   availability, or they want something different, gather the new detail and go
   back to step 4/5 (recommend or check times again) and offer an alternative.
   Never claim a booking is made until `book_table` returns a confirmation id.
7. **After booking**, share 2–3 brief, practical **dining tips** so they're
   prepared (e.g. arrive a few minutes early, mention the reservation name and any
   dietary need to the host, note the perk at the table). Keep tips general — do
   not invent specifics about the restaurant.
8. **Offer another** — ask whether they'd like to book another restaurant for a
   different day. If yes, return to step 3/4 for the new outing.
9. **Close** warmly when they're done.

## Tools
- `remember_guest_details` — save durable facts (pronouns, email is separate, see
  below; location, cuisines, party size, dietary, kids, interests). Save as you go.
- `recall_guest_profile` — check what you already know.
- `set_confirmation_email` — save the guest's email (the unique id for returning
  members and where the confirmation notionally goes). NEVER invent or assume an
  email; use exactly what the guest types. Ask for it before booking.
- `recommend_restaurants` — get a shortlist with perk flags. Call again with
  adjusted criteria if the guest wants different options.
- `check_availability_times` — get open times for a chosen restaurant + date.
- `book_table` — book a specific restaurant at a specific available time. Requires
  the guest's email to be on file first.

We don't actually send email in this demo, but always tell the guest the
confirmation will be sent to their address.
"""


# --- Tool schemas (OpenAI function format) -----------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember_guest_details",
            "description": (
                "Save or update durable facts about the guest to long-term memory. "
                "Pass only the fields you learned this turn; omit the rest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "pronouns": {"type": "string", "description": "e.g. 'she/her', 'they/them'."},
                    "home_location": {"type": "string", "description": "City/area the guest is in."},
                    "party_size": {"type": "integer", "description": "The guest's usual party size."},
                    "dining_atmosphere": {"type": "string", "description": "e.g. 'romantic', 'lively', 'family-friendly'."},
                    "dietary": {"type": "array", "items": {"type": "string"}},
                    "cuisines": {"type": "array", "items": {"type": "string"}},
                    "kids": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "age": {"type": "integer"},
                                "needs_high_chair": {"type": "boolean"},
                            },
                        },
                    },
                    "interests": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_guest_profile",
            "description": "Retrieve everything currently remembered about this guest.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_confirmation_email",
            "description": (
                "Save the guest's email — the unique identifier for returning members "
                "and where the confirmation notionally goes. Returns whether this "
                "email is a returning member and their saved preferences."
            ),
            "parameters": {
                "type": "object",
                "properties": {"email": {"type": "string"}},
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_restaurants",
            "description": (
                "Get a shortlist of restaurants for the guest's criteria, each flagged "
                "with whether it carries a special perk. Call again with adjusted "
                "criteria to refine."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {"type": "string"},
                    "location": {"type": "string", "description": "Neighborhood/area."},
                    "party_size": {"type": "integer"},
                    "date": {"type": "string", "description": "Day or date, e.g. 'Friday' or '2026-08-07'."},
                    "keywords": {"type": "string", "description": "Free-text vibe/occasion/dietary intent."},
                    "max_price_level": {"type": "integer", "description": "0=free .. 4=very expensive."},
                    "min_rating": {"type": "number"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_availability_times",
            "description": (
                "Get the open reservation times for a chosen restaurant on a date. "
                "The restaurant must be one from the latest recommendations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_id": {"type": "string", "description": "place_id from a recommendation."},
                    "date": {"type": "string"},
                    "party_size": {"type": "integer"},
                },
                "required": ["place_id", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_table",
            "description": (
                "Book a specific restaurant at a specific available time. The "
                "restaurant must be from the latest recommendations, the time must be "
                "one that check_availability_times returned, and the guest's email "
                "must already be saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_id": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string", "description": "HH:MM, from the available times."},
                    "party_size": {"type": "integer"},
                    "special_requests": {"type": "string"},
                },
                "required": ["place_id", "date", "time"],
            },
        },
    },
]


# --- Session -----------------------------------------------------------------

@dataclass
class ConciergeSession:
    """One guest's chat session: identity, profile, history, and journey state."""

    member_id: str
    profile: dict[str, Any] | None = None
    messages: list[Any] = field(default_factory=list)
    recommendations: dict[str, Any] = field(default_factory=dict)   # place_id -> rec
    availability: dict[str, Any] | None = None                      # last times shown
    bookings: dict[str, Any] = field(default_factory=dict)          # key -> result (idempotency)

    @property
    def display_name(self) -> str:
        return (self.profile or {}).get("name") or self.member_id


def _guest_typed(session: ConciergeSession, text: str) -> bool:
    """True if `text` actually appears in one of the guest's own messages.

    Stops the model inventing details (notably an email) it was never given.
    """
    needle = (text or "").strip().lower()
    if not needle:
        return False
    return any(
        isinstance(m, HumanMessage) and needle in (m.content or "").lower()
        for m in session.messages
    )


def _profile_context(profile: dict[str, Any] | None) -> str:
    if not profile:
        return "This is a NEW guest — no profile on file yet."
    known = {k: v for k, v in profile.items() if v and k not in ("member_id", "updated_at")}
    return (
        "This is a RETURNING guest. Here is what you already remember (don't re-ask "
        "what you already know; offer to reuse it):\n"
        + json.dumps(known, ensure_ascii=False, indent=2)
    )


# --- Tool dispatch -----------------------------------------------------------

def _handle_remember(session: ConciergeSession, args: dict[str, Any]) -> str:
    updates = {k: v for k, v in args.items() if v not in (None, "", [], {})}
    if not updates:
        return "Nothing to save."
    session.profile = profile_memory.remember(session.member_id, updates)
    return "Saved: " + ", ".join(sorted(updates.keys()))


def _handle_recall(session: ConciergeSession, _args: dict[str, Any]) -> str:
    profile = profile_memory.load(session.member_id)
    session.profile = profile
    if not profile:
        return "No profile on file for this guest yet."
    return json.dumps(profile, ensure_ascii=False)


def _handle_email(session: ConciergeSession, args: dict[str, Any]) -> str:
    email = (args.get("email") or "").strip()
    if not profile_memory.looks_like_email(email):
        return "That doesn't look like a valid email address — could you re-check it?"
    if not _guest_typed(session, email):
        return json.dumps({
            "status": "rejected",
            "message": (
                "That email was not provided by the guest. Never invent an email — "
                "ask the guest for it and use exactly what they type."
            ),
        }, ensure_ascii=False)
    key, profile, returning = profile_memory.adopt_email(session.member_id, email)
    session.member_id = key
    session.profile = profile
    payload: dict[str, Any] = {"status": "saved", "email": key, "returning_member": returning}
    if returning:
        payload["saved_preferences"] = {
            "name": profile.get("name"),
            "cuisines": profile.get("cuisines"),
            "home_location": profile.get("home_location"),
            "party_size": profile.get("party_size"),
            "dietary": profile.get("dietary"),
        }
        payload["note"] = "Returning member — welcome them back and offer to reuse these."
    return json.dumps(payload, ensure_ascii=False)


def _handle_recommend(session: ConciergeSession, args: dict[str, Any]) -> str:
    cuisine = args.get("cuisine")
    keywords = args.get("keywords") or cuisine or "restaurant"
    party_size = args.get("party_size")
    day = None
    if args.get("date"):
        try:
            _, day = reasoning.resolve_date(args["date"])
        except Exception:
            day = None

    search = search_restaurants(
        query=keywords,
        cuisine=cuisine,
        location=args.get("location"),
        max_price_level=args.get("max_price_level"),
        min_rating=args.get("min_rating"),
    )
    candidates = search.get("results", [])
    if not candidates:
        return json.dumps({
            "status": "no_matches",
            "message": "No restaurants matched. Ask the guest to relax a filter "
                       "(cuisine, area, price) and call recommend_restaurants again.",
        }, ensure_ascii=False)

    perks = find_perks(
        query=keywords,
        place_ids=[c["place_id"] for c in candidates],
        party_size=party_size,
        day=day,
    ).get("results", [])
    best_perk: dict[str, dict[str, Any]] = {}
    for p in perks:
        pid = p.get("place_id")
        if pid and p.get("similarity", 0) > best_perk.get(pid, {}).get("similarity", -1):
            best_perk[pid] = p

    session.recommendations = {}
    recs = []
    for c in candidates[:MAX_RECOMMENDATIONS]:
        perk = best_perk.get(c["place_id"])
        rec = {
            "place_id": c["place_id"],
            "name": c["name"],
            "cuisine": (c.get("primary_type") or "").replace("_restaurant", "") or None,
            "rating": c.get("rating"),
            "price_level": c.get("price_level"),
            "has_perk": bool(perk),
            "perk_title": (perk or {}).get("title"),
            "perk_id": (perk or {}).get("perk_id"),
        }
        recs.append(rec)
        session.recommendations[c["place_id"]] = rec

    return json.dumps({
        "status": "ok",
        "source": search.get("source"),
        "recommendations": recs,
        "restaurants_with_perks": [r["name"] for r in recs if r["has_perk"]],
    }, ensure_ascii=False)


def _handle_times(session: ConciergeSession, args: dict[str, Any]) -> str:
    place_id = args.get("place_id")
    rec = session.recommendations.get(place_id)
    if not rec:
        return json.dumps({
            "status": "unknown_restaurant",
            "message": "Only offer times for a restaurant from the latest "
                       "recommendations. Call recommend_restaurants first.",
        }, ensure_ascii=False)
    if not args.get("date"):
        return json.dumps({"status": "need_date", "message": "Ask the guest for a date."},
                          ensure_ascii=False)
    iso, _ = reasoning.resolve_date(args["date"])
    if iso < date.today().isoformat():
        return json.dumps({"status": "date_in_past",
                           "message": "That date is in the past. Re-confirm the date with the guest."},
                          ensure_ascii=False)
    party_size = args.get("party_size") or (session.profile or {}).get("party_size") or 2
    avail = check_availability(place_id, iso, party_size)
    slots = avail.get("available_slots", [])
    session.availability = {"place_id": place_id, "date": iso, "party_size": party_size, "slots": slots}
    if not slots:
        return json.dumps({
            "status": "no_availability", "restaurant": rec["name"], "date": iso,
            "message": "No tables free then. Offer another date/time or an alternate "
                       "restaurant from the recommendations.",
        }, ensure_ascii=False)
    return json.dumps({
        "status": "ok", "restaurant": rec["name"], "date": iso, "available_times": slots,
    }, ensure_ascii=False)


def _handle_book(session: ConciergeSession, args: dict[str, Any]) -> str:
    profile = session.profile or {}
    # Hard gate: never book without a confirmation email on file.
    if not profile.get("email"):
        return json.dumps({
            "status": "email_required",
            "message": "Collect the guest's email via set_confirmation_email before booking.",
        }, ensure_ascii=False)

    place_id = args.get("place_id")
    rec = session.recommendations.get(place_id)
    if not rec:
        return json.dumps({
            "status": "unknown_restaurant",
            "message": "Only book a restaurant from the latest recommendations.",
        }, ensure_ascii=False)
    if not args.get("date") or not args.get("time"):
        return json.dumps({"status": "need_details", "message": "Need both a date and a time."},
                          ensure_ascii=False)

    iso, _ = reasoning.resolve_date(args["date"])
    if iso < date.today().isoformat():
        return json.dumps({"status": "date_in_past",
                           "message": "That date is in the past. Re-confirm the date with the guest."},
                          ensure_ascii=False)
    time = args["time"]
    party_size = (
        args.get("party_size")
        or (session.availability or {}).get("party_size")
        or profile.get("party_size")
        or 2
    )

    # The time must be one we actually offered for this restaurant + date.
    pend = session.availability
    if pend and pend["place_id"] == place_id and pend["date"] == iso and pend["slots"]:
        if time not in pend["slots"]:
            return json.dumps({
                "status": "time_unavailable", "available_times": pend["slots"],
                "message": "That time isn't available; offer one of the available_times.",
            }, ensure_ascii=False)

    key = f"{place_id}|{iso}|{time}"
    if key in session.bookings:  # idempotency
        return json.dumps({**session.bookings[key], "status": "already_booked"}, ensure_ascii=False)

    extras: list[str] = []
    if profile.get("dietary"):
        extras.append(", ".join(profile["dietary"]))
    if args.get("special_requests"):
        extras.append(str(args["special_requests"]))
    high_chairs = sum(1 for k in (profile.get("kids") or []) if k.get("needs_high_chair"))
    if high_chairs:
        extras.append(f"{high_chairs} high chair(s)")
    special = "; ".join(extras) or None

    booking = create_booking(
        place_id=place_id,
        restaurant_name=rec["name"],
        date=iso,
        time=time,
        party_size=party_size,
        guest_name=session.display_name,
        perk_id=rec.get("perk_id"),
        special_requests=special,
    )
    if not booking.get("booked"):
        return json.dumps({
            "status": "failed", "error": booking.get("error"),
            "message": "Booking failed; offer an alternate time or restaurant.",
        }, ensure_ascii=False)

    result = {
        "status": "booked",
        "restaurant": rec["name"],
        "date": iso,
        "time": time,
        "party_size": party_size,
        "confirmation_id": booking.get("confirmation_id"),
        "perk_applied": rec.get("perk_title"),
        "confirmation_sent_to": profile.get("email"),
    }
    session.bookings[key] = result
    # Remember the booking + reusable preferences for next time.
    session.profile = profile_memory.remember(session.member_id, {
        "party_size": party_size,
        "past_bookings": [{
            "restaurant": rec["name"], "confirmation_id": booking.get("confirmation_id"),
            "date": iso, "time": time, "party_size": party_size,
        }],
    })
    return json.dumps(result, ensure_ascii=False)


_HANDLERS = {
    "remember_guest_details": _handle_remember,
    "recall_guest_profile": _handle_recall,
    "set_confirmation_email": _handle_email,
    "recommend_restaurants": _handle_recommend,
    "check_availability_times": _handle_times,
    "book_table": _handle_book,
}


def _dispatch(session: ConciergeSession, name: str, args: dict[str, Any]) -> str:
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        return handler(session, args)
    except Exception as exc:  # keep the chat alive if a tool trips
        return f"Tool '{name}' failed: {exc}"


# --- Conversation turn -------------------------------------------------------

def _run_turn(session: ConciergeSession, llm: Any) -> str:
    """Invoke the model, resolve any tool calls, and return the reply text."""
    for _ in range(MAX_TOOL_HOPS):
        response: AIMessage = llm.invoke(session.messages)
        session.messages.append(response)
        if not response.tool_calls:
            return (response.content or "").strip()
        for call in response.tool_calls:
            result = _dispatch(session, call["name"], call.get("args") or {})
            session.messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
    session.messages.append(
        HumanMessage(content="(Please respond to me directly now, no more tools.)")
    )
    final: AIMessage = llm.invoke(session.messages)
    session.messages.append(final)
    return (final.content or "").strip()


def start_session(member_id: str) -> ConciergeSession:
    """Build a session, load any saved profile, and seed the system prompt."""
    profile = profile_memory.load(member_id)
    session = ConciergeSession(member_id=profile_memory.resolve_key(member_id), profile=profile)
    # Ground the model in the real date so it never invents a calendar date — it
    # should pass the guest's own wording ("Friday") and let the tools resolve it.
    today = date.today()
    context = (
        f"Today is {today:%A}, {today.isoformat()}. When calling a tool that takes a "
        "date, pass the guest's own words (e.g. 'Friday', 'next Saturday'); the "
        "system resolves them against today. NEVER invent a specific calendar date.\n\n"
        "## This guest\n" + _profile_context(profile)
    )
    session.messages = [SystemMessage(content=SYSTEM_PROMPT + "\n\n## Context\n" + context)]
    return session


# --- REPL --------------------------------------------------------------------

def run_chat(name: str | None = None) -> None:
    """Interactive terminal chat with the concierge."""
    llm = get_chat_llm()
    if llm is None:
        print(
            "The interpersonal concierge needs an OpenAI (or OpenRouter) key.\n"
            "Add OPENAI_API_KEY to your .env, then run `uv run python -m agent chat` "
            "again.\n(The one-shot booking demo still runs offline: `uv run python -m "
            "agent`.)"
        )
        return

    llm = llm.bind_tools(TOOL_SCHEMAS)

    if not name:
        try:
            name = input("Welcome to Table for Four. Your name or handle: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
    if not name:
        name = "guest"

    session = start_session(name)
    session.messages.append(
        HumanMessage(content=f"(System: the guest '{name}' just connected. Greet them.)")
    )
    print(f"\nAva: {_run_turn(session, llm)}\n")

    print("(Type 'quit' or 'exit' to leave.)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAva: Lovely chatting — enjoy your meal!")
            return
        if user.lower() in {"quit", "exit", "bye"}:
            print("Ava: Lovely chatting — enjoy your meal!")
            return
        if not user:
            continue
        session.messages.append(HumanMessage(content=user))
        print(f"\nAva: {_run_turn(session, llm)}\n")
