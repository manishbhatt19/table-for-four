"""Ava — the interpersonal Table for Four concierge (conversational front-end).

This is the warm, human-facing layer that sits IN FRONT OF the one-shot booking
orchestrator (`agent.graph.run_concierge`). Where the orchestrator is a
transactional state machine, this is a chat: it welcomes the guest, learns who
they are, and remembers them across sessions.

What it demonstrates:

* **Long-term memory (Chroma).** Everything the guest shares — pronouns, where
  they're writing from, dining atmosphere, dietary needs, favorite cuisines,
  whether kids are joining (ages / high chairs), and fun interests — is saved to
  `agent.profile_memory` so a returning guest is greeted by name with their
  preferences pre-loaded.
* **Tool use.** The model reaches the world only through three tools:
  `remember_guest_details` (write memory), `recall_guest_profile` (read memory),
  and `book_table` (hand off to the booking pipeline). It never invents a
  reservation.
* **Guardrails.** A tightly-scoped system persona keeps Ava on dining/hospitality
  only; off-topic requests are warmly declined and steered back to the table.

Run it:
    uv run python -m agent chat
    uv run python -m agent chat --name "Manish"

Requires an OpenAI (or OpenRouter) key in `.env` — open-ended conversation has no
offline heuristic fallback, unlike the orchestrator's parse/narrate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from agent import profile_memory
from agent.config import get_chat_llm
from agent.graph import run_concierge

MAX_TOOL_HOPS = 5  # guard: bound tool-call chaining within a single guest turn


# --- Persona & guardrails ----------------------------------------------------

SYSTEM_PROMPT = """\
You are **Ava**, a warm, gracious concierge for "Table for Four", a boutique
restaurant-reservation service. You help guests find and book a great table, and
you make them feel genuinely looked after along the way.

## Your one job (hard guardrail)
You ONLY help with dining: finding restaurants, tailoring the choice to the
guest, and booking a table. If a guest asks for anything outside dining and
hospitality — coding, medical/legal/financial advice, general knowledge,
homework, anything off-topic — warmly decline in one sentence and steer back to
the table. Never answer the off-topic question, even partially. For example:
"That's a bit outside my table — but I'd love to help you find somewhere lovely
to eat. What are you in the mood for?"

## How you talk
- Warm, personable, concise. One or two short paragraphs, never a wall of text.
- Ask ONE or at most TWO things at a time — this is a conversation, not a form.
- Always address the guest using the pronouns they gave you. NEVER guess pronouns
  from a name. If you don't know them yet, ask politely and early:
  "And may I ask which pronouns you'd like me to use?"
- Weave in the occasional light, friendly question — a favorite sport, team, or
  hobby — when it feels natural, and remember the answer. Keep it brief; the
  table always comes first.

## What to learn (and remember) about a guest
Gather these naturally over the conversation and save each detail as it comes up:
pronouns; where they're writing from (their location); the dining atmosphere they
like (romantic, lively, quiet, family-friendly…); dietary needs; favorite
cuisines; whether children are joining and their ages, and whether any need a
high chair; and any fun interests. You do NOT need all of it before booking —
gather what matters for a good reservation, save the rest as it surfaces.

## The email step (always the LAST question before booking)
Once the guest has chosen and is ready to book, ask for their email as the final
question, framed warmly as where to send the confirmation — e.g. "Wonderful! And
what's the best email for me to send your confirmation to?" Save it with
`set_confirmation_email`. This email is how we recognize returning guests. If the
tool reports a returning member, warmly acknowledge it. (We don't actually send
mail in this demo — but always tell the guest the confirmation will be sent to
that address.) NEVER invent, assume, or guess an email — use exactly what the
guest types; if you don't have it yet, ask for it. Do not claim a booking is
confirmed until `book_table` returns a confirmation id.

## Using your tools
- The moment a guest shares a durable detail (pronouns, location, a dietary need,
  a cuisine they love, kids/high-chair info, an interest), call
  `remember_guest_details` to save it. Save as you go — don't wait for the end.
- Call `recall_guest_profile` if you're unsure what you already know.
- Collect the email with `set_confirmation_email` BEFORE booking (see above).
- When you have enough to book (at minimum a cuisine or vibe, party size, and a
  day/time), the guest has agreed, AND you have their email, call `book_table`.
  Fold in their saved dietary needs and any high-chair requirement. When it
  succeeds, confirm the details and tell the guest their confirmation is on its
  way to the email they gave.
- Never fabricate a restaurant, availability, or a confirmation number — those
  come only from `book_table`.

Begin by greeting the guest. If they are a returning member, welcome them back by
name and reference something you remember; if they're new, introduce yourself
briefly and make them feel welcome.
"""


# --- Tool schemas (OpenAI function format) -----------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember_guest_details",
            "description": (
                "Save or update durable facts about the guest to long-term memory. "
                "Call whenever the guest shares a lasting preference or detail. "
                "Pass only the fields you learned this turn; omit the rest."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "The guest's name, if given."},
                    "pronouns": {
                        "type": "string",
                        "description": "e.g. 'she/her', 'he/him', 'they/them'.",
                    },
                    "home_location": {
                        "type": "string",
                        "description": "Where the guest is writing from (city/area).",
                    },
                    "dining_atmosphere": {
                        "type": "string",
                        "description": "Preferred vibe, e.g. 'romantic', 'lively', 'quiet', 'family-friendly'.",
                    },
                    "dietary": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Dietary needs, e.g. ['gluten-free', 'vegetarian'].",
                    },
                    "cuisines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Favorite cuisines, e.g. ['italian', 'japanese'].",
                    },
                    "kids": {
                        "type": "array",
                        "description": "Children joining the booking.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "age": {"type": "integer"},
                                "needs_high_chair": {"type": "boolean"},
                            },
                        },
                    },
                    "interests": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Fun facts/interests, e.g. ['soccer - loves Arsenal', 'hiking'].",
                    },
                    "notes": {"type": "string", "description": "Any other useful note."},
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
                "Save the guest's email — the unique identifier used to recognize "
                "returning members and (notionally) send their confirmation. Ask for "
                "this as the LAST step before booking, framed as where to send the "
                "confirmation. Returns whether this email is a returning member."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "The guest's email address."},
                },
                "required": ["email"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_table",
            "description": (
                "Book a table via the reservation pipeline. Call only after the "
                "guest's email is captured (set_confirmation_email) and they have "
                "agreed to a specific booking. Returns a confirmation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cuisine": {"type": "string"},
                    "location": {"type": "string", "description": "Neighborhood/area for the table."},
                    "party_size": {"type": "integer"},
                    "date": {"type": "string", "description": "Day or date, e.g. 'Friday' or '2026-08-07'."},
                    "time": {"type": "string", "description": "Preferred time, e.g. '7pm' or '19:00'."},
                    "atmosphere": {"type": "string", "description": "Desired vibe, if relevant."},
                    "special_requests": {
                        "type": "string",
                        "description": "Dietary needs, high-chair requests, occasion, etc.",
                    },
                },
                "required": ["party_size"],
            },
        },
    },
]


# --- Session -----------------------------------------------------------------

@dataclass
class ConciergeSession:
    """One guest's chat session: identity, loaded profile, and running history."""

    member_id: str
    profile: dict[str, Any] | None = None
    messages: list[Any] = field(default_factory=list)
    bookings: dict[str, Any] = field(default_factory=dict)  # request -> result (idempotency)

    @property
    def display_name(self) -> str:
        return (self.profile or {}).get("name") or self.member_id


def _guest_typed(session: ConciergeSession, text: str) -> bool:
    """True if `text` actually appears in one of the guest's own messages.

    Used to stop the model inventing details (notably an email) it was never
    given — the value must be traceable to something the guest really said.
    """
    needle = (text or "").strip().lower()
    if not needle:
        return False
    return any(
        isinstance(m, HumanMessage) and needle in (m.content or "").lower()
        for m in session.messages
    )


def _profile_context(profile: dict[str, Any] | None) -> str:
    """A compact briefing appended to the system prompt about this guest."""
    if not profile:
        return "This is a NEW guest — no profile on file yet."
    known = {k: v for k, v in profile.items() if v and k not in ("member_id", "updated_at")}
    return (
        "This is a RETURNING guest. Here is what you already remember (do not "
        "re-ask what you already know; confirm lightly if useful):\n"
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
        # The model must not invent an email to satisfy the booking gate.
        return json.dumps({
            "status": "rejected",
            "message": (
                "That email was not provided by the guest. Never invent an email — "
                "ask the guest for it and use exactly what they type."
            ),
        }, ensure_ascii=False)
    # Email is the unique key: rekey the session onto it, merging any existing
    # profile for that address (which is how returning members are recognized).
    key, profile, returning = profile_memory.adopt_email(session.member_id, email)
    session.member_id = key
    session.profile = profile
    return json.dumps({
        "status": "saved",
        "email": key,
        "returning_member": returning,
        "note": (
            "This email matches an existing member — welcome them back and feel free "
            "to reference their saved details."
            if returning else
            "New member saved under this email."
        ),
    }, ensure_ascii=False)


def _handle_book(session: ConciergeSession, args: dict[str, Any]) -> str:
    profile = session.profile or {}
    # Hard gate: never book (an irreversible action) without a confirmation email
    # on file. This enforces the "email is the last question" rule deterministically,
    # rather than trusting the model to follow the instruction — and it prevents the
    # model from inventing an email in its confirmation text.
    if not profile.get("email"):
        return json.dumps({
            "status": "email_required",
            "message": (
                "Cannot book yet: no confirmation email on file. Ask the guest for "
                "their email and call set_confirmation_email FIRST, then book."
            ),
        }, ensure_ascii=False)
    dietary = profile.get("dietary") or []
    # Assemble a natural-language request so the existing pipeline can parse it,
    # folding in remembered dietary needs and any high-chair requirement.
    bits: list[str] = []
    if args.get("cuisine"):
        bits.append(str(args["cuisine"]))
    bits.append("restaurant")
    if args.get("location"):
        bits.append(f"near {args['location']}")
    if args.get("party_size"):
        bits.append(f"for {args['party_size']} people")
    if args.get("date"):
        bits.append(f"on {args['date']}")
    if args.get("time"):
        bits.append(f"at {args['time']}")
    if args.get("atmosphere"):
        bits.append(f"({args['atmosphere']} atmosphere)")

    extras: list[str] = []
    if dietary:
        extras.append(", ".join(dietary))
    if args.get("special_requests"):
        extras.append(str(args["special_requests"]))
    high_chairs = sum(1 for k in (profile.get("kids") or []) if k.get("needs_high_chair"))
    if high_chairs:
        extras.append(f"{high_chairs} high chair(s) needed")
    if extras:
        bits.append("— " + "; ".join(extras))

    request = " ".join(bits)
    # Idempotency: an identical request in this session returns the existing
    # confirmation instead of creating a duplicate reservation.
    if request in session.bookings:
        return json.dumps({"status": "already_booked", **session.bookings[request]},
                          ensure_ascii=False)

    final = run_concierge(
        request,
        guest_name=session.display_name,
        thread_id=session.member_id,
    )
    narrative = final.get("narrative")
    booking = final.get("booking") or {}
    if booking.get("confirmation_id"):
        # Remember the booking so it shows up as history next time.
        session.profile = profile_memory.remember(
            session.member_id,
            {"past_bookings": [{
                "restaurant": (final.get("chosen") or {}).get("restaurant", {}).get("name"),
                "confirmation_id": booking["confirmation_id"],
                "request": request,
            }]},
        )
    result = {
        "request_sent": request,
        "outcome": narrative or "No table could be booked for that request.",
        "confirmation_id": booking.get("confirmation_id"),
        # Text-only in this demo: tell the guest their confirmation goes here.
        "confirmation_sent_to": profile.get("email"),
    }
    if booking.get("confirmation_id"):
        session.bookings[request] = result  # remember for idempotency
    return json.dumps(result, ensure_ascii=False)


_HANDLERS = {
    "remember_guest_details": _handle_remember,
    "recall_guest_profile": _handle_recall,
    "set_confirmation_email": _handle_email,
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
            session.messages.append(
                ToolMessage(content=result, tool_call_id=call["id"])
            )
    # Hit the hop guard — ask the model to wrap up in plain words.
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
    session.messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n\n## This guest\n" + _profile_context(profile))
    ]
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

    # Let Ava greet first (she uses the loaded profile to welcome-back or onboard).
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
