"""Dino — the interpersonal Table for Four concierge (conversational front-end).

A warm, human-facing layer that guides a guest through a full booking *journey*
rather than one-shot auto-booking:

    welcome -> understand intent -> gather missing details (incl. email)
      -> recommend restaurants (flagging which carry a perk)
      -> guest picks one -> show available times (mock data)
      -> book it (or refine and search again) -> share dining tips + what to order
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
* **Live web retrieval (Tavily).** `show_dining_highlights` fetches cited menu
  highlights and photos for a restaurant the guest is actually considering —
  scoped to that restaurant, attributed to its source, and rendered in the UI
  rather than pasted into the reply.
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
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent import menu_card, profile_memory, reasoning
from agent.config import get_chat_llm
from agent.tools import (
    cancel_booking,
    check_availability,
    create_booking,
    find_perks,
    lookup_dining_highlights,
    search_restaurants,
)

MAX_TOOL_HOPS = 6  # guard: bound tool-call chaining within a single guest turn
MAX_RECOMMENDATIONS = 4
CANCELLATION_WINDOW_HOURS = 24  # cancel allowed only more than this far ahead


# --- Persona & guardrails ----------------------------------------------------

SYSTEM_PROMPT = """\
You are **Dino**, the warm, upbeat concierge for "Table for Four", a boutique
restaurant-reservation service — *Dino helps you dine!* You genuinely love good
food and looking after people, and you guide each guest to a wonderful table while
making the whole thing feel easy and fun.

## Your one job (hard guardrail)
You ONLY help with dining: recommending restaurants and booking tables. If a guest
asks for anything outside dining/hospitality — coding, medical/legal/financial
advice, general knowledge, homework — warmly decline in one sentence and steer
back to the table. Never answer the off-topic question, even partially.

## How you talk
- Talk like a friendly human host, not a form or a robot. Warm, natural, with a
  light, playful charm — a little personality goes a long way.
- Use the guest's name once you know it, and react with real enthusiasm to their
  plans (a birthday! a first date! friends in town!).
- Keep it easy: ask ONE or at most TWO things at a time, and never sound clipped or
  scripted. Vary how you phrase things; sound like you mean it.
- Always use the pronouns the guest gave you; NEVER guess pronouns from a name —
  ask politely and early if you don't know them.
- A light, friendly touch is welcome (a favorite cuisine, the occasion), but the
  table always comes first.

## The journey — follow this order
1. **Welcome** the guest warmly and introduce yourself. If the context says this is
   a RETURNING guest (a profile is on file), welcome them back **by name** and
   reference something you remember (a favorite cuisine or their last booking),
   then offer to reuse their usual preferences. A guest is also recognized the
   moment they give a known **email** (see `set_confirmation_email`) — the instant
   that happens, pivot to a warm welcome-back even if you already greeted them.
   Welcoming them back is *not* permission to start booking their usual: go to 2b.
2. **Understand their intent** — what kind of outing is this?
2b. **Ask how they'd like to choose — never assume.** Remembering a guest's usuals
   is for *offering*, not for deciding. Before you search, put the choice to them
   in one friendly question with three ways in:
     - **their usuals** — "shall I look at Italian again, like your last few?"
     - **by area** — "or shall we just find something good near you?"
     - **something new** — "or are you in the mood to try a cuisine you haven't yet?"
   Name their actual saved cuisines when you offer the first option, and wait for
   an answer before searching. If they pick *something new*, deliberately search a
   cuisine that is NOT in their saved list and say that's what you're doing. This
   costs one extra exchange and is worth it: guessing wrong wastes far more of
   their time, and a guest who feels asked rather than assumed about comes back.
   Skip this only if the guest has already told you what they want in this session.
3. **Gather what's missing** (only what wasn't already said): their email, the
   date/time, **party size (how many people) — this is required**, location/area,
   cuisine, and any dietary needs or kids (with ages / high-chair needs). Save
   durable details with `remember_guest_details` as they arrive. Do NOT assume a party
   size — if the guest hasn't told you how many people, ask before you book.
   **Never ask for something you already have.** Every tool result carries a
   `known_so_far` block; treat it as the truth about this session. In particular,
   if `known_so_far.email_on_file` is set, the email is DONE — do not ask for it
   again at any point, including at booking time. Confirm it if you must ("I'll
   send the confirmation to sam@example.com — still the best address?"), but never
   re-request it. Only ask for the email when it is genuinely absent.
4. **Recommend** — call `recommend_restaurants` and present a short shortlist.
   Clearly mention which one or two carry a **special perk** (offer), by name. If
   the result includes a `perk_note` saying the perks are samples, present those as
   a "sample partner offer" so the guest knows it's illustrative. Only ever mention
   restaurants the tool returned.
5. **Guest picks one** → call `check_availability_times` and tell them the open
   times for that restaurant and date. Only offer times the tool returned.
6. **Book** — once they choose an available time, **read the details back and get a
   yes first**: "Booking [restaurant], [date] at [time] for [party size] — shall I
   confirm?" You MUST have the **date**, the **time**, and the **party size** — if
   any is missing, ask; never guess or default the party size. Then call
   `book_table` with the **exact time the guest chose** — never substitute a
   different open slot. (The details you've gathered — date, time, party size — are
   tracked as working memory and echoed back in the tool results; use them, don't
   drift.) If there's no availability, or they want something different, gather the
   new detail and go back to step 4/5 and offer an alternative. Never claim a
   booking is made until `book_table` returns a confirmation id.
7. **After booking**, call `show_dining_highlights` once for the restaurant you
   just booked, then share 2–3 brief, practical **dining tips** so they're
   prepared (e.g. arrive a few minutes early, mention the reservation name and any
   dietary need to the host, note the perk at the table) **plus one "what to
   order" line drawn from the highlights**. Any specific claim about the food must
   come from that tool result — never invent a dish, and say where it came from
   ("diners keep mentioning…", "their site lists…").
8. **Offer another** — ask whether they'd like to book another restaurant for a
   different day. If yes, return to step 3/4 for the new outing.
9. **Cancellations** — if a guest wants to cancel, find the reservation's
   confirmation id (ask, or use `recall_guest_profile`) and call
   `cancel_reservation`. A booking can only be cancelled **more than 24 hours**
   before its time. If the tool returns `too_late`, do NOT say it was cancelled:
   apologize briefly, explain the 24-hour policy, and give the guest the
   restaurant's **phone and website** (from the tool result) to cancel directly.
10. **Close** warmly when they're done.

## Tools
- `remember_guest_details` — save durable facts (pronouns, email is separate, see
  below; location, cuisines, party size, dietary, kids, interests). Save as you go.
- `recall_guest_profile` — check what you already know.
- `set_confirmation_email` — save the guest's email (the unique id for returning
  members and where the confirmation notionally goes). NEVER invent or assume an
  email; use exactly what the guest types. Ask for it before booking. If it returns
  `returning_member: true`, immediately welcome them back by name and mention their
  `last_booking`/saved cuisines before continuing.
- `recommend_restaurants` — get a shortlist with perk flags. Call again with
  adjusted criteria if the guest wants different options.
- `check_availability_times` — get open times for a chosen restaurant + date.
- `book_table` — book a specific restaurant at a specific available time. Requires
  the guest's email to be on file first.
- `cancel_reservation` — cancel a booking by its confirmation id. The 24-hour
  policy is enforced by the system; honor a `too_late` result exactly as described
  in step 9 (never claim a cancellation the tool didn't confirm).
- `show_dining_highlights` — look up what a restaurant is known for (menu
  highlights, signature dishes) with photos, from the public web. Call it whenever
  a guest asks what's good there, what a dish or the room looks like, or asks to
  see pictures — and once automatically after a booking (step 7). It only works
  for restaurants you have already recommended or booked; if it returns
  `unknown_restaurant`, recommend first rather than guessing.
  **Three rules when you use it:** (a) only mention dishes that appear in the
  result; (b) attribute them ("diners keep mentioning…"), never as a promise, since
  menus change; (c) the photos are shown to the guest automatically in the app —
  say something like "I've popped a couple of photos below" rather than pasting
  image links into your reply.

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
                "one that check_availability_times returned, the party size must be "
                "one the guest actually gave you (never guessed), and the guest's "
                "email must already be saved."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_id": {"type": "string"},
                    "date": {"type": "string"},
                    "time": {"type": "string", "description": "HH:MM, from the available times."},
                    "party_size": {"type": "integer", "description": "How many people — required; ask the guest, do not assume."},
                    "special_requests": {"type": "string"},
                },
                "required": ["place_id", "date", "time", "party_size"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reservation",
            "description": (
                "Cancel an existing reservation by its confirmation id. The backend "
                "enforces the 24-hour policy: if the booking is less than 24 hours "
                "away it returns status 'too_late' with the restaurant's phone and "
                "website — relay those and do NOT claim it was cancelled. Look up the "
                "confirmation id via recall_guest_profile if you don't have it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "confirmation_id": {"type": "string", "description": "e.g. 'TF4-0001'."},
                    "reason": {"type": "string", "description": "Optional reason to record."},
                },
                "required": ["confirmation_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_dining_highlights",
            "description": (
                "Look up what a restaurant is known for — menu highlights, signature "
                "dishes — with photos, from the public web. Use it when the guest asks "
                "what's good there, what a dish looks like, or wants to see pictures, "
                "and once after a booking to add a 'what to order' note. Works only "
                "for restaurants already recommended or booked in this conversation. "
                "Photos are rendered to the guest automatically; don't paste image "
                "links. Only repeat dishes that appear in the result, and attribute "
                "them rather than promising them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "place_id": {
                        "type": "string",
                        "description": "Id of a recommended or booked restaurant (preferred).",
                    },
                    "restaurant_name": {
                        "type": "string",
                        "description": "Its name, if you don't have the place_id.",
                    },
                    "focus": {
                        "type": "string",
                        "description": (
                            "What the guest asked about, in their terms — this steers "
                            "the photo search, so be specific about what they want to "
                            "SEE. Use 'what the dining room looks like' / 'the "
                            "interior' / 'the atmosphere' for pictures of the place "
                            "itself, and 'signature dishes' / 'what to order' for food."
                        ),
                    },
                },
                "required": [],
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
    pending: dict[str, Any] = field(default_factory=dict)           # outing being planned
    media: list[dict[str, Any]] = field(default_factory=list)       # web highlights for the UI

    @property
    def display_name(self) -> str:
        return (self.profile or {}).get("name") or self.member_id


# Times the guest may type: "7pm", "7:30 pm", "19:00". A bare number ("7", "4
# people") is deliberately NOT treated as a time — too ambiguous to enforce on.
_TIME_TOKEN = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)?", re.I)


def _parse_time_tokens(text: str) -> set[str]:
    """Extract explicit clock times from text as `HH:MM` (24h).

    Only tokens with am/pm or a colon count. For a colon time without am/pm (e.g.
    "7:30"), both the 24h and the PM reading are emitted so the caller can keep
    whichever is a real slot.
    """
    out: set[str] = set()
    for m in _TIME_TOKEN.finditer(text or ""):
        hour = int(m.group(1))
        minute = int(m.group(2)) if m.group(2) else 0
        has_colon = m.group(2) is not None
        ampm = (m.group(3) or "").lower().replace(".", "")
        if not ampm and not has_colon:
            continue  # bare number — ambiguous, skip
        if minute > 59:
            continue
        if ampm:
            h = hour % 12 + (12 if ampm == "pm" else 0)
            if 0 <= h <= 23:
                out.add(f"{h:02d}:{minute:02d}")
        else:  # colon, no am/pm: keep both readings, slot-matching disambiguates
            if 0 <= hour <= 23:
                out.add(f"{hour:02d}:{minute:02d}")
            if hour < 12:
                out.add(f"{hour + 12:02d}:{minute:02d}")
    return out


def _requested_times(session: ConciergeSession) -> set[str]:
    """Clock times the guest asked for in their last few messages."""
    humans = [m for m in session.messages if isinstance(m, HumanMessage)]
    out: set[str] = set()
    for m in humans[-4:]:
        out |= _parse_time_tokens(m.content or "")
    return out


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
        "This is a RETURNING guest — open by welcoming them back BY NAME and "
        "referencing something you remember (a favorite cuisine or their last "
        "booking). Don't re-ask what you already know; offer to reuse it:\n"
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
        past = profile.get("past_bookings") or []
        payload["saved_preferences"] = {
            "name": profile.get("name"),
            "cuisines": profile.get("cuisines"),
            "home_location": profile.get("home_location"),
            "party_size": profile.get("party_size"),
            "dietary": profile.get("dietary"),
        }
        if past:
            last = past[-1]
            payload["last_booking"] = {
                "restaurant": last.get("restaurant"),
                "date": last.get("date"),
                "status": last.get("status"),
            }
        payload["note"] = (
            "RETURNING member recognized by email. Right now, warmly welcome them "
            "back BY NAME, mention you remember them (a favorite cuisine or their "
            "last booking), and offer to reuse their saved details — even if you "
            "already greeted them generically."
        )
    return json.dumps(payload, ensure_ascii=False)


def _handle_recommend(session: ConciergeSession, args: dict[str, Any]) -> str:
    cuisine = args.get("cuisine")
    keywords = args.get("keywords") or cuisine or "restaurant"
    party_size = args.get("party_size")
    # Working memory: remember the outing's party size / date as they're mentioned.
    if party_size:
        session.pending["party_size"] = party_size
    if args.get("date"):
        session.pending["date"] = args["date"]
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
    # An over-specific keyword string can zero out live results; retry once with a
    # minimal query so we keep the requested cuisine instead of dropping it.
    if not candidates and (cuisine or args.get("location")):
        search = search_restaurants(
            query=cuisine or "restaurant",
            cuisine=cuisine,
            location=args.get("location"),
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
            "address": c.get("address"),
            "phone": c.get("phone"),
            "website": c.get("website"),
            "cuisine": (c.get("primary_type") or "").replace("_restaurant", "") or None,
            "rating": c.get("rating"),
            "price_level": c.get("price_level"),
            "has_perk": bool(perk),
            "perk_sample": False,
            "perk_title": (perk or {}).get("title"),
            "perk_id": (perk or {}).get("perk_id"),
        }
        recs.append(rec)
        session.recommendations[c["place_id"]] = rec

    # Live restaurants have real Google ids that our synthetic (fixture-keyed) perks
    # can't match, so nothing gets flagged. In that case attach a cuisine-matched
    # SAMPLE offer to the top 1-2 recommendations — clearly labeled illustrative.
    if not any(r["has_perk"] for r in recs):
        sample = find_perks(query=keywords, place_ids=None, party_size=party_size, day=day).get("results", [])
        for rec, perk in zip(recs[:2], sample[:2]):
            rec["has_perk"] = True
            rec["perk_sample"] = True
            rec["perk_title"] = perk.get("title")
            rec["perk_id"] = perk.get("perk_id")

    perked = [r["name"] for r in recs if r["has_perk"]]
    uses_samples = any(r["has_perk"] and r["perk_sample"] for r in recs)
    payload: dict[str, Any] = {
        "status": "ok",
        "source": search.get("source"),
        "recommendations": recs,
        "restaurants_with_perks": perked,
    }
    if uses_samples:
        payload["perk_note"] = (
            "These are SAMPLE partner offers (illustrative, not the restaurant's real "
            "promotion). Present them as a 'sample partner offer' when you mention them."
        )
    return json.dumps(payload, ensure_ascii=False)


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
    party_size = args.get("party_size") or session.pending.get("party_size") \
        or (session.profile or {}).get("party_size") or 2
    avail = check_availability(place_id, iso, party_size)
    slots = avail.get("available_slots", [])
    session.availability = {"place_id": place_id, "date": iso, "party_size": party_size, "slots": slots}
    # Working memory for the outing being planned.
    session.pending.update({"place_id": place_id, "restaurant": rec["name"],
                            "date": iso, "party_size": party_size})
    if not slots:
        return json.dumps({
            "status": "no_availability", "restaurant": rec["name"], "date": iso,
            "message": "No tables free then. Offer another date/time or an alternate "
                       "restaurant from the recommendations.",
        }, ensure_ascii=False)
    return json.dumps({
        "status": "ok", "restaurant": rec["name"], "date": iso, "available_times": slots,
        "remembered": {"restaurant": rec["name"], "date": iso, "party_size": party_size},
        "reminder": "Book the exact time the guest picks from available_times; keep "
                    "this date and party size.",
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
    # Party size is required and must be a real value the guest gave (or their saved
    # usual) — never a silent default. Availability's party_size is NOT trusted here
    # because check_availability_times may have used a placeholder.
    party_size = args.get("party_size") or profile.get("party_size")
    if not isinstance(party_size, int) or party_size < 1:
        return json.dumps({
            "status": "need_party_size",
            "message": "No party size on file. Ask the guest how many people the "
                       "booking is for and pass it as party_size — never guess or "
                       "default it.",
        }, ensure_ascii=False)

    # The time must be one we actually offered for this restaurant + date.
    pend = session.availability
    if pend and pend["place_id"] == place_id and pend["date"] == iso and pend["slots"]:
        if time not in pend["slots"]:
            return json.dumps({
                "status": "time_unavailable", "available_times": pend["slots"],
                "message": "That time isn't available; offer one of the available_times.",
            }, ensure_ascii=False)
        # If the guest explicitly asked for an available time, book THAT time — don't
        # silently substitute a different open slot (the random-time bug).
        wanted = _requested_times(session) & set(pend["slots"])
        if wanted and time not in wanted:
            return json.dumps({
                "status": "time_mismatch",
                "requested_times": sorted(wanted),
                "attempted_time": time,
                "message": "The guest asked for a specific available time. Book exactly "
                           "that (one of requested_times), not a different slot.",
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
        address=rec.get("address"),
        restaurant_phone=rec.get("phone"),
        website=rec.get("website"),
        guest_email=profile.get("email"),
    )
    if not booking.get("booked"):
        return json.dumps({
            "status": "failed", "error": booking.get("error"),
            "message": "Booking failed; offer an alternate time or restaurant.",
        }, ensure_ascii=False)

    result = {
        "status": "booked",
        "restaurant": rec["name"],
        "address": rec.get("address"),
        "restaurant_phone": rec.get("phone"),
        "website": rec.get("website"),
        "date": iso,
        "time": time,
        "party_size": party_size,
        "confirmation_id": booking.get("confirmation_id"),
        "perk_applied": rec.get("perk_title"),
        "perk_sample": rec.get("perk_sample", False),
        "confirmation_sent_to": profile.get("email"),
        "cancellation_policy": (
            f"Free to cancel up to {CANCELLATION_WINDOW_HOURS}h before; inside that "
            "window, call the restaurant directly."
        ),
    }
    session.bookings[key] = result
    session.pending.update({"time": time, "confirmation_id": booking.get("confirmation_id")})
    # Remember the booking + reusable preferences for next time. The booked
    # restaurant's cuisine counts as a fresh preference signal — an actual booking
    # is better evidence of taste than anything the guest said in passing — and
    # the store keeps only the most recent few.
    session.profile = profile_memory.remember(session.member_id, {
        "party_size": party_size,
        "cuisines": [rec["cuisine"]] if rec.get("cuisine") else None,
        "past_bookings": [{
            "restaurant": rec["name"], "confirmation_id": booking.get("confirmation_id"),
            "place_id": place_id, "date": iso, "time": time, "party_size": party_size,
            "status": "confirmed",
        }],
    })
    return json.dumps(result, ensure_ascii=False)


def _handle_cancel(session: ConciergeSession, args: dict[str, Any]) -> str:
    conf = (args.get("confirmation_id") or "").strip()
    if not conf:
        return json.dumps({
            "status": "need_confirmation_id",
            "message": "Ask the guest which reservation to cancel (its confirmation "
                       "id), or call recall_guest_profile to find it in their history.",
        }, ensure_ascii=False)

    result = cancel_booking(conf, reason=args.get("reason"))
    status = result.get("status")

    if status == "cancelled":
        # Keep long-term memory consistent with the ledger.
        updated = profile_memory.mark_booking(session.member_id, conf, "cancelled")
        if updated:
            session.profile = updated
        return json.dumps({
            "status": "cancelled",
            "confirmation_id": conf,
            "message": "Reservation cancelled and noted in the guest's history. "
                       "Confirm warmly and offer further help.",
        }, ensure_ascii=False)

    if status == "too_late":
        # Deterministic 24h guardrail: the agent must NOT claim it cancelled — it
        # relays the restaurant's own contact details so the guest can call.
        return json.dumps({
            "status": "too_late",
            "message": result.get("message"),
            "restaurant_name": result.get("restaurant_name"),
            "restaurant_phone": result.get("restaurant_phone"),
            "website": result.get("website"),
            "instruction": "Do not say it was cancelled. Apologize briefly, explain "
                           "the 24-hour policy, and give the guest the restaurant's "
                           "phone (and website) to cancel directly.",
        }, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)  # already_cancelled | not_found


def _resolve_restaurant(
    session: ConciergeSession, place_id: str | None, name: str | None
) -> dict[str, Any] | None:
    """Find a restaurant the guest has actually been shown, by id or by name.

    Deliberately narrow. The web lookup is the one tool that reaches the open
    internet, so it is only ever pointed at a restaurant already on the table —
    which keeps it from becoming a general-purpose search back door around the
    dining-only guardrail.
    """
    if place_id and place_id in session.recommendations:
        return session.recommendations[place_id]

    known: list[dict[str, Any]] = list(session.recommendations.values())
    # Booked restaurants stay reachable after a later search replaced the shortlist.
    known += [
        {
            "place_id": b.get("place_id"),
            "name": b.get("restaurant"),
            "address": b.get("address"),
            "website": b.get("website"),
        }
        for b in session.bookings.values()
    ]
    if place_id:
        for r in known:
            if r.get("place_id") == place_id:
                return r

    wanted = (name or "").strip().lower()
    if not wanted:
        # No hint at all: if there's exactly one thing in play, that's the one.
        return known[0] if len(known) == 1 else None
    for r in known:
        rname = (r.get("name") or "").lower()
        if rname and (wanted == rname or wanted in rname or rname in wanted):
            return r
    return None


def _handle_highlights(session: ConciergeSession, args: dict[str, Any]) -> str:
    rec = _resolve_restaurant(session, args.get("place_id"), args.get("restaurant_name"))
    if rec is None:
        return json.dumps({
            "status": "unknown_restaurant",
            "message": "I can only look up restaurants already recommended or booked "
                       "in this conversation. Call recommend_restaurants first, or ask "
                       "the guest which of the options they meant.",
        }, ensure_ascii=False)

    focus = (args.get("focus") or "").strip() or "menu highlights and signature dishes"
    result = lookup_dining_highlights(
        restaurant_name=rec["name"],
        address=rec.get("address"),
        website=rec.get("website"),
        place_id=rec.get("place_id"),
        focus=focus,
    )

    if not result.get("highlights") and not result.get("images"):
        return json.dumps({
            "status": "nothing_found",
            "restaurant": rec["name"],
            "message": "The web lookup came back empty. Say you couldn't find much "
                       "about the menu online and offer to ask the restaurant when "
                       "they call — do NOT describe dishes from your own knowledge.",
        }, ensure_ascii=False)

    # A generated, cuisine-themed card fronts the web photos: it gives every
    # restaurant one consistent, designed panel carrying the dishes we actually
    # retrieved and the perk, instead of leading with whatever crop the web had.
    card = menu_card.card_for_restaurant(
        restaurant=rec["name"],
        cuisine=rec.get("cuisine"),
        highlights=result.get("highlights", []),
        perk=rec.get("perk_title"),
        perk_is_sample=rec.get("perk_sample", False),
    )

    # Images go to the UI, not into the model's reply — it would only paste raw
    # URLs into the chat. The text half stays in the transcript for grounding.
    session.media.append({
        "restaurant": rec["name"],
        "focus": focus,
        "card": card,
        "images": result.get("images", []),
        "source": result.get("source"),
        "citations": result.get("citations", []),
    })

    return json.dumps({
        "status": "ok",
        "restaurant": rec["name"],
        "source": result.get("source"),
        "scope": result.get("scope"),
        "highlights": result.get("highlights", []),
        "dishes_on_card": card["dishes"],
        "images_shown_to_guest": len(result.get("images", [])),
        "disclaimer": result.get("disclaimer"),
        "note_shown_to_guest": card.get("note"),
        "instruction": "A menu card, the note in `note_shown_to_guest`, and any photos "
                       "are ALREADY displayed to the guest below your reply — never "
                       "paste image URLs, and don't repeat that note back word for "
                       "word. Point at it in one short line ('I've put their standouts "
                       "on a card below') and add at most one dish of your own from "
                       "`highlights`, attributed. If `dishes_on_card` is empty, don't "
                       "pretend to know what's good — say the menu wasn't published "
                       "online.",
    }, ensure_ascii=False)


_HANDLERS = {
    "remember_guest_details": _handle_remember,
    "recall_guest_profile": _handle_recall,
    "set_confirmation_email": _handle_email,
    "recommend_restaurants": _handle_recommend,
    "check_availability_times": _handle_times,
    "book_table": _handle_book,
    "cancel_reservation": _handle_cancel,
    "show_dining_highlights": _handle_highlights,
}


def _working_memory(session: ConciergeSession) -> dict[str, Any]:
    """What we already know, restated to the model on every tool result.

    Long chats drift: the model forgets a detail the guest gave twenty turns ago
    and asks for it again — the email being the one guests notice and resent.
    Rather than trusting it to remember, every tool result carries the facts back.
    """
    profile = session.profile or {}
    known = {
        "guest_name": profile.get("name"),
        "email_on_file": profile.get("email"),
        "party_size": session.pending.get("party_size") or profile.get("party_size"),
        "date": session.pending.get("date"),
        "restaurant": session.pending.get("restaurant"),
    }
    return {k: v for k, v in known.items() if v}


def _dispatch(session: ConciergeSession, name: str, args: dict[str, Any]) -> str:
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        result = handler(session, args)
    except Exception as exc:  # keep the chat alive if a tool trips
        return f"Tool '{name}' failed: {exc}"

    known = _working_memory(session)
    if not known:
        return result
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        return result
    if not isinstance(payload, dict):
        return result
    payload["known_so_far"] = known
    if known.get("email_on_file"):
        payload["do_not_ask"] = (
            f"The guest's email is already on file ({known['email_on_file']}). Do NOT "
            "ask for it again this session — if you need to reference it, confirm it "
            "instead ('I'll send the confirmation to …, still right?')."
        )
    return json.dumps(payload, ensure_ascii=False)


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
    print(f"\nDino: {_run_turn(session, llm)}\n")

    print("(Type 'quit' or 'exit' to leave.)\n")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nDino: Lovely chatting — enjoy your meal!")
            return
        if user.lower() in {"quit", "exit", "bye"}:
            print("Dino: Lovely chatting — enjoy your meal!")
            return
        if not user:
            continue
        session.messages.append(HumanMessage(content=user))
        print(f"\nDino: {_run_turn(session, llm)}\n")
