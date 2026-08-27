"""Dino — the interpersonal Table for Four concierge (conversational front-end).

A warm, human-facing layer that guides a guest through a full booking *journey*
rather than one-shot auto-booking:

    welcome -> have we met before? (email -> saved preferences)
      -> understand intent -> gather missing details (incl. email)
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
  recognized and their usuals can be reused. Three of those — home area, usual
  party size, favourite cuisines — are *standing* preferences: once on file they
  only change when the guest says so, offered once after a booking rather than
  quietly rewritten by wherever they happened to eat this week.
* **Tool use.** The model reaches the world only through the tools below.
* **Live web retrieval (Tavily).** `show_dining_highlights` fetches cited menu
  highlights and photos for a restaurant the guest is actually considering —
  scoped to that restaurant, attributed to its source, and rendered in the UI
  rather than pasted into the reply.
* **Deterministic guardrails.** Dining-only scope; booking gated on a real email;
  emails/restaurants/times must be ones actually surfaced (no fabrication);
  duplicate bookings are idempotent.

Run it:
    uv run python -m table_for_four chat
    uv run python -m table_for_four chat --name "Manish"

Requires an OpenAI (or OpenRouter) key in `.env`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from table_for_four.agent import menu_card, profile_memory, reasoning, roster
from table_for_four.agent.config import get_chat_llm
from table_for_four.agent.tools import (
    cancel_booking,
    check_availability,
    create_booking,
    find_perks,
    lookup_dining_highlights,
    place_photos,
    search_restaurants,
)
from table_for_four.governance import audit, grounding

MAX_TOOL_HOPS = 6  # guard: bound tool-call chaining within a single guest turn
MAX_RECOMMENDATIONS = 4
CANCELLATION_WINDOW_HOURS = 24  # cancel allowed only more than this far ahead
MAX_MEDIA_IMAGES = 3  # a photo strip, not a gallery — matches the web server's cap


# --- Persona & guardrails ----------------------------------------------------
#
# Dino's brief lives in `roster/dino.md` rather than in a string literal here, so
# the host's instructions sit beside the four back-of-house units he delegates to
# and the whole staffing is readable in one place. It is the same text it has
# always been, byte for byte — `tests/golden/dino_system_prompt.txt` proves it, and
# that proof is the point: moving instructions into files cost the demo nothing.
# Read once at import; nothing is assembled per turn.

SYSTEM_PROMPT = roster.build_system_prompt()


# --- Tool schemas (OpenAI function format) -----------------------------------
#
# The shape of each tool stays here — it is structure, not prose. The *description*
# comes from the owning unit's `.md`, which makes those files genuinely model-facing
# through text that was already being sent: the same characters, sourced from the
# roster instead of duplicated beside it. `tests/test_roster.py` holds the schemas
# to their pre-harness character budget so this can never quietly grow the prompt.

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "remember_guest_details",
            "description": roster.tool_description("remember_guest_details"),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "pronouns": {"type": "string", "description": "e.g. 'she/her', 'they/them'."},
                    "home_location": {"type": "string", "description": "City/area the guest is in."},
                    "party_size": {"type": "integer", "description": "The guest's usual party size."},
                    "dining_atmosphere": {"type": "string", "description": "e.g. 'romantic', 'lively', 'family-friendly'."},
                    "dietary": {"type": "array", "items": {"type": "string"}},
                    "cuisines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Kinds of food the guest likes — 'Italian', 'sushi', "
                            "'steakhouse'. NEVER a restaurant's name, and never a "
                            "category like 'restaurant', 'food' or 'fine dining'."
                        ),
                    },
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
            "name": "confirm_preference_updates",
            "description": roster.tool_description("confirm_preference_updates"),
            "parameters": {
                "type": "object",
                "properties": {
                    "home_location": {"type": "string", "description": "New home city/area."},
                    "party_size": {"type": "integer", "description": "New usual party size."},
                    "cuisines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Cuisine(s) to add to their favourites.",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_guest_profile",
            "description": roster.tool_description("recall_guest_profile"),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_confirmation_email",
            "description": roster.tool_description("set_confirmation_email"),
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
            "description": roster.tool_description("recommend_restaurants"),
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant_name": {
                        "type": "string",
                        "description": (
                            "The venue the guest asked for BY NAME ('Osteria Morini'). "
                            "Set this only when they named a specific restaurant; it "
                            "skips the cuisine question entirely."
                        ),
                    },
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
            "description": roster.tool_description("check_availability_times"),
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
            "description": roster.tool_description("book_table"),
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
            "description": roster.tool_description("cancel_reservation"),
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
            "description": roster.tool_description("show_dining_highlights"),
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
    photo_refs: dict[str, list] = field(default_factory=dict)       # place_id -> Places photo handles
    availability: dict[str, Any] | None = None                      # last times shown
    bookings: dict[str, Any] = field(default_factory=dict)          # key -> result (idempotency)
    pending: dict[str, Any] = field(default_factory=dict)           # outing being planned
    media: list[dict[str, Any]] = field(default_factory=list)       # web highlights for the UI
    pref_offer: dict[str, Any] = field(default_factory=dict)        # standing-pref change put to the guest
    asked_returning: bool = False                                   # "have we met before?" put to the guest
    returning_asked_at: int = 0                                     # where in the transcript we asked it
    # Everything a tool has actually offered this session. The grounding check
    # reads these, so they accumulate rather than tracking only the last lookup:
    # recapping the times of a restaurant discussed ten turns ago is legitimate.
    offered_times: set[str] = field(default_factory=set)
    offered_dates: set[str] = field(default_factory=set)
    time_shortcuts: set[str] = field(default_factory=set)           # place|date already confirmed straight through
    # The reserve gate. `confirm_in_ui` is set only by a surface that can actually
    # show a button; without one the old path stands, because a gate nobody can
    # answer is a dead end rather than a safeguard.
    confirm_in_ui: bool = False
    pending_reservation: dict[str, Any] | None = None               # summary awaiting a press
    reserved: set[str] = field(default_factory=set)                 # keys the guest has pressed Reserve on
    trail: audit.Trail = field(default_factory=audit.Trail)         # M4 governance record

    def __post_init__(self) -> None:
        self.trail.member_id = self.member_id

    @property
    def display_name(self) -> str:
        return (self.profile or {}).get("name") or self.member_id


_HHMM = re.compile(r"^\d{2}:\d{2}$")


def to_12h(hhmm: str) -> str:
    """`19:00` -> `7 PM`, `11:30` -> `11:30 AM` — a time the way a guest says one.

    Only ever for display. Slots, the ledger and every comparison stay on 24-hour
    `HH:MM`, because that is what sorts and matches; this is the last inch before
    the words reach a person.
    """
    try:
        hour, minute = (int(part) for part in hhmm.split(":"))
    except (ValueError, AttributeError):
        return hhmm
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12} {suffix}" if minute == 0 else f"{hour12}:{minute:02d} {suffix}"


def _canonical_time(raw: str, slots: list[str] | None) -> str:
    """Turn whatever the model wrote back into an `HH:MM` slot.

    Dino now says "7 PM" to the guest, so sooner or later it will pass "7 PM"
    here too. A written time can be ambiguous ("7:30" is two different hours), so
    the open slots break the tie; with no single answer the string is handed on
    untouched and the checks below refuse it, which is the safe direction.
    """
    if _HHMM.fullmatch(raw or ""):
        return raw
    readings = grounding.clock_times(raw or "")
    narrowed = readings & set(slots or ())
    if len(narrowed) == 1:
        return next(iter(narrowed))
    return next(iter(readings)) if len(readings) == 1 else raw


# Reading a written time is now governance's job — the same parser has to serve
# the booking guard here and the grounding check on the way out, and two copies
# that drifted would let a claim pass one and fail the other. The name stays for
# the callers below.
_parse_time_tokens = grounding.clock_times


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


# Nudge appended by `_run_turn` when the model runs out of tool hops. It arrives as
# a HumanMessage but is *not* the guest speaking, so consent checks must skip it.
_NO_MORE_TOOLS = "(Please respond to me directly now, no more tools.)"

# A plain yes, in the shapes guests actually type. Deliberately generous about
# wording and deliberately strict about needing *something*: with no match we
# leave the saved preference alone, which is the safe direction to be wrong in.
_AFFIRMATIVE = re.compile(
    r"\b(yes|yeah|yep|yup|sure|ok|okay|please|do it|go ahead|sounds good|that works|"
    r"good idea|great|perfect|correct|absolutely|definitely|update|change|switch|"
    r"set|save|make)\b",
    re.I,
)

# An unprompted request to change something ("make Brooklyn my home area"). Merely
# *mentioning* an area or a headcount is how the drift happened in the first place,
# so a bare mention is never enough — the guest has to ask for the change.
_CHANGE_REQUEST = re.compile(r"\b(update|change|switch|set|save|make|from now on)\b", re.I)


def _guest_replies_since(session: ConciergeSession, index: int) -> list[str]:
    """What the guest themselves typed after message `index` (system nudges excluded)."""
    out: list[str] = []
    for message in session.messages[index:]:
        if not isinstance(message, HumanMessage):
            continue
        text = (message.content or "").strip()
        if text and text != _NO_MORE_TOOLS and not text.startswith("(System:"):
            out.append(text)
    return out


def _offer_preference_changes(
    session: ConciergeSession, proposals: dict[str, dict[str, Any]]
) -> None:
    """Record that a standing-preference change is being put to the guest.

    The index matters: consent has to arrive *after* the question, so a model that
    confirms in the same breath as it asked is confirming nothing.
    """
    session.pref_offer = {"proposals": proposals, "asked_at": len(session.messages)}


def _authorized_changes(
    session: ConciergeSession, updates: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split requested preference changes into (authorized, unauthorized).

    A change is authorized two ways, both grounded in the guest's own words:

    * we offered it and they said yes *afterwards*, or
    * they asked for it unprompted — an explicit "make/change/set …" naming the
      new value themselves.

    Everything else — silence, a topic change, the model's own inference — leaves
    the saved value untouched. The guest not answering is a perfectly fine outcome.
    """
    offer = session.pref_offer or {}
    offered = offer.get("proposals") or {}
    replies = _guest_replies_since(session, int(offer.get("asked_at") or 0))
    joined = " ".join(replies)
    said_yes = bool(replies) and bool(_AFFIRMATIVE.search(joined))

    authorized: dict[str, Any] = {}
    unauthorized: dict[str, Any] = {}
    for field_name, value in updates.items():
        values = value if isinstance(value, list) else [value]
        named_it = all(_guest_typed(session, str(v)) for v in values)
        asked_for_it = named_it and bool(_CHANGE_REQUEST.search(joined or "")) \
            and bool(replies)
        if (field_name in offered and said_yes) or asked_for_it:
            authorized[field_name] = value
        else:
            unauthorized[field_name] = value
    return authorized, unauthorized


# --- Cuisine hygiene ---------------------------------------------------------
#
# A guest's favourite cuisines are long-term memory, so only a real cuisine may
# land there. Two things kept getting in: Google's `primary_type`, which is just
# as often a generic category ("restaurant", "bar", "fine_dining_restaurant") as
# a cuisine ("italian_restaurant"), and the model helpfully passing a venue's own
# name. Neither describes taste, and both are visible to the guest next visit
# ("I remember you love restaurant!"), so they're filtered out at every entry
# point rather than at any one of them.

_NOT_A_CUISINE = {
    "restaurant", "restaurants", "food", "foods", "cuisine", "dining", "fine dining",
    "casual dining", "bar", "wine bar", "pub", "cafe", "coffee", "coffee shop",
    "bakery", "diner", "bistro", "brasserie", "eatery", "brunch", "breakfast",
    "lunch", "dinner", "fast food", "takeaway", "meal takeaway", "meal delivery",
    "buffet", "food court", "night club", "point of interest", "establishment",
    "store", "any", "anything", "something new", "other", "none",
}

# Place types that name a cuisine without the `_restaurant` suffix our stripper
# looks for, or whose bare stem reads oddly back to a guest.
_TYPE_CUISINE_ALIASES = {
    "steak_house": "steakhouse",
    "barbecue_restaurant": "barbecue",
    "hamburger_restaurant": "burgers",
    "pizza_restaurant": "pizza",
    "seafood_restaurant": "seafood",
    "sushi_restaurant": "sushi",
    "ramen_restaurant": "ramen",
}

# Trailing filler on a free-text cuisine: "thai food" / "italian restaurant".
_CUISINE_FILLER = re.compile(r"\s*\b(restaurants?|food|cuisine|places?|spots?)\b\s*$")

# What a cuisine actually is, listed out. The denylist above could only ever
# catch what someone had thought to name, and a guest who asked for chicken
# wings one evening had "chicken wings" read back to them as a favourite cuisine
# on their next visit. A dish is not a taste, and there are more dishes than
# anyone can enumerate — so the question is turned around: a value is kept only
# if it names a kitchen. The list fails closed, which is the right direction for
# a permanent file. Missing one costs a guest a preference they can restate;
# admitting one puts a wrong fact about them on the record.
#
# Deliberately a list and not a judgement call for the model. "Is this a
# cuisine?" has an exact answer, and this codebase checks those rather than
# estimating them (see governance/grounding.py for the same reasoning).
_CUISINES = {
    # Kitchens by place — nationality, region, diaspora.
    "afghan", "afghani", "african", "american", "andhra", "argentine", "argentinian",
    "armenian", "asian", "australian", "austrian", "bangladeshi", "basque", "belgian",
    "bolivian", "brazilian", "british", "bulgarian", "burmese", "cajun", "californian",
    "cambodian", "cantonese", "caribbean", "catalan", "chilean", "chinese", "colombian",
    "creole", "croatian", "cuban", "czech", "danish", "dominican", "dutch", "ecuadorian",
    "egyptian", "english", "eritrean", "ethiopian", "european", "filipino", "finnish",
    "french", "galician", "georgian", "german", "ghanaian", "greek", "gujarati",
    "haitian", "hakka", "hawaiian", "hunan", "hungarian", "iberian", "icelandic",
    "indian", "indonesian", "iranian", "iraqi", "irish", "israeli", "italian",
    "jamaican", "japanese", "jewish", "kenyan", "korean", "latin", "latin american",
    "lebanese", "levantine", "malaysian", "mediterranean", "mexican", "middle eastern",
    "modern american", "modern australian", "modern british", "modern european",
    "mongolian", "moroccan", "nepalese", "nepali", "new american", "nigerian", "nordic",
    "north indian", "norwegian", "oaxacan", "pakistani", "palestinian", "pan asian",
    "peruvian", "persian", "polish", "portuguese", "puerto rican", "romanian",
    "russian", "salvadoran", "scandinavian", "scottish", "senegalese", "serbian",
    "shanghainese", "sichuan", "sicilian", "singaporean", "slovak", "somali",
    "south african", "south indian", "southern", "soul", "spanish", "sri lankan",
    "swedish", "swiss", "syrian", "szechuan", "taiwanese", "thai", "tibetan",
    "trinidadian", "tunisian", "turkish", "tuscan", "ukrainian", "uruguayan",
    "uyghur", "venezuelan", "vietnamese", "welsh", "west african", "yemeni",
    # Kitchens by style — how a place cooks, not one thing it serves.
    "barbecue", "bbq", "burger", "burgers", "chophouse", "comfort", "comfort food",
    "deli", "delicatessen", "dim sum", "fondue", "gastropub", "halal", "hot pot",
    "izakaya", "kebab", "kosher", "noodle", "noodles", "pasta", "pizza", "poke",
    "raclette", "ramen", "rotisserie", "seafood", "smokehouse", "soul food",
    "steakhouse", "street food", "sushi", "taco", "tacos", "tapas", "teppanyaki",
    "trattoria", "vegan", "vegetarian", "yakitori",
}


def _norm_text(value: Any) -> str:
    """Lowercase, punctuation-free, single-spaced — for comparing labels."""
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


# A guest says where they want to eat in the shape of a sentence, not a field:
# "somewhere near Midtown", "around soho", "close to my office". What lands in
# their permanent file should be the place, curated, not the phrasing.
_AREA_LEAD = re.compile(
    r"^(somewhere|anywhere|something)?\s*(near|around|close to|next to|by|in|at|"
    r"over in|out in|round)\s+", re.I
)
_AREA_TRAIL = re.compile(r"\s*\b(area|neighbou?rhood|district|side|part of town)\b\s*$", re.I)

# Phrases that read like a place but name none. Saving one of these as a home
# area is worse than saving nothing: it comes back next visit as a fact about
# the guest, and "here" means nothing a month later in a different conversation.
_NOT_AN_AREA = {
    "", "here", "there", "me", "my place", "my office", "my home", "home", "work",
    "my area", "nearby", "close by", "close", "near me", "around here", "anywhere",
    "any", "anything", "somewhere", "wherever", "local", "locally", "this area",
    "the area", "downtown", "uptown", "city", "the city", "town", "everywhere",
}


def _clean_area(value: Any) -> str | None:
    """A place name fit to keep, or None if the text doesn't actually name one.

    The guest's own words are right for *this* search and wrong for their file.
    "somewhere near soho" is a fine thing to say and a poor thing to remember, so
    the phrasing is stripped, the result is title-cased, and anything that turns
    out to name no place at all is refused rather than stored.

    Deliberately not asked of the model. A normalizer that can only ever delete
    words cannot invent a neighbourhood the guest never mentioned, which is a
    guarantee a fluent rewrite could not make.
    """
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = _AREA_LEAD.sub("", text)
    text = _AREA_TRAIL.sub("", text)
    text = text.strip(" ,.;:-")
    if _norm_text(text) in _NOT_AN_AREA or len(text) > 60:
        return None
    # Leave anything already carrying capitals alone — "SoHo" and "NoMad" are
    # spelled that way, and title-casing would quietly correct them to nonsense.
    return text if any(c.isupper() for c in text) else text.title()


def _clean_cuisine(value: Any) -> str | None:
    """A cuisine label, or None if the text doesn't actually name a cuisine.

    Checked against `_CUISINES` rather than merely against the things we know
    aren't one: "chicken wings" is nobody's cuisine, and no denylist was ever
    going to have it on it.
    """
    label = _norm_text(value)
    if not label or label in _NOT_A_CUISINE:
        return None
    label = _CUISINE_FILLER.sub("", label).strip() or label
    if label in _NOT_A_CUISINE:
        return None
    return label if label in _CUISINES else None


def _cuisine_from_place_type(primary_type: str | None) -> str | None:
    """Read a cuisine off a Google place type, or None if it's just a category."""
    kind = re.sub(r"[^a-z0-9]+", "_", (primary_type or "").lower()).strip("_")
    if not kind:
        return None
    if kind in _TYPE_CUISINE_ALIASES:
        return _TYPE_CUISINE_ALIASES[kind]
    if kind.endswith("_restaurant"):
        return _clean_cuisine(kind[: -len("_restaurant")].replace("_", " "))
    return None  # "restaurant", "bar", "cafe", "food" — a category, not a cuisine


def _is_restaurant_name(session: ConciergeSession, label: str) -> bool:
    """True if `label` is really the name of a venue in play, not a cuisine.

    Exact matches only for a one-word label — plenty of restaurants are called
    "Italian Kitchen", and a guest who loves Italian shouldn't lose the memory
    because of where they ate.
    """
    target = _norm_text(label)
    if not target:
        return False
    names = [r.get("name") for r in session.recommendations.values()]
    names += [b.get("restaurant") for b in session.bookings.values()]
    multiword = " " in target
    for name in names:
        known = _norm_text(name)
        if known and (target == known or (multiword and target in known)):
            return True
    return False


def _clean_cuisines(session: ConciergeSession, value: Any) -> list[str]:
    """Filter a proposed cuisines list down to values worth remembering."""
    incoming = value if isinstance(value, list) else [value]
    out: list[str] = []
    for item in incoming:
        cuisine = _clean_cuisine(item)
        if cuisine and not _is_restaurant_name(session, item) and cuisine not in out:
            out.append(cuisine)
    return out


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

def _curated(session: ConciergeSession, args: dict[str, Any]) -> dict[str, Any]:
    """The parts of a write worth keeping, in the form worth keeping them in.

    Two fields arrive as the guest's own phrasing and must not be filed that way:
    a cuisine that is really a category or a venue's name, and an area that is
    really a sentence ("somewhere near soho") or nothing at all ("nearby"). Both
    are dropped rather than stored badly — a profile is read back to the guest
    later, and a wrong fact about them is worse than a missing one.
    """
    updates = {k: v for k, v in args.items() if v not in (None, "", [], {})}
    if "cuisines" in updates:
        cuisines = _clean_cuisines(session, updates["cuisines"])
        if cuisines:
            updates["cuisines"] = cuisines
        else:
            updates.pop("cuisines")  # "restaurant", a venue's name — not a taste
    if "home_location" in updates:
        area = _clean_area(updates["home_location"])
        if area:
            updates["home_location"] = area
        else:
            updates.pop("home_location")  # "nearby", "my place" — names nowhere
    return updates


def _handle_remember(session: ConciergeSession, args: dict[str, Any]) -> str:
    updates = _curated(session, args)
    # A cuisine that didn't survive curation has to be said out loud, or Dino
    # promises a memory that was never filed: "I'll remember you love chicken
    # wings", against a field that stayed empty.
    refused = (
        " NOT saved: what they named isn't a cuisine — it's a dish, a venue or a "
        "category. Don't tell them you'll remember it as a favourite cuisine."
        if args.get("cuisines") and "cuisines" not in updates else ""
    )
    if not updates:
        return "Nothing to save." + refused

    # Standing preferences (home area, usual party size, the cuisine top three) are
    # never overwritten by a passing mention — the guest has to agree to the change.
    blocked = profile_memory.sticky_conflicts(session.profile, updates)
    allowed = {k: v for k, v in updates.items() if k not in blocked}
    if allowed:
        session.profile = profile_memory.remember(session.member_id, allowed)
    if not blocked:
        return "Saved: " + ", ".join(sorted(allowed)) + refused

    _offer_preference_changes(session, blocked)
    return json.dumps({
        "status": "saved_with_confirmation_needed",
        "saved": sorted(allowed),
        "not_saved": blocked,
        "message": refused + (
            "The fields in `not_saved` are STANDING preferences with a different "
            "value already on file, so they were left unchanged. Ask the guest once, "
            "lightly, whether they'd like the saved value updated. If they clearly "
            "say yes, call confirm_preference_updates with just those fields; if they "
            "don't answer or move on, leave it and never ask again."
        ),
    }, ensure_ascii=False)


def _handle_confirm_prefs(session: ConciergeSession, args: dict[str, Any]) -> str:
    """Apply a standing-preference change the guest actually asked for."""
    updates = _curated(session, args)
    if not updates:
        return json.dumps({
            "status": "nothing_to_update",
            "message": "Pass the field(s) the guest agreed to change.",
        }, ensure_ascii=False)

    authorized, unauthorized = _authorized_changes(session, updates)
    if authorized:
        session.profile = profile_memory.remember(session.member_id, authorized)
        # One offer, one answer: don't let a later "yes" to something else reuse it.
        session.pref_offer = {}

    if not unauthorized:
        return json.dumps({
            "status": "updated",
            "updated": authorized,
            "message": "Saved. Acknowledge it in half a sentence and move on.",
        }, ensure_ascii=False)

    return json.dumps({
        "status": "not_authorized" if not authorized else "partly_updated",
        "updated": authorized,
        "unchanged": sorted(unauthorized),
        "message": (
            "The guest hasn't clearly agreed to change the field(s) in `unchanged`, "
            "so their saved value stands. That's a fine outcome — do NOT pester them. "
            "Only if you genuinely haven't asked yet, ask once; otherwise carry on "
            "with the booking and drop it."
        ),
    }, ensure_ascii=False)


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
    key, profile, returning, conflicts = profile_memory.adopt_email(session.member_id, email)
    session.member_id = key
    session.profile = profile
    session.asked_returning = True  # identity settled; don't gate the search on it
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
    if conflicts:
        # What they've said so far this session differs from their standing profile.
        # Recognising someone is the worst possible moment to overwrite them, so the
        # saved values stand and the difference becomes part of the welcome back.
        _offer_preference_changes(session, conflicts)
        payload["preference_check"] = {
            "proposals": conflicts,
            "instruction": (
                "What the guest mentioned this session differs from the standing "
                "preferences on their file, which were left UNCHANGED. Fold this into "
                "your welcome back as ONE light question — 'I've got you down as "
                "Manhattan, table for two — still right, or shall I make tonight's the "
                "new usual?' Ask once, then carry on with tonight's booking either way "
                "(this outing uses what they told you today regardless). If they don't "
                "answer, change nothing and never raise it again. Only on a clear yes, "
                "call confirm_preference_updates with just the fields they agreed to."
            ),
        }
    return json.dumps(payload, ensure_ascii=False)


def _shortlist(
    session: ConciergeSession,
    candidates: list[dict[str, Any]],
    keywords: str,
    party_size: int | None,
    day: str | None,
    limit: int = MAX_RECOMMENDATIONS,
    exclude: set[str] = frozenset(),
) -> list[dict[str, Any]]:
    """Turn raw search hits into perk-flagged rows, registered as bookable.

    Shared by the shortlist and by the alternatives offered when a restaurant has
    no tables free: an alternative the guest can't then book would be a tease, and
    only rows in `session.recommendations` survive the checks in `book_table`.
    """
    picked = [c for c in candidates if c.get("place_id") not in exclude][:limit]
    if not picked:
        return []
    perks = find_perks(
        query=keywords,
        place_ids=[c["place_id"] for c in picked],
        party_size=party_size,
        day=day,
    ).get("results", [])
    best_perk: dict[str, dict[str, Any]] = {}
    for p in perks:
        pid = p.get("place_id")
        if pid and p.get("similarity", 0) > best_perk.get(pid, {}).get("similarity", -1):
            best_perk[pid] = p

    recs = []
    for c in picked:
        perk = best_perk.get(c["place_id"])
        rec = {
            "place_id": c["place_id"],
            "name": c["name"],
            "address": c.get("address"),
            "phone": c.get("phone"),
            "website": c.get("website"),
            "cuisine": _cuisine_from_place_type(c.get("primary_type")),
            "rating": c.get("rating"),
            "price_level": c.get("price_level"),
            "has_perk": bool(perk),
            "perk_sample": False,
            "perk_title": (perk or {}).get("title"),
            "perk_id": (perk or {}).get("perk_id"),
        }
        recs.append(rec)
        session.recommendations[c["place_id"]] = rec
        # Photo handles live in their own map rather than on the rec. They are long
        # opaque strings the model has no use for — it never sees an image — and
        # four restaurants' worth would cost tokens on every shortlist. Keeping
        # them beside it rather than in a copy of it matters: callers finish the
        # rec after this returns (sample perks are attached that way), and a copy
        # silently dropped those edits.
        session.photo_refs[c["place_id"]] = c.get("photos") or []
    return recs


def _handle_recommend(session: ConciergeSession, args: dict[str, Any]) -> str:
    # A guest who names the restaurant has already made every choice the shortlist
    # exists to help with. Asking them what cuisine they fancy, or which of their
    # usuals to reuse, reads as not having listened — so the name short-circuits
    # both the taste questions and the "have we met?" gate, and we go straight to
    # looking the place up. The email is still required before booking (step 6).
    named = (args.get("restaurant_name") or "").strip()

    # Have we met before? A guest who has booked with Dino already has cuisines, an
    # area, a party size and dietary needs on file — all of it worth searching WITH
    # rather than discovering again. But we only find them by email, so ask for it
    # BEFORE the first search instead of at booking time, when it's too late to
    # shape the shortlist. Asked once per session, and never when we already have a
    # profile: a guest saying "no, I'm new" is a fine answer that unblocks the search.
    if not named and not (session.profile or {}).get("email"):
        if not session.asked_returning:
            session.asked_returning = True
            session.returning_asked_at = len(session.messages)
            return json.dumps({
                "status": "ask_if_returning",
                "message": (
                    "Before searching, ask the guest ONE friendly question: have they "
                    "dined with us before? If yes, ask for the email they used and call "
                    "set_confirmation_email — their saved cuisines, area, usual party "
                    "size and dietary needs then shape this search (offer them as in "
                    "step 2b). If they're new or would rather not say, just carry on and "
                    "search — don't ask again, and don't re-ask for the email later."
                ),
            }, ensure_ascii=False)
        # Asking is only worth anything if we wait for the answer: the search stays
        # shut until the guest has actually spoken since the question.
        if not _guest_replies_since(session, session.returning_asked_at):
            return json.dumps({
                "status": "awaiting_answer",
                "message": (
                    "You've asked whether they've dined with us before, but the guest "
                    "hasn't answered yet. Stop calling tools and put the question to "
                    "them; search once they've replied."
                ),
            }, ensure_ascii=False)

    # Searching by name means searching for that place, not for its category: a
    # cuisine filter here can only exclude the very restaurant being asked for.
    cuisine = None if named else args.get("cuisine")
    keywords = named or args.get("keywords") or cuisine or "restaurant"
    party_size = args.get("party_size")
    # Working memory: remember the outing's party size / date as they're mentioned.
    if party_size:
        session.pending["party_size"] = party_size
    if args.get("date"):
        session.pending["date"] = args["date"]
    if args.get("location"):
        # Where *this outing* is — not where the guest lives. Kept apart from the
        # saved home area on purpose; see the preference check after booking.
        session.pending["location"] = args["location"]
    # What the guest asked for is the fallback taste signal when the place's own
    # Google type is the useless generic "restaurant".
    requested_cuisine = _clean_cuisine(cuisine)
    if requested_cuisine:
        session.pending["cuisine"] = requested_cuisine
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
    if not candidates and named:
        # The area may just be loosely worded ("Soho" vs the listed address), so try
        # the name alone. What we must NOT do is fall back to a generic search and
        # present strangers as though they were the restaurant they asked for.
        candidates = search_restaurants(query=named).get("results", [])
    elif not candidates and (cuisine or args.get("location")):
        # An over-specific keyword string can zero out live results; retry once with a
        # minimal query so we keep the requested cuisine instead of dropping it.
        search = search_restaurants(
            query=cuisine or "restaurant",
            cuisine=cuisine,
            location=args.get("location"),
        )
        candidates = search.get("results", [])
    if not candidates:
        if named:
            return json.dumps({
                "status": "restaurant_not_found",
                "restaurant_name": named,
                "message": (
                    f"Couldn't find '{named}'. Say so plainly, check you have the name "
                    "and area right, and offer to look for somewhere similar nearby — "
                    "then call recommend_restaurants with a cuisine and location "
                    "instead. Never present a different restaurant as the one they asked for."
                ),
            }, ensure_ascii=False)
        return json.dumps({
            "status": "no_matches",
            "message": "No restaurants matched. Ask the guest to relax a filter "
                       "(cuisine, area, price) and call recommend_restaurants again.",
        }, ensure_ascii=False)

    session.recommendations = {}
    recs = _shortlist(session, candidates, keywords, party_size, day)

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
    if named:
        payload["named_lookup"] = named
        # A name is not always one restaurant — chains and common names ("Nobu")
        # legitimately match several. Asking which branch they meant is the one
        # question this path still owes the guest; it is not a taste question.
        if len(recs) == 1:
            payload["instruction"] = (
                f"The guest asked for '{named}' by name, so this is a lookup, not a "
                "shortlist. Confirm you've found it in one line and do NOT ask about "
                "cuisine, their usuals, or what kind of place they fancy — they've told "
                "you. The only things still missing are the party size and the "
                "date/time; ask for whichever you don't have, then call "
                "check_availability_times. You'll need their email before booking, but "
                "that can wait until then."
            )
        else:
            payload["instruction"] = (
                f"'{named}' matched {len(recs)} places, so ask which one they meant — "
                "name them by address or neighbourhood, nothing else. That is the ONLY "
                "question owed here: still do NOT ask about cuisine, their usuals, or "
                "what kind of place they fancy. Once they pick, carry on with party "
                "size and date/time, then call check_availability_times."
            )
    return json.dumps(payload, ensure_ascii=False)


ALTERNATIVES_WHEN_FULL = 3


def _similar_nearby(
    session: ConciergeSession, rec: dict[str, Any], party_size: int | None, iso: str
) -> list[dict[str, Any]]:
    """Restaurants of the same kind, in the same area, when the first choice is full.

    "Similar" is deliberately the guest's own two constraints — cuisine and area —
    rather than a taste judgement of ours. The cuisine comes from the place they
    picked, falling back to what they searched for; the area is this outing's
    location, which may simply be absent when the guest named a restaurant and no
    neighbourhood. Cuisine alone is still a fair basis; neither is not.
    """
    cuisine = _clean_cuisine(rec.get("cuisine")) or session.pending.get("cuisine")
    location = session.pending.get("location")
    if not (cuisine or location):
        return []  # nothing to be similar *to*; don't offer arbitrary restaurants

    query = f"{cuisine} restaurant" if cuisine else "restaurant"
    try:
        found = search_restaurants(query=query, cuisine=cuisine, location=location)
    except Exception:
        return []  # a dead search must not take the availability answer down with it
    day = None
    try:
        _, day = reasoning.resolve_date(iso)
    except Exception:
        day = None
    return _shortlist(
        session,
        found.get("results", []),
        query,
        party_size,
        day,
        limit=ALTERNATIVES_WHEN_FULL,
        exclude={rec.get("place_id")},
    )


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
    # Open times depend on the party size — the backend seats a party of eight far
    # more sparsely than a couple — so a placeholder here is not a harmless default.
    # It showed the guest times they could not actually have: offered for a made-up
    # two, refused at the pass for the real eight, then offered again from the same
    # placeholder. That is the loop a guest booking a big table kept hitting, and it
    # only ever broke when they happened to pick a slot that survived both rules.
    # Worse, the placeholder was filed in `pending` as though the guest had said it.
    party_size = args.get("party_size") or session.pending.get("party_size") \
        or (session.profile or {}).get("party_size")
    if not isinstance(party_size, int) or party_size < 1:
        return json.dumps({
            "status": "need_party_size",
            "message": "Ask the guest how many people are coming before offering "
                       "times — which tables are free depends on it. Then call "
                       "check_availability_times again with party_size.",
        }, ensure_ascii=False)
    avail = check_availability(place_id, iso, party_size)
    slots = avail.get("available_slots", [])
    session.availability = {"place_id": place_id, "date": iso, "party_size": party_size, "slots": slots}
    # What we have actually offered, for the grounding check on the way out.
    session.offered_times.update(slots)
    session.offered_dates.add(iso)
    # Working memory for the outing being planned.
    session.pending.update({"place_id": place_id, "restaurant": rec["name"],
                            "date": iso, "party_size": party_size})
    if not slots:
        # A closed door is not an answer. Before handing the guest back a "no", look
        # for places of the same kind in the same area — that's what they'd ask for
        # next anyway, and having it ready turns a dead end into a choice.
        alternatives = _similar_nearby(session, rec, party_size, iso)
        payload: dict[str, Any] = {
            "status": "no_availability", "restaurant": rec["name"], "date": iso,
            "message": "No tables free then.",
        }
        if alternatives:
            payload["alternatives"] = alternatives
            payload["instruction"] = (
                f"{rec['name']} is full on {iso}. Say so briefly, then offer these "
                "similar places in the same area — by name, mentioning any perk — and "
                "ask whether they'd like one of them or would rather try a different "
                "date at their first choice. They are already bookable: call "
                "check_availability_times with the place_id the guest picks. Do not "
                "invent alternatives beyond this list."
            )
        else:
            payload["instruction"] = (
                "Nothing similar came back nearby either. Offer another date or time "
                "at the same restaurant, or ask whether to widen the area."
            )
        return json.dumps(payload, ensure_ascii=False)
    payload = {
        "status": "ok", "restaurant": rec["name"], "date": iso,
        "available_times": slots,
        "available_times_display": [to_12h(s) for s in slots],
        "remembered": {"restaurant": rec["name"], "date": iso, "party_size": party_size},
        "reminder": "Say times to the guest the way available_times_display writes "
                    "them (7 PM, not 19:00). Book the exact time the guest picks; "
                    "keep this date and party size.",
    }

    # If the guest already named a time and it's free, they have chosen. Telling
    # the model not to read the list back didn't hold — it still did — so the list
    # is simply not here to read. Their time is the only one in the payload; the
    # full set stays on the session for the booking guard, and the model can ask
    # for it again if the guest actually wants alternatives. Once per restaurant
    # and date, so "what else is open?" gets a real answer on the second call.
    already_asked = sorted(_requested_times(session) & set(slots))
    shortcut = f"{place_id}|{iso}"
    if already_asked and shortcut not in session.time_shortcuts:
        session.time_shortcuts.add(shortcut)
        chosen = already_asked[0]
        payload["available_times"] = [chosen]
        payload["available_times_display"] = [to_12h(chosen)]
        payload["guest_already_chose"] = to_12h(chosen)
        payload["other_times_open"] = len(slots) - 1
        payload["instruction"] = (
            f"The guest asked for {to_12h(chosen)} and it is free, so that is the "
            "only time listed here — the others are deliberately withheld. Do NOT "
            "read a list back or ask them to pick again; they have already chosen. "
            "Confirm that time, read the booking details back for a yes, and spend "
            "the question you just saved on something they'd like (photos, or what "
            "people order there). If they ask what else is open, call "
            "check_availability_times again and the full list comes back."
        )
    return json.dumps(payload, ensure_ascii=False)


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

    # The time must be one we actually offered for this restaurant on this date.
    # Everything below used to hang off `if pend and ...`, which meant a model that
    # skipped step 5 entirely — never asked for a day, never showed the guest a
    # single open slot — fell straight past the checks with a plausible-looking
    # "19:00" and booked it. That is the same bug as inventing a restaurant, and it
    # gets the same answer: no offered slots, no booking. Demo feedback, and the
    # reason the refusal lives here rather than only in the brief.
    pend = session.availability
    if not (pend and pend["place_id"] == place_id and pend["date"] == iso):
        return json.dumps({
            "status": "need_availability_check",
            "requested_date": iso,
            "message": "No open times have been offered for this restaurant on this "
                       "date. Confirm the date with the guest, call "
                       "check_availability_times, show them what came back, and book "
                       "only the time they pick. Never assume a date or a time.",
        }, ensure_ascii=False)
    if pend["party_size"] != party_size:
        # The slot list on file was computed for a different number of people, so it
        # is not the list this booking may choose from. Catching it here turns what
        # used to be a bare 409 from the backend — which the model could only retry
        # blindly — into an instruction it can actually act on.
        return json.dumps({
            "status": "party_size_changed",
            "times_were_for": pend["party_size"],
            "booking_is_for": party_size,
            "message": "Those open times were looked up for a different party size. "
                       "Call check_availability_times again with the real party_size "
                       "and let the guest pick from what comes back.",
        }, ensure_ascii=False)
    if not pend["slots"]:
        # We did look, and there was nothing. Booking anyway would invent a table.
        return json.dumps({
            "status": "no_availability", "date": iso,
            "message": "Nothing was free at this restaurant on this date. Offer the "
                       "alternatives or another date — do not book.",
        }, ensure_ascii=False)
    # Dino talks to the guest in "7 PM" now, so it will pass that here sooner or
    # later. The ledger keys on 19:00; the open slots settle any ambiguity.
    time = _canonical_time(time, pend["slots"])
    if time not in pend["slots"]:
        return json.dumps({
            "status": "time_unavailable",
            "available_times": pend["slots"],
            "available_times_display": [to_12h(s) for s in pend["slots"]],
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

    # Assembled before the gate rather than after it, so the guest is shown the
    # same special requests that will actually reach the restaurant.
    extras: list[str] = []
    if profile.get("dietary"):
        extras.append(", ".join(profile["dietary"]))
    if args.get("special_requests"):
        extras.append(str(args["special_requests"]))
    high_chairs = sum(1 for k in (profile.get("kids") or []) if k.get("needs_high_chair"))
    if high_chairs:
        extras.append(f"{high_chairs} high chair(s)")
    special = "; ".join(extras) or None

    # The hand on the door. Every check above says the booking is *possible*; none
    # of them says the guest wants it. In the graph path that question is asked by
    # `gate_node`, which interrupts and waits — but the chat path had no equivalent,
    # so a model that decided a guest had agreed simply booked. "Shall I confirm?"
    # answered in prose is the model's reading of consent, not consent.
    #
    # So the surface that can show a summary and two buttons asks for one. Only
    # the app sets `confirm_in_ui`; the REPL and the tests keep the old path, where
    # there is no button to press.
    if session.confirm_in_ui and key not in session.reserved:
        session.pending_reservation = {
            "key": key,
            "restaurant": rec["name"],
            "address": rec.get("address"),
            "date": iso,
            "time": time,
            "time_display": to_12h(time),
            "party_size": party_size,
            "guest_name": session.display_name,
            "perk": rec.get("perk_title"),
            "perk_sample": rec.get("perk_sample", False),
            "special_requests": special,
            "email": profile.get("email"),
            "photo": _gate_photo(session, rec),
            # Exactly what was validated, kept so pressing Reserve replays this
            # booking rather than whatever the model reconstructs a turn later.
            "args": {**args, "place_id": place_id, "date": iso, "time": time,
                     "party_size": party_size},
        }
        session.trail.record("reservation_gate", actor="booker", stage="shown",
                             restaurant=rec["name"], date=iso, time=time,
                             party_size=party_size)
        return json.dumps({
            "status": "awaiting_confirmation",
            # The photo is for the guest's eyes, not the model's: a signed image
            # URL in the transcript is tokens spent on something Dino is forbidden
            # to repeat, and the one thing it might do with it is paste it.
            "summary": {k: v for k, v in session.pending_reservation.items()
                        if k != "photo"},
            "message": (
                "NOT booked yet. The full details are on screen with a photo and a "
                "Reserve button, and nothing is reserved until the guest presses it. Say "
                "in one short line that you've put the details below for them to "
                "confirm. Do NOT claim the table is booked, do NOT invent a "
                "confirmation id, and do not call book_table again until they act."
            ),
        }, ensure_ascii=False)

    session.pending_reservation = None  # the door is open; don't leave it showing
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
        # A refusal at the pass is the moment a guest most needs to be told where
        # they stand, and "Booking failed; offer an alternate time" left them
        # guessing whether they had a table. Two things are owed here: that nothing
        # was reserved, said plainly, and times that are open *now* — re-offering
        # the list that just failed is what made the retry feel like a loop.
        fresh = check_availability(place_id, iso, party_size).get("available_slots", [])
        session.availability = {"place_id": place_id, "date": iso,
                                "party_size": party_size, "slots": fresh}
        session.offered_times.update(fresh)
        stands = (f"NOT booked — nothing is reserved. {time} is no longer available "
                  f"for {party_size} on {iso}. Tell the guest both parts plainly: no "
                  "reservation was made, and that time has gone.")
        return json.dumps({
            "status": "not_booked",
            "error": booking.get("error"),
            "attempted_time": time,
            "available_times": fresh,
            "available_times_display": [to_12h(s) for s in fresh],
            "message": stands + (
                " Then offer the times in available_times and let them pick another. "
                "Never imply the table is being held."
                if fresh else
                " Nothing else is open there that day, so offer another date or one "
                "of the other restaurants."
            ),
        }, ensure_ascii=False)

    result = {
        "status": "booked",
        "restaurant": rec["name"],
        "address": rec.get("address"),
        "restaurant_phone": rec.get("phone"),
        "website": rec.get("website"),
        "date": iso,
        "time": time,
        "time_display": to_12h(time),
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

    # What this booking says about the guest. A first value is simply learned — an
    # actual booking is better evidence of taste than anything said in passing. But
    # once a standing preference is on file, one outing must not quietly rewrite it:
    # those become a question for the guest (step 7b) rather than a silent update.
    signals: dict[str, Any] = {"party_size": party_size}
    # The place's own type first (it's what they actually ate), falling back to the
    # cuisine the guest searched for when Google only says "restaurant". Both go
    # through the same filter: the fallback used to skip it, which is how a search
    # phrase — "chicken wings" — ended up on file as a favourite cuisine.
    booked_cuisine = next(iter(_clean_cuisines(
        session, [rec.get("cuisine"), session.pending.get("cuisine")])), None)
    if booked_cuisine:
        signals["cuisines"] = [booked_cuisine]
    # Curated, not quoted: what goes in the file is the place, not the sentence
    # the guest happened to ask in. A phrase that names nowhere is simply dropped.
    area = _clean_area(session.pending.get("location"))
    if area:
        signals["home_location"] = area

    conflicts = profile_memory.sticky_conflicts(session.profile, signals)
    learn = {k: v for k, v in signals.items() if k not in conflicts}
    learn["past_bookings"] = [{
        "restaurant": rec["name"], "confirmation_id": booking.get("confirmation_id"),
        "place_id": place_id, "date": iso, "time": time, "party_size": party_size,
        "status": "confirmed",
    }]
    session.profile = profile_memory.remember(session.member_id, learn)

    # The table is booked; now tell them about the place they're going. This used
    # to be step 7 of the brief and nothing more — an instruction the model
    # followed most of the time, which in this codebase has repeatedly meant "not
    # when it mattered". Doing it here makes it part of the confirmation instead
    # of a thing Dino might remember, and it costs no extra model call: the
    # dishes and photos arrive in the same result, so the reply that confirms the
    # booking is also the reply that shows them, in one hop rather than two.
    shown = _show_highlights(session, rec, "signature dishes and what the place looks like")
    if shown is not None:
        highlights, card = shown
        result["dining_highlights"] = {
            "dishes_on_card": card["dishes"],
            "highlights": highlights.get("highlights", []),
            "note_shown_to_guest": card.get("note"),
            "source": highlights.get("source"),
            "disclaimer": highlights.get("disclaimer"),
        }
        result["instruction"] = (
            "Photos and a menu card for this restaurant are ALREADY displayed to "
            "the guest below your reply — never paste image URLs. Lead with the "
            "good part: a line about what people order there, drawn from "
            "`dining_highlights` and attributed ('diners keep mentioning…'), then "
            "a couple of brief practical tips (arrive a few minutes early, mention "
            "the reservation name and any dietary need, note the perk at the "
            "table), and close with the booking summary — restaurant, date, time, "
            "party size, confirmation id, and the perk applied if there is one. "
            "Never invent a dish: if `dishes_on_card` is empty, just say the menu "
            "wasn't published online and give the summary."
        )
    else:
        result["instruction"] = (
            "No menu details came back for this restaurant. Do NOT describe dishes "
            "from your own knowledge — give the practical tips and the booking "
            "summary (restaurant, date, time, party size, confirmation id, and the "
            "perk applied if there is one)."
        )

    if conflicts:
        _offer_preference_changes(session, conflicts)
        result["preference_check"] = {
            "proposals": conflicts,
            "instruction": (
                "This outing differs from the guest's saved standing preferences, "
                "which were left UNCHANGED. Once you've confirmed the booking, ask "
                "in ONE short, light question whether they'd like the saved value "
                "updated (e.g. 'want me to make Brooklyn your home area from now "
                "on?'). Ask once and move on. If they don't answer, change nothing "
                "and never raise it again. Only on a clear yes, call "
                "confirm_preference_updates with just the fields they agreed to."
            ),
        }
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


def _gate_photo(session: ConciergeSession, rec: dict[str, Any]) -> dict[str, Any] | None:
    """One picture to go with the question, at the moment it is asked.

    A summary table is the honest thing to put in front of someone before an
    irreversible write, and it is also the least appetising thing in the app:
    nobody says yes to a spreadsheet. So the gate carries a single image of the
    place. One, not a strip — the question on screen is "is this the table you
    want?", and a gallery answers a different question.

    Cheapest source first. Anything already fetched for this restaurant this
    session costs nothing and is what the guest is already looking at; failing
    that, one photo resolved against the place id, which is Curator's capability
    and borrowed as Curator rather than reached for from behind the pass; failing
    that — offline, no key — the generated card, which needs neither.
    """
    for item in reversed(session.media):
        if item.get("restaurant") != rec["name"]:
            continue
        for img in item.get("images") or []:
            return img
        if item.get("card"):
            return item["card"]

    with roster.acting_as("curator"):
        # One reference, one round trip. The gate is not the place to spend three.
        photos = place_photos((session.photo_refs.get(rec.get("place_id")) or [])[:1])
    if photos:
        return photos[0]

    return menu_card.card_for_restaurant(
        restaurant=rec["name"],
        cuisine=rec.get("cuisine"),
        highlights=[],
        perk=rec.get("perk_title"),
        perk_is_sample=rec.get("perk_sample", False),
    )


def _show_highlights(
    session: ConciergeSession, rec: dict[str, Any], focus: str
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Look a restaurant up, put the card and photos in front of the guest.

    Shared by the guest asking ("what's good there?") and by the confirmation,
    which shows this without being asked. Returns None when the lookup found
    nothing at all, so the caller can say so rather than invent something.

    Runs as Curator whoever calls it. This is Curator's work — it is the only
    unit allowed near the open web — and Booker, who calls it after a booking,
    is explicitly forbidden from it. Stepping into the right unit is the honest
    way to do that; reaching for the capability from inside Booker would be the
    dishonest one, and the broker would refuse it anyway.
    """
    with roster.acting_as("curator"):
        result = lookup_dining_highlights(
            restaurant_name=rec["name"],
            address=rec.get("address"),
            website=rec.get("website"),
            place_id=rec.get("place_id"),
            focus=focus,
        )
        # Photos of the restaurant itself go first. Tavily has only the name to
        # search on, which is how a guest ends up looking at another branch or a
        # namesake in another city; Places resolves against the place id, so
        # these are the right restaurant by construction. Web images stay on as
        # the top-up, and offline this is empty so placeholders are untouched.
        own_photos = place_photos(session.photo_refs.get(rec.get("place_id")))

    images = own_photos + [
        img for img in result.get("images", [])
        if img.get("url") not in {p["url"] for p in own_photos}
    ]
    if not result.get("highlights") and not images:
        return None

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
        "images": images[:MAX_MEDIA_IMAGES],
        "source": result.get("source"),
        # So the UI can tell the guest where the photos actually came from —
        # the restaurant's own domain reads very differently from "the web".
        "website": rec.get("website"),
        "citations": result.get("citations", []),
    })
    return result, card


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
    shown = _show_highlights(session, rec, focus)
    if shown is None:
        return json.dumps({
            "status": "nothing_found",
            "restaurant": rec["name"],
            "message": "The web lookup came back empty. Say you couldn't find much "
                       "about the menu online and offer to ask the restaurant when "
                       "they call — do NOT describe dishes from your own knowledge.",
        }, ensure_ascii=False)
    result, card = shown

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
    "confirm_preference_updates": _handle_confirm_prefs,
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


def _outcome_of(result: str) -> str:
    """The handler's own status word, for the trail. Handlers all return JSON."""
    try:
        payload = json.loads(result)
    except (ValueError, TypeError):
        return "unparsed"
    return payload.get("status", "ok") if isinstance(payload, dict) else "ok"


def _dispatch(session: ConciergeSession, name: str, args: dict[str, Any]) -> str:
    """Hand one tool call to the unit that owns it, and run it under that unit's grant.

    This was already the single chokepoint for every tool call in the chat path, so
    it is the natural place to say *who is acting*. Inside the `acting_as` block the
    broker in `agent.tools` and `agent.profile_memory` will refuse anything that
    unit was not granted — the Curator cannot book a table even if a future edit
    wires it a way to try.
    """
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    try:
        with roster.acting_as(roster.unit_for_handler(name)):
            result = handler(session, args)
    except roster.NotGranted:
        # A unit reaching past its grant is a bug in us, not a tool having a bad day.
        # Swallowing it into a chat message would hide exactly the thing the harness
        # exists to catch, so it goes up loudly.
        raise
    except Exception as exc:  # keep the chat alive if a tool trips
        return f"Tool '{name}' failed: {exc}"

    # The trail wants the same thing the grant check wanted, and this is already
    # the one place that knows it: which unit acted, on what, and how it went.
    # Arguments are recorded, results are not — a result can be long, and what
    # governance needs to answer is what was *asked for* by whom.
    session.trail.record(
        "tool_call",
        actor=roster.unit_for_handler(name),
        tool=name,
        args=args,
        outcome=_outcome_of(result),
    )

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
    """Invoke the model, resolve any tool calls, and return the reply text.

    The whole turn runs as Dino, who is granted no capability at all — he talks, and
    everything else is somebody's job. `_dispatch` steps into the owning unit for
    the length of a tool call and back out again, so the only code that can touch
    the world is code a unit was declared for.
    """
    with roster.acting_as("dino"):
        for _ in range(MAX_TOOL_HOPS):
            response: AIMessage = llm.invoke(session.messages)
            session.messages.append(response)
            if not response.tool_calls:
                return _vetted(session, response.content or "")
            for call in response.tool_calls:
                result = _dispatch(session, call["name"], call.get("args") or {})
                session.messages.append(ToolMessage(content=result, tool_call_id=call["id"]))
        session.messages.append(HumanMessage(content=_NO_MORE_TOOLS))
        final: AIMessage = llm.invoke(session.messages)
        session.messages.append(final)
        return _vetted(session, final.content or "")


def _vetted(session: ConciergeSession, reply: str) -> str:
    """The last thing between the model and the guest.

    Every refusal in this file guards an *action* — a booking that names a
    restaurant nobody surfaced, at a time nobody offered. None of them guard the
    sentence. This does: a time, date, confirmation id or email in the reply that
    no tool ever returned is removed before the guest reads it, and recorded
    either way, so the trail says what was checked and not merely what was sent.
    """
    verdict = grounding.check(reply.strip(), grounding.GroundedFacts.gather(session))
    if not verdict.grounded:
        session.trail.record("grounding", actor="dino", **verdict.as_audit())
    return verdict.reply


def press_reserve(session: ConciergeSession) -> str:
    """The guest pressed Reserve: record the consent, then make the booking.

    The press goes into the transcript in the guest's own voice, so the booking is
    answerable to something they did. But the booking itself happens *here* rather
    than by asking the model to call `book_table` again — the first version did
    that, and when the model replied conversationally instead of calling the tool,
    the press did nothing and the guest was handed back the time picker.

    Nothing is being decided at this point. The guest approved these exact details
    a moment ago; carrying them out is not a judgement, and a step with no judgement
    in it should not depend on a model choosing to take it.

    Appends the guest's message itself, so the caller must not append it again.
    """
    pending = session.pending_reservation or {}
    session.reserved.add(pending.get("key", ""))
    session.pending_reservation = None
    session.trail.record(
        "reservation_gate", actor=None, stage="approved",
        restaurant=pending.get("restaurant"), date=pending.get("date"),
        time=pending.get("time"), party_size=pending.get("party_size"),
    )

    said = "Yes — reserve it, please."
    session.messages.append(HumanMessage(content=said))
    # `_dispatch`, not the handler directly: it runs Booker under its own grant and
    # writes the tool call to the governance trail like any other.
    result = _dispatch(session, "book_table", dict(pending.get("args") or {}))
    # The order here is the guest's, not the data model's. Before the gate, the
    # food led and the summary closed; now they have just pressed a button and the
    # first thing they want is to be told it worked.
    session.messages.append(HumanMessage(content=(
        "(System: the guest pressed Reserve. `book_table` has ALREADY run and the "
        "table is booked — do NOT call it again, and do not ask them to confirm "
        f"anything else.\n\nResult:\n{result}\n\n"
        "Write the confirmation reply now, warmly, in this order:\n"
        "1. Say plainly that it is booked, with the restaurant, the date, the time "
        "(as '7 PM', not 19:00) and the party size.\n"
        "2. If `perk_applied` is set, name that perk and say it has been applied — "
        "it is why they chose this table, and leaving it out reads as though it "
        "quietly didn't.\n"
        "3. One line on what people order there, taken from `dining_highlights` and "
        "attributed ('diners keep mentioning…'). If there is nothing there, say the "
        "menu wasn't published online rather than inventing a dish.\n"
        "4. Two short practical tips — arriving a few minutes early, giving the "
        "reservation name, mentioning any dietary need or the perk at the table.\n"
        "5. The confirmation id, and that it has been emailed to them.\n\n"
        "Photos and a menu card are already on screen below your reply. Never paste "
        "image URLs.)"
    )))
    return said


def press_change_my_mind(session: ConciergeSession) -> str:
    """The guest declined at the gate. Nothing is reserved, and the trail says so."""
    pending = session.pending_reservation or {}
    session.pending_reservation = None
    session.trail.record(
        "reservation_gate", actor=None, stage="declined",
        restaurant=pending.get("restaurant"), date=pending.get("date"),
        time=pending.get("time"),
    )
    return "Actually, hold on — I'd like to change something before we book."


def start_session(member_id: str, confirm_in_ui: bool = False) -> ConciergeSession:
    """Build a session, load any saved profile, and seed the system prompt."""
    profile = profile_memory.load(member_id)
    session = ConciergeSession(
        member_id=profile_memory.resolve_key(member_id),
        profile=profile,
        # A recognized guest needs no "have we met?" — we already have their file.
        asked_returning=bool((profile or {}).get("email")),
        confirm_in_ui=confirm_in_ui,
    )
    # Which surface this is decides where the guest's yes comes from, and the model
    # has no way to see that for itself. Told here rather than hard-wired into the
    # brief because the brief is one file serving both: on the web app the Reserve
    # card is the question, so asking it in prose first makes the guest confirm the
    # same table twice; in the terminal there is no card, so the spoken yes is the
    # only gate there is and must still be asked for.
    surface = (
        "This surface shows a **Reserve card**: `book_table` will NOT book, it puts "
        "the reservation on screen for the guest to press Reserve. Do not ask for a "
        "spoken yes before calling it — the card is the ask (step 6a)."
        if confirm_in_ui else
        "This surface is a plain terminal with no Reserve card: `book_table` books "
        "immediately. Read the details back and get a spoken yes BEFORE calling it "
        "(step 6a)."
    )
    # Ground the model in the real date so it never invents a calendar date — it
    # should pass the guest's own wording ("Friday") and let the tools resolve it.
    today = date.today()
    context = (
        f"Today is {today:%A}, {today.isoformat()}. When calling a tool that takes a "
        "date, pass the guest's own words (e.g. 'Friday', 'next Saturday'); the "
        "system resolves them against today. NEVER invent a specific calendar date.\n\n"
        f"## This surface\n{surface}\n\n"
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
            "Add OPENAI_API_KEY to your .env, then run `uv run python -m table_for_four "
            "chat` again.\n(The one-shot booking demo still runs offline: `uv run python "
            "-m table_for_four`.)"
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
