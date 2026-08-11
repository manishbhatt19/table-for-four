"""Reasoning helpers with an LLM path and a deterministic heuristic fallback.

Every function takes an optional `llm`; when it is None (no API key, or the caller
forced offline mode) a rule-based version runs instead. This keeps the whole
orchestrator runnable and testable offline, and uses the LLM only for the two
places it most improves the experience: understanding a messy request (`parse`)
and writing the final summary (`narrate`). That's ~2 LLM calls per run.
"""

from __future__ import annotations

import json
import re
from datetime import date, timedelta
from typing import Any

_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}
_DAY_ABBR = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_CUISINES = [
    "italian", "french", "japanese", "sushi", "chinese", "thai", "indian",
    "mexican", "mediterranean", "american", "korean", "spanish",
]
_DIETARY = ["gluten-free", "gluten free", "vegetarian", "vegan", "halal", "kosher", "dairy-free"]


# --- Heuristic parse ---------------------------------------------------------

def _parse_party_size(text: str) -> int | None:
    m = re.search(r"(\d+)\s*(?:people|persons|guests|ppl|pax|diners)", text)
    if m:
        return int(m.group(1))
    m = re.search(r"(?:party of|table for|for)\s+(\d+)", text)
    if m:
        return int(m.group(1))
    for word, n in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b\s*(?:people|persons|guests)", text):
            return n
    return None


def _parse_date(text: str, ref: date) -> tuple[str, str]:
    """Return (iso_date, day_abbr). Defaults to the next Friday if none given."""
    for name, wd in _WEEKDAYS.items():
        if name in text:
            delta = (wd - ref.weekday()) % 7 or 7
            d = ref + timedelta(days=delta)
            return d.isoformat(), _DAY_ABBR[wd]
    if "tomorrow" in text:
        d = ref + timedelta(days=1)
        return d.isoformat(), _DAY_ABBR[d.weekday()]
    # default: upcoming Friday
    delta = (4 - ref.weekday()) % 7 or 7
    d = ref + timedelta(days=delta)
    return d.isoformat(), "Fri"


def _parse_time(text: str) -> str | None:
    m = re.search(r"(\d{1,2}):(\d{2})", text)
    if m:
        return f"{int(m.group(1)):02d}:{m.group(2)}"
    m = re.search(r"(\d{1,2})\s*(am|pm)", text)
    if m:
        hour = int(m.group(1)) % 12
        if m.group(2) == "pm":
            hour += 12
        return f"{hour:02d}:00"
    return None


def resolve_date(text: str, ref: date | None = None) -> tuple[str, str]:
    """Resolve a free-text or ISO date to `(iso_date, day_abbr)`.

    Accepts an explicit `YYYY-MM-DD` or a relative day ("friday", "tomorrow").
    Falls back to the next Friday when nothing is recognized (same rule the
    request parser uses).
    """
    ref = ref or date.today()
    t = (text or "").strip().lower()
    m = re.match(r"(\d{4}-\d{2}-\d{2})", t)
    if m:
        iso = m.group(1)
        return iso, _DAY_ABBR[date.fromisoformat(iso).weekday()]
    return _parse_date(t, ref)


def _heuristic_parse(request: str, ref: date) -> dict[str, Any]:
    text = request.lower()
    iso_date, day = _parse_date(text, ref)

    cuisine = next((c for c in _CUISINES if c in text), None)
    dietary = sorted({d.replace(" ", "-") for d in _DIETARY if d in text})

    location = None
    m = re.search(r"(?:near|in|around)\s+([a-z0-9 ]+?)(?:,|\.|;| for | on | at |$)", text)
    if m:
        location = m.group(1).strip().title()

    max_price = None
    if any(w in text for w in ("cheap", "budget", "affordable", "inexpensive")):
        max_price = 2

    # "top-rated"/"best" sets a high bar the modest fixtures can't clear, which
    # exercises the orchestrator's refine-retry loop (it relaxes and tries again).
    min_rating = None
    if any(w in text for w in ("top-rated", "top rated", "highly rated",
                               "best rated", "5 star", "five star")):
        min_rating = 4.8

    return {
        "cuisine": cuisine,
        "location": location,
        "party_size": _parse_party_size(text) or 2,
        "date": iso_date,
        "day": day,
        "time": _parse_time(text),
        "dietary": dietary,
        "special_requests": ", ".join(dietary) or None,
        "max_price_level": max_price,
        "min_rating": min_rating,
        "open_now": None,
    }


# --- LLM parse ---------------------------------------------------------------

_PARSE_SYSTEM = (
    "You extract structured dining constraints from a request. Return ONLY JSON "
    "with keys: cuisine (str|null), location (str|null), party_size (int), "
    "date (YYYY-MM-DD or null), day (Mon..Sun or null), time (HH:MM 24h or null), "
    "dietary (list of strings), max_price_level (0-4 or null), min_rating "
    "(float or null). Today is {today}. Resolve relative days (e.g. 'Friday') to a "
    "date."
)


def parse(request: str, llm: Any | None = None, ref: date | None = None) -> dict[str, Any]:
    """Parse a free-text request into constraints (LLM if available, else rules)."""
    ref = ref or date.today()
    if llm is None:
        return _heuristic_parse(request, ref)
    try:
        msg = llm.invoke(
            [
                ("system", _PARSE_SYSTEM.format(today=ref.isoformat())),
                ("human", request),
            ]
        )
        raw = msg.content if hasattr(msg, "content") else str(msg)
        data = json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group(0))
        # Backfill anything the model omitted using the heuristic parse.
        base = _heuristic_parse(request, ref)
        base.update({k: v for k, v in data.items() if v is not None})
        base["special_requests"] = ", ".join(base.get("dietary") or []) or None
        return base
    except Exception:
        return _heuristic_parse(request, ref)


# --- Ranking (heuristic; simple and explainable) -----------------------------

def choose_best(
    candidates: list[dict[str, Any]], perks: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Pick the best candidate.

    Prefers a restaurant that has a matching perk (so the personalization is
    visible in the decision), then falls back to rating and review count.
    """
    perked = {p.get("place_id") for p in (perks or [])}
    return max(
        candidates,
        key=lambda r: (
            r["place_id"] in perked,
            r.get("rating") or 0,
            r.get("rating_count") or 0,
        ),
    )


def best_perk_for(place_id: str, perks: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [p for p in perks if p.get("place_id") == place_id]
    if not matches:
        return None
    return max(matches, key=lambda p: p.get("similarity", 0))


# --- Narrative ---------------------------------------------------------------

def _heuristic_narrative(restaurant: dict, perk: dict | None, booking: dict) -> str:
    b = booking.get("booking", booking)
    perk_line = ""
    if perk:
        perk_line = f" Your offer '{perk['title']}' is noted on the reservation."
    return (
        f"Booked {restaurant['name']} for {b['party_size']} on {b['date']} at "
        f"{b['time']} (rating {restaurant.get('rating', 'n/a')}).{perk_line} "
        f"Confirmation {booking.get('confirmation_id', b.get('confirmation_id'))}."
    )


def narrate(
    restaurant: dict, perk: dict | None, booking: dict, llm: Any | None = None
) -> str:
    """One friendly sentence or two summarizing the booking."""
    if llm is None:
        return _heuristic_narrative(restaurant, perk, booking)
    try:
        facts = {
            "restaurant": restaurant.get("name"),
            "rating": restaurant.get("rating"),
            "booking": booking.get("booking", booking),
            "perk": (perk or {}).get("title"),
        }
        msg = llm.invoke(
            [
                (
                    "system",
                    "You are a warm restaurant concierge. In 2 short sentences, "
                    "confirm the booking to the guest. Mention the perk if present. "
                    "Do not invent details beyond the facts given.",
                ),
                ("human", json.dumps(facts)),
            ]
        )
        return (msg.content if hasattr(msg, "content") else str(msg)).strip()
    except Exception:
        return _heuristic_narrative(restaurant, perk, booking)
