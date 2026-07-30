"""Restaurant search MCP server.

Exposes a single curated tool, `search_restaurants`, backed by Google Places
API (New). When no API key is configured it serves an offline fixture whose
shape mirrors the live API, so the same normalization path runs in both modes.

Run standalone (stdio transport):
    uv run mcp_servers/search_server.py

Inspect interactively:
    uv run mcp dev mcp_servers/search_server.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# --- Configuration -----------------------------------------------------------

PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
PLACES_TEXT_SEARCH_URL = "https://places.googleapis.com/v1/places:searchText"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "places_sample.json"

# Field mask controls both the response shape AND the billing tier on the live
# API. We request only what the concierge actually reasons over -- nothing more.
PLACES_FIELD_MASK = ",".join(
    f"places.{f}"
    for f in (
        "id",
        "displayName",
        "formattedAddress",
        "primaryType",
        "types",
        "priceLevel",
        "rating",
        "userRatingCount",
        "currentOpeningHours.openNow",
        "location",
        "websiteUri",
        "nationalPhoneNumber",
    )
)

# Places API (New) returns price as an enum; we normalize to 0-4 for reasoning.
_PRICE_ENUM_TO_INT = {
    "PRICE_LEVEL_FREE": 0,
    "PRICE_LEVEL_INEXPENSIVE": 1,
    "PRICE_LEVEL_MODERATE": 2,
    "PRICE_LEVEL_EXPENSIVE": 3,
    "PRICE_LEVEL_VERY_EXPENSIVE": 4,
}

mcp = FastMCP("restaurant-search")


# --- Normalization (shared by live + fixture paths) --------------------------

def _normalize_place(raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw Places result into the concierge's clean restaurant shape."""
    return {
        "place_id": raw.get("id"),
        "name": (raw.get("displayName") or {}).get("text"),
        "address": raw.get("formattedAddress"),
        "primary_type": raw.get("primaryType"),
        "types": raw.get("types", []),
        "price_level": _PRICE_ENUM_TO_INT.get(raw.get("priceLevel", "")),
        "rating": raw.get("rating"),
        "rating_count": raw.get("userRatingCount"),
        "open_now": (raw.get("currentOpeningHours") or {}).get("openNow"),
        "location": raw.get("location"),
        "website": raw.get("websiteUri"),
        "phone": raw.get("nationalPhoneNumber"),
    }


# A few cuisines whose Google `type` doesn't literally contain the requested word,
# so a hard cuisine filter still matches them. Everything else matches by substring.
_CUISINE_ALIASES = {
    "sushi": ("sushi", "japanese"),
    "japanese": ("japanese", "sushi"),
}


def _matches_cuisine(r: dict[str, Any], cuisine: str | None) -> bool:
    """True if the restaurant plausibly serves the requested cuisine.

    Matches the cuisine word (or an alias) against the place's Google `types`,
    `primary_type`, or name — e.g. "italian" keeps `italian_restaurant` but drops
    `french_restaurant`. With no cuisine given, everything passes.
    """
    if not cuisine:
        return True
    tokens = _CUISINE_ALIASES.get(cuisine.lower(), (cuisine.lower(),))
    haystack = " ".join(
        [r.get("primary_type") or "", " ".join(r.get("types") or []), r.get("name") or ""]
    ).lower()
    return any(tok in haystack for tok in tokens)


def _passes_filters(
    r: dict[str, Any],
    *,
    cuisine: str | None,
    max_price_level: int | None,
    min_rating: float | None,
    open_now: bool | None,
) -> bool:
    if not _matches_cuisine(r, cuisine):
        return False
    if max_price_level is not None and r["price_level"] is not None:
        if r["price_level"] > max_price_level:
            return False
    if min_rating is not None and r["rating"] is not None:
        if r["rating"] < min_rating:
            return False
    if open_now is True and r["open_now"] is not True:
        return False
    return True


# --- Data sources ------------------------------------------------------------

def _search_fixture(query: str) -> list[dict[str, Any]]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    places = data.get("places", [])
    q = query.lower()
    # Cheap text relevance so the offline path behaves query-sensitively.
    scored = [
        p
        for p in places
        if not q
        or q in json.dumps(p).lower()
        or any(tok in json.dumps(p).lower() for tok in q.split())
    ]
    return [_normalize_place(p) for p in (scored or places)]


def _search_live(query: str, max_results: int) -> list[dict[str, Any]]:
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": PLACES_API_KEY,
        "X-Goog-FieldMask": PLACES_FIELD_MASK,
    }
    body = {
        "textQuery": query,
        "maxResultCount": max_results,
        "includedType": "restaurant",
    }
    resp = httpx.post(
        PLACES_TEXT_SEARCH_URL, headers=headers, json=body, timeout=15.0
    )
    resp.raise_for_status()
    return [_normalize_place(p) for p in resp.json().get("places", [])]


# --- Tool --------------------------------------------------------------------

@mcp.tool()
def search_restaurants(
    query: str,
    cuisine: str | None = None,
    location: str | None = None,
    max_price_level: int | None = None,
    min_rating: float | None = None,
    open_now: bool | None = None,
    max_results: int = 8,
) -> dict[str, Any]:
    """Search for restaurants matching a dining request.

    Args:
        query: Free-text dining request (e.g. "Italian dinner near Midtown").
        cuisine: Optional cuisine (e.g. "italian"). Added to the query AND applied
            as a hard filter — results must match the cuisine (by Google type/name).
        location: Optional location hint appended to the query (e.g. "Midtown NYC").
        max_price_level: Keep results at or below this price tier (0=free .. 4=very expensive).
        min_rating: Keep results at or above this Google rating (0.0-5.0).
        open_now: If true, keep only places currently open.
        max_results: Maximum number of results to return (1-20).

    Returns:
        A dict with `source` ("live"|"fixture"), the effective `query`, and a
        `results` list of normalized restaurants. `source` is included so the
        governance/audit layer can record whether a booking decision rested on
        live or mock data.
    """
    max_results = max(1, min(max_results, 20))
    parts = [query]
    if cuisine:
        parts.append(cuisine)
    if location:
        parts.append(f"near {location}")
    effective_query = " ".join(parts).strip()

    if PLACES_API_KEY:
        source = "live"
        results = _search_live(effective_query, max_results)
    else:
        source = "fixture"
        results = _search_fixture(effective_query)

    results = [
        r
        for r in results
        if _passes_filters(
            r,
            cuisine=cuisine,
            max_price_level=max_price_level,
            min_rating=min_rating,
            open_now=open_now,
        )
    ][:max_results]

    return {
        "source": source,
        "query": effective_query,
        "result_count": len(results),
        "results": results,
    }


if __name__ == "__main__":
    mcp.run()
