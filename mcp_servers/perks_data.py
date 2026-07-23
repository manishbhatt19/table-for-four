"""Synthetic perks (offers/coupons) dataset for the perks RAG layer.

`generate_perks()` returns a deterministic, reproducible set of synthetic perks
keyed to the fixture restaurants (by `place_id`). The data is intentionally
crafted to give the retrieval layer two things to exercise:

* **Semantic signal** in the free-text `blurb` (gluten-free, celebrations,
  groups/families, romantic, business, wine) — what the vector search matches on.
* **Structured metadata** for filtering (`min_party_size`, `expiry`,
  `valid_days`, `dine_in_only`, `perk_type`) — including deliberately **expired**
  and **large-party-gated** perks so the metadata filters are provably working.

All perks are **synthetic** and labeled as such downstream (`source: "synthetic"`).
They are offers attached to *fictional* fixture restaurants — no real business is
represented.

Run as a script to (re)write the committed seed file:
    uv run mcp_servers/perks_data.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SEED_PATH = Path(__file__).parent / "fixtures" / "perks_seed.json"

# Absolute expiry anchors keep the dataset deterministic: a "valid" perk never
# silently expires and an "expired" one never comes back, so tests stay stable
# regardless of when they run.
_VALID = "2099-12-31"
_EXPIRED = "2020-01-01"


def _perk(
    perk_id: str,
    place_id: str,
    restaurant_name: str,
    title: str,
    blurb: str,
    *,
    perk_type: str,
    discount_pct: int = 0,
    min_party_size: int = 1,
    dine_in_only: bool = True,
    valid_days: str = "",  # CSV of 3-letter days; "" == any day
    expiry: str = _VALID,
    active: bool = True,
) -> dict[str, Any]:
    return {
        "perk_id": perk_id,
        "place_id": place_id,
        "restaurant_name": restaurant_name,
        "title": title,
        "blurb": blurb,
        "perk_type": perk_type,  # percent_off | freebie | prix_fixe
        "discount_pct": discount_pct,
        "min_party_size": min_party_size,
        "dine_in_only": dine_in_only,
        "valid_days": valid_days,
        "expiry": expiry,
        "active": active,
    }


def generate_perks() -> list[dict[str, Any]]:
    """Return the deterministic synthetic perks dataset."""
    perks: list[dict[str, Any]] = []

    # --- Osteria Midtown (Italian, moderate) ---------------------------------
    perks += [
        _perk(
            "perk-osteria-01", "fixture-osteria-1", "Osteria Midtown",
            "Weekend Family Feast",
            "Family-style Italian sharing platters of pasta and antipasti — "
            "generous portions built for groups, celebrations, and big tables.",
            perk_type="percent_off", discount_pct=15,
            min_party_size=4, valid_days="Fri,Sat,Sun",
        ),
        _perk(
            "perk-osteria-02", "fixture-osteria-1", "Osteria Midtown",
            "Aperitivo Welcome",
            "A complimentary Aperol spritz to start — a relaxed, cozy aperitivo "
            "hour that suits a casual date night or catching up with a friend.",
            perk_type="freebie", min_party_size=2,
        ),
        _perk(
            "perk-osteria-03", "fixture-osteria-1", "Osteria Midtown",
            "Seasonal Truffle Prix Fixe",
            "A three-course seasonal tasting menu featuring fresh shaved truffle "
            "over handmade pasta — an indulgent seasonal Italian dinner.",
            perk_type="prix_fixe",
        ),
    ]

    # --- Trattoria del Sole (Italian, expensive) -----------------------------
    perks += [
        _perk(
            "perk-trattoria-01", "fixture-trattoria-2", "Trattoria del Sole",
            "Sommelier Wine Pairing",
            "A complimentary regional wine flight paired by our sommelier — an "
            "elegant, romantic upscale dinner for couples who love wine.",
            perk_type="freebie", min_party_size=2,
        ),
        _perk(
            "perk-trattoria-02", "fixture-trattoria-2", "Trattoria del Sole",
            "Restaurant Week Prix Fixe (past)",
            "A special Restaurant Week three-course prix fixe at a reduced price "
            "— a limited-time seasonal promotion.",
            perk_type="percent_off", discount_pct=25, expiry=_EXPIRED,
        ),
        _perk(
            "perk-trattoria-03", "fixture-trattoria-2", "Trattoria del Sole",
            "Private Banquet for Large Parties",
            "A private banquet menu for large groups, corporate dinners, and "
            "milestone celebrations — reserved for bigger parties.",
            perk_type="percent_off", discount_pct=10, min_party_size=6,
        ),
    ]

    # --- Nonna's Gluten-Free Kitchen (Italian/pizza, moderate) ---------------
    perks += [
        _perk(
            "perk-nonna-01", "fixture-pizzeria-3", "Nonna's Gluten-Free Kitchen",
            "Gluten-Free Tasting Flight",
            "A complimentary gluten-free tasting flight from our fully "
            "celiac-safe kitchen — perfect for a birthday or celebration when a "
            "guest needs gluten-free, with no cross-contamination worries.",
            perk_type="freebie", min_party_size=1,
        ),
        _perk(
            "perk-nonna-02", "fixture-pizzeria-3", "Nonna's Gluten-Free Kitchen",
            "Family Gluten-Free Pizza Night",
            "A kid-friendly, gluten-free pizza night for families — a relaxed "
            "weeknight dinner everyone at the table can share safely.",
            perk_type="percent_off", discount_pct=20,
            min_party_size=4, valid_days="Mon,Tue,Wed,Thu",
        ),
    ]

    # --- Le Petit Bistro (French, very expensive) ----------------------------
    perks += [
        _perk(
            "perk-bistro-01", "fixture-bistro-4", "Le Petit Bistro",
            "Date Night Dégustation",
            "A candlelit French tasting menu for two — an intimate, romantic "
            "dégustation ideal for anniversaries and special date nights.",
            perk_type="prix_fixe", min_party_size=2,
        ),
        _perk(
            "perk-bistro-02", "fixture-bistro-4", "Le Petit Bistro",
            "Weekday Business Lunch",
            "An efficient weekday prix-fixe lunch — a quiet, professional setting "
            "well suited to business meetings and working lunches.",
            perk_type="percent_off", discount_pct=15,
            valid_days="Mon,Tue,Wed,Thu,Fri",
        ),
        _perk(
            "perk-bistro-03", "fixture-bistro-4", "Le Petit Bistro",
            "Champagne Celebration Toast",
            "A complimentary champagne toast to mark the occasion — for "
            "anniversaries, engagements, and milestone celebrations.",
            perk_type="freebie", min_party_size=2,
        ),
    ]

    return perks


def write_seed(path: Path = SEED_PATH) -> Path:
    """Write the synthetic perks dataset to the committed seed JSON."""
    perks = generate_perks()
    payload = {
        "_comment": (
            "Synthetic perks dataset (offers/coupons) for fictional fixture "
            "restaurants. Generated deterministically by perks_data.py. All perks "
            "are synthetic and labeled source='synthetic' downstream."
        ),
        "perks": perks,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


if __name__ == "__main__":
    out = write_seed()
    print(f"Wrote {len(generate_perks())} synthetic perks -> {out}")
