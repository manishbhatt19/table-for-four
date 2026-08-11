"""Menu cards — a generated, cuisine-themed graphic for the chat UI.

Photos scraped off the web are honest but scruffy: inconsistent crops, mismatched
colour, the odd interior shot where you wanted food. A *card* fixes the frame. For
each restaurant we render one designed graphic — a themed header, the dish
highlights we actually retrieved, and the perk if there is one — so every
recommendation looks composed rather than assembled from whatever the internet
had lying around.

Cards are plain SVG rendered to a `data:` URI: no image library, no network, no
files on disk, and they render identically offline and live. There are six
themes; every cuisine maps onto one of them, with a neutral fall-back for
anything unrecognised.

The dish lines come from retrieved text, never from invention — see
`extract_dishes`. A card with no retrieved dishes simply shows fewer lines.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

CARD_WIDTH = 640
HEADER_HEIGHT = 99   # title block plus its accent rule
ROW_HEIGHT = 34
PERK_BAND = 64
MAX_DISHES = 4
DISH_CHARS = 44  # a dish line longer than this is trimmed rather than wrapped


# --- Themes ------------------------------------------------------------------
# Six palettes, each a (background, panel, ink, accent, motif) set. `motif` is a
# single glyph drawn large and faint behind the text — enough to signal the
# cuisine at a glance without shipping any image assets.

THEMES: dict[str, dict[str, str]] = {
    "mediterranean": {"bg": "#2e1f18", "panel": "#3b2a20", "ink": "#f6ece2",
                      "accent": "#e0a458", "motif": "🍝"},
    "french":        {"bg": "#1f2536", "panel": "#2a3145", "ink": "#eef1f8",
                      "accent": "#c9a227", "motif": "🥖"},
    "east_asian":    {"bg": "#231d28", "panel": "#312838", "ink": "#f4eef7",
                      "accent": "#e2606d", "motif": "🍣"},
    "latin":         {"bg": "#2b1d1a", "panel": "#3a2823", "ink": "#fdf0e6",
                      "accent": "#f2842f", "motif": "🌮"},
    "spice":         {"bg": "#2d1512", "panel": "#3d1f19", "ink": "#fdf0e8",
                      "accent": "#d4553a", "motif": "🍛"},
    "green":         {"bg": "#1b2620", "panel": "#26342c", "ink": "#eef6f0",
                      "accent": "#7fbf7f", "motif": "🌿"},
    "default":       {"bg": "#22262b", "panel": "#2f343b", "ink": "#f0f2f5",
                      "accent": "#9fb3c8", "motif": "🍽"},
}

# Which theme a cuisine wears. Keys are matched as substrings against the
# restaurant's cuisine/type, so "italian_restaurant" and "italian" both land.
_CUISINE_THEME: list[tuple[str, str]] = [
    ("italian", "mediterranean"), ("pizza", "mediterranean"), ("greek", "mediterranean"),
    ("french", "french"), ("bistro", "french"), ("steak", "french"),
    ("japanese", "east_asian"), ("sushi", "east_asian"), ("ramen", "east_asian"),
    ("chinese", "east_asian"), ("korean", "east_asian"),
    ("mexican", "latin"), ("taco", "latin"), ("spanish", "latin"), ("latin", "latin"),
    ("indian", "spice"), ("thai", "spice"), ("curry", "spice"), ("vietnamese", "spice"),
    ("vegan", "green"), ("vegetarian", "green"), ("plant", "green"), ("salad", "green"),
]


def theme_for(cuisine: str | None) -> dict[str, str]:
    """Pick a theme from a cuisine or Google place type. Never raises."""
    text = (cuisine or "").lower()
    for token, theme in _CUISINE_THEME:
        if token in text:
            return THEMES[theme]
    return THEMES["default"]


# --- Dish extraction ---------------------------------------------------------

# Menus on the web read as a run of "dish … price, dish … price": "Beef
# Carpaccio, Goldbar Squash, Horseradish $24" or "Salmon Crudo with citrus miso
# cream : 33.00". So the dish is whatever sits *between* two prices.
#
# A price must carry a currency mark or decimal places. Bare integers are not
# prices — that's how a street number, a zip code or a phone number ends up
# looking like the cost of dinner.
_PRICE_TOKEN = re.compile(r"[$£€]\s?\d{1,4}(?:\.\d{2})?|\b\d{1,3}\.\d{2}\b")

# Words that mark a phrase as page furniture rather than something you can eat.
_NOT_A_DISH = re.compile(
    r"\b(street|ave|avenue|road|suite|floor|phone|tel|reservation|hours|open|"
    r"closed|monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"copyright|delivery|catering|gift card|est|since|skip to|logo|instagram|"
    r"facebook|subscribe|newsletter|click|privacy)\b",
    re.I,
)

_MAX_SEGMENT = 72  # longer than this and it's prose, not a menu line


# Menus are printed under section headings, and scraping runs them straight into
# the first dish: "Mains Grilled Cauliflower", "To Start Oysters".
_SECTION_HEADING = re.compile(
    r"^(mains?|starters?|appetiz?ers?|to start|to begin|sides?|desserts?|"
    r"entrées?|entrees?|first course|second course|small plates|large plates|"
    r"raw bar|for the table|specials?|lunch|dinner|brunch)\b[\s:–—-]*",
    re.I,
)


def _clean_dish(segment: str) -> str | None:
    """Tidy one between-prices segment into a dish line, or reject it."""
    dish = " ".join(segment.replace("\\", " ").replace("*", " ").split())
    dish = _SECTION_HEADING.sub("", dish).strip(" ,;:-–—•|/·")
    # Menu runs often leave a connector at the front ("with citrus…"); start the
    # line at the first capital so it reads as a name.
    if dish and not dish[0].isupper():
        cut = re.search(r"[A-Z]", dish)
        dish = dish[cut.start():] if cut else ""
        dish = dish.strip(" ,;:-–—•|/·")
    if not (6 <= len(dish) <= _MAX_SEGMENT):
        return None
    if any(ch.isdigit() for ch in dish):  # addresses, opening hours, phone numbers
        return None
    if _NOT_A_DISH.search(dish):
        return None
    return dish if len(dish) <= DISH_CHARS else dish[: DISH_CHARS - 1] + "…"


def extract_dishes(highlights: list[dict[str, Any]], limit: int = MAX_DISHES) -> list[str]:
    """Pull dish names out of retrieved snippets.

    Strictly extractive: every line returned is a substring of text we actually
    retrieved, so a card can't invent a dish the restaurant doesn't serve. When
    the snippets are navigation chrome rather than a menu, this correctly returns
    nothing and the card renders without dish lines.
    """
    dishes: list[str] = []
    seen: set[str] = set()
    for h in highlights or []:
        text = " ".join((h.get("snippet") or "").split())
        cursor = 0
        for price in _PRICE_TOKEN.finditer(text):
            dish = _clean_dish(text[cursor:price.start()])
            cursor = price.end()
            if not dish or dish.lower() in seen:
                continue
            seen.add(dish.lower())
            dishes.append(dish)
            if len(dishes) >= limit:
                return dishes
    return dishes


# --- Rendering ---------------------------------------------------------------

def _esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _fit(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def build_menu_card(
    restaurant: str,
    cuisine: str | None = None,
    dishes: list[str] | None = None,
    perk: str | None = None,
    perk_is_sample: bool = False,
    subtitle: str | None = None,
) -> str:
    """Render a menu card as an SVG `data:` URI, ready for an <img src>."""
    t = theme_for(cuisine)
    dishes = (dishes or [])[:MAX_DISHES]

    # Height follows the content: a two-dish card shouldn't carry two dishes'
    # worth of empty space below them.
    body_rows = max(len(dishes), 1)
    height = HEADER_HEIGHT + 40 + body_rows * ROW_HEIGHT + (PERK_BAND if perk else 14)

    rows: list[str] = []
    y = HEADER_HEIGHT + 56
    for dish in dishes:
        rows.append(
            f'<circle cx="52" cy="{y - 5}" r="3" fill="{t["accent"]}"/>'
            f'<text x="68" y="{y}" font-family="Georgia,serif" font-size="19" '
            f'fill="{t["ink"]}">{_esc(dish)}</text>'
        )
        y += ROW_HEIGHT
    if not dishes:
        rows.append(
            f'<text x="52" y="{y}" font-family="Georgia,serif" font-size="18" '
            f'font-style="italic" fill="{t["ink"]}" opacity="0.6">'
            "Ask about tonight&#39;s specials when you arrive</text>"
        )

    perk_band = ""
    if perk:
        label = "Sample partner offer" if perk_is_sample else "Perk included"
        perk_band = (
            f'<rect x="0" y="{height - PERK_BAND}" width="{CARD_WIDTH}" height="{PERK_BAND}" '
            f'fill="{t["accent"]}"/>'
            f'<text x="40" y="{height - 38}" font-family="Segoe UI,Helvetica,sans-serif" '
            f'font-size="11" letter-spacing="1.6" fill="{t["bg"]}" opacity="0.75">'
            f'{_esc(label.upper())}</text>'
            f'<text x="40" y="{height - 17}" font-family="Segoe UI,Helvetica,sans-serif" '
            f'font-size="17" font-weight="600" fill="{t["bg"]}">'
            f'{_esc(_fit(perk, 52))}</text>'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{CARD_WIDTH}" '
        f'height="{height}" viewBox="0 0 {CARD_WIDTH} {height}">'
        f'<rect width="{CARD_WIDTH}" height="{height}" fill="{t["bg"]}"/>'
        # Motif: large, faint, bottom-right — texture, not decoration you read.
        f'<text x="{CARD_WIDTH - 34}" y="{height - (PERK_BAND if perk else 0) - 18}" '
        f'text-anchor="end" font-size="118" opacity="0.08">{t["motif"]}</text>'
        f'<rect x="0" y="0" width="{CARD_WIDTH}" height="96" fill="{t["panel"]}"/>'
        f'<rect x="0" y="96" width="{CARD_WIDTH}" height="3" fill="{t["accent"]}"/>'
        f'<text x="40" y="46" font-family="Georgia,serif" font-size="27" '
        f'fill="{t["ink"]}">{_esc(_fit(restaurant, 34))}</text>'
        f'<text x="40" y="72" font-family="Segoe UI,Helvetica,sans-serif" font-size="12" '
        f'letter-spacing="2.2" fill="{t["accent"]}">'
        f'{_esc((subtitle or "MENU HIGHLIGHTS").upper())}</text>'
        + "".join(rows)
        + perk_band
        + "</svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def highlight_note(dishes: list[str], perk: str | None = None) -> str:
    """The one-line 'what people order here' note shown beside the card.

    Built from the extracted dishes rather than written by the model, so it says
    only what was retrieved — and so it appears even when the model's own reply
    skims past it. Reads as a quiet aside, which is why the UI sets it in italic
    next to the card instead of burying it in the chat prose.
    """
    named = [d.rstrip("…").strip() for d in (dishes or [])[:3] if d]
    if not named:
        return (
            "Their menu isn't published online — worth asking what's good "
            "when you arrive."
        )
    if len(named) == 1:
        listed = named[0]
    elif len(named) == 2:
        listed = f"{named[0]} and {named[1]}"
    else:
        listed = f"{named[0]}, {named[1]} and {named[2]}"
    note = f"Diners keep mentioning {listed} — worth a look when you order."
    if perk:
        note += f" Your perk, {perk}, applies at the table."
    return note


def card_for_restaurant(
    restaurant: str,
    cuisine: str | None,
    highlights: list[dict[str, Any]],
    perk: str | None = None,
    perk_is_sample: bool = False,
) -> dict[str, Any]:
    """Build the card and describe it, in the same shape as a retrieved photo."""
    dishes = extract_dishes(highlights)
    caption = (
        f"{restaurant} — {', '.join(dishes[:2])}" if dishes else f"{restaurant} — menu card"
    )
    return {
        "url": build_menu_card(restaurant, cuisine, dishes, perk, perk_is_sample),
        "description": _fit(caption, 110),
        "source": "menu card",
        "dishes": dishes,
        "note": highlight_note(dishes, perk),
    }
