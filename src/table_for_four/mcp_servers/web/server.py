"""Web-highlights MCP server — live menu highlights and photos via Tavily.

Exposes a single curated tool, `lookup_dining_highlights`, that answers the
questions the perks store and the reservation backend can't: *what's good here?*
and *what does it look like?* It searches the public web through Tavily and
returns a small, cited set of highlights plus a few images.

Like `search_server`, it runs live when `TAVILY_API_KEY` is set and falls back to
an offline fixture otherwise, so the whole journey still demos with no key. In
fixture mode the images are locally generated placeholder graphics — nothing is
hotlinked, and nothing pretends to be a real photo.

Two deliberate constraints, both about honesty:

* **Web content is never presented as the restaurant's own.** Every highlight
  carries the domain it came from, and the payload carries a `disclaimer` the
  concierge is instructed to honour.
* **Searches are scoped to the restaurant's own site when we know it** (Places
  gives us `websiteUri`), falling back to the open web only if that returns
  nothing — the same "narrow first, widen if empty" shape the search server uses
  for its cuisine filter.

Run standalone (stdio transport):
    uv run mcp_servers/web_server.py

Inspect interactively:
    uv run mcp dev mcp_servers/web_server.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlsplit

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

# --- Configuration -----------------------------------------------------------

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()
TAVILY_SEARCH_URL = "https://api.tavily.com/search"
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "web_highlights_sample.json"

# Kept small on purpose: this text is read back to a chatty model, so every extra
# result is tokens spent for a detail no guest asked for.
MAX_HIGHLIGHTS = 3
MAX_IMAGES = 3
SNIPPET_CHARS = 400
REQUEST_TIMEOUT = 20.0

# Measured against live restaurant sites, counting how much real menu text came
# back: `fast` is quickest but returns mostly site chrome ("powered by ...",
# nav links) rather than dishes; `advanced` costs 2 credits for an inconsistent
# gain. `basic` is the documented 1-credit tier and the one that actually returns
# priced menu items, which is the entire point of the lookup.
SEARCH_DEPTH = "basic"

DISCLAIMER = (
    "Menu details and photos come from the public web, not from the restaurant's "
    "official feed. Mention them as 'what people are saying' and never state them "
    "as guaranteed — menus change."
)

mcp = FastMCP("dining-web-highlights")


# --- Helpers -----------------------------------------------------------------

def _domain(url: str) -> str:
    """The bare host of a URL, without `www.` — what we credit a snippet to."""
    host = urlsplit(url or "").netloc.lower()
    return host[4:] if host.startswith("www.") else host


_PALETTE = [
    ("#f2e9df", "#8a6a4a"), ("#e8eee9", "#4e7a5c"), ("#eceaf3", "#5f5a86"),
    ("#f6ece9", "#a2604f"), ("#eaf0f4", "#4a6f8a"),
]


def _placeholder_image(label: str) -> str:
    """A self-contained SVG data URI standing in for a photo in offline mode.

    Deterministic (same label, same colours) and obviously a placeholder, so an
    offline demo shows the layout working without pretending to have a real
    photograph of a real dish.
    """
    bg, fg = _PALETTE[sum(ord(c) for c in label) % len(_PALETTE)]
    # Wrap the label onto at most two lines so long dish names stay readable.
    words, lines, current = label.split(), [], ""
    for word in words:
        if len(current) + len(word) + 1 > 18 and current:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
        if len(lines) == 2:
            break
    if current and len(lines) < 2:
        lines.append(current)
    spans = "".join(
        f'<tspan x="160" dy="{0 if i == 0 else 24}">{_xml_escape(t)}</tspan>'
        for i, t in enumerate(lines)
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="200">'
        f'<rect width="320" height="200" fill="{bg}"/>'
        f'<circle cx="160" cy="78" r="30" fill="none" stroke="{fg}" stroke-width="2"/>'
        f'<circle cx="160" cy="78" r="18" fill="none" stroke="{fg}" stroke-width="1.5"/>'
        f'<text x="160" y="140" text-anchor="middle" font-family="Segoe UI,Helvetica,sans-serif"'
        f' font-size="15" fill="{fg}">{spans}</text>'
        f'<text x="160" y="184" text-anchor="middle" font-family="Segoe UI,Helvetica,sans-serif"'
        f' font-size="10" fill="{fg}" opacity="0.7">illustrative placeholder</text>'
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + quote(svg)


def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _clean(text: str) -> str:
    """Collapse whitespace and trim a web snippet to a readable length."""
    flat = re.sub(r"\s+", " ", text or "").strip()
    return flat[: SNIPPET_CHARS - 1] + "…" if len(flat) > SNIPPET_CHARS else flat


# Page-derived images include everything a page carries, not just photographs:
# social icons, logos, spacers, map tiles. They look broken in a photo strip.
_NOT_A_PHOTO = re.compile(
    r"\b(icon|logo|sprite|favicon|pixel|badge|avatar|spacer|arrow|button|"
    r"thumbnail|map)\b",
    re.I,
)
_ASSET_HOSTS = ("gstatic.com", "maps.wikimedia.org", "w3.org", "fonts.googleapis.com")


# A word-boundary match misses the way real sites name these files —
# "faviconnew.webp", "katz_WEBicons_Shipping.png", "logopos.webp" all sailed
# through and landed in a photo strip. These tokens match as substrings instead.
# "map" is deliberately not among them: "mapo tofu" and "maple" are food.
_FURNITURE = re.compile(r"(favicon|sprite|logo|icons)", re.I)


def _is_photo(url: str, description: str) -> bool:
    if url.lower().split("?")[0].endswith(".svg"):  # vector = furniture, not a photo
        return False
    if any(host in _domain(url) for host in _ASSET_HOSTS):
        return False
    if _FURNITURE.search(url):
        return False
    return not _NOT_A_PHOTO.search(f"{url} {description}")


def _normalize_images(raw: list[Any], fallback_caption: str = "") -> list[dict[str, Any]]:
    """Tavily returns images as bare URLs, or as objects once descriptions are on.

    Only the top-level image search gets described; images lifted off a result page
    arrive as bare URLs. `fallback_caption` (the title of the page they came from)
    stands in for those, so every photo still has a caption and alt text.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw or []:
        url = item if isinstance(item, str) else (item or {}).get("url", "")
        description = "" if isinstance(item, str) else (item or {}).get("description", "")
        if not url or not url.startswith("http") or url in seen:
            continue
        if not _is_photo(url, description or fallback_caption):
            continue
        seen.add(url)
        out.append({
            "url": url,
            "description": _clean(description) or _clean(fallback_caption) or "Photo from the web",
            "source": _domain(url),
        })
        if len(out) >= MAX_IMAGES:
            break
    return out


# --- The restaurant's own site -----------------------------------------------
#
# Scoping the Tavily *search* to the restaurant's domain was only ever half the
# job: `include_domains` filters which pages match, but the images that come back
# are whatever the index happens to hold, and the separate image query ignores
# the domain entirely. So a guest kept being shown another branch, or a namesake
# in another city.
#
# Fetching the site ourselves fixes that at the source. What a restaurant puts in
# its own `og:image` is the picture it chose to represent itself — and it is on
# their domain, so it cannot be somewhere else's dining room. That is the same
# structural guarantee Places photos have, from the other direction.

# Deliberately short: this sits inside a live conversation, and a slow site must
# cost the guest a second, not their dining tips.
SITE_TIMEOUT = 6.0
SITE_MAX_BYTES = 600_000  # a head-and-shoulders read; we only need <head> really
SITE_AGENT = "TableForFour/1.0 (restaurant concierge; +https://example.invalid/bot)"

_OG_IMAGE = re.compile(
    r"""<meta[^>]+(?:property|name)\s*=\s*["'](?:og:image(?::url)?|twitter:image)["'][^>]*>""",
    re.I,
)
_CONTENT_ATTR = re.compile(r"""content\s*=\s*["']([^"']+)["']""", re.I)
_JSON_LD = re.compile(
    r"""<script[^>]+type\s*=\s*["']application/ld\+json["'][^>]*>(.*?)</script>""",
    re.I | re.S,
)
_IMG_SRC = re.compile(r"""<img[^>]+src\s*=\s*["']([^"']+)["']""", re.I)

# Found by running this against real restaurant sites, which is the only way this
# kind of thing gets found: Gramercy Tavern's page yielded its own homepage URL
# and a Facebook tracking pixel alongside the actual photograph.
_TRACKER_HOSTS = (
    "facebook.com", "facebook.net", "google-analytics.com", "googletagmanager.com",
    "doubleclick.net", "bing.com", "hotjar.com", "segment.io", "pinterest.com",
    "tiktok.com", "snapchat.com", "criteo.com", "clarity.ms",
)
# Restaurant sites overwhelmingly sit on a handful of hosted platforms, whose
# image URLs carry no file extension at all.
_IMAGE_HOSTS = (
    "getbento.com", "squarespace-cdn.com", "squarespace.com", "wixstatic.com",
    "cloudinary.com", "imgix.net", "shopify.com", "cloudfront.net",
    "googleusercontent.com", "resengo.com", "toasttab.com",
)
_IMAGE_EXT = re.compile(r"\.(jpe?g|png|webp|avif|gif)(\?|#|$)", re.I)


def _is_image_url(url: str, *, declared: bool) -> bool:
    """Is this plausibly a photograph rather than a page, a pixel, or a script?

    `declared` marks a URL the site itself nominated as its image (og:image or
    schema.org). Those get the benefit of the doubt about their shape, because
    hosted platforms serve perfectly good photos from extensionless URLs. A URL
    merely scraped off an <img> tag has to look like an image to count.
    """
    host = _domain(url)
    if any(tracker in host for tracker in _TRACKER_HOSTS):
        return False
    path = urlsplit(url).path
    if path in ("", "/"):
        return False  # a link to the site itself, not a picture on it
    if declared:
        return True
    return bool(_IMAGE_EXT.search(url)) or any(h in host for h in _IMAGE_HOSTS)


def _same_site(requested: str, final: str) -> bool:
    """Did the fetch end up on the domain we asked for?

    Compared on the last two labels, so `example.com` → `www.example.com` and a
    plain http → https upgrade both count as the same place, while a redirect off
    to somebody else's domain does not.
    """
    def registrable(url: str) -> str:
        return ".".join(_domain(url).split(".")[-2:])

    return bool(registrable(requested)) and registrable(requested) == registrable(final)


def _absolute(url: str, base: str) -> str:
    """Resolve a possibly-relative image URL against the page it came from."""
    try:
        return urljoin(base, (url or "").strip())
    except ValueError:
        return ""


def _json_ld_images(html: str) -> list[str]:
    """Images declared in schema.org markup, which restaurants use for menus."""
    found: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            value = node.get("image")
            if isinstance(value, str):
                found.append(value)
            elif isinstance(value, list):
                found.extend(v for v in value if isinstance(v, str))
            elif isinstance(value, dict) and isinstance(value.get("url"), str):
                found.append(value["url"])
            for child in node.values():
                walk(child)
        elif isinstance(node, list):
            for child in node:
                walk(child)

    for block in _JSON_LD.findall(html):
        try:
            walk(json.loads(block))
        except (ValueError, TypeError):
            continue  # a malformed block is common and not worth failing over
    return found


def site_images(website: str, restaurant_name: str = "") -> list[dict[str, Any]]:
    """Photos published by the restaurant itself, read from its own page.

    Order is by how deliberate the choice was: `og:image` is what the owner
    picked to represent the page, schema.org `image` is what they declared to
    search engines, and only then the page's own `<img>` tags, which are as
    likely to be a logo as a plate and lean on `_is_photo` to sort it out.

    Every failure here is swallowed. A restaurant with a dead site, a JS-only
    page, or a slow host should cost the guest nothing — the caller simply falls
    through to the sources it already had, which is also what happens offline.
    """
    if not (website or "").strip():
        return []
    try:
        resp = httpx.get(
            website,
            timeout=SITE_TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": SITE_AGENT, "Accept": "text/html,*/*"},
        )
        resp.raise_for_status()
        if "html" not in resp.headers.get("content-type", "").lower():
            return []
        html = resp.text[:SITE_MAX_BYTES]
    except (httpx.HTTPError, UnicodeDecodeError, ValueError):
        return []

    base = str(resp.url)
    # A restaurant that closed and let its domain lapse now serves whoever bought
    # it. The Spotted Pig's address redirects to a parked site, and without this
    # a guest would have been shown a squatter's banners as the restaurant's own
    # photographs — the exact failure this whole function exists to prevent, just
    # arriving by a different road. If the redirect left the site we asked for,
    # we cannot vouch for anything on it.
    if not _same_site(website, base):
        return []
    # (url, was it the site's own nomination?) — order is by how deliberate the
    # choice was, so the picture the owner picked leads.
    candidates: list[tuple[str, bool]] = []
    for tag in _OG_IMAGE.findall(html):
        match = _CONTENT_ATTR.search(tag)
        if match:
            candidates.append((match.group(1), True))
    candidates += [(url, True) for url in _json_ld_images(html)]
    candidates += [(url, False) for url in _IMG_SRC.findall(html)]

    caption = f"From {restaurant_name}'s own site" if restaurant_name else "From the restaurant's own site"
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw, declared in candidates:
        url = _absolute(raw, base)
        if not url.startswith("http"):
            continue
        # Deduped on the path, not the whole URL: og:image and twitter:image are
        # usually the same photograph with different resize parameters, and a
        # strip that shows a guest the same room twice looks broken.
        key = f"{_domain(url)}{urlsplit(url).path}"
        if key in seen:
            continue
        if not _is_image_url(url, declared=declared) or not _is_photo(url, caption):
            continue
        seen.add(key)
        out.append({"url": url, "description": caption, "source": _domain(base)})
        if len(out) >= MAX_IMAGES:
            break
    return out


# --- Data sources ------------------------------------------------------------

def _lookup_fixture(place_id: str | None, restaurant_name: str) -> dict[str, Any]:
    data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    entry = (data.get("restaurants") or {}).get(place_id or "")
    if entry is None:
        # Fall back to a name match, then to the generic entry, so an unknown
        # restaurant still returns something usefully shaped.
        wanted = (restaurant_name or "").lower()
        entry = next(
            (
                e for e in (data.get("restaurants") or {}).values()
                if wanted and wanted in json.dumps(e).lower()
            ),
            data.get("default", {}),
        )
    dishes = entry.get("dishes") or [restaurant_name or "House special"]
    return {
        "highlights": [
            {**h, "snippet": _clean(h.get("snippet", ""))}
            for h in (entry.get("highlights") or [])[:MAX_HIGHLIGHTS]
        ],
        "images": [
            {
                "url": _placeholder_image(dish),
                "description": f"{dish} (illustrative placeholder — no live photo offline)",
                "source": "placeholder",
            }
            for dish in dishes[:MAX_IMAGES]
        ],
    }


def _post(body: dict[str, Any]) -> httpx.Response:
    return httpx.post(
        TAVILY_SEARCH_URL,
        headers={"Authorization": f"Bearer {TAVILY_API_KEY}"},
        json=body,
        timeout=REQUEST_TIMEOUT,
    )


def _search_live(
    query: str, *, include_domains: list[str] | None, want_images: bool
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": query,
        "search_depth": SEARCH_DEPTH,
        "max_results": MAX_HIGHLIGHTS,
        "include_images": want_images,
        # Descriptions are the image captions and the alt text a broken photo
        # falls back to — worth the extra round trip.
        "include_image_descriptions": want_images,
    }
    if include_domains:
        body["include_domains"] = include_domains

    resp = _post(body)
    resp.raise_for_status()
    return resp.json()


# Plenty of restaurant sites publish their menu as a PDF, an image, or inside a
# booking widget, so a domain-scoped search comes back with navigation furniture
# instead of food. These two patterns tell those apart.
_MENU_SIGNAL = re.compile(
    r"\b(appetizer|entr[ée]e|starter|dessert|tasting menu|prix fixe|signature|"
    r"specialit|specialty|classics|known for|must[- ]order|order the|served with|"
    r"dishes|menu features)\b",
    re.I,
)
_PRICE = re.compile(r"\$\s?\d|\b\d{2}\.\d{2}\b")
_CHROME = re.compile(
    r"skip to content|powered by|main content starts here|all rights reserved|"
    r"privacy policy|follow us|cookie",
    re.I,
)


def _content_score(highlights: list[dict[str, Any]]) -> float:
    """How much a result set actually tells us about the food, 0 = nothing.

    Scored per snippet and reduced with `max`, not summed over the whole set: the
    concierge quotes one snippet, so one good one is worth more than three
    mediocre ones — and a single chrome-heavy result shouldn't drag down the good
    one sitting next to it.
    """
    best = 0.0
    for h in highlights:
        text = h.get("snippet", "")
        if not text.strip():
            continue
        score = min(len(text) / 2000, 1.0)  # substance, capped
        if _MENU_SIGNAL.search(text):
            score += 2.0
        if _PRICE.search(text):  # a priced list is the strongest evidence of a menu
            score += 2.0
        if _CHROME.search(text):
            score -= 2.0
        best = max(best, score)
    return best


# Below this, a result set has no clear evidence of dishes and is worth a retry.
MENU_CONTENT_THRESHOLD = 2.0


def _normalize_live(payload: dict[str, Any]) -> dict[str, Any]:
    """Split a Tavily payload into highlights and its two distinct image sources.

    The distinction matters. `page_images` are lifted from the pages that actually
    matched the search, so they depict *this* restaurant. `query_images` come from
    a separate image search that ignores `include_domains` entirely, so for a name
    shared by several branches — or several restaurants — they can easily be
    somewhere else. Page images are therefore always preferred.
    """
    highlights = [
        {
            "title": _clean(r.get("title", "")),
            "snippet": _clean(r.get("content", "")),
            "url": r.get("url", ""),
            "source": _domain(r.get("url", "")),
        }
        for r in (payload.get("results") or [])[:MAX_HIGHLIGHTS]
        if r.get("content")
    ]
    page_images: list[dict[str, Any]] = []
    for r in payload.get("results") or []:
        page_images += _normalize_images(r.get("images") or [], _clean(r.get("title", "")))
    return {
        "highlights": highlights,
        "page_images": page_images[:MAX_IMAGES],
        "query_images": _normalize_images(payload.get("images") or []),
    }


def _merge_images(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Concatenate image groups in priority order, deduped, capped."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        for img in group:
            if img["url"] in seen:
                continue
            seen.add(img["url"])
            out.append(img)
            if len(out) >= MAX_IMAGES:
                return out
    return out


# What the guest is actually asking to see, which is not the same query as "what
# should I order here" — a menu query returns anonymous plated-food stock shots.
_ROOM_WORDS = re.compile(
    r"\b(interior|dining room|inside|ambien|atmosphere|decor|d[ée]cor|space|room|"
    r"venue|terrace|patio|bar|look|looks like|vibe|setting)\b",
    re.I,
)


def _image_query(restaurant_name: str, address: str | None, focus: str) -> str:
    """A query built for photos of *this* restaurant, not of food in general.

    The address is the important part: without it an image search for a common
    name returns whichever branch — or whichever unrelated restaurant — happens to
    be better indexed.
    """
    aspect = "dining room interior" if _ROOM_WORDS.search(focus or "") else "signature dishes"
    return " ".join(
        p for p in [restaurant_name, address, "restaurant", aspect, "photos"] if p
    ).strip()


# --- Tool --------------------------------------------------------------------

@mcp.tool()
def lookup_dining_highlights(
    restaurant_name: str,
    address: str | None = None,
    website: str | None = None,
    place_id: str | None = None,
    focus: str = "menu highlights and signature dishes",
    include_images: bool = True,
) -> dict[str, Any]:
    """Look up what a restaurant is known for, with photos, from the public web.

    Args:
        restaurant_name: The restaurant to look up. Required.
        address: Optional address or area, to disambiguate chains and common names.
        website: Optional official site. When given, the text search is scoped to
            that domain first and only widens to the open web if it comes back
            with nothing about the food.
        place_id: Optional id, used to select the offline fixture entry.
        focus: What to look for — e.g. "menu highlights", "signature dishes",
            "what the room looks like". Wording matters: anything mentioning the
            room, decor or atmosphere switches the photo search from food to
            interiors.
        include_images: Whether to fetch photos alongside the text.

    Returns:
        A dict with `source` ("live"|"fixture"), the `query` used, the search
        `scope` ("official_site"|"open_web"|"fixture"), the `image_query` if a
        separate photo search was needed, up to three cited `highlights`, up to
        three `images`, and a `disclaimer` describing how the material may be
        presented. `source` and the per-item domains are included so the
        governance/audit layer can record what the concierge was told and by whom.
    """
    if not (restaurant_name or "").strip():
        return {
            "source": "none",
            "error": "restaurant_name is required.",
            "highlights": [],
            "images": [],
        }

    query = " ".join(p for p in [restaurant_name, address, focus] if p).strip()

    if not TAVILY_API_KEY:
        data = _lookup_fixture(place_id, restaurant_name)
        scope = "fixture"
        source = "fixture"
    else:
        source = "live"
        site = _domain(website or "")
        scope = "official_site" if site else "open_web"
        data = _normalize_live(
            _search_live(query, include_domains=[site] if site else None,
                         want_images=include_images)
        )
        # The restaurant's own site leads. `og:image` is a picture it chose to
        # represent itself, served from its own domain, so unlike anything a
        # search index returns it cannot be a different branch or a namesake.
        # Then photos Tavily lifted off the pages that actually matched.
        images = _merge_images(site_images(website or "", restaurant_name),
                               data["page_images"])

        # Narrow first, widen if the official site gave us nothing to say. That
        # covers both an empty answer and the commoner case: a site whose menu is
        # a PDF or an image, so the search returns only navigation furniture.
        if site and _content_score(data["highlights"]) < MENU_CONTENT_THRESHOLD:
            wider = _normalize_live(
                _search_live(query, include_domains=None, want_images=include_images)
            )
            if _content_score(wider["highlights"]) > _content_score(data["highlights"]):
                scope = "open_web"
                data = wider
            images = _merge_images(images, wider["page_images"])

        # Still short of photos? The menu query is the wrong one to ask an image
        # search — it returns anonymous plated food. Ask a second, location-anchored
        # question instead, and take page-derived hits over generic ones.
        image_query = ""
        if include_images and len(images) < MAX_IMAGES:
            image_query = _image_query(restaurant_name, address, focus)
            shots = _normalize_live(
                _search_live(image_query, include_domains=None, want_images=True)
            )
            images = _merge_images(images, shots["page_images"], shots["query_images"])

        # Only as a last resort: images from the unscoped image search, which may
        # be a different branch or a namesake entirely.
        data["images"] = _merge_images(images, data["query_images"])
        if image_query:
            data["image_query"] = image_query

    if not include_images:
        data["images"] = []

    return {
        "source": source,
        "scope": scope,
        "restaurant": restaurant_name,
        "query": query,
        "image_query": data.get("image_query"),
        "highlights": data["highlights"],
        "images": data["images"],
        "citations": [h["url"] for h in data["highlights"] if h.get("url")],
        "disclaimer": DISCLAIMER,
    }


if __name__ == "__main__":
    mcp.run()
