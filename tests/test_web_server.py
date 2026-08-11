"""Offline tests for the web-highlights MCP server.

The live path is exercised by monkeypatching the single HTTP call, so nothing
here touches Tavily or the network. The fixture path is exercised as-is: it is
what runs when no `TAVILY_API_KEY` is set, which is how the demo is graded.
"""

import pytest

from table_for_four.mcp_servers.web import server as web_server
from table_for_four.mcp_servers.web.server import lookup_dining_highlights


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Default every test to the keyless (fixture) path unless it opts out."""
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "")


def test_fixture_mode_returns_cited_highlights_and_placeholder_images():
    out = lookup_dining_highlights(
        restaurant_name="Nonna's Gluten-Free Kitchen",
        place_id="fixture-pizzeria-3",
    )
    assert out["source"] == "fixture"
    assert out["scope"] == "fixture"
    assert out["highlights"], "fixture entry should yield highlights"
    assert all(h["url"] for h in out["highlights"])
    assert out["citations"]
    # Offline images are self-contained placeholders, never hotlinked photos.
    assert out["images"]
    assert all(img["url"].startswith("data:image/svg+xml") for img in out["images"])
    assert all(img["source"] == "placeholder" for img in out["images"])


def test_unknown_restaurant_still_returns_usable_shape():
    out = lookup_dining_highlights(restaurant_name="Somewhere We Never Seeded")
    assert out["source"] == "fixture"
    assert isinstance(out["highlights"], list)
    assert isinstance(out["images"], list)


def test_images_can_be_suppressed():
    out = lookup_dining_highlights(
        restaurant_name="Verdant", place_id="fixture-verdant-10", include_images=False
    )
    assert out["images"] == []
    assert out["highlights"]


def test_missing_name_is_rejected():
    out = lookup_dining_highlights(restaurant_name="  ")
    assert out["error"]
    assert out["highlights"] == []


def test_live_path_normalizes_results_and_images(monkeypatch):
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")
    calls: list[dict] = []
    captured: dict = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        calls.append(json)
        return _FakeResponse({
            "results": [
                {
                    "title": "Osteria Midtown  review",
                    "content": "The   cacio e pepe is finished tableside, $28.",
                    "url": "https://www.example-guide.com/osteria",
                    "images": [
                        "https://cdn.example-guide.com/room.jpg",
                        "https://cdn.example-guide.com/room.jpg",  # duplicate
                        "not-a-url",                                # malformed
                    ],
                }
            ],
            "images": [
                {"url": "https://stock.example.net/pasta.jpg", "description": "Some pasta"},
            ],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(
        restaurant_name="Osteria Midtown",
        website="https://example.com/osteria-midtown",
    )

    assert out["source"] == "live"
    assert captured["url"] == web_server.TAVILY_SEARCH_URL
    assert captured["headers"]["Authorization"] == "Bearer tvly-test"
    # Knowing the official site scopes the FIRST search to that domain.
    assert calls[0]["include_domains"] == ["example.com"]
    assert out["scope"] == "official_site"
    # Whitespace collapsed, domain credited, `www.` stripped.
    assert out["highlights"][0]["snippet"] == "The cacio e pepe is finished tableside, $28."
    assert out["highlights"][0]["source"] == "example-guide.com"
    # Duplicate and malformed image URLs are dropped.
    urls = [i["url"] for i in out["images"]]
    assert urls.count("https://cdn.example-guide.com/room.jpg") == 1
    assert "not-a-url" not in urls
    # A page-derived photo outranks the generic image-search result, and inherits
    # its page's title as a caption since bare URLs carry no description.
    assert urls[0] == "https://cdn.example-guide.com/room.jpg"
    assert out["images"][0]["description"] == "Osteria Midtown review"


def test_page_furniture_is_not_mistaken_for_a_photo(monkeypatch):
    # Result pages carry social icons, logos and map tiles alongside real photos.
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({
            "results": [{
                "title": "Dining Room", "url": "https://own.example/room",
                "content": "Signature lasagna, served with basil $26",
                "images": [
                    "https://own.example/assets/instagram-icon.png",
                    "https://own.example/brand/logo.svg",
                    "https://maps.wikimedia.org/img/osm-intl.png",
                    "https://own.example/photos/dining-room.jpg",
                ],
            }],
            "images": [],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(
        restaurant_name="Trattoria", address="1 High St", website="https://own.example"
    )
    assert [i["url"] for i in out["images"]] == ["https://own.example/photos/dining-room.jpg"]


def test_page_images_are_preferred_over_generic_image_search(monkeypatch):
    # The heart of "photos of THIS restaurant": images lifted off the matched page
    # depict the place we searched for; the top-level image search ignores the
    # domain filter and can return a namesake or another branch entirely.
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")

    def fake_post(url, headers=None, json=None, timeout=None):
        return _FakeResponse({
            "results": [{
                "title": "Menu", "url": "https://own.example/menu",
                "content": "Signature lasagna, served with basil $26",
                "images": ["https://own.example/dining-room.jpg"],
            }],
            "images": [{"url": "https://elsewhere.example/other-branch.jpg",
                        "description": "A different branch"}],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(
        restaurant_name="Trattoria", address="1 High St", website="https://own.example"
    )
    assert [i["source"] for i in out["images"]][0] == "own.example"


def test_second_located_image_search_runs_only_when_photos_are_short(monkeypatch):
    # No photos on the matched pages, so a dedicated image query fires — anchored
    # on the address, and asking for the room when that's what the guest wanted.
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return _FakeResponse({
            "results": [{"title": "Menu", "url": "https://own.example/menu",
                         "content": "Signature lasagna, served with basil $26"}],
            "images": [{"url": "https://press.example/room.jpg", "description": "The room"}],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(
        restaurant_name="Trattoria",
        address="1 High St, Springfield",
        website="https://own.example",
        focus="what the dining room looks like",
    )

    assert len(calls) == 2, "a second, image-specific search should run"
    image_query = calls[1]["query"]
    assert "1 High St, Springfield" in image_query   # anchored to this location
    assert "dining room interior" in image_query     # asks for the room, not food
    assert out["image_query"] == image_query
    assert out["images"][0]["source"] == "press.example"


def test_live_does_not_widen_when_the_official_site_answers(monkeypatch):
    # Guard on credits and latency: a good domain-scoped answer must not trigger
    # a second search.
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        return _FakeResponse({
            "results": [{"title": "Menu", "url": "https://example.com/menu",
                         "content": "Beef Carpaccio, Goldbar Squash, Horseradish $24"}],
            "images": [],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(
        restaurant_name="Osteria", website="https://example.com/x", include_images=False
    )

    assert len(calls) == 1
    assert out["scope"] == "official_site"


def test_live_widens_when_official_site_returns_only_navigation_chrome(monkeypatch):
    # The common real-world case: the menu is a PDF or an image, so a domain-scoped
    # search returns site furniture. Keep the restaurant's own photos, take the
    # open web's text.
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if json.get("include_domains"):
            return _FakeResponse({
                "results": [{"title": "Carbone NYC", "url": "https://carbone.example/",
                             "content": "Skip to content Carbone Logo Reservations Instagram",
                             "images": ["https://carbone.example/lasagna.jpg",
                                        "https://carbone.example/room.jpg",
                                        "https://carbone.example/bar.jpg"]}],
                "images": [],
            })
        return _FakeResponse({
            "results": [{"title": "Where to eat", "url": "https://guide.example.org/x",
                         "content": "Order the spicy rigatoni vodka, their signature dish."}],
            "images": [{"url": "https://guide.example.org/stock.jpg", "description": "Stock"}],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(restaurant_name="Carbone", website="https://carbone.example")

    assert len(calls) == 2, "text widened once; photos were already satisfied"
    assert out["scope"] == "open_web"
    assert "spicy rigatoni" in out["highlights"][0]["snippet"]
    # The restaurant's own photos survived the switch to open-web text.
    assert {i["source"] for i in out["images"]} == {"carbone.example"}


def test_live_keeps_official_text_when_the_open_web_is_no_better(monkeypatch):
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")

    def fake_post(url, headers=None, json=None, timeout=None):
        if json.get("include_domains"):
            return _FakeResponse({
                "results": [{"title": "Home", "url": "https://own.example/",
                             "content": "A neighbourhood restaurant since 1998."}],
                "images": [],
            })
        return _FakeResponse({"results": [], "images": []})

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(restaurant_name="Quiet Place", website="https://own.example")

    assert out["scope"] == "official_site"
    assert "neighbourhood restaurant" in out["highlights"][0]["snippet"]


def test_live_widens_to_open_web_when_official_site_is_empty(monkeypatch):
    monkeypatch.setattr(web_server, "TAVILY_API_KEY", "tvly-test")
    calls: list[dict] = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(json)
        if json.get("include_domains"):  # official site knows nothing
            return _FakeResponse({"results": [], "images": []})
        return _FakeResponse({
            "results": [{"title": "Guide", "content": "Try the tasting menu.",
                         "url": "https://guide.example.org/x"}],
            "images": [],
        })

    monkeypatch.setattr(web_server.httpx, "post", fake_post)
    out = lookup_dining_highlights(
        restaurant_name="Le Petit Bistro", website="https://example.com/le-petit-bistro",
        include_images=False,
    )

    assert len(calls) == 2, "should retry once without the domain filter"
    assert "include_domains" not in calls[1]
    assert out["scope"] == "open_web"
    assert out["highlights"][0]["snippet"] == "Try the tasting menu."


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload
