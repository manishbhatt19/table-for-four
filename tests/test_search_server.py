"""Offline smoke tests for the search MCP server.

These call the tool's underlying function directly (no MCP client needed) and
exercise the fixture path, so they run with no API key and no network.
"""

from table_for_four.mcp_servers.search.server import (
    _photo_refs,
    place_photos,
    search_restaurants,
)


def test_returns_fixture_results_offline():
    out = search_restaurants(query="Italian dinner near Midtown")
    assert out["source"] == "fixture"
    assert out["result_count"] >= 1
    assert all("name" in r and "price_level" in r for r in out["results"])


def test_price_filter_excludes_expensive():
    out = search_restaurants(query="dinner", max_price_level=2)
    assert all(
        r["price_level"] is None or r["price_level"] <= 2 for r in out["results"]
    )


def test_open_now_filter():
    out = search_restaurants(query="dinner", open_now=True)
    assert all(r["open_now"] is True for r in out["results"])


def test_min_rating_filter():
    out = search_restaurants(query="dinner", min_rating=4.6)
    assert all(
        r["rating"] is None or r["rating"] >= 4.6 for r in out["results"]
    )


def test_cuisine_filter_excludes_other_cuisines():
    # An Italian request must not return the French bistro — cuisine is a hard filter.
    out = search_restaurants(query="dinner for four", cuisine="italian")
    assert out["result_count"] >= 1
    types = {t for r in out["results"] for t in (r["types"] or [])}
    assert "italian_restaurant" in types
    assert "french_restaurant" not in types
    assert all("Bistro" not in (r["name"] or "") for r in out["results"])


# --- Places photos -----------------------------------------------------------

def test_photo_refs_keep_the_credit_with_the_handle():
    # These photos are largely taken by diners, so the attribution has to travel
    # with the reference or whatever renders it has nothing to credit.
    refs = _photo_refs([
        {"name": "places/abc/photos/one",
         "authorAttributions": [{"displayName": "A Diner"}]},
        {"name": "places/abc/photos/two", "authorAttributions": []},
        {"widthPx": 100},  # no handle, nothing to fetch
    ])
    assert [r["ref"] for r in refs] == ["places/abc/photos/one", "places/abc/photos/two"]
    assert refs[0]["attribution"] == "A Diner"
    assert refs[1]["attribution"] == ""


def test_no_photos_without_a_key_so_the_offline_demo_still_runs():
    # Invariant 4: every path degrades to fixtures with no key. Offline this must
    # return nothing rather than reach for the network.
    assert place_photos([{"ref": "places/abc/photos/one"}]) == []


def test_resolving_a_photo_never_puts_the_api_key_where_a_browser_can_see_it(monkeypatch):
    # The media endpoint 302s straight to the image, so handing a browser that URL
    # would publish our API key in an <img src>. skipHttpRedirect asks for JSON
    # instead, keeping the key in the header on this side.
    import table_for_four.mcp_servers.search.server as ss

    seen: dict = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"photoUri": "https://lh3.googleusercontent.com/signed-photo"}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(url=url, params=params or {}, headers=headers or {})
        return _Resp()

    monkeypatch.setattr(ss, "PLACES_API_KEY", "test-key")
    monkeypatch.setattr(ss.httpx, "get", fake_get)

    out = ss.place_photos([{"ref": "places/abc/photos/one", "attribution": "A Diner"}])

    assert seen["params"]["skipHttpRedirect"] == "true"
    assert seen["headers"]["X-Goog-Api-Key"] == "test-key"
    assert "test-key" not in seen["url"]                  # not in the path...
    assert "test-key" not in str(seen["params"])          # ...and not in the query
    assert out[0]["url"] == "https://lh3.googleusercontent.com/signed-photo"
    assert "test-key" not in out[0]["url"]                # nor in what we hand out
    assert out[0]["description"] == "Photo from Google Places, by A Diner"


def test_one_unavailable_photo_does_not_cost_the_guest_their_dining_tips(monkeypatch):
    import httpx

    import table_for_four.mcp_servers.search.server as ss

    class _Ok:
        def raise_for_status(self):
            return None

        def json(self):
            return {"photoUri": "https://example.com/good.jpg"}

    def fake_get(url, **_kw):
        if "bad" in url:
            raise httpx.HTTPError("gone")
        return _Ok()

    monkeypatch.setattr(ss, "PLACES_API_KEY", "test-key")
    monkeypatch.setattr(ss.httpx, "get", fake_get)

    out = ss.place_photos([{"ref": "places/abc/photos/bad"}, {"ref": "places/abc/photos/ok"}])
    assert [p["url"] for p in out] == ["https://example.com/good.jpg"]


if __name__ == "__main__":
    # Allow `uv run tests/test_search_server.py` as a quick manual check.
    import json

    print(json.dumps(search_restaurants("Italian near Midtown, gluten-free"), indent=2))
