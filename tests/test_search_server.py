"""Offline smoke tests for the search MCP server.

These call the tool's underlying function directly (no MCP client needed) and
exercise the fixture path, so they run with no API key and no network.
"""

from mcp_servers.search_server import search_restaurants


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


if __name__ == "__main__":
    # Allow `uv run tests/test_search_server.py` as a quick manual check.
    import json

    print(json.dumps(search_restaurants("Italian near Midtown, gluten-free"), indent=2))
