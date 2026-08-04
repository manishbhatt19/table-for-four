"""Offline tests for the perks RAG MCP server.

These build an **ephemeral** (in-memory) Chroma collection from the synthetic
seed and call the retrieval core directly — no persistent store, no MCP client,
no network beyond the one-time local embedding-model download that Chroma caches.

A fixed `today` is injected so the expiry filter is deterministic.
"""

import chromadb
import pytest

from mcp_servers.perks_server import build_collection, query_perks

# Any date after the "expired" anchor (2020) and before the "valid" anchor (2099).
TODAY = "2026-07-22"


@pytest.fixture(scope="module")
def collection():
    client = chromadb.EphemeralClient()
    return build_collection(client)


def test_semantic_matches_gluten_free_celebration(collection):
    # The vector side should surface the celiac-safe / birthday perk for this
    # intent, over the generic Italian and French offers.
    perks = query_perks(
        collection,
        "birthday celebration dinner, one guest is gluten-free",
        today=TODAY,
    )
    assert len(perks) >= 1
    assert perks[0]["place_id"] == "fixture-pizzeria-3"  # Nonna's Gluten-Free Kitchen
    assert all(p["source"] == "synthetic" for p in perks)


def test_party_size_filter_excludes_larger_minimums(collection):
    perks = query_perks(
        collection,
        "dinner for a group",
        party_size=2,
        max_results=20,
        today=TODAY,
    )
    # The 6-top banquet and party-of-4 offers must be filtered out for a pair.
    assert perks, "expected at least one perk a party of 2 qualifies for"
    assert all(p["min_party_size"] <= 2 for p in perks)


def test_expired_perk_never_returned(collection):
    perks = query_perks(
        collection,
        "restaurant week prix fixe deal",
        place_ids=["fixture-trattoria-2"],
        max_results=20,
        today=TODAY,
    )
    ids = {p["perk_id"] for p in perks}
    assert "perk-trattoria-02" not in ids  # expiry 2020-01-01
    assert all(p["expiry"] >= TODAY for p in perks)


def test_place_ids_restrict_results(collection):
    perks = query_perks(
        collection,
        "special dinner",
        place_ids=["fixture-bistro-4"],
        max_results=20,
        today=TODAY,
    )
    assert perks
    assert all(p["place_id"] == "fixture-bistro-4" for p in perks)


def test_day_filter_excludes_weekday_only_perk(collection):
    # The gluten-free family pizza night is valid Mon-Thu only.
    perks = query_perks(
        collection,
        "family gluten-free pizza night",
        day="Sat",
        max_results=20,
        today=TODAY,
    )
    for p in perks:
        assert not p["valid_days"] or "Sat" in p["valid_days"].split(",")
    assert "perk-nonna-02" not in {p["perk_id"] for p in perks}


def test_semantic_weight_tunes_ranking(collection):
    # Every result carries the blended score and its two component signals.
    perks = query_perks(collection, "dinner", party_size=6, max_results=20, today=TODAY)
    assert perks
    assert all({"similarity", "metadata_fit", "score"} <= p.keys() for p in perks)

    # At weight 1.0 the blend is pure semantic: score collapses onto similarity.
    pure_semantic = query_perks(
        collection, "dinner", party_size=6, max_results=20, semantic_weight=1.0, today=TODAY
    )
    assert all(p["score"] == p["similarity"] for p in pure_semantic)

    # At weight 0.0 ranking is pure metadata-fit: for a party of six, a group offer
    # (min_party_size == 6) fits best and rises to the top — a different #1 than the
    # semantic ordering would pick.
    pure_meta = query_perks(
        collection, "dinner", party_size=6, max_results=20, semantic_weight=0.0, today=TODAY
    )
    assert pure_meta[0]["min_party_size"] == 6
    assert pure_meta[0]["score"] >= pure_meta[-1]["score"]


if __name__ == "__main__":
    import json

    client = chromadb.EphemeralClient()
    col = build_collection(client)
    out = query_perks(
        col, "romantic anniversary dinner with wine", party_size=2, today=TODAY
    )
    print(json.dumps(out, indent=2))
