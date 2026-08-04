"""Perks (offers/coupons) MCP server — hybrid RAG over synthetic perks.

Exposes a single tool, `find_perks`, that combines:

* **Semantic vector search** over each perk's unstructured `blurb`
  (cuisine / vibe / dietary / occasion), and
* **Structured metadata filtering** (place, party size, day, expiry, active),

in a local **Chroma** vector database. This hybrid is the point: the vector side
matches intent ("gluten-free birthday"), the metadata side enforces hard
constraints (party of 6, not expired) — so the vector DB only carries weight
where semantics genuinely add value.

Data is **synthetic** (see perks_data.py); every result is labeled
`source: "synthetic"` so the governance layer records that a suggestion rested on
mock offer data. Embeddings are the local `all-MiniLM-L6-v2` model: no API key,
fully offline after a one-time model download (~80MB) on first run.

Run standalone (stdio transport):
    uv run mcp_servers/perks_server.py

Inspect interactively:
    uv run mcp dev mcp_servers/perks_server.py
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

from mcp_servers.perks_data import generate_perks

# --- Configuration -----------------------------------------------------------

SEED_PATH = Path(__file__).parent / "fixtures" / "perks_seed.json"
CHROMA_PATH = Path(__file__).parent / ".chroma_perks"
COLLECTION_NAME = "restaurant_perks"

# Fields carried as Chroma metadata (everything except the embedded blurb). Chroma
# metadata values must be scalars (str/int/float/bool) -- no None, no lists.
_METADATA_FIELDS = (
    "place_id",
    "restaurant_name",
    "title",
    "perk_type",
    "discount_pct",
    "min_party_size",
    "dine_in_only",
    "valid_days",
    "expiry",
    "active",
)

_EMBED = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2, local

# Default blend of the two retrieval signals: how much the *semantic* similarity
# counts vs the structured *metadata* fit when ranking. Semantic-dominant by
# default (the vector side is the point); dial down to let hard-constraint fit
# (party snugness, day specificity) reorder more aggressively.
DEFAULT_SEMANTIC_WEIGHT = 0.7

mcp = FastMCP("restaurant-perks")

_collection: Collection | None = None  # lazy persistent singleton


# --- Data loading ------------------------------------------------------------

def _load_perks() -> list[dict[str, Any]]:
    """Load perks from the committed seed, falling back to generating them."""
    if SEED_PATH.exists():
        return json.loads(SEED_PATH.read_text(encoding="utf-8")).get("perks", [])
    return generate_perks()


def _metadata(perk: dict[str, Any]) -> dict[str, Any]:
    return {field: perk[field] for field in _METADATA_FIELDS}


def build_collection(client: chromadb.ClientAPI) -> Collection:
    """Create (or reuse) the perks collection on `client` and load the seed once.

    Uses cosine space so we can report a 0-1 similarity. Idempotent: perks are
    added only when the collection is empty, so re-runs don't duplicate.
    """
    col = client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=_EMBED,
        metadata={"hnsw:space": "cosine"},
    )
    if col.count() == 0:
        perks = _load_perks()
        col.add(
            ids=[p["perk_id"] for p in perks],
            documents=[p["blurb"] for p in perks],
            metadatas=[_metadata(p) for p in perks],
        )
    return col


def get_collection() -> Collection:
    """Lazily build/open the persistent perks collection (singleton)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = build_collection(client)
    return _collection


# --- Retrieval (shared by the tool and tests) --------------------------------

def _build_where(
    place_ids: list[str] | None, party_size: int | None
) -> dict[str, Any] | None:
    """Metadata pre-filter applied inside the vector query."""
    conds: list[dict[str, Any]] = [{"active": True}]
    if place_ids:
        conds.append({"place_id": {"$in": list(place_ids)}})
    if party_size is not None:
        # keep perks whose minimum party size the group can satisfy
        conds.append({"min_party_size": {"$lte": int(party_size)}})
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


def _row_to_perk(pid: str, doc: str, meta: dict[str, Any], distance: float) -> dict[str, Any]:
    perk = {"perk_id": pid, "blurb": doc, **meta}
    perk["similarity"] = round(1.0 - distance, 3)  # cosine distance -> similarity
    perk["source"] = "synthetic"
    return perk


def _metadata_fit(perk: dict[str, Any], party_size: int | None, day: str | None) -> float:
    """Structured-fit score in [0,1] — the non-semantic half of the hybrid rank.

    Rewards perks whose hard fields snugly match the request context: a group
    offer for a group, a day-specific offer on that day. Neutral (0.5) for any
    dimension the request doesn't constrain, so an unfiltered query is ranked on
    semantics alone.
    """
    # Party snugness: min_party_size approaching the group's size fits better than
    # a catch-all min-1 offer (a 6-top banquet is a great fit for six).
    if party_size:
        party = min(1.0, (perk.get("min_party_size") or 1) / party_size)
    else:
        party = 0.5

    # Day specificity: an offer valid *specifically* on the requested day beats a
    # generic any-day one (survivors are already filtered to be day-valid).
    valid = perk.get("valid_days") or ""
    if day:
        day_fit = 1.0 if (valid and day in valid.split(",")) else 0.6
    else:
        day_fit = 0.5

    return (party + day_fit) / 2.0


def query_perks(
    collection: Collection,
    query: str,
    *,
    place_ids: list[str] | None = None,
    party_size: int | None = None,
    day: str | None = None,
    max_results: int = 5,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    today: str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval core: vector search + metadata filter + date/day post-filter.

    Ranking blends two signals: the semantic `similarity` and a structured
    `metadata_fit`, combined as
    `score = semantic_weight * similarity + (1 - semantic_weight) * metadata_fit`.
    `semantic_weight` is clamped to [0,1]; at 1.0 ranking is pure vector search.

    `today` (ISO date) is injectable for deterministic testing; defaults to the
    real current date.
    """
    n = max(1, min(max_results, 20))
    where = _build_where(place_ids, party_size)
    # Over-fetch so the Python-side expiry/day filters still leave enough results.
    res = collection.query(
        query_texts=[query], n_results=n * 3, where=where
    )

    perks = [
        _row_to_perk(pid, res["documents"][0][i], res["metadatas"][0][i], res["distances"][0][i])
        for i, pid in enumerate(res["ids"][0])
    ]

    # Post-filters that Chroma's metadata `where` can't express cleanly.
    cutoff = today or date.today().isoformat()
    perks = [p for p in perks if p["expiry"] >= cutoff]  # ISO dates sort lexically
    if day:
        perks = [
            p for p in perks
            if not p["valid_days"] or day in p["valid_days"].split(",")
        ]

    # Hybrid re-rank: blend the semantic score with the structured-fit score.
    w = max(0.0, min(1.0, semantic_weight))
    for p in perks:
        p["metadata_fit"] = round(_metadata_fit(p, party_size, day), 3)
        p["score"] = round(w * p["similarity"] + (1.0 - w) * p["metadata_fit"], 3)
    perks.sort(key=lambda p: p["score"], reverse=True)
    return perks[:n]


# --- Tool --------------------------------------------------------------------

@mcp.tool()
def find_perks(
    query: str,
    place_ids: list[str] | None = None,
    party_size: int | None = None,
    day: str | None = None,
    max_results: int = 5,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
) -> dict[str, Any]:
    """Find restaurant perks (offers/coupons) matching a dining request.

    Combines semantic matching on the perk description with hard-constraint
    filtering, over a synthetic perks store.

    Args:
        query: Natural-language intent (e.g. "birthday dinner, one guest is
            gluten-free"). This is what the vector search matches on.
        place_ids: Optional restaurant ids to restrict to — typically the
            candidate set returned by `search_restaurants`, so perks line up with
            the shortlist.
        party_size: Group size; excludes perks that require a larger party.
        day: Three-letter day (e.g. "Fri"); excludes perks not valid that day.
        max_results: Maximum perks to return (1-20).
        semantic_weight: 0-1 blend of the semantic vs metadata-fit signals when
            ranking (default 0.7, semantic-dominant; 1.0 = pure vector search).

    Returns:
        A dict with `source` ("synthetic"), the `query`, a `result_count`, and a
        `results` list of matching perks, each with a 0-1 `similarity`, a
        `metadata_fit`, the blended `score` it was ranked on, and its structured
        fields (perk_type, discount_pct, min_party_size, expiry, …).
    """
    perks = query_perks(
        get_collection(),
        query,
        place_ids=place_ids,
        party_size=party_size,
        day=day,
        max_results=max_results,
        semantic_weight=semantic_weight,
    )
    return {
        "source": "synthetic",
        "query": query,
        "result_count": len(perks),
        "results": perks,
    }


if __name__ == "__main__":
    mcp.run()
