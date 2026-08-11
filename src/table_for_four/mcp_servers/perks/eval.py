"""Retrieval evaluation for the perks RAG layer.

A small, labeled benchmark that measures whether the hybrid retriever surfaces
the *right* restaurant for a given dining intent. Each case pairs a natural
query (plus any hard filters) with the set of `place_id`s that a good retriever
should return — so we can score retrieval quality with standard IR metrics
rather than eyeballing it:

* **hit@k**   — did any relevant restaurant appear in the top-k? (coverage)
* **prec@k**  — what fraction of the top-k were relevant? (precision)
* **MRR**     — 1 / rank of the first relevant hit, averaged (ranking quality)

The labels are deliberately by *restaurant*, not perk id: an intent like
"omakase sushi for two" is satisfied by any perk from the sushi house. All data
is the synthetic seed, so this runs fully offline and deterministically.

Run it:
    uv run python -m mcp_servers.perks_eval          # default weight
    uv run python -m mcp_servers.perks_eval --sweep  # compare semantic weights
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import chromadb

from table_for_four.mcp_servers.perks.server import DEFAULT_SEMANTIC_WEIGHT, build_collection, query_perks

# Fixed eval date: between the "expired" (2020) and "valid" (2099) seed anchors,
# so the expiry filter behaves identically every run.
EVAL_TODAY = "2026-07-22"
DEFAULT_K = 3


@dataclass(frozen=True)
class EvalCase:
    """One labeled query: the intent, its hard filters, and the relevant places."""

    query: str
    relevant: frozenset[str]        # place_ids a good retriever should surface
    party_size: int | None = None
    day: str | None = None


# Labeled benchmark over the 10-restaurant synthetic store. Each intent targets a
# cuisine/occasion that one (or a few) restaurants clearly own.
EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase("birthday celebration dinner, one guest is gluten-free",
             frozenset({"fixture-pizzeria-3"})),
    EvalCase("romantic anniversary dinner with a bottle of wine", party_size=2,
             relevant=frozenset({"fixture-trattoria-2", "fixture-bistro-4"})),
    EvalCase("omakase sushi tasting for two", party_size=2,
             relevant=frozenset({"fixture-sakura-5"})),
    EvalCase("cheap street tacos night out with friends", party_size=4,
             relevant=frozenset({"fixture-taqueria-6"})),
    EvalCase("vegetarian indian curry feast with lots of spice",
             frozenset({"fixture-spice-7"})),
    EvalCase("dry-aged steak dinner to entertain a business client",
             frozenset({"fixture-primecut-8"})),
    EvalCase("spicy thai food with good vegan options",
             frozenset({"fixture-orchid-9"})),
    EvalCase("healthy plant-based vegan tasting menu",
             frozenset({"fixture-verdant-10"})),
    EvalCase("family-style italian sharing platters for a group", party_size=4,
             relevant=frozenset({"fixture-osteria-1"})),
    EvalCase("private corporate dinner for a large party", party_size=8,
             relevant=frozenset({"fixture-primecut-8", "fixture-spice-7",
                                  "fixture-trattoria-2"})),
)


# --- Metrics -----------------------------------------------------------------

def _hit_at_k(places: list[str], relevant: frozenset[str], k: int) -> float:
    return 1.0 if any(p in relevant for p in places[:k]) else 0.0


def _precision_at_k(places: list[str], relevant: frozenset[str], k: int) -> float:
    top = places[:k]
    if not top:
        return 0.0
    return sum(1 for p in top if p in relevant) / len(top)


def _reciprocal_rank(places: list[str], relevant: frozenset[str]) -> float:
    for i, p in enumerate(places, start=1):
        if p in relevant:
            return 1.0 / i
    return 0.0


def evaluate_case(
    collection: Any, case: EvalCase, *, k: int, semantic_weight: float, today: str
) -> dict[str, Any]:
    perks = query_perks(
        collection, case.query,
        party_size=case.party_size, day=case.day,
        max_results=k, semantic_weight=semantic_weight, today=today,
    )
    places = [p["place_id"] for p in perks]
    return {
        "query": case.query,
        "top_places": places,
        "hit": _hit_at_k(places, case.relevant, k),
        "precision": _precision_at_k(places, case.relevant, k),
        "rr": _reciprocal_rank(places, case.relevant),
    }


def run_eval(
    collection: Any, *, k: int = DEFAULT_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT, today: str = EVAL_TODAY,
) -> dict[str, Any]:
    """Score every case and return per-case rows plus aggregate metrics."""
    rows = [
        evaluate_case(collection, c, k=k, semantic_weight=semantic_weight, today=today)
        for c in EVAL_CASES
    ]
    n = len(rows) or 1
    aggregate = {
        f"hit@{k}": round(sum(r["hit"] for r in rows) / n, 3),
        f"prec@{k}": round(sum(r["precision"] for r in rows) / n, 3),
        "mrr": round(sum(r["rr"] for r in rows) / n, 3),
    }
    return {"k": k, "semantic_weight": semantic_weight, "cases": rows, "aggregate": aggregate}


# --- CLI report --------------------------------------------------------------

def _print_report(result: dict[str, Any]) -> None:
    k = result["k"]
    print(f"\nPerks retrieval eval | k={k} | semantic_weight={result['semantic_weight']} "
          f"| {len(result['cases'])} cases\n")
    print(f"  {'hit':>4}  {'rr':>5}  query")
    print(f"  {'-'*4}  {'-'*5}  {'-'*46}")
    for r in result["cases"]:
        mark = " HIT" if r["hit"] else "miss"
        print(f"  {mark}  {r['rr']:>5.2f}  {r['query'][:46]}")
    agg = result["aggregate"]
    print("\n  " + "  |  ".join(f"{key} {val}" for key, val in agg.items()) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate perks retrieval quality")
    ap.add_argument("-k", type=int, default=DEFAULT_K, help="top-k to score (default 3)")
    ap.add_argument("--sweep", action="store_true",
                    help="compare several semantic-weight settings")
    args = ap.parse_args()

    collection = build_collection(chromadb.EphemeralClient())

    if args.sweep:
        print("\nSemantic-weight sweep (1.0 = pure vector, 0.0 = pure metadata-fit):")
        print(f"\n  {'weight':>6}  {'hit@'+str(args.k):>7}  {'prec@'+str(args.k):>8}  {'mrr':>5}")
        print(f"  {'-'*6}  {'-'*7}  {'-'*8}  {'-'*5}")
        for w in (1.0, 0.7, 0.5, 0.3, 0.0):
            agg = run_eval(collection, k=args.k, semantic_weight=w)["aggregate"]
            print(f"  {w:>6.1f}  {agg['hit@'+str(args.k)]:>7}  "
                  f"{agg['prec@'+str(args.k)]:>8}  {agg['mrr']:>5}")
        print()
    else:
        _print_report(run_eval(collection, k=args.k))


if __name__ == "__main__":
    main()
