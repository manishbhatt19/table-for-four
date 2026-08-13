"""Inspect the perks RAG: see a query's ranked results and *why* they ranked.

A small CLI window into the hybrid retriever — it prints, for one query, the
top-k perks with the three numbers behind each row: the semantic `sim`, the
structured `fit`, and the blended `score` they were ordered on. Handy for demos
and for sanity-checking how the `--weight` knob re-ranks results.

Runs offline against an ephemeral store built from the committed seed, so it's
deterministic and writes nothing to disk.

Examples:
    uv run python -m table_for_four.mcp_servers.perks.inspect "romantic dinner with wine" --party 2
    uv run python -m table_for_four.mcp_servers.perks.inspect "tacos with friends" --day Tue --weight 0.3
    uv run python -m table_for_four.mcp_servers.perks.inspect "gluten-free birthday" --blurb
"""

from __future__ import annotations

import argparse
import sys

import chromadb

# Restaurant/perk text has accented characters (e.g. "Dégustation"); force UTF-8
# so they render on Windows' default cp1252 console instead of mojibake.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

from table_for_four.mcp_servers.perks.server import DEFAULT_SEMANTIC_WEIGHT, build_collection, query_perks


def _truncate(text: str, width: int) -> str:
    return text if len(text) <= width else text[: width - 3] + "..."


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect perks RAG retrieval for a query")
    ap.add_argument("query", help="natural-language dining intent")
    ap.add_argument("--party", type=int, default=None, help="party size filter")
    ap.add_argument("--day", default=None, help="3-letter day filter, e.g. Fri")
    ap.add_argument("--weight", type=float, default=DEFAULT_SEMANTIC_WEIGHT,
                    help="semantic weight 0-1 (default 0.7; 1.0 = pure vector)")
    ap.add_argument("-k", type=int, default=5, help="how many results to show")
    ap.add_argument("--blurb", action="store_true", help="also print each perk's blurb")
    args = ap.parse_args()

    collection = build_collection(chromadb.EphemeralClient())
    perks = query_perks(
        collection, args.query,
        party_size=args.party, day=args.day,
        max_results=args.k, semantic_weight=args.weight,
    )

    filt = []
    if args.party is not None:
        filt.append(f"party={args.party}")
    if args.day:
        filt.append(f"day={args.day}")
    filt.append(f"weight={args.weight}")
    print(f'\nPerks retrieval | query="{args.query}" | {" ".join(filt)} | k={args.k}\n')

    if not perks:
        print("  (no perks matched the filters)\n")
        return

    print(f"  {'#':>2}  {'score':>5}  {'sim':>5}  {'fit':>5}  {'restaurant':<20}  perk")
    print(f"  {'-'*2}  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*20}  {'-'*28}")
    for i, p in enumerate(perks, start=1):
        print(f"  {i:>2}  {p['score']:>5.3f}  {p['similarity']:>5.3f}  {p['metadata_fit']:>5.3f}  "
              f"{_truncate(p['restaurant_name'], 20):<20}  {_truncate(p['title'], 28)}")
        if args.blurb:
            print(f"        - {p['blurb']}")
    print()
    print("  score = weight*sim + (1-weight)*fit   |   sim = semantic   fit = metadata\n")


if __name__ == "__main__":
    main()
