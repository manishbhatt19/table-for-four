"""Runnable concierge demo — the single command to drive the full loop.

Usage:
    uv run python -m agent                       # default sample request
    uv run python -m agent "sushi for 2 friday"  # custom request
    uv run python -m agent --guest "Manish" "Italian for 4 friday 7pm"

Runs the LangGraph orchestrator end-to-end and prints the reasoning trace and the
final confirmation. Uses the LLM automatically if a key is configured in .env,
otherwise the deterministic heuristic path (shown as MODE).
"""

from __future__ import annotations

import argparse

from agent.graph import run_concierge

DEFAULT_REQUEST = "Italian, near Midtown, 4 people, Friday 7pm, one guest is gluten-free"


def main() -> None:
    ap = argparse.ArgumentParser(description="Table for Four — concierge demo")
    ap.add_argument("request", nargs="?", default=DEFAULT_REQUEST, help="dining request")
    ap.add_argument("--guest", default="Guest", help="name for the reservation")
    ap.add_argument("--heuristic", action="store_true", help="force offline heuristic mode")
    args = ap.parse_args()

    final = run_concierge(
        args.request,
        guest_name=args.guest,
        use_llm=False if args.heuristic else None,
    )

    print(f"\nREQUEST : {args.request}")
    print(f"GUEST   : {args.guest}")
    print(f"MODE    : {final.get('reasoning_mode')}\n")
    print("REASONING TRACE")
    for line in final.get("log", []):
        print(f"  {line}")
    print("\nCONFIRMATION")
    print(f"  {final.get('narrative') or 'No booking was made.'}")


if __name__ == "__main__":
    main()
