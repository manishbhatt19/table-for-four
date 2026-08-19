"""Runnable concierge demos — the commands to drive the agent.

One-shot booking loop (offline-capable):
    uv run python -m table_for_four                       # default sample request
    uv run python -m table_for_four "sushi for 2 friday"  # custom request
    uv run python -m table_for_four --guest "Manish" "Italian for 4 friday 7pm"

Interpersonal chat concierge (needs an OpenAI key; remembers guests in Chroma):
    uv run python -m table_for_four chat
    uv run python -m table_for_four chat --name "Manish"

The one-shot form runs the LangGraph orchestrator end-to-end and prints the
reasoning trace and the final confirmation. It uses the LLM automatically if a key
is configured in .env, otherwise the deterministic heuristic path (shown as MODE).
"""

from __future__ import annotations

import argparse
import sys

from table_for_four.agent.graph import run_concierge

DEFAULT_REQUEST = "Italian, near Midtown, 4 people, Friday 7pm, one guest is gluten-free"


def _chat(argv: list[str]) -> None:
    from table_for_four.agent.concierge_chat import run_chat

    ap = argparse.ArgumentParser(
        prog="table_for_four chat", description="Interpersonal concierge chat"
    )
    ap.add_argument("--name", default=None, help="guest name/handle (else prompted)")
    args = ap.parse_args(argv)
    run_chat(name=args.name)


def _ask_at_the_gate(proposal: dict) -> bool:
    """The M4 human gate, on a terminal.

    The run has genuinely stopped by the time this is called — the graph is
    checkpointed mid-flight and will not proceed until this returns. Anything but
    a clear yes is a no, including EOF, which is what a piped or unattended run
    produces: those should decline rather than book a table nobody watched.
    """
    # ASCII on purpose: this prints to a Windows console during the recorded
    # demo, where an em-dash arrives as mojibake.
    print("\nAPPROVAL NEEDED - this books a real table")
    for label in ("restaurant", "date", "time", "party_size", "guest_name", "perk"):
        if proposal.get(label):
            print(f"  {label:11}: {proposal[label]}")
    try:
        return input("  Confirm this reservation? [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:
        print("  (no answer — declining)")
        return False


def main() -> None:
    # `chat` subcommand routes to the conversational front-end; anything else is
    # treated as a one-shot dining request (preserving the original usage).
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        _chat(sys.argv[2:])
        return

    ap = argparse.ArgumentParser(description="Table for Four — concierge demo")
    ap.add_argument("request", nargs="?", default=DEFAULT_REQUEST, help="dining request")
    ap.add_argument("--guest", default="Guest", help="name for the reservation")
    ap.add_argument("--heuristic", action="store_true", help="force offline heuristic mode")
    ap.add_argument("--yes", action="store_true",
                    help="approve the booking without asking (unattended runs)")
    args = ap.parse_args()

    final = run_concierge(
        args.request,
        guest_name=args.guest,
        use_llm=False if args.heuristic else None,
        approve=True if args.yes else _ask_at_the_gate,
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
