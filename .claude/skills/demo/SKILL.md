---
name: demo
description: Launch or rehearse the Table for Four capstone demo — the Streamlit chat UI, LangGraph Studio, or the CLI trace. Use when asked to run the demo, record it, show the agent working end to end, or reset demo state between takes.
---

# Running the Table for Four demo

The recorded demo is a graded capstone deliverable. Three surfaces, each showing a
different thing. Pick by what the moment needs; don't run all three by reflex.

| Surface | Command | Shows |
|---|---|---|
| **Streamlit chat** (primary) | `uv run streamlit run src/table_for_four/ui/chat_app.py` | Dino end to end, plus the live memory panel filling in from Chroma as the guest talks |
| **LangGraph Studio** | `uv run langgraph dev` → graph `concierge` | The orchestrator as a state machine: nodes, the refine loop, the checkpointer |
| **CLI trace** | `uv run python -m table_for_four "Italian for 4, Friday 7pm"` | The full reasoning trace and confirmation, in text — good for a close up |

## Pre-flight

- `uv sync` first. `OPENAI_API_KEY` (or OpenRouter) in `.env` is **required** for the
  chat surface — without it Dino can't run.
- `GOOGLE_PLACES_API_KEY` and `TAVILY_API_KEY` are optional. Without them, search and
  web highlights serve offline fixtures and the demo still completes. Live keys make
  the shortlist real; say which mode you're in when narrating.
- Never print or read `.env` on a shared screen. `.env.example` shows the shape.

## The happy path worth recording

This sequence exercises every built milestone in one unbroken take:

1. **New guest.** Dino welcomes, then asks "have you dined with us before?" *before*
   searching — the point being that a returning guest's preferences are only
   reachable by email, and after the shortlist is on screen it's too late to use them.
2. **Give an email, state the outing.** Cuisine, area, party size, a date, a dietary
   need. Dino asks how they'd like to choose rather than assuming.
3. **Shortlist.** Note which entries carry a perk — that's the RAG layer surfacing a
   matched offer, not a hardcoded flag.
4. **Pick one → open times → confirm.** Dino reads the booking back and waits for a
   yes. No booking exists until a confirmation id comes back.
5. **After booking**: the generated menu card and cited highlights appear beneath the
   reply, with sources. Then Dino offers *once* to update a standing preference if
   this outing differed from what's on file — and changes nothing if the guest
   doesn't answer.
6. **Restart the session, give the same email.** Recognised by name, with their
   saved cuisines and last booking. That's the long-term memory closing the loop.

## Resetting between takes

A second take on a fresh guest needs the stores cleared, or Dino will greet a guest
who "returns" mid-recording:

```bash
rm -rf src/table_for_four/agent/.chroma_profiles          # member profiles
rm -f  src/table_for_four/mcp_servers/booking/backend/bookings.db   # reservations ledger
```

Both rebuild themselves on next run. **Leave
`src/table_for_four/mcp_servers/perks/.chroma_perks` alone** unless the seed changed
— it reindexes from `fixtures/perks_seed.json` and costs a slow first query.

Alternatively point the ledger elsewhere for a throwaway run:
`BOOKING_DB_PATH=/tmp/demo.db uv run streamlit run src/table_for_four/ui/chat_app.py`

## Before recording

Run `uv run pytest -q` and confirm green. A broken offline path shows up on camera as
a stack trace three minutes into a take.
