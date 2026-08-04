# Table for Four — Agentic Restaurant Concierge

**CMU AI Agent Certification — Capstone Project**
Author: Manish Bhatt

An MCP-first, LangGraph-orchestrated agent that takes a natural-language dining
request ("Italian, near Midtown, 4 people, Friday 7pm, one guest is
gluten-free"), searches real restaurant data, reasons over fit, and books a
table through a self-built mock reservation service — with a human-in-the-loop
confirmation step and a full governance/audit trail.

> **Built with an agent, about an agent.** This repo is scaffolded and
> developed using Claude Code (agentic coding in VS Code) — a small live
> demonstration of agentic tooling in the dev workflow itself.

## Architecture (target)

```
User (chat) → Orchestrator Agent (LangGraph)
                 ├── Search MCP Server   → Google Places API (New)
                 └── Booking MCP Server  → mock FastAPI reservation backend
              → Governance / Audit Layer (logs every tool call + human approval)
              → Human-in-the-loop confirmation gate → booking finalized
```

Search uses **real** Google Places data; the transactional booking step uses a
**self-built mock backend** by design — OpenTable/Resy/SevenRooms are
partner-gated and Yelp dropped its free tier. This is a standard, defensible
pattern for agent prototyping against partner-gated downstream systems.

## Status

| Milestone | State |
|---|---|
| 1 — Search MCP server (Google Places, offline-testable) | ✅ working |
| 2 — Mock booking FastAPI + booking MCP server | ✅ working |
| 2.5 — Perks/RAG: synthetic perks → Chroma → `find_perks` MCP tool | ✅ working |
| 3 — LangGraph orchestrator, end-to-end happy path | ✅ working |
| 4 — Human-in-loop gate + governance/audit logging | ⬜ next |
| 5 (stretch) — member preferences, model comparison, illustrative imagery, demo | ⬜ |

## Getting started

Prereqs: [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned via
`.python-version` and provisioned by uv automatically.

```bash
uv sync                       # create the env from the lockfile
cp .env.example .env          # optional: add GOOGLE_PLACES_API_KEY for live data
uv run pytest -q              # run the offline test suite
```

### The search server

Runs in **offline fixture mode** with no API key, and switches to **live**
Google Places automatically once `GOOGLE_PLACES_API_KEY` is set — the same
normalization code path runs in both modes.

```bash
# inspect the server interactively with the MCP dev inspector
uv run mcp dev mcp_servers/search_server.py

# or run it as a stdio MCP server (what the orchestrator will spawn)
uv run mcp_servers/search_server.py
```

To enable live data, see [docs/google_places_setup.md](docs/google_places_setup.md).

### Run the concierge (M3)

The orchestrator chains search → perks → booking through a LangGraph state machine
with a refine-retry loop and per-thread working memory. Run it end-to-end:

```bash
uv run python -m agent                                  # default sample request
uv run python -m agent "Italian for 4, Friday 7pm"      # custom request
uv run python -m agent --heuristic "sushi for 2"        # force offline mode
```

It prints the full reasoning trace and the booking confirmation. With an
`OPENAI_API_KEY` (or OpenRouter) set in `.env` it uses the LLM to parse the request
and write the confirmation; with no key it runs a deterministic heuristic path, so
the whole loop works offline.

## Interpersonal concierge — Dino

`agent/concierge_chat.py` is the warm, conversational front-end: Dino guides a guest
through a full booking *journey* (understand intent → gather details → recommend →
pick → check times → book → tips), keeping the guest in the loop at each choice and
remembering them across sessions in Chroma. It needs an `OPENAI_API_KEY` (or
OpenRouter) in `.env`.

```bash
uv run python -m agent chat                 # terminal REPL
uv run python -m agent chat --name "Manish" # skip the name prompt
uv run streamlit run agent/chat_app.py      # web chat UI (live memory panel)
```

The Streamlit UI (`agent/chat_app.py`) is a thin wrapper over the same session API
(`start_session` + `_run_turn`) — the sidebar shows the guest's long-term profile
filling in live from Chroma as they talk.

## Bookings ledger & cancellation policy

Reservations persist to a **SQLite ledger** in the mock backend
([mock_booking_api/app.py](mock_booking_api/app.py)) — a real relational
system-of-record (restaurant name/address/phone, date/time, party, guest email,
`status`, and cancellation timestamps), still zero-setup and offline (path via
`BOOKING_DB_PATH`; tests use an in-memory DB). Chroma stays reserved for semantic
work (perks + profiles); the transactional ledger is SQL.

Cancellation is governed by a **24-hour policy enforced in the backend**, not the
model: `POST /bookings/{id}/cancel` cancels a booking that's more than 24h away and
stamps the ledger; inside that window it refuses and returns the restaurant's phone
and website so the guest can call directly. Dino exposes this as `cancel_reservation`
and relays the "call the restaurant" path verbatim — Dino never claims a
cancellation the backend didn't confirm — and the guest's long-term memory is kept
in sync with the ledger.

## Perks RAG — retrieval you can measure and inspect

The perks layer is a hybrid RAG over a synthetic offers store (10 restaurants, 24
perks): semantic vector search (local MiniLM embeddings in Chroma) blended with
structured metadata + time filters, re-ranked by a tunable `semantic_weight`. It's
both **evaluated** and **inspectable** — see [docs/week3_rag.md](docs/week3_rag.md).

```bash
uv run python -m mcp_servers.perks_eval                       # hit@k / precision / MRR
uv run python -m mcp_servers.perks_eval --sweep               # weight vs. precision trade-off
uv run python -m mcp_servers.perks_inspect "tacos with friends" --day Tue
uv run python -m mcp_servers.perks_inspect "romantic dinner with wine" --party 2 --blurb
```

Long-term member memory ([agent/profile_memory.py](agent/profile_memory.py)) runs
on the same stack with two retrieval modes: key lookup and semantic recall
(`find_members("the guest who loves Sicilian wine")`).

## Repo layout

```
table-for-four/
├── mcp_servers/
│   ├── search_server.py      # ✅ Google Places search tool (MCP)
│   ├── perks_server.py       # ✅ perks RAG tool: Chroma hybrid + tunable weight (MCP)
│   ├── perks_data.py         # ✅ deterministic synthetic perks generator
│   ├── perks_eval.py         # ✅ labeled retrieval eval (hit@k / precision / MRR)
│   ├── perks_inspect.py      # ✅ CLI: inspect a query's ranked results + scores
│   ├── booking_server.py     # ✅ booking + cancellation tools over the backend (MCP)
│   └── fixtures/             # offline fixtures: places + perks_seed.json
├── mock_booking_api/         # ✅ FastAPI reservation backend + SQLite ledger (self-built mock)
├── agent/                    # ✅ (M3) LangGraph orchestrator
│   ├── graph.py              #     state machine + refine-retry loop + checkpointer
│   ├── reasoning.py          #     heuristic + LLM parse / rank / narrate
│   ├── tools.py              #     in-process MCP tool registry
│   ├── config.py             #     provider-agnostic LLM loader (OpenAI/OpenRouter)
│   └── __main__.py           #     `python -m agent` demo CLI
├── agent/                    # (M3) LangGraph orchestrator
├── governance/               # (M4) audit log + human-approval gate
├── tests/
└── docs/
```
