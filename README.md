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
| 2 — Mock booking FastAPI + booking MCP server | ⬜ next |
| 2.5 — Perks/RAG: synthetic perks → Chroma → `find_perks` MCP tool | ✅ working |
| 3 — LangGraph orchestrator, end-to-end happy path | ⬜ |
| 4 — Human-in-loop gate + governance/audit logging | ⬜ |
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

## Repo layout

```
table-for-four/
├── mcp_servers/
│   ├── search_server.py      # ✅ Google Places search tool (MCP)
│   ├── perks_server.py       # ✅ perks RAG tool: Chroma hybrid search (MCP)
│   ├── perks_data.py         # ✅ deterministic synthetic perks generator
│   └── fixtures/             # offline fixtures: places + perks_seed.json
├── mock_booking_api/         # (M2) FastAPI reservation backend
├── agent/                    # (M3) LangGraph orchestrator
├── governance/               # (M4) audit log + human-approval gate
├── tests/
└── docs/
```
