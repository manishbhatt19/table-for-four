# Project Progress & Handoff — Table for Four

_Snapshot as of 2026-07-17. This document records what has been built so far,
how it's wired together, and what comes next — so any session (human or agent)
can pick up without re-deriving the state._

## What this project is

An MCP-first, LangGraph-orchestrated agentic restaurant concierge (CMU AI Agent
Certification capstone). It takes a natural-language dining request, searches
real restaurant data, reasons over fit, and books a table through a self-built
mock reservation service — with a human-in-the-loop confirmation step and a full
governance/audit trail.

## Capstone submission deliverables (don't lose track)

The final submission is graded on more than working code. All of these are required:

| # | Deliverable | State |
|---|---|---|
| A | **Context + architecture walkthrough** (written/spoken) | 🟡 design doc drafted; needs final polish + narration |
| B | **Recorded demo video** — full end-to-end loop on screen | 🟡 command ready: `uv run python -m agent`; record after M4 gate |
| C | **Attached document** — `docs/project_scoping_and_design.md` → PDF | 🟡 living master done; export final PDF near end |
| D | **GitHub repo link** (shareable) | ⬜ publish (private) via GitHub Desktop; get link |
| E | **Recorded video link** attached with submission | ⬜ after B |

**Build implication:** keep a single, narratable **end-to-end happy path** runnable
at every step so the demo (B) is always recordable. When M3 lands, ensure one
command/flow runs: request → search → perks → human approval → booking confirmation.

## Milestone status

| Milestone | State |
|---|---|
| 1 — Search MCP server (Google Places, offline-testable) | ✅ Done |
| 2 — Mock booking FastAPI + booking MCP server | ✅ Done |
| 2.5 — Perks/RAG (synthetic perks → Chroma → `find_perks`) | ✅ Done |
| 3 — LangGraph orchestrator, end-to-end happy path | ✅ Done |
| 4 — Human-in-loop gate + governance/audit logging | ⬜ Next |
| 5 (stretch) — member preferences, model comparison, illustrative imagery, demo | ⬜ |

### Milestone 3 — LangGraph orchestrator (done)
- `agent/graph.py` — the concierge **state machine**: `parse → search → (refine ⟲
  search)* → match_perks → rank → propose → gate → book → audit`. Working memory is
  `ConciergeState` persisted by a **MemorySaver checkpointer** per thread; a
  **conditional refine-retry loop** (guarded by `MAX_ITERATIONS`) relaxes
  constraints and re-searches when a query returns nothing.
- `agent/reasoning.py` — **LLM + heuristic duality**: LLM (when a key is set) parses
  the request and writes the confirmation; a deterministic rule-based path runs
  otherwise, so the whole loop works offline. Ranking prefers a restaurant that has
  a matched perk, then rating.
- `agent/tools.py` — in-process **tool registry** (search / perks / booking).
- `agent/config.py` — provider-agnostic LLM loader (OpenAI or OpenRouter via `.env`).
- `agent/__main__.py` — `uv run python -m agent [request]` demo CLI (prints the
  reasoning trace + confirmation) — **the narratable command for the demo video**.
- `tests/test_orchestrator.py` — 5 offline end-to-end tests (heuristic mode),
  including a refine-loop test. Full suite: **24 passing**.
- The `gate` node auto-approves in M3 (placeholder); M4 makes it a real human
  interrupt + the governance/audit trail.

### Milestone 2 — Mock booking (done)
- `mock_booking_api/app.py` — self-built **FastAPI** reservation service:
  `GET /availability`, `POST /bookings`, `GET /bookings/{id}`, in-memory store,
  **deterministic** availability (pure function of place_id+date, tighter for large
  parties). `create_booking` is the system's one irreversible write.
- `mcp_servers/booking_server.py` — `check_availability`, `create_booking`,
  `get_booking` MCP tools with **live/offline duality**: `BOOKING_API_URL` → real
  HTTP; unset → drives the app **in-process** via Starlette TestClient (offline,
  no port). Each result carries `backend` ("live"|"mock").
- `tests/test_booking.py` — 10 offline tests (backend + MCP tools). Full suite:
  **19 passing**.
- Run the backend standalone: `uv run uvicorn mock_booking_api.app:app --port 8000`.

### Milestone 2.5 — Perks/RAG (done)
- `mcp_servers/perks_data.py` — deterministic synthetic perks generator; writes
  `mcp_servers/fixtures/perks_seed.json` (11 perks keyed to fixture restaurants).
- `mcp_servers/perks_server.py` — `find_perks` MCP tool: **hybrid retrieval** over
  a local **Chroma** store — semantic vector search on each perk's `blurb` **+**
  metadata filters (`place_ids`, `party_size`, `day`, expiry, active). Local
  `all-MiniLM-L6-v2` embeddings (no API key; ~80MB model downloaded once, then
  offline). Every result labeled `source: "synthetic"`.
- Vector store persists to `mcp_servers/.chroma_perks/` (gitignored, regenerable
  from the seed).
- `tests/test_perks_server.py` — 5 offline tests (semantic match, party-size
  filter, expired-perk exclusion, place-id restriction, day filter). Full suite:
  **9 passing**.

## What has been built (Milestone 1)

### Repo scaffolding
- `pyproject.toml` — Python ≥3.12, deps: `httpx`, `mcp[cli]`, `python-dotenv`;
  dev dep `pytest`. `pytest` configured with `pythonpath = ["."]`.
- `.python-version` (3.12), `uv.lock`, `.gitignore`, `.env.example`.
- `README.md` — architecture diagram, milestone table, getting-started.
- `docs/google_places_setup.md` — step-by-step guide to enable live Google
  Places API (New) and drop a key into `.env`.

### Search MCP server — `mcp_servers/search_server.py`
- Exposes one MCP tool: **`search_restaurants`**.
- **Dual-mode by design:** runs in **offline fixture mode** with no API key, and
  automatically switches to **live** Google Places API (New) when
  `GOOGLE_PLACES_API_KEY` is set. Same normalization path in both modes.
- Live calls use an explicit **field mask** (`X-Goog-FieldMask`) requesting only
  the fields the concierge reasons over — this controls both response shape and
  billing SKU (cost-control / responsible-AI point for the writeup).
- Normalizes raw Places results into a clean restaurant shape (`place_id`,
  `name`, `address`, `price_level` mapped from enum → 0–4, `rating`, `open_now`,
  etc.).
- Post-search filters: `max_price_level`, `min_rating`, `open_now`, `max_results`
  (clamped 1–20).
- Returns `source` ("live"|"fixture") in the payload so the future governance
  layer can record whether a booking decision rested on live or mock data.
- `mcp_servers/fixtures/places_sample.json` — offline fixture mirroring the real
  API response shape.

### Tests — `tests/test_search_server.py`
- Four offline smoke tests calling the tool function directly (no MCP client, no
  network): fixture results, price filter, open-now filter, min-rating filter.

## Known environment note
- The project uses **`uv`** for env/deps. In this session `uv` was **not on the
  PATH** for either the Bash or PowerShell tools, so `uv run pytest` could not be
  executed here. Verify locally with:
  ```bash
  uv sync
  uv run pytest -q
  ```
  If `uv` isn't found, install from https://docs.astral.sh/uv/ or ensure its
  install dir (typically `~/.local/bin` on the shell PATH) is exported.

## Not yet started
- No git commits yet (`master` branch, all files untracked).
- `mock_booking_api/`, `agent/`, `governance/` directories referenced in the
  README layout do **not** exist yet — they are Milestones 2–4.

## Suggested next steps
1. Make the first git commit to capture Milestone 1.
2. Milestone 2: build the mock booking FastAPI backend + a booking MCP server.
