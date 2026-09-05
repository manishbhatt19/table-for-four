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

## Architecture

```
Guest ──▶ Dino, the conversational path      or   LangGraph orchestrator, scripted
          (two model calls per turn,              (parse → search → perks → rank →
           six tool steps at most)                 propose → GATE → book → audit)
                              │
                              ▼
          the permission broker — every tool call runs as a declared unit,
          and a unit reaching past its grant raises rather than proceeding
                              │
   ┌───────────────┬──────────┴─────┬─────────────────┬────────────────────┐
   ▼               ▼                ▼                 ▼                    ▼
 search MCP     perks MCP        web MCP         booking MCP         profile store
 Google Places  Chroma, local    Tavily,         FastAPI service     Chroma, keyed
 API (New)      embeddings       cited snippets  + SQLite ledger     by email
                              │
                              ▼
     Governance across all of it — a human approval gate that genuinely stops
     the run, an append-only audit trail, and a grounding check on every reply
```

Search uses **real** Google Places data; the transactional booking step uses a
**self-built mock backend** by design — OpenTable/Resy/SevenRooms are
partner-gated and Yelp dropped its free tier. This is a standard, defensible
pattern for agent prototyping against partner-gated downstream systems.

Every path degrades to fixtures without an API key, so the whole system — search,
perks, web highlights, booking, and the model itself — runs offline.

## Status

| Milestone | State |
|---|---|
| 1 — Search MCP server (Google Places, offline-testable) | ✅ shipped |
| 2 — Mock booking FastAPI + booking MCP server | ✅ shipped |
| 2.5 — Perks/RAG: synthetic perks → Chroma → `find_perks` MCP tool | ✅ shipped |
| 3 — LangGraph orchestrator, end-to-end happy path | ✅ shipped |
| 3.5 — Conversational concierge (Dino) + long-term member memory | ✅ shipped |
| 3.6 — Web highlights: Tavily menu/photo lookup → generated menu cards | ✅ shipped |
| 4 — Agentic harness: five declared units with enforced permissions | ✅ shipped |
| 4 — Human-in-loop gate + governance/audit logging + reply grounding | ✅ shipped |
| 5 — Capstone deck and demo | ✅ shipped |

**236 tests**, all offline: no API key, no network, about 30 seconds.

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
uv run mcp dev src/table_for_four/mcp_servers/search/server.py

# or run it as a stdio MCP server (what the orchestrator will spawn)
uv run src/table_for_four/mcp_servers/search/server.py
```

To enable live data, see [docs/google_places_setup.md](docs/google_places_setup.md).

### Run the concierge

The orchestrator chains search → perks → **approval** → booking through a LangGraph
state machine with a refine-retry loop and per-thread working memory. Run it
end-to-end:

```bash
uv run python -m table_for_four                                  # default sample request
uv run python -m table_for_four "Italian for 4, Friday 7pm"      # custom request
uv run python -m table_for_four --heuristic "sushi for 2"        # force offline mode
uv run python -m table_for_four --yes "sushi for 2"              # unattended: pre-approve
uv run langgraph dev                                             # the graph in LangGraph Studio
```

It prints the full reasoning trace, stops at the gate to ask the person at the
keyboard, and only then books. With an `OPENAI_API_KEY` (or OpenRouter) set in
`.env` it uses the LLM to parse the request and write the confirmation; with no key
it runs a deterministic heuristic path, so the whole loop works offline.

## Interpersonal concierge — Dino

`src/table_for_four/agent/concierge_chat.py` is the warm, conversational front-end: Dino guides a guest
through a full booking *journey* (understand intent → gather details → recommend →
pick → check times → book → tips), keeping the guest in the loop at each choice and
remembering them across sessions in Chroma. It needs an `OPENAI_API_KEY` (or
OpenRouter) in `.env`.

```bash
uv run python -m table_for_four chat                 # terminal REPL
uv run python -m table_for_four chat --name "Manish" # skip the name prompt
uv run streamlit run src/table_for_four/ui/chat_app.py      # web chat UI (live memory panel)
```

The Streamlit UI (`src/table_for_four/ui/chat_app.py`) is a thin wrapper over the same session API
(`start_session` + `_run_turn`) — the sidebar shows the guest's long-term profile
filling in live from Chroma as they talk.

## Five units, and what each is allowed to touch

Capability is split by **authority rather than skill**. Each unit is one markdown
file in [src/table_for_four/agent/roster/](src/table_for_four/agent/roster/):
frontmatter declares what it may call, the body is its brief.

| Unit | Role | Holds | Cannot |
|---|---|---|---|
| **Dino** | Host, the only unit that runs a model | *nothing* | every capability in the system |
| **Scout** | Finds the table, never takes it | search, perks, availability | book, cancel, write to memory |
| **Curator** | Knows the food, never meets the guest | web highlights, photos | search, book, write to memory |
| **Steward** | Keeper of the member's own record | remember, adopt an identity | search, book, reach the web |
| **Booker** | The only unit that commits the guest | create and cancel bookings | search, reach the web, adopt an identity |

The unit that reasons owns no tools, so it can only *ask* a unit that holds one —
misbehaviour is a `NotGranted` exception rather than a policy violation. Booker
cannot search, so a booking may only name a restaurant Scout already surfaced.
Widening a permission means editing the file and saying why.

## Governance — the gate, the trail, and the reply check

[src/table_for_four/governance/](src/table_for_four/governance/) is the layer that
sits across everything above.

- **Nothing books without a person saying yes.** In the graph, `gate_node` issues a
  real LangGraph `interrupt`: the run checkpoints and genuinely stops until a caller
  resumes it with a decision. Anything that is not clearly a yes — no approver, an
  unparsable answer, end of input — declines. In chat, the summary goes on screen
  with Reserve and Change my mind. There is no default that books, on either path.
- **An append-only audit trail** (`audit.py`) records every tool call, approval and
  finding, each naming the acting unit, the member and the time, optionally mirrored
  to a JSON Lines file via `TF4_AUDIT_LOG`. The Streamlit sidebar shows it live.
- **Every reply is checked before the guest reads it** (`grounding.py`). A time,
  date, confirmation id or email that no tool returned is removed from the reply and
  recorded either way. It is deterministic code rather than a second model, because
  each of those has an exact answer already in session state. Dish and restaurant
  names are **not** covered, and the module says so rather than implying otherwise.

## Bookings ledger & cancellation policy

Reservations persist to a **SQLite ledger** in the mock backend
([src/table_for_four/mcp_servers/booking/backend/app.py](src/table_for_four/mcp_servers/booking/backend/app.py)) — a real relational
system-of-record (restaurant name/address/phone, date/time, party, guest email,
`status`, and cancellation timestamps), still zero-setup and offline (path via
`BOOKING_DB_PATH`; tests use an in-memory DB). Chroma stays reserved for semantic
work (perks + profiles); the transactional ledger is SQL.

Cancellation is governed by a **24-hour policy enforced in the backend**, not the
model: `POST /bookings/{id}/cancel` cancels a booking that's more than 24h away and
stamps the ledger; inside that window it refuses and returns the restaurant's phone
and website so the guest can call directly. Dino exposes this as `cancel_reservation`
and relays the "call the restaurant" path verbatim — Dino never claims a
cancellation the backend didn't confirm. A cancellation made through Dino also flips
the status in the guest's own history (`mark_booking`), so the two agree; a booking
cancelled out of band does not, and reconciling that is still open.

## Perks RAG — retrieval you can measure and inspect

The perks layer is a hybrid RAG over a synthetic offers store (10 restaurants, 24
perks): semantic vector search (local MiniLM embeddings in Chroma) blended with
structured metadata + time filters, re-ranked by a tunable `semantic_weight`. It's
both **evaluated** and **inspectable** — the eval harness and the inspector below
are the evidence, and both run offline.

```bash
uv run python -m table_for_four.mcp_servers.perks.eval                       # hit@k / precision / MRR
uv run python -m table_for_four.mcp_servers.perks.eval --sweep               # weight vs. precision trade-off
uv run python -m table_for_four.mcp_servers.perks.inspect "tacos with friends" --day Tue
uv run python -m table_for_four.mcp_servers.perks.inspect "romantic dinner with wine" --party 2 --blurb
```

Long-term member memory ([src/table_for_four/agent/profile_memory.py](src/table_for_four/agent/profile_memory.py)) runs
on the same stack with two retrieval modes: key lookup and semantic recall
(`find_members("the guest who loves Sicilian wine")`).

## Live web highlights — menu, dishes, photos (Tavily)

The third retrieval surface is the **open web**. When a guest asks *"what's good
there?"* or *"can I see the place?"*, and once automatically after a booking,
Dino calls `show_dining_highlights` ([src/table_for_four/mcp_servers/web/server.py](src/table_for_four/mcp_servers/web/server.py))
— a Tavily search that returns a few **cited** menu highlights plus photos.

Three constraints keep it honest and on-scope:

- **It can only be pointed at a restaurant already recommended or booked** in this
  conversation, so it can't become a general web-search back door around the
  dining-only guardrail.
- **Scoped to the restaurant's own site first** (Places gives us `websiteUri`),
  widening to the open web only if that returns nothing.
- **Nothing is passed off as official.** Every snippet carries its source domain,
  photos are captioned with where they came from, and the model is told to
  attribute dishes ("diners keep mentioning…") rather than promise them.

Photos render in the Streamlit UI beneath Dino's reply, never as pasted URLs.
Without a `TAVILY_API_KEY` the tool serves offline fixture highlights with locally
generated placeholder graphics, so the demo still runs end-to-end with no key.

```bash
uv run mcp dev src/table_for_four/mcp_servers/web/server.py    # inspect the tool interactively
```

## Repo layout

Structured to mirror the architecture in
[docs/project_scoping_and_design.md](docs/project_scoping_and_design.md) §5: the
orchestrator, one package per MCP server, and each server owning the data it uses.

```
table-for-four/
├── src/table_for_four/
│   ├── __main__.py                 # `python -m table_for_four` demo CLI + chat
│   ├── agent/                      # the orchestrator and the concierge
│   │   ├── graph.py                #   LangGraph state machine + refine loop + checkpointer
│   │   ├── state.py                #   typed graph state (working memory)
│   │   ├── reasoning.py            #   heuristic + LLM parse / rank / narrate
│   │   ├── tools.py                #   in-process MCP tool registry
│   │   ├── config.py               #   provider-agnostic LLM loader (OpenAI/OpenRouter)
│   │   ├── concierge_chat.py       #   Dino: the turn-by-turn booking journey
│   │   ├── profile_memory.py       #   long-term member memory (Chroma)
│   │   ├── roster/                 #   the five units: one .md each, permissions
│   │   │                           #   in frontmatter, brief in the body
│   │   ├── menu_card.py            #   generated cuisine-themed menu cards (SVG)
│   │   └── calendar_invite.py      #   .ics builder for a confirmed booking
│   ├── governance/                 # the gate, the trail, and the reply check
│   │   ├── audit.py                #   append-only records, actor on every line
│   │   └── grounding.py            #   every reply checked against tool results
│   ├── mcp_servers/                # one package per tool server, data included
│   │   ├── search/                 #   Google Places + offline places fixture
│   │   ├── perks/                  #   perks RAG: server, seed data, eval, inspector,
│   │   │                           #   fixtures, and the .chroma_perks index
│   │   ├── booking/                #   booking + cancellation tools
│   │   │   └── backend/            #     the FastAPI reservation service + SQLite ledger
│   │   └── web/                    #   Tavily menu highlights + offline fixture
│   └── ui/
│       └── chat_app.py             # Streamlit chat UI (live memory panel)
├── tests/                          # 236 offline tests, no keys or network needed
└── docs/                           # design reference, harness and testing notes,
                                    # architecture diagram, capstone deck
```

Each MCP server keeps its own fixtures and persistent store beside it, so a server
is self-contained: delete the directory and nothing else has to change.
