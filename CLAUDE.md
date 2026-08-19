# Table for Four — agent working notes

CMU AI Agent Certification capstone (author: Manish Bhatt). An MCP-first,
LangGraph-orchestrated restaurant concierge: a natural-language dining request →
real restaurant search → perks RAG → human-confirmed booking → governance trail.

`README.md` and `docs/` are written for people. This file is for you.

## Commands

Everything runs through `uv`. The whole suite is offline — no keys, no network.

```bash
uv sync                                              # env from the lockfile
uv run pytest -q                                     # full suite (119 tests, ~30s)
uv run pytest tests/test_profile_memory.py -q        # concierge + memory only

uv run python -m table_for_four                      # one-shot orchestrator demo
uv run python -m table_for_four chat                 # Dino, terminal REPL
uv run streamlit run src/table_for_four/ui/chat_app.py   # Dino, web UI (demo surface)
uv run langgraph dev                                 # LangGraph Studio (graph: `concierge`)

uv run python -m table_for_four.mcp_servers.perks.eval          # hit@k / prec@k / MRR
uv run python -m table_for_four.mcp_servers.perks.inspect "tacos with friends" --day Tue
uv run mcp dev src/table_for_four/mcp_servers/search/server.py  # MCP inspector
```

## Where things live

`src/` layout mirrors `docs/project_scoping_and_design.md` §5. One package per MCP
server, each owning its own fixtures and store — delete a server directory and
nothing else has to change.

| Path | What it is |
|---|---|
| `agent/graph.py` | LangGraph state machine: parse → search → refine⟲ → perks → rank → propose → gate → book → audit |
| `agent/concierge_chat.py` | Dino — the conversational front-end. Tool schemas, handlers, session state |
| `agent/roster/` | The five declared units, one `.md` each: grants in frontmatter, brief + tool descriptions in the body. `dino.md` **is** the system prompt |
| `agent/profile_memory.py` | Long-term member memory in Chroma, keyed by email |
| `agent/tools.py` | In-process tool registry — the agent's only route to the world |
| `mcp_servers/{search,perks,booking,web}/` | One package per server; booking owns the FastAPI + SQLite ledger |
| `src/table_for_four/governance/` | M4 audit log + approval gate — **goes here**, not at repo root (README's tree is out of date on this) |

## Invariants — do not break these

These are the product, not preferences. Tests guard most of them; if a change makes
one awkward, that's a signal to stop, not to route around it.

1. **The model never invents.** Not a restaurant, a time, an email, a dish, or a
   confirmation id. Every one of those comes from a tool result. Dining tips about
   food must be attributed to `show_dining_highlights` output.
2. **The guest chooses at every branch point.** Shortlist, time, booking
   confirmation. Remembering a guest's usuals is for *offering*, never deciding.
3. **Standing preferences need consent.** Home area, usual party size, and
   favourite cuisines change only when the guest agrees or asks — see
   `sticky_conflicts` in `profile_memory.py` and `_authorized_changes` in
   `concierge_chat.py`. A *first* value is learned freely; that isn't a change.
   Consent is checked against the guest's own words, never the model's claim.
4. **Offline first.** Every path degrades to fixtures without an API key: Places →
   `places_sample.json`, Tavily → `web_highlights_sample.json`, LLM → the heuristic
   path in `reasoning.py`. A change that makes a key mandatory breaks the demo and
   the test suite.
5. **Dining only.** The scope guardrail in the system prompt is load-bearing;
   `show_dining_highlights` is deliberately restricted to restaurants already
   surfaced in the conversation so it can't become a general web-search back door.
6. **The backend owns policy, not the model.** The 24 hour cancellation window is
   enforced in the FastAPI backend. Dino relays refusals verbatim and never claims
   a cancellation the ledger didn't confirm.
7. **A unit only touches what its roster entry grants.** `_dispatch` runs each
   handler as its owning unit; `agent/tools.py` and the `profile_memory` write
   wrappers call `roster.require` before acting. Widening a grant means editing the
   `.md` and saying why — never adding a call and letting it through.
8. **Editing `roster/dino.md` breaks the golden test, on purpose.** Prompt changes
   are fine; they just have to be deliberate and land in their own commit, with
   `tests/golden/dino_system_prompt.txt` regenerated in the same one.
9. **Nothing books without a human saying yes.** `gate_node` interrupts and the
   run genuinely stops; only an explicit approval resumes it. `run_concierge`
   has no default that books — no approver means declined. A convenient default
   would quietly undo the gate, so there isn't one.
10. **The reply is checked, not just the action.** `_vetted` runs every reply
    through `governance.grounding` before the guest sees it: a time, date,
    confirmation id or email no tool returned is removed and recorded. It is
    deterministic on purpose — a claim with an exact answer is checked, never
    estimated by a second model.

## Conventions

- **Comments explain *why*, in prose.** This codebase reads like it was written by
  someone thinking out loud about the guest — match that register. Comments name
  the failure being prevented ("a birthday dinner in Brooklyn doesn't mean they've
  moved"), not the syntax. No comment restates the line below it.
- **Tests are named as behaviour and open with the bug they guard.** See
  `test_merely_naming_an_area_is_not_permission_to_change_it`. Deterministic and
  offline: monkeypatch `profile_memory.remember`, patch `search_restaurants`,
  never call a real model.
- **Docs are weekly submissions.** `docs/weekN_<topic>.md`, with a title block
  (course, week, author), a framing blockquote, numbered sections, and a PDF
  alongside. Grounded in the working code, argued rather than asserted.
- **Commits** describe the guest-visible change and why it mattered, not the files
  touched.

## Gotchas

- Package is `table_for_four`. Module paths are fully qualified everywhere
  (`python -m table_for_four`, `python -m table_for_four.mcp_servers.perks.eval`) —
  the pre-restructure short forms were cleaned out; don't reintroduce them.
- `_run_turn` is the only place the chat path calls a model (two call sites, bounded
  by `MAX_TOOL_HOPS = 6`). Keep it that way — see `docs/week4_tree_of_thought.md`
  for why added model calls are a design regression here, not an upgrade.
- Chroma stores are on disk beside their server. Tests use ephemeral clients; if a
  test starts depending on demo data, it's wrong.
- `.env` holds live API keys. Read `.env.example` instead — a hook blocks the real one.

## Current state

M1–M3.6 shipped (search, booking, perks RAG, orchestrator, Dino, web highlights),
plus the week 4 harness: `agent/roster/` — five declared units, enforced grants, and
`actors` on the audit line, at zero added model cost (`docs/week4_agentic_harness.md`
§11 records the measurements and the one place the plan gave way to the code).
`docs/week4_tree_of_thought.md` is the reasoned decision *against* tree search and
stays a decision, not a build.

**M4 shipped** — `src/table_for_four/governance/`:

- `gate_node` interrupts for real (LangGraph `interrupt` + checkpointer). Only an
  explicit yes resumes it; no approver, a malformed resume, or EOF all decline.
  The CLI asks the person at the keyboard; `--yes` is for unattended runs.
- `audit.py` — append-only records (`event`, `actor`, `member_id`, `at`), held on
  the session and on graph state, optionally mirrored to `$TF4_AUDIT_LOG` as JSONL.
  Every chat tool call is recorded with the unit that ran it.
- `grounding.py` — the reply check. Covers times, ISO dates, confirmation ids and
  emails. **Does not cover dish or restaurant names, or phone numbers** — those
  need entity extraction or a judgement, and the module says so rather than
  implying full coverage. The Streamlit sidebar shows the trail live.

Still open: reconciling the ledger with profile memory, and the M4 writeup in
`docs/`.
