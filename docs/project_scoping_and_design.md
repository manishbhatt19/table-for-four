# Table for Four — Project Scoping & Initial Agent Design

**CMU AI Agent Certification — Capstone**
**Author:** Manish Bhatt
**Document version:** 1.0 · Initial scoping & design
**Status:** Design approved; Milestone 1 (search) complete; build of perks/RAG and booking layers pending.

---

## 1. Executive summary

**Table for Four** is an agentic restaurant concierge. A user makes a natural-language
dining request — *"Italian, near Midtown, 4 people, Friday 7pm, one guest is
gluten-free"* — and the agent searches real restaurant data, reasons over which
options fit, surfaces relevant perks (coupons/offers), and books a table through a
reservation service. A **human-in-the-loop confirmation gate** sits before any
booking is finalized, and a **governance/audit layer** records every tool call and
approval.

The system is deliberately built **MCP-first** (tools exposed as Model Context
Protocol servers) and **orchestrated with LangGraph**, so the reasoning loop, tool
use, human approval, and auditing are explicit and inspectable rather than hidden
inside a single prompt.

> **Built with an agent, about an agent.** The project is itself scaffolded and
> developed using an agentic coding tool (Claude Code in VS Code) — a live,
> in-workflow demonstration of the same technology the capstone studies.

---

## 2. Problem statement

Booking a restaurant for a group is a small but real multi-constraint decision
problem. A person must reconcile cuisine, location, party size, timing, budget,
dietary restrictions, ratings, and availability — then act on it (book) — often
across several tabs and apps. It is:

- **Multi-constraint:** several soft and hard constraints interact.
- **Retrieval-heavy:** the right answer depends on current, real-world data.
- **Transactional:** it ends in an irreversible action (a reservation) that a
  responsible system should not take without explicit human confirmation.

This makes it an ideal, well-bounded testbed for an agent that must **search,
reason, personalize, and act under human oversight** — exercising the full agent
loop rather than a single retrieval or generation step.

---

## 3. Goals & non-goals

### 3.1 Goals
- Turn a free-text dining request into a **ranked, reasoned set of options** using
  **real** restaurant data.
- **Personalize** those options with relevant perks/offers via **semantic
  retrieval (RAG)** over an unstructured perks store.
- **Book** a table through a reservation interface, gated by **explicit human
  confirmation**.
- Produce a **complete audit trail** of tool calls, data provenance, and approvals.
- Keep the system **runnable offline with no API keys** for development, testing,
  and grading, while supporting **live data** when keys are provided.

### 3.2 Non-goals (explicitly out of scope)
- **Real transactional booking** against OpenTable/Resy/SevenRooms. These are
  partner-gated; we use a **self-built mock reservation backend** by design (see
  §7).
- Payment processing, real coupon redemption, or any real financial transaction.
- A production web/mobile UI. The interface is a chat/CLI-level interaction
  sufficient to demonstrate the agent loop.
- Multi-city scale, real-time availability guarantees, or account/identity systems
  beyond a lightweight member-preferences notion (stretch goal).

---

## 4. Target user & primary scenario

**Primary user:** an individual arranging a meal for a small group who wants a
single natural-language request handled end-to-end.

**Primary scenario (happy path):**
1. User: *"Italian, near Midtown, 4 people, Friday 7pm, one guest is gluten-free."*
2. Agent searches restaurants, applies constraints (cuisine, area, price, rating,
   open-now), and returns a shortlist with reasoning.
3. Agent retrieves and attaches **relevant perks** ("2 of these 3 have offers that
   fit a gluten-free group of 4").
4. User picks an option.
5. Agent proposes a booking and **pauses for explicit confirmation**.
6. On approval, the booking is finalized and a confirmation returned.
7. Every step — searches, perk matches, the approval, the booking — is logged.

---

## 5. System architecture

```
User (chat / CLI)
      │  natural-language dining request
      ▼
┌─────────────────────────────────────────────────────────────┐
│         Orchestrator Agent  (LangGraph state machine)         │
│   plan → search → match perks → propose → [HUMAN GATE] → book │
└─────────────────────────────────────────────────────────────┘
   │              │                  │                    │
   ▼              ▼                  ▼                    ▼
Search MCP     Perks MCP         Booking MCP         Governance /
 server         server            server             Audit layer
   │              │                  │                    │
   ▼              ▼                  ▼                    ▼
Google Places  Chroma vector     Mock FastAPI        Structured log:
API (New)      DB (synthetic     reservation         every tool call,
+ offline      perks, local      backend             data source, and
fixture        embeddings)                            human approval
```

**Design principles**
- **MCP-first.** Every external capability (search, perks, booking) is a discrete
  MCP tool server. The orchestrator only knows tools, not vendors — swappable and
  independently testable.
- **Offline-first, live-optional.** Each data path has a keyless offline mode
  (fixtures / synthetic data / mock backend) and a live mode behind an env var.
  The **same normalization code runs in both modes**.
- **Provenance is first-class.** Every result carries a `source` field
  (`live` | `fixture` | `synthetic`) so the audit layer can record what a decision
  rested on.
- **The irreversible step is gated.** No booking is finalized without an explicit
  human approval event.

---

## 6. Agent design detail

### 6.1 Orchestration (LangGraph)
The agent is modeled as an explicit **state graph**, not a single prompt loop.
Planned nodes:

| Node | Responsibility |
|---|---|
| `parse_request` | Extract structured constraints from free text (cuisine, area, party size, time, dietary, budget). |
| `search` | Call the Search MCP tool; get candidate restaurants. |
| `match_perks` | Call the Perks MCP tool with the user's intent + candidate `place_id`s; attach fitting offers. |
| `rank_and_explain` | Reason over fit; produce a ranked shortlist with rationale. |
| `propose_booking` | Draft a booking for the chosen option. |
| **`human_gate`** | **Interrupt**; present the booking for explicit approval/decline. |
| `book` | On approval, call the Booking MCP tool; return confirmation. |
| `audit` | (cross-cutting) log every transition, tool call, and decision. |

State is carried explicitly between nodes so the flow is inspectable and resumable —
a core reason for choosing LangGraph over an implicit agent loop.

### 6.2 Tools (MCP servers)

**A. Search — `search_restaurants`** *(built, Milestone 1)*
- Backed by **Google Places API (New)**, with an **offline fixture** fallback.
- Normalizes raw results to a clean shape (`place_id`, `name`, `address`,
  `price_level` 0–4, `rating`, `open_now`, `location`, …).
- Server-side-style filters: `max_price_level`, `min_rating`, `open_now`,
  `max_results`.
- Uses an explicit **field mask** so live calls fetch only needed fields — this
  controls both the response shape and the **billing SKU** (cost control).
- Returns `source: "live" | "fixture"`.

**B. Perks — `find_perks`** *(to build, Milestone 2.5)*
- Backed by a **Chroma vector database** of **synthetic** perks.
- **Hybrid retrieval:** semantic vector similarity over an unstructured `blurb`
  (cuisine/vibe/dietary/occasion) **combined with** structured metadata filtering
  (`min_party_size`, `expiry`, `dine_in_only`, `perk_type`, `discount_pct`,
  `valid_days`, `active`).
- **Local embeddings** (`all-MiniLM-L6-v2`) — no API key, fully offline.
- Perks are keyed to restaurants by `place_id`. Returns match score and
  `source: "synthetic"`.

**C. Booking — `create_booking` / `check_availability`** *(to build, Milestone 2)*
- Backed by a **self-built mock FastAPI reservation backend** (see §7).
- Exposes availability lookup and reservation creation with a confirmation id.
- Deterministic/mocked so the transactional path is fully testable offline.

### 6.3 Human-in-the-loop gate
Before any booking is written, the graph **interrupts** and surfaces the proposed
reservation (restaurant, time, party size, any applied perk) for the user to
**approve or decline**. Approval is a recorded event; a decline routes back to
selection. This makes the one irreversible action a deliberate, auditable human
decision — the central responsible-AI control of the system.

### 6.4 Governance & audit layer
A cross-cutting layer records a structured entry for every consequential step:
- which **tool** was called, with what arguments;
- the **data source/provenance** (`live` / `fixture` / `synthetic`);
- the **human approval** (or decline) event and timestamp;
- the final **booking outcome**.

This yields an end-to-end trail explaining *why* the agent did what it did and *on
what data* — directly addressing accountability and transparency requirements.

---

## 7. Data strategy

| Capability | Source | Rationale |
|---|---|---|
| Restaurant search | **Real** — Google Places API (New), with offline fixture | Real-world retrieval is core to the value; fixture keeps it keyless/testable. |
| Perks / coupons | **Synthetic**, clearly labeled | No cleanly-licensable free coupon dataset exists; scraping real coupon sites is ToS-gated and weakens the governance story. Labeled synthetic data is the more defensible, reproducible choice. |
| Booking backend | **Self-built mock** (FastAPI) | Real reservation systems (OpenTable/Resy/SevenRooms) are partner-gated; Yelp dropped its free tier. Mocking a partner-gated downstream is a standard, defensible agent-prototyping pattern. |

**Provenance honesty.** Every payload carries an explicit `source`. The system never
presents synthetic or mock data as if it were live — this labeling *is* part of the
responsible-AI design, not an afterthought.

**Synthetic perks generation.** A seeded (reproducible) generator produces several
perks per fixture restaurant, spanning perk types, with some deliberately **expired**
and some **party-size-gated** (to prove the metadata filters work) and varied
dietary/occasion themes (to give semantic retrieval real signal).

---

## 8. Technology stack

| Concern | Choice | Why |
|---|---|---|
| Language / runtime | Python ≥ 3.12, managed by **uv** | Fast, reproducible, lockfile-based env. |
| Tool protocol | **MCP** (`mcp[cli]`, FastMCP) | Standard, inspectable tool interface; decouples agent from vendors. |
| Orchestration | **LangGraph** | Explicit, resumable state graph with a native interrupt for the human gate. |
| Vector store | **Chroma** (local, persistent) | Simple local RAG with built-in embeddings; no external service. |
| Embeddings | `all-MiniLM-L6-v2` (local) | Free, offline, no API key — consistent with offline-first philosophy. |
| Booking backend | **FastAPI** (mock) | Lightweight, self-contained transactional service. |
| HTTP | `httpx` | Async-capable modern client for the Places call. |
| Config | `python-dotenv` | Keeps keys out of code and version control. |
| Testing | `pytest` | Offline test suites per component. |

---

## 9. Milestone plan

| # | Milestone | State |
|---|---|---|
| 1 | Search MCP server (Google Places, offline-testable) | ✅ Complete |
| 2 | Mock booking FastAPI + booking MCP server | ⬜ Next |
| 2.5 | **Perks / RAG:** synthetic perks → Chroma → `find_perks` MCP tool | ✅ Complete |
| 3 | LangGraph orchestrator; end-to-end happy path (search → perks → book) | ⬜ |
| 4 | Human-in-the-loop gate + governance/audit logging | ⬜ |
| 5 | *(Stretch)* member preferences, model comparison, **illustrative restaurant imagery**, polished demo | ⬜ |

### 9.1 Milestone 5 — illustrative restaurant imagery *(stretch)*
A UX-polish layer that shows a representative image alongside each option to make
the concierge feel richer in the demo. Scoped deliberately to respect
representational integrity (see §10):
- **Storage:** images live in a **folder**; each restaurant record carries a
  `photo` *reference/path* — image bytes are **never** stored in the vector DB.
- **Sourcing:** *live mode* uses **real Google Places photos** (authentic, of the
  actual place); *offline mode* uses **generic, cuisine-themed illustrative
  images**, clearly labeled — never presented as a depiction of a specific named
  restaurant.
- **Sequencing:** presentation polish, not agent capability — built only after the
  core loop (search → perks → human gate → book) works end-to-end.
- *(Optional stretch-within-stretch:* multimodal CLIP embeddings to enable
  visually-similar retrieval — the one design that would let images genuinely earn
  a place in the vector store.)

---

## 10. Responsible AI & governance considerations

- **Human authority over irreversible actions.** Bookings require explicit
  confirmation; the agent proposes, the human disposes.
- **Data provenance & honesty.** `source` labeling on every payload; synthetic and
  mock data are never disguised as live.
- **Auditability.** A structured trail of tool calls, sources, and approvals makes
  every decision explainable after the fact.
- **Cost control & least-privilege data access.** The Places field mask requests
  only the fields the agent reasons over, minimizing both data collected and billing.
- **Secret hygiene.** API keys live only in a gitignored `.env`; the recommended key
  is restricted to the single API it needs, capping blast radius if leaked.
- **Reproducibility.** Offline modes and a seeded synthetic generator make runs
  deterministic and gradable without credentials.
- **Representational integrity (imagery).** Any AI-generated or stock restaurant
  image is **illustrative only** and explicitly labeled as such — the system never
  fabricates a picture and presents it as a depiction of a specific, real, named
  restaurant. Only *live* Google Places photos are shown as actual images of a
  place. This guardrail is a first-class design constraint on the Milestone 5
  imagery layer, not an afterthought.

---

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Live vs. synthetic **join-key mismatch** (Places `place_id`s differ from fixture ids, so perks won't match live results) | Key perks to the fixture set for the offline demo; decide the live-matching strategy explicitly before wiring M3. |
| **Vector DB used for its own sake** (perks that are purely structured don't need embeddings) | Hybrid design: embed genuinely unstructured blurbs, filter on structured metadata — vector search only where semantics add value. |
| **Heavier dependency** (`chromadb` pulls in onnxruntime) | Accepted; local embeddings avoid an external service and keep the offline story intact. |
| Over-automation of the **irreversible booking** step | Non-negotiable human gate before any write; declines are first-class. |
| **Misrepresentation via imagery** (a fabricated image implying it depicts a specific real restaurant) | Illustrative-only, clearly-labeled generic imagery offline; real Google Places photos only in live mode; never caption a synthetic image as a specific named venue (see §10). |
| **Scope creep** from stretch ideas | Milestones are ordered; stretch items (M5) only after the core loop works end-to-end. |

---

## 12. Success criteria

The capstone is successful when:
1. A natural-language request flows end-to-end: **parse → search → perk match →
   proposal → human approval → booking confirmation.**
2. The system runs **fully offline with no API keys** (fixtures + synthetic perks +
   mock backend) and **switches to live search** when a key is present.
3. **Perk matching demonstrably uses both** semantic similarity and metadata filters
   (e.g. a gluten-free group-of-4 query surfaces a fitting, non-expired, party-size-valid
   offer).
4. No booking is ever finalized without a recorded **human approval**.
5. A **governance/audit trail** exists for every consequential step with its data
   provenance.
6. Each component ships with **passing offline tests**.

---

*End of document — v1.0.*
