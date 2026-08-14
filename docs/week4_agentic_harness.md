# Table for Four — The Agentic Harness

**CMU AI Agent Certification — Capstone · Week 4 Plan**
**Author:** Manish Bhatt

> Week 4 is about the **agentic harness**: how an agent is packaged, what it is
> allowed to do, and where its instructions live. This document records the design
> decision and the phased build, under one hard constraint — the harness must cost
> nothing extra in tokens or latency. Planned against commit `ab5008e`; all three
> phases are now built, and §11 records what shipped and where it diverged.

---

## 1. The finding that shaped the plan

The obvious move for a harness week is to split the concierge into an agent plus
sub-agents, each with its own model. In this codebase that is strictly a cost
increase, and the code says so plainly.

The chat path has **exactly two LLM call sites**, both inside `_run_turn`
(`concierge_chat.py:1395` and `:1403`). Everything else is deterministic:
`_handle_highlights`, `menu_card.card_for_restaurant`, `_handle_recommend`,
`_handle_book`, and the whole profile-memory layer make no model calls at all. The
LLM calls in `reasoning.py` belong to the Studio pipeline in `graph.py`, which the
chat front-end does not use.

So there is no existing second model call to draw a sub-agent boundary around. Any
sub-agent given its own model is a new round trip per guest turn — more latency in a
live demo, more spend, and no new capability.

## 2. The thesis

**The sub-agents already exist. This week declares them; it does not add them.**

The nine handlers in `_HANDLERS` (`concierge_chat.py:1331`) are already five distinct
units of work with different privileges and different failure modes. They are simply
anonymous — their boundaries exist only in the author's head. The harness work makes
them named, declared in `.md`, and enforced. None of them gets a model, so nothing
gets slower or more expensive.

## 3. The cost invariant

The check every phase must pass before it ships:

| Measure | Before | After |
|---|---|---|
| Model calls per guest turn | 1 … `MAX_TOOL_HOPS`+1 | **identical** — no new call sites |
| System-prompt tokens | *N* | **identical bytes**, proven by a golden test |
| Tool-schema tokens | *M* | **≤ *M***, asserted in a test |
| Added latency | — | ~5 small file reads, once per process, module-cached |
| Added runtime per tool call | — | one dict lookup |

If a phase cannot hold this table, it does not ship.

## 4. The storyline: front of house, back of house

Dino is the **host**. The guest only ever talks to Dino. Behind the pass, specialists
do bounded jobs — they never greet the guest, never write to the member book, never
touch the reservation system unless that is their job.

The metaphor is load-bearing rather than decorative: it maps one-to-one onto the tool
grants, so least privilege gets to be described as staffing rather than as security
jargon. It also fits the product story already told in the scoping doc — a concierge
who remembers you and asks before assuming.

| Unit | Restaurant role | Owns today | Cannot |
|---|---|---|---|
| **Dino** | Host, front of house | the conversation, the journey, the consent question | — (the only unit with a model) |
| **Scout** | Reservations desk | `recommend_restaurants`, `check_availability_times` | book, write memory |
| **Curator** | Kitchen / menu research | `show_dining_highlights` + `menu_card` | book, write memory, address the guest |
| **Steward** | The maître d's book | `remember_guest_details`, `confirm_preference_updates`, `recall_guest_profile`, `set_confirmation_email` | search, book |
| **Booker** | The pass | `book_table`, `cancel_reservation` | search, write preferences |

Steward is where the consent gate shipped in `ab5008e` lives. The most recent piece of
work is therefore the centre of the harness story rather than a bolt-on.

## 5. Phase 1 — instructions become files (~1–2h)

```
src/table_for_four/agent/roster/
  __init__.py      # loader: parse frontmatter + body, cache at import
  dino.md          # host: persona, guardrails, journey, standing-prefs policy
  scout.md  curator.md  steward.md  booker.md
```

Move `SYSTEM_PROMPT` (`concierge_chat.py:70`) out of the Python string literal into
`dino.md`, split along its existing `##` sections. `start_session` assembles it exactly
as before.

**The guarantee.** Snapshot today's prompt to a golden file and assert
`build_system_prompt() == GOLDEN` byte for byte. That is a literal proof of zero token
change, and it demos well: the refactor is provably free.

Packaging needs no work — `mcp_servers/perks/fixtures/perks_seed.json` already ships as
package data under the same hatchling config (`packages = ["src/table_for_four"]`).

## 6. Phase 2 — grants become enforced (~2–3h)

Frontmatter carries the capability grant:

```yaml
---
name: curator
role: Back-of-house menu research
tools: [lookup_dining_highlights]
never: [create_booking, remember]
---
```

Two enforcement points, both free:

1. **`_dispatch` tags the acting unit.** `_dispatch` (`concierge_chat.py:1362`) is
   already the single chokepoint for every tool call, and already post-processes
   results to attach `known_so_far`. It gains a handler → unit lookup.
2. **A broker in `tools.py` checks the grant.** `create_booking` called while acting as
   `curator` raises, rather than merely never happening to be called.

The tests that then become writable: *the curator cannot book*, *the scout cannot write
a preference*, *the booker cannot search*. All three pass by construction today — the
point is that they would now **keep** passing.

**What this does and does not constrain.** Dino is the only unit with a model, so these
grants constrain *code paths*, not the model's imagination. What constrains the model is
the tool-schema list it is handed, which is a separate mechanism. The writeup should say
so explicitly; the distinction is what separates an understood harness from a decorated
one.

## 7. Phase 3 — close the loop, and set up M4 (~1–2h)

- **Tool descriptions sourced from the roster `.md`.** Each unit's body becomes the
  `description` for its tools in `TOOL_SCHEMAS`, making the `.md` files genuinely
  model-facing — through text that is already being sent. Guarded by an assertion that
  total schema characters do not exceed today's baseline.
- **`actor` on the audit line.** `audit_node` (`graph.py:191`) and the M4 governance
  trail record *which unit* acted. One field, and it is the join between this week's
  work and the M4 milestone.

## 8. Explicitly out of scope

No sub-agent gets its own model. No LangGraph restructuring. No prompt rewriting —
Phase 1 is a move, not an edit, and the golden test enforces that. Behaviour is
bit-identical at the end of Phase 1 and behaviour-identical after Phase 3.

## 9. Risks

- **Reads as cosmetic if the grants are not enforced.** Phase 2 is what makes it real;
  Phase 1 alone leaves a folder of markdown. Do not ship 1 without 2.
- **The golden test is load-bearing.** Editing the prompt in the same commit as the move
  destroys the proof. Move first, edit later, in separate commits.
- **Five units for nine handlers is near the ceiling** this codebase justifies. If a unit
  ends up owning a single handler and no distinct privilege, fold it into its neighbour.

## 10. Sequencing

Phase 1 alone is a complete, provable deliverable if the week runs short. Phases 1 → 2
are the minimum for the harness to be more than documentation. Phase 3 is what carries
into M4 governance.

---

## 11. What shipped

All three phases. `src/table_for_four/agent/roster/` holds five `.md` files and a
loader; the five units are declared, their grants are enforced at the tool registry
and the profile store, and the audit line names the actor.

**The cost table held, measured rather than asserted.**

| Measure | Before | After |
|---|---|---|
| LLM call sites in the chat path | 2 | 2 |
| System-prompt characters | 12,966 | 12,966 — golden test, byte for byte |
| Serialized tool-schema characters | 6,624 | 6,624 |
| Roster file reads | — | 5, at import |
| Added work per tool call | — | one `ContextVar` read, one set lookup |

Three implementation notes worth recording, because each is a place the plan met
the code and the code won.

**The broker sits in two modules, not one.** The plan put it in `tools.py`. But
`profile_memory`'s writes are effects too — arguably the ones a guest would mind
most — and they never pass through the tool registry. So `remember`, `adopt_email`
and `mark_booking` call `roster.require` directly. Reads stay ungoverned: the
harness constrains what a unit can *change*, not what it can look up.

**Outside a declared unit, nothing is checked.** `require()` returns immediately
when no unit is acting, so the perks eval script and the tests still call tools
directly. This is deliberate. The roster constrains the units it declares; it is
not a sandbox around the process, and a harness that quietly claimed to be one
would be the decorated version of this week's work.

**The Booker writes to the member book, and the roster says so.** §4's table had it
holding no memory capability. In the code, `_handle_book` files the reservation in
the guest's history and learns a first-ever cuisine or party size from it — an
actual booking being better evidence of taste than anything said in passing. Rather
than fictionalise the grant, `booker.md` grants `remember` and explains the line it
still cannot cross: `sticky_conflicts` strips standing preferences out before the
write and hands them back for Dino to ask about. So the guarantee that a birthday
dinner in Brooklyn doesn't move the guest to Brooklyn is enforced by the memory
layer, not the grant. The grant's job is narrower and still real — Scout and Curator
have no business in the member book at all, and now cannot reach it.

**What the tests can now say.** `tests/test_roster.py` (18 tests) asserts the two
cost invariants and the boundaries: the Curator cannot book or write memory, the
Scout cannot write a preference, the Booker cannot search or adopt an identity, the
Steward can do neither of the other two jobs, Dino holds nothing at all. Two more
guard the plumbing that makes those meaningful — every handler runs as the unit that
owns it, and a grant breach propagates instead of being swallowed into a polite chat
message. All of them passed the moment they were written; the point was never to
find a bug, it was to stop one from being possible.
