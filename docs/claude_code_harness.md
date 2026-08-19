# Table for Four — The Claude Code Harness

**Author:** Manish Bhatt · Committed in `1bb34b8`, Aug 13 2026

> Claude Code has been the development environment for this project from the start.
> This document records what was committed to make that configuration part of the
> repo rather than ambient setup on one laptop — the instruction file, the subagent
> roster, the demo skill, and the commit gate. Everything here is checked in and
> inspectable; nothing described below is a plan.

A note on the word, because the repo uses it twice. **This** harness is the
development-time one: how Claude Code is configured to work on the project.
`docs/week4_agentic_harness.md` describes a different thing — a roster of declared
units *inside* the product, at runtime — and is still unbuilt.

---

## 1. What was added

| Path | What it does |
|---|---|
| `CLAUDE.md` | Project instructions loaded into every session — commands, layout, invariants |
| `.claude/agents/concierge-consistency.md` | Read-only auditor for prompt/schema/handler drift |
| `.claude/agents/memory-invariants.md` | Verifies the consent and memory rules, runs the offline suite |
| `.claude/agents/submission-doc.md` | Drafts weekly capstone docs in house style; owns the PDF pipeline |
| `.claude/skills/demo/SKILL.md` | The three demo surfaces, the happy path, resetting between takes |
| `.claude/settings.json` | Permissions, a deny rule on the live `.env`, the pre-commit test gate |
| `.gitignore` | Excludes `settings.local.json` only — the shared config is committed on purpose |

## 2. `CLAUDE.md` — the instructions, in the repo

The file answers what a new agent session would otherwise have to rediscover by
grepping: the `uv` commands, where each thing lives, and the conventions the
codebase actually follows (comments explain *why* in prose; tests are named as
behaviour and open with the bug they guard; docs are weekly submissions).

Its centre is the six **invariants** — stated as the product rather than as
preferences, so a change that makes one awkward reads as a signal to stop rather
than something to route around:

1. The model never invents — not a restaurant, time, email, dish, or confirmation id.
2. The guest chooses at every branch point.
3. Standing preferences need consent, checked against the guest's own words.
4. Offline first — every path degrades to fixtures without an API key.
5. Dining only — the scope guardrail is load-bearing.
6. The backend owns policy, not the model — the 24 hour cancellation window is
   enforced in FastAPI, and Dino relays refusals verbatim.

It also carries the gotchas that are expensive to learn twice: `_run_turn` is the
only place the chat path calls a model and should stay that way; module paths are
fully qualified; Chroma stores are on disk beside their server and tests must use
ephemeral clients; `.env` is off limits and `.env.example` is the thing to read.

## 3. `.claude/agents/` — three subagents, each with its own tool grant

Each is a markdown file whose frontmatter declares `tools:` and `model:`. The grant
is the point: the two auditors cannot edit anything, by configuration rather than by
instruction.

**`concierge-consistency`** (`Read, Grep, Glob` — sonnet). `concierge_chat.py` holds
three descriptions of the same behaviour — the `SYSTEM_PROMPT` journey, the
`TOOL_SCHEMAS` the model can call, and the `_handle_*` functions that run. Nothing
in the code enforces agreement between them, so a tool renamed in one place, a
handler status the prompt never mentions, or a schema parameter the handler doesn't
read all ship silently and surface as strange behaviour mid-demo. The agent checks
those specific mismatches, reports `file:line` for both sides, and fixes nothing.

**`memory-invariants`** (`Read, Grep, Glob, Bash` — sonnet). Restates the six memory
rules — first values learned freely, standing preferences only via
`confirm_preference_updates`, consent from the guest's own words, only real cuisines
on file, a rolling window of three, email as identity key — then reads the changed
code *and* runs `tests/test_profile_memory.py`. It is told explicitly that green
tests are necessary but not sufficient, and to look for a new write path that
reaches around `sticky_conflicts` or a new cuisine source that skips
`_clean_cuisines`. Findings are reported as the guest scenario they produce.

**`submission-doc`** (`Read, Grep, Glob, Write, Edit, Bash` — opus). The only one
that writes. It encodes the house style for `docs/weekN_*.md` — title block, framing
blockquote, numbered sections, claims checkable by opening the file, unbuilt things
marked plainly — and the headless-Chrome PDF pipeline, including the instruction to
verify the rendered PDF rather than trust it, and to check the dash convention
before rendering.

## 4. `.claude/skills/demo/` — the recorded demo, written down

The demo is a graded deliverable, so the sequence that exercises every shipped
milestone in one unbroken take is captured rather than re-improvised: ask about a
prior visit *before* searching, take the email and the outing, shortlist with a RAG
perk visible, pick → times → confirm, menu card and cited highlights, the single
preference offer afterwards, then restart the session and be recognised by email.

It also records the two things that ruin a take: which stores to clear between
guests (`.chroma_profiles`, `bookings.db` — but leave `.chroma_perks` alone), and
running `uv run pytest -q` before recording, because a broken offline path shows up
on camera as a stack trace three minutes in.

## 5. `.claude/settings.json` — permissions and the commit gate

Allow-listed without prompting: `uv run pytest*`, `git status*`, `git diff*`,
`git log*`. Denied outright: reading or editing `./.env`, which holds live keys —
`.env.example` shows the shape instead.

The substantive piece is a `PreToolUse` hook matching `Bash(git commit *)`. It runs
the offline suite to `.git/tf4-precommit.log` and, on failure, emits a
`permissionDecision: "deny"` with a reason pointing at the log:

```json
"if": "Bash(git commit *)",
"command": "uv run pytest -q > .git/tf4-precommit.log 2>&1 || echo '{...\"permissionDecision\":\"deny\"...}'"
```

That gate is the same argument the product makes one level down: a consequential
action, a check that is not the model's judgement, and a refusal it cannot talk its
way past. It is the human-gate pattern of M4, applied to the repo itself.

## 6. Why commit any of it

Two reasons. It makes the harness **inspectable rather than claimed** — a grader or
a collaborator can read the tool grants and the deny rules instead of taking a
sentence in a writeup on faith. And it makes the invariants **survive the session**:
the consent rule, the offline-first rule, and the no-invention rule are now loaded
into every future agent context rather than living in the author's memory of what
the last conversation decided.

Personal overrides stay out via `.gitignore` (`settings.local.json`); the shared
configuration is committed deliberately.
