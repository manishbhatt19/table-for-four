"""The concierge orchestrator — a LangGraph state machine.

Flow:
    parse -> search -> (refine ⟲ search)* -> match_perks -> rank
          -> propose -> gate -> book -> audit

Three module concepts made concrete:
* **Tools:** each node calls the in-process tool registry (`agent.tools`).
* **Memory:** `ConciergeState` is working memory, persisted by a `MemorySaver`
  checkpointer keyed per conversation thread (enables the future human-gate
  pause/resume).
* **Reasoning loop:** a conditional edge lets `search` **refine and retry** when a
  query returns nothing, bounded by `MAX_ITERATIONS`.

The `gate` node auto-approves in M3 (placeholder); M4 swaps it for a real human
`interrupt` plus the governance/audit trail. `audit` is a lightweight log for now.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from table_for_four.agent import reasoning, roster
from table_for_four.governance import audit
from table_for_four.agent.config import get_llm
from table_for_four.agent.state import ConciergeState
from table_for_four.agent.tools import check_availability, create_booking, find_perks, search_restaurants

MAX_ITERATIONS = 2  # refine-retry guard so the loop can never spin


def _log(state: ConciergeState, msg: str) -> list[str]:
    return list(state.get("log", [])) + [msg]


def _audit(state: ConciergeState, event: str, *, actor: str | None, **detail: Any) -> list[dict[str, Any]]:
    """Append one governance record to the run's trail, never rewriting an earlier one."""
    return list(state.get("audit", [])) + [
        audit.emit(event, actor=actor, member_id=state.get("guest_name"), **detail)
    ]


# What counts as a yes. Deliberately narrow and deliberately explicit: anything
# this doesn't recognise is a refusal, so a malformed resume can't book a table.
_YES = {"y", "yes", "approve", "approved", "confirm", "confirmed", "ok", "book it", "true"}


def _is_approval(decision: Any) -> bool:
    if isinstance(decision, bool):
        return decision
    if isinstance(decision, str):
        return decision.strip().lower() in _YES
    if isinstance(decision, dict):
        if isinstance(decision.get("approved"), bool):
            return decision["approved"]
        return _is_approval(decision.get("decision"))
    return False


def _describe(decision: Any) -> Any:
    """A record of what the human actually sent, kept JSON-shaped for the trail."""
    return decision if isinstance(decision, (bool, str, int, float, type(None))) else str(decision)


def _acted(state: ConciergeState, unit: str) -> list[str]:
    """Record that a roster unit did something in this run, first appearance wins.

    The graph reaches the world through the same brokered registry the chat path
    does, so the nodes that call a tool declare which unit they are acting as. That
    makes "who did this?" answerable from the state rather than inferred from the
    node name — which is precisely the field the M4 governance trail needs.
    """
    prior = list(state.get("actors", []))
    return prior if unit in prior else prior + [unit]


def _select_llm(state: ConciergeState) -> Any | None:
    """LLM if configured and not force-disabled for this run."""
    if state.get("use_llm") is False:
        return None
    return get_llm()


# --- Nodes -------------------------------------------------------------------

def parse_node(state: ConciergeState) -> dict[str, Any]:
    llm = _select_llm(state)
    constraints = reasoning.parse(state["request"], llm=llm)
    mode = "llm" if llm is not None else "heuristic"
    return {
        "constraints": constraints,
        "iterations": 0,
        "reasoning_mode": mode,
        "log": _log(state, f"[parse:{mode}] {constraints}"),
    }


def search_node(state: ConciergeState) -> dict[str, Any]:
    c = state["constraints"]
    with roster.acting_as("scout"):
        out = search_restaurants(
            query=state["request"],
            cuisine=c.get("cuisine"),
            location=c.get("location"),
            max_price_level=c.get("max_price_level"),
            min_rating=c.get("min_rating"),
            open_now=c.get("open_now"),
        )
    return {
        "candidates": out["results"],
        "actors": _acted(state, "scout"),
        "log": _log(state, f"[search:{out['source']}] {out['result_count']} candidates"),
    }


def refine_node(state: ConciergeState) -> dict[str, Any]:
    """Relax one constraint and try again — the reasoning loop's body."""
    c = dict(state["constraints"])
    note = "broadened query"
    if c.get("min_rating") is not None:
        c["min_rating"] = None
        note = "dropped min_rating"
    elif c.get("max_price_level") is not None:
        c["max_price_level"] = min(4, c["max_price_level"] + 1)
        note = "raised price ceiling"
    elif c.get("open_now"):
        c["open_now"] = None
        note = "dropped open-now"
    elif c.get("cuisine"):
        c["cuisine"] = None
        note = "dropped cuisine filter"
    return {
        "constraints": c,
        "iterations": state.get("iterations", 0) + 1,
        "log": _log(state, f"[refine #{state.get('iterations', 0) + 1}] {note}"),
    }


def route_after_search(state: ConciergeState) -> Literal["refine", "match_perks"]:
    if state.get("candidates"):
        return "match_perks"
    if state.get("iterations", 0) < MAX_ITERATIONS:
        return "refine"
    return "match_perks"  # give up refining; downstream handles empty gracefully


def match_perks_node(state: ConciergeState) -> dict[str, Any]:
    c = state["constraints"]
    candidates = state.get("candidates", [])
    if not candidates:
        return {"perks": [], "log": _log(state, "[perks] skipped (no candidates)")}
    with roster.acting_as("scout"):
        out = find_perks(
            query=state["request"],
            place_ids=[r["place_id"] for r in candidates],
            party_size=c.get("party_size"),
            day=c.get("day"),
        )
    return {
        "perks": out["results"],
        "actors": _acted(state, "scout"),
        "log": _log(state, f"[perks:{out['source']}] {out['result_count']} matched"),
    }


def rank_node(state: ConciergeState) -> dict[str, Any]:
    candidates = state.get("candidates", [])
    if not candidates:
        return {"chosen": None, "log": _log(state, "[rank] nothing to rank")}
    perks = state.get("perks", [])
    best = reasoning.choose_best(candidates, perks)
    perk = reasoning.best_perk_for(best["place_id"], perks)
    line = f"[rank] chose {best['name']}"
    if perk:
        line += f" + perk '{perk['title']}'"
    return {"chosen": {"restaurant": best, "perk": perk}, "log": _log(state, line)}


def propose_node(state: ConciergeState) -> dict[str, Any]:
    chosen = state.get("chosen")
    if not chosen:
        return {"proposal": None, "log": _log(state, "[propose] no restaurant to book")}
    r = chosen["restaurant"]
    c = state["constraints"]
    party_size = c.get("party_size", 2)
    with roster.acting_as("scout"):
        avail = check_availability(r["place_id"], c["date"], party_size)
    slots = avail.get("available_slots", [])
    requested = c.get("time")
    time = requested if requested in slots else (slots[0] if slots else None)
    proposal = {
        "place_id": r["place_id"],
        "restaurant_name": r["name"],
        "date": c["date"],
        "time": time,
        "party_size": party_size,
        "guest_name": state.get("guest_name", "Guest"),
        "perk_id": (chosen.get("perk") or {}).get("perk_id"),
        "special_requests": c.get("special_requests"),
    }
    note = f"at {time} on {c['date']}" if time else "but no slots are open"
    return {
        "proposal": proposal,
        "actors": _acted(state, "scout"),
        "log": _log(state, f"[propose] {r['name']} {note}"),
    }


def gate_node(state: ConciergeState) -> dict[str, Any]:
    """Stop, show a person what is about to happen, and wait to be told.

    This is the milestone. Everything before it is reversible — a search, a
    shortlist, a proposal — and `create_booking` is the one step that isn't, so
    it is the one step the agent may not take on its own say-so. The node
    interrupts: LangGraph checkpoints the state and the run genuinely stops here
    until a caller resumes it with a decision.

    Two properties worth stating, because they are what make this a gate rather
    than a prompt. The model is not asked and cannot answer — the decision
    arrives from outside the graph. And a resume that isn't clearly a yes is a
    no: silence, an empty resume, anything unparsed all deny, because the safe
    direction to be wrong in is the one where no table gets booked.
    """
    proposal = state.get("proposal")
    if not proposal or not proposal.get("time"):
        return {
            "approved": False,
            "log": _log(state, "[gate] nothing to approve"),
            "audit": _audit(state, "approval", actor=None,
                            outcome="nothing_to_approve"),
        }

    chosen = state.get("chosen") or {}
    perk = (chosen.get("perk") or {})
    request = {
        "question": "Confirm this reservation?",
        "restaurant": proposal.get("restaurant_name"),
        "date": proposal.get("date"),
        "time": proposal.get("time"),
        "party_size": proposal.get("party_size"),
        "guest_name": proposal.get("guest_name"),
        "perk": perk.get("title") or perk.get("perk_id"),
        "irreversible": True,
    }
    decision = interrupt(request)
    approved = _is_approval(decision)

    return {
        "approved": approved,
        "approval": {"shown": request, "decision": decision, "approved": approved},
        "log": _log(state, f"[gate] {'approved' if approved else 'declined'} by human"),
        "audit": _audit(state, "approval", actor=None, shown=request,
                        decision=_describe(decision), approved=approved),
    }


def route_after_gate(state: ConciergeState) -> Literal["book", "end"]:
    return "book" if state.get("approved") else "end"


def book_node(state: ConciergeState) -> dict[str, Any]:
    proposal = state["proposal"]
    with roster.acting_as("booker"):
        result = create_booking(**proposal)
    if not result.get("booked"):
        return {
            "booking": result,
            "actors": _acted(state, "booker"),
            "log": _log(state, f"[book] failed: {result.get('error')}"),
        }
    chosen = state["chosen"]
    narrative = reasoning.narrate(
        chosen["restaurant"], chosen.get("perk"), result, llm=_select_llm(state)
    )
    return {
        "booking": result,
        "narrative": narrative,
        "actors": _acted(state, "booker"),
        "log": _log(state, f"[book] confirmed {result['confirmation_id']}"),
    }


def audit_node(state: ConciergeState) -> dict[str, Any]:
    """Close the run with a record of what happened and who is answerable for it."""
    booking = state.get("booking") or {}
    chosen = state.get("chosen") or {}
    summary = {
        "request": state["request"],
        "reasoning_mode": state.get("reasoning_mode"),
        "refine_iterations": state.get("iterations", 0),
        "chosen": (chosen.get("restaurant") or {}).get("name"),
        "perk_applied": (chosen.get("perk") or {}).get("perk_id"),
        "confirmation_id": booking.get("confirmation_id"),
        "booking_backend": booking.get("backend"),
        # Who acted, not just what happened. An audit line that cannot name the
        # actor is a diary; this is the field M4's governance trail is built on.
        "actors": state.get("actors", []),
        # A booking with no approval behind it is the thing this trail exists to
        # make impossible to miss, so it is stated rather than inferred.
        "approved_by_human": bool((state.get("approval") or {}).get("approved")),
    }
    return {
        "log": _log(state, f"[audit] {json.dumps(summary)}"),
        "audit": _audit(state, "run_complete", actor=None, **summary),
    }


# --- Graph -------------------------------------------------------------------

def build_state_graph() -> StateGraph:
    """Assemble the (uncompiled) concierge StateGraph — nodes and edges."""
    g = StateGraph(ConciergeState)
    g.add_node("parse", parse_node)
    g.add_node("search", search_node)
    g.add_node("refine", refine_node)
    g.add_node("match_perks", match_perks_node)
    g.add_node("rank", rank_node)
    g.add_node("propose", propose_node)
    g.add_node("gate", gate_node)
    g.add_node("book", book_node)
    g.add_node("audit", audit_node)

    g.add_edge(START, "parse")
    g.add_edge("parse", "search")
    g.add_conditional_edges("search", route_after_search,
                            {"refine": "refine", "match_perks": "match_perks"})
    g.add_edge("refine", "search")           # the reasoning loop
    g.add_edge("match_perks", "rank")
    g.add_edge("rank", "propose")
    g.add_edge("propose", "gate")
    g.add_conditional_edges("gate", route_after_gate, {"book": "book", "end": END})
    g.add_edge("book", "audit")
    g.add_edge("audit", END)
    return g


def build_graph(checkpointer: Any | None = None):
    """Build and compile the concierge graph. Pass a checkpointer for persistence."""
    return build_state_graph().compile(checkpointer=checkpointer or MemorySaver())


# Entry point for LangGraph Studio / `langgraph dev`: the platform supplies its own
# persistence, so we hand it the uncompiled builder (no in-code checkpointer).
studio_graph = build_state_graph()


def run_concierge(
    request: str,
    guest_name: str = "Guest",
    thread_id: str = "demo",
    use_llm: bool | None = None,
    approve: Any = None,
) -> ConciergeState:
    """Run the full concierge loop for one request and return the final state.

    `use_llm=False` forces the deterministic heuristic path (used by tests);
    `None` auto-selects the LLM when a key is configured.

    `approve` answers the human gate, and there is no default that books:

    * a callable — handed the proposal, returns the decision (the CLI passes one
      that asks the person at the keyboard);
    * `True` / `False` — a standing answer, for tests and unattended runs;
    * `None` — nobody is there to ask, so the booking is declined and the trail
      records that it was never approved rather than that it was refused.

    Leaving this out cannot silently book a table. That is the entire point of
    the gate, and a convenient default would quietly undo it.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: ConciergeState = {
        "request": request,
        "guest_name": guest_name,
        "use_llm": use_llm,
        "log": [],
        "audit": [],
    }
    state = graph.invoke(initial, config)

    # The gate genuinely stops the run. Resume it with whatever the caller's
    # approver says — looping because a graph may interrupt more than once.
    while (pending := state.get("__interrupt__")):
        payload = getattr(pending[0], "value", {})
        decision = approve(payload) if callable(approve) else approve
        # A bare `None` cannot be resumed on — LangGraph raises inside its own
        # loop — and "nobody answered" is a refusal in any case, so say so.
        state = graph.invoke(Command(resume=False if decision is None else decision), config)
    return state
