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

from agent import reasoning
from agent.config import get_llm
from agent.state import ConciergeState
from agent.tools import check_availability, create_booking, find_perks, search_restaurants

MAX_ITERATIONS = 2  # refine-retry guard so the loop can never spin


def _log(state: ConciergeState, msg: str) -> list[str]:
    return list(state.get("log", [])) + [msg]


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
    out = find_perks(
        query=state["request"],
        place_ids=[r["place_id"] for r in candidates],
        party_size=c.get("party_size"),
        day=c.get("day"),
    )
    return {
        "perks": out["results"],
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
    return {"proposal": proposal, "log": _log(state, f"[propose] {r['name']} {note}")}


def gate_node(state: ConciergeState) -> dict[str, Any]:
    # M3 placeholder: auto-approve. M4 replaces this with a real human interrupt
    # and records the approval in the governance/audit trail.
    proposal = state.get("proposal")
    approved = bool(proposal and proposal.get("time"))
    msg = "auto-approved (M3 placeholder for human gate)" if approved else "nothing to approve"
    return {"approved": approved, "log": _log(state, f"[gate] {msg}")}


def route_after_gate(state: ConciergeState) -> Literal["book", "end"]:
    return "book" if state.get("approved") else "end"


def book_node(state: ConciergeState) -> dict[str, Any]:
    proposal = state["proposal"]
    result = create_booking(**proposal)
    if not result.get("booked"):
        return {"booking": result, "log": _log(state, f"[book] failed: {result.get('error')}")}
    chosen = state["chosen"]
    narrative = reasoning.narrate(
        chosen["restaurant"], chosen.get("perk"), result, llm=_select_llm(state)
    )
    return {
        "booking": result,
        "narrative": narrative,
        "log": _log(state, f"[book] confirmed {result['confirmation_id']}"),
    }


def audit_node(state: ConciergeState) -> dict[str, Any]:
    # M3: lightweight audit line. M4: full structured governance trail.
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
    }
    return {"log": _log(state, f"[audit] {json.dumps(summary)}")}


# --- Graph -------------------------------------------------------------------

def build_graph(checkpointer: Any | None = None):
    """Build and compile the concierge graph. Pass a checkpointer for persistence."""
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

    return g.compile(checkpointer=checkpointer or MemorySaver())


def run_concierge(
    request: str,
    guest_name: str = "Guest",
    thread_id: str = "demo",
    use_llm: bool | None = None,
) -> ConciergeState:
    """Run the full concierge loop for one request and return the final state.

    `use_llm=False` forces the deterministic heuristic path (used by tests);
    `None` auto-selects the LLM when a key is configured.
    """
    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    initial: ConciergeState = {
        "request": request,
        "guest_name": guest_name,
        "use_llm": use_llm,
        "log": [],
    }
    return graph.invoke(initial, config)
