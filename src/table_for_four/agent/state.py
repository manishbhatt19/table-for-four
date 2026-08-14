"""Working-memory state for the concierge graph.

This TypedDict IS the agent's working memory: it is carried across every node and
persisted by the checkpointer per conversation thread, which is what lets the
(future M4) human gate pause and resume. Nodes return partial updates that
LangGraph merges into this state.
"""

from __future__ import annotations

from typing import Any, TypedDict


class ConciergeState(TypedDict, total=False):
    # --- inputs ---
    request: str                       # raw natural-language dining request
    guest_name: str                    # name the reservation is held under
    use_llm: bool | None               # force heuristic (False) / auto (None)

    # --- parsed / retrieved ---
    constraints: dict[str, Any]        # parsed request: cuisine, party_size, date, ...
    candidates: list[dict[str, Any]]   # restaurants from search
    perks: list[dict[str, Any]]        # perks from find_perks

    # --- decision / action ---
    chosen: dict[str, Any] | None      # {"restaurant": ..., "perk": ...}
    proposal: dict[str, Any] | None    # booking proposal (pre-approval)
    approved: bool                     # human-gate outcome (auto in M3)
    booking: dict[str, Any] | None     # booking result

    # --- bookkeeping ---
    iterations: int                    # refine-retry counter (loop guard)
    reasoning_mode: str                # "llm" | "heuristic"
    narrative: str                     # human-facing summary
    log: list[str]                     # step-by-step trace
    actors: list[str]                  # roster units that acted, in order (audit trail)
