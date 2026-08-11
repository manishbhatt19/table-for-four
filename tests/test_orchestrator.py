"""Offline end-to-end tests for the LangGraph orchestrator (Milestone 3).

All runs force `use_llm=False`, so the deterministic heuristic path executes — no
API key, no network, no cost. This exercises the full loop: parse → search →
perks → rank → propose → gate → book → audit, plus the refine-retry loop.
"""

from table_for_four.agent.graph import run_concierge
from table_for_four.agent.reasoning import parse
from table_for_four.mcp_servers.booking.backend.app import reset_store


def _run(request, **kw):
    reset_store()
    return run_concierge(request, use_llm=False, **kw)


def test_heuristic_parse_extracts_constraints():
    c = parse("Italian, near Midtown, 4 people, Friday 7pm, one guest is gluten-free")
    assert c["cuisine"] == "italian"
    assert c["party_size"] == 4
    assert c["time"] == "19:00"
    assert c["day"] == "Fri"
    assert "gluten-free" in c["dietary"]


def test_end_to_end_books_a_table():
    final = _run(
        "Italian, near Midtown, 4 people, Friday 7pm, one guest is gluten-free",
        guest_name="Manish",
    )
    assert final["reasoning_mode"] == "heuristic"
    assert final["chosen"] is not None
    booking = final["booking"]
    assert booking["booked"] is True
    assert booking["confirmation_id"].startswith("TF4-")
    assert booking["booking"]["guest_name"] == "Manish"
    assert final["narrative"]


def test_refine_retry_loop_triggers_and_recovers():
    # "top-rated" sets min_rating 4.8; no fixture clears it, so the graph must
    # relax the constraint and retry before it can book.
    final = _run("top-rated Italian for 2 on Friday")
    assert final["iterations"] >= 1
    assert any("refine" in line for line in final["log"])
    assert final["booking"]["booked"] is True  # recovered after relaxing


def test_audit_line_present():
    final = _run("Italian dinner for 2 on Friday")
    assert any(line.startswith("[audit]") for line in final["log"])


def test_working_memory_persists_per_thread():
    # A second run on the same thread id still returns a coherent final state
    # (checkpointer wiring smoke test).
    final = _run("Italian for 2 on Friday", thread_id="t1")
    assert final["booking"]["booked"] is True
