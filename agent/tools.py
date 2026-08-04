"""In-process tool registry — the orchestrator's view of the MCP tool servers.

M3 calls the three MCP tools **in-process** (the same functions the stdio servers
expose) through this thin registry. That keeps the graph decoupled from tool
internals and fast/offline to test, while the tools remain standalone MCP servers.
A real stdio MCP-client backend can be swapped in here later without touching any
node code.
"""

from __future__ import annotations

from mcp_servers.booking_server import (
    cancel_booking,
    check_availability,
    create_booking,
    get_booking,
)
from mcp_servers.perks_server import find_perks
from mcp_servers.search_server import search_restaurants

# The capability registry the orchestrator reasons about.
TOOLS = {
    "search_restaurants": search_restaurants,
    "find_perks": find_perks,
    "check_availability": check_availability,
    "create_booking": create_booking,
    "get_booking": get_booking,
    "cancel_booking": cancel_booking,
}

__all__ = [
    "TOOLS",
    "search_restaurants",
    "find_perks",
    "check_availability",
    "create_booking",
    "get_booking",
    "cancel_booking",
]
