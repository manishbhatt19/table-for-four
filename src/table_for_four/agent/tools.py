"""In-process tool registry — the orchestrator's view of the MCP tool servers.

M3 calls the three MCP tools **in-process** (the same functions the stdio servers
expose) through this thin registry. That keeps the graph decoupled from tool
internals and fast/offline to test, while the tools remain standalone MCP servers.
A real stdio MCP-client backend can be swapped in here later without touching any
node code.

Because this is the agent's only route to the world, it is also the right place to
ask *who is reaching for it*. Each tool below is wrapped by the roster broker
(`agent.roster`): when a declared unit is acting, a capability it was not granted
raises `NotGranted` rather than merely never happening to be called. Outside any
unit — a test, the perks eval script — nothing is checked and the call proceeds.
The wrapper costs one set lookup.
"""

from __future__ import annotations

from table_for_four.agent import roster
from table_for_four.mcp_servers.booking.server import (
    cancel_booking as _cancel_booking,
    check_availability as _check_availability,
    create_booking as _create_booking,
    get_booking as _get_booking,
)
from table_for_four.mcp_servers.perks.server import find_perks as _find_perks
from table_for_four.mcp_servers.search.server import search_restaurants as _search_restaurants
from table_for_four.mcp_servers.web.server import (
    lookup_dining_highlights as _lookup_dining_highlights,
)

search_restaurants = roster.brokered("search_restaurants", _search_restaurants)
find_perks = roster.brokered("find_perks", _find_perks)
lookup_dining_highlights = roster.brokered("lookup_dining_highlights", _lookup_dining_highlights)
check_availability = roster.brokered("check_availability", _check_availability)
create_booking = roster.brokered("create_booking", _create_booking)
get_booking = roster.brokered("get_booking", _get_booking)
cancel_booking = roster.brokered("cancel_booking", _cancel_booking)

# The capability registry the orchestrator reasons about.
TOOLS = {
    "search_restaurants": search_restaurants,
    "find_perks": find_perks,
    "lookup_dining_highlights": lookup_dining_highlights,
    "check_availability": check_availability,
    "create_booking": create_booking,
    "get_booking": get_booking,
    "cancel_booking": cancel_booking,
}

__all__ = [
    "TOOLS",
    "search_restaurants",
    "find_perks",
    "lookup_dining_highlights",
    "check_availability",
    "create_booking",
    "get_booking",
    "cancel_booking",
]
