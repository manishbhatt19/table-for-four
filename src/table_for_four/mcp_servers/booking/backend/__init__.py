"""Mock reservation backend for Table for Four.

A self-built FastAPI service standing in for a partner-gated reservation system
(OpenTable/Resy/SevenRooms). Deterministic and in-memory so the transactional
booking path is fully testable offline.
"""

from table_for_four.mcp_servers.booking.backend.app import app, reset_store

__all__ = ["app", "reset_store"]
