"""Shared test fixtures.

Unit tests must be deterministic and offline. Now that a real
`GOOGLE_PLACES_API_KEY` may live in `.env`, the search server would otherwise run
in **live** mode during tests — hitting the paid Places API and returning
non-deterministic data. This autouse fixture forces the offline fixture path for
every test by blanking the module-level key.

Also point the mock booking backend at an in-memory SQLite database so tests
never read or write the real `bookings.db` ledger. This env var must be set
before `mock_booking_api.app` is first imported; pytest imports this conftest
before collecting test modules, so it is.
"""

import os

os.environ.setdefault("BOOKING_DB_PATH", ":memory:")

import mcp_servers.search_server as search_server
import pytest


@pytest.fixture(autouse=True)
def _force_offline_search(monkeypatch):
    monkeypatch.setattr(search_server, "PLACES_API_KEY", "")
