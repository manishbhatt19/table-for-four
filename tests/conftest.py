"""Shared test fixtures.

Unit tests must be deterministic and offline. Now that a real
`GOOGLE_PLACES_API_KEY` may live in `.env`, the search server would otherwise run
in **live** mode during tests — hitting the paid Places API and returning
non-deterministic data. This autouse fixture forces the offline fixture path for
every test by blanking the module-level key.
"""

import mcp_servers.search_server as search_server
import pytest


@pytest.fixture(autouse=True)
def _force_offline_search(monkeypatch):
    monkeypatch.setattr(search_server, "PLACES_API_KEY", "")
