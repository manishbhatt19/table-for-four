"""Booking MCP server — reservation tools over the mock backend.

Exposes the transactional half of the concierge: check availability, create a
booking, and fetch one. Backed by the self-built FastAPI reservation service
(`mock_booking_api`).

Live/offline duality (mirrors the search server):
* If `BOOKING_API_URL` is set, tools call that running service over HTTP.
* Otherwise they drive the FastAPI app **in-process** via Starlette's TestClient —
  a real request/response through the ASGI stack, no port, no second process, and
  fully offline-testable. The same tool code runs in both modes.

Each result carries a `backend` field ("live" | "mock") so the governance/audit
layer can record how a booking was made. `create_booking` is the system's one
irreversible write; in the full agent it is gated behind human approval (M4).

Run standalone (stdio transport):
    uv run mcp_servers/booking_server.py
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()

BOOKING_API_URL = os.getenv("BOOKING_API_URL", "").strip()

mcp = FastMCP("restaurant-booking")

_client: Any = None
_backend: str = "mock"


def _get_client() -> Any:
    """Lazily build the HTTP client: live httpx or in-process TestClient."""
    global _client, _backend
    if _client is None:
        if BOOKING_API_URL:
            _backend = "live"
            _client = httpx.Client(base_url=BOOKING_API_URL, timeout=15.0)
        else:
            _backend = "mock"
            from fastapi.testclient import TestClient

            from mock_booking_api.app import app

            _client = TestClient(app)
    return _client


# --- Tools -------------------------------------------------------------------

@mcp.tool()
def check_availability(place_id: str, date: str, party_size: int = 2) -> dict[str, Any]:
    """Check open reservation slots for a restaurant on a date.

    Args:
        place_id: Restaurant id (as returned by `search_restaurants`).
        date: Reservation date, `YYYY-MM-DD`.
        party_size: Number of guests (1-20).

    Returns:
        A dict with `backend` ("live"|"mock"), the echoed request, and
        `available_slots` (list of `HH:MM` times still open).
    """
    client = _get_client()
    resp = client.get(
        "/availability",
        params={"place_id": place_id, "date": date, "party_size": party_size},
    )
    resp.raise_for_status()
    data = resp.json()
    data["backend"] = _backend
    return data


@mcp.tool()
def create_booking(
    place_id: str,
    restaurant_name: str,
    date: str,
    time: str,
    party_size: int,
    guest_name: str,
    perk_id: str | None = None,
    special_requests: str | None = None,
    address: str | None = None,
    restaurant_phone: str | None = None,
    website: str | None = None,
    guest_email: str | None = None,
) -> dict[str, Any]:
    """Create a reservation (the irreversible write).

    Args:
        place_id: Restaurant id.
        restaurant_name: Human-readable restaurant name (for the confirmation).
        date: `YYYY-MM-DD`.
        time: `HH:MM`, must be one of the restaurant's open slots for that date.
        party_size: Number of guests (1-20).
        guest_name: Name the reservation is held under.
        perk_id: Optional perk/offer being applied (from `find_perks`).
        special_requests: Optional free-text note (e.g. dietary needs).
        address: Restaurant address, saved to the ledger (from `search_restaurants`).
        restaurant_phone: Restaurant phone, saved for direct-cancellation contact.
        website: Restaurant website, saved for direct-cancellation contact.
        guest_email: Guest email the reservation is booked under (ledger key).

    Returns:
        On success: `{backend, booked: true, confirmation_id, booking}`.
        If the slot is not available: `{backend, booked: false, error:
        "slot_unavailable", detail}` — so the agent can pick another time rather
        than crash.
    """
    client = _get_client()
    payload = {
        "place_id": place_id,
        "restaurant_name": restaurant_name,
        "date": date,
        "time": time,
        "party_size": party_size,
        "guest_name": guest_name,
        "perk_id": perk_id,
        "special_requests": special_requests,
        "address": address,
        "restaurant_phone": restaurant_phone,
        "website": website,
        "guest_email": guest_email,
    }
    resp = client.post("/bookings", json=payload)

    if resp.status_code == 409:
        return {
            "backend": _backend,
            "booked": False,
            "error": "slot_unavailable",
            "detail": resp.json().get("detail"),
        }
    resp.raise_for_status()
    booking = resp.json()
    return {
        "backend": _backend,
        "booked": True,
        "confirmation_id": booking["confirmation_id"],
        "booking": booking,
    }


@mcp.tool()
def get_booking(confirmation_id: str) -> dict[str, Any]:
    """Fetch a previously created reservation by its confirmation id.

    Returns the booking dict (with `backend`), or `{backend, found: false}` if no
    such booking exists.
    """
    client = _get_client()
    resp = client.get(f"/bookings/{confirmation_id}")
    if resp.status_code == 404:
        return {"backend": _backend, "found": False}
    resp.raise_for_status()
    booking = resp.json()
    booking["backend"] = _backend
    return booking


@mcp.tool()
def cancel_booking(
    confirmation_id: str, reason: str | None = None, now: str | None = None
) -> dict[str, Any]:
    """Cancel a reservation, subject to the 24-hour policy (enforced by the backend).

    A booking can be cancelled only while it is more than 24 hours away. Inside
    that window the backend refuses and returns the restaurant's contact details so
    the guest can call directly — the agent must relay those, not retry.

    Args:
        confirmation_id: The reservation to cancel.
        reason: Optional free-text cancellation reason, recorded in the ledger.
        now: Optional ISO datetime override for the 24h check (testing only).

    Returns:
        - Cancelled: `{backend, status: "cancelled", booking}`.
        - Too late: `{backend, status: "too_late", restaurant_name,
          restaurant_phone, website, hours_until_booking, message}` — the agent
          should tell the guest to contact the restaurant directly.
        - `{backend, status: "already_cancelled" | "not_found"}`.
    """
    client = _get_client()
    body: dict[str, Any] = {}
    if reason is not None:
        body["reason"] = reason
    if now is not None:
        body["now"] = now
    resp = client.post(f"/bookings/{confirmation_id}/cancel", json=body)

    if resp.status_code == 404:
        return {"backend": _backend, "status": "not_found"}
    if resp.status_code == 409:
        detail = resp.json().get("detail", {})
        if detail.get("error") == "already_cancelled":
            return {"backend": _backend, "status": "already_cancelled",
                    "cancelled_at": detail.get("cancelled_at")}
        # within_24h: surface the restaurant's own contact channels.
        return {
            "backend": _backend,
            "status": "too_late",
            "message": detail.get("message"),
            "hours_until_booking": detail.get("hours_until_booking"),
            "restaurant_name": detail.get("restaurant_name"),
            "restaurant_phone": detail.get("restaurant_phone"),
            "website": detail.get("website"),
        }
    resp.raise_for_status()
    booking = resp.json()
    return {"backend": _backend, "status": "cancelled", "booking": booking}


if __name__ == "__main__":
    mcp.run()
