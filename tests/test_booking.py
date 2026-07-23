"""Offline tests for the mock booking backend and the booking MCP tools.

The backend is exercised directly through Starlette's TestClient (no network,
no port). The MCP tools are called as plain functions; with `BOOKING_API_URL`
unset they drive the same app in-process, so `backend` is reported as "mock".
"""

import pytest
from fastapi.testclient import TestClient

from mcp_servers import booking_server
from mock_booking_api.app import DINNER_SLOTS, app, available_slots, reset_store

PLACE = "fixture-osteria-1"


@pytest.fixture(autouse=True)
def _clean_store():
    reset_store()
    yield
    reset_store()


@pytest.fixture
def client():
    return TestClient(app)


def _first_open_slot(place: str, party_size: int) -> tuple[str, str]:
    """Find a date with at least one open slot; return (date, slot)."""
    for day in range(1, 29):
        date = f"2026-08-{day:02d}"
        slots = available_slots(place, date, party_size)
        if slots:
            return date, slots[0]
    raise AssertionError("no open slot found in range")


# --- Backend (direct) --------------------------------------------------------

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_availability_is_deterministic_subset(client):
    r1 = client.get("/availability", params={"place_id": PLACE, "date": "2026-08-01", "party_size": 2})
    r2 = client.get("/availability", params={"place_id": PLACE, "date": "2026-08-01", "party_size": 2})
    slots = r1.json()["available_slots"]
    assert r1.json() == r2.json()  # deterministic
    assert set(slots) <= set(DINNER_SLOTS)


def test_create_booking_success(client):
    date, slot = _first_open_slot(PLACE, 4)
    resp = client.post("/bookings", json={
        "place_id": PLACE, "restaurant_name": "Osteria Midtown",
        "date": date, "time": slot, "party_size": 4, "guest_name": "Manish",
    })
    assert resp.status_code == 201
    body = resp.json()
    assert body["confirmation_id"].startswith("TF4-")
    assert body["status"] == "confirmed"


def test_booking_unavailable_slot_conflicts(client):
    # A lunch time is never in the dinner-slot set -> always unavailable.
    resp = client.post("/bookings", json={
        "place_id": PLACE, "restaurant_name": "Osteria Midtown",
        "date": "2026-08-01", "time": "12:00", "party_size": 2, "guest_name": "Manish",
    })
    assert resp.status_code == 409


def test_booking_roundtrip(client):
    date, slot = _first_open_slot(PLACE, 2)
    created = client.post("/bookings", json={
        "place_id": PLACE, "restaurant_name": "Osteria Midtown",
        "date": date, "time": slot, "party_size": 2, "guest_name": "Manish",
    }).json()
    fetched = client.get(f"/bookings/{created['confirmation_id']}").json()
    assert fetched == created


def test_party_size_validation(client):
    resp = client.get("/availability", params={"place_id": PLACE, "date": "2026-08-01", "party_size": 0})
    assert resp.status_code == 422


# --- Booking MCP tools (in-process) ------------------------------------------

def test_tool_check_availability_reports_mock_backend():
    out = booking_server.check_availability(PLACE, "2026-08-01", party_size=2)
    assert out["backend"] == "mock"
    assert set(out["available_slots"]) <= set(DINNER_SLOTS)


def test_tool_create_then_get():
    date, slot = _first_open_slot(PLACE, 2)
    booked = booking_server.create_booking(
        place_id=PLACE, restaurant_name="Osteria Midtown",
        date=date, time=slot, party_size=2, guest_name="Manish",
        perk_id="perk-osteria-02",
    )
    assert booked["booked"] is True
    conf = booked["confirmation_id"]
    fetched = booking_server.get_booking(conf)
    assert fetched["confirmation_id"] == conf
    assert fetched["perk_id"] == "perk-osteria-02"


def test_tool_create_unavailable_returns_structured_error():
    out = booking_server.create_booking(
        place_id=PLACE, restaurant_name="Osteria Midtown",
        date="2026-08-01", time="12:00", party_size=2, guest_name="Manish",
    )
    assert out["booked"] is False
    assert out["error"] == "slot_unavailable"


def test_tool_get_missing_booking():
    assert booking_server.get_booking("TF4-9999")["found"] is False
