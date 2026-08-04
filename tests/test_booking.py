"""Offline tests for the mock booking backend and the booking MCP tools.

The backend is exercised directly through Starlette's TestClient (no network,
no port). The MCP tools are called as plain functions; with `BOOKING_API_URL`
unset they drive the same app in-process, so `backend` is reported as "mock".
"""

from datetime import datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from mcp_servers import booking_server
from mock_booking_api.app import (
    LUNCH_SLOTS,
    SERVICE_SLOTS,
    app,
    available_slots,
    reset_store,
)

PLACE = "fixture-osteria-1"
NEVER_A_SLOT = "23:30"  # outside lunch and dinner service -> never bookable
PHONE = "(212) 555-0142"
WEBSITE = "https://example.com/osteria-midtown"


def _book(client, party_size=2, **extra):
    """Create a booking on the first open slot; return its response dict."""
    date, slot = _first_open_slot(PLACE, party_size)
    payload = {
        "place_id": PLACE, "restaurant_name": "Osteria Midtown",
        "date": date, "time": slot, "party_size": party_size, "guest_name": "Manish",
        "restaurant_phone": PHONE, "website": WEBSITE, **extra,
    }
    return client.post("/bookings", json=payload).json()


def _clock(date: str, time: str, hours_before: float) -> str:
    """An ISO 'now' a given number of hours before a booking's date/time."""
    return (datetime.fromisoformat(f"{date}T{time}") - timedelta(hours=hours_before)).isoformat()


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
    assert set(slots) <= set(SERVICE_SLOTS)


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
    # A time outside all service hours is never bookable -> always a conflict.
    resp = client.post("/bookings", json={
        "place_id": PLACE, "restaurant_name": "Osteria Midtown",
        "date": "2026-08-01", "time": NEVER_A_SLOT, "party_size": 2, "guest_name": "Manish",
    })
    assert resp.status_code == 409


def test_lunch_slots_are_offered(client):
    # Regression for demo feedback: availability must include lunch sittings, not
    # only dinner — across the month at least one lunch time is open somewhere.
    seen_lunch = set()
    for day in range(1, 29):
        slots = client.get("/availability", params={
            "place_id": PLACE, "date": f"2026-08-{day:02d}", "party_size": 2,
        }).json()["available_slots"]
        seen_lunch |= set(slots) & set(LUNCH_SLOTS)
    assert seen_lunch, "expected lunch times to be offered somewhere in the month"


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
    assert set(out["available_slots"]) <= set(SERVICE_SLOTS)


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
        date="2026-08-01", time=NEVER_A_SLOT, party_size=2, guest_name="Manish",
    )
    assert out["booked"] is False
    assert out["error"] == "slot_unavailable"


def test_tool_get_missing_booking():
    assert booking_server.get_booking("TF4-9999")["found"] is False


# --- Persistence + cancellation policy (backend) -----------------------------

def test_booking_persists_ledger_fields(client):
    b = _book(client, party_size=2, address="127 W 44th St", guest_email="g@x.com")
    fetched = client.get(f"/bookings/{b['confirmation_id']}").json()
    assert fetched["status"] == "confirmed"
    assert fetched["address"] == "127 W 44th St"
    assert fetched["guest_email"] == "g@x.com"
    assert fetched["restaurant_phone"] == PHONE
    assert fetched["created_at"] and fetched["cancelled_at"] is None


def test_cancel_more_than_24h_ahead_succeeds(client):
    b = _book(client)
    now = _clock(b["date"], b["time"], hours_before=48)
    resp = client.post(f"/bookings/{b['confirmation_id']}/cancel", json={"now": now, "reason": "plans changed"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "cancelled"
    assert body["cancelled_at"] and body["cancellation_reason"] == "plans changed"
    # Persisted: a re-fetch shows it cancelled.
    assert client.get(f"/bookings/{b['confirmation_id']}").json()["status"] == "cancelled"


def test_cancel_within_24h_refused_with_restaurant_contact(client):
    b = _book(client)
    now = _clock(b["date"], b["time"], hours_before=10)
    resp = client.post(f"/bookings/{b['confirmation_id']}/cancel", json={"now": now})
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["error"] == "within_24h"
    assert detail["restaurant_phone"] == PHONE
    assert detail["website"] == WEBSITE
    # And the booking is left confirmed — no partial cancellation.
    assert client.get(f"/bookings/{b['confirmation_id']}").json()["status"] == "confirmed"


def test_cancel_is_idempotent_guarded(client):
    b = _book(client)
    now = _clock(b["date"], b["time"], hours_before=48)
    first = client.post(f"/bookings/{b['confirmation_id']}/cancel", json={"now": now})
    second = client.post(f"/bookings/{b['confirmation_id']}/cancel", json={"now": now})
    assert first.status_code == 200
    assert second.status_code == 409
    assert second.json()["detail"]["error"] == "already_cancelled"


def test_cancel_missing_booking_404(client):
    assert client.post("/bookings/TF4-9999/cancel", json={}).status_code == 404


def test_list_bookings_filters_by_email_and_status(client):
    _book(client, guest_email="a@x.com")
    b2 = _book(client, guest_email="b@x.com")
    now = _clock(b2["date"], b2["time"], hours_before=48)
    client.post(f"/bookings/{b2['confirmation_id']}/cancel", json={"now": now})

    mine = client.get("/bookings", params={"email": "a@x.com"}).json()
    assert [b["guest_email"] for b in mine] == ["a@x.com"]
    cancelled = client.get("/bookings", params={"status": "cancelled"}).json()
    assert [b["confirmation_id"] for b in cancelled] == [b2["confirmation_id"]]


# --- Cancellation via the MCP tool -------------------------------------------

def test_tool_cancel_success():
    date, slot = _first_open_slot(PLACE, 2)
    booked = booking_server.create_booking(
        place_id=PLACE, restaurant_name="Osteria Midtown", date=date, time=slot,
        party_size=2, guest_name="Manish", restaurant_phone=PHONE, website=WEBSITE,
    )
    conf = booked["confirmation_id"]
    out = booking_server.cancel_booking(conf, now=_clock(date, slot, 48))
    assert out["status"] == "cancelled"
    assert out["booking"]["status"] == "cancelled"


def test_tool_cancel_too_late_returns_contact():
    date, slot = _first_open_slot(PLACE, 2)
    booked = booking_server.create_booking(
        place_id=PLACE, restaurant_name="Osteria Midtown", date=date, time=slot,
        party_size=2, guest_name="Manish", restaurant_phone=PHONE, website=WEBSITE,
    )
    out = booking_server.cancel_booking(booked["confirmation_id"], now=_clock(date, slot, 5))
    assert out["status"] == "too_late"
    assert out["restaurant_phone"] == PHONE
    assert out["website"] == WEBSITE


def test_tool_cancel_not_found():
    assert booking_server.cancel_booking("TF4-9999")["status"] == "not_found"
