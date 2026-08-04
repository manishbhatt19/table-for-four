"""Mock reservation backend (FastAPI) with a SQLite bookings ledger.

Endpoints:
    GET  /health                          -> liveness
    GET  /availability                    -> open time slots for a restaurant/date
    POST /bookings                        -> create a reservation (the write)
    GET  /bookings/{confirmation_id}      -> fetch a reservation
    GET  /bookings                        -> list reservations (filter by email/status)
    POST /bookings/{confirmation_id}/cancel -> cancel, subject to the 24h policy

Design notes:
* **Deterministic availability.** Which slots are taken is a pure function of
  (place_id, date) via a stable hash — no randomness — so tests and demos are
  reproducible. Larger parties see fewer slots, mimicking real constraints.
* **SQLite ledger.** Bookings persist to a single SQLite file (path via
  `BOOKING_DB_PATH`, default `mock_booking_api/bookings.db`) — a real relational
  system-of-record with a `status` column and cancellation history, but zero-setup
  and fully offline. `reset_store()` drops the table (used by tests).
* **Cancellation policy.** A reservation can be cancelled up to **24 hours** before
  its date/time. Inside that window the backend refuses and returns the
  restaurant's phone/website so the guest can call directly. The rule is enforced
  here (deterministically), never left to the model.
* Booking is the one **irreversible-by-the-agent** action; cancellation is the
  controlled reversal, recorded in the same ledger.

Run standalone (for a live demo):
    uv run uvicorn mock_booking_api.app:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Table for Four — Mock Reservation API", version="0.2.0")

# Standard dinner service slots offered by every (fictional) restaurant.
DINNER_SLOTS = [
    "17:00", "17:30", "18:00", "18:30", "19:00",
    "19:30", "20:00", "20:30", "21:00",
]
MAX_PARTY_SIZE = 20
CANCELLATION_WINDOW_HOURS = 24

DB_PATH = os.getenv("BOOKING_DB_PATH", "").strip() or str(
    Path(__file__).parent / "bookings.db"
)

# Fields returned for a booking, in a stable order (row -> dict).
_COLUMNS = (
    "confirmation_id", "place_id", "restaurant_name", "address", "restaurant_phone",
    "website", "date", "time", "party_size", "guest_name", "guest_email", "perk_id",
    "special_requests", "status", "created_at", "cancelled_at", "cancellation_reason",
)

_DDL = """
CREATE TABLE IF NOT EXISTS bookings (
    seq                 INTEGER PRIMARY KEY AUTOINCREMENT,
    confirmation_id     TEXT UNIQUE,
    place_id            TEXT NOT NULL,
    restaurant_name     TEXT NOT NULL,
    address             TEXT,
    restaurant_phone    TEXT,
    website             TEXT,
    date                TEXT NOT NULL,
    time                TEXT NOT NULL,
    party_size          INTEGER NOT NULL,
    guest_name          TEXT NOT NULL,
    guest_email         TEXT,
    perk_id             TEXT,
    special_requests    TEXT,
    status              TEXT NOT NULL DEFAULT 'confirmed',
    created_at          TEXT NOT NULL,
    cancelled_at        TEXT,
    cancellation_reason TEXT
);
"""

# --- SQLite store ------------------------------------------------------------

_conn: sqlite3.Connection | None = None
_lock = threading.Lock()  # serialize writes (TestClient/uvicorn may use threads)


def _connect() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        if DB_PATH != ":memory:":
            Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _conn.row_factory = sqlite3.Row
        _conn.execute(_DDL)
        _conn.commit()
    return _conn


def reset_store() -> None:
    """Drop and recreate the bookings table (test helper; resets the id counter)."""
    conn = _connect()
    with _lock:
        conn.execute("DROP TABLE IF EXISTS bookings")
        conn.execute(_DDL)
        conn.commit()


def _row_to_booking(row: sqlite3.Row) -> dict[str, Any]:
    return {col: row[col] for col in _COLUMNS}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


# --- Deterministic availability ----------------------------------------------

def _taken_slots(place_id: str, date: str) -> set[str]:
    """Deterministically decide which slots are already booked for (place, date)."""
    digest = hashlib.sha256(f"{place_id}|{date}".encode()).hexdigest()
    seed = int(digest, 16)
    return {slot for i, slot in enumerate(DINNER_SLOTS) if (seed >> i) & 1}


def available_slots(place_id: str, date: str, party_size: int) -> list[str]:
    """Open slots for a restaurant on a date, tightened for larger parties."""
    taken = _taken_slots(place_id, date)
    slots = [s for s in DINNER_SLOTS if s not in taken]
    if party_size >= 8:
        # Large parties are harder to seat: offer every other remaining slot.
        slots = slots[::2]
    return slots


# --- Schemas -----------------------------------------------------------------

class AvailabilityResponse(BaseModel):
    place_id: str
    date: str
    party_size: int
    available_slots: list[str]


class BookingRequest(BaseModel):
    place_id: str
    restaurant_name: str
    date: str  # YYYY-MM-DD
    time: str  # HH:MM
    party_size: int = Field(ge=1, le=MAX_PARTY_SIZE)
    guest_name: str
    address: str | None = None
    restaurant_phone: str | None = None
    website: str | None = None
    guest_email: str | None = None
    perk_id: str | None = None
    special_requests: str | None = None


class Booking(BaseModel):
    confirmation_id: str
    place_id: str
    restaurant_name: str
    address: str | None = None
    restaurant_phone: str | None = None
    website: str | None = None
    date: str
    time: str
    party_size: int
    guest_name: str
    guest_email: str | None = None
    perk_id: str | None = None
    special_requests: str | None = None
    status: str
    created_at: str
    cancelled_at: str | None = None
    cancellation_reason: str | None = None


class CancelRequest(BaseModel):
    # Injectable clock so the 24h policy is testable; omitted in real use.
    now: str | None = None
    reason: str | None = None


# --- Booking helpers ---------------------------------------------------------

def _fetch(conn: sqlite3.Connection, confirmation_id: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM bookings WHERE confirmation_id = ?", (confirmation_id,)
    ).fetchone()
    return _row_to_booking(row) if row else None


def _hours_until(date: str, time: str, now: datetime) -> float:
    booking_dt = datetime.fromisoformat(f"{date}T{time}")
    return (booking_dt - now) / timedelta(hours=1)


# --- Endpoints ---------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/availability", response_model=AvailabilityResponse)
def get_availability(place_id: str, date: str, party_size: int = 2) -> AvailabilityResponse:
    if not 1 <= party_size <= MAX_PARTY_SIZE:
        raise HTTPException(422, f"party_size must be 1-{MAX_PARTY_SIZE}")
    return AvailabilityResponse(
        place_id=place_id,
        date=date,
        party_size=party_size,
        available_slots=available_slots(place_id, date, party_size),
    )


@app.post("/bookings", response_model=Booking, status_code=201)
def create_booking(req: BookingRequest) -> Booking:
    open_slots = available_slots(req.place_id, req.date, req.party_size)
    if req.time not in open_slots:
        raise HTTPException(
            409,
            f"{req.time} is not available for a party of {req.party_size} on "
            f"{req.date}. Open slots: {open_slots}",
        )
    conn = _connect()
    with _lock:
        cur = conn.execute(
            """INSERT INTO bookings
               (place_id, restaurant_name, address, restaurant_phone, website,
                date, time, party_size, guest_name, guest_email, perk_id,
                special_requests, status, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?, 'confirmed', ?)""",
            (req.place_id, req.restaurant_name, req.address, req.restaurant_phone,
             req.website, req.date, req.time, req.party_size, req.guest_name,
             req.guest_email, req.perk_id, req.special_requests, _now_iso()),
        )
        confirmation_id = f"TF4-{cur.lastrowid:04d}"
        conn.execute(
            "UPDATE bookings SET confirmation_id = ? WHERE seq = ?",
            (confirmation_id, cur.lastrowid),
        )
        conn.commit()
        booking = _fetch(conn, confirmation_id)
    return Booking(**booking)


@app.get("/bookings/{confirmation_id}", response_model=Booking)
def get_booking(confirmation_id: str) -> Booking:
    booking = _fetch(_connect(), confirmation_id)
    if booking is None:
        raise HTTPException(404, f"No booking {confirmation_id}")
    return Booking(**booking)


@app.get("/bookings", response_model=list[Booking])
def list_bookings(email: str | None = None, status: str | None = None) -> list[Booking]:
    query = "SELECT * FROM bookings"
    conds, params = [], []
    if email:
        conds.append("guest_email = ?")
        params.append(email)
    if status:
        conds.append("status = ?")
        params.append(status)
    if conds:
        query += " WHERE " + " AND ".join(conds)
    query += " ORDER BY seq"
    rows = _connect().execute(query, params).fetchall()
    return [Booking(**_row_to_booking(r)) for r in rows]


@app.post("/bookings/{confirmation_id}/cancel", response_model=Booking)
def cancel_booking(confirmation_id: str, req: CancelRequest | None = None) -> Booking:
    req = req or CancelRequest()
    conn = _connect()
    booking = _fetch(conn, confirmation_id)
    if booking is None:
        raise HTTPException(404, f"No booking {confirmation_id}")
    if booking["status"] == "cancelled":
        raise HTTPException(409, {
            "error": "already_cancelled",
            "confirmation_id": confirmation_id,
            "cancelled_at": booking["cancelled_at"],
        })

    now = datetime.fromisoformat(req.now) if req.now else datetime.now()
    hours = _hours_until(booking["date"], booking["time"], now)
    if hours < CANCELLATION_WINDOW_HOURS:
        # Inside the 24h window (or already past): the agent must NOT self-cancel;
        # hand the guest the restaurant's own contact details to call directly.
        raise HTTPException(409, {
            "error": "within_24h",
            "message": (
                f"Cancellations are only possible more than {CANCELLATION_WINDOW_HOURS} "
                "hours ahead. Please contact the restaurant directly to cancel."
            ),
            "hours_until_booking": round(hours, 1),
            "restaurant_name": booking["restaurant_name"],
            "restaurant_phone": booking["restaurant_phone"],
            "website": booking["website"],
        })

    with _lock:
        conn.execute(
            "UPDATE bookings SET status='cancelled', cancelled_at=?, "
            "cancellation_reason=? WHERE confirmation_id=?",
            (_now_iso(), req.reason, confirmation_id),
        )
        conn.commit()
        booking = _fetch(conn, confirmation_id)
    return Booking(**booking)
