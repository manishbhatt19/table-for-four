"""Mock reservation backend (FastAPI).

Endpoints:
    GET  /health                          -> liveness
    GET  /availability                    -> open time slots for a restaurant/date
    POST /bookings                        -> create a reservation (the write)
    GET  /bookings/{confirmation_id}      -> fetch a reservation
    GET  /bookings                         -> list reservations (demo/debug)

Design notes:
* **Deterministic availability.** Which slots are taken is a pure function of
  (place_id, date) via a stable hash — no randomness — so tests and demos are
  reproducible. Larger parties see fewer slots, mimicking real constraints.
* **In-memory store.** Bookings live in a module-level dict; `reset_store()` clears
  it (used by tests). This is a *mock* — no persistence, no real reservation.
* Booking is the one **irreversible action** in the system; in the full agent it is
  wrapped by a human-approval gate (M4). This service just records the write.

Run standalone (for a live demo):
    uv run uvicorn mock_booking_api.app:app --reload --port 8000
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(title="Table for Four — Mock Reservation API", version="0.1.0")

# Standard dinner service slots offered by every (fictional) restaurant.
DINNER_SLOTS = [
    "17:00", "17:30", "18:00", "18:30", "19:00",
    "19:30", "20:00", "20:30", "21:00",
]
MAX_PARTY_SIZE = 20

# --- In-memory store ---------------------------------------------------------

_BOOKINGS: dict[str, dict[str, Any]] = {}
_SEQ = {"n": 0}


def reset_store() -> None:
    """Clear all bookings (test helper)."""
    _BOOKINGS.clear()
    _SEQ["n"] = 0


def _next_confirmation_id() -> str:
    _SEQ["n"] += 1
    return f"TF4-{_SEQ['n']:04d}"


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
    perk_id: str | None = None
    special_requests: str | None = None


class Booking(BookingRequest):
    confirmation_id: str
    status: str


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
    booking = Booking(
        confirmation_id=_next_confirmation_id(),
        status="confirmed",
        **req.model_dump(),
    )
    _BOOKINGS[booking.confirmation_id] = booking.model_dump()
    return booking


@app.get("/bookings/{confirmation_id}", response_model=Booking)
def get_booking(confirmation_id: str) -> Booking:
    booking = _BOOKINGS.get(confirmation_id)
    if booking is None:
        raise HTTPException(404, f"No booking {confirmation_id}")
    return Booking(**booking)


@app.get("/bookings", response_model=list[Booking])
def list_bookings() -> list[Booking]:
    return [Booking(**b) for b in _BOOKINGS.values()]
