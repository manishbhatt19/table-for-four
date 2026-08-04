"""Build a downloadable calendar hold (.ics) from a reservation.

Produces a standards-compliant iCalendar VEVENT (RFC 5545) the guest can save to
any calendar app — Google, Apple, Outlook — with no account or connector needed.
The event uses *floating* local time (no timezone marker), which every calendar
interprets as the viewer's own local time — the right behavior for a table booked
at "7pm" wherever the guest is.

`build_ics(booking)` takes the same booking dict the concierge already produces
(restaurant, address, date, time, party_size, confirmation_id, perk_applied, …),
so the chat UI can offer a "Save to calendar" download straight from session state.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

DEFAULT_DURATION_MINUTES = 90
PRODID = "-//Table for Four//Concierge//EN"


def _escape(text: str) -> str:
    """Escape per RFC 5545: backslash, comma, semicolon, and newlines."""
    return (
        str(text)
        .replace("\\", "\\\\")
        .replace(",", "\\,")
        .replace(";", "\\;")
        .replace("\n", "\\n")
    )


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S")


def build_ics(booking: dict[str, Any], duration_minutes: int = DEFAULT_DURATION_MINUTES) -> str:
    """Return the .ics text for one reservation.

    Requires at least `restaurant`, `date` (YYYY-MM-DD), and `time` (HH:MM). Other
    fields (address, party_size, confirmation_id, perk_applied, restaurant_phone,
    website) enrich the event when present.
    """
    start = datetime.fromisoformat(f"{booking['date']}T{booking['time']}")
    end = start + timedelta(minutes=duration_minutes)

    restaurant = booking.get("restaurant") or "Restaurant"
    summary = f"Reservation — {restaurant}"

    desc: list[str] = []
    if booking.get("party_size"):
        desc.append(f"Party of {booking['party_size']}")
    if booking.get("confirmation_id"):
        desc.append(f"Confirmation: {booking['confirmation_id']}")
    if booking.get("perk_applied"):
        desc.append(f"Perk: {booking['perk_applied']}")
    if booking.get("restaurant_phone"):
        desc.append(f"Restaurant: {booking['restaurant_phone']}")
    if booking.get("website"):
        desc.append(booking["website"])
    desc.append("Booked with Ava, your Table for Four concierge.")
    description = "\n".join(desc)

    uid = f"{booking.get('confirmation_id') or 'tf4'}@tableforfour"

    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        f"PRODID:{PRODID}",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "BEGIN:VEVENT",
        f"UID:{uid}",
        f"DTSTAMP:{_stamp(datetime.now())}",
        f"DTSTART:{_stamp(start)}",
        f"DTEND:{_stamp(end)}",
        f"SUMMARY:{_escape(summary)}",
        f"DESCRIPTION:{_escape(description)}",
    ]
    if booking.get("address"):
        lines.append(f"LOCATION:{_escape(booking['address'])}")
    lines += ["STATUS:CONFIRMED", "END:VEVENT", "END:VCALENDAR"]

    # iCalendar requires CRLF line endings.
    return "\r\n".join(lines) + "\r\n"


def ics_filename(booking: dict[str, Any]) -> str:
    """A tidy download filename for a reservation's .ics."""
    conf = booking.get("confirmation_id") or "reservation"
    return f"reservation-{conf}.ics"
