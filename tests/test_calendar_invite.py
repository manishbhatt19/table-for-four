"""Tests for the downloadable calendar hold (.ics) generator."""

from table_for_four.agent.calendar_invite import build_ics, ics_filename

BOOKING = {
    "restaurant": "Osteria Midtown",
    "address": "127 W 44th St, New York, NY",
    "date": "2026-09-04",
    "time": "19:00",
    "party_size": 4,
    "confirmation_id": "TF4-0001",
    "perk_applied": "Weekend Family Feast",
    "restaurant_phone": "(212) 555-0142",
}


def test_ics_is_a_valid_vevent_with_correct_times():
    ics = build_ics(BOOKING)
    assert ics.startswith("BEGIN:VCALENDAR")
    assert "BEGIN:VEVENT" in ics and "END:VEVENT" in ics
    assert ics.strip().endswith("END:VCALENDAR")
    # 19:00 start, +90 min default -> 20:30 end (floating local time, no Z).
    assert "DTSTART:20260904T190000" in ics
    assert "DTEND:20260904T203000" in ics
    # iCalendar requires CRLF line endings.
    assert "\r\n" in ics and "\n\n" not in ics


def test_ics_carries_reservation_details():
    ics = build_ics(BOOKING)
    assert "SUMMARY:Reservation — Osteria Midtown" in ics
    assert "TF4-0001" in ics
    assert "Weekend Family Feast" in ics          # the perk the guest earned
    assert "Party of 4" in ics
    # Commas in the address are escaped per RFC 5545.
    assert "LOCATION:127 W 44th St\\, New York\\, NY" in ics


def test_ics_duration_is_configurable():
    ics = build_ics(BOOKING, duration_minutes=120)
    assert "DTEND:20260904T210000" in ics  # 19:00 + 2h


def test_ics_minimal_booking():
    ics = build_ics({"restaurant": "X", "date": "2026-09-04", "time": "12:00"})
    assert "DTSTART:20260904T120000" in ics
    assert "DTEND:20260904T133000" in ics
    assert "BEGIN:VEVENT" in ics


def test_ics_filename_uses_confirmation_id():
    assert ics_filename(BOOKING) == "reservation-TF4-0001.ics"
    assert ics_filename({}) == "reservation-reservation.ics"
