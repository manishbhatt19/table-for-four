"""Dino — a Streamlit chat UI for the interpersonal Table for Four concierge.

A thin web wrapper around the existing, UI-agnostic session API in
`table_for_four.agent.concierge_chat`:

    start_session(member_id) -> ConciergeSession   # loads any saved profile
    _run_turn(session, llm)  -> reply text          # one model turn + tool calls

The conversation, memory, tool-calling, and guardrails all live in
`concierge_chat`; this file only renders them. The left panel shows the guest's
**long-term profile** filling in live from Chroma as they talk — the memory
story made visible.

Run it:
    uv run streamlit run src/table_for_four/ui/chat_app.py
"""

from __future__ import annotations

import html
from typing import Any

import streamlit as st

from table_for_four.agent.calendar_invite import build_ics, ics_filename
from table_for_four.agent.concierge_chat import (
    TOOL_SCHEMAS,
    ConciergeSession,
    _requested_times,
    _run_turn,
    press_change_my_mind,
    press_reserve,
    start_session,
    to_12h,
)
from table_for_four.agent.config import get_chat_llm

from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Table for Four · Dino", page_icon="🍽️", layout="wide")


# --- Profile panel -----------------------------------------------------------

def _render_profile(profile: dict[str, Any] | None) -> None:
    """Render the guest's long-term profile — the live memory story."""
    st.subheader("🧠 What Dino remembers")

    if not profile:
        st.caption("Nothing yet — Dino learns as you chat, and remembers you next time.")
        return

    def row(label: str, value: Any) -> None:
        if value in (None, "", [], {}):
            return
        if isinstance(value, list):
            value = ", ".join(str(v) for v in value)
        st.markdown(f"**{label}**  \n{value}")

    row("Name", profile.get("name"))
    row("Pronouns", profile.get("pronouns"))
    row("Email", profile.get("email"))
    row("Home area", profile.get("home_location"))
    row("Usual party size", profile.get("party_size"))
    row("Favourite cuisines", profile.get("cuisines"))
    row("Dietary needs", profile.get("dietary"))
    row("Atmosphere", profile.get("dining_atmosphere"))
    row("Interests", profile.get("interests"))

    kids = profile.get("kids") or []
    if kids:
        summary = []
        for k in kids:
            age = k.get("age")
            chair = " (high chair)" if k.get("needs_high_chair") else ""
            summary.append(f"age {age}{chair}" if age is not None else f"child{chair}")
        row("Dining with children", summary)

    bookings = profile.get("past_bookings") or []
    if bookings:
        st.markdown("**Past bookings**")
        for b in bookings[-5:]:
            line = f"· {b.get('restaurant', 'a restaurant')}"
            if b.get("date"):
                line += f" — {b['date']}"
            if b.get("time"):
                line += f" {b['time']}"
            st.caption(line)

    if profile.get("updated_at"):
        st.divider()
        st.caption(f"Profile saved to Chroma · updated {profile['updated_at']}")


# --- Reservations & perks panel ----------------------------------------------

def _render_reservations(session: ConciergeSession) -> None:
    """Show this session's bookings — celebrate any perk, offer a calendar hold."""
    bookings = [b for b in session.bookings.values() if b.get("confirmation_id")]
    if not bookings:
        return

    st.subheader("🎟️ Reservations & perks")
    cancelled = {
        b.get("confirmation_id")
        for b in ((session.profile or {}).get("past_bookings") or [])
        if b.get("status") == "cancelled"
    }

    # A little celebration the first time a perked booking appears.
    celebrated = st.session_state.setdefault("celebrated_perks", set())
    fresh_perks = {
        b["confirmation_id"] for b in bookings
        if b.get("perk_applied") and b["confirmation_id"] not in cancelled
    }
    if fresh_perks - celebrated:
        st.balloons()
        celebrated |= fresh_perks

    for b in bookings:
        conf = b["confirmation_id"]
        with st.container(border=True):
            st.markdown(
                f"**{b.get('restaurant', 'Restaurant')}**  \n"
                f"{b.get('date', '')} · {b.get('time', '')} · party of {b.get('party_size', '?')}"
            )
            if b.get("perk_applied"):
                label = "Sample partner offer" if b.get("perk_sample") else "Perk unlocked"
                st.success(f"🎟️ **{label}:** {b['perk_applied']}")
            if conf in cancelled:
                st.caption(f"❌ Cancelled · {conf}")
            else:
                st.caption(f"Confirmation {conf}")
                st.download_button(
                    "📅 Save to calendar (.ics)",
                    data=build_ics(b),
                    file_name=ics_filename(b),
                    mime="text/calendar",
                    key=f"ics-{conf}",
                    use_container_width=True,
                )


# --- Web highlights (photos) -------------------------------------------------

def _render_media(items: list[dict[str, Any]]) -> None:
    """Render photos fetched from the web for a restaurant under Dino's reply.

    Rendered as plain <img> rather than `st.image` on purpose: a hotlinked web
    photo can 404 or be blocked at any time, and a broken <img> degrades to its
    alt text instead of throwing. Every photo is captioned with the domain it
    came from — these are not the restaurant's official pictures.
    """
    for item in items:
        card = item.get("card")
        images = item.get("images") or []
        if not card and not images:
            continue

        # The generated card leads: it's the one panel we control the design of.
        # The "what people order" line sits beside it, set in italic serif — it
        # reads as an aside rather than disappearing into the chat prose.
        if card:
            note = card.get("note", "")
            aside = (
                f'<p style="flex:1 1 210px;min-width:190px;margin:6px 0 0;'
                f'font-family:Georgia,\'Times New Roman\',serif;font-style:italic;'
                f'font-size:1.02rem;line-height:1.55;opacity:0.88;'
                f'border-left:3px solid rgba(128,128,128,0.35);padding-left:13px">'
                f"{html.escape(note)}</p>"
            ) if note else ""
            st.markdown(
                f'<div style="display:flex;gap:16px;flex-wrap:wrap;'
                f'align-items:flex-start;margin:4px 0 10px">'
                f'<img src="{html.escape(card["url"], quote=True)}" '
                f'alt="{html.escape(card.get("description", "Menu card"))}" '
                f'style="width:100%;max-width:430px;border-radius:12px;display:block">'
                f"{aside}</div>",
                unsafe_allow_html=True,
            )

        cards = "".join(
            f'<figure style="margin:0;flex:1 1 180px;max-width:240px">'
            f'<img src="{html.escape(img["url"], quote=True)}" '
            f'alt="{html.escape(img.get("description", "Photo"))}" '
            f'loading="lazy" '
            f'style="width:100%;height:150px;object-fit:cover;border-radius:10px;'
            f'border:1px solid rgba(128,128,128,0.25)">'
            f'<figcaption style="font-size:0.72rem;opacity:0.7;margin-top:4px;'
            f'line-height:1.3">{html.escape(_shorten(img.get("description", "")))}'
            f'<br><span style="opacity:0.75">{html.escape(img.get("source", "web"))}</span>'
            f"</figcaption></figure>"
            for img in images
        )
        if images:
            st.markdown(
                f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:2px 0 8px">{cards}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"📸 {item.get('restaurant', 'Restaurant')} · {_provenance(item, images)}")


def _reserve_gate(session: ConciergeSession) -> tuple[str, bool] | None:
    """Show what is about to be booked, and wait to be told.

    The last thing before an irreversible write, and the only place in the chat
    surface where the guest's consent is a thing they *did* rather than a thing
    the model read into a sentence. Everything shown here comes from the same
    dict the booking is built from, so the summary cannot drift from the booking.
    """
    pending = session.pending_reservation
    if not pending:
        return None
    # Returns (what the guest said, was it already put in the transcript).

    rows = [
        ("Restaurant", pending.get("restaurant")),
        ("Where", pending.get("address")),
        ("Date", pending.get("date")),
        ("Time", pending.get("time_display") or pending.get("time")),
        ("Party", f"{pending.get('party_size')} people"),
        ("Under the name", pending.get("guest_name")),
        ("Confirmation to", pending.get("email")),
        ("Notes", pending.get("special_requests")),
    ]
    perk = pending.get("perk")
    if perk:
        rows.append(("Offer applied", perk + (" (sample)" if pending.get("perk_sample") else "")))

    body = "".join(
        f'<tr><td style="padding:2px 12px 2px 0;opacity:0.65;white-space:nowrap">{html.escape(label)}</td>'
        f'<td style="padding:2px 0"><b>{html.escape(str(value))}</b></td></tr>'
        for label, value in rows if value
    )
    st.markdown(
        '<div style="border:1px solid rgba(128,128,128,0.35);border-radius:10px;'
        'padding:12px 14px;margin:6px 0 10px">'
        '<div style="font-size:0.95rem;margin-bottom:8px">🍽️ <b>Please confirm your reservation</b></div>'
        '<div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start">'
        f'{_gate_art(pending)}'
        f'<table style="font-size:0.88rem;border-collapse:collapse">{body}</table></div>'
        '<div style="font-size:0.78rem;opacity:0.65;margin-top:10px">'
        'Nothing is booked until you press Reserve.</div></div>',
        unsafe_allow_html=True,
    )

    left, right, _ = st.columns([1, 1, 2])
    if left.button("✅ Reserve", key="gate-reserve", type="primary", use_container_width=True):
        # Reserve books straight away and puts the guest's words in the transcript
        # itself, so this one is already spoken by the time it comes back.
        return press_reserve(session), True
    if right.button("↩️ Change my mind", key="gate-cancel", use_container_width=True):
        return press_change_my_mind(session), False
    return None


def _gate_art(pending: dict[str, Any]) -> str:
    """The one picture beside the details, sized for what it actually is.

    A Places photo is a photograph and crops happily to a fixed frame; the
    generated menu card is a wide 640px panel whose type becomes unreadable if
    it's cropped to the same shape. Same slot, two shapes — and the credit line
    stays under the photograph, because these are somebody's pictures.
    """
    photo = pending.get("photo") or {}
    url = photo.get("url")
    if not url:
        return ""
    alt = photo.get("description") or pending.get("restaurant") or "Restaurant"
    if photo.get("source") == "menu card":
        return (
            f'<img src="{html.escape(url, quote=True)}" alt="{html.escape(alt)}" '
            'style="flex:0 0 auto;width:100%;max-width:300px;height:auto;'
            'border-radius:10px;display:block">'
        )
    return (
        '<figure style="margin:0;flex:0 0 auto">'
        f'<img src="{html.escape(url, quote=True)}" alt="{html.escape(alt)}" '
        'style="width:200px;height:145px;object-fit:cover;border-radius:10px;'
        'border:1px solid rgba(128,128,128,0.25);display:block">'
        '<figcaption style="font-size:0.7rem;opacity:0.65;margin-top:3px;'
        f'max-width:200px;line-height:1.3">{html.escape(_shorten(alt))}</figcaption>'
        "</figure>"
    )


def _provenance(item: dict[str, Any], images: list[dict[str, Any]]) -> str:
    """Say where these photos actually came from.

    This caption used to read "from the public web, not the restaurant" for
    everything, which was honest when everything did come from a search index.
    Now the best of them come from Google against the place id, or from the
    restaurant's own domain, and telling a guest otherwise would be its own
    small untruth — in the direction of underselling, but untrue either way.
    """
    if item.get("source") == "fixture":
        return "placeholder images (offline mode)"
    sources = {(img.get("source") or "").lower() for img in images}
    if "google places" in sources:
        return "from Google Places and the web"
    site = _domain(item.get("website") or "")
    if site and site in sources:
        return f"from {site} and the web"
    return "from the public web, not the restaurant"


def _domain(url: str) -> str:
    host = url.split("//")[-1].split("/")[0].lower()
    return host[4:] if host.startswith("www.") else host


def _render_trail(session: ConciergeSession) -> None:
    """The governance trail, where the guest's own journey can be checked against it.

    Worth showing rather than only writing to a log: the claim this project makes
    is that every action has a named actor and every reply was checked, and a
    panel that fills in as the conversation runs is that claim being demonstrated
    instead of asserted.
    """
    records = session.trail.records
    if not records:
        return
    stripped = [r for r in records if r.event == "grounding"]
    label = f"🧾 Governance trail ({len(records)})"
    if stripped:
        label += " · ⚠️"
    with st.expander(label, expanded=False):
        if stripped:
            st.caption(
                f"{len(stripped)} reply(s) had a claim no tool result supported. "
                "The sentence was removed before it reached the chat."
            )
        for record in reversed(records[-25:]):
            when = record.at[11:19]  # HH:MM:SS is enough at this scale
            if record.event == "tool_call":
                st.markdown(
                    f"`{when}` **{record.actor}** · {record.detail.get('tool')} "
                    f"→ _{record.detail.get('outcome')}_"
                )
            elif record.event == "grounding":
                kinds = ", ".join(
                    f["kind"] for f in record.detail.get("findings", [])
                ) or "—"
                st.markdown(f"`{when}` ⚠️ **ungrounded** ({kinds}) — sentence removed")
                for gone in record.detail.get("removed", []):
                    st.caption(f"↳ {_shorten(gone, 120)}")
            else:
                st.markdown(f"`{when}` {record.event}")


def _shorten(text: str, limit: int = 90) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- Open times, as something you can tap ------------------------------------

# Wide enough to read a clock time, narrow enough that a full service fits on a
# few rows without the labels truncating.
CHIPS_PER_ROW = 5


def _time_chips(session: ConciergeSession) -> str | None:
    """Offer the open times as buttons; return the one the guest tapped, if any.

    Deliberately not a shortcut past the conversation. A tap is fed back through
    exactly the same path as typed text, because the guest's own words are what
    several guardrails read: `_requested_times` scans the transcript so the pass
    can refuse a booking at a time they didn't ask for, and a chip that quietly
    set state instead of speaking would walk straight around it.

    Only what a tool actually returned is ever shown here — these are the slots
    from the last `check_availability_times`, not a guess at what a restaurant
    might be able to do.
    """
    pend = session.availability or {}
    slots = pend.get("slots") or []
    if not slots:
        return None
    # Once this outing is on the ledger the times are history, not an offer.
    if any(k.startswith(f"{pend.get('place_id')}|{pend.get('date')}|")
           for k in session.bookings):
        return None
    # These stay up even once the guest has named a time. Hiding them then was an
    # attempt to stop the picker outlasting its question, but it meant a guest who
    # said "6pm" got the alternatives read out as prose instead — a list to scroll
    # where a moment earlier there were buttons to tap. The reserve gate already
    # takes the picker down at the point it would start nagging, so the times can
    # simply stay tappable for as long as they're on offer.
    where = (session.pending.get("restaurant")
             if session.pending.get("place_id") == pend.get("place_id") else None)
    chosen = _requested_times(session) & set(slots)
    st.caption(
        "🕑 Open times"
        + (f" at {where}" if where else "")
        + (f" on {pend['date']}" if pend.get("date") else "")
        + (" — yours is set; tap another only if you'd like to change it."
           if chosen else " — tap one, or just tell Dino.")
    )

    picked = None
    for start in range(0, len(slots), CHIPS_PER_ROW):
        # A full-width row every time, so a short last row stays left-aligned
        # under the one above it rather than stretching to fill.
        cols = st.columns(CHIPS_PER_ROW)
        for col, slot in zip(cols, slots[start:start + CHIPS_PER_ROW]):
            # Labelled the way a guest reads a time, keyed on the way the ledger
            # stores one — and it sends the label, which `clock_times` resolves
            # straight back to the slot, so the booking guard still sees the choice.
            key = f"slot-{pend.get('place_id')}-{pend.get('date')}-{slot}"
            if col.button(to_12h(slot), key=key, use_container_width=True):
                picked = to_12h(slot)
    return picked


# --- Session bootstrap -------------------------------------------------------

def _start(name: str) -> None:
    """Create a concierge session, get Dino's greeting, and store it in state."""
    session = start_session(name, confirm_in_ui=True)  # this surface has a Reserve button
    session.messages.append(
        HumanMessage(content=f"(System: the guest '{name}' just connected. Greet them.)")
    )
    greeting = _run_turn(session, st.session_state.llm)
    st.session_state.session = session
    st.session_state.display = [
        {"role": "assistant", "content": greeting, "media": list(session.media)}
    ]


# --- App ---------------------------------------------------------------------

def main() -> None:
    st.title("🍽️ Table for Four")
    st.caption("Dino, your interpersonal reservation concierge — with long-term memory.")

    # Bind the (warm) chat model once. None means no API key configured.
    if "llm" not in st.session_state:
        llm = get_chat_llm()
        st.session_state.llm = llm.bind_tools(TOOL_SCHEMAS) if llm is not None else None

    if st.session_state.llm is None:
        st.error(
            "Dino needs an OpenAI (or OpenRouter) key to talk. Add `OPENAI_API_KEY` "
            "to your `.env`, then restart this app.\n\n"
            "The offline booking demo still runs without a key: "
            "`uv run python -m table_for_four`."
        )
        return

    # Sidebar: identity gate + live profile panel.
    with st.sidebar:
        if "session" not in st.session_state:
            st.subheader("Welcome")
            st.write("Tell Dino who you are to begin. Use your email to be recognised "
                     "on a return visit.")
            with st.form("identity"):
                name = st.text_input("Your name, handle, or email", value="")
                start = st.form_submit_button("Start chatting", use_container_width=True)
            if start and name.strip():
                with st.spinner("Dino is getting ready…"):
                    _start(name.strip())
                st.rerun()
        else:
            _render_profile(st.session_state.session.profile)
            _render_reservations(st.session_state.session)
            _render_trail(st.session_state.session)
            st.divider()
            if st.button("End session / new guest", use_container_width=True):
                for k in ("session", "display"):
                    st.session_state.pop(k, None)
                st.rerun()

    if "session" not in st.session_state:
        st.info("👈 Introduce yourself in the sidebar to start your booking journey.")
        return

    # Render the conversation so far.
    for msg in st.session_state.display:
        avatar = "🧑" if msg["role"] == "user" else "🦖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg.get("media"):
                _render_media(msg["media"])

    # Guest turn — typed, tapped from the open times, or answered at the gate.
    # The gate comes first: while a reservation is waiting on a decision, that is
    # the only thing being asked, and offering times underneath it would confuse
    # what the buttons apply to.
    session_now: ConciergeSession = st.session_state.session
    # Watermark the media before the gate, not after. Pressing Reserve books
    # immediately and gathers the restaurant's photos on the way out, so a mark
    # taken further down counts them as already-seen and silently drops them —
    # which is exactly how a confirmed booking arrived with no pictures.
    seen_media = len(session_now.media)
    answered, spoken = _reserve_gate(session_now) or (None, False)
    tapped = None if session_now.pending_reservation else _time_chips(session_now)
    if prompt := (answered or tapped or st.chat_input("Message Dino…")):
        st.session_state.display.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        session: ConciergeSession = st.session_state.session
        if not spoken:
            session.messages.append(HumanMessage(content=prompt))
        with st.chat_message("assistant", avatar="🦖"):
            with st.spinner("Dino is thinking…"):
                reply = _run_turn(session, st.session_state.llm)
            st.markdown(reply)
            fresh = session.media[seen_media:]
            if fresh:
                _render_media(fresh)
        st.session_state.display.append(
            {"role": "assistant", "content": reply, "media": fresh}
        )
        # Profile may have changed this turn — refresh the sidebar panel.
        st.rerun()


main()
