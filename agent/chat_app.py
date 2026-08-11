"""Dino — a Streamlit chat UI for the interpersonal Table for Four concierge.

A thin web wrapper around the existing, UI-agnostic session API in
`agent.concierge_chat`:

    start_session(member_id) -> ConciergeSession   # loads any saved profile
    _run_turn(session, llm)  -> reply text          # one model turn + tool calls

The conversation, memory, tool-calling, and guardrails all live in
`concierge_chat`; this file only renders them. The left panel shows the guest's
**long-term profile** filling in live from Chroma as they talk — the memory
story made visible.

Run it:
    uv run streamlit run agent/chat_app.py
"""

from __future__ import annotations

import html
import sys
from pathlib import Path
from typing import Any

# Streamlit runs this file directly, so it puts `agent/` (the script's own dir) on
# sys.path rather than the project root — which makes `import agent...` fail. Put
# the project root (this file's parent's parent) first so the package resolves.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st

from agent.calendar_invite import build_ics, ics_filename
from agent.concierge_chat import (
    TOOL_SCHEMAS,
    ConciergeSession,
    _run_turn,
    start_session,
)
from agent.config import get_chat_llm

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
            label = "placeholder images (offline mode)" if item.get("source") == "fixture" \
                else "from the public web, not the restaurant"
            st.caption(f"📸 {item.get('restaurant', 'Restaurant')} · {label}")


def _shorten(text: str, limit: int = 90) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


# --- Session bootstrap -------------------------------------------------------

def _start(name: str) -> None:
    """Create a concierge session, get Dino's greeting, and store it in state."""
    session = start_session(name)
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
            "`uv run python -m agent`."
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

    # Guest turn.
    if prompt := st.chat_input("Message Dino…"):
        st.session_state.display.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        session: ConciergeSession = st.session_state.session
        session.messages.append(HumanMessage(content=prompt))
        # Anything the web-highlights tool collects during this turn is new media.
        seen_media = len(session.media)
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
