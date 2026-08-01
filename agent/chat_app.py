"""Ava — a Streamlit chat UI for the interpersonal Table for Four concierge.

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

from typing import Any

import streamlit as st

from agent.concierge_chat import (
    TOOL_SCHEMAS,
    ConciergeSession,
    _run_turn,
    start_session,
)
from agent.config import get_chat_llm

from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Table for Four · Ava", page_icon="🍽️", layout="wide")


# --- Profile panel -----------------------------------------------------------

def _render_profile(profile: dict[str, Any] | None) -> None:
    """Render the guest's long-term profile — the live memory story."""
    st.subheader("🧠 What Ava remembers")

    if not profile:
        st.caption("Nothing yet — Ava learns as you chat, and remembers you next time.")
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


# --- Session bootstrap -------------------------------------------------------

def _start(name: str) -> None:
    """Create a concierge session, get Ava's greeting, and store it in state."""
    session = start_session(name)
    session.messages.append(
        HumanMessage(content=f"(System: the guest '{name}' just connected. Greet them.)")
    )
    greeting = _run_turn(session, st.session_state.llm)
    st.session_state.session = session
    st.session_state.display = [{"role": "assistant", "content": greeting}]


# --- App ---------------------------------------------------------------------

def main() -> None:
    st.title("🍽️ Table for Four")
    st.caption("Ava, your interpersonal reservation concierge — with long-term memory.")

    # Bind the (warm) chat model once. None means no API key configured.
    if "llm" not in st.session_state:
        llm = get_chat_llm()
        st.session_state.llm = llm.bind_tools(TOOL_SCHEMAS) if llm is not None else None

    if st.session_state.llm is None:
        st.error(
            "Ava needs an OpenAI (or OpenRouter) key to talk. Add `OPENAI_API_KEY` "
            "to your `.env`, then restart this app.\n\n"
            "The offline booking demo still runs without a key: "
            "`uv run python -m agent`."
        )
        return

    # Sidebar: identity gate + live profile panel.
    with st.sidebar:
        if "session" not in st.session_state:
            st.subheader("Welcome")
            st.write("Tell Ava who you are to begin. Use your email to be recognised "
                     "on a return visit.")
            with st.form("identity"):
                name = st.text_input("Your name, handle, or email", value="")
                start = st.form_submit_button("Start chatting", use_container_width=True)
            if start and name.strip():
                with st.spinner("Ava is getting ready…"):
                    _start(name.strip())
                st.rerun()
        else:
            _render_profile(st.session_state.session.profile)
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
        avatar = "🧑" if msg["role"] == "user" else "🌸"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    # Guest turn.
    if prompt := st.chat_input("Message Ava…"):
        st.session_state.display.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        session: ConciergeSession = st.session_state.session
        session.messages.append(HumanMessage(content=prompt))
        with st.chat_message("assistant", avatar="🌸"):
            with st.spinner("Ava is thinking…"):
                reply = _run_turn(session, st.session_state.llm)
            st.markdown(reply)
        st.session_state.display.append({"role": "assistant", "content": reply})
        # Profile may have changed this turn — refresh the sidebar panel.
        st.rerun()


main()
