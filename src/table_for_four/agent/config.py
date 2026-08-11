"""LLM configuration — provider-agnostic, offline-friendly.

`get_llm()` returns a LangChain chat model built from `.env`, or **None** when no
key is configured. None is the signal for the orchestrator to fall back to its
deterministic heuristic path, so the whole graph still runs offline with no key
and no cost.

Both OpenAI and OpenRouter speak the OpenAI-compatible API, so the only
difference is the base URL — selected by `LLM_PROVIDER`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _build_llm(temperature: float) -> Any | None:
    """Construct a chat model from `.env` at the given temperature, or None."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()

    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            return None
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    else:  # default: openai
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        base_url = None

    # Imported lazily so the project runs without langchain when staying offline.
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": model, "api_key": key, "temperature": temperature}
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=1)
def get_llm() -> Any | None:
    """Return a configured chat model (temperature 0), or None if no key is set.

    Temperature 0 keeps the orchestrator's parse/narrate deterministic.
    """
    return _build_llm(temperature=0.0)


@lru_cache(maxsize=1)
def get_chat_llm() -> Any | None:
    """Return a warmer chat model for the interpersonal concierge, or None.

    The conversational front-end wants some personality, so it runs a little
    warmer than the deterministic orchestrator path.
    """
    return _build_llm(temperature=0.6)
