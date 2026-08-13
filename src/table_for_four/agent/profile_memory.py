"""Long-term member memory — a persistent Chroma-backed profile store.

This is the concierge's *cross-session* memory: who a guest is and what they
prefer, so a returning member is welcomed by name and never has to re-state a
standing dietary need, a favorite cuisine, or that their table always needs a
high chair.

Each member gets ONE profile document, keyed by a normalized handle
(`member_id`). The full profile is stored as JSON on the document's metadata;
a human-readable summary is the embedded document text, so profiles can also be
recalled *semantically* later (e.g. "the guest who loves Sicilian wine"). We
reuse the same local `all-MiniLM-L6-v2` embeddings as the perks store — no API
key, offline after a one-time model download.

Storage layout (Chroma collection `member_profiles`):
    id        = normalized member handle
    document  = summary text (embedded)
    metadata  = {profile_json, name, home_location, updated_at}

The pure functions take a `Collection` so tests can drive an ephemeral store;
the module-level wrappers use a lazily-opened persistent singleton.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

CHROMA_PATH = Path(__file__).parent / ".chroma_profiles"
COLLECTION_NAME = "member_profiles"

# Preference fields that accumulate rather than overwrite: an update UNIONs new
# values into the existing list (deduped, order-preserving) instead of replacing.
_LIST_FIELDS = ("dietary", "cuisines", "interests", "past_bookings")

# List fields that hold *current taste* rather than a permanent record: keep only
# the N most recently mentioned, newest last. Deliberately excludes `dietary`
# (an allergy is not a passing preference) and `past_bookings` (that's history).
_RECENCY_CAPPED = {"cuisines": 3}

# Standing preferences: once a value is on file it describes the guest, not the
# outing. A trip to Brooklyn for a friend's birthday must not silently rewrite
# where the guest lives, and a table for six once must not become their usual.
# `sticky_conflicts` flags such a write so the caller can ASK before it lands.
STICKY_SCALARS = ("home_location", "party_size")

_EMBED = embedding_functions.DefaultEmbeddingFunction()  # all-MiniLM-L6-v2, local

_collection: Collection | None = None  # lazy persistent singleton


# --- Identity ----------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def normalize_id(handle: str) -> str:
    """Fold a free-text name/handle into a stable id (lowercase, hyphenated)."""
    slug = re.sub(r"[^a-z0-9]+", "-", handle.strip().lower()).strip("-")
    return slug or "guest"


def looks_like_email(value: str) -> bool:
    return bool(_EMAIL_RE.match((value or "").strip()))


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def resolve_key(member_id: str) -> str:
    """The canonical storage key for an identity.

    Email is the unique identifier: an email-shaped id keys by the normalized
    email, so returning guests are recognized regardless of the display name they
    gave. Anything else (a provisional name/handle) keys by its slug.
    """
    value = (member_id or "").strip()
    return normalize_email(value) if looks_like_email(value) else normalize_id(value)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- Collection lifecycle ----------------------------------------------------

def build_collection(client: chromadb.ClientAPI) -> Collection:
    """Create or reuse the member-profiles collection on `client`."""
    return client.get_or_create_collection(
        COLLECTION_NAME,
        embedding_function=_EMBED,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection() -> Collection:
    """Lazily open the persistent profiles collection (process singleton)."""
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _collection = build_collection(client)
    return _collection


# --- Summary (embedded document) ---------------------------------------------

def profile_summary(profile: dict[str, Any]) -> str:
    """A short human-readable digest of a profile — the embedded document text."""
    parts: list[str] = []
    if profile.get("name"):
        parts.append(f"Member {profile['name']}")
    if profile.get("pronouns"):
        parts.append(f"pronouns {profile['pronouns']}")
    if profile.get("home_location"):
        parts.append(f"based in {profile['home_location']}")
    if profile.get("cuisines"):
        parts.append("likes " + ", ".join(profile["cuisines"]))
    if profile.get("dietary"):
        parts.append("dietary: " + ", ".join(profile["dietary"]))
    if profile.get("dining_atmosphere"):
        parts.append(f"prefers a {profile['dining_atmosphere']} atmosphere")
    kids = profile.get("kids") or []
    if kids:
        ages = ", ".join(str(k.get("age")) for k in kids if k.get("age") is not None)
        parts.append(f"dining with {len(kids)} child(ren)" + (f" (ages {ages})" if ages else ""))
    if profile.get("interests"):
        parts.append("interests: " + ", ".join(profile["interests"]))
    if profile.get("email"):
        parts.append(f"email {profile['email']}")
    return ". ".join(parts) or "Member profile"


# --- CRUD --------------------------------------------------------------------

def load_profile(collection: Collection, member_id: str) -> dict[str, Any] | None:
    """Return the stored profile for a member, or None if unknown."""
    res = collection.get(ids=[resolve_key(member_id)])
    if not res["ids"]:
        return None
    return json.loads(res["metadatas"][0]["profile_json"])


def save_profile(
    collection: Collection, member_id: str, profile: dict[str, Any]
) -> dict[str, Any]:
    """Upsert a full profile document (idempotent per member)."""
    pid = resolve_key(member_id)
    stored = {**profile, "member_id": pid, "updated_at": _now()}
    collection.upsert(
        ids=[pid],
        documents=[profile_summary(stored)],
        metadatas=[{
            "profile_json": json.dumps(stored, ensure_ascii=False),
            "name": str(stored.get("name") or ""),
            "email": str(stored.get("email") or ""),
            "home_location": str(stored.get("home_location") or ""),
            "updated_at": stored["updated_at"],
        }],
    )
    return stored


def _merge(current: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Merge updates into current: list fields UNION, scalars overwrite.

    None values in `updates` are ignored so a partial update never blanks a
    field the guest already gave us.

    Fields in `_RECENCY_CAPPED` are treated as *recent taste* rather than a
    permanent record: a repeat mention moves to the end of the list and only the
    most recent few are kept. Tastes drift, and a guest who has booked Italian
    three times running shouldn't still be pegged as a sushi person from one visit
    a year ago. Uncapped fields (dietary above all) keep everything — an allergy is
    not a preference and must never be aged out.
    """
    merged = dict(current)
    for key, value in updates.items():
        if value is None:
            continue
        if key in _LIST_FIELDS:
            incoming = value if isinstance(value, list) else [value]
            cap = _RECENCY_CAPPED.get(key)
            existing = list(merged.get(key) or [])
            if cap:
                # Re-mentioning something makes it recent again: drop the old
                # position, re-append, then keep the tail. Matching ignores case,
                # so "Italian" and "italian" are one favourite, not two.
                existing = [
                    item for item in existing
                    if not any(_same_value(item, new) for new in incoming)
                ]
                existing += [item for item in incoming if item is not None]
                merged[key] = existing[-cap:]
            else:
                for item in incoming:
                    if not any(_same_value(item, seen) for seen in existing):
                        existing.append(item)
                merged[key] = existing
        else:
            merged[key] = value
    return merged


def update_profile(
    collection: Collection, member_id: str, updates: dict[str, Any]
) -> dict[str, Any]:
    """Merge `updates` into a member's profile (creating it if new) and persist."""
    current = load_profile(collection, member_id) or {"member_id": resolve_key(member_id)}
    return save_profile(collection, member_id, _merge(current, updates))


def _same_value(a: Any, b: Any) -> bool:
    """Equality that ignores case/whitespace for text ("Midtown" == "midtown")."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def sticky_conflicts(
    profile: dict[str, Any] | None, updates: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Which parts of `updates` would rewrite a standing preference already on file.

    Returns `{field: {"saved": current, "proposed": incoming}}` for every field the
    caller should CONFIRM with the guest before writing:

    * `home_location` / `party_size` — a value is already stored and differs.
    * `cuisines` — the list is already at its recency cap, so adding a new one
      would push the guest's oldest favourite out.

    A first value is never a conflict: there's nothing to overwrite, so learning
    it is just learning. Everything else stays free to update as it always was —
    only these three drift in a way guests notice (see `STICKY_SCALARS`).
    """
    current = profile or {}
    out: dict[str, dict[str, Any]] = {}

    for field_name in STICKY_SCALARS:
        proposed = updates.get(field_name)
        saved = current.get(field_name)
        if proposed in (None, "", [], {}) or saved in (None, "", [], {}):
            continue
        if not _same_value(proposed, saved):
            out[field_name] = {"saved": saved, "proposed": proposed}

    incoming = updates.get("cuisines")
    if incoming:
        cap = _RECENCY_CAPPED["cuisines"]
        saved_list = list(current.get("cuisines") or [])
        fresh = [
            c for c in (incoming if isinstance(incoming, list) else [incoming])
            if not any(_same_value(c, existing) for existing in saved_list)
        ]
        if fresh and len(saved_list) >= cap:
            out["cuisines"] = {"saved": saved_list, "proposed": fresh}

    return out


def set_email(
    collection: Collection, current_member_id: str, email: str
) -> tuple[str, dict[str, Any], bool]:
    """Adopt `email` as the guest's canonical identity.

    Merges the current (name-keyed) profile into any existing profile stored under
    that email, saves the result under the email key, and removes the provisional
    name-keyed document. Returns `(email_key, merged_profile, was_returning)` where
    `was_returning` is True if a profile already existed for that email.
    """
    email_key = normalize_email(email)
    current = load_profile(collection, current_member_id) or {}
    existing = load_profile(collection, email_key)
    was_returning = existing is not None

    carry = {k: v for k, v in current.items() if k not in ("member_id", "updated_at")}
    merged = _merge(existing or {}, {**carry, "email": email_key})
    saved = save_profile(collection, email_key, merged)

    old_key = resolve_key(current_member_id)
    if old_key != email_key and current:
        try:
            collection.delete(ids=[old_key])
        except Exception:
            pass
    return email_key, saved, was_returning


# --- Semantic recall ---------------------------------------------------------

def search_profiles(
    collection: Collection, query: str, n_results: int = 5
) -> list[dict[str, Any]]:
    """Semantically recall members by free-text intent.

    Each profile is stored with an embedded summary (see `profile_summary`), so a
    query like "the guest who loves Sicilian wine" ranks members by meaning rather
    than exact key. Returns `[{member_id, name, similarity, profile}]`, best first.
    This is the profile store's second retrieval mode alongside key lookup — the
    same Chroma + local-embeddings stack as the perks RAG, applied to memory.
    """
    count = collection.count()
    if count == 0:
        return []
    res = collection.query(query_texts=[query], n_results=min(n_results, count))
    out: list[dict[str, Any]] = []
    for i, pid in enumerate(res["ids"][0]):
        meta = res["metadatas"][0][i]
        profile = json.loads(meta["profile_json"])
        out.append({
            "member_id": pid,
            "name": meta.get("name") or profile.get("name"),
            "similarity": round(1.0 - res["distances"][0][i], 3),  # cosine -> 0-1
            "profile": profile,
        })
    return out


# --- Persistent-singleton wrappers (used by the chat tools) ------------------

def load(member_id: str) -> dict[str, Any] | None:
    return load_profile(get_collection(), member_id)


def remember(member_id: str, updates: dict[str, Any]) -> dict[str, Any]:
    return update_profile(get_collection(), member_id, updates)


def adopt_email(current_member_id: str, email: str) -> tuple[str, dict[str, Any], bool]:
    return set_email(get_collection(), current_member_id, email)


def find_members(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    return search_profiles(get_collection(), query, n_results)


def set_booking_status(
    collection: Collection, member_id: str, confirmation_id: str, status: str
) -> dict[str, Any] | None:
    """Flip a stored past-booking's status (e.g. to 'cancelled') and persist.

    A targeted in-place edit + full save, deliberately *not* the union-merge path
    (`update_profile`), which would append a duplicate rather than update the row.
    """
    profile = load_profile(collection, member_id)
    if not profile:
        return None
    changed = False
    for booking in profile.get("past_bookings", []):
        if booking.get("confirmation_id") == confirmation_id:
            booking["status"] = status
            changed = True
    return save_profile(collection, member_id, profile) if changed else profile


def mark_booking(member_id: str, confirmation_id: str, status: str) -> dict[str, Any] | None:
    return set_booking_status(get_collection(), member_id, confirmation_id, status)
