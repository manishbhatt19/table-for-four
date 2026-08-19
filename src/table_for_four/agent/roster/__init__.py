"""The roster — who works here, and what each of them is allowed to touch.

Table for Four has always had five units of work; until now they were anonymous,
their boundaries living only in the author's head. This package declares them.
Each `.md` file is one unit: frontmatter says what it may call, the body is its
brief — and for the four back-of-house units, the body also carries the tool
descriptions the model is handed, so these files are genuinely model-facing rather
than documentation *about* the model.

**No unit here gets a model.** Dino is still the only thing in the system that
talks to an LLM (`_run_turn`, two call sites). The roster costs one file read per
unit at import and one set lookup per tool call — see `docs/week4_agentic_harness.md`
for why a sub-agent with its own model would have been a pure cost increase.

Two things this gives us that comments could not:

* **A grant is enforced, not observed.** `require()` is called inside the tool
  registry and the profile store, so `create_booking` while acting as `curator`
  raises rather than merely never happening to get called.
* **An audit line can name an actor.** `acting_unit()` answers "who did this?"
  at the moment of the effect, which is the join into the M4 governance trail.

What this does *not* constrain is the model's imagination — these grants govern
code paths. What constrains the model is the list of tool schemas it is handed,
which is a separate mechanism (`TOOL_SCHEMAS` in `concierge_chat`).
"""

from __future__ import annotations

import functools
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

_ROSTER_DIR = Path(__file__).parent

# Every effect a unit can have on the world: the in-process tool registry
# (`agent.tools`) plus the three profile-store writes. Reads are deliberately not
# in here — the harness governs what a unit can *change*, not what it can look up,
# and a lookup the guest already consented to is not a privilege worth spending a
# grant on.
CAPABILITIES = frozenset({
    "search_restaurants",
    "find_perks",
    "lookup_dining_highlights",
    "place_photos",
    "check_availability",
    "create_booking",
    "get_booking",
    "cancel_booking",
    "remember",
    "adopt_email",
    "mark_booking",
})

_TOOL_HEADING = "## Tool: "


class NotGranted(RuntimeError):
    """A unit reached for a capability its roster entry does not grant."""


@dataclass(frozen=True)
class Unit:
    """One declared unit of work — a member of staff, in the storyline."""

    name: str
    role: str
    tools: frozenset[str]     # capabilities granted
    never: frozenset[str]     # capabilities explicitly withheld (documentation with teeth)
    handlers: tuple[str, ...]  # the model-facing tool names this unit answers
    brief: str                 # prose body, up to the first tool section
    tool_text: dict[str, str]  # handler name -> the description the model is sent

    def grants(self, capability: str) -> bool:
        return capability in self.tools


# --- Loading -----------------------------------------------------------------

def _parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """Split a `---`-delimited header off the body.

    A deliberately tiny parser rather than a YAML dependency: the header holds
    strings and flat lists and nothing else, and adding a package to read five
    files of our own would be a poor trade.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("roster file must open with a '---' frontmatter block")
    try:
        end = lines.index("---", 1)
    except ValueError:
        raise ValueError("unterminated frontmatter block") from None

    meta: dict[str, Any] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, _, raw = line.partition(":")
        value = raw.strip()
        if value.startswith("[") and value.endswith("]"):
            meta[key.strip()] = [v.strip() for v in value[1:-1].split(",") if v.strip()]
        else:
            meta[key.strip()] = value
    return meta, "\n".join(lines[end + 1:]).lstrip("\n")


def _split_tool_sections(body: str) -> tuple[str, dict[str, str]]:
    """Separate the unit's brief from its `## Tool: <name>` sections.

    Each section's prose is unwrapped back to the single line the model actually
    receives, so the `.md` can be wrapped for a human reader without changing a
    byte of what goes over the wire.
    """
    brief_lines: list[str] = []
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in body.splitlines():
        if line.startswith(_TOOL_HEADING):
            current = line[len(_TOOL_HEADING):].strip()
            sections[current] = []
        elif current is None:
            brief_lines.append(line)
        else:
            sections[current].append(line)
    unwrapped = {
        name: " ".join(part.strip() for part in lines if part.strip())
        for name, lines in sections.items()
    }
    return "\n".join(brief_lines).strip() + "\n", unwrapped


def _load_unit(path: Path) -> Unit:
    meta, body = _parse_frontmatter(path.read_text(encoding="utf-8"))
    brief, tool_text = _split_tool_sections(body)

    tools = frozenset(meta.get("tools") or ())
    never = frozenset(meta.get("never") or ())
    unknown = (tools | never) - CAPABILITIES
    if unknown:
        raise ValueError(f"{path.name}: unknown capability {sorted(unknown)}")
    if tools & never:
        raise ValueError(f"{path.name}: {sorted(tools & never)} is both granted and denied")

    handlers = tuple(meta.get("handlers") or ())
    if set(handlers) != set(tool_text):
        raise ValueError(
            f"{path.name}: handlers {sorted(handlers)} do not match the "
            f"'## Tool:' sections {sorted(tool_text)}"
        )
    return Unit(
        name=meta["name"],
        role=meta.get("role", ""),
        tools=tools,
        never=never,
        handlers=handlers,
        brief=brief,
        tool_text=tool_text,
    )


def _load_roster() -> dict[str, Unit]:
    units: dict[str, Unit] = {}
    for path in sorted(_ROSTER_DIR.glob("*.md")):
        unit = _load_unit(path)
        if unit.name != path.stem:
            raise ValueError(f"{path.name}: declares name '{unit.name}'")
        units[unit.name] = unit
    return units


UNITS: dict[str, Unit] = _load_roster()  # read once per process, at import

# One handler belongs to exactly one unit — that is what makes "who acted?" answerable.
_HANDLER_UNIT: dict[str, str] = {}
for _unit in UNITS.values():
    for _handler in _unit.handlers:
        if _handler in _HANDLER_UNIT:
            raise ValueError(f"handler '{_handler}' claimed by two units")
        _HANDLER_UNIT[_handler] = _unit.name


def unit_for_handler(handler: str) -> str | None:
    """Which unit answers a model-facing tool name."""
    return _HANDLER_UNIT.get(handler)


def tool_description(handler: str) -> str:
    """The description sent to the model for a tool, sourced from its unit's `.md`."""
    unit = UNITS[_HANDLER_UNIT[handler]]
    return unit.tool_text[handler]


def build_system_prompt() -> str:
    """Dino's persona, guardrails and journey — the host's brief, verbatim."""
    return UNITS["dino"].brief


# --- Who is acting -----------------------------------------------------------

# A ContextVar rather than a module global: Streamlit gives each guest session its
# own thread, and one guest's Booker must never be able to act under another's
# context. Threads start from the default, which is exactly what we want.
_ACTING: ContextVar[str | None] = ContextVar("acting_unit", default=None)


@contextmanager
def acting_as(unit: str | None) -> Iterator[None]:
    """Run a block as `unit`, so every effect inside it is checked against its grant."""
    token = _ACTING.set(unit)
    try:
        yield
    finally:
        _ACTING.reset(token)


def acting_unit() -> str | None:
    """The unit currently acting, or None outside any unit's turn."""
    return _ACTING.get()


def require(capability: str) -> None:
    """Refuse a capability the acting unit was not granted.

    Outside any unit — a test calling a tool directly, the perks eval script — there
    is no grant to check against and the call proceeds. The harness constrains the
    units it declares; it is not a sandbox around the whole process, and pretending
    otherwise would be the decorative version of this week's work.
    """
    acting = _ACTING.get()
    if acting is None:
        return
    unit = UNITS.get(acting)
    if unit is None or not unit.grants(capability):
        raise NotGranted(
            f"'{acting}' is not granted '{capability}' — "
            f"{unit.role if unit else 'unknown unit'}. "
            f"Granted: {sorted(unit.tools) if unit else []}."
        )


def brokered(capability: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool so calling it checks the acting unit's grant first."""
    @functools.wraps(fn)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        require(capability)
        return fn(*args, **kwargs)
    return guarded


__all__ = [
    "CAPABILITIES",
    "NotGranted",
    "UNITS",
    "Unit",
    "acting_as",
    "acting_unit",
    "brokered",
    "build_system_prompt",
    "require",
    "tool_description",
    "unit_for_handler",
]
