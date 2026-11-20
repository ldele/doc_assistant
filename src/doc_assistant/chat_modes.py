"""Chat modes (A7, `docs/specs/chat-modes.md`; ADR-066 choice 6).

A mode is a name and a framing: the persona-and-task text that replaces the answer prompt's first
line. Everything after it — the grounding rule, the citing block `synthesis._CITATION_RE` parses,
the fence sentence — is `prompts.ANSWER_PROMPT_FROZEN` and the same in every mode.

Two modes are built in and live here, in code: *Research assistant* (the default, today's prompt
byte for byte) and *Critical reading*. A person's own modes are rows of `chat_modes` in
`library.db`. A conversation keeps its mode on `conversation_meta.mode_id`; empty, or an id that no
longer resolves, means the default.

**Evaluated.** A built-in mode has a gate run on the eval panel before it ships, so its answers are
recorded *evaluated*; a person's own mode has none and is recorded *not evaluated*.

**The check (decision 7).** On save, text that asks to skip citations or to use knowledge from
outside the sources earns a warning that the frozen rules still apply. It is a plain phrase check
with false positives, so it never refuses: the mode is saved either way. What *is* refused is a
format error: the prompt's own placeholders, the fence's tags, an empty or over-long text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import select
from sqlalchemy.orm import Session as OrmSession

from doc_assistant.db.models import ChatModeRow, ConversationMeta
from doc_assistant.db.session import session_scope
from doc_assistant.prompts import ANSWER_FRAMING_DEFAULT, build_answer_prompt

log = structlog.get_logger(__name__)

# Structural caps, not tuned: a framing is a persona and a task, a few sentences. 1,000 characters
# is about 250 tokens, a small share of the smallest window the app fits a turn to (A5).
FRAMING_MAX_CHARS = 1000
NAME_MAX_CHARS = 60

CRITICAL_READING_FRAMING = (
    "You are a research assistant who reads the sources critically. Where the sources "
    "themselves show it, say where they agree, where they disagree, or where they are limited "
    "(their sample, method or date), and cite each such point like any other claim. When the "
    "sources show none of this, say nothing about it: no section, no heading, and no remark "
    "that they do not."
)


@dataclass(frozen=True)
class ChatMode:
    """One chat mode. ``built_in`` modes are evaluated; a person's own are not."""

    id: str
    name: str
    framing: str
    built_in: bool

    @property
    def evaluated(self) -> bool:
        return self.built_in

    def answer_prompt(self) -> ChatPromptTemplate:
        """The answer prompt this mode builds: its framing, then the frozen remainder."""
        return build_answer_prompt(_template_safe(self.framing))


RESEARCH_ASSISTANT = ChatMode(
    id="research-assistant",
    name="Research assistant",
    framing=ANSWER_FRAMING_DEFAULT,
    built_in=True,
)
CRITICAL_READING = ChatMode(
    id="critical-reading",
    name="Critical reading",
    framing=CRITICAL_READING_FRAMING,
    built_in=True,
)
BUILT_IN: tuple[ChatMode, ...] = (RESEARCH_ASSISTANT, CRITICAL_READING)
DEFAULT_MODE = RESEARCH_ASSISTANT


class ModeError(ValueError):
    """A mode that cannot be saved or found, with a sentence a person can act on."""


class ModeNotFound(ModeError):
    """No mode has this id or name."""


@dataclass(frozen=True)
class SavedMode:
    """A saved mode and the check's warnings (empty when nothing was flagged)."""

    mode: ChatMode
    warnings: list[str]


# ============================================================
# The text: what is refused, what is flagged
# ============================================================

_PLACEHOLDER_RE = re.compile(r"\{\s*(context|question|history)\s*\}", re.IGNORECASE)
_FENCE_TAG_RE = re.compile(r"</?\s*source\b", re.IGNORECASE)

# Decision 7's check. Each pattern names what it flags; a hit is a warning, never a refusal.
_SKIP_CITATIONS = re.compile(
    r"\b(?:do\s*n[o']t|don't|never|no\s+need\s+to|without|skip|omit|avoid|stop|no)\s+"
    r"(?:\w+\s+){0,2}?(?:cit(?:e|es|ing|ations?)|references?|sources?\s+numbers?|brackets?)\b",
    re.IGNORECASE,
)
_OUTSIDE_KNOWLEDGE = re.compile(
    r"\b(?:(?:your|own|general|prior|background|world|outside|external)\s+knowledge"
    r"|from\s+(?:your\s+)?(?:memory|training)"
    r"|(?:beyond|outside(?:\s+of)?|ignore|ignoring|not\s+limited\s+to)\s+the\s+sources?"
    r"|what\s+you\s+(?:already\s+)?know)\b",
    re.IGNORECASE,
)

WARN_SKIP_CITATIONS = (
    "This text seems to ask the model not to cite. The frozen rules still apply: every claim is "
    "asked for its [n] citation in every mode."
)
WARN_OUTSIDE_KNOWLEDGE = (
    "This text seems to ask the model to use knowledge from outside the sources. The frozen rules "
    "still apply: answers use only the sources in every mode."
)


def check_framing(framing: str) -> list[str]:
    """The check's warnings for ``framing`` (decision 7); empty when nothing is flagged."""
    warnings: list[str] = []
    if _SKIP_CITATIONS.search(framing):
        warnings.append(WARN_SKIP_CITATIONS)
    if _OUTSIDE_KNOWLEDGE.search(framing):
        warnings.append(WARN_OUTSIDE_KNOWLEDGE)
    return warnings


def validate_framing(framing: str) -> str:
    """``framing`` stripped, or :class:`ModeError` for a format error (never for its meaning)."""
    text = (framing or "").strip()
    if not text:
        raise ModeError("A mode needs its text: who the assistant is and what it does.")
    if len(text) > FRAMING_MAX_CHARS:
        raise ModeError(
            f"A mode's text is at most {FRAMING_MAX_CHARS} characters; this one has {len(text)}."
        )
    if _PLACEHOLDER_RE.search(text):
        raise ModeError(
            "A mode's text cannot contain {context}, {question} or {history}: those are where the "
            "app puts the sources, the question and the history."
        )
    if _FENCE_TAG_RE.search(text):
        raise ModeError(
            "A mode's text cannot contain <source> or </source>: those tags mark the sources."
        )
    return text


def _validate_name(name: str) -> str:
    text = " ".join((name or "").split())
    if not text:
        raise ModeError("A mode needs a name.")
    if len(text) > NAME_MAX_CHARS:
        raise ModeError(f"A mode's name is at most {NAME_MAX_CHARS} characters.")
    return text


def _template_safe(framing: str) -> str:
    """``framing`` with its braces doubled, so the template reads them as text. Placeholders are
    refused at save (`validate_framing`); any other brace is a person's own character."""
    return framing.replace("{", "{{").replace("}", "}}")


# ============================================================
# Reading and writing modes
# ============================================================


def _from_row(row: ChatModeRow) -> ChatMode:
    return ChatMode(id=str(row.id), name=row.name, framing=row.framing, built_in=False)


def list_modes() -> list[ChatMode]:
    """Every mode: the built-in ones first, then a person's own by name."""
    with session_scope() as session:
        rows = session.scalars(select(ChatModeRow).order_by(ChatModeRow.name_key)).all()
        own = [_from_row(r) for r in rows]
    return [*BUILT_IN, *own]


def get_mode(mode_id: str) -> ChatMode | None:
    """The mode with ``mode_id``, built in or a person's own, or ``None``."""
    for mode in BUILT_IN:
        if mode.id == mode_id:
            return mode
    with session_scope() as session:
        row = session.get(ChatModeRow, mode_id)
        return _from_row(row) if row is not None else None


def find_mode(name: str) -> ChatMode | None:
    """The mode named ``name`` (case and spacing not mattering), or one whose id it is."""
    key = " ".join((name or "").split()).lower()
    if not key:
        return None
    for mode in BUILT_IN:
        if mode.name.lower() == key or mode.id == key:
            return mode
    with session_scope() as session:
        row = session.scalars(select(ChatModeRow).where(ChatModeRow.name_key == key)).first()
        if row is not None:
            return _from_row(row)
    return get_mode(name.strip())


def _name_taken(session: OrmSession, key: str, *, except_id: str | None = None) -> bool:
    if any(m.name.lower() == key for m in BUILT_IN):
        return True
    row = session.scalars(select(ChatModeRow).where(ChatModeRow.name_key == key)).first()
    return row is not None and str(row.id) != except_id


def create_mode(name: str, framing: str) -> SavedMode:
    """Save a person's own mode. Raises :class:`ModeError` for a format error or a taken name."""
    clean_name = _validate_name(name)
    text = validate_framing(framing)
    key = clean_name.lower()
    with session_scope() as session:
        if _name_taken(session, key):
            raise ModeError(f"A mode named “{clean_name}” already exists.")
        row = ChatModeRow(name=clean_name, name_key=key, framing=text)
        session.add(row)
        session.flush()
        mode = _from_row(row)
    warnings = check_framing(text)
    log.info("chat_mode_created", mode_id=mode.id, flagged=len(warnings))
    return SavedMode(mode, warnings)


def update_mode(mode_id: str, *, name: str | None = None, framing: str | None = None) -> SavedMode:
    """Rename or edit a person's own mode. A built-in mode cannot be changed."""
    if any(m.id == mode_id for m in BUILT_IN):
        raise ModeError("A built-in mode cannot be edited.")
    clean_name = _validate_name(name) if name is not None else None
    text = validate_framing(framing) if framing is not None else None
    with session_scope() as session:
        row = session.get(ChatModeRow, mode_id)
        if row is None:
            raise ModeNotFound("No mode has this id.")
        if clean_name is not None:
            key = clean_name.lower()
            if _name_taken(session, key, except_id=mode_id):
                raise ModeError(f"A mode named “{clean_name}” already exists.")
            row.name, row.name_key = clean_name, key
        if text is not None:
            row.framing = text
        session.flush()
        mode = _from_row(row)
    warnings = check_framing(mode.framing)
    log.info("chat_mode_updated", mode_id=mode.id, flagged=len(warnings))
    return SavedMode(mode, warnings)


def delete_mode(mode_id: str) -> int:
    """Delete a person's own mode; returns how many conversations used it. Those go back to the
    default mode from their next turn (their ``mode_id`` is cleared)."""
    if any(m.id == mode_id for m in BUILT_IN):
        raise ModeError("A built-in mode cannot be deleted.")
    with session_scope() as session:
        row = session.get(ChatModeRow, mode_id)
        if row is None:
            raise ModeNotFound("No mode has this id.")
        metas = session.scalars(
            select(ConversationMeta).where(ConversationMeta.mode_id == mode_id)
        ).all()
        for meta in metas:
            meta.mode_id = None
        session.delete(row)
    log.info("chat_mode_deleted", mode_id=mode_id, conversations=len(metas))
    return len(metas)


def deleted_mode_note(name: str, conversations: int) -> str:
    """The sentence that says what deleting a mode did to the conversations using it."""
    if conversations == 0:
        return f"Deleted the mode “{name}”. No conversation was using it."
    noun = "conversation goes" if conversations == 1 else "conversations go"
    return (
        f"Deleted the mode “{name}”. {conversations} {noun} back to "
        f"{DEFAULT_MODE.name} from the next turn."
    )


# ============================================================
# A conversation's mode
# ============================================================


def conversation_mode(session_id: str) -> ChatMode:
    """The mode a conversation's next turn uses: its stored mode, else the default."""
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        mode_id = meta.mode_id if meta is not None else None
    if not mode_id:
        return DEFAULT_MODE
    return get_mode(mode_id) or DEFAULT_MODE


def set_conversation_mode(session_id: str, mode_id: str | None) -> ChatMode:
    """Set a conversation's mode from its next turn; ``None`` or ``""`` means the default.
    Raises :class:`ModeNotFound` for an id no mode has."""
    mode = DEFAULT_MODE
    if mode_id:
        found = get_mode(mode_id)
        if found is None:
            raise ModeNotFound("No mode has this id.")
        mode = found
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        if meta is None:
            meta = ConversationMeta(session_id=session_id)
            session.add(meta)
        meta.mode_id = None if mode is DEFAULT_MODE else mode.id
    return mode


# ============================================================
# `/mode`
# ============================================================


def _mode_line(mode: ChatMode, current: ChatMode) -> str:
    kind = "built in" if mode.built_in else "yours, not evaluated"
    marker = " ← this conversation" if mode.id == current.id else ""
    return f"- **{mode.name}** ({kind}){marker}"


def mode_command(session_id: str, arg: str) -> str:
    """The `/mode` command: list the modes, or switch this conversation from its next turn."""
    current = conversation_mode(session_id)
    modes = list_modes()
    listing = "\n".join(_mode_line(m, current) for m in modes)
    if not arg.strip():
        return (
            f"**Chat mode:** {current.name}.\n\n{listing}\n\n"
            "`/mode <name>` switches this conversation from its next turn."
        )
    mode = find_mode(arg)
    if mode is None:
        return f"No mode is named “{arg.strip()}”. The modes are:\n\n{listing}"
    set_conversation_mode(session_id, mode.id)
    tail = (
        "" if mode.evaluated else " It is your own mode, so its answers are marked not evaluated."
    )
    if mode.id == current.id:
        return f"This conversation already uses **{mode.name}**.{tail}"
    return f"From the next turn this conversation uses **{mode.name}**.{tail}"
