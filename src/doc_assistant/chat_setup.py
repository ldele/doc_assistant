"""How a chat is set up: its mode and folders, fixed for the chat (A8a).

The contract is `docs/specs/chat-interface-pass.md` § "The contract — A8a". A chat's mode and
folders are chosen on the start screen and **stored on the chat** by its first turn
(`conversation_meta.mode_id` and `scope_json`; ADR-025's addendum amends S9). Later turns read the
stored setup and ignore the request's scope field, except a one-off ``/folder <name>: <question>``
(A7b), which scopes its own turn only. Changing either means *Continue in a new chat*, which can
carry the old chat's exchanges into the new chat's memory (G3), where A5 fits them like any earlier
message.

**The stored value.** ``scope_json`` holds the folder ids the chat searches, as
``{"folder_ids": [...]}``, or the JSON text ``null`` for the whole library. SQL ``NULL`` means no
setup was stored: a chat from before A8, or one with no turn yet. A chat from before A8 has its
setup worked out from its last answer record (its mode and scope); reading it writes nothing back.

**Never wider (ADR-025 F2).** Folder ids are stored as chosen. A folder deleted later narrows or
empties the search when the turn resolves the scope, and the line at the top of the chat says
which folder held nothing; nothing here falls back to the whole library.

**Defaults** (`settings.json` → ``new_chat``) are per library, since the file sits in the library's
data home. A default folder that no longer exists is dropped and named.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog
from sqlalchemy import select

from doc_assistant import app_settings
from doc_assistant.chat_modes import (
    DEFAULT_MODE,
    ChatMode,
    find_mode,
    get_mode,
    list_modes,
    set_conversation_mode,
)
from doc_assistant.db.models import AnswerRecord, ConversationMeta
from doc_assistant.db.session import session_scope
from doc_assistant.retrieval_scope import ScopeView, read_scope

log = structlog.get_logger(__name__)

#: The ``settings.json`` key the New chat defaults live under (W2).
DEFAULTS_KEY = "new_chat"

# The model's own "Sources:" / "References:" block at the end of an answer. A carried exchange
# leaves it out (the spec: "the question and the answer text, without their citation blocks").
_SOURCES_BLOCK = re.compile(r"^\W*(?:sources?|references?)\W*:", re.IGNORECASE)


@dataclass(frozen=True)
class ChatSetup:
    """A chat's mode and folders. ``folder_ids is None`` = the whole library.

    ``stored``: written on the chat (by its first turn, or by *Continue in a new chat*). A chat
    from before A8 reads ``stored=False`` with its setup worked out from its last record
    (``derived=True``). ``carried_exchanges``: how many exchanges *Continue in a new chat*
    carried into it (0 when none or not continued)."""

    mode: ChatMode = DEFAULT_MODE
    folder_ids: list[str] | None = None
    stored: bool = False
    derived: bool = False
    continued_from: str | None = None
    carried_exchanges: int = 0


def _dump_scope(folder_ids: list[str] | None) -> str:
    # `[]` (a scope that names no folder, from a record that could not be read) stays a scope:
    # it searches nothing. Only `None` is the whole library.
    if folder_ids is None:
        return "null"
    return json.dumps({"folder_ids": list(dict.fromkeys(i for i in folder_ids if i))})


def _load_scope(raw: str | None) -> list[str] | None:
    """The stored folder ids; ``None`` for the whole library. Text that cannot be read gives an
    empty list, which searches nothing: a chat that says it was scoped never widens (F2)."""
    try:
        value = json.loads(raw or "null")
    except ValueError:
        return []
    if value is None:
        return None
    if isinstance(value, dict) and isinstance(value.get("folder_ids"), list):
        return [str(i) for i in value["folder_ids"] if isinstance(i, str) and i]
    return []


#: Stands in for a stored scope that names no folder (text that could not be read), so the turn
#: resolves it as an unknown folder and searches nothing, rather than `[]`, which the resolver
#: reads as the whole library.
UNREADABLE_FOLDER_ID = "unreadable-scope"


def scope_folder_ids(setup: ChatSetup) -> list[str] | None:
    """The folder ids a turn under ``setup`` hands the scope resolver: never wider than stored."""
    if setup.folder_ids is None:
        return None
    return setup.folder_ids or [UNREADABLE_FOLDER_ID]


def _is_command(text: str | None) -> bool:
    return bool(text) and str(text).lstrip().startswith("/")


def _derive_from_records(rows: list[Any]) -> tuple[str | None, list[str] | None, bool]:
    """``(mode name, folder ids, found)`` from a chat's records, newest first.

    The last turn that ran as an ordinary question counts: a rule-answered turn (A6) records no
    scope, and a one-off ``/folder X: q`` (A7b) scoped its own turn only, so neither says where
    the chat searched. A concept scope was always a one-off command, so it is never the chat's."""
    for row in rows:
        if row.records_json is not None or _is_command(row.original_query or row.query):
            continue
        scope = read_scope(row.retrieval_scope_json)
        if scope is None or scope.concept is not None:
            return row.chat_mode, None, True
        # A record that says it was scoped but names no folder searches nothing (F2): `[]`.
        return row.chat_mode, [f.folder_id for f in scope.folders], True
    return None, None, False


def has_turns(session_id: str) -> bool:
    """Whether the chat has an answered turn on record."""
    with session_scope() as session:
        found = session.execute(
            select(AnswerRecord.id).where(AnswerRecord.session_id == session_id).limit(1)
        ).first()
    return found is not None


def read_setup(session_id: str) -> ChatSetup:
    """The chat's setup: stored, else worked out from its last record (a chat from before A8),
    else the default (a chat with no turn yet). Writes nothing."""
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        mode_id = meta.mode_id if meta is not None else None
        scope_raw = meta.scope_json if meta is not None else None
        continued_from = meta.continued_from if meta is not None else None
        carried = int(meta.carried_exchanges or 0) if meta is not None else 0
        rows: list[Any] = []
        if scope_raw is None:
            rows = list(
                session.execute(
                    select(
                        AnswerRecord.chat_mode,
                        AnswerRecord.retrieval_scope_json,
                        AnswerRecord.records_json,
                        AnswerRecord.original_query,
                        AnswerRecord.query,
                    )
                    .where(AnswerRecord.session_id == session_id)
                    .order_by(AnswerRecord.created_at.desc())
                ).all()
            )
    mode = (get_mode(mode_id) if mode_id else None) or DEFAULT_MODE
    if scope_raw is not None:
        return ChatSetup(
            mode=mode,
            folder_ids=_load_scope(scope_raw),
            stored=True,
            continued_from=continued_from,
            carried_exchanges=carried,
        )
    mode_name, folder_ids, found = _derive_from_records(rows)
    if found:
        derived_mode = (find_mode(mode_name) if mode_name else None) or mode
        return ChatSetup(
            mode=derived_mode,
            folder_ids=folder_ids,
            derived=True,
            continued_from=continued_from,
            carried_exchanges=carried,
        )
    return ChatSetup(mode=mode, continued_from=continued_from, carried_exchanges=carried)


def write_setup(
    session_id: str,
    *,
    mode_id: str | None,
    folder_ids: list[str] | None,
    continued_from: str | None = None,
    carried_exchanges: int | None = None,
) -> None:
    """Store a chat's setup. ``mode_id`` ``None`` (or the default's id) stores the default."""
    stored_mode = None if (not mode_id or mode_id == DEFAULT_MODE.id) else mode_id
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        if meta is None:
            meta = ConversationMeta(session_id=session_id)
            session.add(meta)
        meta.mode_id = stored_mode
        meta.scope_json = _dump_scope(folder_ids)
        if continued_from is not None:
            meta.continued_from = continued_from
        if carried_exchanges is not None:
            meta.carried_exchanges = carried_exchanges


def turn_setup(
    session_id: str, request_folder_ids: list[str] | None, request_mode_id: str | None
) -> ChatSetup:
    """The setup a turn runs under, stored on the chat if it was not yet (the first turn).

    A stored setup wins and the request's scope and mode are ignored. A chat from before A8
    keeps the setup its last record shows, now stored. A new chat takes the request's: the start
    screen's choice. A mode id that resolves to nothing falls back to the default (logged): a
    turn never fails on its setup."""
    setup = read_setup(session_id)
    if setup.stored:
        return setup
    if setup.derived:
        folder_ids, mode = setup.folder_ids, setup.mode
    else:
        folder_ids = list(dict.fromkeys(request_folder_ids or [])) or None
        # The mode set on the chat before its first turn (`/mode <name>` on an empty chat),
        # unless the request names one: the start screen's chip.
        mode = setup.mode
        if request_mode_id:
            found = get_mode(request_mode_id)
            if found is None:
                log.warning("chat_setup_unknown_mode", mode_id=request_mode_id)
            else:
                mode = found
    write_setup(session_id, mode_id=mode.id, folder_ids=folder_ids)
    log.info(
        "chat_setup_stored",
        session_id=session_id,
        mode_id=mode.id,
        folders=len(folder_ids or []),
        derived=setup.derived,
    )
    return ChatSetup(
        mode=mode,
        folder_ids=folder_ids,
        stored=True,
        continued_from=setup.continued_from,
        carried_exchanges=setup.carried_exchanges,
    )


def current_scope(folder_ids: list[str] | None) -> ScopeView | None:
    """``folder_ids`` resolved now, as a turn would resolve them: each folder's count, the union
    and the folders that hold nothing (F2). ``None`` = the whole library. What the start screen's
    count and the line at the top of the chat show."""
    from doc_assistant.chat_controller.helpers import _resolve_scope

    if folder_ids is None:
        return None
    return _resolve_scope(folder_ids or [UNREADABLE_FOLDER_ID])[1]


def chat_is_set(session_id: str) -> bool:
    """Whether the chat's mode is fixed: it has a turn, or a stored setup (G2's "has turns")."""
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        if meta is not None and meta.scope_json is not None:
            return True
    return has_turns(session_id)


# ============================================================
# `/mode` (G2): an open chat never changes mode
# ============================================================


@dataclass(frozen=True)
class ModeReply:
    """What `/mode` answers, and what the window does with it: ``continue`` opens *Continue in
    a new chat* prefilled with ``mode_id``; ``set`` sets the start screen's mode chip."""

    text: str
    mode_id: str | None = None
    action: Literal["continue", "set"] | None = None


NEWLINE = chr(10)
PARA = NEWLINE * 2


def _mode_line(mode: ChatMode, current: ChatMode) -> str:
    kind = "built in" if mode.built_in else "yours, not evaluated"
    marker = " ← this chat" if mode.id == current.id else ""
    return f"- **{mode.name}** ({kind}){marker}"


def mode_command(session_id: str, arg: str) -> ModeReply:
    """``/mode`` lists the modes and names the chat's own. ``/mode <name>`` on a chat that has
    turns changes nothing and offers *Continue in a new chat* with that mode; on a chat with no
    turn yet it sets the mode the chat starts with."""
    current = read_setup(session_id).mode
    listing = NEWLINE.join(_mode_line(m, current) for m in list_modes())
    is_set = chat_is_set(session_id)
    if not arg.strip():
        how = (
            "`/mode <name>` opens *Continue in a new chat* with that mode: an open chat keeps "
            "its mode."
            if is_set
            else "`/mode <name>` sets the mode this chat starts with."
        )
        return ModeReply(f"**Chat mode:** {current.name}.{PARA}{listing}{PARA}{how}")
    mode = find_mode(arg)
    if mode is None:
        return ModeReply(f"No mode is named “{arg.strip()}”. The modes are:{PARA}{listing}")
    tail = (
        "" if mode.evaluated else " It is your own mode, so its answers are marked not evaluated."
    )
    if is_set:
        if mode.id == current.id:
            return ModeReply(f"This chat already answers as **{mode.name}**.")
        return ModeReply(
            f"This chat answers as **{current.name}**. To answer as **{mode.name}**, continue "
            "in a new chat.",
            mode_id=mode.id,
            action="continue",
        )
    set_conversation_mode(session_id, mode.id)
    return ModeReply(f"This chat will answer as **{mode.name}**.{tail}", mode.id, "set")


# ============================================================
# Carry-over (G3)
# ============================================================


def _without_sources_block(answer: str) -> str:
    lines = (answer or "").splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if _SOURCES_BLOCK.match(lines[i].strip()):
            return "\n".join(lines[:i]).rstrip()
    return (answer or "").rstrip()


def _exchange_question(row: AnswerRecord) -> str:
    # A one-off scoped command (A7b) keeps its command text as `original_query`; the live history
    # held the question alone, so the carried exchange does too.
    if row.original_query and not _is_command(row.original_query):
        return row.original_query
    return row.query


@dataclass(frozen=True)
class Carried:
    """A chat's exchanges as memory messages (user, assistant, …), oldest first, and the ids of
    the answer records they were read from, in the same order."""

    messages: list[dict[str, str]] = field(default_factory=list)
    record_ids: list[str] = field(default_factory=list)

    @property
    def exchanges(self) -> int:
        return len(self.record_ids)


def carried_exchanges(session_id: str) -> Carried:
    """A chat's exchanges, read from its answer records, oldest first.

    Each is the question and the answer text without its sources block. A turn that retrieved
    nothing is left out, as the live history leaves it out: nothing was answered. Nothing is
    summarised and no model is called."""
    messages: list[dict[str, str]] = []
    record_ids: list[str] = []
    with session_scope() as session:
        rows = (
            session.execute(
                select(AnswerRecord)
                .where(AnswerRecord.session_id == session_id)
                .order_by(AnswerRecord.created_at.asc())
            )
            .scalars()
            .all()
        )
        for row in rows:
            answered = row.records_json is not None or (row.retrieved_chunks_json or "[]") != "[]"
            if not answered or not (row.answer or "").strip():
                continue
            messages.append({"role": "user", "content": _exchange_question(row)})
            messages.append({"role": "assistant", "content": _without_sources_block(row.answer)})
            record_ids.append(str(row.id))
    return Carried(messages, record_ids)


def carried_held(
    carried_messages: int, history_len: int, kept_messages: int
) -> dict[str, int] | None:
    """How many carried exchanges a turn's prompt held, for its record (beside A5's counts).

    The first ``carried_messages`` messages of the history were carried; the rewrite call holds
    the newest ``kept_messages``. ``None`` when the chat carried nothing."""
    if carried_messages <= 0:
        return None
    total = carried_messages // 2
    first_kept = history_len - kept_messages
    held = {i // 2 for i in range(max(0, first_kept), carried_messages)}
    return {"exchanges": total, "kept": len(held), "dropped": total - len(held)}


@dataclass(frozen=True)
class ContinuedChat:
    """A new chat continued from another: its setup and the memory it starts with."""

    setup: ChatSetup
    carried: Carried = field(default_factory=Carried)


def continue_chat(
    from_session_id: str,
    new_session_id: str,
    *,
    mode_id: str | None,
    folder_ids: list[str] | None,
    carry: bool,
) -> ContinuedChat:
    """Set up ``new_session_id`` as continued from ``from_session_id``; the old chat is not
    changed. With ``carry`` its exchanges become the new chat's starting memory."""
    mode = DEFAULT_MODE
    if mode_id:
        found = get_mode(mode_id)
        if found is None:
            log.warning("chat_setup_unknown_mode", mode_id=mode_id)
        else:
            mode = found
    carried = carried_exchanges(from_session_id) if carry else Carried()
    ids = list(dict.fromkeys(folder_ids or [])) or None
    write_setup(
        new_session_id,
        mode_id=mode.id,
        folder_ids=ids,
        continued_from=from_session_id,
        carried_exchanges=carried.exchanges,
    )
    log.info(
        "chat_continued",
        from_session=from_session_id,
        new_session=new_session_id,
        carried=carried.exchanges,
    )
    return ContinuedChat(
        setup=ChatSetup(
            mode=mode,
            folder_ids=ids,
            stored=True,
            continued_from=from_session_id,
            carried_exchanges=carried.exchanges,
        ),
        carried=carried,
    )


@dataclass(frozen=True)
class ContinuedFrom:
    """The chat a chat was continued from, as its line names it."""

    session_id: str
    title: str | None
    mode_name: str
    scope: ScopeView | None


@dataclass(frozen=True)
class SetupView:
    """A setup ready to show: the setup, its folders resolved now, and where it came from."""

    setup: ChatSetup
    scope: ScopeView | None
    continued_from: ContinuedFrom | None = None


def describe_setup(setup: ChatSetup) -> SetupView:
    """``setup`` with its scope resolved now (each folder's count, the union, the folders that
    hold nothing) and the chat it was continued from named."""
    origin: ContinuedFrom | None = None
    if setup.continued_from:
        old = read_setup(setup.continued_from)
        origin = ContinuedFrom(
            session_id=setup.continued_from,
            title=conversation_title(setup.continued_from),
            mode_name=old.mode.name,
            scope=current_scope(old.folder_ids),
        )
    return SetupView(setup=setup, scope=current_scope(setup.folder_ids), continued_from=origin)


def conversation_title(session_id: str) -> str | None:
    """A chat's title as the list shows it: the title set on it, else its first question."""
    from doc_assistant.conversations import _truncate

    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        if meta is not None and meta.title_override:
            return meta.title_override
        row = session.execute(
            select(AnswerRecord.original_query, AnswerRecord.query)
            .where(AnswerRecord.session_id == session_id)
            .order_by(AnswerRecord.created_at.asc())
            .limit(1)
        ).first()
    return _truncate(row[0] or row[1]) if row is not None else None


# ============================================================
# The chat list: each chat's mode and folders, for grouping (the rail's Group)
# ============================================================


@dataclass(frozen=True)
class ListedSetup:
    """What the rail groups a chat by: its mode's name and its folders (empty = whole library)."""

    mode_name: str
    folder_ids: list[str] = field(default_factory=list)


def setups_for(session_ids: list[str]) -> dict[str, ListedSetup]:
    """Each chat's mode name and folder ids, in three queries whatever the number of chats."""
    ids = [s for s in dict.fromkeys(session_ids) if s]
    if not ids:
        return {}
    with session_scope() as session:
        metas = {
            str(m.session_id): (m.mode_id, m.scope_json)
            for m in session.execute(
                select(ConversationMeta).where(ConversationMeta.session_id.in_(ids))
            ).scalars()
        }
        unstored = [s for s in ids if metas.get(s, (None, None))[1] is None]
        by_session: dict[str, list[Any]] = {}
        if unstored:
            for row in session.execute(
                select(
                    AnswerRecord.session_id,
                    AnswerRecord.chat_mode,
                    AnswerRecord.retrieval_scope_json,
                    AnswerRecord.records_json,
                    AnswerRecord.original_query,
                    AnswerRecord.query,
                )
                .where(AnswerRecord.session_id.in_(unstored))
                .order_by(AnswerRecord.created_at.desc())
            ).all():
                by_session.setdefault(str(row.session_id), []).append(row)
    out: dict[str, ListedSetup] = {}
    for sid in ids:
        mode_id, scope_raw = metas.get(sid, (None, None))
        mode = (get_mode(mode_id) if mode_id else None) or DEFAULT_MODE
        if scope_raw is not None:
            out[sid] = ListedSetup(mode.name, _load_scope(scope_raw) or [])
            continue
        mode_name, folder_ids, found = _derive_from_records(by_session.get(sid, []))
        out[sid] = ListedSetup(
            (mode_name or mode.name) if found else mode.name, list(folder_ids or [])
        )
    return out


# ============================================================
# New chat defaults (W2) — `settings.json` → `new_chat`
# ============================================================


@dataclass(frozen=True)
class NewChatDefaults:
    """The setup New chat starts from. ``missing_folders``: default folders that no longer
    exist, dropped and named. ``mode_missing``: a default mode that no longer exists (the
    default mode is used). ``nothing_left``: every default folder is gone; ``folder_ids`` then
    keeps the gone ids, which search nothing, rather than the whole library (never wider, F2)."""

    mode: ChatMode = DEFAULT_MODE
    folder_ids: list[str] | None = None
    missing_folders: list[str] = field(default_factory=list)
    mode_missing: str | None = None
    nothing_left: bool = False


def get_new_chat_defaults() -> NewChatDefaults:
    """The New chat defaults, with any folder or mode that no longer exists dropped and named."""
    from doc_assistant.library import folders_doc_hashes

    raw = app_settings.load_user_settings().get(DEFAULTS_KEY)
    if not isinstance(raw, dict):
        return NewChatDefaults()
    mode = DEFAULT_MODE
    mode_missing: str | None = None
    mode_id = raw.get("mode_id")
    if isinstance(mode_id, str) and mode_id:
        found = get_mode(mode_id)
        if found is None:
            mode_missing = str(raw.get("mode_name") or mode_id)
        else:
            mode = found
    wanted = [i for i in raw.get("folder_ids") or [] if isinstance(i, str) and i]
    if not wanted:
        return NewChatDefaults(mode=mode, mode_missing=mode_missing)
    names = folders_doc_hashes(wanted).names
    kept = [i for i in wanted if i in names]
    names_raw = raw.get("folder_names")
    remembered: dict[str, Any] = names_raw if isinstance(names_raw, dict) else {}
    missing = [
        str(remembered.get(i) or "a folder that no longer exists")
        for i in wanted
        if i not in names
    ]
    if missing:
        log.info("new_chat_default_folders_missing", missing=len(missing))
    return NewChatDefaults(
        mode=mode,
        # Every default folder gone: the gone ids are kept, so a first question searches nothing
        # and says which folder no longer exists, rather than the whole library (F2, the spec's
        # hard constraint: the defaults never fall back to the whole library).
        folder_ids=kept or wanted,
        missing_folders=missing,
        mode_missing=mode_missing,
        nothing_left=not kept,
    )


def set_new_chat_defaults(mode_id: str | None, folder_ids: list[str] | None) -> NewChatDefaults:
    """Save the New chat defaults. Unknown folder ids are not saved; an unknown mode id is a
    :class:`chat_modes.ModeNotFound`. The folders' names are kept beside their ids, so a folder
    deleted later can still be named when it is dropped."""
    from doc_assistant.chat_modes import ModeNotFound
    from doc_assistant.library import folders_doc_hashes

    mode = DEFAULT_MODE
    if mode_id:
        found = get_mode(mode_id)
        if found is None:
            raise ModeNotFound("No mode has this id.")
        mode = found
    wanted = list(dict.fromkeys(i for i in folder_ids or [] if i))
    names = folders_doc_hashes(wanted).names if wanted else {}
    kept = [i for i in wanted if i in names]
    settings = app_settings.load_user_settings()
    settings[DEFAULTS_KEY] = {
        "mode_id": mode.id,
        "mode_name": mode.name,
        "folder_ids": kept,
        "folder_names": {i: names[i] for i in kept},
    }
    app_settings.save_user_settings(settings)
    log.info("new_chat_defaults_saved", mode_id=mode.id, folders=len(kept))
    return get_new_chat_defaults()
