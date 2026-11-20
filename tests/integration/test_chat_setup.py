"""A chat's setup: stored by its first turn, defaults, `/mode`, carry-over (A8a).

The contract is `docs/specs/chat-interface-pass.md` § "The contract — A8a" and its guard tests.
Deterministic: a spy pipeline records what retrieval and the rewrite were handed; a temp SQLite
backs the records; no network, no model, no paid call.
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from doc_assistant import chat_controller, chat_modes, chat_setup, context
from doc_assistant.chat_controller import ChatController, Result, Session
from doc_assistant.chat_modes import CRITICAL_READING, DEFAULT_MODE
from doc_assistant.db import session as session_mod
from doc_assistant.db.models import AnswerRecord, Base, ConversationMeta, Folder
from doc_assistant.db.models import Document as DocRow
from doc_assistant.db.session import session_scope
from doc_assistant.prompts import REWRITE_PROMPT


@pytest.fixture(autouse=True)
def _no_paid_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("doc_assistant.llm.reviewer_available", lambda provider=None: False)


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    engine = create_engine(f"sqlite:///{tmp_path / 'setup.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield engine
    engine.dispose()


class SpyRAG:
    """One passage per document in scope; records every scope and every rewrite's history."""

    def __init__(self, hashes: list[str]) -> None:
        self.hashes = hashes
        self.llm = types.SimpleNamespace(model="fake-model")
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"
        self.scope_calls: list[frozenset[str] | None] = []
        self.rewrites: list[list[dict[str, str]]] = []

    def chunk_count(self) -> int:
        return len(self.hashes)

    def rewrite(self, question: str, history: list[dict[str, str]], counter: object = None) -> str:
        self.rewrites.append([dict(m) for m in history])
        return question

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = 10,
        *,
        use_multi_query: bool | None = None,
        scope: frozenset[str] | None = None,
    ) -> list[tuple[Document, float]]:
        self.scope_calls.append(scope)
        wanted = self.hashes if scope is None else sorted(h for h in self.hashes if h in scope)
        return [
            (
                Document(
                    page_content=f"Passage from {h}.",
                    metadata={"document_id": h, "chunk_index": 0, "filename": h, "doc_hash": h},
                ),
                0.9,
            )
            for h in wanted
        ]

    def stream_answer(
        self, question: str, docs: list[Document], counter: object = None, **kw: object
    ) -> Iterator[str]:
        yield f"Answer to {question} [1].\n\nSources:\n[1] {docs[0].metadata['filename']}"


def _doc(name: str) -> str:
    with session_scope() as session:
        doc = DocRow(
            filename=name,
            source_original=f"C:/lib/{name}",
            doc_hash=f"h-{name}",
            format="pdf",
            title=f"Title of {name}",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _folder(name: str, doc_ids: list[str]) -> str:
    with session_scope() as session:
        folder = Folder(name=name)
        session.add(folder)
        session.flush()
        for d in doc_ids:
            got = session.get(DocRow, d)
            assert got is not None
            got.folders.append(folder)
        return str(folder.id)


def _delete_folder(folder_id: str) -> None:
    with session_scope() as session:
        got = session.get(Folder, folder_id)
        assert got is not None
        session.delete(got)


def _turn(controller: ChatController, session: Session, text: str, **kw: Any) -> Any:
    events = list(controller.handle_message(session, text, **kw))
    return next(e.result for e in events if isinstance(e, Result))


def _meta(session_id: str) -> tuple[str | None, str | None, str | None, int | None] | None:
    with session_scope() as session:
        meta = session.get(ConversationMeta, session_id)
        if meta is None:
            return None
        return meta.mode_id, meta.scope_json, meta.continued_from, meta.carried_exchanges


def _rows(session_id: str) -> list[tuple[str, str, str | None]]:
    with session_scope() as session:
        return [
            (str(r.id), r.answer, r.context_json)
            for r in session.execute(
                select(AnswerRecord)
                .where(AnswerRecord.session_id == session_id)
                .order_by(AnswerRecord.created_at.asc())
            ).scalars()
        ]


# --- the stored setup ------------------------------------------------------------------- #


def test_the_first_turn_writes_the_setup_and_later_turns_ignore_the_request(temp_db) -> None:
    a, b = _doc("a.pdf"), _doc("b.pdf")
    thesis = _folder("Thesis", [a])
    methods = _folder("Methods", [b])
    rag = SpyRAG(["h-a.pdf", "h-b.pdf"])
    controller, session = ChatController(rag=rag), Session(session_id="chat-1")

    _turn(controller, session, "q1", scope_folder_ids=[thesis], mode_id=CRITICAL_READING.id)
    assert _meta("chat-1") == (
        CRITICAL_READING.id,
        json.dumps({"folder_ids": [thesis]}),
        None,
        None,
    )
    # A later turn's request scope and mode are ignored: the chat keeps its own.
    _turn(controller, session, "q2", scope_folder_ids=[methods], mode_id=DEFAULT_MODE.id)
    _turn(controller, session, "q3", scope_folder_ids=None)
    assert rag.scope_calls == [frozenset({"h-a.pdf"})] * 3
    assert chat_modes.conversation_mode("chat-1") == CRITICAL_READING


def test_a_one_off_folder_command_scopes_only_its_own_turn(temp_db) -> None:
    a, b = _doc("a.pdf"), _doc("b.pdf")
    _folder("Thesis", [a])
    methods = _folder("Methods", [b])
    rag = SpyRAG(["h-a.pdf", "h-b.pdf"])
    controller, session = ChatController(rag=rag), Session(session_id="chat-2")

    _turn(controller, session, "/folder Thesis: what is X?", scope_folder_ids=[methods])
    _turn(controller, session, "and Y?")
    # The command searched Thesis once; the chat's setup is the start screen's (Methods).
    assert rag.scope_calls == [frozenset({"h-a.pdf"}), frozenset({"h-b.pdf"})]
    assert chat_setup.read_setup("chat-2").folder_ids == [methods]


def test_an_unscoped_default_first_turn_stores_the_whole_library(temp_db) -> None:
    rag = SpyRAG(["h-a.pdf"])
    controller, session = ChatController(rag=rag), Session(session_id="chat-3")
    _turn(controller, session, "q1")
    assert _meta("chat-3") == (None, "null", None, None)
    setup = chat_setup.read_setup("chat-3")
    assert setup.stored and setup.folder_ids is None and setup.mode == DEFAULT_MODE


def test_a_deleted_setup_folder_narrows_never_widens(temp_db) -> None:
    a = _doc("a.pdf")
    _doc("b.pdf")
    thesis = _folder("Thesis", [a])
    rag = SpyRAG(["h-a.pdf", "h-b.pdf"])
    controller, session = ChatController(rag=rag), Session(session_id="chat-4")
    _turn(controller, session, "q1", scope_folder_ids=[thesis])
    _delete_folder(thesis)

    result = _turn(controller, session, "q2")
    assert rag.scope_calls[-1] == frozenset()  # searched nothing, not the whole library
    assert result.answer == "That folder no longer exists, so nothing was searched."
    view = chat_setup.describe_setup(chat_setup.read_setup("chat-4"))
    assert view.scope is not None and view.scope.dropped and view.scope.doc_count == 0


def test_an_unreadable_stored_scope_searches_nothing(temp_db) -> None:
    rag = SpyRAG(["h-a.pdf"])
    with session_scope() as s:
        s.add(ConversationMeta(session_id="chat-5", scope_json="{not json"))
    _turn(ChatController(rag=rag), Session(session_id="chat-5"), "q1")
    assert rag.scope_calls == [frozenset()]


# --- chats from before A8 ------------------------------------------------------------------ #


def _seed_record(session_id: str, minutes: int, **kw: Any) -> None:
    with session_scope() as session:
        session.add(
            AnswerRecord(
                session_id=session_id,
                query=kw.pop("query", "q"),
                answer=kw.pop("answer", "a [1]"),
                retrieved_chunks_json=kw.pop("chunks", json.dumps([{"filename": "a.pdf"}])),
                created_at=datetime(2026, 11, 1, 10, 0, 0) + timedelta(minutes=minutes),
                **kw,
            )
        )


def test_a_pre_a8_chat_gets_its_line_from_its_last_record_and_nothing_is_written(
    temp_db,
) -> None:
    thesis = _folder("Thesis", [_doc("a.pdf")])
    old_scope = json.dumps({"folder_id": thesis, "folder_name": "Thesis", "doc_count": 1})
    _seed_record("old-1", 0, chat_mode="Research assistant")
    _seed_record("old-1", 1, chat_mode="Critical reading", retrieval_scope_json=old_scope)
    # A one-off command and a rule-answered turn after it do not say where the chat searched.
    _seed_record("old-1", 2, query="/folder Thesis", records_json=json.dumps({"lookup": "x"}))

    setup = chat_setup.read_setup("old-1")
    assert setup.derived and not setup.stored
    assert setup.mode == CRITICAL_READING and setup.folder_ids == [thesis]
    assert _meta("old-1") is None  # reading wrote nothing back
    listed = chat_setup.setups_for(["old-1"])["old-1"]
    assert (listed.mode_name, listed.folder_ids) == ("Critical reading", [thesis])


def test_a_pre_a8_chat_with_no_record_mode_reads_the_default(temp_db) -> None:
    _seed_record("old-2", 0)
    setup = chat_setup.read_setup("old-2")
    assert setup.derived and setup.mode == DEFAULT_MODE and setup.folder_ids is None


# --- defaults ------------------------------------------------------------------------------- #


def test_defaults_round_trip_and_a_missing_folder_is_dropped_and_named(temp_db) -> None:
    thesis = _folder("Thesis", [_doc("a.pdf")])
    methods = _folder("Methods", [_doc("b.pdf")])
    saved = chat_setup.set_new_chat_defaults(CRITICAL_READING.id, [thesis, methods, "nope"])
    assert saved.mode == CRITICAL_READING and saved.folder_ids == [thesis, methods]

    _delete_folder(methods)
    got = chat_setup.get_new_chat_defaults()
    assert got.folder_ids == [thesis]
    assert got.missing_folders == ["Methods"]
    assert not got.nothing_left


def test_defaults_whose_folders_are_all_gone_search_nothing_not_everything(temp_db) -> None:
    thesis = _folder("Thesis", [_doc("a.pdf")])
    chat_setup.set_new_chat_defaults(None, [thesis])
    _delete_folder(thesis)
    got = chat_setup.get_new_chat_defaults()
    assert got.nothing_left and got.missing_folders == ["Thesis"]
    assert got.folder_ids == [thesis]  # kept: it resolves to nothing, never to the whole library
    rag = SpyRAG(["h-a.pdf"])
    _turn(ChatController(rag=rag), Session(session_id="d1"), "q", scope_folder_ids=got.folder_ids)
    assert rag.scope_calls == [frozenset()]


def test_a_default_mode_that_no_longer_exists_is_named(temp_db) -> None:
    mode = chat_modes.create_mode("Tutor", "You are a tutor.").mode
    chat_setup.set_new_chat_defaults(mode.id, None)
    chat_modes.delete_mode(mode.id)
    got = chat_setup.get_new_chat_defaults()
    assert got.mode == DEFAULT_MODE and got.mode_missing == "Tutor"


def test_no_defaults_is_the_default_mode_and_the_whole_library(temp_db) -> None:
    got = chat_setup.get_new_chat_defaults()
    assert got.mode == DEFAULT_MODE and got.folder_ids is None and got.missing_folders == []


# --- /mode (G2) ----------------------------------------------------------------------------- #


def test_mode_on_a_chat_with_turns_changes_nothing_and_offers_a_new_chat(temp_db) -> None:
    rag = SpyRAG(["h-a.pdf"])
    controller, session = ChatController(rag=rag), Session(session_id="chat-6")
    _turn(controller, session, "q1")

    result = _turn(controller, session, "/mode critical reading")
    assert result.answer == (
        "This chat answers as **Research assistant**. To answer as **Critical reading**, "
        "continue in a new chat."
    )
    assert result.mode_offer == ("continue", CRITICAL_READING.id)
    assert chat_modes.conversation_mode("chat-6") == DEFAULT_MODE
    assert "← this chat" in _turn(controller, session, "/mode").answer


def test_mode_on_an_empty_chat_sets_the_mode_and_writes_no_scope(temp_db) -> None:
    controller, session = ChatController(rag=SpyRAG([])), Session(session_id="chat-7")
    result = _turn(controller, session, "/mode critical reading")
    assert result.mode_offer == ("set", CRITICAL_READING.id)
    assert chat_modes.conversation_mode("chat-7") == CRITICAL_READING
    assert _meta("chat-7") == (CRITICAL_READING.id, None, None, None)  # no setup stored yet


# --- carry-over (G3) ------------------------------------------------------------------------ #


def _old_chat(controller: ChatController, session_id: str, n: int) -> None:
    session = Session(session_id=session_id)
    for i in range(n):
        _turn(controller, session, f"old question {i}")


def test_a_new_chat_carries_the_old_exchanges_and_the_old_chat_is_unchanged(temp_db) -> None:
    rag = SpyRAG(["h-a.pdf"])
    controller = ChatController(rag=rag)
    _old_chat(controller, "old", 2)
    before = _rows("old"), _meta("old")

    continued = chat_setup.continue_chat(
        "old", "new", mode_id=CRITICAL_READING.id, folder_ids=None, carry=True
    )
    assert continued.carried.exchanges == 2
    assert continued.carried.messages[0] == {"role": "user", "content": "old question 0"}
    # The answer text without its sources block.
    assert continued.carried.messages[1]["content"] == "Answer to old question 0 [1]."
    assert _meta("new") == (CRITICAL_READING.id, "null", "old", 2)

    session = Session(
        session_id="new",
        history=list(continued.carried.messages),
        carried_messages=len(continued.carried.messages),
    )
    rag.rewrites.clear()
    _turn(controller, session, "and then?")
    assert [m["content"] for m in rag.rewrites[0] if m["role"] == "user"] == [
        "old question 0",
        "old question 1",
    ]
    record = json.loads(_rows("new")[0][2] or "{}")
    assert record["carried"] == {"exchanges": 2, "kept": 2, "dropped": 0}
    assert (_rows("old"), _meta("old")) == before  # the old chat is not changed


def test_carried_exchanges_are_fitted_oldest_first_on_a_small_window(
    temp_db, monkeypatch: pytest.MonkeyPatch
) -> None:
    rag = SpyRAG(["h-a.pdf"])
    controller = ChatController(rag=rag)
    _old_chat(controller, "old", 4)
    carried = chat_setup.carried_exchanges("old")
    assert carried.exchanges == 4

    question = "and then?"
    fixed = context.estimate_tokens(REWRITE_PROMPT.format(history="", question=question))
    per_message = max(context._message_tokens(m) for m in carried.messages)
    window = context.Window(fixed + 3 * per_message, "set")
    monkeypatch.setattr(context, "window_for", lambda provider, server_window=None: window)
    monkeypatch.setattr(chat_controller.controller, "_reply_reserve", lambda p, m: 0)

    session = Session(
        session_id="new", history=list(carried.messages), carried_messages=len(carried.messages)
    )
    rag.rewrites.clear()
    _turn(controller, session, question)
    held = rag.rewrites[0]
    assert held == carried.messages[-len(held) :]  # the newest kept, the oldest dropped first
    record = json.loads(_rows("new")[0][2] or "{}")
    assert record["carried"]["exchanges"] == 4
    assert record["carried"]["dropped"] >= 2
    assert record["carried"]["kept"] + record["carried"]["dropped"] == 4


def test_a_turn_that_carried_nothing_records_no_carried_count(temp_db) -> None:
    rag = SpyRAG(["h-a.pdf"])
    controller, session = ChatController(rag=rag), Session(session_id="plain")
    _turn(controller, session, "q1")
    _turn(controller, session, "q2")
    assert "carried" not in json.loads(_rows("plain")[1][2] or "{}")


def test_a_turn_that_retrieved_nothing_is_not_carried(temp_db) -> None:
    _seed_record("old-3", 0, query="asked", answer="answered [1]")
    _seed_record("old-3", 1, query="nothing", answer="No documents matched.", chunks="[]")
    carried = chat_setup.carried_exchanges("old-3")
    assert [m["content"] for m in carried.messages] == ["asked", "answered [1]"]


def test_carried_held_counts_whole_and_partial_exchanges() -> None:
    assert chat_setup.carried_held(0, 4, 4) is None
    assert chat_setup.carried_held(8, 10, 6) == {"exchanges": 4, "kept": 2, "dropped": 2}
    assert chat_setup.carried_held(8, 8, 3) == {"exchanges": 4, "kept": 2, "dropped": 2}
    assert chat_setup.carried_held(4, 6, 2) == {"exchanges": 2, "kept": 0, "dropped": 2}


# --- the API -------------------------------------------------------------------------------- #


@pytest.fixture
def api(temp_db) -> tuple[TestClient, SpyRAG]:
    rag = SpyRAG(["h-a.pdf", "h-b.pdf"])
    return TestClient(create_app(controller=ChatController(rag=rag))), rag  # type: ignore[arg-type]


def _send(client: TestClient, session_id: str, text: str, **body: Any) -> str:
    r = client.post("/api/chat", json={"text": text, "session_id": session_id, **body})
    assert r.status_code == 200
    return r.text


def test_api_first_turn_takes_the_start_screens_mode_and_folders(api) -> None:
    client, rag = api
    a = _doc("a.pdf")
    thesis = _folder("Thesis", [a])
    _send(client, "s1", "q1", scope_folder_ids=[thesis], mode_id=CRITICAL_READING.id)
    _send(client, "s1", "q2", scope_folder_ids=None)
    assert rag.scope_calls == [frozenset({"h-a.pdf"})] * 2

    setup = client.get("/api/conversations/s1/setup").json()
    assert setup["mode"]["id"] == CRITICAL_READING.id and setup["stored"] is True
    assert setup["folder_ids"] == [thesis]
    assert setup["scope"]["doc_count"] == 1
    detail = client.get("/api/conversations/s1").json()
    assert detail["setup"]["folder_ids"] == [thesis]
    listed = client.get("/api/conversations").json()
    assert listed[0]["mode_name"] == "Critical reading"
    assert listed[0]["folders"] == [{"folder_id": thesis, "name": "Thesis"}]


def test_api_scope_preview_counts_the_union_once(api) -> None:
    client, _ = api
    a, b, c = _doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")
    one = _folder("One", [a, b])
    two = _folder("Two", [b, c])
    body = client.get("/api/conversations/scope", params={"folder_ids": [one, two]}).json()
    assert body["doc_count"] == 3
    assert [f["doc_count"] for f in body["folders"]] == [2, 2]
    assert client.get("/api/conversations/scope").json() is None


def test_api_mode_reply_carries_its_offer(api) -> None:
    client, _ = api
    stream = _send(client, "s2", "/mode critical reading")
    assert '"mode_offer":{"action":"set","mode_id":"critical-reading"}' in stream


def test_api_continue_carries_and_returns_the_carried_turns(api) -> None:
    client, rag = api
    _send(client, "old", "first question")
    _send(client, "old", "second question")
    r = client.post(
        "/api/conversations/old/continue",
        json={"session_id": "new", "mode_id": CRITICAL_READING.id, "folder_ids": None},
    )
    assert r.status_code == 200
    body = r.json()
    assert [t["question"] for t in body["carried"]] == ["first question", "second question"]
    assert body["setup"]["continued_from"]["session_id"] == "old"
    assert body["setup"]["continued_from"]["mode_name"] == "Research assistant"
    assert body["setup"]["carried_exchanges"] == 2

    rag.rewrites.clear()
    _send(client, "new", "and then?")
    assert [m["content"] for m in rag.rewrites[0] if m["role"] == "user"] == [
        "first question",
        "second question",
    ]
    assert (
        client.post(
            "/api/conversations/old/continue", json={"session_id": "x", "mode_id": "gone"}
        ).status_code
        == 404
    )


def test_api_continue_without_carrying_starts_empty(api) -> None:
    client, rag = api
    _send(client, "old", "first question")
    body = client.post(
        "/api/conversations/old/continue", json={"session_id": "new", "carry": False}
    ).json()
    assert body["carried"] == [] and body["setup"]["carried_exchanges"] == 0
    rag.rewrites.clear()
    _send(client, "new", "fresh")
    assert rag.rewrites == []  # no earlier message, so no rewrite call


def test_api_continue_this_chat_carries_its_own_exchanges(api, temp_db) -> None:
    """Open question 1: a reopened chat continued remembers its own exchanges (the resume
    spec's decision 1, an empty memory, is superseded)."""
    client, rag = api
    _seed_record("past", 0, query="what is X?", answer="X is Y [1].")
    r = client.post("/api/conversations/past/resume")
    assert r.json() == {"session_id": "past", "carried_exchanges": 1}
    rag.rewrites.clear()
    _send(client, "past", "and Z?")
    assert rag.rewrites[0] == [
        {"role": "user", "content": "what is X?"},
        {"role": "assistant", "content": "X is Y [1]."},
    ]
    row = _rows("past")[-1]
    assert json.loads(row[2] or "{}")["carried"] == {"exchanges": 1, "kept": 1, "dropped": 0}


def test_api_defaults_round_trip(api) -> None:
    client, _ = api
    thesis = _folder("Thesis", [_doc("a.pdf")])
    assert client.get("/api/settings/new-chat").json()["mode_id"] == DEFAULT_MODE.id
    saved = client.put(
        "/api/settings/new-chat", json={"mode_id": CRITICAL_READING.id, "folder_ids": [thesis]}
    ).json()
    assert (saved["mode_id"], saved["folder_ids"]) == (CRITICAL_READING.id, [thesis])
    assert client.put("/api/settings/new-chat", json={"mode_id": "gone"}).status_code == 404


def test_api_zero_documents_start_and_first_question(api) -> None:
    client, _ = api
    assert client.get("/api/conversations/scope").json() is None
    assert client.get("/api/settings/new-chat").json()["folder_ids"] is None


def test_the_setup_columns_are_added_to_an_old_database_and_read_back_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy import text

    from doc_assistant.db.migrations import _apply_additive_columns

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        for column in ("scope_json", "continued_from", "carried_exchanges"):
            conn.execute(text(f"ALTER TABLE conversation_meta DROP COLUMN {column}"))
        conn.execute(
            text(
                "INSERT INTO conversation_meta (session_id, pinned, archived, updated_at) "
                "VALUES ('conv-old', 0, 0, '2026-11-01 10:00:00')"
            )
        )
    added = _apply_additive_columns(engine)
    assert {
        "conversation_meta.scope_json",
        "conversation_meta.continued_from",
        "conversation_meta.carried_exchanges",
    } <= set(added)
    assert _apply_additive_columns(engine) == [], "idempotent"
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    setup = chat_setup.read_setup("conv-old")
    assert not setup.stored and setup.continued_from is None and setup.carried_exchanges == 0
    engine.dispose()
