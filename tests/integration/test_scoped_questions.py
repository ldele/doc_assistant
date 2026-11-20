"""Guard tests for A7b (`docs/specs/chat-scoped-questions.md`, ADR-066 choice 7).

Several folders per turn searched as one set (G1), a concept's documents as a scope (G2), the
`/folder <name>: <question>` and `/concept <name>: <question>` commands (G3), the chat list's
labels and filters (G4), the record's new shape read beside the old one, and MCP's additive
`scope` field (G6). The unscoped turn is pinned unchanged by `test_turn_parity.py`.

Offline: a temp SQLite library and a fake retrieval that records the scope it was handed. No
models, no network, $0.
"""

from __future__ import annotations

import json
import random
import time
import types
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, event, insert, select
from sqlalchemy.orm import sessionmaker

from doc_assistant import chat_controller
from doc_assistant.chat_controller import ChatController, Result, Session
from doc_assistant.chat_controller.helpers import _no_sources_answer, _resolve_scope, _scope_note
from doc_assistant.db import session as session_mod
from doc_assistant.db.models import (
    AnswerRecord,
    Base,
    Concept,
    ConceptAlias,
    ConceptPresenceRow,
    Folder,
    document_folders,
)
from doc_assistant.db.models import Document as DocRow
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge import concept_graph_view
from doc_assistant.knowledge.concept_graph_view import GraphStaleness
from doc_assistant.retrieval_scope import (
    GRAPH_STALE_NOTE,
    ScopeConcept,
    ScopeFolder,
    ScopeView,
    read_scope,
    scope_dict,
    scope_label,
    scope_summary,
)


@pytest.fixture(autouse=True)
def _no_paid_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-source turn trips a confidence signal, which would call a billed reviewer."""
    monkeypatch.setattr("doc_assistant.llm.reviewer_available", lambda provider=None: False)


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    engine = create_engine(f"sqlite:///{tmp_path / 'scoped.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield engine
    engine.dispose()


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The concept graph's staleness verdict, set per test: ``None`` = never built."""
    state: dict[str, Any] = {"verdict": _verdict(stale=False)}
    monkeypatch.setattr(concept_graph_view, "graph_staleness", lambda: state["verdict"])
    return state


def _verdict(*, stale: bool) -> GraphStaleness:
    return GraphStaleness(
        stale=stale, n_concepts_in_db=1, n_concepts_in_skeleton=1, added_labels=(), removed_ids=()
    )


class SpyRAG:
    """Returns one passage per document in scope (or one per seeded hash when unscoped) and
    records every scope it was handed, so a test can assert what retrieval saw."""

    def __init__(self, hashes: list[str]) -> None:
        self.hashes = hashes
        self.llm = types.SimpleNamespace(model="fake-model")
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"
        self.scope_calls: list[frozenset[str] | None] = []
        self.queries: list[str] = []

    def chunk_count(self) -> int:
        return len(self.hashes)

    def rewrite(self, question: str, history: list[dict[str, str]], counter: object = None) -> str:
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
        self.queries.append(query)
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
        yield "Answer [1]."


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


def _concept(label: str, present_in: list[str]) -> str:
    with session_scope() as session:
        c = Concept(label=label, kind="concept", graph_include=True)
        session.add(c)
        session.flush()
        session.add(ConceptAlias(concept_id=c.id, alias=label, breadth="exact"))
        for d in present_in:
            session.add(
                ConceptPresenceRow(
                    concept_id=c.id, document_id=d, chunk_keys_json="[]", n_mentions=1
                )
            )
        return str(c.id)


def _result(controller: ChatController, text: str, **kw: Any) -> Any:
    events = list(controller.handle_message(Session(), text, **kw))
    return next(e.result for e in events if isinstance(e, Result))


def _records() -> list[AnswerRecord]:
    with session_scope() as session:
        rows = list(session.execute(select(AnswerRecord)).scalars())
        session.expunge_all()
        return rows


# --- G1: several folders, one set ------------------------------------------------------- #


def test_two_folders_that_share_a_document_search_it_once(temp_db) -> None:
    a, b, c = _doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")
    one = _folder("One", [a, b])
    two = _folder("Two", [b, c])

    scope, view = _resolve_scope([one, two])

    assert scope == frozenset({"h-a.pdf", "h-b.pdf", "h-c.pdf"})
    assert view is not None and view.doc_count == 3, "b is counted once"
    assert [(f.folder_name, f.doc_count) for f in view.folders] == [("One", 2), ("Two", 2)]


def test_a_folder_that_holds_nothing_is_left_out_and_named(temp_db) -> None:
    a = _doc("a.pdf")
    live = _folder("Thesis", [a])
    empty = _folder("Drafts", [])
    rag = SpyRAG(["h-a.pdf", "h-z.pdf"])

    result = _result(
        ChatController(rag=rag), "how?", scope_folder_ids=["never-existed", live, empty]
    )

    assert rag.scope_calls == [frozenset({"h-a.pdf"})], "the live folder's documents only"
    assert result.scope.doc_count == 1
    note = result.provenance_card_md
    assert "Searched **Thesis** only: 1 document." in note
    assert "A folder that no longer exists was left out." in note
    assert "**Drafts** held no documents and was left out." in note


def test_every_folder_bad_searches_nothing(temp_db) -> None:
    empty = _folder("Drafts", [])
    rag = SpyRAG(["h-a.pdf"])

    result = _result(ChatController(rag=rag), "how?", scope_folder_ids=["gone-1", "gone-2", empty])

    assert rag.scope_calls == [frozenset()], "an empty scope, never None (F2, S3)"
    assert result.sources == [] and result.route == "no_sources"
    assert result.answer.startswith("None of the chosen folders holds a document")


def test_duplicate_folder_ids_count_once(temp_db) -> None:
    folder = _folder("Thesis", [_doc("a.pdf"), _doc("b.pdf")])

    _scope, view = _resolve_scope([folder, folder, folder])

    assert view is not None and len(view.folders) == 1 and view.doc_count == 2


def test_none_or_an_empty_list_is_the_whole_library(temp_db) -> None:
    assert _resolve_scope(None) == (None, None)
    assert _resolve_scope([]) == (None, None)


def test_the_union_is_one_query_whatever_the_number_of_folders(temp_db) -> None:
    ids = [_folder(f"F{i}", [_doc(f"d{i}.pdf")]) for i in range(6)]
    statements: list[str] = []

    def count(conn, cursor, statement, *args) -> None:
        statements.append(statement)

    event.listen(temp_db, "before_cursor_execute", count)
    try:
        _resolve_scope(ids)
    finally:
        event.remove(temp_db, "before_cursor_execute", count)

    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 2, "the folders' names, then their members: never one per folder"
    assert sum("document_folders" in s for s in selects) == 1


# --- G2: a concept's documents ---------------------------------------------------------- #


def test_a_concept_scope_is_its_presence_documents_only(temp_db, graph) -> None:
    a, b = _doc("a.pdf"), _doc("b.pdf")
    _doc("broad.pdf")  # only a broad form reaches it: kept beside presence, never in it
    concept_id = _concept("synapse", [a, b])

    scope, view = _resolve_scope(None, concept_id)

    assert scope == frozenset({"h-a.pdf", "h-b.pdf"})
    assert view is not None and view.concept is not None
    assert (view.concept.name, view.concept.graph_stale, view.doc_count) == ("synapse", False, 2)
    assert view.folders == []


def test_a_concept_with_no_presence_searches_nothing_and_says_rebuild(temp_db, graph) -> None:
    _doc("a.pdf")
    _concept("synapse", [])
    rag = SpyRAG(["h-a.pdf"])

    result = _result(ChatController(rag=rag), "/concept synapse: what is it?")

    assert rag.scope_calls == [frozenset()]
    assert result.route == "no_sources" and result.sources == []
    assert "has no documents yet" in result.answer
    assert "Rebuild the concept graph" in result.answer


def test_a_graph_never_built_gives_an_empty_concept_scope(temp_db, graph) -> None:
    a = _doc("a.pdf")
    concept_id = _concept("synapse", [a])  # a presence row left from somewhere: not trusted
    graph["verdict"] = None

    scope, view = _resolve_scope(None, concept_id)

    assert scope == frozenset()
    assert view is not None and view.doc_count == 0


def test_a_stale_graph_is_said_beside_the_concept(temp_db, graph) -> None:
    concept_id = _concept("synapse", [_doc("a.pdf")])
    graph["verdict"] = _verdict(stale=True)

    _scope, view = _resolve_scope(None, concept_id)

    assert view is not None and view.concept is not None and view.concept.graph_stale
    assert GRAPH_STALE_NOTE in _scope_note(view)


# --- G3: the commands ------------------------------------------------------------------- #


def test_folder_command_with_a_question_scopes_the_turn(temp_db) -> None:
    a = _doc("a.pdf")
    folder = _folder("Thesis", [a])
    rag = SpyRAG(["h-a.pdf", "h-other.pdf"])

    result = _result(ChatController(rag=rag), "/folder thesis: What is X?")

    assert rag.scope_calls == [frozenset({"h-a.pdf"})]
    assert rag.queries == ["What is X?"], "the question as typed, case kept, without the command"
    assert [f.folder_id for f in result.scope.folders] == [folder]
    (rec,) = _records()
    assert rec.query == "What is X?"
    assert rec.original_query == "/folder thesis: What is X?"
    assert read_scope(rec.retrieval_scope_json) == result.scope


def test_a_further_separator_stays_in_the_question(temp_db) -> None:
    _folder("Thesis", [_doc("a.pdf")])
    rag = SpyRAG(["h-a.pdf"])

    _result(ChatController(rag=rag), "/folder Thesis: define: synapse")

    assert rag.queries == ["define: synapse"]


@pytest.mark.parametrize("text", ["/folder Thesis", "/folder Thesis:", "/folder Thesis:   "])
def test_folder_command_without_a_question_stays_a6s_lookup(temp_db, text) -> None:
    _folder("Thesis", [_doc("a.pdf")])
    rag = SpyRAG(["h-a.pdf"])

    result = _result(ChatController(rag=rag), text)

    assert rag.scope_calls == [], "no retrieval"
    assert result.route == "library"
    assert result.answer.startswith("The folder **Thesis** holds **1** documents")


def test_an_unknown_folder_lists_the_folders_and_asks_nothing(temp_db) -> None:
    _folder("Thesis", [_doc("a.pdf")])
    rag = SpyRAG(["h-a.pdf"])

    result = _result(ChatController(rag=rag), "/folder Nope: what is X?")

    assert rag.scope_calls == []
    assert result.route == "library"
    assert "No folder is named “Nope”. The folders are: Thesis." in result.answer
    (rec,) = _records()
    assert json.loads(rec.records_json or "{}")["args"] == {
        "folder": "Nope",
        "question": "what is X?",
    }


def test_an_ambiguous_folder_asks_which_and_asks_nothing(temp_db, monkeypatch) -> None:
    from doc_assistant.read_surface import records
    from doc_assistant.read_surface.shapes import FolderItem, FolderList

    twins = [FolderItem("f1", "Notes", 2), FolderItem("f2", "notes", 3)]
    monkeypatch.setattr(records, "list_folders", lambda: FolderList(folders=twins, total=2))
    rag = SpyRAG(["h-a.pdf"])

    result = _result(ChatController(rag=rag), "/folder Notes: what is X?")

    assert rag.scope_calls == []
    assert "Which one do you mean?" in result.answer
    assert json.loads(_records()[0].records_json or "{}")["asked"] is True


def test_an_unknown_concept_lists_the_concepts_and_asks_nothing(temp_db, graph) -> None:
    _concept("synapse", [])
    rag = SpyRAG(["h-a.pdf"])

    result = _result(ChatController(rag=rag), "/concept dendrite: what is it?")

    assert rag.scope_calls == []
    assert "“dendrite” is not a concept you have taken on" in result.answer
    assert "The concepts you have taken on are: synapse." in result.answer


def test_concept_command_with_a_question_scopes_to_its_documents(temp_db, graph) -> None:
    a, b = _doc("a.pdf"), _doc("b.pdf")
    _concept("synapse", [a])
    rag = SpyRAG(["h-a.pdf", "h-b.pdf"])

    result = _result(ChatController(rag=rag), "/concept Synapse: how does it work?")

    assert rag.scope_calls == [frozenset({"h-a.pdf"})]
    assert result.scope.concept.name == "synapse"
    assert "Searched the documents where **synapse** is present: 1 document." in (
        result.provenance_card_md
    )
    assert b  # b is in the library and was not searched


def test_a_command_scope_lasts_its_own_turn(temp_db) -> None:
    _folder("Thesis", [_doc("a.pdf")])
    rag = SpyRAG(["h-a.pdf", "h-b.pdf"])
    controller, session = ChatController(rag=rag), Session()

    list(controller.handle_message(session, "/folder Thesis: q1"))
    list(controller.handle_message(session, "q2"))

    assert rag.scope_calls == [frozenset({"h-a.pdf"}), None]


# --- the record: both shapes, one reader ------------------------------------------------ #


def test_the_record_round_trips_in_the_new_shape() -> None:
    view = ScopeView(
        folders=[ScopeFolder("f1", "A", 9), ScopeFolder("f2", None, 0)],
        doc_count=9,
    )
    concept = ScopeView(concept=ScopeConcept("c1", "synapse", True), doc_count=4)

    assert read_scope(json.dumps(scope_dict(view))) == view
    assert read_scope(json.dumps(scope_dict(concept))) == concept
    assert read_scope(None) is None and read_scope("null") is None and read_scope("") is None


def test_an_old_shape_record_reads_as_a_one_folder_scope() -> None:
    old = json.dumps({"folder_id": "f1", "folder_name": "Thesis", "doc_count": 5})

    assert read_scope(old) == ScopeView(folders=[ScopeFolder("f1", "Thesis", 5)], doc_count=5)


def test_a_record_that_cannot_be_read_never_reads_as_unscoped() -> None:
    assert read_scope("{not json") == ScopeView()
    assert read_scope(json.dumps(["odd"])) == ScopeView()


def test_history_reads_an_old_record_through_the_reader(temp_db) -> None:
    from doc_assistant.conversations import get_conversation

    with session_scope() as session:
        session.add(
            AnswerRecord(
                session_id="s1",
                query="q",
                answer="a",
                retrieved_chunks_json="[]",
                retrieval_scope_json=json.dumps(
                    {"folder_id": "f1", "folder_name": "Thesis", "doc_count": 5}
                ),
            )
        )

    detail = get_conversation("s1")

    assert detail is not None
    (turn,) = detail.turns
    assert turn.retrieval_scope == ScopeView(folders=[ScopeFolder("f1", "Thesis", 5)], doc_count=5)


def test_the_words_are_pinned() -> None:
    two = ScopeView(folders=[ScopeFolder("a", "A", 9), ScopeFolder("b", "B", 7)], doc_count=14)
    assert scope_summary(two) == "Searched **A**, **B** only: 14 documents."
    assert scope_label(two) == "A, B (14 documents)"
    dropped = ScopeView(folders=[ScopeFolder("a", "A", 1), ScopeFolder("c", "C", 0)], doc_count=1)
    assert scope_summary(dropped) == (
        "Searched **A** only: 1 document. **C** held no documents and was left out."
    )
    gone = ScopeView(folders=[ScopeFolder("x", None, 0)])
    assert (
        scope_summary(gone) == "Nothing was searched. A folder that no longer exists was left out."
    )
    assert scope_label(gone) == "a folder that no longer exists (0 documents)"
    concept = ScopeView(concept=ScopeConcept("c", "synapse", False), doc_count=6)
    assert (
        scope_summary(concept)
        == "Searched the documents where **synapse** is present: 6 documents."
    )
    assert _no_sources_answer(gone, corpus_empty=False) == (
        "That folder no longer exists, so nothing was searched."
    )


# --- G6: MCP's turn ---------------------------------------------------------------------- #


def test_mcp_turns_carry_scope_and_folder_only_for_one_folder(temp_db) -> None:
    from doc_assistant import read_surface as rs

    base = datetime(2026, 11, 20, 9)
    scopes = [
        None,
        ScopeView(folders=[ScopeFolder("f1", "Thesis", 3)], doc_count=3),
        ScopeView(
            folders=[ScopeFolder("f1", "Thesis", 3), ScopeFolder("f2", "B", 2)], doc_count=4
        ),
        ScopeView(concept=ScopeConcept("c1", "synapse"), doc_count=2),
    ]
    with session_scope() as session:
        for i, scope in enumerate(scopes):
            raw = scope_dict(scope)
            session.add(
                AnswerRecord(
                    session_id="s1",
                    query=f"q{i}",
                    answer="a",
                    retrieved_chunks_json="[]",
                    retrieval_scope_json=json.dumps(raw) if raw is not None else None,
                    created_at=base + timedelta(minutes=i),
                )
            )
        session.add(
            AnswerRecord(
                session_id="s1",
                query="old",
                answer="a",
                retrieved_chunks_json="[]",
                retrieval_scope_json=json.dumps(
                    {"folder_id": "f1", "folder_name": "Thesis", "doc_count": 3}
                ),
                created_at=base + timedelta(minutes=9),
            )
        )

    record = rs.get_conversation("s1")

    assert record is not None
    got = [(t.folder, t.scope) for t in record.turns]
    assert got[0] == (None, None)
    assert got[1][0] == "Thesis" and got[1][1] is not None
    assert (got[1][1].folders, got[1][1].concept, got[1][1].documents) == (["Thesis"], None, 3)
    assert got[2][0] is None, "folder is set for exactly one folder only"
    assert got[2][1] is not None and got[2][1].folders == ["Thesis", "B"]
    assert got[3][0] is None and got[3][1] is not None and got[3][1].concept == "synapse"
    assert got[4][0] == "Thesis", "an old record keeps its folder"


# --- G4: the app's chat list ------------------------------------------------------------ #


def test_the_chat_list_has_labels_and_filters_as_the_read_surface_does(temp_db) -> None:
    from apps.api.main import create_app
    from fastapi.testclient import TestClient

    from doc_assistant import read_surface as rs

    a, b, c = _doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")
    thesis = _folder("Thesis", [a, b])
    base = datetime(2026, 11, 20, 9)
    cited = {"s1": [a], "s2": [b, c], "s3": [c], "s4": [a, c]}
    with session_scope() as session:
        for i, (sid, docs) in enumerate(cited.items()):
            session.add(
                AnswerRecord(
                    session_id=sid,
                    query=f"question {sid}",
                    answer="a",
                    retrieved_chunks_json=json.dumps([{"doc_id": d} for d in docs]),
                    created_at=base + timedelta(minutes=i),
                )
            )
    client = TestClient(create_app(controller=ChatController(rag=SpyRAG([]))))  # type: ignore[arg-type]

    every = client.get("/api/conversations").json()
    s2 = next(r for r in every if r["session_id"] == "s2")
    assert {d["document_id"] for d in s2["labels"]["documents"]} == {b, c}
    assert s2["labels"]["folders"] == [{"folder_id": thesis, "name": "Thesis"}]

    def app(**params: str) -> set[str]:
        return {r["session_id"] for r in client.get("/api/conversations", params=params).json()}

    def surface(**params: str) -> set[str]:
        return {c.conversation_id for c in rs.list_conversations(**params).conversations}

    for params in (
        {"folder_id": thesis},
        {"document_id": c},
        {"folder_id": thesis, "document_id": c},
    ):
        assert app(**params) == surface(**params), params
    assert app(folder_id=thesis) == {"s1", "s2", "s4"}
    assert app(document_id=c) == {"s2", "s3", "s4"}
    assert app(folder_id=thesis, document_id=c) == {"s2", "s4"}


# --- scale: 10,000 documents in 20 folders ---------------------------------------------- #

#: Bounds for the timed test, stated in the DEVLOG entry. Generous against the measured times so
#: a slow CI runner does not flake; a per-folder or per-document query would blow far past them.
UNION_BOUND_S = 0.5
LIST_BOUND_S = 2.0


def test_scale_union_and_filtered_list_on_10000_documents(temp_db) -> None:
    from doc_assistant.conversations import list_conversations

    rnd = random.Random(7)
    n_docs, n_folders = 10_000, 20
    doc_ids = [f"doc-{i:05d}" for i in range(n_docs)]
    folder_ids = [f"folder-{i:02d}" for i in range(n_folders)]
    with session_scope() as session:
        session.execute(
            insert(DocRow),
            [
                {
                    "id": d,
                    "filename": f"{d}.pdf",
                    "source_original": f"C:/lib/{d}.pdf",
                    "doc_hash": f"h-{d}",
                    "format": "pdf",
                }
                for d in doc_ids
            ],
        )
        session.execute(
            insert(Folder), [{"id": f, "name": f"Folder {i}"} for i, f in enumerate(folder_ids)]
        )
        # Each document in one folder, and one in ten also in the next: the union must dedupe.
        links = [
            {"document_id": d, "folder_id": folder_ids[i % n_folders]}
            for i, d in enumerate(doc_ids)
        ]
        links += [
            {"document_id": d, "folder_id": folder_ids[(i + 1) % n_folders]}
            for i, d in enumerate(doc_ids)
            if i % 10 == 0
        ]
        session.execute(insert(document_folders), links)
        base = datetime(2026, 11, 1)
        session.execute(
            insert(AnswerRecord),
            [
                {
                    "session_id": f"s{i:04d}",
                    "query": f"q{i}",
                    "answer": "a",
                    "retrieved_chunks_json": json.dumps(
                        [{"doc_id": rnd.choice(doc_ids)} for _ in range(10)]
                    ),
                    "created_at": base + timedelta(minutes=i),
                }
                for i in range(1_000)
            ],
        )

    five = folder_ids[:5]
    started = time.perf_counter()
    scope, view = _resolve_scope(five)
    union_s = time.perf_counter() - started

    started = time.perf_counter()
    listed = list_conversations(folder_id=folder_ids[3])
    list_s = time.perf_counter() - started

    expected = {f"h-{d}" for i, d in enumerate(doc_ids) if i % n_folders < 5}
    expected |= {
        f"h-{d}" for i, d in enumerate(doc_ids) if i % 10 == 0 and (i + 1) % n_folders < 5
    }
    assert scope == frozenset(expected) and view is not None and view.doc_count == len(expected)
    assert listed and all(
        any(f.folder_id == folder_ids[3] for f in s.labels.folders) for s in listed
    )
    print(
        f"\nA7b scale: union of 5/20 folders over 10,000 docs {union_s:.3f}s; "
        f"filtered list over 1,000 conversations {list_s:.3f}s"
    )
    assert union_s < UNION_BOUND_S, union_s
    assert list_s < LIST_BOUND_S, list_s
