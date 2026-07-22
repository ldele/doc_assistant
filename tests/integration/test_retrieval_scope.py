"""Integration tests for folder-scoped retrieval (ADR-025 F2, feature-corpus-folders-scope.md).

Covers the honesty contract end to end — resolution (S2), the never-fall-back-to-unscoped rule
(S3), the untouched unscoped path (S4), the persisted provenance scope (S7), the structured
answer chip on **both** synthesis paths (S8), and the API round-trip.

The pipeline-level arm filtering lives in ``tests/unit/test_pipeline_scope.py``; here the
pipeline is a fake that records the ``scope`` it was handed, so these tests are about the turn,
the record, and the wire. No models, no LLM, no network.
"""

from __future__ import annotations

import json
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.documents import Document
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from doc_assistant import chat_controller
from doc_assistant.chat_controller import ChatController, RagOverrides, Result, Session
from doc_assistant.db import session as session_mod
from doc_assistant.db.models import AnswerRecord, Base
from doc_assistant.db.models import Document as DocRow
from doc_assistant.db.session import session_scope
from doc_assistant.library import add_documents_to_folder, create_folder, delete_folder


@pytest.fixture(autouse=True)
def _no_paid_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hard stop on the reviewer. A single-source turn trips a confidence signal, which calls the
    reviewer — a real, billed API request. These tests are about scoping, not the reviewer, and a
    test that *can* reach a paid provider is a defect regardless of whether it fires today."""
    monkeypatch.setattr("doc_assistant.llm.reviewer_available", lambda provider=None: False)


@pytest.fixture(autouse=True)
def _isolate_user_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """E3: _resolve_turn_knobs reads the persisted answer-layer default per turn — isolate it
    from the dev box's real settings.json so scoping tests can't depend on local state."""
    monkeypatch.setattr(chat_controller.app_settings, "SETTINGS_PATH", tmp_path / "settings.json")


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'scope.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield tmp_path
    engine.dispose()


class FakeRAG:
    """Records the scope handed to retrieval, so a test can assert what the arms would see."""

    def __init__(self, tokens: list[str] | None = None) -> None:
        self._tokens = tokens or ["Answer [1]."]
        self.llm = types.SimpleNamespace(model="fake-model")
        self.provider = "anthropic"
        self.model = "claude-haiku-4-5-20251001"
        self.scope_calls: list[frozenset[str] | None] = []

    def chunk_count(self) -> int:
        return 3

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
        # An empty scope means the real pipeline retrieves nothing — mirror that here, or the
        # test would silently exercise a friendlier world than production.
        if scope is not None and not scope:
            return []
        docs = [
            (
                Document(
                    page_content="Neurons meet at synapses.",
                    metadata={
                        "document_id": "d1",
                        "chunk_index": 0,
                        "filename": "a.pdf",
                        "page": 1,
                        "doc_hash": "h-a.pdf",
                    },
                ),
                0.91,
            )
        ]
        return [(d, s) for d, s in docs if scope is None or d.metadata["doc_hash"] in scope]

    def stream_answer(
        self, question: str, docs: list[Document], counter: object = None, llm: object = None
    ) -> Iterator[str]:
        yield from self._tokens


def _seed_doc(filename: str) -> str:
    with session_scope() as session:
        doc = DocRow(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"h-{filename}",
            format="pdf",
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _folder_with(*filenames: str) -> str:
    folder = create_folder("Demo corpus")
    add_documents_to_folder(folder.id, [_seed_doc(f) for f in filenames])
    return folder.id


def _turn(
    controller: ChatController,
    *,
    scope_folder_id: str | None = None,
    overrides: RagOverrides | None = None,
) -> object:
    events = list(
        controller.handle_message(
            Session(),
            "how do neurons connect?",
            overrides=overrides,
            scope_folder_id=scope_folder_id,
        )
    )
    return next(e.result for e in events if isinstance(e, Result))


def _scope_json() -> list[str | None]:
    """The persisted ``retrieval_scope_json`` of every record, read inside the session."""
    with session_scope() as session:
        return [r.retrieval_scope_json for r in session.execute(select(AnswerRecord)).scalars()]


# --- S2 / S7 / S8: a scoped turn resolves, records, and announces ------------------------ #


def test_scoped_turn_resolves_membership_records_and_announces(monkeypatch, temp_db) -> None:
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    folder_id = _folder_with("a.pdf", "b.pdf")
    rag = FakeRAG()
    controller = ChatController(rag=rag)

    result = _turn(controller, scope_folder_id=folder_id)

    # S2 — the backend resolved the folder to its members' doc_hashes.
    assert rag.scope_calls == [frozenset({"h-a.pdf", "h-b.pdf"})]
    # S8 — the structured chip the renderer shows, on the ai path.
    assert result.scope is not None
    assert (result.scope.folder_id, result.scope.folder_name, result.scope.doc_count) == (
        folder_id,
        "Demo corpus",
        2,
    )
    # ...and the scope is stated in the provenance card, not only in a structured field.
    assert "Retrieval scope" in result.provenance_card_md
    assert "Demo corpus" in result.provenance_card_md
    # S7 — persisted, so the answer can never later be mistaken for a whole-library one.
    (raw,) = _scope_json()
    assert json.loads(raw or "null") == {
        "folder_id": folder_id,
        "folder_name": "Demo corpus",
        "doc_count": 2,
    }


def test_human_mode_scoped_turn_also_announces_and_records(monkeypatch, temp_db) -> None:
    """The second result builder — a chip on only one path is a lie on the other (S8)."""
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    folder_id = _folder_with("a.pdf")
    controller = ChatController(rag=FakeRAG())

    result = _turn(
        controller, scope_folder_id=folder_id, overrides=RagOverrides(synthesis_mode="human")
    )

    assert result.mode == "human"
    assert result.scope is not None and result.scope.doc_count == 1
    # Human mode renders no provenance card, so the note has to ride the answer itself.
    assert "Retrieval scope" in result.answer
    (raw,) = _scope_json()
    assert json.loads(raw or "null")["folder_name"] == "Demo corpus"


# --- S3: never fall back to unscoped ----------------------------------------------------- #


def test_deleted_folder_searches_nothing_rather_than_everything(monkeypatch, temp_db) -> None:
    """The core integrity assertion: an unhonourable scope must not become 'searched all'."""
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    folder_id = _folder_with("a.pdf")
    delete_folder(folder_id)
    rag = FakeRAG()
    controller = ChatController(rag=rag)

    result = _turn(controller, scope_folder_id=folder_id)

    assert rag.scope_calls == [frozenset()]  # empty, NOT None
    assert result.sources == []
    # The chip still fires, and says the folder is gone rather than naming a folder.
    assert result.scope is not None
    assert result.scope.folder_name is None
    assert result.scope.doc_count == 0
    assert json.loads(_scope_json()[0] or "null")["folder_name"] is None


def test_empty_folder_searches_nothing(monkeypatch, temp_db) -> None:
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    folder = create_folder("Empty")
    _seed_doc("a.pdf")  # exists in the library, but not in this folder
    rag = FakeRAG()
    controller = ChatController(rag=rag)

    result = _turn(controller, scope_folder_id=folder.id)

    assert rag.scope_calls == [frozenset()]
    assert result.sources == []
    assert result.scope is not None and result.scope.doc_count == 0


def test_unknown_folder_id_searches_nothing(monkeypatch, temp_db) -> None:
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    _seed_doc("a.pdf")
    rag = FakeRAG()
    controller = ChatController(rag=rag)

    result = _turn(controller, scope_folder_id="never-existed")

    assert rag.scope_calls == [frozenset()]
    assert result.sources == []


# --- S4: the unscoped turn is untouched -------------------------------------------------- #


def test_unscoped_turn_passes_none_records_null_and_says_nothing(monkeypatch, temp_db) -> None:
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    _folder_with("a.pdf")  # a folder exists; not selecting it must change nothing
    rag = FakeRAG()
    controller = ChatController(rag=rag)

    result = _turn(controller)

    assert rag.scope_calls == [None]
    assert result.scope is None
    assert "Retrieval scope" not in result.provenance_card_md
    assert _scope_json() == [None]


def test_scope_does_not_leak_between_turns(monkeypatch, temp_db) -> None:
    """The ADR-010 isolation obligation, applied to the scope: a scoped turn must not make the
    next turn scoped. Nothing is stored on the session, so this proves the wiring, not a reset."""
    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    folder_id = _folder_with("a.pdf")
    rag = FakeRAG()
    controller = ChatController(rag=rag)
    session = Session()

    list(controller.handle_message(session, "q1", scope_folder_id=folder_id))
    list(controller.handle_message(session, "q2"))

    assert rag.scope_calls == [frozenset({"h-a.pdf"}), None]


# --- API round-trip ---------------------------------------------------------------------- #


def test_chat_route_forwards_the_scope_and_returns_it_on_the_wire(monkeypatch, temp_db) -> None:
    from apps.api.main import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    folder_id = _folder_with("a.pdf")
    controller = ChatController(rag=FakeRAG())
    client = TestClient(create_app(controller=controller))  # type: ignore[arg-type]

    body = {"text": "q", "session_id": "s1", "scope_folder_id": folder_id}
    with client.stream("POST", "/api/chat", json=body) as r:
        assert r.status_code == 200
        payload = next(
            json.loads(line[len("data: ") :])
            for block in r.iter_lines()
            for line in [block]
            if line.startswith("data: ") and '"scope"' in line
        )
    assert payload["scope"] == {
        "folder_id": folder_id,
        "folder_name": "Demo corpus",
        "doc_count": 1,
    }


def test_chat_route_defaults_to_unscoped(monkeypatch, temp_db) -> None:
    from apps.api.main import create_app
    from fastapi.testclient import TestClient

    monkeypatch.setattr(chat_controller, "is_library_query", lambda t: False)
    controller = ChatController(rag=FakeRAG())
    client = TestClient(create_app(controller=controller))  # type: ignore[arg-type]

    with client.stream("POST", "/api/chat", json={"text": "q", "session_id": "s1"}) as r:
        assert r.status_code == 200
        list(r.iter_lines())
    assert _scope_json() == [None]


# --- A/B compare (the user's 2026-07-20 decision: scope both sides) ----------------------- #


def test_compare_scopes_both_sides_and_labels_the_card(monkeypatch, temp_db) -> None:
    """An unscoped diff shown while a folder scope is active would describe retrieval the next
    answer will not perform."""
    folder_id = _folder_with("a.pdf")
    rag = FakeRAG()
    controller = ChatController(rag=rag)

    result = controller.compare_retrieval("q", RagOverrides(top_k=3), folder_id)

    assert rag.scope_calls == [frozenset({"h-a.pdf"}), frozenset({"h-a.pdf"})]  # A and B
    assert result.scope_label == "Demo corpus (1 document)"


def test_compare_without_a_scope_is_unchanged(monkeypatch, temp_db) -> None:
    rag = FakeRAG()
    result = ChatController(rag=rag).compare_retrieval("q", RagOverrides())

    assert rag.scope_calls == [None, None]
    assert result.scope_label is None


# --- KI-23: the API migrates the live schema on startup ------------------------------------ #


def test_api_startup_applies_pending_additive_columns(tmp_path: Path, monkeypatch) -> None:
    """KI-23 (filed as KI-20) — before this, additive columns only ever landed via `ingest`, so
    a user who pulled an update and just chatted kept a stale schema. F2 put a column on the
    answer path, where that breaks every turn."""
    from apps.api.main import create_app
    from fastapi.testclient import TestClient
    from sqlalchemy import text as sql_text

    db = tmp_path / "stale.db"
    engine = create_engine(f"sqlite:///{db}", future=True)
    Base.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(sql_text("ALTER TABLE answer_records DROP COLUMN retrieval_scope_json"))
        cols = {r[1] for r in conn.execute(sql_text("PRAGMA table_info(answer_records)"))}
    assert "retrieval_scope_json" not in cols  # a genuinely stale schema
    engine.dispose()

    monkeypatch.setattr("doc_assistant.config.SQLITE_PATH", str(db))
    monkeypatch.setattr("doc_assistant.db.migrations.SQLITE_PATH", str(db))
    fresh = create_engine(f"sqlite:///{db}", future=True)
    monkeypatch.setattr(session_mod, "_engine", fresh)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=fresh, autoflush=False, autocommit=False, future=True),
    )

    # Entering the TestClient context runs the lifespan.
    with TestClient(create_app(controller=ChatController(rag=FakeRAG()))) as client:  # type: ignore[arg-type]
        client.get("/api/health")

    with fresh.begin() as conn:
        cols = {r[1] for r in conn.execute(sql_text("PRAGMA table_info(answer_records)"))}
    assert "retrieval_scope_json" in cols
    fresh.dispose()


# --- migration --------------------------------------------------------------------------- #


def test_scope_column_lands_on_a_pre_existing_answer_records_table(tmp_path: Path) -> None:
    """The additive-column path: an old DB gains the column, and its existing rows read back as
    unscoped rather than failing."""
    from sqlalchemy import text

    from doc_assistant.db.migrations import _apply_additive_columns

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE answer_records (id VARCHAR PRIMARY KEY, query TEXT, answer TEXT, "
                "retrieved_chunks_json TEXT, created_at DATETIME)"
            )
        )
        conn.execute(
            text("INSERT INTO answer_records (id, query, answer) VALUES ('r1', 'q', 'a')")
        )

    _apply_additive_columns(engine)
    _apply_additive_columns(engine)  # idempotent

    with engine.begin() as conn:
        value = conn.execute(text("SELECT retrieval_scope_json FROM answer_records")).scalar()
    assert value is None
    engine.dispose()
