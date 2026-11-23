"""The read surface's passages against the five evidence cases (A8', `docs/security.md` T6).

What an outside AI program is handed must stay true when the library moves under it:

1. a document whose text changed: an old passage never passes as current;
2. two documents that disagree: both are returned, never merged into one;
3. a passage with no page: the page is said to be unknown, never invented;
4. a document removed: gone from every result;
5. the same call twice: the same passages in the same order.

And the answer record's half of case 1: a past answer still names the documents it cited after
a per-part and a full re-ingest.

Real ingest and a real `RAGPipeline` over temp stores: a fake embedder (no model download), a
word-overlap reranker (no cross-encoder), no chat model. The engine's own wiring is used where
it matters: the API's re-ingest worker for case 1, `delete_document` with the engine's `forget`
for case 4.
"""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest
from doc_assistant import read_surface as rs
from doc_assistant.db.models import AnswerRecord, Base, Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest.cache import write_cache

_FAKE_DIM = 16


class _OverlapReranker:
    """Scores a pair by the share of the query's words in the passage: deterministic, offline."""

    def predict(self, pairs: list[list[str]]) -> list[float]:
        out = []
        for query, text in pairs:
            q = set(re.findall(r"\w+", query.lower()))
            t = set(re.findall(r"\w+", text.lower()))
            out.append(len(q & t) / (len(q) or 1))
        return out


@dataclass
class _Library:
    docs: Path
    cache: Path
    tmp: Path

    def write(self, name: str, text: str) -> Path:
        """A source holding ``text`` and a fresh cache of it: ingest reads the cache verbatim,
        and a re-extraction (the *text* part) reads the same text from the file."""
        src = self.docs / name
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text(text, encoding="utf-8")
        cached = ingest.get_cache_path(src)
        cached.parent.mkdir(parents=True, exist_ok=True)
        write_cache(cached, text, source=src)
        return src

    def ingest(self, *files: Path) -> None:
        stats = ingest.main(files=list(files)) if files else ingest.main()
        assert not stats.get("error"), stats

    def pipeline(self) -> Any:
        from doc_assistant.pipeline import RAGPipeline

        rag = RAGPipeline()
        rag.reranker = _OverlapReranker()  # type: ignore[assignment]
        return rag

    def document_id(self, filename: str) -> str:
        with session_scope() as session:
            got = session.execute(
                select(Document.id).where(Document.filename == filename)
            ).scalar_one()
        return str(got)


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Library]:
    docs, cache = tmp_path / "sources", tmp_path / "cache"
    chroma, pc_chroma = tmp_path / "chroma", tmp_path / "chroma_pc"
    for d in (docs, cache, chroma, pc_chroma):
        d.mkdir(parents=True, exist_ok=True)
    # One worker: extraction runs in this process, not a spawned pool (seconds per ingest here).
    monkeypatch.setenv("DOC_INGEST_WORKERS", "1")
    monkeypatch.setattr(config, "DOCS_PATH", docs)
    monkeypatch.setattr(config, "CACHE_PATH", cache)
    monkeypatch.setattr(config, "CHROMA_PATH", str(chroma))
    monkeypatch.setattr(config, "PC_CHROMA_PATH", str(pc_chroma))
    monkeypatch.setattr("doc_assistant.pipeline.CHROMA_PATH", str(chroma))
    monkeypatch.setattr("doc_assistant.pipeline.PC_CHROMA_PATH", str(pc_chroma))
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    fake = DeterministicFakeEmbedding(size=_FAKE_DIM)
    monkeypatch.setattr(ingest, "get_embeddings", lambda name=None: fake)
    monkeypatch.setattr("doc_assistant.pipeline.get_embeddings", lambda name=None: fake)
    monkeypatch.setattr("doc_assistant.pipeline.build_chat_model", lambda *_a, **_k: object())

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    yield _Library(docs=docs, cache=cache, tmp=tmp_path)
    engine.dispose()


_V1 = """<!-- page:1 -->
# Saturation in ranking

Term frequency saturation in BM25 is controlled by the parameter k1, set to 1.2 here.
"""

_V2 = """<!-- page:1 -->
# Saturation in ranking

The revised text says the saturation parameter k1 is now 2.0 for long fields.
"""

_QUESTION = "what is the saturation parameter k1"


def _texts(result: rs.SearchResult) -> list[str]:
    return [p.fenced for p in result.passages]


def _ids(result: rs.SearchResult) -> list[str | None]:
    return [p.passage_id for p in result.passages]


# --------------------------------------------------------------------------- 1. a changed text


def _rerun_through_the_api(library: _Library, rag_holder: dict[str, Any], doc_id: str) -> None:
    """Run the *text* part through the API's re-ingest worker, as the window does."""
    from apps.api.main import create_app
    from fastapi.testclient import TestClient

    from doc_assistant.chat_controller import ChatController

    def factory() -> ChatController:
        controller = ChatController.__new__(ChatController)
        controller.rag = library.pipeline()
        rag_holder["rag"] = controller.rag
        return controller

    first = factory()
    app = create_app(controller=first, controller_factory=factory)
    with TestClient(app) as client:
        started = client.post(
            "/api/library/documents/reingest", json={"document_ids": [doc_id], "parts": ["text"]}
        )
        assert started.status_code == 202, started.text
        for _ in range(200):
            status = client.get("/api/library/reingest/status").json()
            if status["state"] in ("done", "error"):
                break
            time.sleep(0.05)
        assert status["state"] == "done", status
        # The live engine after the run: the app replaces its controller, or keeps the first.
        rag_holder.setdefault("rag", first.rag)
        rag_holder["live"] = app.state.controller.rag


def test_a_changed_text_never_serves_its_old_passage(library: _Library) -> None:
    src = library.write("bm25.md", _V1)
    library.ingest(src)
    doc_id = library.document_id("bm25.md")
    rag = library.pipeline()
    before = rs.search_library(rag.retrieve_with_scores, _QUESTION)
    assert any("1.2" in t for t in _texts(before))
    old_id = before.passages[0].passage_id
    assert old_id is not None

    # The document's text changes and the person re-runs its text, as the window does.
    library.write("bm25.md", _V2)
    holder: dict[str, Any] = {}
    _rerun_through_the_api(library, holder, doc_id)
    live = holder["live"]

    after = rs.search_library(live.retrieve_with_scores, _QUESTION)
    assert after.passages, "the re-extracted text must be searchable"
    assert all("1.2" not in t for t in _texts(after)), "a superseded passage was served"
    assert any("2.0" in t for t in _texts(after)), _texts(after)

    # An id handed out before the change must not now answer with the new text as if it were
    # the passage the client was given.
    assert rs.get_passage(live.db, old_id) is None
    current = after.passages[0].passage_id
    assert current is not None and current != old_id
    now = rs.get_passage(live.db, current)
    assert now is not None and "2.0" in now.fenced


# --------------------------------------------------------------------------- 2. disagreement


def test_two_documents_that_disagree_are_both_returned(library: _Library) -> None:
    # The same file name in two folders: a Zotero library holds many `main.pdf`-like names.
    a = library.write("one/notes.md", _V1)
    b = library.write("two/notes.md", _V2)
    library.ingest(a, b)
    rag = library.pipeline()
    result = rs.search_library(rag.retrieve_with_scores, _QUESTION)
    documents = {p.document_id for p in result.passages}
    assert len(documents) == 2, [p.fenced for p in result.passages]
    texts = " ".join(_texts(result))
    assert "1.2" in texts and "2.0" in texts


# --------------------------------------------------------------------------- 3. no page


def test_a_passage_with_no_page_says_so(library: _Library) -> None:
    src = library.write("plain.md", "# Plain\n\nThe saturation parameter k1 is 1.2 here.\n")
    library.ingest(src)
    rag = library.pipeline()
    result = rs.search_library(rag.retrieve_with_scores, _QUESTION)
    assert result.passages
    assert all(p.page is None for p in result.passages)
    got = rs.get_passage(rag.db, str(result.passages[0].passage_id))
    assert got is not None and got.page is None


# --------------------------------------------------------------------------- 4. removed


def test_a_removed_document_is_gone_from_every_result(library: _Library) -> None:
    keep = library.write("keep.md", _V2)
    drop = library.write("drop.md", _V1)
    library.ingest(keep, drop)
    rag = library.pipeline()
    before = rs.search_library(rag.retrieve_with_scores, _QUESTION)
    dropped = library.document_id("drop.md")
    held = [p.passage_id for p in before.passages if p.document_id == dropped]
    assert held

    from doc_assistant.library.documents import delete_document

    assert delete_document(dropped, rag.db, forget=rag.forget_document) is not None
    after = rs.search_library(rag.retrieve_with_scores, _QUESTION)
    assert after.passages
    assert all(p.document_id != dropped for p in after.passages)
    assert all(rs.get_passage(rag.db, str(pid)) is None for pid in held)
    assert rs.get_document(dropped) is None
    listed = rs.list_documents()
    assert dropped not in {d.document_id for d in listed.documents}


# --------------------------------------------------------------------------- 5. twice


def test_the_same_call_twice_gives_the_same_passages_in_order(library: _Library) -> None:
    files = [
        library.write(f"d{i}.md", f"# D{i}\n\nThe saturation parameter k1 appears in d{i}.\n")
        for i in range(6)
    ]
    library.ingest(*files)
    rag = library.pipeline()
    first = rs.search_library(rag.retrieve_with_scores, _QUESTION, top_k=10)
    second = rs.search_library(rag.retrieve_with_scores, _QUESTION, top_k=10)
    assert first.passages
    assert _ids(first) == _ids(second)
    assert [p.score for p in first.passages] == [p.score for p in second.passages]
    # A fresh engine over the same stores, as after a relaunch.
    third = rs.search_library(library.pipeline().retrieve_with_scores, _QUESTION, top_k=10)
    assert _ids(third) == _ids(first)


# --------------------------------------------------------------------------- the records


def _answer_citing(doc_id: str, filename: str, excerpt: str) -> str:
    import json

    with session_scope() as session:
        rec = AnswerRecord(
            query=_QUESTION,
            answer="k1 controls saturation [1].",
            session_id="conv-1",
            retrieved_chunks_json=json.dumps(
                [{"doc_id": doc_id, "filename": filename, "chunk_excerpt": excerpt}]
            ),
        )
        session.add(rec)
    return "conv-1"


def test_a_past_answer_still_names_its_document_after_a_re_ingest(library: _Library) -> None:
    src = library.write("bm25.md", _V1)
    library.ingest(src)
    doc_id = library.document_id("bm25.md")
    excerpt = "Term frequency saturation in BM25 is controlled by the parameter k1"
    conv = _answer_citing(doc_id, "bm25.md", excerpt)
    first = rs.get_conversation(conv)
    assert first is not None
    assert first.turns[0].cited[0].page == 1

    # A per-part text re-run that changes the text: the id is kept (ADR-047), the excerpt is the
    # record's own copy, and the page it can no longer place is said to be unknown.
    library.write("bm25.md", _V2)
    from doc_assistant.reingest import rerun

    rerun([doc_id], ["text"])
    assert library.document_id("bm25.md") == doc_id
    after = rs.get_conversation(conv)
    assert after is not None
    cited = after.turns[0].cited[0]
    assert cited.document_id == doc_id
    assert cited.title != "unknown"
    assert cited.excerpt == excerpt
    assert cited.page is None

    # A full re-ingest of the whole folder keeps the same id again.
    library.write("bm25.md", _V1)
    library.ingest()
    assert library.document_id("bm25.md") == doc_id
    again = rs.get_conversation(conv)
    assert again is not None
    assert again.turns[0].cited[0].document_id == doc_id
    assert again.turns[0].cited[0].page == 1
    assert {d.document_id for d in again.labels.documents} == {doc_id}
