"""Integration tests for the Library browser endpoints (feature-library-browser.md, L1).

Seed ``Document`` rows into a temp DB and give the controller a fake Chroma handle whose ``.get``
returns canned child chunks per ``document_id``. The read path never generates or writes — a
minimal fake suffices and a write-trap fake proves it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope


class FakeChroma:
    """A read-only stand-in for the live Chroma handle: ``.get`` returns canned chunks per doc."""

    def __init__(self, by_doc: dict[str, list[dict[str, Any]]]) -> None:
        self._by_doc = by_doc

    def get(self, *, where: dict[str, Any], include: list[str]) -> dict[str, Any]:
        doc_id = where["document_id"]
        chunks = self._by_doc.get(doc_id, [])
        return {
            "documents": [c["text"] for c in chunks],
            "metadatas": [{k: v for k, v in c.items() if k != "text"} for c in chunks],
        }


class FakeRag:
    def __init__(self, db: FakeChroma) -> None:
        self.db = db


class FakeController:
    def __init__(self, by_doc: dict[str, list[dict[str, Any]]]) -> None:
        self.rag = FakeRag(FakeChroma(by_doc))

    def chunk_count(self) -> int:
        return 0


@pytest.fixture
def temp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.db.models import Base

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", future=True)
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    yield
    engine.dispose()


def _seed_doc(
    filename: str,
    *,
    archived: bool = False,
    chunk_count: int | None = None,
    title: str | None = None,
) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"hash-{filename}",
            format="pdf",
            is_archived=archived,
            chunk_count=chunk_count,
            title=title,
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _child(p: int, c: int, text: str, parent_text: str = "PARENT") -> dict[str, Any]:
    return {"parent_index": p, "child_index": c, "parent_text": parent_text, "text": text}


def test_library_endpoints(temp_db: None) -> None:
    id_a = _seed_doc("a.pdf", chunk_count=2)
    id_b = _seed_doc("b.pdf", chunk_count=0)  # a known doc with no stored chunks
    _seed_doc("z.pdf", archived=True)  # archived -> excluded from the list

    by_doc = {
        id_a: [
            _child(0, 1, "second child"),
            _child(0, 0, "first child"),
        ],
    }
    client = TestClient(create_app(controller=FakeController(by_doc)))  # type: ignore[arg-type]

    # --- list: archived excluded, ordered by filename ---
    docs = client.get("/api/library/documents").json()
    assert [d["filename"] for d in docs] == ["a.pdf", "b.pdf"]

    # --- detail: chunks grouped into ordered parent blocks ---
    detail = client.get(f"/api/library/documents/{id_a}").json()
    assert detail["filename"] == "a.pdf"
    assert detail["child_count"] == 2
    assert len(detail["parents"]) == 1
    block = detail["parents"][0]
    assert block["parent_index"] == 0
    assert block["parent_text"] == "PARENT"
    assert [c["text"] for c in block["children"]] == [
        "first child",
        "second child",
    ]  # by child_index

    # --- a known doc with zero stored chunks: empty parents, NOT a 404 ---
    empty = client.get(f"/api/library/documents/{id_b}")
    assert empty.status_code == 200
    assert empty.json()["parents"] == []
    assert empty.json()["child_count"] == 0

    # --- unknown doc -> 404 ---
    assert client.get("/api/library/documents/does-not-exist").status_code == 404

    # --- guard: the reads mutated nothing ---
    with session_scope() as session:
        assert session.query(Document).count() == 3


# ============================================================
# ADR-027 D1 (E4) — GET /api/library/documents/{id}/connections
# ============================================================


def test_document_connections_route_returns_bundle(temp_db) -> None:
    from doc_assistant.db.models import Citation, DocSimilarity
    from doc_assistant.embeddings import get_active_model_name

    a = _seed_doc("subject.pdf")
    b = _seed_doc("neighbour.pdf")
    c = _seed_doc("citer.pdf")
    with session_scope() as session:
        session.add(
            DocSimilarity(
                source_document_id=a,
                target_document_id=b,
                embedding_model=get_active_model_name(),  # what the route scopes to
                score=0.93,
            )
        )
        session.add(Citation(source_document_id=a, target_document_id=b, target_title="Nb"))
        session.add(Citation(source_document_id=c, target_document_id=a, target_title="Subj"))
        session.add(Citation(source_document_id=a, target_title="External only", target_year=2020))

    client = TestClient(create_app(controller=FakeController({})))
    r = client.get(f"/api/library/documents/{a}/connections")
    assert r.status_code == 200
    body = r.json()
    assert [x["document_id"] for x in body["related"]] == [b]
    assert body["related"][0]["filename"] == "neighbour.pdf"
    assert [x["document_id"] for x in body["cited_by"]] == [c]
    assert body["cited_by"][0]["n_citations"] == 1
    # The neighbourhood, not the bibliography: what `a` cites is /references (2026-08-10).
    assert "cites" not in body
    assert "external_refs" not in body


def test_document_connections_route_404_for_unknown_doc(temp_db) -> None:
    client = TestClient(create_app(controller=FakeController({})))
    assert client.get("/api/library/documents/nope/connections").status_code == 404


def test_document_connections_route_empty_bundle_for_bare_doc(temp_db) -> None:
    # A known doc with no sidecar rows: 200 + all-empty lists (honest degrade), never a 404/500.
    a = _seed_doc("bare.pdf")
    client = TestClient(create_app(controller=FakeController({})))
    r = client.get(f"/api/library/documents/{a}/connections")
    assert r.status_code == 200
    body = r.json()
    assert body["related"] == [] and body["cited_by"] == []


# ============================================================
# GET /api/library/documents/{id}/references — the References block
# ============================================================


def test_document_references_route_lists_the_whole_bibliography(temp_db) -> None:
    # Every extracted reference, resolved or not, in one list — and the resolved one carries
    # the document_id that makes it a link.
    from doc_assistant.db.models import Citation

    a = _seed_doc("subject.pdf")
    # The title matters: a stored resolution is re-checked against the document it names
    # before it is presented as a link (library.citations.resolution_is_credible).
    owned = _seed_doc("owned.pdf", title="Owned paper")
    with session_scope() as session:
        session.add(
            Citation(
                source_document_id=a,
                target_document_id=owned,
                target_title="Owned paper",
                target_year=2021,
            )
        )
        session.add(Citation(source_document_id=a, target_title="Elsewhere", target_year=2020))
        # No year at all — sorts last (nulls_last), and still appears: an unparseable line is
        # a reference the paper makes.
        session.add(Citation(source_document_id=a, raw_citation_text="[3] unparseable"))

    client = TestClient(create_app(controller=FakeController({})))
    r = client.get(f"/api/library/documents/{a}/references")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and body["shown"] == 3 and body["in_library"] == 1
    assert [x["document_id"] for x in body["references"]] == [owned, None, None]
    assert body["references"][0]["filename"] == "owned.pdf"


def test_document_references_route_404_for_unknown_doc(temp_db) -> None:
    client = TestClient(create_app(controller=FakeController({})))
    assert client.get("/api/library/documents/nope/references").status_code == 404


def test_document_references_route_empty_list_for_bare_doc(temp_db) -> None:
    # A document whose references were never extracted: 200 + an empty list, never a 404.
    a = _seed_doc("bare.pdf")
    client = TestClient(create_app(controller=FakeController({})))
    r = client.get(f"/api/library/documents/{a}/references")
    assert r.status_code == 200
    assert r.json() == {"references": [], "total": 0, "in_library": 0, "shown": 0}
