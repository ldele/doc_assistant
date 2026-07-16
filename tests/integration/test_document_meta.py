"""Integration tests for document metadata overrides + reveal (ADR-013, library.py + routes).

Seeds ``Document`` rows into a temp DB. Covers: override upsert with dedup-against-default, blank
reverts, effective = override ?? auto in ``list_documents``, reset, source-path resolution +
reveal, and the PATCH / reset / reveal API routes (200 + 404). Never touches the real DB or a live
file manager (reveal is monkeypatched).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient

from doc_assistant.db.models import Document, DocumentMeta
from doc_assistant.db.session import session_scope


class _FakeController:
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
    title: str | None = None,
    authors: str | None = None,
    year: int | None = None,
    source_original: str = "/tmp/nope.pdf",
) -> str:
    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=source_original,
            doc_hash=f"hash-{filename}",
            format="pdf",
            title=title,
            authors=authors,
            year=year,
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _summary(doc_id: str):
    from doc_assistant.library import list_documents

    return next(s for s in list_documents() if s.id == doc_id)


# --- override logic -------------------------------------------------------------------------- #


def test_set_meta_stores_override_only_when_it_differs(temp_db: None) -> None:
    from doc_assistant.library import set_document_meta

    doc_id = _seed_doc("a.pdf", title="Auto Title", authors="Auto Author", year=2020)

    # Re-saving values equal to the auto defaults creates NO override row (not customized).
    set_document_meta(doc_id, title="Auto Title", authors="Auto Author", year=2020)
    s = _summary(doc_id)
    assert not s.customized
    assert (s.title, s.authors, s.year) == ("Auto Title", "Auto Author", 2020)

    # Changing the title stores just that override; effective picks it up.
    set_document_meta(doc_id, title="Fixed Title", authors="Auto Author", year=2020)
    s = _summary(doc_id)
    assert s.customized
    assert s.title == "Fixed Title"
    assert s.authors == "Auto Author"  # unchanged field stays the auto value


def test_blank_string_reverts_a_field(temp_db: None) -> None:
    from doc_assistant.library import set_document_meta

    doc_id = _seed_doc("a.pdf", title="Auto Title")
    set_document_meta(doc_id, title="Custom")
    assert _summary(doc_id).title == "Custom"
    # A blank title clears the override -> back to the auto default.
    set_document_meta(doc_id, title="   ")
    s = _summary(doc_id)
    assert s.title == "Auto Title"
    assert not s.customized


def test_year_override_and_none_leaves_unchanged(temp_db: None) -> None:
    from doc_assistant.library import set_document_meta

    doc_id = _seed_doc("a.pdf", title="T", year=2020)
    set_document_meta(doc_id, title="T", year=1999)
    assert _summary(doc_id).year == 1999


def test_clear_document_meta_resets_all(temp_db: None) -> None:
    from doc_assistant.library import clear_document_meta, set_document_meta

    doc_id = _seed_doc("a.pdf", title="Auto", authors="Auto A", year=2020)
    set_document_meta(doc_id, title="Custom", authors="Custom A", year=1999)
    assert _summary(doc_id).customized
    clear_document_meta(doc_id)
    s = _summary(doc_id)
    assert not s.customized
    assert (s.title, s.authors, s.year) == ("Auto", "Auto A", 2020)
    # the row is gone
    with session_scope() as session:
        assert session.get(DocumentMeta, doc_id) is None


# --- source resolution + reveal -------------------------------------------------------------- #


def test_resolve_source_path_prefers_existing_then_none(tmp_path: Path) -> None:
    from doc_assistant.library import resolve_source_path

    real = tmp_path / "paper.pdf"
    real.write_text("x")
    assert resolve_source_path(str(real), "paper.pdf") == real
    assert resolve_source_path(str(tmp_path / "gone.pdf"), "also-gone.pdf") is None


def test_reveal_document_source(
    temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import doc_assistant.library as lib

    real = tmp_path / "paper.pdf"
    real.write_text("x")
    doc_id = _seed_doc("paper.pdf", source_original=str(real))
    calls: list[Path] = []
    monkeypatch.setattr(lib, "_reveal_in_file_manager", lambda p: calls.append(p))

    assert lib.reveal_document_source(doc_id) is True
    assert calls == [real]  # revealed the resolved path
    # a doc whose source file is missing -> False, no reveal
    missing_id = _seed_doc("gone.pdf", source_original=str(tmp_path / "missing.pdf"))
    assert lib.reveal_document_source(missing_id) is False


# --- API routes ------------------------------------------------------------------------------ #


def _client() -> TestClient:
    return TestClient(create_app(controller=_FakeController()))  # type: ignore[arg-type]


def test_patch_and_reset_routes(temp_db: None) -> None:
    doc_id = _seed_doc("a.pdf", title="Auto")
    client = _client()

    r = client.patch(f"/api/library/documents/{doc_id}", json={"title": "Edited"})
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert _summary(doc_id).title == "Edited"

    r = client.post(f"/api/library/documents/{doc_id}/reset-metadata")
    assert r.status_code == 200
    assert _summary(doc_id).title == "Auto"

    assert client.patch("/api/library/documents/nope", json={"title": "x"}).status_code == 404
    assert client.post("/api/library/documents/nope/reset-metadata").status_code == 404


def test_reveal_route(temp_db: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import doc_assistant.library as lib

    real = tmp_path / "paper.pdf"
    real.write_text("x")
    doc_id = _seed_doc("paper.pdf", source_original=str(real))
    monkeypatch.setattr(lib, "_reveal_in_file_manager", lambda p: None)
    client = _client()

    assert client.post(f"/api/library/documents/{doc_id}/reveal").status_code == 200
    # unknown doc / missing file -> 404
    assert client.post("/api/library/documents/nope/reveal").status_code == 404
    missing_id = _seed_doc("gone.pdf", source_original=str(tmp_path / "missing.pdf"))
    assert client.post(f"/api/library/documents/{missing_id}/reveal").status_code == 404


def test_list_documents_carries_year_and_customized(temp_db: None) -> None:
    from doc_assistant.library import set_document_meta

    doc_id = _seed_doc("a.pdf", title="Auto", year=2020)
    s = _summary(doc_id)
    assert s.year == 2020 and s.customized is False
    set_document_meta(doc_id, title="Edited")
    assert _summary(doc_id).customized is True
