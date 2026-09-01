"""The source-viewer routes (ROADMAP 18, ADR-050) — over the wire, against a real PDF.

The unit suite covers the page arithmetic with a fake store. What only an integration test can
show is the part the pane actually depends on: that `/page/{n}` returns PNG **bytes** for a real
file, that two different pages are two different images, and that every failure arrives as a 404
whose detail is a sentence rather than a broken image.

The availability split is the other thing asserted here, because it is the row's own requirement:
an unknown *document* is a 404, while a document whose *file* has moved is a 200 that says so.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


class _FakeChroma:
    """Answers the one shape `_chunk_metadata` asks, over rows a test puts in.

    Rows rather than a stubbed `locate_chunk`: the route resolves `locate_chunk` through the
    **package** re-export, so patching the owning module misses it entirely (the trap in
    `src/doc_assistant/CLAUDE.md`). Feeding the real function real metadata avoids the question.
    """

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def get(self, **kwargs: Any) -> dict[str, Any]:
        where = kwargs.get("where") or {}
        clauses = where.get("$and")
        if not clauses:
            return {"metadatas": []}
        wanted = {k: v for clause in clauses for k, v in clause.items()}
        hits = [r for r in self.rows if all(r.get(k) == v for k, v in wanted.items())]
        return {"metadatas": hits[: kwargs.get("limit", 1)]}


class _FakeRag:
    def __init__(self) -> None:
        self.db = _FakeChroma()


class FakeController:
    """The minimum surface `create_app` needs; the viewer touches none of it."""

    def __init__(self) -> None:
        self.chunk_count = 0
        self.rag = _FakeRag()

    def corpus_stats(self) -> dict[str, int]:
        return {"chunk_count": 0}


@pytest.fixture
def temp_database(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_connection, connection_record):
        cur = dbapi_connection.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    from doc_assistant.db.models import LIBRARY_ROOT_ID, Base, SourceRoot

    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine, future=True)() as s:
        s.add(SourceRoot(id=LIBRARY_ROOT_ID, path=str(Path(path).parent), kind="library"))
        s.commit()
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


def _make_pdf(path: Path, pages: int = 3) -> Path:
    """A real multi-page PDF — the viewer renders bytes, so a fake one would prove nothing."""
    import pymupdf

    doc = pymupdf.open()  # type: ignore[no-untyped-call]
    for n in range(pages):
        page = doc.new_page()
        # Distinct content per page: the test asserts two pages render *differently*.
        page.insert_text((72, 144 + 40 * n), f"Page {n + 1} of this document", fontsize=28)
    doc.save(str(path))
    doc.close()
    return path


def _add_document(source: Path, *, fmt: str = "pdf", pages: int | None = 3) -> str:
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    doc_id = "doc-under-test"
    with session_scope() as session:
        session.add(
            Document(
                id=doc_id,
                filename=source.name,
                source_original=str(source),
                source_cache="",
                doc_hash="hash-under-test",
                format=fmt,
                page_count=pages,
            )
        )
    return doc_id


@pytest.fixture
def client(temp_database, tmp_path, monkeypatch):
    monkeypatch.setenv("DOC_SOURCE_DIR", str(tmp_path))
    return TestClient(create_app(controller=FakeController()))


def _seed_chunk(client: TestClient, **row: Any) -> None:
    """Put one chunk in the fake store the app was built with."""
    client.app.state.controller.rag.db.rows.append(row)


# The shape the extractor writes: a marker, then that page's text.
_MARKED_CACHE = "\n<!-- page:1 -->\nFirst page.\n\n<!-- page:2 -->\nSecond page body text.\n"


# --- the header (D3/D4) ------------------------------------------------------------------------ #


def test_a_reachable_pdf_reports_itself_renderable(client, tmp_path):
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    r = client.get(f"/api/library/documents/{doc_id}/source")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["pageable"] is True
    assert body["page_count"] == 3
    assert body["reason"] is None


def test_an_unknown_document_is_a_404(client):
    assert client.get("/api/library/documents/nope/source").status_code == 404


def test_a_document_whose_file_moved_is_a_200_that_says_so(client, tmp_path):
    """The row's own requirement: a missing file degrades to a sentence, not a broken pane."""
    pdf = _make_pdf(tmp_path / "paper.pdf")
    doc_id = _add_document(pdf)
    pdf.unlink()

    r = client.get(f"/api/library/documents/{doc_id}/source")
    assert r.status_code == 200, "a missing file is not a missing document"
    body = r.json()
    assert body["available"] is False
    assert body["reason"] and "paper.pdf" in body["reason"], "the reason must name the path"


def test_a_format_without_pages_is_not_pageable(client, tmp_path):
    """ADR-050 D3 — an EPUB is a document the pane shows as text, not a failure."""
    book = tmp_path / "book.epub"
    book.write_bytes(b"not really an epub, and never opened")
    doc_id = _add_document(book, fmt="epub", pages=None)

    body = client.get(f"/api/library/documents/{doc_id}/source").json()
    assert body["available"] is True, "the file is right there"
    assert body["pageable"] is False, "but it has no pages"


# --- the pages themselves (D1) ----------------------------------------------------------------- #


def test_a_page_comes_back_as_png_bytes(client, tmp_path):
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    r = client.get(f"/api/library/documents/{doc_id}/page/1")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert r.content[:4] == b"\x89PNG", "the pane sets this as an <img src>"
    assert len(r.content) > 1000


def test_two_pages_are_two_different_images(client, tmp_path):
    """Guards the off-by-one: `page` is 1-based, and page 2 is not page 1."""
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    first = client.get(f"/api/library/documents/{doc_id}/page/1").content
    second = client.get(f"/api/library/documents/{doc_id}/page/2").content
    assert first != second


@pytest.mark.parametrize("page", [0, -1, 4, 999])
def test_a_page_outside_the_document_is_a_404_naming_the_range(client, tmp_path, page):
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    r = client.get(f"/api/library/documents/{doc_id}/page/{page}")
    assert r.status_code == 404
    assert "1-3" in r.json()["detail"], "the refusal says what the document actually has"


def test_rendering_a_non_pdf_is_refused_rather_than_attempted(client, tmp_path):
    book = tmp_path / "book.epub"
    book.write_bytes(b"not really an epub")
    doc_id = _add_document(book, fmt="epub", pages=None)

    r = client.get(f"/api/library/documents/{doc_id}/page/1")
    assert r.status_code == 404
    assert "EPUB format has no pages" in r.json()["detail"]


def test_rendering_a_moved_file_reports_the_move(client, tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf")
    doc_id = _add_document(pdf)
    pdf.unlink()

    r = client.get(f"/api/library/documents/{doc_id}/page/1")
    assert r.status_code == 404
    assert "paper.pdf" in r.json()["detail"]


def test_rendering_an_unknown_document_is_a_404(client):
    assert client.get("/api/library/documents/nope/page/1").status_code == 404


# --- the citation entry point (D2) ------------------------------------------------------------- #


def test_an_unknown_chunk_key_is_a_404(client):
    """The fake store knows nothing, so every key is unknown here."""
    r = client.get("/api/library/chunk-page", params={"key": "doc-1:p3"})
    assert r.status_code == 404


def test_a_located_chunk_returns_its_document_and_page(client, tmp_path):
    """The shape the chat card depends on: a document id it never had, plus a page.

    Goes the whole way — cache markers on disk, offset in the metadata, page derived.
    """
    cache = tmp_path / "doc.md"
    cache.write_text(_MARKED_CACHE, encoding="utf-8")
    _seed_chunk(
        client,
        document_id="doc-7",
        parent_index=3,
        source_cache=str(cache),
        parent_char_start=str(_MARKED_CACHE.index("Second page body")),
    )
    r = client.get("/api/library/chunk-page", params={"key": "doc-7:p3"})
    assert r.status_code == 200
    assert r.json() == {"document_id": "doc-7", "page": 2}


def test_a_cited_figure_locates_by_its_stored_page(client):
    """The case the chat card could not offer before — a figure has no position in the text."""
    _seed_chunk(client, document_id="doc-9", parent_index=1, chunk_type="figure", page="12")
    r = client.get("/api/library/chunk-page", params={"key": "doc-9:p1"})
    assert r.status_code == 200
    assert r.json() == {"document_id": "doc-9", "page": 12}


def test_a_known_but_unplaceable_chunk_is_a_200_with_a_null_page(client):
    """Not a 404: its document can still be opened, at page 1, claiming nothing (ADR-050 D2)."""
    _seed_chunk(client, document_id="doc-7", parent_index=3)
    r = client.get("/api/library/chunk-page", params={"key": "doc-7:p3"})
    assert r.status_code == 200
    assert r.json() == {"document_id": "doc-7", "page": None}


# --- zoom: a sharper render on demand (ROADMAP 18, second pass) -------------------------------- #


def test_a_higher_dpi_returns_a_bigger_image(client, tmp_path):
    """Zoom is not magnification: the pane asks the server to draw the page again, larger."""
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    small = client.get(f"/api/library/documents/{doc_id}/page/1")
    large = client.get(f"/api/library/documents/{doc_id}/page/1", params={"dpi": 300})
    assert small.status_code == large.status_code == 200
    assert len(large.content) > len(small.content)


def test_an_absurd_dpi_is_clamped_rather_than_refused(client, tmp_path):
    """A zoom level is not a validation error, and the ceiling is what stops a work generator.

    Render cost grows with the square of dpi, so the bound has to hold at the route — but the
    honest answer to "sharper than we draw" is the sharpest we draw, not a 4xx.
    """
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    huge = client.get(f"/api/library/documents/{doc_id}/page/1", params={"dpi": 100000})
    ceiling = client.get(f"/api/library/documents/{doc_id}/page/1", params={"dpi": 400})
    assert huge.status_code == 200
    assert huge.content == ceiling.content, "an absurd dpi must render at the ceiling, not beyond"


def test_a_tiny_or_negative_dpi_is_clamped_to_the_floor(client, tmp_path):
    doc_id = _add_document(_make_pdf(tmp_path / "paper.pdf"))
    floor = client.get(f"/api/library/documents/{doc_id}/page/1", params={"dpi": 72})
    for bad in (1, 0, -300):
        r = client.get(f"/api/library/documents/{doc_id}/page/1", params={"dpi": bad})
        assert r.status_code == 200, f"dpi={bad} should render, not fail"
        assert r.content == floor.content
