"""The letter-spacing routes (ADR-056 points 5, 7, 9, 10) — over the wire, against a real PDF.

A PDF whose title is drawn letter by letter is extracted for real, so the record the routes read
is the one the extractor writes. Then every change the control makes goes through the routes and
is read back twice: in the ignore-list file, and in what the next extraction writes.
"""

from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path

import pymupdf
import pytest
from apps.api.main import create_app
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from doc_assistant import config
from doc_assistant.ingest import cache

SIZE = 10.0
FONT = pymupdf.Font("helv")


class FakeController:
    """The minimum surface `create_app` needs; the spacing routes touch none of it."""

    def __init__(self) -> None:
        self.chunk_count = 0

    def corpus_stats(self) -> dict[str, int]:
        return {"chunk_count": 0}


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A database, a library folder, a cache and an ignore list, all under ``tmp_path``."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{db_path}", future=True)
    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    from doc_assistant.db.models import Base

    Base.metadata.create_all(engine)
    sources = tmp_path / "sources"
    sources.mkdir()
    monkeypatch.setattr(config, "DOCS_PATH", sources)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(config, "REJOIN_IGNORE_PATH", tmp_path / "rejoin_ignore.json")
    monkeypatch.setenv("DOC_SOURCE_DIR", str(sources))
    cache._ignore_list_memo.clear()
    yield sources
    cache._ignore_list_memo.clear()
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(db_path)


def _title_pdf(path: Path) -> Path:
    """A page whose first line is NEURONS drawn letter by letter, 0.3 em apart, then a sentence."""
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72.0
    for ch in "NEURONS":
        page.insert_text((x, 100), ch, fontsize=SIZE, fontname="helv")
        x += FONT.text_length(ch, fontsize=SIZE) + 0.3 * SIZE
    page.insert_text((72, 140), "A sentence of body text.", fontsize=SIZE, fontname="helv")
    doc.save(path)
    doc.close()
    return path


def _document(path: Path, fmt: str = "pdf") -> str:
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    cached = cache.get_cache_path(path)
    with session_scope() as session:
        session.add(
            Document(
                id="doc-1",
                filename=path.name,
                source_original=str(path),
                source_cache=str(cached),
                doc_hash="h",
                format=fmt,
                page_count=1,
            )
        )
    return "doc-1"


@pytest.fixture
def setup(home: Path):
    pdf = _title_pdf(home / "paper.pdf")
    doc_id = _document(pdf)
    text = cache.load_or_extract(pdf)
    client = TestClient(create_app(controller=FakeController()))
    return client, doc_id, pdf, text


def _ignore_file() -> dict:
    return json.loads(Path(config.REJOIN_IGNORE_PATH).read_text(encoding="utf-8"))


# --- reading ---------------------------------------------------------------------------------


def test_extraction_records_the_run_and_the_route_shows_it(setup) -> None:
    client, doc_id, pdf, text = setup
    assert "NEURONS" in text
    record = cache.read_spacing_record(cache.get_cache_path(pdf))
    assert [(r["spaced"], r["state"], r["page"]) for r in record] == [
        ("N E U R O N S", "closed", 1)
    ]

    body = client.get(f"/api/library/documents/{doc_id}/spacing").json()
    assert body["pageable"] is True and body["keep_all"] is False
    [row] = body["rows"]
    assert (row["spaced"], row["written"], row["state"], row["pending"]) == (
        "N E U R O N S",
        "NEURONS",
        "closed",
        None,
    )
    assert row["pages"] == [1] and len(row["crop"]) == 4


def test_an_unknown_document_is_a_404(setup) -> None:
    client, *_ = setup
    assert client.get("/api/library/documents/nope/spacing").status_code == 404


def test_a_document_that_is_not_a_pdf_has_nothing_to_show_or_keep(home: Path) -> None:
    epub = home / "book.epub"
    epub.write_bytes(b"not really")
    doc_id = _document(epub, fmt="epub")
    client = TestClient(create_app(controller=FakeController()))
    assert client.get(f"/api/library/documents/{doc_id}/spacing").json()["pageable"] is False
    r = client.post(f"/api/library/documents/{doc_id}/spacing/keep", json={"spaced": "A B C"})
    assert r.status_code == 409


def test_a_crop_is_a_png_of_part_of_the_page(setup) -> None:
    client, doc_id, *_ = setup
    crop = client.get(f"/api/library/documents/{doc_id}/spacing").json()["rows"][0]["crop"]
    whole = client.get(f"/api/library/documents/{doc_id}/page/1")
    part = client.get(
        f"/api/library/documents/{doc_id}/page/1", params={"clip": ",".join(map(str, crop))}
    )
    assert part.status_code == 200 and part.content[:4] == b"\x89PNG"
    assert len(part.content) < len(whole.content)
    bad = client.get(f"/api/library/documents/{doc_id}/page/1", params={"clip": "1,2,three"})
    assert bad.status_code == 422


# --- changing the list -----------------------------------------------------------------------


def test_keep_spaced_writes_an_entry_and_takes_effect_at_the_next_extraction(setup) -> None:
    client, doc_id, pdf, _ = setup
    row = client.post(
        f"/api/library/documents/{doc_id}/spacing/keep", json={"spaced": "N E U R O N S"}
    ).json()["rows"][0]
    assert (row["state"], row["pending"]) == ("closed", "kept")
    [entry] = _ignore_file()["documents"]
    assert entry["keep"] == ["N E U R O N S"] and entry["size"] == pdf.stat().st_size

    assert not cache.is_cache_fresh(pdf, cache.get_cache_path(pdf)), "the entry re-extracts"
    text = cache.load_or_extract(pdf)
    assert "NEURONS" not in text and "N E U R O N S" in text
    row = client.get(f"/api/library/documents/{doc_id}/spacing").json()["rows"][0]
    assert (row["state"], row["pending"]) == ("kept", None)


def test_your_spacing_is_applied_and_a_letter_change_is_refused(setup) -> None:
    client, doc_id, pdf, _ = setup
    url = f"/api/library/documents/{doc_id}/spacing/edit"
    refused = client.post(url, json={"spaced": "N E U R O N S", "chosen": "NEURONZ"})
    assert refused.status_code == 422
    assert not Path(config.REJOIN_IGNORE_PATH).exists(), "a refused edit writes nothing"

    row = client.post(url, json={"spaced": "N E U R O N S", "chosen": "NEU RONS"}).json()["rows"][
        0
    ]
    assert (row["pending"], row["chosen"]) == ("yours", "NEU RONS")
    assert "NEU RONS" in cache.load_or_extract(pdf)


def test_remove_gives_the_run_back_to_the_rule(setup) -> None:
    client, doc_id, *_ = setup
    base = f"/api/library/documents/{doc_id}/spacing"
    client.post(f"{base}/keep", json={"spaced": "N E U R O N S"})
    row = client.post(f"{base}/remove", json={"spaced": "N E U R O N S"}).json()["rows"][0]
    assert row["pending"] is None
    assert _ignore_file()["documents"] == [], "an entry that keeps nothing is no entry"


def test_keep_every_run_spaced(setup) -> None:
    client, doc_id, *_ = setup
    base = f"/api/library/documents/{doc_id}/spacing"
    body = client.post(f"{base}/keep-all", json={"keep": True}).json()
    assert body["keep_all"] is True and body["rows"][0]["pending"] == "kept"
    assert _ignore_file()["documents"][0]["keep"] == ["*"]
    body = client.post(f"{base}/keep-all", json={"keep": False}).json()
    assert body["keep_all"] is False and body["rows"][0]["pending"] is None


def test_settings_lists_removes_all_and_undoes(setup) -> None:
    client, doc_id, *_ = setup
    client.post(f"/api/library/documents/{doc_id}/spacing/keep", json={"spaced": "N E U R O N S"})
    [entry] = client.get("/api/library/spacing").json()["entries"]
    assert (entry["file"], entry["document_id"], entry["kept"]) == (
        "paper.pdf",
        doc_id,
        ["N E U R O N S"],
    )

    removed = client.post("/api/library/spacing/remove", json={"sha256": None}).json()
    assert (removed["removed"], removed["documents"]) == (1, ["paper.pdf"])
    assert client.get("/api/library/spacing").json()["entries"] == []

    restored = client.post("/api/library/spacing/restore", json={"previous": removed["previous"]})
    assert [e["file"] for e in restored.json()["entries"]] == ["paper.pdf"]


def test_an_unreadable_list_is_never_overwritten(setup) -> None:
    client, doc_id, *_ = setup
    Path(config.REJOIN_IGNORE_PATH).write_text("{broken", encoding="utf-8")
    r = client.post(
        f"/api/library/documents/{doc_id}/spacing/keep", json={"spaced": "N E U R O N S"}
    )
    assert r.status_code == 409 and "cannot be read" in r.json()["detail"]
    assert Path(config.REJOIN_IGNORE_PATH).read_text(encoding="utf-8") == "{broken"


def test_a_list_of_the_wrong_shape_is_refused_and_kept(setup) -> None:
    """Valid JSON that is not a list of entries: a sentence, not a 500, and the file stays."""
    client, doc_id, *_ = setup
    text = json.dumps({"version": 1, "documents": ["not an entry"]})
    Path(config.REJOIN_IGNORE_PATH).write_text(text, encoding="utf-8")
    assert client.get("/api/library/spacing").status_code == 409
    r = client.post(
        f"/api/library/documents/{doc_id}/spacing/keep", json={"spaced": "N E U R O N S"}
    )
    assert r.status_code == 409 and "expected form" in r.json()["detail"]
    assert Path(config.REJOIN_IGNORE_PATH).read_text(encoding="utf-8") == text


def test_a_restore_of_the_wrong_shape_is_refused(setup) -> None:
    client, *_ = setup
    r = client.post("/api/library/spacing/restore", json={"previous": {"documents": ["x"]}})
    assert r.status_code == 422
    assert not Path(config.REJOIN_IGNORE_PATH).exists()


def test_a_document_whose_file_moved_is_still_matched_in_the_list(setup) -> None:
    """The source's stored path is stale; `resolve_source_path` finds it in the library folder."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    client, doc_id, *_ = setup
    client.post(f"/api/library/documents/{doc_id}/spacing/keep", json={"spaced": "N E U R O N S"})
    with session_scope() as session:
        doc = session.get(Document, doc_id)
        assert doc is not None
        doc.source_original = "/elsewhere/paper.pdf"
    entries = client.get("/api/library/spacing").json()["entries"]
    assert [e["document_id"] for e in entries] == [doc_id]


# --- what removing a document leaves ---------------------------------------------------------


class _EmptyChroma:
    def get(self, **_: object) -> dict[str, list[str]]:
        return {"ids": []}


def test_removing_a_document_removes_its_record(setup) -> None:
    from doc_assistant.library.documents import purge_document_record

    _, doc_id, pdf, _ = setup
    cached = cache.get_cache_path(pdf)
    record = cache.spacing_record_path(cached)
    assert record.exists()
    purge_document_record(doc_id, _EmptyChroma(), doc_hash="h", source_cache=str(cached))
    assert not cached.exists()
    assert not record.exists()


def test_a_removed_sources_orphan_clean_removes_its_record(setup) -> None:
    from doc_assistant.ingest.cleanup import cleanup_orphans_chroma

    _, _, pdf, _ = setup
    cached = cache.get_cache_path(pdf)
    meta = {"doc_hash": "h", "source_original": "/gone/paper.pdf", "source_cache": str(cached)}

    class _Store(_EmptyChroma):
        def get(self, **_: object) -> dict[str, object]:
            return {"ids": ["c1"], "metadatas": [meta]}

        def delete(self, **_: object) -> None:
            pass

    cleanup_orphans_chroma(_Store(), ["h"], also_clean_cache=True)
    assert not cached.exists()
    assert not cache.spacing_record_path(cached).exists()
