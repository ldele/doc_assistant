"""A letter-spacing rule still applies after the document is ingested again (T-009, A8'').

The rule is the ignore list, keyed by the PDF's sha256 and part of the extraction fingerprint
(ADR-056), so it should hold through every way a document's text is read again: a per-part text
re-run, an index of the whole folder, a rebuild, and an extraction from scratch. Read in the code
until now; this drives each one on a real PDF.

Real extraction and ingest over temp stores, a fake embedder (no model download), one extraction
worker (in this process, as the installed build runs).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pymupdf
import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest
from doc_assistant.db.models import Base, Document
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import cache as cache_mod

SIZE = 10.0
FONT = pymupdf.Font("helv")
TITLE = "NEURONS"
BODY = "The neuron doctrine holds that the nervous system is made of discrete cells."


def _spaced_pdf(path: Path) -> Path:
    """A page whose first line is TITLE drawn one letter at a time, then a line of body text."""
    doc = pymupdf.open()
    page = doc.new_page()
    x = 72.0
    for ch in TITLE:
        page.insert_text((x, 100), ch, fontsize=SIZE, fontname="helv")
        x += FONT.text_length(ch, fontsize=SIZE) + 0.3 * SIZE
    page.insert_text((72, 130), BODY, fontsize=SIZE, fontname="helv")
    doc.save(path)
    doc.close()
    return path


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    docs = tmp_path / "sources"
    for d in (docs, tmp_path / "cache", tmp_path / "chroma", tmp_path / "chroma_pc"):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DOC_INGEST_WORKERS", "1")
    monkeypatch.setattr(config, "DOCS_PATH", docs)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PC_CHROMA_PATH", str(tmp_path / "chroma_pc"))
    monkeypatch.setattr(config, "REJOIN_IGNORE_PATH", tmp_path / "rejoin_ignore.json")
    fake = DeterministicFakeEmbedding(size=16)
    monkeypatch.setattr(ingest, "get_embeddings", lambda name=None: fake)
    cache_mod._ignore_list_memo.clear()
    cache_mod.extraction_fingerprint.cache_clear()

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", future=True)
    Base.metadata.create_all(engine)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(
        session_mod,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    yield docs
    cache_mod._ignore_list_memo.clear()
    cache_mod.extraction_fingerprint.cache_clear()
    engine.dispose()


def _row() -> tuple[str, str]:
    """``(document id, its cached text)`` for the one document."""
    with session_scope() as session:
        doc = session.execute(select(Document)).scalar_one()
        doc_id, cache = str(doc.id), doc.source_cache
    assert cache is not None
    return doc_id, Path(cache).read_text(encoding="utf-8")


def _closed(text: str) -> bool:
    return TITLE in text


def test_a_keep_spaced_rule_holds_through_every_re_ingest(library: Path) -> None:
    from doc_assistant.library import spacing
    from doc_assistant.reingest import rerun

    pdf = _spaced_pdf(library / "neurons.pdf")
    assert not ingest.main(files=[pdf]).get("error")
    doc_id, text = _row()
    assert _closed(text), "without a rule the spaced title is closed"

    # The person keeps every run of this document as extracted; it applies at the next re-run.
    spacing.set_keep_all(doc_id, True)
    cache_mod._ignore_list_memo.clear()

    outcome = rerun([doc_id], ["text"]).outcomes[0]
    assert outcome.status == "ok", outcome.detail
    same_id, text = _row()
    assert same_id == doc_id
    assert not _closed(text), "a per-part text re-run applies the rule"

    for label, run in (
        ("an index of the folder", lambda: ingest.main()),
        ("a rebuild", lambda: ingest.main(force_rebuild=True)),
    ):
        assert not run().get("error"), label
        same_id, text = _row()
        assert same_id == doc_id, label
        assert not _closed(text), f"the rule still applies after {label}"

    # An extraction from scratch: the cached text is gone, so the PDF is read again.
    Path(_cache_path()).unlink()
    assert not ingest.main().get("error")
    _, text = _row()
    assert not _closed(text), "a fresh extraction reads the rule from the list"
    record = spacing.document_spacing(doc_id)
    assert record is not None and record.keep_all


def _cache_path() -> str:
    with session_scope() as session:
        cache = session.execute(select(Document.source_cache)).scalar_one()
    assert cache is not None
    return str(cache)
