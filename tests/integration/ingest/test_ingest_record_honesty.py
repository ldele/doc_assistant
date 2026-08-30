"""Regression (KI-53): what the ingest record *says about* a document must be true of it.

Both halves of KI-53 were found on the first EPUB/HTML round trip through the UI, and both are
about the row rather than the pipeline — extraction itself was clean:

* ``extraction_health`` opened with ``chunk_count <= 1 -> broken``, so a 2 KB web article that
  extracted perfectly into one full chunk was filed as a failure and rendered as "html · broken";
* ``extractor_used`` was hardcoded to ``config.PDF_EXTRACTOR``, so an EPUB and an HTML file both
  recorded ``pymupdf`` — a PDF extractor that never touched them.

This drives the real extractor over the committed fixtures (no cache bypass) so the assertions are
about what an *ingest* records, not about what the helper functions return in isolation.

Deterministic and offline: fake embedder, temp data dirs, temp SQLite. No paid call.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest
from doc_assistant.db.models import Base
from doc_assistant.db.models import Document as DBDocument

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures" / "documents"


@pytest.fixture
def isolated_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``ingest`` at temp data dirs + a temp SQLite, with a fake embedder."""
    docs = tmp_path / "sources"
    cache = tmp_path / "cache"
    chroma = tmp_path / "chroma"
    pc_chroma = tmp_path / "chroma_pc"
    for d in (docs, cache, chroma, pc_chroma):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(config, "DOCS_PATH", docs)
    monkeypatch.setattr(config, "CACHE_PATH", cache)
    monkeypatch.setattr(config, "CHROMA_PATH", str(chroma))
    monkeypatch.setattr(config, "PC_CHROMA_PATH", str(pc_chroma))
    monkeypatch.setattr(
        ingest, "get_embeddings", lambda name=None: DeterministicFakeEmbedding(size=16)
    )

    db_path = tmp_path / "library.db"
    engine = create_engine(f"sqlite:///{db_path}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    try:
        yield docs
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _row(filename: str) -> tuple[str | None, str | None, int | None]:
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        return session.execute(
            select(
                DBDocument.extractor_used,
                DBDocument.extraction_health,
                DBDocument.chunk_count,
            ).where(DBDocument.filename == filename)
        ).one()


@pytest.mark.parametrize(
    ("fixture", "expected_extractor"),
    [("article.html", "bs4"), ("treatise.epub", "ebooklib+bs4")],
)
def test_a_short_document_is_recorded_as_what_it_is(
    isolated_ingest: Path, fixture: str, expected_extractor: str
) -> None:
    """The row names the extractor that ran, and does not call a complete extraction broken."""
    shutil.copy(FIXTURES / fixture, isolated_ingest / fixture)

    stats = ingest.main()
    assert stats["added"] == 1, stats

    extractor_used, health, chunk_count = _row(fixture)
    # Health first: it is the half a user reads. Pre-fix this row rendered as "html · broken",
    # and asserting the extractor first would mask it behind the provenance failure.
    assert health == "healthy", (
        f"{fixture} extracted into {chunk_count} chunk(s) and read {health}"
    )
    assert extractor_used == expected_extractor
