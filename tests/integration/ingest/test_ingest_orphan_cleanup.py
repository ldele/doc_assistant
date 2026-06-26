"""Regression: incremental ingest must drop a document's OLD hash when its
cached *content* changes — not only when its source file is deleted.

Bug (DEVLOG 2026-06-13 "Full-corpus Marker table extraction + re-ingest (step 2)"):
document identity is ``doc_hash(text)`` over the cached markdown. Splicing tables
into a cached ``.md`` changes that hash, so a plain ``python -m doc_assistant.ingest``
*added* the new-hash document but left the pre-splice copy behind — two hashes per
changed file in both Chroma stores and SQLite, retrieval seeing duplicates. The only
cure was ``ingest --rebuild`` (a full wipe + re-embed). The orphan sweep now also
removes hashes no current source still produces, so an incremental run cleans the
stale copy.

Deterministic and offline: a fake embedder (no HuggingFace download), isolated temp
data dirs, and a temp SQLite bound to the global session machinery.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest
from doc_assistant.db.models import Base, Figure
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.ingest import figures

_DOC_V1 = """<!-- page:1 -->
# Retrieval study

Background prose on dense retrieval, written for the first ingest.

## Results
The accuracy numbers live in a table on the next page.
"""

# Same document after a (simulated) Marker table-splice: extra content => new hash.
_DOC_V2 = (
    _DOC_V1
    + """
<!-- table:marker:page=2:begin -->

| Model | Top-20 | Top-100 |
| --- | --- | --- |
| DPR | 78.4 | 85.4 |

<!-- table:marker:page=2:end -->
"""
)


@pytest.fixture
def isolated_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``ingest`` at temp data dirs + a temp SQLite, with a fake embedder.

    Yields the temp DOCS_PATH so a test can drop source files under it.
    """
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
    # Fake embedder: stable vectors, no HF download. Dimension is arbitrary.
    monkeypatch.setattr(
        ingest, "get_embeddings", lambda name=None: DeterministicFakeEmbedding(size=16)
    )

    # Rebind the global session machinery to a fresh, isolated SQLite.
    db_path = tmp_path / "library.db"
    from sqlalchemy import create_engine

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


def _write_cached_source(docs: Path, name: str, content: str) -> Path:
    """Create a ``.md`` source and a *fresh* cache so ingest reads ``content`` verbatim.

    The cache is written after the source, so ``is_cache_fresh`` is True and the real
    extractor never runs — the test controls the document text directly.
    """
    src = docs / name
    src.write_text("placeholder — bypassed by the fresh cache\n", encoding="utf-8")
    cached = ingest.get_cache_path(src)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(content, encoding="utf-8")
    return src


def _store_hashes(persist_dir: str) -> set[str]:
    """Read the set of ``doc_hash`` values present in a Chroma store."""
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=DeterministicFakeEmbedding(size=16),
        collection_name=ingest.get_collection_name(ingest.get_active_model_name()),
    )
    return ingest.get_indexed_hashes(db)


def _doc_hashes_for(filename: str) -> list[str]:
    """Every SQLite ``Document.doc_hash`` recorded for ``filename``."""
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(
            select(DBDocument.doc_hash).where(DBDocument.filename == filename)
        ).all()
    return [r[0] for r in rows]


def test_content_change_leaves_exactly_one_hash(isolated_ingest: Path) -> None:
    """Mutating a doc's cache then re-ingesting incrementally drops the stale copy."""
    docs = isolated_ingest
    src = _write_cached_source(docs, "paper.md", _DOC_V1)
    old_hash = ingest.doc_hash(_DOC_V1)

    ingest.main()  # first ingest — establishes the old hash
    assert _doc_hashes_for("paper.md") == [old_hash]

    # Simulate a Marker table-splice: rewrite the cache (source untouched, so the
    # cache stays fresh and the new content is what the re-ingest sees).
    ingest.get_cache_path(src).write_text(_DOC_V2, encoding="utf-8")
    new_hash = ingest.doc_hash(_DOC_V2)
    assert new_hash != old_hash

    ingest.main()  # incremental re-ingest — no --rebuild

    # Exactly one Document survives for the file, and it is the NEW content.
    assert _doc_hashes_for("paper.md") == [new_hash]
    # Both vector stores agree: only the new hash, the stale copy is gone.
    assert _store_hashes(config.CHROMA_PATH) == {new_hash}
    assert _store_hashes(config.PC_CHROMA_PATH) == {new_hash}
    # A content change must NOT delete the live cache the new hash came from.
    assert ingest.get_cache_path(src).read_text(encoding="utf-8") == _DOC_V2


def test_deleted_source_still_cleaned_and_cache_removed(isolated_ingest: Path) -> None:
    """The original orphan path (source file gone) still works after the refactor."""
    docs = isolated_ingest
    src = _write_cached_source(docs, "paper.md", _DOC_V1)
    cache = ingest.get_cache_path(src)

    ingest.main()
    assert _doc_hashes_for("paper.md") == [ingest.doc_hash(_DOC_V1)]
    assert cache.exists()

    # Source removed entirely; cache left behind (the real cleanup deletes it).
    src.unlink()
    ingest.main()

    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        remaining = session.execute(select(func.count()).select_from(DBDocument)).scalar_one()
    assert remaining == 0
    assert _store_hashes(config.CHROMA_PATH) == set()
    assert _store_hashes(config.PC_CHROMA_PATH) == set()
    # Gone source => its orphaned cache is swept too.
    assert not cache.exists()


def test_orphan_cleanup_sweeps_figure_dir_for_gone_source(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A gone document's on-disk figure PNG dir is swept; a live document's is kept (G1).

    Figure *rows* FK-cascade with the Document; the cropped PNGs under
    ``FIGURE_DIR/{doc_hash}/`` have no DB cascade, so the orphan sweep must remove them
    or they leak forever.
    """
    docs = isolated_ingest
    fig_root = docs.parent / "figures"
    monkeypatch.setattr(figures, "FIGURE_DIR", fig_root)

    gone = _write_cached_source(docs, "gone.md", _DOC_V1)
    _write_cached_source(docs, "keep.md", _DOC_V2)  # distinct content => distinct hash
    ingest.main()
    gone_hash = ingest.doc_hash(_DOC_V1)
    keep_hash = ingest.doc_hash(_DOC_V2)

    # Simulate extract_figures having written a PNG dir per document.
    for h in (gone_hash, keep_hash):
        d = figures.figure_dir(h)
        d.mkdir(parents=True, exist_ok=True)
        (d / "page1_fig0.png").write_bytes(b"\x89PNG fake")

    gone.unlink()  # source removed => gone_hash becomes a 'gone' orphan
    ingest.main()

    assert not figures.figure_dir(gone_hash).exists()  # swept with the orphan
    assert figures.figure_dir(keep_hash).exists()  # live document untouched
    assert (figures.figure_dir(keep_hash) / "page1_fig0.png").exists()


def test_orphan_cleanup_sweeps_figure_dir_on_content_change(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A content change orphans the old hash; its now-dead figure dir is swept too (G1)."""
    docs = isolated_ingest
    fig_root = docs.parent / "figures"
    monkeypatch.setattr(figures, "FIGURE_DIR", fig_root)

    src = _write_cached_source(docs, "paper.md", _DOC_V1)
    ingest.main()
    old_hash = ingest.doc_hash(_DOC_V1)
    old_dir = figures.figure_dir(old_hash)
    old_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / "page1_fig0.png").write_bytes(b"fake")

    # Splice tables into the cache (source untouched, cache stays fresh) => new hash;
    # old_hash is now a 'stale' orphan and its figure dir no longer matches any content.
    ingest.get_cache_path(src).write_text(_DOC_V2, encoding="utf-8")
    ingest.main()

    assert not old_dir.exists()


def test_figure_rows_cascade_on_document_delete(isolated_ingest: Path) -> None:
    """Deleting a Document removes its Figure rows — the cascade orphan cleanup relies on.

    ``cleanup_orphans_sqlite`` deletes orphan Document rows; the figure-dir sweep (G1)
    only handles the on-disk PNGs because the rows are expected to cascade. Asserted here
    directly since nothing else covered it.
    """
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = DBDocument(filename="f.pdf", source_original="/x/f.pdf", doc_hash="hh", format="pdf")
        session.add(doc)
        session.flush()
        doc_id = doc.id
        session.add(Figure(document_id=doc_id, doc_hash="hh", page=1, kind="figure"))

    def _figure_count() -> int:
        with session_scope() as session:
            return session.execute(
                select(func.count()).select_from(Figure).where(Figure.document_id == doc_id)
            ).scalar_one()

    assert _figure_count() == 1

    with session_scope() as session:
        session.delete(session.get(DBDocument, doc_id))

    assert _figure_count() == 0


def test_figure_dir_delete_failure_does_not_abort_sweep(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A locked/undeletable figure dir is logged and skipped — the sweep continues (G1).

    Guards the per-hash try/except (not ``rmtree(ignore_errors=True)``): one orphan's
    failed delete must neither raise nor stop the remaining orphans being swept.
    """
    docs = isolated_ingest
    fig_root = docs.parent / "figures"
    monkeypatch.setattr(figures, "FIGURE_DIR", fig_root)

    g1 = _write_cached_source(docs, "gone1.md", _DOC_V1)
    g2 = _write_cached_source(docs, "gone2.md", _DOC_V2)
    ingest.main()
    h1 = ingest.doc_hash(_DOC_V1)
    h2 = ingest.doc_hash(_DOC_V2)
    for h in (h1, h2):
        d = figures.figure_dir(h)
        d.mkdir(parents=True, exist_ok=True)
        (d / "page1_fig0.png").write_bytes(b"fake")

    g1.unlink()
    g2.unlink()  # both sources gone => both are orphans

    real_rmtree = ingest.cleanup.shutil.rmtree

    def selective_rmtree(path: object, *args: object, **kwargs: object) -> None:
        if str(h1) in str(path):
            raise OSError("simulated locked figure dir")
        real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(ingest.cleanup.shutil, "rmtree", selective_rmtree)

    ingest.main()  # must not raise despite the h1 rmtree failure

    assert figures.figure_dir(h1).exists()  # delete failed -> dir kept, no crash
    assert not figures.figure_dir(h2).exists()  # loop continued -> h2 swept
