"""KI-24 — ``ingest --rebuild`` rebuilds the index; it must not reset the library.

The rebuild branch used to run ``delete(DBDocument)``, which FK-cascaded away folder membership
(ADR-025), tags, keywords, citations and figures, orphaned the FK-less ``document_meta`` overrides
against ids that no longer existed, and reset user columns (``is_archived``, ``notes``) — all
invisible afterwards, because the folders themselves survived and merely looked empty. It also
dropped every **figure chunk** from the rebuilt index, because ``figure_units()`` queries by
document id and the rows were gone.

Rows the rebuild does not reproduce are swept afterwards instead, classified the way
``cleanup_orphans_sqlite`` classifies them (stale / gone) — with one deliberate difference: a file
that is still on disk but produced nothing *this run* is **kept**, because deleting a user's
folders and metadata on a transient extraction failure is exactly the loss this fix is about.

Same offline harness as ``test_ingest_orphan_cleanup.py``: fake embedder, temp data dirs, temp
SQLite bound to the global session machinery. No HuggingFace download, no LLM.
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
from doc_assistant.db.models import Base, DocumentMeta, Figure, Folder
from doc_assistant.db.models import Document as DBDocument
from doc_assistant.db.session import session_scope
from doc_assistant.ingest import figures

_DOC_A = """<!-- page:1 -->
# Dense retrieval

Prose about dense passage retrieval, long enough to chunk.
"""

_DOC_B = """<!-- page:1 -->
# Re-ranking

Prose about cross-encoder re-ranking, different content so the hash differs.
"""


@pytest.fixture
def isolated_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
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
    monkeypatch.setattr(figures, "FIGURE_DIR", tmp_path / "figures")

    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
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


def _drop_chroma_client_cache() -> None:
    """Release chromadb's per-path shared system between two in-process ``ingest.main()`` calls.

    chromadb caches one system per persist path, so the second call reattaches to the store the
    rebuild's ``rmtree`` was supposed to remove (and on Windows the open handles make that
    ``ignore_errors=True`` rmtree a no-op anyway). Production never hits this — ``--rebuild`` is a
    CLI entrypoint in a fresh process and no API route exposes it — but the sweep must be correct
    either way, which is why it keys on "what this run produced" rather than assuming an empty
    store.
    """
    from chromadb.api.client import SharedSystemClient

    SharedSystemClient.clear_system_cache()


def _empty_chroma_stores() -> None:
    """Stand in for the rebuild's ``rmtree`` when it cannot bite in-process (see above).

    Needed only by the test that must observe *re-chunking*; the row-sweep tests deliberately run
    against the harder, half-wiped state.
    """
    for path in (config.CHROMA_PATH, config.PC_CHROMA_PATH):
        store = Chroma(
            persist_directory=path,
            embedding_function=DeterministicFakeEmbedding(size=16),
            collection_name=ingest.get_collection_name(ingest.get_active_model_name()),
        )
        ids = store.get(include=[])["ids"]
        if ids:
            store.delete(ids=list(ids))


def _write_cached_source(docs: Path, name: str, content: str) -> Path:
    """Create a source file plus a *fresh* cache, so the real extractor never runs."""
    src = docs / name
    src.write_text("placeholder — bypassed by the fresh cache\n", encoding="utf-8")
    cached = ingest.get_cache_path(src)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(content, encoding="utf-8")
    return src


def _rows() -> dict[str, tuple[str, str]]:
    """``filename -> (document id, doc_hash)`` for every library row."""
    with session_scope() as session:
        return {
            str(f): (str(i), str(h))
            for f, i, h in session.execute(
                select(DBDocument.filename, DBDocument.id, DBDocument.doc_hash)
            ).all()
        }


def _row_count() -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(DBDocument)).scalar_one())


def _pc_chunks(doc_hash: str) -> list[dict[str, object]]:
    db = Chroma(
        persist_directory=config.PC_CHROMA_PATH,
        embedding_function=DeterministicFakeEmbedding(size=16),
        collection_name=ingest.get_collection_name(ingest.get_active_model_name()),
    )
    found = db.get(where={"doc_hash": doc_hash}, include=["metadatas"])
    return [m for m in found["metadatas"] if m]


# --- the KI-24 guard -------------------------------------------------------------------------- #


def test_rebuild_preserves_folders_tags_metadata_and_user_columns(isolated_ingest: Path) -> None:
    """The whole point: everything the user authored survives a --rebuild, ids included."""
    docs = isolated_ingest
    _write_cached_source(docs, "a.md", _DOC_A)
    _write_cached_source(docs, "b.md", _DOC_B)
    ingest.main()

    before = _rows()
    assert set(before) == {"a.md", "b.md"}

    from doc_assistant.library import add_documents_to_folder, create_folder, folder_document_ids

    folder = create_folder("My reading")
    add_documents_to_folder(folder.id, [before["a.md"][0], before["b.md"][0]])
    with session_scope() as session:
        session.add(DocumentMeta(document_id=before["a.md"][0], title_override="My own title"))
        row = session.get(DBDocument, before["b.md"][0])
        assert row is not None
        row.is_archived = True
        row.notes = "read this one twice"

    _drop_chroma_client_cache()
    ingest.main(force_rebuild=True)

    after = _rows()
    assert after == before, "a rebuild must reuse the same Document ids, not mint new ones"
    assert sorted(folder_document_ids(folder.id)) == sorted([before["a.md"][0]])  # b is archived
    with session_scope() as session:
        assert session.execute(select(func.count()).select_from(Folder)).scalar_one() == 1
        members = session.get(Folder, folder.id)
        assert members is not None
        assert len(members.documents) == 2  # membership itself is intact, archived included
        meta = session.get(DocumentMeta, before["a.md"][0])
        assert meta is not None and meta.title_override == "My own title"
        b_row = session.get(DBDocument, before["b.md"][0])
        assert b_row is not None
        assert b_row.is_archived is True
        assert b_row.notes == "read this one twice"


def test_rebuild_keeps_figure_chunks_in_the_index(isolated_ingest: Path) -> None:
    """The silent retrieval-quality half of KI-24.

    Figures are keyed by document id, so the old bulk delete meant ``figure_units()`` found none
    during the rebuild and the reindexed corpus carried **no figure chunks** — regenerating them
    needs the paid VLM describe pass.
    """
    docs = isolated_ingest
    _write_cached_source(docs, "a.md", _DOC_A)
    ingest.main()

    document_id, doc_hash = _rows()["a.md"]
    with session_scope() as session:
        session.add(
            Figure(
                document_id=document_id,
                doc_hash=doc_hash,
                page=1,
                kind="figure",
                image_path="figures/a/fig1.png",
                caption="Figure 1: retrieval recall",
                vlm_description="A described figure, produced by the (paid) VLM pass.",
            )
        )
    _empty_chroma_stores()
    _drop_chroma_client_cache()
    ingest.main(force_rebuild=True)

    kinds = [m.get("chunk_type") for m in _pc_chunks(doc_hash)]
    assert "figure" in kinds, "the rebuilt index lost its figure chunks"
    with session_scope() as session:
        assert session.execute(select(func.count()).select_from(Figure)).scalar_one() == 1


# --- what a rebuild still sweeps --------------------------------------------------------------- #


def test_rebuild_sweeps_the_row_of_a_deleted_source(isolated_ingest: Path) -> None:
    docs = isolated_ingest
    _write_cached_source(docs, "a.md", _DOC_A)
    gone = _write_cached_source(docs, "b.md", _DOC_B)
    ingest.main()
    assert _row_count() == 2

    gone.unlink()
    _drop_chroma_client_cache()
    ingest.main(force_rebuild=True)

    assert set(_rows()) == {"a.md"}


def test_rebuild_sweeps_the_stale_hash_of_changed_content(isolated_ingest: Path) -> None:
    """The case the bulk delete used to be the only cure for — still cured."""
    docs = isolated_ingest
    src = _write_cached_source(docs, "a.md", _DOC_A)
    ingest.main()
    old_hash = _rows()["a.md"][1]

    ingest.get_cache_path(src).write_text(_DOC_A + "\nAn added paragraph.\n", encoding="utf-8")
    _drop_chroma_client_cache()
    ingest.main(force_rebuild=True)

    rows = _rows()
    assert set(rows) == {"a.md"}
    assert rows["a.md"][1] != old_hash
    assert _row_count() == 1


def test_rebuild_keeps_a_row_whose_file_is_present_but_produced_nothing(
    isolated_ingest: Path,
) -> None:
    """New protection: a transient extraction failure must not delete the user's library data.

    The old bulk delete removed the row unconditionally, taking folder membership with it.
    """
    docs = isolated_ingest
    src = _write_cached_source(docs, "a.md", _DOC_A)
    ingest.main()
    document_id = _rows()["a.md"][0]

    from doc_assistant.library import add_documents_to_folder, create_folder, folder_document_ids

    folder = create_folder("My reading")
    add_documents_to_folder(folder.id, [document_id])

    # The file is still there, but now extracts to nothing -> "skipped", never reproduced.
    ingest.get_cache_path(src).write_text("   \n", encoding="utf-8")
    _drop_chroma_client_cache()
    ingest.main(force_rebuild=True)

    assert set(_rows()) == {"a.md"}
    assert folder_document_ids(folder.id) == [document_id]


def test_incremental_ingest_still_sweeps_orphans_after_the_change(isolated_ingest: Path) -> None:
    """Guard: the rebuild change must not weaken the ordinary (non-rebuild) orphan sweep."""
    docs = isolated_ingest
    _write_cached_source(docs, "a.md", _DOC_A)
    gone = _write_cached_source(docs, "b.md", _DOC_B)
    ingest.main()

    gone.unlink()
    ingest.main()

    assert set(_rows()) == {"a.md"}
