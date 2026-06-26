"""Regression (F1): the SQLite Document row is committed only AFTER both Chroma writes
land, so a vector-store failure never leaves an orphaned Document row with no chunks —
and the intersection dedup gate self-heals the partial write on the next run.

Bug: ``process_one_document`` committed the Document row (``upsert_document_in_sqlite``)
*before* ``db.add_documents`` / ``pc_db.add_documents``. A Chroma failure was caught and
the document skipped, but the committed row remained — a document with zero chunks. The
row write is now the last step.

Deterministic and offline: a fake embedder (no HuggingFace download), isolated temp data
dirs, a temp SQLite bound to the global session machinery, and a monkeypatched
``Chroma.add_documents`` to simulate the failure. No real Chroma server / LLM / paid call.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import ingest
from doc_assistant.db.models import Base, Figure
from doc_assistant.db.models import Document as DBDocument

_DOC = """<!-- page:1 -->
# A short paper

Background prose with enough text to chunk and embed for the write-ordering test.

## Results
A couple of sentences so the splitter has real content to work with.
"""


@pytest.fixture
def isolated_ingest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Point ``ingest`` at temp data dirs + a temp SQLite, with a fake embedder."""
    docs = tmp_path / "sources"
    cache = tmp_path / "cache"
    chroma = tmp_path / "chroma"
    pc_chroma = tmp_path / "chroma_pc"
    for d in (docs, cache, chroma, pc_chroma):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(ingest, "DOCS_PATH", docs)
    monkeypatch.setattr(ingest, "CACHE_PATH", cache)
    monkeypatch.setattr(ingest, "CHROMA_PATH", str(chroma))
    monkeypatch.setattr(ingest, "PC_CHROMA_PATH", str(pc_chroma))
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


def _write_cached_source(docs: Path, name: str, content: str) -> Path:
    """A ``.md`` source + a *fresh* cache so ingest reads ``content`` verbatim."""
    src = docs / name
    src.write_text("placeholder — bypassed by the fresh cache\n", encoding="utf-8")
    cached = ingest.get_cache_path(src)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(content, encoding="utf-8")
    return src


def _document_count() -> int:
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        return session.execute(select(func.count()).select_from(DBDocument)).scalar_one()


def _open_store(persist_dir: str) -> Chroma:
    return Chroma(
        persist_directory=persist_dir,
        embedding_function=DeterministicFakeEmbedding(size=16),
        collection_name=ingest.get_collection_name(ingest.get_active_model_name()),
    )


def _store_hashes(persist_dir: str) -> set[str]:
    return ingest.get_indexed_hashes(_open_store(persist_dir))


def _doc_id_for(filename: str) -> str:
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        return session.execute(
            select(DBDocument.id).where(DBDocument.filename == filename)
        ).scalar_one()


def _chunk_count_for(filename: str) -> int:
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        return session.execute(
            select(DBDocument.chunk_count).where(DBDocument.filename == filename)
        ).scalar_one()


def _pc_figure_chunk_ids(h: str) -> list[str]:
    """figure_id values of any chunk_type='figure' chunks for ``h`` in the pc store."""
    data = _open_store(ingest.PC_CHROMA_PATH).get(where={"doc_hash": h}, include=["metadatas"])
    return [
        str(m.get("figure_id")) for m in data["metadatas"] if m and m.get("chunk_type") == "figure"
    ]


def test_baseline_chroma_failure_leaves_no_orphan_row(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first vector write failing commits no Document row (no orphan)."""
    docs = isolated_ingest
    _write_cached_source(docs, "paper.md", _DOC)

    def boom(self: Chroma, *args: object, **kwargs: object) -> object:
        raise RuntimeError("simulated Chroma failure (baseline store)")

    monkeypatch.setattr(Chroma, "add_documents", boom)

    stats = ingest.main()

    assert sum(stats.values()) == 1  # single-doc fixture (pins the call-counter target)
    assert stats["error"] == 1
    assert stats["added"] == 0
    # The row is committed only after both Chroma writes succeed, so a failed write
    # leaves zero Document rows — no orphan.
    assert _document_count() == 0


def test_pc_store_failure_leaves_no_orphan_and_self_heals(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SECOND store failing (after the first succeeds) still leaves no orphan row —
    and the next clean run completes the partial write via the intersection dedup gate."""
    docs = isolated_ingest
    _write_cached_source(docs, "paper.md", _DOC)
    expected_hash = ingest.doc_hash(_DOC)

    # Fail the 2nd add_documents call of the run — the parent-child store. (Within one
    # document, the baseline store is written first, then the parent-child store.)
    original = Chroma.add_documents
    calls = {"n": 0}

    def flaky(self: Chroma, *args: object, **kwargs: object) -> object:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated Chroma failure (parent-child store)")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Chroma, "add_documents", flaky)

    stats = ingest.main()
    assert (
        sum(stats.values()) == 1
    )  # single-doc fixture: the 2nd add_documents call is the pc write
    assert stats["error"] == 1
    assert _document_count() == 0  # no orphaned row despite the baseline write landing

    # Exactly the partial state the intersection dedup gate exists to repair: the hash
    # is in the baseline store but not the parent-child store.
    assert expected_hash in _store_hashes(ingest.CHROMA_PATH)
    assert expected_hash not in _store_hashes(ingest.PC_CHROMA_PATH)

    # Second run, no failure injected: the hash is missing from the intersection, so the
    # document is reprocessed, both writes land, and the row is finally committed.
    monkeypatch.setattr(Chroma, "add_documents", original)
    stats2 = ingest.main()
    assert stats2["added"] == 1
    assert _document_count() == 1
    assert expected_hash in _store_hashes(ingest.CHROMA_PATH)
    assert expected_hash in _store_hashes(ingest.PC_CHROMA_PATH)


def test_sqlite_commit_failure_self_heals_via_reconciliation(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both Chroma writes land but the final SQLite commit fails → an *inverse* orphan: the
    hash is in BOTH stores (so in the dedup intersection) with no Document row. The next run's
    inverse-orphan reconciliation drops it from the dedup set, reprocesses the doc, and finally
    commits the row — no `--rebuild` needed.

    The SQLite-side mirror of ``test_pc_store_failure_leaves_no_orphan_and_self_heals``. Fails on
    pre-reconciliation code: run two would skip the in-intersection hash and never commit the row
    (``_document_count() == 0`` after the second run). The orphan cleanup pass cannot heal it —
    the source is present and its content is unchanged, so it is neither gone nor stale.
    """
    docs = isolated_ingest
    _write_cached_source(docs, "paper.md", _DOC)
    expected_hash = ingest.doc_hash(_DOC)

    # Fail ONLY the final row commit — it runs after both Chroma writes have already landed.
    original_upsert = ingest.upsert_document_in_sqlite
    state = {"fail": True}

    def maybe_boom(*args: object, **kwargs: object) -> object:
        if state["fail"]:
            raise RuntimeError("simulated SQLite commit failure (after both Chroma writes)")
        return original_upsert(*args, **kwargs)

    monkeypatch.setattr(ingest, "upsert_document_in_sqlite", maybe_boom)

    stats = ingest.main()
    assert stats["error"] == 1
    assert _document_count() == 0  # the commit failed, so no row
    # The inverse orphan: present in BOTH stores, yet with no Document row.
    assert expected_hash in _store_hashes(ingest.CHROMA_PATH)
    assert expected_hash in _store_hashes(ingest.PC_CHROMA_PATH)

    # Second run, commit no longer failing: reconciliation removes the no-row hash from the
    # dedup intersection, so the doc is reprocessed and the row is finally committed.
    state["fail"] = False
    stats2 = ingest.main()
    assert stats2["added"] == 1
    assert _document_count() == 1
    assert expected_hash in _store_hashes(ingest.CHROMA_PATH)
    assert expected_hash in _store_hashes(ingest.PC_CHROMA_PATH)


def test_recorded_chunk_count_excludes_figure_chunks(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The committed Document.chunk_count is the prose-chunk count, not prose + figures.

    Pins the B1 ``baseline_chunk_count`` snapshot (taken before figure chunks are
    appended). A regression to ``len(documents)`` after the figure append would inflate
    the recorded count.
    """
    docs = isolated_ingest
    _write_cached_source(docs, "paper.md", _DOC)
    h = ingest.doc_hash(_DOC)

    # Inject one described figure so the figure-chunk path runs without a DB/VLM round-trip.
    monkeypatch.setattr(
        ingest, "figure_units", lambda doc_id: [("Figure 1. A described figure.", 1, "fig-xyz")]
    )

    assert ingest.main()["added"] == 1

    baseline_total = len(_open_store(ingest.CHROMA_PATH).get(where={"doc_hash": h})["ids"])
    # The figure WAS materialised into the store (so this isn't a vacuous test)...
    assert "fig-xyz" in _pc_figure_chunk_ids(h)
    # ...yet the recorded chunk_count excludes it: total == prose chunks + 1 figure.
    assert baseline_total == _chunk_count_for("paper.md") + 1


def test_reingest_reuses_document_id_keeping_figures_linked(
    isolated_ingest: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-ingest reuses the existing Document.id so id-keyed sidecars (figures) stay linked.

    Drives the B1 reuse branch (``_existing_document_id(h)`` returning the existing id).
    A plain same-content re-ingest is skipped by the intersection dedup gate, so the hash
    is dropped from both Chroma stores first (row preserved) to force reprocessing. If the
    branch regressed to always minting a fresh uuid, ``figure_units(new_id)`` would find
    nothing and the figure chunk would not reappear.
    """
    from doc_assistant.db.session import session_scope

    docs = isolated_ingest
    _write_cached_source(docs, "paper.md", _DOC)
    h = ingest.doc_hash(_DOC)

    assert ingest.main()["added"] == 1
    doc_id = _doc_id_for("paper.md")

    # A described figure now exists for this document (as scripts/describe_figures would write).
    with session_scope() as session:
        session.add(
            Figure(
                id="fig-seed-1",
                document_id=doc_id,
                doc_hash=h,
                page=1,
                kind="figure",
                vlm_description="A described figure for the id-reuse test.",
            )
        )

    # Drop the hash from BOTH vector stores so it falls out of the dedup intersection and
    # gets reprocessed — but the SQLite Document row (and its Figure) survives.
    _open_store(ingest.CHROMA_PATH).delete(where={"doc_hash": h})
    _open_store(ingest.PC_CHROMA_PATH).delete(where={"doc_hash": h})

    ingest.main(skip_cleanup=True)  # skip_cleanup so orphan cleanup can't touch the row

    # Same row reused (not duplicated), and the figure re-materialised — proving the
    # reused id resolved figure_units() against the existing Figure sidecar.
    assert _document_count() == 1
    assert _doc_id_for("paper.md") == doc_id
    assert "fig-seed-1" in _pc_figure_chunk_ids(h)
