"""Per-part re-ingest (ADR-048, ROADMAP 20/21).

The load-bearing test here is `test_re_extraction_purges_the_superseded_chunks`. `ingest.main`
sweeps orphans only when `files is None`, and a per-document re-extract is the `files is not None`
branch — so without an explicit purge the previous `doc_hash`'s chunks stay in both stores and stay
retrievable, silently indexing the document twice. That is not a hypothetical: it is what the
selective path does by construction.

Deterministic and offline: fake embedder, temp data dirs, temp SQLite. Sources are `.md`, which
`extract_text` reads verbatim — so "the file changed, therefore the hash moves" is exact rather
than dependent on a PDF extractor's output.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_chroma import Chroma
from langchain_core.embeddings import DeterministicFakeEmbedding
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import config, ingest, reingest
from doc_assistant.db.models import Base, Citation
from doc_assistant.db.models import Document as DBDocument

_BODY = """# A short paper

Background prose with enough text to chunk and embed for the re-ingest tests.

## Results
A couple of sentences so the splitter has real content to work with.

## References
1. Delacroix, M. (2019). A prior study of things. Journal of Things, 4(2), 1-10.
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    docs = tmp_path / "sources"
    for d in (docs, tmp_path / "cache", tmp_path / "chroma", tmp_path / "chroma_pc"):
        d.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "DOCS_PATH", docs)
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(config, "CHROMA_PATH", str(tmp_path / "chroma"))
    monkeypatch.setattr(config, "PC_CHROMA_PATH", str(tmp_path / "chroma_pc"))
    monkeypatch.setattr(
        ingest, "get_embeddings", lambda name=None: DeterministicFakeEmbedding(size=16)
    )

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


def _ingest_one(docs: Path, name: str = "paper.md", body: str = _BODY) -> tuple[str, str]:
    """Ingest one source and return its (document_id, doc_hash)."""
    (docs / name).write_text(body, encoding="utf-8")
    stats = ingest.main()
    assert stats["added"] == 1, stats
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        row = session.execute(
            select(DBDocument.id, DBDocument.doc_hash).where(DBDocument.filename == name)
        ).one()
    return str(row[0]), str(row[1])


def _chunks_for(hash_: str) -> dict[str, int]:
    """Chunk counts for a doc_hash in each store, by store path."""
    from doc_assistant.embeddings import get_active_model_name, get_collection_name

    collection = get_collection_name(get_active_model_name())
    out: dict[str, int] = {}
    for label, path in (("baseline", config.CHROMA_PATH), ("pc", config.PC_CHROMA_PATH)):
        store = Chroma(
            persist_directory=str(path), embedding_function=None, collection_name=collection
        )
        out[label] = len(store.get(where={"doc_hash": hash_}, include=[]).get("ids", []))
    return out


# --- the registry ---------------------------------------------------------------------------- #


def test_every_declared_part_has_a_runner_and_a_cost() -> None:
    """The registry is what the API and the UI both read; a part with no runner is a 500."""
    assert {p.id for p in reingest.PARTS} == set(reingest._RUNNERS)
    assert len({p.id for p in reingest.PARTS}) == len(reingest.PARTS), "duplicate part id"
    for part in reingest.PARTS:
        assert part.label and part.blurb, part.id
        # The cost statement is the whole reason the parts are separate (ADR-048) — an empty one
        # turns the control back into four equivalent-looking checkboxes.
        assert part.cost, f"{part.id} has no cost statement"
    assert [p.id for p in reingest.PARTS if p.moves_identity] == ["text"]


def test_the_registry_order_is_cheapest_first() -> None:
    """Two contracts ride on this order and neither is visible from the list itself.

    `rerun` executes in registry order, which is why `text` is last — it replaces the cache the
    cheap parts read. And the client's cost summary quotes the *last selected* part as the dearest
    one, so a part inserted out of cost order makes the dialog understate the wait. Pinned as a
    literal because the costs are prose and cannot be compared.
    """
    assert [p.id for p in reingest.PARTS] == [
        "metadata",
        "crops",
        "figures",
        "references",
        "text",
    ]


def test_an_unknown_part_is_refused_before_anything_runs(env: Path) -> None:
    doc_id, _ = _ingest_one(env)
    with pytest.raises(reingest.UnknownPart):
        reingest.rerun([doc_id], ["metadata", "nonsense"])


def test_unknown_document_ids_are_dropped_not_fatal(env: Path) -> None:
    doc_id, _ = _ingest_one(env)
    result = reingest.rerun([doc_id, "no-such-document"], ["metadata"])
    assert {o.document_id for o in result.outcomes} == {doc_id}


def test_text_runs_last_whatever_order_it_was_asked_in(env: Path) -> None:
    """Re-extraction rewrites the cache the cheap parts read; running them first derives from
    text the user asked to replace."""
    doc_id, _ = _ingest_one(env)
    result = reingest.rerun([doc_id], ["text", "metadata", "references"])
    assert [o.part for o in result.outcomes] == ["metadata", "references", "text"]


# --- the parts ------------------------------------------------------------------------------- #


def test_metadata_rerun_writes_the_extracted_defaults(env: Path) -> None:
    from doc_assistant.db.session import session_scope

    doc_id, _ = _ingest_one(env)
    with session_scope() as session:
        session.execute(
            DBDocument.__table__.update().where(DBDocument.id == doc_id).values(title=None)
        )

    result = reingest.rerun([doc_id], ["metadata"])
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes
    with session_scope() as session:
        title = session.execute(
            select(DBDocument.title).where(DBDocument.id == doc_id)
        ).scalar_one()
    assert title, "the re-run did not write a title back"


def test_a_part_that_has_no_cached_text_skips_with_a_reason(env: Path) -> None:
    """A skip must say why. 'Nothing happened' is the failure mode this feature cannot afford."""
    from doc_assistant.db.session import session_scope

    doc_id, _ = _ingest_one(env)
    with session_scope() as session:
        cache = session.execute(
            select(DBDocument.source_cache).where(DBDocument.id == doc_id)
        ).scalar_one()
    Path(str(cache)).unlink()

    result = reingest.rerun([doc_id], ["metadata", "references"])
    assert [o.status for o in result.outcomes] == ["skipped", "skipped"]
    for outcome in result.outcomes:
        assert "Text & chunks" in outcome.detail, outcome.detail


def test_references_rerun_replaces_this_documents_rows(env: Path) -> None:
    from doc_assistant.db.session import session_scope

    doc_id, _ = _ingest_one(env)
    reingest.rerun([doc_id], ["references"])
    with session_scope() as session:
        first = (
            session.execute(select(Citation).where(Citation.source_document_id == doc_id))
            .scalars()
            .all()
        )
    assert first, "the fixture's reference list produced no citations"

    # A re-run replaces rather than merges — merging would keep whatever the previous parse got
    # wrong, which is the thing being re-run to fix.
    reingest.rerun([doc_id], ["references"])
    with session_scope() as session:
        second = (
            session.execute(select(Citation).where(Citation.source_document_id == doc_id))
            .scalars()
            .all()
        )
    assert len(second) == len(first)


def test_figures_are_skipped_for_a_non_pdf_with_the_reason(env: Path) -> None:
    doc_id, _ = _ingest_one(env)
    result = reingest.rerun([doc_id], ["figures"])
    assert result.outcomes[0].status == "skipped"
    assert "PDF-only" in result.outcomes[0].detail


# --- the one that pays for the module ---------------------------------------------------------- #


def test_re_extraction_purges_the_superseded_chunks(env: Path) -> None:
    """The whole reason `_rerun_text` exists rather than just calling `ingest.main(files=[...])`.

    `ingest.main` runs `cleanup_orphans_*` only when `files is None`. Re-extraction moves
    `doc_hash` (ADR-042) and ADR-047 keeps the row attached — so without the purge the OLD hash's
    chunks stay in both stores, and the document is indexed twice.
    """
    doc_id, before = _ingest_one(env)
    assert sum(_chunks_for(before).values()) > 0, "fixture did not index"

    # Change the source, so re-extraction genuinely yields different text and a different hash.
    (env / "paper.md").write_text(
        _BODY.replace("Background prose", "Rewritten prose"), encoding="utf-8"
    )

    result = reingest.rerun([doc_id], ["text"])
    assert [o.status for o in result.outcomes] == ["ok"], result.outcomes

    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        after = str(
            session.execute(
                select(DBDocument.doc_hash).where(DBDocument.id == doc_id)
            ).scalar_one()
        )
    assert after != before, "the fixture did not actually move the hash"

    stale = _chunks_for(before)
    assert stale == {"baseline": 0, "pc": 0}, f"superseded chunks survived: {stale}"
    assert sum(_chunks_for(after).values()) > 0, "the re-extracted document is not indexed"
    assert "superseded" in result.outcomes[0].detail


def test_re_extraction_of_an_unchanged_document_says_so(env: Path) -> None:
    """The common case: nothing changed, so the hash holds and there is nothing to sweep."""
    doc_id, before = _ingest_one(env)
    result = reingest.rerun([doc_id], ["text"])
    assert result.outcomes[0].status == "ok"
    assert "unchanged" in result.outcomes[0].detail
    assert sum(_chunks_for(before).values()) > 0


def test_a_failing_part_does_not_abandon_the_rest_of_the_batch(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 21 runs this over a selection; one unreadable document must not lose the other 39."""
    first, _ = _ingest_one(env, "a.md")
    second, _ = _ingest_one(env, "b.md", _BODY.replace("Background", "Other"))

    def boom(doc: object) -> tuple[str, str]:
        if getattr(doc, "filename", "") == "a.md":
            raise RuntimeError("simulated failure")
        return "ok", "fine"

    monkeypatch.setitem(reingest._RUNNERS, "metadata", boom)
    result = reingest.rerun([first, second], ["metadata"])
    assert [o.status for o in result.outcomes] == ["error", "ok"]
    assert result.errors == 1 and result.ok == 1


def test_progress_reports_every_document_part_pair(env: Path) -> None:
    """The status bar needs a real denominator — `documents x parts`, not documents."""
    first, _ = _ingest_one(env, "a.md")
    second, _ = _ingest_one(env, "b.md", _BODY.replace("Background", "Other"))
    seen: list[tuple[int, int, str | None]] = []
    reingest.rerun(
        [first, second], ["metadata", "references"], on_progress=lambda *a: seen.append(a)
    )
    assert seen[0][1] == 4, "total should be documents x parts"
    assert seen[-1][0] == 4, "the last tick should report everything done"
    assert all(d <= t for d, t, _ in seen)
