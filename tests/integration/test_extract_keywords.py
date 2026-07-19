"""Integration guard tests for keyword extraction (KI-13 vocabulary-seed producer).

Extraction writes `Keyword(source="extracted")` rows linked to documents; idempotent; the
rows then surface through the concept-skeleton `--promote` seam. Temp file-backed SQLite
swapped into the global session (mirrors test_seed_concepts.py). `load_document_texts` is
monkeypatched to a toy corpus so no cache files / Chroma are needed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
import doc_assistant.knowledge.keywords as kw
from doc_assistant.db.models import Base, Document, Keyword, document_keywords
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_skeleton import list_keyword_candidates
from doc_assistant.knowledge.keywords import extract_keywords

# A toy corpus: "colbert"/"hyde" are distinctive (df=1); "retrieval" is ubiquitous (df=3);
# "bm25" is shared mid-frequency (df=2) — the corpus-band target.
CORPUS = {
    "d1": ("colbert.pdf", "colbert late interaction retrieval retrieval ranking bm25"),
    "d2": ("hyde.pdf", "hyde hypothetical document retrieval generation"),
    "d3": ("dpr.pdf", "dense passage retrieval dense encoder bm25"),
}


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    # Seed Document rows keyed by the toy corpus ids.
    with session_scope() as session:
        session.add_all(
            Document(
                id=doc_id,
                filename=fname,
                source_original=fname,
                doc_hash=doc_id,
                format="pdf",
            )
            for doc_id, (fname, _text) in CORPUS.items()
        )
    # Bypass cached-markdown IO with the toy corpus.
    monkeypatch.setattr(
        kw,
        "load_document_texts",
        lambda document_ids=None: [(d, f, t) for d, (f, t) in CORPUS.items()],
    )
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _links() -> int:
    with session_scope() as session:
        return int(
            session.execute(select(func.count()).select_from(document_keywords)).scalar() or 0
        )


def test_dry_run_writes_nothing(env: Path) -> None:
    result = extract_keywords(apply=False, top_k=5, ngram_max=2, min_chars=3)
    assert result.n_documents == 3
    assert result.total_written == 0
    with session_scope() as session:
        assert (session.execute(select(func.count()).select_from(Keyword)).scalar() or 0) == 0


def test_apply_writes_keyword_rows_and_links(env: Path) -> None:
    result = extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    assert result.total_written > 0
    with session_scope() as session:
        names = {k.name for k in session.execute(select(Keyword)).scalars()}
        assert all(k.source == "extracted" for k in session.execute(select(Keyword)).scalars())
    # Distinctive terms surface; the ubiquitous "retrieval" is out-ranked out of top-5.
    assert "colbert" in names
    assert "hyde" in names


def test_apply_is_idempotent(env: Path) -> None:
    extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    links_after_first = _links()
    again = extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    assert again.total_written == 0  # docs already have extracted keywords → skipped
    assert _links() == links_after_first


def test_force_reextracts(env: Path) -> None:
    extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    baseline = _links()
    forced = extract_keywords(apply=True, force=True, top_k=5, ngram_max=2, min_chars=3)
    assert forced.total_written > 0  # cleared + rewritten
    assert _links() == baseline  # same corpus → same link count


def test_extracted_keywords_feed_the_promote_seam(env: Path) -> None:
    extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    candidates = {c.name for c in list_keyword_candidates()}
    assert "colbert" in candidates  # KI-13 loop closed: extractor -> seed_concepts candidate
    assert all(not c.promoted for c in list_keyword_candidates())  # candidate only


def test_force_sweeps_orphaned_keywords_but_keeps_promoted(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_assistant.db.models import Concept

    extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    with session_scope() as session:
        names = {k.name for k in session.execute(select(Keyword)).scalars()}
    assert {"colbert", "hyde"} <= names

    # Promote "colbert" to a curated Concept — the sweep must never delete a promoted form.
    with session_scope() as session:
        session.add(Concept(label="colbert", source="keyword"))

    # A new corpus that drops "colbert" and "hyde": both lose every link on a force re-extract.
    new_corpus = {
        "d1": ("colbert.pdf", "late interaction retrieval ranking bm25"),
        "d2": ("hyde.pdf", "hypothetical document retrieval generation"),
        "d3": ("dpr.pdf", "dense passage retrieval dense encoder bm25"),
    }
    monkeypatch.setattr(
        kw,
        "load_document_texts",
        lambda document_ids=None: [(d, f, t) for d, (f, t) in new_corpus.items()],
    )
    result = extract_keywords(apply=True, force=True, top_k=5, ngram_max=2, min_chars=3)

    assert result.removed_orphans >= 1
    with session_scope() as session:
        remaining = {k.name for k in session.execute(select(Keyword)).scalars()}
    assert "hyde" not in remaining  # orphan + not promoted → swept
    assert "colbert" in remaining  # orphan BUT matches a promoted Concept label → kept


def test_no_sweep_without_force_or_on_single_doc(env: Path) -> None:
    # A plain apply (no force) never sweeps; a --doc run is excluded from the sweep too.
    result = extract_keywords(apply=True, top_k=5, ngram_max=2, min_chars=3)
    assert result.removed_orphans == 0
    single = extract_keywords(
        apply=True, force=True, document_id="d1", top_k=5, ngram_max=2, min_chars=3
    )
    assert single.removed_orphans == 0  # single-document force run is excluded


def test_corpus_band_mode_links_shared_terms_across_docs(env: Path) -> None:
    # max_df = floor(0.7 * 3) = 2, min_df = 2 → the band is exactly df==2.
    extract_keywords(
        apply=True,
        top_k=20,
        ngram_max=1,
        min_chars=3,
        mode="corpus_band",
        min_df=2,
        max_df_frac=0.7,
    )
    with session_scope() as session:
        rows = {k.name: k for k in session.execute(select(Keyword)).scalars()}
        assert "bm25" in rows  # shared mid-frequency term selected
        assert len(rows["bm25"].documents) == 2  # linked to BOTH docs → cross-document edge
        assert "retrieval" not in rows  # df=3 hub excluded (> max_df)
        assert "colbert" not in rows  # df=1 singleton excluded (< min_df)
