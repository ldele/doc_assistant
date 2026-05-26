"""End-to-end Phase 4 pipeline against a temp SQLite DB.

For each scenario:
  1. Generate synthetic markdown via tests.fixtures.synthetic_corpus.
  2. Run metadata extractor over each paper -> populate Documents.
  3. Run citation extractor + matcher over each paper -> populate Citations.
  4. Assert the resulting graph topology matches the planned `cites` field.

This exercises the matcher in conditions the natural corpus can't: chains,
cycles, isolated nodes, and the four reference-format variants. Catches
regressions that the unit tests would miss because they don't run the
DOI / author+year / fuzzy-title fallback chain against a real DB.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

# We rebind the session module's engine before importing anything that
# touches `session_scope()`, so each test gets its own isolated SQLite.
import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Citation, Document
from doc_assistant.metadata_extractor import extract_metadata
from tests.fixtures.synthetic_corpus import (
    FakePaper,
    chain_scenario,
    cycle_scenario,
    isolated_scenario,
    mixed_format_scenario,
    render_corpus,
)

# ============================================================
# Per-test temp DB
# ============================================================


@pytest.fixture
def temp_db(tmp_path: Path) -> Iterator[Path]:
    """Bind the global session machinery to a fresh SQLite file."""
    db_path = tmp_path / "library.db"
    url = f"sqlite:///{db_path}"
    engine = create_engine(url, echo=False, future=True)
    Base.metadata.create_all(engine)

    orig_engine = session_mod._engine
    orig_factory = session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True
    )
    try:
        yield db_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


# ============================================================
# Pipeline runner
# ============================================================


def _run_pipeline(papers: list[FakePaper]) -> dict[str, str]:
    """Populate the bound DB with Documents + Citations. Returns paper_id -> doc_id map."""
    # Import inside to ensure the module picks up the rebound session
    from doc_assistant.citations import extract_from_markdown, match_to_library
    from doc_assistant.db.session import session_scope

    corpus = render_corpus(papers)
    paper_id_to_doc_id: dict[str, str] = {}

    # 1. Insert Documents + extract metadata
    with session_scope() as session:
        for p in papers:
            filename = f"{p.paper_id}.pdf"
            md = corpus[f"{p.paper_id}.md"]
            meta = extract_metadata(md, filename=filename)
            doc = Document(
                filename=filename,
                source_original=f"/synthetic/{filename}",
                source_cache=f"/synthetic/cache/{p.paper_id}.md",
                doc_hash=f"hash_{p.paper_id[:12]}",  # synthetic, length doesn't matter for tests
                format="pdf",
                title=meta.title,
                authors=meta.authors,
                year=meta.year,
                doi=meta.doi,
                extraction_health="healthy",
            )
            session.add(doc)
            session.flush()
            paper_id_to_doc_id[p.paper_id] = doc.id

    # 2. Extract + match + persist Citations
    for p in papers:
        doc_id = paper_id_to_doc_id[p.paper_id]
        md = corpus[f"{p.paper_id}.md"]
        result = extract_from_markdown(doc_id, md)
        with session_scope() as session:
            for c in result.citations:
                target_id = match_to_library(c)
                session.add(
                    Citation(
                        source_document_id=doc_id,
                        target_document_id=target_id,
                        raw_citation_text=c.raw_text,
                        target_doi=c.doi,
                        target_title=c.title,
                        target_authors=c.authors,
                        target_year=c.year,
                        extraction_method=c.extraction_method,
                        confidence=c.confidence,
                    )
                )

    return paper_id_to_doc_id


def _internal_edges(paper_id_to_doc_id: dict[str, str]) -> set[tuple[str, str]]:
    """Read back resolved internal citation edges as (src_paper_id, tgt_paper_id)."""
    from doc_assistant.db.session import session_scope

    doc_id_to_paper = {v: k for k, v in paper_id_to_doc_id.items()}
    with session_scope() as session:
        stmt = select(Citation.source_document_id, Citation.target_document_id).where(
            Citation.target_document_id.is_not(None)
        )
        return {
            (doc_id_to_paper[src], doc_id_to_paper[tgt])
            for src, tgt in session.execute(stmt).all()
        }


def _planned_edges(papers: list[FakePaper]) -> set[tuple[str, str]]:
    """Edges declared in the FakePaper specs."""
    return {(p.paper_id, cited) for p in papers for cited in p.cites}


# ============================================================
# Scenarios
# ============================================================


def test_chain_scenario_all_edges_resolved(temp_db: Path) -> None:
    """A -> B -> C -> D. Every edge should resolve via DOI."""
    papers = chain_scenario()
    mapping = _run_pipeline(papers)
    actual = _internal_edges(mapping)
    expected = _planned_edges(papers)
    assert actual == expected, f"missing: {expected - actual}, extra: {actual - expected}"


def test_cycle_scenario_bidirectional_and_dedup(temp_db: Path) -> None:
    """X <-> Y plus Z -> {X, Y}. Two cycle edges + two outbound from Z."""
    papers = cycle_scenario()
    mapping = _run_pipeline(papers)
    actual = _internal_edges(mapping)
    expected = _planned_edges(papers)
    assert actual == expected


def test_isolated_scenario_no_edges(temp_db: Path) -> None:
    """Single paper with no cites and no citers."""
    papers = isolated_scenario()
    _run_pipeline(papers)

    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        n_docs = session.execute(select(func.count()).select_from(Document)).scalar_one()
        n_cites = session.execute(select(func.count()).select_from(Citation)).scalar_one()
    assert n_docs == 1
    assert n_cites == 0  # nothing to extract; nothing matched


def test_mixed_format_clean_and_multicolumn_resolve(temp_db: Path) -> None:
    """All four format variants cite paper_a. Clean + multi-column should resolve via DOI."""
    papers = mixed_format_scenario()
    mapping = _run_pipeline(papers)
    actual = _internal_edges(mapping)

    # Clean and multi-column include the DOI in refs -> should match.
    assert ("paper_clean", "paper_a") in actual
    assert ("paper_mcol", "paper_a") in actual


def test_mixed_format_no_heading_yields_zero_edges(temp_db: Path) -> None:
    """Tier-1 cannot detect a References section without the heading."""
    papers = mixed_format_scenario()
    mapping = _run_pipeline(papers)
    actual = _internal_edges(mapping)

    # paper_nohdr cites paper_a but the refs section is undetectable.
    assert ("paper_nohdr", "paper_a") not in actual


def test_mixed_format_lncs_resolves_via_author_year(temp_db: Path) -> None:
    """LNCS format strips DOI from refs -> matcher must fall back to author+year."""
    papers = mixed_format_scenario()
    mapping = _run_pipeline(papers)
    actual = _internal_edges(mapping)

    # paper_lncs cites paper_a (Origin, O., 2005). DOI not in LNCS refs.
    # author+year fallback in match_to_library should still resolve it.
    assert ("paper_lncs", "paper_a") in actual


# ============================================================
# Empty-DB sanity
# ============================================================


def test_match_to_library_against_empty_db_returns_none(temp_db: Path) -> None:
    """Matcher must not blow up when there's nothing in the library."""
    from doc_assistant.citations import ParsedCitation, match_to_library

    c = ParsedCitation(raw_text="some text", doi="10.0/x", title="t", authors="Foo", year=2020)
    assert match_to_library(c) is None
