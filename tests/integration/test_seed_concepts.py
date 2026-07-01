"""Integration guard tests for vocabulary seeding (concept-graph redesign Decision 1).

A Keyword is a candidate only — never auto-written as a Concept; only an explicit promote
writes a Concept + seed alias. Temp file-backed SQLite swapped into the global session.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.concept_skeleton import list_keyword_candidates, promote_keyword
from doc_assistant.db.models import Base, Concept, ConceptAlias, Keyword
from doc_assistant.db.session import session_scope


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Path]:
    engine = create_engine(f"sqlite:///{tmp_path / 'library.db'}", echo=False, future=True)
    Base.metadata.create_all(engine)
    orig_engine, orig_factory = session_mod._engine, session_mod._SessionLocal
    session_mod._engine = engine
    session_mod._SessionLocal = sessionmaker(
        bind=engine, autoflush=False, autocommit=False, future=True, expire_on_commit=False
    )
    try:
        yield tmp_path
    finally:
        session_mod._engine = orig_engine
        session_mod._SessionLocal = orig_factory
        engine.dispose()


def _seed_keywords(*names: str) -> None:
    with session_scope() as session:
        session.add_all(Keyword(name=name, source="extracted") for name in names)


def _count(model: type) -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(model)).scalar() or 0)


def test_keywords_surface_as_candidates_no_concepts_written(env: Path) -> None:
    _seed_keywords("RAG", "BM25")
    candidates = list_keyword_candidates()
    assert {c.name for c in candidates} == {"RAG", "BM25"}
    assert all(not c.promoted for c in candidates)
    # Listing candidates must NOT write any Concept (Decision 1).
    assert _count(Concept) == 0


def test_promote_writes_concept_and_seed_alias(env: Path) -> None:
    _seed_keywords("RAG", "BM25")
    concept_id = promote_keyword("RAG")
    assert concept_id is not None

    with session_scope() as session:
        concepts = list(session.execute(select(Concept)).scalars())
        aliases = list(session.execute(select(ConceptAlias)).scalars())
    assert {c.label for c in concepts} == {"RAG"}  # only the promoted one
    assert concepts[0].source == "keyword"
    assert [a.alias for a in aliases] == ["RAG"]  # seed alias == the surface form
    # The un-promoted keyword leaves no Concept.
    assert _count(Concept) == 1


def test_promote_is_idempotent(env: Path) -> None:
    _seed_keywords("RAG")
    first = promote_keyword("RAG")
    second = promote_keyword("RAG")
    assert first == second  # get-or-create by label
    assert _count(Concept) == 1
    assert _count(ConceptAlias) == 1


def test_promote_unknown_keyword_is_noop(env: Path) -> None:
    _seed_keywords("RAG")
    assert promote_keyword("does-not-exist") is None
    assert _count(Concept) == 0


def test_candidate_marks_promoted_after_promote(env: Path) -> None:
    _seed_keywords("RAG", "BM25")
    promote_keyword("RAG")
    by_name = {c.name: c for c in list_keyword_candidates()}
    assert by_name["RAG"].promoted is True
    assert by_name["BM25"].promoted is False
