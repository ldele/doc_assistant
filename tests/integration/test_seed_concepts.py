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
from doc_assistant.db.models import Base, Concept, ConceptAlias, Keyword
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_skeleton import (
    add_concept,
    list_keyword_candidates,
    load_glossary,
    promote_keyword,
)


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


def test_add_concept_creates_curated_glossary_entry(env: Path) -> None:
    add_concept("BM25", definition="A sparse lexical ranker.", aliases=["Okapi BM25"])
    with session_scope() as session:
        c = session.execute(select(Concept)).scalar_one()
        assert c.label == "BM25"
        assert c.source == "manual"  # directly curated, not promoted from a keyword
        assert c.definition == "A sparse lexical ranker."
        aliases = {a.alias for a in session.execute(select(ConceptAlias)).scalars()}
    # The label is an implicit surface form; only extra synonyms are stored as aliases.
    assert aliases == {"Okapi BM25"}


def test_add_concept_idempotent_updates_definition_and_unions_aliases(env: Path) -> None:
    first = add_concept("dense retrieval", aliases=["DPR"])
    second = add_concept(
        "dense retrieval",
        definition="Embedding-based retrieval.",
        aliases=["dense passage retrieval"],
    )
    assert first == second  # get-or-create by label
    assert _count(Concept) == 1
    with session_scope() as session:
        c = session.execute(select(Concept)).scalar_one()
        assert c.definition == "Embedding-based retrieval."  # filled on re-add
        aliases = {a.alias for a in session.execute(select(ConceptAlias)).scalars()}
    assert aliases == {"DPR", "dense passage retrieval"}  # union, never removed


def test_load_glossary_returns_sorted_entries(env: Path) -> None:
    add_concept("re-ranking", definition="Reorder candidates.", aliases=["reranking"])
    add_concept("BM25")
    glossary = load_glossary()
    assert [e.label for e in glossary] == ["BM25", "re-ranking"]  # sorted by label (casefold)
    assert glossary[0].definition is None
    rr = glossary[1]
    assert rr.definition == "Reorder candidates."
    assert rr.aliases == ["reranking"]
