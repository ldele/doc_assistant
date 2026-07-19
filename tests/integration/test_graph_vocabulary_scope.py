"""Guard tests for the ADR-018 graph-vocabulary scope.

Tag families (ADR-015) and concept-graph nodes are the same ``Concept`` rows. The graph scopes
itself to ``graph_include`` rows; families deliberately stay unfiltered. The load-bearing test
here is that asymmetry — scoping the graph must not shrink the Manage-keywords view, which is
the regression that made deleting the bulk-promoted concepts unavailable as a fix.

Temp file-backed SQLite swapped into the global session (mirrors ``test_seed_concepts.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Concept, Keyword
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_skeleton import (
    add_concept,
    backfill_graph_include,
    load_concepts,
    promote_keyword,
    set_graph_include,
)
from doc_assistant.library import create_keyword_family, list_keyword_families


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


def _labels() -> set[str]:
    concepts, _ = load_concepts()
    return {label for _, label in concepts}


def _set_null(label: str) -> None:
    """Force a row to the pre-migration state (flag IS NULL)."""
    with session_scope() as session:
        row = session.execute(select(Concept).where(Concept.label == label)).scalar_one()
        row.graph_include = None


# --- the filter itself -------------------------------------------------------------------


def test_load_concepts_returns_only_included(env: Path) -> None:
    add_concept("BM25")  # defaults to included
    add_concept("speckles", graph_include=False)
    assert _labels() == {"BM25"}


def test_null_flag_reads_as_excluded(env: Path) -> None:
    """Every row predating the migration lands NULL — opt-in means NULL stays out."""
    add_concept("BM25")
    _set_null("BM25")
    assert _labels() == set()


def test_aliases_only_for_included_concepts(env: Path) -> None:
    """An excluded concept contributes no surface form, or it would still match text."""
    add_concept("BM25", aliases=["Okapi BM25"])
    add_concept("speckles", aliases=["speckle"], graph_include=False)
    _, aliases = load_concepts()
    assert [a for group in aliases.values() for a in group] == ["Okapi BM25"]


def test_empty_vocabulary_is_an_empty_graph_not_an_error(env: Path) -> None:
    add_concept("speckles", graph_include=False)
    assert load_concepts() == ([], {})


# --- which creation path opts in (the ADR-018 table) -------------------------------------


def test_add_concept_defaults_to_included(env: Path) -> None:
    add_concept("dense retrieval")
    assert _labels() == {"dense retrieval"}


def test_promote_keyword_is_excluded(env: Path) -> None:
    """A promotion is a *candidate* by contract — and the path --promote-all drives."""
    with session_scope() as session:
        session.add(Keyword(name="hyaline", source="extracted"))
    assert promote_keyword("hyaline") is not None
    assert _labels() == set()


def test_keyword_family_is_excluded(env: Path) -> None:
    """Grouping keywords is library organisation, not a claim about the map."""
    create_keyword_family("retrieval")
    assert _labels() == set()


def test_bulk_promotion_cannot_reflood_the_graph(env: Path) -> None:
    """The 2026-07-05 regression, as a test: promote everything, graph stays curated."""
    with session_scope() as session:
        session.add_all(
            Keyword(name=n, source="extracted") for n in ("speckles", "hyaline", "unlabeled")
        )
    add_concept("BM25")
    for name in ("speckles", "hyaline", "unlabeled"):
        promote_keyword(name)
    assert _labels() == {"BM25"}


# --- families stay whole (the load-bearing guard) -----------------------------------------


def test_scoping_the_graph_does_not_shrink_keyword_families(env: Path) -> None:
    """ADR-018's whole premise: both features keep the rows they need."""
    add_concept("BM25")
    create_keyword_family("retrieval")
    with session_scope() as session:
        session.add(Keyword(name="hyaline", source="extracted"))
    promote_keyword("hyaline")

    assert {f.canonical for f in list_keyword_families()} == {"BM25", "retrieval", "hyaline"}
    assert _labels() == {"BM25"}


# --- the write surface --------------------------------------------------------------------


def test_set_graph_include_round_trip(env: Path) -> None:
    concept_id = add_concept("speckles", graph_include=False)
    assert set_graph_include(concept_id, True) is True
    assert _labels() == {"speckles"}
    assert set_graph_include(concept_id, False) is True
    assert _labels() == set()


def test_set_graph_include_unknown_concept(env: Path) -> None:
    assert set_graph_include("no-such-id", True) is False


# --- the backfill -------------------------------------------------------------------------


def test_backfill_splits_by_source(env: Path) -> None:
    add_concept("BM25")  # source="manual"
    with session_scope() as session:
        session.add(Keyword(name="hyaline", source="extracted"))
    promote_keyword("hyaline")  # source="keyword"
    _set_null("BM25")
    _set_null("hyaline")

    assert backfill_graph_include(apply=True) == (1, 1)
    assert _labels() == {"BM25"}


def test_backfill_is_dry_run_by_default(env: Path) -> None:
    add_concept("BM25")
    _set_null("BM25")
    assert backfill_graph_include() == (1, 0)
    assert _labels() == set()  # nothing written


def test_backfill_is_idempotent_and_preserves_overrides(env: Path) -> None:
    """Only NULL rows are touched, so a user's opt-in survives a second run."""
    with session_scope() as session:
        session.add(Keyword(name="hyaline", source="extracted"))
    concept_id = promote_keyword("hyaline")
    assert concept_id is not None
    _set_null("hyaline")
    backfill_graph_include(apply=True)  # -> excluded (source="keyword")

    set_graph_include(concept_id, True)  # the user disagrees
    assert backfill_graph_include(apply=True) == (0, 0)  # no NULL rows left
    assert _labels() == {"hyaline"}  # the override survived


# --- stage-0 candidate ranking (impure wrapper) --------------------------------------------


def test_rank_keyword_candidates_reads_reach_vocabulary_and_authors(env: Path) -> None:
    """The SQL half of stage 0: document reach, promotion state, and the authors harvest."""
    from doc_assistant.db.models import Document, Keyword, document_keywords
    from doc_assistant.knowledge.concept_curation import rank_keyword_candidates

    with session_scope() as session:
        for i in (1, 2):
            session.add(
                Document(
                    id=f"d{i}",
                    filename=f"d{i}.pdf",
                    source_original=f"/tmp/d{i}.pdf",
                    doc_hash=f"h{i}",
                    format="pdf",
                    authors="Ziyang Wang and Jane Smith",
                )
            )
        session.add_all(
            [
                Keyword(id=1, name="mamba", source="extracted"),
                Keyword(id=2, name="ziyang wang", source="extracted"),
                Keyword(id=3, name="18653 v1", source="extracted"),
            ]
        )
    with session_scope() as session:
        session.execute(
            document_keywords.insert(),
            [
                {"document_id": "d1", "keyword_id": 1},
                {"document_id": "d2", "keyword_id": 1},  # mamba spans 2 docs
                {"document_id": "d1", "keyword_id": 2},
                {"document_id": "d1", "keyword_id": 3},
            ],
        )
    add_concept("mamba")  # promoted AND in the graph

    ranked = {c.name: c for c in rank_keyword_candidates()}
    assert next(c.name for c in rank_keyword_candidates()) == "mamba"  # highest reach leads
    assert ranked["mamba"].doc_count == 2
    assert (ranked["mamba"].promoted, ranked["mamba"].in_graph) == (True, True)
    assert ranked["ziyang wang"].author_like is True  # harvested from documents.authors
    assert ranked["18653 v1"].artifact is True
    assert ranked["18653 v1"].author_like is False
