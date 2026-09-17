"""ROADMAP 53 (1) — a concept merge keeps what the user curated, and can be undone.

``apply_merges`` used to fold aliases and delete the dropped ``Concept``. The delete cascades:
``concept_hierarchy`` is ``ON DELETE CASCADE``, so the dropped concept's taxonomy placements
vanished, and ``gap_triage`` (no foreign key) was left pointing at nothing. These tests pin the
fix against a real SQLite with foreign keys on (``db/session.py`` enables them per connection),
so the cascade the old code relied on actually fires here. Non-vacuous (checked 2026-09-17):
patch ``concept_merge._repoint`` to plan no moves — the old fold-and-delete — and the placement
assertions in ``test_a_merge_moves_placements_to_the_survivor`` and
``test_undo_splits_the_merge_back_exactly`` both fail.

Deterministic + offline: no embedder, no LLM — plans are built by hand.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import (
    Base,
    Concept,
    ConceptAlias,
    ConceptHierarchy,
    ConceptMerge,
    GapRow,
    GapTriage,
)
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_curation import CurationPlan, MergePlan, apply_plan
from doc_assistant.knowledge.concept_merge import (
    MergeUndoError,
    apply_merges,
    list_merges,
    undo_merge,
)
from doc_assistant.knowledge.taxonomy import add_hierarchy_edge


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


def _concept(
    cid: str,
    label: str,
    *,
    aliases: tuple[str, ...] = (),
    kind: str = "concept",
    graph_include: bool = False,
    definition: str | None = None,
) -> None:
    with session_scope() as s:
        s.add(
            Concept(
                id=cid,
                label=label,
                kind=kind,
                graph_include=graph_include,
                definition=definition,
                aliases=[ConceptAlias(alias=a) for a in aliases],
            )
        )


def _edge(source: str, target: str, edge_type: str = "in_field", origin: str = "curated") -> None:
    with session_scope() as s:
        add_hierarchy_edge(s, source, target, edge_type, origin=origin)


def _edges() -> set[tuple[str, str, str, str]]:
    with session_scope() as s:
        return {
            (r.source_id, r.target_id, r.type, r.origin)
            for r in s.execute(select(ConceptHierarchy)).scalars()
        }


def _aliases(cid: str) -> set[str]:
    with session_scope() as s:
        return set(
            s.execute(select(ConceptAlias.alias).where(ConceptAlias.concept_id == cid)).scalars()
        )


def _triage() -> set[tuple[str, str, str]]:
    with session_scope() as s:
        return {(r.concept_id, r.kind, r.status) for r in s.execute(select(GapTriage)).scalars()}


def _plan(keep: str, drop: str) -> MergePlan:
    return MergePlan(keep_id=keep, keep_label=keep, drop_id=drop, drop_label=drop)


def _two_concepts_in_a_taxonomy() -> None:
    """``text embedding`` (kept) and ``text embeddings`` (dropped), each placed somewhere."""
    _concept("field-ml", "Machine learning", kind="domain")
    _concept("field-ir", "Information retrieval", kind="domain")
    _concept("broad", "representation learning", graph_include=True)
    _concept("narrow", "sentence embedding")
    _concept("keep", "text embedding", aliases=("text embed",), graph_include=False)
    _concept(
        "drop",
        "text embeddings",
        aliases=("embeddings of text",),
        graph_include=True,
        definition="Vectors that stand for a passage.",
    )
    _edge("keep", "field-ml")  # the survivor's own placement
    _edge("drop", "field-ir", origin="proposed")  # the dropped concept's placements
    _edge("drop", "broad", "is_a")
    _edge("narrow", "drop", "is_a")  # something placed *under* the dropped concept


def test_a_merge_moves_placements_to_the_survivor(env: Path) -> None:
    _two_concepts_in_a_taxonomy()

    outcome = apply_merges([_plan("keep", "drop")])

    assert outcome.n_merged == 1 and outcome.skipped == ()
    assert _edges() == {
        ("keep", "field-ml", "in_field", "curated"),
        ("keep", "field-ir", "in_field", "proposed"),  # origin travels with the placement
        ("keep", "broad", "is_a", "curated"),
        ("narrow", "keep", "is_a", "curated"),
    }
    with session_scope() as s:
        assert s.get(Concept, "drop") is None
        keep = s.get(Concept, "keep")
        assert keep is not None
        assert keep.graph_include is True  # the dropped one was on the graph
        assert keep.definition == "Vectors that stand for a passage."
    assert _aliases("keep") == {"text embed", "text embeddings", "embeddings of text"}


def test_an_edge_between_the_two_concepts_is_dropped_not_made_a_self_edge(env: Path) -> None:
    _concept("keep", "dense retrieval")
    _concept("drop", "dense passage retrieval")
    _edge("drop", "keep", "is_a")

    assert apply_merges([_plan("keep", "drop")]).n_merged == 1
    assert _edges() == set()


def test_a_merge_that_would_close_a_cycle_is_refused_and_writes_nothing(env: Path) -> None:
    # keep is_a mid, mid is_a drop: moving drop's incoming edge to keep would make keep is_a keep.
    _concept("keep", "a")
    _concept("mid", "b")
    _concept("drop", "c", aliases=("c-alias",))
    _edge("keep", "mid", "is_a")
    _edge("mid", "drop", "is_a")
    before = _edges()

    outcome = apply_merges([_plan("keep", "drop")])

    assert outcome.n_merged == 0
    assert [reason for _plan_, reason in outcome.skipped] == [
        "moving its taxonomy placements would close a cycle"
    ]
    assert _edges() == before
    assert _aliases("drop") == {"c-alias"} and _aliases("keep") == set()
    assert list_merges() == []


def test_a_field_node_is_never_merged(env: Path) -> None:
    _concept("keep", "psychology")
    _concept("field", "Psychology", kind="domain")

    outcome = apply_merges([_plan("keep", "field")])

    assert outcome.n_merged == 0 and len(outcome.skipped) == 1
    with session_scope() as s:
        assert s.get(Concept, "field") is not None


def test_triage_follows_the_concept_and_the_survivors_own_verdict_wins(env: Path) -> None:
    _concept("keep", "bm25")
    _concept("drop", "okapi bm25")
    with session_scope() as s:
        s.add(GapTriage(concept_id="drop", kind="single_source", status="dismissed"))
        s.add(GapTriage(concept_id="drop", kind="isolated", status="dismissed"))
        s.add(GapTriage(concept_id="keep", kind="isolated", status="promoted"))
        s.add(
            GapRow(
                id="suggestion",
                concept_id="drop",
                tier="t2a",
                determinism="stochastic",
                kind="suggested_concept",
                status="promoted",
            )
        )

    apply_merges([_plan("keep", "drop")])

    assert _triage() == {
        ("keep", "single_source", "dismissed"),  # moved
        ("keep", "isolated", "promoted"),  # the survivor's own verdict, not overwritten
    }
    with session_scope() as s:
        assert s.get(GapRow, "suggestion").concept_id == "keep"  # type: ignore[union-attr]


def test_undo_splits_the_merge_back_exactly(env: Path) -> None:
    _two_concepts_in_a_taxonomy()
    with session_scope() as s:
        s.add(GapTriage(concept_id="drop", kind="single_source", status="dismissed"))
    edges_before, triage_before = _edges(), _triage()
    keep_aliases_before, drop_aliases_before = _aliases("keep"), _aliases("drop")

    (merge_id,) = apply_merges([_plan("keep", "drop")]).merged
    summary = undo_merge(merge_id)

    assert summary.undone_at is not None
    assert _edges() == edges_before
    assert _triage() == triage_before
    assert _aliases("keep") == keep_aliases_before
    assert _aliases("drop") == drop_aliases_before
    with session_scope() as s:
        keep, drop = s.get(Concept, "keep"), s.get(Concept, "drop")
        assert keep is not None and drop is not None
        assert (keep.graph_include, keep.definition) == (False, None)
        assert (drop.label, drop.graph_include) == ("text embeddings", True)
        assert drop.definition == "Vectors that stand for a passage."

    with pytest.raises(MergeUndoError, match="already undone"):
        undo_merge(merge_id)


def test_the_record_says_what_moved(env: Path) -> None:
    _two_concepts_in_a_taxonomy()

    (merge_id,) = apply_merges([_plan("keep", "drop")]).merged

    (summary,) = list_merges()
    assert summary.id == merge_id and summary.undone_at is None
    assert (summary.n_placements, summary.n_aliases_added) == (3, 2)
    with session_scope() as s:
        record = s.get(ConceptMerge, merge_id)
        assert record is not None
        data = json.loads(record.record_json)
    assert data["dropped"]["label"] == "text embeddings"


def test_undo_refuses_when_the_survivor_is_gone(env: Path) -> None:
    _concept("keep", "a")
    _concept("drop", "b")
    (merge_id,) = apply_merges([_plan("keep", "drop")]).merged
    with session_scope() as s:
        s.delete(s.get(Concept, "keep"))

    with pytest.raises(MergeUndoError, match="no longer exists"):
        undo_merge(merge_id)


def test_apply_plan_reports_merges_through_the_outcome(env: Path) -> None:
    _concept("keep", "a", graph_include=True)
    _concept("drop", "b")
    _concept("noise", "2015 volume", graph_include=True)

    demoted, outcome = apply_plan(
        CurationPlan(artifacts=[("noise", "2015 volume")], merges=[_plan("keep", "drop")])
    )

    assert (demoted, outcome.n_merged) == (1, 1)
