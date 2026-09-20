"""Guard tests for the deterministic ``is_a`` proposer (ROADMAP 51, ADR-028 D2/D8).

The rule is lexical, so most of it is testable with no DB at all; the write half needs one (the
seam's cycle + endpoint-kind checks are what make a proposal safe to write unattended). No model,
no network — this pass never had either.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from doc_assistant.db.models import Base, Concept, ConceptHierarchy
from doc_assistant.knowledge.isa_propose import (
    head_suffix_candidates,
    load_concept_labels,
    run_propose_isa,
    write_candidates,
)


@pytest.fixture
def temp_db(monkeypatch):
    """A fresh temp SQLite with the current schema, engine + session factory patched."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    Base.metadata.create_all(engine)
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


def _pairs(candidates) -> set[tuple[str, str]]:
    return {(c.narrow_label, c.broad_label) for c in candidates}


# ============================================================
# The rule
# ============================================================


def test_a_shared_head_makes_the_longer_label_narrower():
    candidates = head_suffix_candidates(
        [
            ("a", "beta oscillations"),
            ("b", "oscillations"),
            ("c", "passage re-ranking"),
            ("d", "re-ranking"),
        ]
    )
    assert _pairs(candidates) == {
        ("beta oscillations", "oscillations"),
        ("passage re-ranking", "re-ranking"),
    }
    assert candidates[0].head == "oscillations"


def test_a_shared_prefix_is_not_a_kind_of():
    """`self-sorting memory` is a kind of memory, not a kind of `self-sorting` — the modifier is
    what narrows, so containment at the front proves nothing (merge baseline 2026-09-17)."""
    candidates = head_suffix_candidates(
        [
            ("a", "self-sorting memory"),
            ("b", "self-sorting"),
            ("c", "saturation index"),
            ("d", "saturation"),
        ]
    )
    assert candidates == []


def test_case_and_hyphens_do_not_hide_a_head():
    candidates = head_suffix_candidates([("a", "Dense Cross-Encoder"), ("b", "cross-encoder")])
    assert _pairs(candidates) == {("Dense Cross-Encoder", "cross-encoder")}


def test_one_token_labels_and_duplicate_labels_propose_nothing():
    """A one-token label has no modifier to drop; two concepts sharing a label have no proper
    suffix between them — that is a merge question (ROADMAP 53), not a hierarchy one."""
    assert head_suffix_candidates([("a", "pose"), ("b", "pose")]) == []
    assert head_suffix_candidates([("a", "retrieval"), ("b", "ranking")]) == []


def test_only_the_nearest_broader_concept_is_proposed():
    """`deep contrastive learning` reaches `learning` through `contrastive learning`, so the
    long way round is left out of the review list — it is redundant, not wrong."""
    candidates = head_suffix_candidates(
        [("a", "deep contrastive learning"), ("b", "contrastive learning"), ("c", "learning")]
    )
    assert _pairs(candidates) == {
        ("deep contrastive learning", "contrastive learning"),
        ("contrastive learning", "learning"),
    }


def test_one_label_gets_one_proposed_parent_but_a_tie_keeps_both():
    """A label's suffixes are nested by construction, so its candidate parents form a chain and
    the nearest one is unique. The exception is two concepts under the *same* label — a merge
    question (ROADMAP 53); both are proposed and the review decides."""
    chain = head_suffix_candidates(
        [("a", "multi-animal pose estimation"), ("b", "pose estimation"), ("c", "estimation")]
    )
    assert _pairs(chain) == {
        ("multi-animal pose estimation", "pose estimation"),
        ("pose estimation", "estimation"),
    }
    tie = head_suffix_candidates([("a", "human pose"), ("b", "pose"), ("c", "Pose")])
    assert {c.broad_id for c in tie if c.narrow_id == "a"} == {"b", "c"}


def test_the_order_is_stable():
    unordered = head_suffix_candidates(
        [("a", "zebra pose"), ("b", "pose"), ("c", "alpha pose"), ("d", "human pose")]
    )
    assert [c.narrow_label for c in unordered] == ["alpha pose", "human pose", "zebra pose"]


# ============================================================
# The write half
# ============================================================


def test_proposals_are_written_as_proposed_and_never_overwrite_curated(temp_db):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge

    with session_scope() as session:
        session.add(Concept(id="n1", label="beta oscillations", kind="concept"))
        session.add(Concept(id="b1", label="oscillations", kind="concept"))
        session.add(Concept(id="n2", label="human pose", kind="concept"))
        session.add(Concept(id="b2", label="pose", kind="concept"))
        session.flush()
        add_hierarchy_edge(session, "n2", "b2", "is_a")  # the user got there first

        written, skipped = write_candidates(
            session, head_suffix_candidates(load_concept_labels(session))
        )
        assert (written, skipped) == (2, 0)
        rows = session.execute(select(ConceptHierarchy)).scalars().all()
        origins = {(r.source_id, r.target_id): r.origin for r in rows}
        assert origins == {("n1", "b1"): "proposed", ("n2", "b2"): "curated"}


def test_a_cycle_is_refused_and_the_rest_of_the_batch_survives(temp_db):
    """`pose estimation` is_a `estimation` and an existing `estimation` is_a `pose estimation`
    cannot both hold. The seam refuses the second; the unrelated candidate still lands."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge

    with session_scope() as session:
        session.add(Concept(id="pe", label="pose estimation", kind="concept"))
        session.add(Concept(id="e", label="estimation", kind="concept"))
        session.add(Concept(id="n", label="nuclear speckles", kind="concept"))
        session.add(Concept(id="s", label="speckles", kind="concept"))
        session.flush()
        add_hierarchy_edge(session, "e", "pe", "is_a")  # broader-than, curated, wrong way round

        written, skipped = write_candidates(
            session, head_suffix_candidates(load_concept_labels(session))
        )
        assert (written, skipped) == (1, 1)
        rows = (
            session.execute(select(ConceptHierarchy).where(ConceptHierarchy.origin == "proposed"))
            .scalars()
            .all()
        )
        assert [(r.source_id, r.target_id) for r in rows] == [("n", "s")]


def test_dry_run_writes_nothing_and_apply_is_idempotent(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(Concept(id="n1", label="beta oscillations", kind="concept"))
        session.add(Concept(id="b1", label="oscillations", kind="concept"))
        session.add(Concept(id="d1", label="Computing", kind="domain"))

    dry = run_propose_isa()
    assert len(dry.candidates) == 1 and dry.applied is False and dry.n_written == 0
    with session_scope() as session:
        assert session.execute(select(ConceptHierarchy)).scalars().all() == []

    first = run_propose_isa(apply=True)
    second = run_propose_isa(apply=True)
    assert (first.n_written, second.n_written) == (1, 1)  # idempotent on the unique key
    with session_scope() as session:
        rows = session.execute(select(ConceptHierarchy)).scalars().all()
        assert len(rows) == 1 and rows[0].type == "is_a" and rows[0].origin == "proposed"


def test_graph_only_narrows_the_vocabulary(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            Concept(id="n1", label="beta oscillations", kind="concept", graph_include=True)
        )
        session.add(Concept(id="b1", label="oscillations", kind="concept", graph_include=False))
        session.flush()
        assert len(load_concept_labels(session)) == 2
        assert load_concept_labels(session, graph_only=True) == [("n1", "beta oscillations")]
