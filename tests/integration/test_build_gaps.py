"""Integration test for the gap-detection build (ADR-004) — the impure orchestration.

Drives ``gaps.build_gaps`` against a temp SQLite (curated vocabulary + persisted
``answer_claims``) and a temp concept-skeleton ``skeleton.json``. Asserts Tier-1 +
Tier-2a-floor rows written, idempotency (replace deterministic rows, not append),
dry-run writes nothing, a missing skeleton raises, and ``--suggest`` (the
not-yet-built stochastic ceiling) raises rather than silently doing nothing.

Deterministic + offline: no network, no real LLM, no Chroma.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant import concept_skeleton as cs
from doc_assistant.db.models import AnswerClaim, AnswerRecord, Base, Concept, ConceptAlias, GapRow
from doc_assistant.db.session import session_scope
from doc_assistant.gaps import build_gaps
from doc_assistant.synthesis import MARKER_OK, MARKER_UNSUPPORTED


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Temp SQLite bound to the global session machinery."""
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


# One isolated concept ("iso"), one sole-source pair ("sole"/"shared"), one
# below-min_degree leaf ("leaf", degree 1 < min_degree=2).
_COOC = frozenset({"cooccurrence"})


def _write_skeleton(root: Path) -> None:
    nodes = [
        cs.ConceptNode("iso", "Iso", ("d1",), 0, -1),
        cs.ConceptNode("sole", "Sole", ("d1",), 0, -1),
        cs.ConceptNode("shared", "Shared", ("d1", "d2"), 0, -1),
        cs.ConceptNode("hub", "Hub", ("d1", "d2", "d3"), 0, -1),
        cs.ConceptNode("leaf", "Leaf", ("d1",), 0, -1),
    ]
    edges = [
        cs.SkeletonEdge("sole", "shared", _COOC, cs.edge_weight(_COOC, 2), 2),
        cs.SkeletonEdge("hub", "leaf", _COOC, cs.edge_weight(_COOC, 2), 2),
    ]
    skeleton = cs.analyze_skeleton(nodes, edges, seed=42)
    root.mkdir(parents=True, exist_ok=True)
    (root / cs.SKELETON_NAME).write_text(
        json.dumps(cs.skeleton_to_dict(skeleton)), encoding="utf-8"
    )


def _seed_curated_concepts() -> None:
    # The curated vocabulary the Tier-2a floor presence-matches claim text against —
    # independent of (but named to line up with) the skeleton fixture's concept ids.
    with session_scope() as session:
        session.add_all(
            [
                Concept(id="iso", label="Iso", source="manual"),
                Concept(id="sole", label="Sole", source="manual"),
                Concept(id="shared", label="Shared", source="manual"),
                Concept(id="hub", label="Hub", source="manual"),
                Concept(id="leaf", label="Leaf", source="manual"),
            ]
        )
        session.add(ConceptAlias(concept_id="shared", alias="shared concept"))


def _seed_claims() -> None:
    with session_scope() as session:
        answer = AnswerRecord(id=str(uuid4()), query="q", answer="a", model_name="m")
        session.add(answer)
        session.flush()
        session.add_all(
            [
                AnswerClaim(
                    answer_record_id=answer.id,
                    claim_index=0,
                    claim_text="Shared is under-discussed.",
                    marker=MARKER_UNSUPPORTED,
                ),
                AnswerClaim(
                    answer_record_id=answer.id,
                    claim_index=1,
                    claim_text="Hub is well documented.",
                    marker=MARKER_OK,  # cited — must not produce a gap
                ),
            ]
        )


def _count_rows() -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(GapRow)).scalar_one())


def test_apply_writes_tier1_and_tier2a_rows(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")

    result = build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    assert result.applied
    kinds = {g.kind for g in result.gaps}
    assert "isolated" in kinds  # iso: degree 0
    assert "single_source" in kinds  # sole: one doc
    assert "under_connected" in kinds  # leaf: degree 1 < 2
    assert "unsourced_claim" in kinds  # the unsupported claim about "shared"

    with session_scope() as session:
        rows = list(session.execute(select(GapRow)).scalars())
    assert len(rows) == len(result.gaps)
    assert all(r.determinism == "deterministic" for r in rows)
    unsourced = [r for r in rows if r.kind == "unsourced_claim"]
    assert unsourced and unsourced[0].concept_id == "shared"


def test_cited_claim_produces_no_unsourced_gap(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")

    result = build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    assert all(g.concept_id != "hub" or g.kind != "unsourced_claim" for g in result.gaps)


def test_idempotent_no_llm(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")

    first = build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    first_count = _count_rows()
    second = build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    assert _count_rows() == first_count  # replaced, not appended
    assert second.graph_version == first.graph_version
    assert len(second.gaps) == len(first.gaps)


def test_dry_run_writes_nothing(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")

    result = build_gaps(apply=False, skeleton_dir=env / "skeleton", min_degree=2)
    assert not result.applied
    assert result.gaps  # computed in-memory...
    assert _count_rows() == 0  # ...but nothing persisted


def test_missing_skeleton_raises(env: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_gaps(apply=True, skeleton_dir=env / "no-skeleton-here", min_degree=2)


def test_suggest_flag_raises_not_implemented(env: Path) -> None:
    _write_skeleton(env / "skeleton")
    with pytest.raises(NotImplementedError):
        build_gaps(apply=False, skeleton_dir=env / "skeleton", min_degree=2, suggest=True)


def test_stochastic_rows_survive_a_deterministic_rebuild(env: Path) -> None:
    # A promoted stochastic suggestion (the deferred Tier-2a ceiling's output shape)
    # must not be wiped by a deterministic rebuild — only deterministic rows replace.
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    with session_scope() as session:
        session.add(
            GapRow(
                concept_id="candidate-concept",
                tier="t2a",
                determinism="stochastic",
                kind="suggested_concept",
                status="promoted",
            )
        )
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    with session_scope() as session:
        stochastic = list(
            session.execute(select(GapRow).where(GapRow.determinism == "stochastic")).scalars()
        )
    assert len(stochastic) == 1
    assert stochastic[0].status == "promoted"


def test_no_document_or_concept_mutation(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    with session_scope() as session:
        concept_count = session.execute(select(func.count()).select_from(Concept)).scalar_one()
    assert concept_count == 5  # the sidecar never touches the curated vocabulary
