"""E0.1 / KI-20 / review CS-5 — curation demotes non-concepts, it does not delete them.

``concept_curation.apply_plan`` is the seam the ``curate_concepts`` runner drives on ``--apply``.
Its artifact + ``classify_noise`` stages used to hard-delete the ``Concept`` rows they flagged —
which also deletes the row's ADR-015 **keyword family** and cascades presence/edges/gaps, against
ADR-018's *demote* verb. Because ``classify_noise`` is exactly the stage that mislabels real
specialist vocabulary (``cre``/``dbs``/``ntsr1``/``pddl``), a false positive there was
irrecoverable data loss.

These guards pin the fix: a flagged concept keeps its row + aliases (its family) and is merely
excluded from the graph (``graph_include=False``). Non-vacuous — point ``apply_plan`` at the old
``remove_concepts`` and ``test_noise_verdict_demotes_and_keeps_the_family`` fails (row is gone).
The reserved explicit-deletion primitive (``remove_concepts``) is covered too, so the distinction
is a tested contract, not a comment.

Deterministic + offline: no network, no real LLM.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import Base, Concept, ConceptAlias
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_curation import (
    CurationPlan,
    apply_plan,
    demote_concepts,
    remove_concepts,
)


@pytest.fixture
def env(tmp_path: Path) -> Iterator[Path]:
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


def _seed_family(concept_id: str, label: str, aliases: list[str]) -> None:
    """A curated graph concept that is also an ADR-015 keyword family (label + aliases)."""
    with session_scope() as session:
        session.add(Concept(id=concept_id, label=label, source="manual", graph_include=True))
        session.add_all(ConceptAlias(concept_id=concept_id, alias=a) for a in aliases)


def _get(concept_id: str) -> Concept | None:
    with session_scope() as session:
        return session.get(Concept, concept_id)


def _alias_count(concept_id: str) -> int:
    with session_scope() as session:
        return int(
            session.execute(
                select(func.count())
                .select_from(ConceptAlias)
                .where(ConceptAlias.concept_id == concept_id)
            ).scalar_one()
        )


def test_noise_verdict_demotes_and_keeps_the_family(env: Path) -> None:
    # `cre` is a real specialist term the LLM classifier is known to mislabel as noise.
    _seed_family("cre-id", "cre", ["cre recombinase", "cre-lox"])
    plan = CurationPlan(llm_noise=[("cre-id", "cre")])

    demoted, merged = apply_plan(plan)
    assert (demoted, merged) == (1, 0)

    row = _get("cre-id")
    assert row is not None  # the row survived (a delete would have removed it — fails today)
    assert row.graph_include is False  # ...but it is out of the graph vocabulary (ADR-018)
    assert _alias_count("cre-id") == 2  # its keyword family is intact


def test_artifact_verdict_demotes_not_deletes(env: Path) -> None:
    _seed_family("art-id", "2015 volume", [])
    plan = CurationPlan(artifacts=[("art-id", "2015 volume")])

    demote_concepts(plan.demote_ids)

    row = _get("art-id")
    assert row is not None and row.graph_include is False


def test_demote_is_idempotent_and_ignores_unknown_ids(env: Path) -> None:
    _seed_family("c1", "keep", [])
    assert demote_concepts({"c1", "ghost"}) == 1  # only the row that exists is counted
    assert demote_concepts({"c1"}) == 1  # a second run is a no-op on the flag, still counted
    assert _get("c1").graph_include is False  # type: ignore[union-attr]


def test_remove_concepts_is_the_reserved_hard_delete(env: Path) -> None:
    # The explicit-deletion primitive still deletes — reserved for a separately-confirmed purge,
    # never wired to the noise classifier. This is the behaviour demote_concepts replaced.
    _seed_family("gone-id", "purge me", ["alias-a"])
    assert remove_concepts({"gone-id"}) == 1
    assert _get("gone-id") is None
    assert _alias_count("gone-id") == 0  # aliases cascade with the row


def test_empty_plan_writes_nothing(env: Path) -> None:
    _seed_family("c1", "keep", ["a"])
    assert apply_plan(CurationPlan()) == (0, 0)
    row = _get("c1")
    assert row is not None and row.graph_include is True  # untouched
