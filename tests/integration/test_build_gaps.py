"""Integration test for the gap-detection build (ADR-004) — the impure orchestration.

Drives ``gaps.build_gaps`` against a temp SQLite (curated vocabulary + persisted
``answer_claims``) and a temp concept-skeleton ``skeleton.json``. Asserts Tier-1 +
Tier-2a-floor rows written, idempotency (replace deterministic rows, not append),
dry-run writes nothing, a missing skeleton raises, and the Tier-2a stochastic
ceiling (``--suggest``) writes/upserts stochastic rows via a scripted ``LLMClient``
— zero calls without ``--apply``, a promoted row survives a re-suggest.

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
from doc_assistant.db.models import AnswerClaim, AnswerRecord, Base, Concept, ConceptAlias, GapRow
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge import concept_skeleton as cs
from doc_assistant.knowledge.gaps import build_gaps
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
                Concept(id="iso", label="Iso", source="manual", graph_include=True),
                Concept(id="sole", label="Sole", source="manual", graph_include=True),
                Concept(id="shared", label="Shared", source="manual", graph_include=True),
                Concept(id="hub", label="Hub", source="manual", graph_include=True),
                Concept(id="leaf", label="Leaf", source="manual", graph_include=True),
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


def test_suggest_without_apply_makes_zero_calls_and_writes_nothing(env: Path) -> None:
    # Dry run + --suggest: the contract every other enrichment CLI shares — no LLM call,
    # nothing persisted, even though suggest=True.
    _write_skeleton(env / "skeleton")

    class _ExplodingClient:
        def complete(self, messages: object, *, temperature: float, max_tokens: int) -> str:
            raise AssertionError("must not be called on a dry run")

    result = build_gaps(
        apply=False,
        skeleton_dir=env / "skeleton",
        min_degree=2,
        suggest=True,
        client=_ExplodingClient(),
    )
    assert result.n_suggested == 0
    assert _count_rows() == 0


def test_suggest_with_apply_requires_a_client(env: Path) -> None:
    _write_skeleton(env / "skeleton")
    with pytest.raises(ValueError, match="requires an LLMClient"):
        build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2, suggest=True)


class _FakeSuggestClient:
    """Returns the same canned suggestion for every ``under_connected`` concept asked about."""

    def __init__(
        self, response: str = '{"kind": "thin_area", "target": "fusion", "rating": 0.6}'
    ) -> None:
        self._response = response
        self.n_calls = 0

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str:
        self.n_calls += 1
        return self._response


def test_suggest_on_writes_stochastic_without_a_paid_call(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    client = _FakeSuggestClient()

    result = build_gaps(
        apply=True, skeleton_dir=env / "skeleton", min_degree=2, suggest=True, client=client
    )

    # Fixture edges (sole-shared, hub-leaf) give every non-isolated node degree 1 < min_degree=2,
    # so all four ("hub", "leaf", "shared", "sole") route through the ceiling; "iso" (degree 0)
    # is `isolated`, not `under_connected`, and is excluded.
    assert result.n_suggested == 4
    assert client.n_calls == 4
    with session_scope() as session:
        stochastic = list(
            session.execute(select(GapRow).where(GapRow.determinism == "stochastic")).scalars()
        )
    assert {r.concept_id for r in stochastic} == {"hub", "leaf", "shared", "sole"}
    leaf = next(r for r in stochastic if r.concept_id == "leaf")
    assert leaf.kind == "thin_area"
    assert leaf.status == "surfaced"
    assert leaf.rating == 0.6


def test_promoted_stochastic_survives_rebuild_and_resuggest(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    client = _FakeSuggestClient()

    build_gaps(
        apply=True, skeleton_dir=env / "skeleton", min_degree=2, suggest=True, client=client
    )
    with session_scope() as session:
        row = session.execute(
            select(GapRow).where(GapRow.determinism == "stochastic", GapRow.concept_id == "leaf")
        ).scalar_one()
        row.status = "promoted"

    # A deterministic rebuild + a re-suggest must not clobber the promotion.
    build_gaps(
        apply=True, skeleton_dir=env / "skeleton", min_degree=2, suggest=True, client=client
    )

    with session_scope() as session:
        stochastic = list(
            session.execute(select(GapRow).where(GapRow.determinism == "stochastic")).scalars()
        )
    assert len(stochastic) == 4  # re-suggest neither duplicates nor drops the other three
    leaf = next(r for r in stochastic if r.concept_id == "leaf")
    assert leaf.status == "promoted"  # survived both the rebuild and the re-suggest


def test_stochastic_rows_survive_a_deterministic_rebuild(env: Path) -> None:
    # A promoted stochastic suggestion on a LIVE graph concept must not be wiped by a
    # deterministic rebuild — only deterministic rows replace. (E0.2 changed the contract: a
    # stochastic row whose anchor left the vocabulary IS reaped now — see the reconcile test
    # below. "leaf" is graph_include=True, so it stays.)
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    with session_scope() as session:
        session.add(
            GapRow(
                concept_id="leaf",
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


def test_orphaned_stochastic_gap_is_reconciled_away(env: Path) -> None:
    """E0.2 / KI-17: a stochastic gap anchored on a concept that has left the graph vocabulary
    (excluded via graph_include=False, or deleted) is deleted on the next deterministic `--apply`
    — while a stochastic gap on a still-included concept keeps its status. This is the fix for the
    live symptom of 27 gaps served over a 13-node skeleton (10 orphaned). Fails today: no reconcile
    pass runs on a deterministic-only apply, so both rows would survive."""
    from doc_assistant.knowledge.concept_skeleton import set_graph_include

    _seed_curated_concepts()
    _write_skeleton(env / "skeleton")
    set_graph_include("hub", False)  # hub leaves the graph vocabulary — its gap is now an orphan
    with session_scope() as session:
        session.add_all(
            [
                GapRow(  # orphan: anchored on the now-excluded "hub"
                    concept_id="hub",
                    tier="t2a",
                    determinism="stochastic",
                    kind="thin_area",
                    status="promoted",  # even a human decision does not keep an orphan alive
                ),
                GapRow(  # live: anchored on still-included "sole"
                    concept_id="sole",
                    tier="t2a",
                    determinism="stochastic",
                    kind="thin_area",
                    status="promoted",
                ),
            ]
        )

    result = build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    assert result.n_reconciled == 1

    with session_scope() as session:
        stochastic = {
            r.concept_id: r
            for r in session.execute(
                select(GapRow).where(GapRow.determinism == "stochastic")
            ).scalars()
        }
    assert "hub" not in stochastic  # the orphan was reaped
    assert stochastic["sole"].status == "promoted"  # the live triage survived


def test_no_document_or_concept_mutation(env: Path) -> None:
    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    with session_scope() as session:
        concept_count = session.execute(select(func.count()).select_from(Concept)).scalar_one()
    assert concept_count == 5  # the sidecar never touches the curated vocabulary


# ============================================================
# ADR-017 C1 (E5) — gap triage override sidecar
# ============================================================


def test_triage_override_wins_over_row_status(env: Path) -> None:
    # load_gaps resolves the EFFECTIVE status: an override in the gap_triage sidecar beats the
    # deterministic row's own "surfaced" (ADR-017 C1).
    from doc_assistant.knowledge.gaps import load_gaps, set_gap_status

    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)

    set_gap_status("sole", "single_source", "dismissed")
    sole = [g for g in load_gaps() if g.concept_id == "sole" and g.kind == "single_source"]
    assert sole and sole[0].status == "dismissed"
    # An un-triaged gap is unaffected — the override is consulted only when present.
    iso = [g for g in load_gaps() if g.concept_id == "iso" and g.kind == "isolated"]
    assert iso and iso[0].status == "surfaced"


def test_dismissal_survives_a_deterministic_rebuild(env: Path) -> None:
    # THE POINT of the separate sidecar (ADR-017 C1): deterministic gap rows are delete-and-replace
    # on build_gaps, but the triage override lives in a table the rebuild never touches, so a
    # dismissal is durable across the acquire loop's rebuild.
    from doc_assistant.knowledge.gaps import load_gaps, set_gap_status

    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)
    set_gap_status("sole", "single_source", "dismissed")

    # Rebuild the deterministic rows from scratch (as the in-app rebuild route does).
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)

    sole = [g for g in load_gaps() if g.concept_id == "sole" and g.kind == "single_source"]
    assert sole and sole[0].status == "dismissed"  # the row was replaced; the verdict persisted


def test_reset_deletes_the_override(env: Path) -> None:
    from doc_assistant.knowledge.gaps import load_gap_overrides, load_gaps, set_gap_status

    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)

    set_gap_status("sole", "single_source", "promoted")
    assert ("sole", "single_source") in load_gap_overrides()
    set_gap_status("sole", "single_source", "surfaced")  # reset
    assert ("sole", "single_source") not in load_gap_overrides()  # row deleted
    sole = [g for g in load_gaps() if g.concept_id == "sole" and g.kind == "single_source"]
    assert sole and sole[0].status == "surfaced"  # back to the detector's default


def test_set_gap_status_rejects_an_invalid_status(env: Path) -> None:
    from doc_assistant.knowledge.gaps import set_gap_status

    with pytest.raises(ValueError, match="invalid gap status"):
        set_gap_status("sole", "single_source", "bogus")  # type: ignore[arg-type]


def test_gap_list_resolves_labels(env: Path) -> None:
    # The list surface resolves the concept UUID → label (the graph payload carries labels only on
    # nodes; a flat list needs them attached). A concept_id with no Concept falls back to itself.
    from doc_assistant.knowledge.concept_graph_view import load_gap_list

    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)

    items = load_gap_list()
    assert items  # gaps exist
    by_concept = {(it.gap.concept_id, it.gap.kind): it for it in items}
    assert by_concept[("sole", "single_source")].label == "Sole"  # resolved from the vocabulary


def test_triage_is_keyed_on_concept_and_kind(env: Path) -> None:
    # A concept can carry more than one gap kind; triaging one kind must not touch another.
    from doc_assistant.knowledge.gaps import load_gaps, set_gap_status

    _seed_curated_concepts()
    _seed_claims()
    _write_skeleton(env / "skeleton")
    build_gaps(apply=True, skeleton_dir=env / "skeleton", min_degree=2)

    # "leaf" carries under_connected; give it a second, made-up kind via a different key and verify
    # isolation by dismissing only one.
    set_gap_status("leaf", "under_connected", "dismissed")
    gaps = {(g.concept_id, g.kind): g.status for g in load_gaps()}
    assert gaps[("leaf", "under_connected")] == "dismissed"
    # iso's own gap is a different (concept, kind) and stays surfaced.
    assert gaps[("iso", "isolated")] == "surfaced"
