"""Integration guard tests for the Node-A skeleton build.

Seeds a curated vocabulary + Citation/DocSimilarity into a temp file-backed SQLite, injects
a fake presence loader (no Chroma), and exercises the real DB read/write paths. Asserts:
the deterministic build makes zero LLM calls, writes the derived sidecar + skeleton.json,
is byte-identical on a re-run, and keeps the two lifecycles distinct (curated rows survive a
--force rebuild; derived rows are dropped + rebuilt). The chunk store is never touched.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

import doc_assistant.db.session as session_mod
from doc_assistant.db.models import (
    Base,
    Citation,
    Concept,
    ConceptAlias,
    ConceptEdge,
    ConceptPresenceRow,
    ConceptWrittenForm,
    DocSimilarity,
    Document,
)
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.concept_skeleton import build_concept_skeleton


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


def _seed() -> dict[str, str]:
    """Two docs, two curated concepts (RAG + DPR), a citation + a similarity edge d1->d2."""
    ids: dict[str, str] = {}
    with session_scope() as session:
        for name in ("d1", "d2"):
            doc = Document(
                filename=f"{name}.pdf",
                source_original=f"{name}.pdf",
                doc_hash=f"h{name}",
                format="pdf",
            )
            session.add(doc)
            session.flush()
            ids[name] = str(doc.id)

        # graph_include is opt-in (ADR-018) and independent of `source` — a promoted
        # keyword is graph vocabulary here precisely because the flag, not the source,
        # decides membership.
        rag = Concept(label="RAG", source="keyword", graph_include=True)
        dpr = Concept(label="DPR", source="manual", graph_include=True)
        session.add_all([rag, dpr])
        session.flush()
        session.add(ConceptAlias(concept_id=rag.id, alias="retrieval-augmented generation"))

        session.add(Citation(source_document_id=ids["d1"], target_document_id=ids["d2"]))
        session.add(
            DocSimilarity(
                source_document_id=ids["d1"],
                target_document_id=ids["d2"],
                embedding_model="bge-base",
                score=0.71,
            )
        )
    return ids


def _fake_presence(ids: dict[str, str]):
    """RAG+DPR co-occur in d1:p0; RAG alone in d2:p0 → RAG spans d1,d2 and DPR sits in d1."""

    def loader(document_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
        return [
            (f"{ids['d1']}:p0", ids["d1"], "RAG and DPR are evaluated together here."),
            (f"{ids['d2']}:p0", ids["d2"], "RAG appears again in a second paper."),
        ]

    return loader


def _count(model: type) -> int:
    with session_scope() as session:
        return int(session.execute(select(func.count()).select_from(model)).scalar() or 0)


def test_node_a_build_writes_sidecar_and_provenance(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    result = build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )

    assert result.applied is True
    assert result.n_concepts == 2  # both curated concepts are nodes
    assert result.n_edges == 1  # the single RAG-DPR co-occurrence edge
    assert (skeleton_dir / "skeleton.json").exists()
    assert _count(ConceptEdge) == 1
    assert _count(ConceptPresenceRow) == 3  # RAG@d1, RAG@d2, DPR@d1

    # The edge carries citation + similarity provenance (d1->d2 spans RAG[d1,d2] / DPR[d1]).
    with session_scope() as session:
        edge = session.execute(select(ConceptEdge)).scalar_one()
        provenance = set(json.loads(edge.provenance_json))
        strength = json.loads(edge.strength_json)
    assert provenance == {"cooccurrence", "citation", "similarity"}
    # R4: graded strength persisted per doc-pair token; this saturated toy graph → 1.0.
    # Co-occurrence is the base fact and carries no strength entry.
    assert strength == {"citation": 1.0, "similarity": 1.0}


def test_build_is_byte_identical_on_rerun(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    kwargs = dict(apply=True, min_cooccurrence=1, presence_loader=_fake_presence(ids))

    build_concept_skeleton(skeleton_dir=skeleton_dir, **kwargs)
    first = (skeleton_dir / "skeleton.json").read_text(encoding="utf-8")
    build_concept_skeleton(skeleton_dir=skeleton_dir, **kwargs)
    second = (skeleton_dir / "skeleton.json").read_text(encoding="utf-8")

    assert first == second  # timestamp-free graph_version → byte-identical rebuild
    assert _count(ConceptEdge) == 1  # replace-not-append (no row duplication)
    assert _count(ConceptPresenceRow) == 3


def test_force_rebuild_keeps_curated_drops_derived(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )
    curated_concepts = _count(Concept)
    curated_aliases = _count(ConceptAlias)

    build_concept_skeleton(
        apply=True,
        force=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )

    # Curated vocabulary survives a --force rebuild; derived rows are regenerated.
    assert _count(Concept) == curated_concepts == 2
    assert _count(ConceptAlias) == curated_aliases == 1
    assert _count(ConceptEdge) == 1
    assert _count(ConceptPresenceRow) == 3


def test_dry_run_writes_nothing(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    result = build_concept_skeleton(
        apply=False,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )
    assert result.applied is False
    assert result.n_edges == 1  # computed...
    assert not (skeleton_dir / "skeleton.json").exists()  # ...but nothing written
    assert _count(ConceptEdge) == 0
    assert _count(ConceptPresenceRow) == 0


def test_plain_rebuild_preserves_node_b_stance(env: Path) -> None:
    """E0.5b: a plain ``--apply`` (Node A, no ``--enrich``) must NOT wipe Node-B stance. It
    recomputes structure only, so it re-attaches the stance/relation of edges that still exist
    instead of dropping them — otherwise corpus-wide epistemics goes dark on every in-app rebuild
    (the G6-run footgun). Fails today: without ``_reattach_stance`` the rebuilt edge is stance-less
    and its ``concept_edges.stance_json`` is NULL."""
    ids = _seed()
    skeleton_dir = env / "skeleton"
    presence = _fake_presence(ids)

    r1 = build_concept_skeleton(
        apply=True, min_cooccurrence=1, presence_loader=presence, skeleton_dir=skeleton_dir
    )
    assert r1.skeleton.edges[0].stance_by_doc == ()  # Node A carries no stance
    assert r1.skeleton.edges[0].relation is None

    # Simulate a Node-B enrich by injecting stance/relation into the on-disk skeleton.json (the
    # artifact the next rebuild reads) + the concept_edges row.
    path = skeleton_dir / "skeleton.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    stance = [[ids["d1"], "supports"], [ids["d2"], "contradicts"]]
    for e in data["edges"]:
        e["stance"] = stance
        e["relation"] = "is evaluated with"
        e["provenance"] = sorted(set(e["provenance"]) | {"llm_relation"})
    path.write_text(json.dumps(data), encoding="utf-8")

    # A plain rebuild — no --enrich.
    r2 = build_concept_skeleton(
        apply=True, min_cooccurrence=1, presence_loader=presence, skeleton_dir=skeleton_dir
    )
    e2 = r2.skeleton.edges[0]
    assert e2.relation == "is evaluated with"
    assert e2.stance_by_doc == ((ids["d1"], "supports"), (ids["d2"], "contradicts"))
    assert "llm_relation" in e2.provenance

    # ...and the preserved stance reached the concept_edges DB row, not just skeleton.json.
    with session_scope() as session:
        row = session.execute(select(ConceptEdge)).scalar_one()
    assert row.stance_json is not None
    assert json.loads(row.stance_json) == stance


def test_build_never_touches_chunk_store(env: Path) -> None:
    ids = _seed()
    skeleton_dir = env / "skeleton"
    build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_fake_presence(ids),
        skeleton_dir=skeleton_dir,
    )
    # Sidecar-only: no Chroma dir created, Document rows unchanged.
    assert not (env / "chroma").exists()
    assert not (env / "chroma_pc").exists()
    assert _count(Document) == 2


def test_written_forms_are_stored_beside_the_label_never_into_it(env: Path) -> None:
    """ADR-053 decision 3 / ADR-043: the build votes how the prose writes each label, stores it in
    its own table, on the node and in skeleton.json, and matches presence with it. The curated
    label is never rewritten."""
    with session_scope() as session:
        doc_ids = []
        for name in ("n1", "v1"):
            doc = Document(
                filename=f"{name}.pdf",
                source_original=f"{name}.pdf",
                doc_hash=f"h{name}",
                format="pdf",
            )
            session.add(doc)
            session.flush()
            doc_ids.append(str(doc.id))
        din = Concept(label="din", source="keyword", graph_include=True)
        session.add(din)
        session.flush()
        din_id = str(din.id)
    n1, v1 = doc_ids

    def loader(document_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
        return [
            (
                f"{n1}:p0",
                n1,
                "Recordings from the dIN population show a steady rhythm. "
                "In every larva the dIN cells fire first.",
            ),
            (f"{v1}:p0", v1, "A diet with vitamin Din adults was not studied in this cohort."),
        ]

    skeleton_dir = env / "skeleton"
    result = build_concept_skeleton(
        apply=True, min_cooccurrence=1, presence_loader=loader, skeleton_dir=skeleton_dir
    )

    (node,) = result.skeleton.nodes
    assert node.written == "dIN"
    assert node.doc_ids == (n1,)  # the vitamin-Din document is not a mention of dIN
    assert (result.n_written_forms, result.n_cased_forms) == (1, 1)
    with session_scope() as session:
        concept = session.get(Concept, din_id)
        assert concept is not None and concept.label == "din"  # never rewritten
        row = session.execute(select(ConceptWrittenForm)).scalar_one()
        assert (row.concept_id, row.form, row.written) == (din_id, "din", "dIN")
        assert json.loads(row.votes_json) == {"dIN": 1, "Din": 1}
    data = json.loads((skeleton_dir / "skeleton.json").read_text(encoding="utf-8"))
    assert data["nodes"][0]["written"] == "dIN"


# ============================================================
# ADR-054 — exact and broad forms
# ============================================================


def _seed_distillation() -> dict[str, str]:
    """Three documents and two concepts. `knowledge distillation` is named in d1 only; its alias
    `distillation` occurs alone in d2 and d3; `pruning` shares a chunk with it in d1 and d2."""
    ids: dict[str, str] = {}
    with session_scope() as session:
        for name in ("d1", "d2", "d3"):
            doc = Document(
                filename=f"{name}.pdf",
                source_original=f"{name}.pdf",
                doc_hash=f"h{name}",
                format="pdf",
            )
            session.add(doc)
            session.flush()
            ids[name] = str(doc.id)
        kd = Concept(label="knowledge distillation", source="manual", graph_include=True)
        pruning = Concept(label="pruning", source="manual", graph_include=True)
        session.add_all([kd, pruning])
        session.flush()
        session.add(ConceptAlias(concept_id=kd.id, alias="distillation"))
        ids["kd"], ids["pruning"] = str(kd.id), str(pruning.id)
    return ids


def _distillation_chunks(ids: dict[str, str]):
    def loader(document_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
        return [
            (
                f"{ids['d1']}:p0",
                ids["d1"],
                "In this study knowledge distillation is combined with pruning of the ranker.",
            ),
            (
                f"{ids['d2']}:p0",
                ids["d2"],
                "We apply distillation to the ranker first, and pruning comes afterwards.",
            ),
            (f"{ids['d3']}:p0", ids["d3"], "The cost of distillation falls with the batch size."),
        ]

    return loader


def _set_breadth(concept_id: str, alias: str, breadth: str | None) -> None:
    with session_scope() as session:
        row = session.execute(
            select(ConceptAlias).where(
                ConceptAlias.concept_id == concept_id, ConceptAlias.alias == alias
            )
        ).scalar_one()
        row.breadth = breadth


def _build(env: Path, ids: dict[str, str]):
    skeleton_dir = env / "skeleton"
    result = build_concept_skeleton(
        apply=True,
        min_cooccurrence=1,
        presence_loader=_distillation_chunks(ids),
        skeleton_dir=skeleton_dir,
    )
    return result, (skeleton_dir / "skeleton.json").read_text(encoding="utf-8")


def test_a_broad_form_is_counted_beside_presence_never_in_it(env: Path) -> None:
    """ADR-054: once the user marks `distillation` broad, `knowledge distillation` is present only
    where it is named. The two documents the bare word reaches are listed beside it, and nothing
    that is counted from presence — the edge, the presence rows — sees them."""
    from doc_assistant.knowledge.concept_skeleton import load_broad_forms, load_concepts

    ids = _seed_distillation()
    before, _ = _build(env, ids)
    kd = next(n for n in before.skeleton.nodes if n.id == ids["kd"])
    assert kd.doc_ids == tuple(sorted([ids["d1"], ids["d2"], ids["d3"]]))
    assert (kd.broad_doc_ids, kd.broad_forms) == ((), ())
    assert before.skeleton.edges[0].n_cooccurrence_chunks == 2  # d1 and d2

    _set_breadth(ids["kd"], "distillation", "broad")
    _concepts, aliases = load_concepts()
    assert aliases.get(ids["kd"], []) == []  # the loader everything counts through
    assert load_broad_forms() == {ids["kd"]: ["distillation"]}

    after, text = _build(env, ids)
    kd = next(n for n in after.skeleton.nodes if n.id == ids["kd"])
    assert kd.doc_ids == (ids["d1"],)
    assert kd.broad_doc_ids == tuple(sorted([ids["d2"], ids["d3"]]))
    assert kd.broad_forms == ("distillation",)
    assert after.skeleton.edges[0].n_cooccurrence_chunks == 1  # d2 no longer links the two
    assert (after.n_broad_concepts, after.n_broad_documents) == (1, 2)
    with session_scope() as session:
        rows = session.execute(
            select(ConceptPresenceRow.document_id).where(
                ConceptPresenceRow.concept_id == ids["kd"]
            )
        ).all()
    assert [r[0] for r in rows] == [ids["d1"]]
    node = next(n for n in json.loads(text)["nodes"] if n["id"] == ids["kd"])
    assert node["broad_doc_ids"] == sorted([ids["d2"], ids["d3"]])
    assert node["broad_forms"] == ["distillation"]


def test_nothing_moves_until_a_form_is_marked_broad(env: Path) -> None:
    """An unclassified form is exact, and so is one the user marked exact: the build is byte for
    byte the one a library without breadth produces. Clearing a broad mark puts it back."""
    ids = _seed_distillation()
    _, unclassified = _build(env, ids)
    assert "broad" not in unclassified  # no key is written for a node with nothing beside it

    _set_breadth(ids["kd"], "distillation", "exact")
    assert _build(env, ids)[1] == unclassified

    _set_breadth(ids["kd"], "distillation", "broad")
    assert _build(env, ids)[1] != unclassified

    _set_breadth(ids["kd"], "distillation", None)
    assert _build(env, ids)[1] == unclassified


def test_the_view_says_which_concept_it_is_behind_on_until_the_rebuild(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: a build records what each concept was counted through, a mark made afterwards
    is reported against that concept, and the rebuild that applies it clears the report."""
    from doc_assistant.knowledge.concept_graph_view import load_graph_view

    monkeypatch.setattr("doc_assistant.config.CONCEPT_SKELETON_DIR", env / "skeleton")
    ids = _seed_distillation()
    _, text = _build(env, ids)
    assert set(json.loads(text)["meta"]["forms"]) == {ids["kd"], ids["pruning"]}
    view = load_graph_view()
    assert view is not None
    assert (view.staleness.stale, view.staleness.forms_changed_ids) == (False, ())

    _set_breadth(ids["kd"], "distillation", "broad")
    view = load_graph_view()
    assert view is not None
    assert (view.staleness.stale, view.staleness.forms_changed_ids) == (True, (ids["kd"],))

    _build(env, ids)
    view = load_graph_view()
    assert view is not None
    assert (view.staleness.stale, view.staleness.forms_changed_ids) == (False, ())


def test_a_broad_mark_on_the_name_itself_is_ignored(env: Path) -> None:
    """The name is always exact: an alias row that repeats the label cannot take the concept's
    own documents away, whatever it is marked."""
    from doc_assistant.knowledge.concept_skeleton import load_broad_forms

    ids = _seed_distillation()
    with session_scope() as session:
        session.add(
            ConceptAlias(concept_id=ids["kd"], alias="Knowledge Distillation", breadth="broad")
        )
    assert load_broad_forms() == {}
    result, _ = _build(env, ids)
    kd = next(n for n in result.skeleton.nodes if n.id == ids["kd"])
    assert ids["d1"] in kd.doc_ids and kd.broad_doc_ids == ()
