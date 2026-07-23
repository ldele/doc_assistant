"""Guard tests for the taxonomy layer (ADR-028 increment 1) — the `kind` column, the
`concept_hierarchy` / `document_field` tables, the `knowledge/taxonomy.py` write seam, and the
`scripts/seed_taxonomy.py` seeder. Each test fails against the pre-increment code.

DB-backed (a temp SQLite, engine monkeypatched — the repo's isolation pattern); no Chroma, no LLM.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event, inspect, select, text
from sqlalchemy.orm import sessionmaker

from doc_assistant.db.models import (
    Base,
    Concept,
    ConceptEdge,
    ConceptHierarchy,
    Document,
    DocumentField,
)
from doc_assistant.knowledge.taxonomy import (
    NotADomainError,
    TaxonomyCycleError,
    add_hierarchy_edge,
    attach_document_field,
    load_taxonomy,
    presence_nodes,
    remove_hierarchy_edge,
)


@pytest.fixture
def temp_db(monkeypatch):
    """Isolate the DB: a fresh temp SQLite with the current schema (create_all), FKs on, the
    global engine + session factory monkeypatched so `session_scope()` hits it."""
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


def _concept(session, cid: str, kind: str = "concept", label: str | None = None) -> Concept:
    c = Concept(id=cid, label=label or cid.upper(), kind=kind)
    session.add(c)
    session.flush()
    return c


# ============================================================
# Test 1 — the additive `kind` migration backfills existing rows to "concept"
# ============================================================


def test_migration_adds_kind_and_backfills_existing_rows(tmp_path):
    """A DB whose `concepts` predates `kind`: the migration adds it and every existing row
    reads "concept" (not NULL) — the KI-25 backfill-in-the-same-change discipline. Non-vacuous:
    the column is asserted absent before the migration runs."""
    from doc_assistant.db.migrations import _apply_additive_columns

    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}", future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE concepts (id VARCHAR PRIMARY KEY, label VARCHAR NOT NULL, "
                "source VARCHAR NOT NULL DEFAULT 'manual')"
            )
        )
        conn.execute(text("INSERT INTO concepts (id, label) VALUES ('c1', 'Old concept')"))

    assert "kind" not in {c["name"] for c in inspect(engine).get_columns("concepts")}

    added = _apply_additive_columns(engine)

    assert "concepts.kind" in added
    assert "kind" in {c["name"] for c in inspect(engine).get_columns("concepts")}
    with engine.connect() as conn:
        assert (
            conn.execute(text("SELECT kind FROM concepts WHERE id='c1'")).scalar_one() == "concept"
        )
    engine.dispose()


# ============================================================
# Test 2 — create_all produces the two new tables with their constraints
# ============================================================


def test_new_tables_created_with_constraints(temp_db):
    insp = inspect(create_engine(f"sqlite:///{temp_db}", future=True))
    tables = set(insp.get_table_names())
    assert {"concept_hierarchy", "document_field"} <= tables

    ch_cols = {c["name"] for c in insp.get_columns("concept_hierarchy")}
    assert {"id", "source_id", "target_id", "type", "created_at"} <= ch_cols
    df_cols = {c["name"] for c in insp.get_columns("document_field")}
    assert {"id", "document_id", "concept_id", "origin", "created_at"} <= df_cols

    ch_uniques = {
        tuple(u["column_names"]) for u in insp.get_unique_constraints("concept_hierarchy")
    }
    assert ("source_id", "target_id", "type") in ch_uniques
    df_uniques = {tuple(u["column_names"]) for u in insp.get_unique_constraints("document_field")}
    assert ("document_id", "concept_id") in df_uniques


# ============================================================
# Test 3 — a cycle is rejected (acyclicity invariant, ADR-028 D3)
# ============================================================


def test_cycle_is_rejected(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "a")
        _concept(s, "b")
        _concept(s, "c")
        add_hierarchy_edge(s, "a", "b", "is_a")  # a -> b
        add_hierarchy_edge(s, "b", "c", "is_a")  # b -> c
        with pytest.raises(TaxonomyCycleError) as exc:
            add_hierarchy_edge(s, "c", "a", "is_a")  # c -> a closes a -> b -> c -> a
        assert exc.value.source_id == "c" and exc.value.target_id == "a"


def test_self_edge_is_a_cycle(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "x")
        with pytest.raises(TaxonomyCycleError):
            add_hierarchy_edge(s, "x", "x", "in_field")


# ============================================================
# Test 4 — polyhierarchy: a concept may have two parents
# ============================================================


def test_polyhierarchy_two_parents_allowed(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "child")
        _concept(s, "p1", kind="domain")
        _concept(s, "p2", kind="domain")
        add_hierarchy_edge(s, "child", "p1", "in_field")
        add_hierarchy_edge(s, "child", "p2", "in_field")  # second parent — no error
        edges = (
            s.execute(select(ConceptHierarchy).where(ConceptHierarchy.source_id == "child"))
            .scalars()
            .all()
        )
        assert {e.target_id for e in edges} == {"p1", "p2"}


def test_add_hierarchy_edge_is_idempotent_and_rejects_bad_type(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "a")
        _concept(s, "b")
        first = add_hierarchy_edge(s, "a", "b", "is_a")
        again = add_hierarchy_edge(s, "a", "b", "is_a")  # idempotent — same row
        assert first.id == again.id
        assert len(s.execute(select(ConceptHierarchy)).scalars().all()) == 1
        with pytest.raises(ValueError):
            add_hierarchy_edge(s, "a", "b", "related")  # not a hierarchy edge type


def test_remove_hierarchy_edge(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "a")
        _concept(s, "b")
        add_hierarchy_edge(s, "a", "b", "is_a")
        assert remove_hierarchy_edge(s, "a", "b", "is_a") == 1
        assert remove_hierarchy_edge(s, "a", "b", "is_a") == 0  # already gone
        assert s.execute(select(ConceptHierarchy)).scalars().all() == []


# ============================================================
# Test 5 — presence_nodes() excludes domain nodes
# ============================================================


def test_presence_nodes_excludes_domains(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "real", kind="concept")
        _concept(s, "field", kind="domain")
        got = presence_nodes(s)
        assert [c.id for c in got] == ["real"]


# ============================================================
# Test 6 — a skeleton rebuild preserves the curated hierarchy (the whole point)
# ============================================================


def test_skeleton_rebuild_preserves_hierarchy(temp_db, tmp_path):
    """The KI-17/KI-20 guard: `concept_hierarchy` is curated data that MUST survive a
    `build_concept_skeleton` rebuild — which drops + rebuilds the derived `concept_edges`."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.concept_skeleton import build_concept_skeleton

    with session_scope() as s:
        _concept(s, "c1")
        _concept(s, "c2")
        add_hierarchy_edge(s, "c1", "c2", "is_a")
        # a derived edge that the rebuild SHOULD wipe (proves the rebuild actually ran)
        s.add(
            ConceptEdge(
                source_concept_id="c1", target_concept_id="c2", provenance_json="[]", weight=1.0
            )
        )

    build_concept_skeleton(
        apply=True,
        concept_loader=lambda: ([], {}),
        presence_loader=lambda document_ids=None: [],
        doc_graph_loader=lambda: ([], []),
        doc_years_loader=lambda: {},
        stance_loader=lambda root: {},
        skeleton_dir=tmp_path / "skeleton",
    )

    with session_scope() as s:
        surviving = s.execute(select(ConceptHierarchy)).scalars().all()
        rebuilt_edges = s.execute(select(ConceptEdge)).scalars().all()
    assert len(surviving) == 1  # the curated hierarchy edge survived
    assert rebuilt_edges == []  # the derived edge was dropped by the rebuild


# ============================================================
# Test 7 — document_field rejects a non-domain target; accepts a domain
# ============================================================


def test_attach_document_field_rejects_non_domain(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _concept(s, "concept_node", kind="concept")
        with pytest.raises(NotADomainError):
            attach_document_field(s, "doc-1", "concept_node")


def test_attach_document_field_accepts_domain_and_is_idempotent(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        s.add(
            Document(
                id="doc-1", filename="p.pdf", source_original="p.pdf", doc_hash="h", format="pdf"
            )
        )
        _concept(s, "ml", kind="domain")
        s.flush()
        link1 = attach_document_field(s, "doc-1", "ml")
        link2 = attach_document_field(s, "doc-1", "ml")  # idempotent
        assert link1.id == link2.id
        assert len(s.execute(select(DocumentField)).scalars().all()) == 1


# ============================================================
# Test 8 — the seeder is idempotent (matches the bundled data file's own structure)
# ============================================================


def test_seed_taxonomy_idempotent(temp_db):
    from scripts.seed_taxonomy import load_seed, seed_taxonomy

    data = load_seed()
    n_div = len(data["divisions"])
    n_grp = len(data["groups"])

    first = seed_taxonomy(apply=True)
    second = seed_taxonomy(apply=True)  # re-run is a no-op

    assert first == (n_div + n_grp, n_grp, n_grp)
    assert second == first

    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        domains = s.execute(select(Concept).where(Concept.kind == "domain")).scalars().all()
        edges = s.execute(select(ConceptHierarchy)).scalars().all()
    assert len(domains) == n_div + n_grp  # no duplicates across the two runs
    assert len(edges) == n_grp
    # Every seeded field node is a domain, and every group edge rolls up to a real division.
    graph = None
    with session_scope() as s:
        graph = load_taxonomy(s)
    assert all(graph.nodes[n]["kind"] == "domain" for n in graph.nodes)


def test_seed_taxonomy_seeds_all_23_divisions(temp_db):
    from scripts.seed_taxonomy import anzsrc_node_id, load_seed, seed_taxonomy

    seed_taxonomy(apply=True)
    data = load_seed()
    assert len(data["divisions"]) == 23  # ANZSRC 2020 FoR has 23 divisions

    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        for div in data["divisions"]:
            node = s.get(Concept, anzsrc_node_id(div["code"]))
            assert node is not None and node.kind == "domain" and node.source == "anzsrc"


# ============================================================
# Test 9 — the bundled data file carries the CC-BY attribution (a licence obligation)
# ============================================================


def test_seed_data_file_has_ccby_attribution():
    from scripts.seed_taxonomy import load_seed

    meta = load_seed()["_meta"]
    assert "CC BY" in meta["license"]
    assert meta["license_url"]
    assert "Australian Bureau of Statistics" in meta["attribution"]
    assert "CC BY" in meta["attribution"]
    assert meta["source_url"].startswith("https://")


# ============================================================
# Consumer guards — domain nodes must not leak into the always-on concept surfaces (ADR-028 D4)
# ============================================================


def test_list_keyword_families_excludes_domains(temp_db):
    """The Library keyword-families list must not surface abstract taxonomy field nodes — else the
    seeded ~236 ANZSRC domains would flood the filter."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import list_keyword_families

    with session_scope() as s:
        _concept(s, "real-concept", kind="concept", label="Dense retrieval")
        _concept(s, "anzsrc-46", kind="domain", label="Information and computing sciences")

    families = list_keyword_families()
    assert [f.canonical for f in families] == ["Dense retrieval"]


def test_load_glossary_excludes_domains(temp_db):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.concept_skeleton import load_glossary

    with session_scope() as s:
        _concept(s, "real-concept", kind="concept", label="BM25")
        _concept(s, "anzsrc-49", kind="domain", label="Mathematical sciences")

    labels = [e.label for e in load_glossary()]
    assert labels == ["BM25"]


# ============================================================
# taxonomy_view read model (increment 2a) — forest + set-semantics rollup coverage
# ============================================================


def _field(session, fid: str, label: str | None = None) -> Concept:
    return _concept(session, fid, kind="domain", label=label or fid.upper())


def test_taxonomy_view_zero_state_and_unassigned(temp_db):
    """A seeded field forest with no members: totals honest, rollups 0, concepts unassigned."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge
    from doc_assistant.knowledge.taxonomy_view import load_taxonomy_view

    with session_scope() as s:
        _field(s, "div", "Information and computing sciences")
        _field(s, "grp", "Machine learning")
        add_hierarchy_edge(s, "grp", "div", "in_field")  # group -> division
        _concept(s, "c1", kind="concept", label="Embeddings")
        _concept(s, "c2", kind="concept", label="BM25")

    view = load_taxonomy_view()
    assert len(view.fields) == 2
    assert view.roots == ("div",)  # the division has no broader parent
    assert view.n_concepts_total == 2
    assert view.n_unassigned_concepts == 2  # neither concept attached yet
    assert all(f.n_concepts_rollup == 0 and f.n_documents_rollup == 0 for f in view.fields)


def test_taxonomy_view_rollup_crosses_group_to_division(temp_db):
    """Attaching a concept to a group rolls up to the division (DoD 2)."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge
    from doc_assistant.knowledge.taxonomy_view import load_taxonomy_view

    with session_scope() as s:
        _field(s, "div")
        _field(s, "grp")
        add_hierarchy_edge(s, "grp", "div", "in_field")
        _concept(s, "c1", kind="concept")
        add_hierarchy_edge(s, "c1", "grp", "in_field")  # attach concept to the group

    by_id = {f.id: f for f in load_taxonomy_view().fields}
    assert by_id["grp"].n_concepts_direct == 1
    assert by_id["div"].n_concepts_direct == 0
    assert by_id["div"].n_concepts_rollup == 1  # rollup crosses grp -> div
    assert load_taxonomy_view().n_unassigned_concepts == 0


def test_taxonomy_view_rollup_dedups_polyhierarchy(temp_db):
    """A concept under two groups of one division counts ONCE at the division (DoD 6)."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge
    from doc_assistant.knowledge.taxonomy_view import load_taxonomy_view

    with session_scope() as s:
        _field(s, "div")
        _field(s, "g1")
        _field(s, "g2")
        add_hierarchy_edge(s, "g1", "div", "in_field")
        add_hierarchy_edge(s, "g2", "div", "in_field")
        _concept(s, "c1", kind="concept")
        add_hierarchy_edge(s, "c1", "g1", "in_field")
        add_hierarchy_edge(s, "c1", "g2", "in_field")  # same concept, two parents

    by_id = {f.id: f for f in load_taxonomy_view().fields}
    assert by_id["div"].n_concepts_rollup == 1  # deduped by id, not 2


def test_taxonomy_view_document_rollup(temp_db):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge, attach_document_field
    from doc_assistant.knowledge.taxonomy_view import load_taxonomy_view

    with session_scope() as s:
        s.add(Document(id="d1", filename="p.pdf", source_original="p", doc_hash="h", format="pdf"))
        _field(s, "div")
        _field(s, "grp")
        add_hierarchy_edge(s, "grp", "div", "in_field")
        s.flush()
        attach_document_field(s, "d1", "grp")

    by_id = {f.id: f for f in load_taxonomy_view().fields}
    assert by_id["grp"].n_documents_direct == 1
    assert by_id["div"].n_documents_rollup == 1  # doc rolls up to the division


def test_load_field_detail(temp_db):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge
    from doc_assistant.knowledge.taxonomy_view import load_field_detail

    with session_scope() as s:
        _field(s, "grp", "Machine learning")
        _concept(s, "c1", kind="concept", label="Embeddings")
        add_hierarchy_edge(s, "c1", "grp", "in_field")

    detail = load_field_detail("grp")
    assert detail is not None
    assert detail.label == "Machine learning"
    assert detail.concepts == (("c1", "Embeddings"),)
    # a concept id or an unknown id is not a field -> None (distinct from a real-but-empty field)
    assert load_field_detail("c1") is None
    assert load_field_detail("nope") is None
