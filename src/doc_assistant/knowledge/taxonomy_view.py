"""Taxonomy read model (ADR-028 increment 2a) — the field forest + coverage a UI renders.

Read-only assembly over the curated taxonomy (`concept_hierarchy` + `document_field`): the
domain-node forest, each field's directly-attached concepts/documents, and the **set-semantics
rollup** (ADR-028 D6) — a field's coverage is the *distinct* set of concepts/documents for which it
is an ancestor (itself or any narrower descendant), deduped by id. No fractional counts, no forced
primary parent.

Pure read (own `session_scope`), zero-LLM, zero-network. The write seam is `knowledge/taxonomy.py`;
the HTTP shell is `apps/api/routers/taxonomy.py`. Currently every rollup is 0 (nothing is attached
yet — the honest zero-state), which is a legitimate value, not a missing artifact.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
from sqlalchemy import func, select

from doc_assistant.db.models import Document, DocumentField
from doc_assistant.db.session import session_scope
from doc_assistant.knowledge.taxonomy import load_taxonomy


@dataclass(frozen=True)
class TaxonomyField:
    """One field node (a ``kind="domain"`` concept) with its structure + coverage counts."""

    id: str
    label: str
    parent_ids: tuple[str, ...]  # broader fields (this ``--in_field-->`` them)
    child_ids: tuple[str, ...]  # narrower fields (they ``--in_field-->`` this)
    n_concepts_direct: int  # concepts attached straight to this field
    n_documents_direct: int  # documents attached straight to this field
    n_concepts_rollup: int  # distinct concepts under this field or any narrower descendant
    n_documents_rollup: int  # distinct documents under this field or any narrower descendant


@dataclass(frozen=True)
class TaxonomyView:
    """The whole field forest + corpus-level totals."""

    fields: tuple[TaxonomyField, ...]
    roots: tuple[str, ...]  # field ids with no broader (in_field) parent — the divisions
    n_concepts_total: int  # curated text-bearing concepts (kind="concept")
    n_documents_total: int  # documents in the corpus (the classification denominator)
    n_unassigned_concepts: int  # concepts with no in_field edge to any field yet


@dataclass(frozen=True)
class FieldDetail:
    """One field's directly-attached members (for a drill-in), plus its rollup counts."""

    id: str
    label: str
    concepts: tuple[tuple[str, str], ...]  # direct (concept_id, label)
    documents: tuple[tuple[str, str], ...]  # direct (document_id, title-or-filename)
    n_concepts_rollup: int
    n_documents_rollup: int


def _kind(graph: nx.DiGraph, node: str) -> str:
    kind = graph.nodes[node].get("kind")
    return str(kind) if kind is not None else ""


def _field_doc_map(session) -> dict[str, set[str]]:  # type: ignore[no-untyped-def]
    """field_id -> set of directly-attached document ids (from ``document_field``)."""
    out: dict[str, set[str]] = {}
    for concept_id, document_id in session.execute(
        select(DocumentField.concept_id, DocumentField.document_id)
    ).all():
        out.setdefault(concept_id, set()).add(document_id)
    return out


def load_taxonomy_view() -> TaxonomyView:
    """Assemble the field forest with direct + rolled-up concept/document coverage."""
    with session_scope() as session:
        graph = load_taxonomy(session)
        field_docs = _field_doc_map(session)
        n_documents_total = int(
            session.execute(select(func.count()).select_from(Document)).scalar_one()
        )

    domains = [n for n in graph.nodes if _kind(graph, n) == "domain"]
    concepts = [n for n in graph.nodes if _kind(graph, n) == "concept"]

    # in_field-only view for the field↔field structure (parents/children) and root detection.
    in_field = nx.DiGraph()
    in_field.add_nodes_from(graph.nodes(data=True))
    for src, tgt, data in graph.edges(data=True):
        if data.get("type") == "in_field":
            in_field.add_edge(src, tgt)

    def direct_concepts(field: str) -> set[str]:
        # concepts C with C --in_field--> field (C is a predecessor of `field` in in_field)
        return {p for p in in_field.predecessors(field) if _kind(graph, p) == "concept"}

    fields: list[TaxonomyField] = []
    roots: list[str] = []
    for field in domains:
        parents = tuple(
            sorted(s for s in in_field.successors(field) if _kind(graph, s) == "domain")
        )
        children = tuple(
            sorted(p for p in in_field.predecessors(field) if _kind(graph, p) == "domain")
        )
        if not parents:
            roots.append(field)

        # Rollup: ancestors(field) in the FULL hierarchy graph = every node that can reach `field`
        # (its narrower descendants + their concepts) — set-union, deduped by id (ADR-028 D6).
        under = nx.ancestors(graph, field) | {field}
        concepts_rollup = {n for n in under if _kind(graph, n) == "concept"}
        domains_under = {n for n in under if _kind(graph, n) == "domain"}
        docs_rollup: set[str] = set()
        for d in domains_under:
            docs_rollup |= field_docs.get(d, set())

        fields.append(
            TaxonomyField(
                id=field,
                label=str(graph.nodes[field].get("label", "")),
                parent_ids=parents,
                child_ids=children,
                n_concepts_direct=len(direct_concepts(field)),
                n_documents_direct=len(field_docs.get(field, set())),
                n_concepts_rollup=len(concepts_rollup),
                n_documents_rollup=len(docs_rollup),
            )
        )

    fields.sort(key=lambda f: f.label.casefold())
    roots.sort(key=lambda fid: str(graph.nodes[fid].get("label", "")).casefold())

    # A concept is "assigned" once it has any in_field edge to a field.
    assigned = {
        c for c in concepts if any(_kind(graph, s) == "domain" for s in in_field.successors(c))
    }
    return TaxonomyView(
        fields=tuple(fields),
        roots=tuple(roots),
        n_concepts_total=len(concepts),
        n_documents_total=n_documents_total,
        n_unassigned_concepts=len(concepts) - len(assigned),
    )


def load_field_detail(field_id: str) -> FieldDetail | None:
    """One field's directly-attached concepts + documents + rollup counts; ``None`` if not a field.

    ``None`` distinguishes "not a domain node" (a wrong id, or a concept id) from "a real but empty
    field" (which returns a detail with empty member lists) — the UI needs the two apart."""
    with session_scope() as session:
        graph = load_taxonomy(session)
        if field_id not in graph.nodes or _kind(graph, field_id) != "domain":
            return None
        field_docs = _field_doc_map(session)

        in_field = nx.DiGraph()
        in_field.add_nodes_from(graph.nodes(data=True))
        for src, tgt, data in graph.edges(data=True):
            if data.get("type") == "in_field":
                in_field.add_edge(src, tgt)

        direct_concept_ids = sorted(
            p for p in in_field.predecessors(field_id) if _kind(graph, p) == "concept"
        )
        concepts = tuple(
            (cid, str(graph.nodes[cid].get("label", ""))) for cid in direct_concept_ids
        )

        doc_ids = sorted(field_docs.get(field_id, set()))
        titles: dict[str, str] = {}
        if doc_ids:
            for did, title, filename in session.execute(
                select(Document.id, Document.title, Document.filename).where(
                    Document.id.in_(doc_ids)
                )
            ).all():
                titles[did] = title or filename
        documents = tuple((did, titles.get(did, did)) for did in doc_ids)

        under = nx.ancestors(graph, field_id) | {field_id}
        concepts_rollup = {n for n in under if _kind(graph, n) == "concept"}
        domains_under = {n for n in under if _kind(graph, n) == "domain"}
        docs_rollup: set[str] = set()
        for d in domains_under:
            docs_rollup |= field_docs.get(d, set())

    return FieldDetail(
        id=field_id,
        label=str(graph.nodes[field_id].get("label", "")),
        concepts=concepts,
        documents=documents,
        n_concepts_rollup=len(concepts_rollup),
        n_documents_rollup=len(docs_rollup),
    )
