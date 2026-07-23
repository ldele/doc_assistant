"""Curated taxonomy layer — the classification DAG over the concept graph (ADR-028).

The write seam + read accessors for the curated hierarchy (``concept_hierarchy``) and the
document→field links (``document_field``). Pure, session-scoped, **zero-LLM, zero-network**:
all logic lives here per the thin-shell rule; ``scripts/seed_taxonomy.py`` is the CLI over it.

Two invariants live here and nowhere else:

- **Acyclicity** — the ``is_a``/``in_field`` hierarchy is a DAG. :func:`add_hierarchy_edge` rejects
  any edge that would close a cycle (ADR-028 Decision 3). There is *no* maximum depth — a hard
  cap is the corpus-tuned magic number the robustness contract bans, and "depth" is multi-valued
  under polyhierarchy; acyclicity alone guarantees traversal termination and well-defined roots.
- **Presence-kind guard** — :func:`presence_nodes` is the single canonical accessor returning only
  ``kind="concept"`` rows, so the domain-exclusion (ADR-028 Decision 4) is written once, centrally,
  not scattered as N ``WHERE kind`` clauses across every presence/gap detector.

Edge orientation is ``source → target`` = *narrower → broader* (concept → its field / a field →
its parent field), so a walk toward roots is a walk along the edges.
"""

from __future__ import annotations

import networkx as nx
from sqlalchemy import select
from sqlalchemy.orm import Session

from doc_assistant.db.models import Concept, ConceptHierarchy, DocumentField

# The two curated hierarchy edge types (ADR-028 Decision 2). ``related`` (the associative
# Node-A/B layer) is deliberately NOT here — it lives in the derived ``concept_edges``.
HIERARCHY_EDGE_TYPES: frozenset[str] = frozenset({"is_a", "in_field"})
DOCUMENT_FIELD_ORIGINS: frozenset[str] = frozenset({"curated", "proposed"})


class TaxonomyCycleError(ValueError):
    """Raised when an edge would make the ``is_a``/``in_field`` hierarchy cyclic (ADR-028 D3).

    Carries ``path`` — the node ids of the cycle the edge would close, for a legible message.
    """

    def __init__(self, source_id: str, target_id: str, path: list[str]) -> None:
        self.source_id = source_id
        self.target_id = target_id
        self.path = path
        super().__init__(
            f"hierarchy edge {source_id} -> {target_id} would create a cycle: {' -> '.join(path)}"
        )


class NotADomainError(ValueError):
    """Raised when a document is attached to a field node that is not ``kind="domain"``."""


def _hierarchy_edges(session: Session) -> list[tuple[str, str]]:
    """All curated hierarchy edges as ``(source_id, target_id)`` pairs (both edge types)."""
    rows = session.execute(select(ConceptHierarchy.source_id, ConceptHierarchy.target_id)).all()
    return [(r[0], r[1]) for r in rows]


def add_hierarchy_edge(
    session: Session, source_id: str, target_id: str, edge_type: str
) -> ConceptHierarchy:
    """Add one curated hierarchy edge, rejecting anything that would close a cycle.

    ``source --edge_type--> target``: ``is_a`` = concept → broader concept; ``in_field`` =
    concept/field → broader field. The sole sanctioned writer of ``concept_hierarchy`` — the
    acyclicity invariant (ADR-028 D3) is enforced *here*, so no other path can smuggle a cycle in.

    Idempotent on the unique key ``(source_id, target_id, type)``: a re-add returns the existing
    row untouched. Flushes so a bad foreign key (a source/target that is not a real concept)
    surfaces as an ``IntegrityError`` within this call, not later.

    Raises:
        ValueError: ``edge_type`` is not one of :data:`HIERARCHY_EDGE_TYPES`.
        TaxonomyCycleError: the edge would make the hierarchy cyclic (incl. a self-edge).
    """
    if edge_type not in HIERARCHY_EDGE_TYPES:
        raise ValueError(
            f"edge_type must be one of {sorted(HIERARCHY_EDGE_TYPES)}, got {edge_type!r}"
        )

    existing = session.execute(
        select(ConceptHierarchy).where(
            ConceptHierarchy.source_id == source_id,
            ConceptHierarchy.target_id == target_id,
            ConceptHierarchy.type == edge_type,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    # Cycle check over the whole hierarchy (is_a + in_field): add the candidate to the current
    # DAG and test. Whole-graph rather than an incident subgraph — clearest correct form, and this
    # is a curation action, not a hot path (a bulk seed of E edges is O(E^2), trivial for the ~213
    # seed edges; a 10k-node bound is an RIGOR_TODO measurement, not a design cap — ADR-028 D3).
    graph: nx.DiGraph = nx.DiGraph()
    graph.add_edges_from(_hierarchy_edges(session))
    graph.add_edge(source_id, target_id)
    if not nx.is_directed_acyclic_graph(graph):
        try:
            cycle = nx.find_cycle(graph, source=source_id)
            path = [edge[0] for edge in cycle] + [cycle[-1][1]]
        except nx.NetworkXNoCycle:  # pragma: no cover - guarded by the is_dag check above
            path = [source_id, target_id]
        raise TaxonomyCycleError(source_id, target_id, path)

    edge = ConceptHierarchy(source_id=source_id, target_id=target_id, type=edge_type)
    session.add(edge)
    session.flush()
    return edge


def remove_hierarchy_edge(session: Session, source_id: str, target_id: str, edge_type: str) -> int:
    """Delete a curated hierarchy edge by its unique key. Returns the number of rows removed."""
    rows = list(
        session.execute(
            select(ConceptHierarchy).where(
                ConceptHierarchy.source_id == source_id,
                ConceptHierarchy.target_id == target_id,
                ConceptHierarchy.type == edge_type,
            )
        ).scalars()
    )
    for row in rows:
        session.delete(row)
    session.flush()
    return len(rows)


def attach_document_field(
    session: Session, document_id: str, field_id: str, origin: str = "curated"
) -> DocumentField:
    """Link a document to a taxonomy field (a ``kind="domain"`` node). Idempotent per pair.

    Validates that ``field_id`` resolves to a domain node — a document must attach to a *field*,
    not to a text-bearing concept (ADR-028 D6). A re-attach of the same ``(document, field)`` pair
    returns the existing row untouched (its ``origin`` is not overwritten — a curated row keeps
    winning over a later proposal).

    Raises:
        ValueError: ``origin`` is not one of :data:`DOCUMENT_FIELD_ORIGINS`.
        NotADomainError: ``field_id`` is missing or is not a ``kind="domain"`` concept.
    """
    if origin not in DOCUMENT_FIELD_ORIGINS:
        raise ValueError(f"origin must be one of {sorted(DOCUMENT_FIELD_ORIGINS)}, got {origin!r}")

    kind = session.execute(select(Concept.kind).where(Concept.id == field_id)).scalar_one_or_none()
    if kind != "domain":
        raise NotADomainError(
            f"document_field target {field_id!r} must be a kind='domain' node, got {kind!r}"
        )

    existing = session.execute(
        select(DocumentField).where(
            DocumentField.document_id == document_id,
            DocumentField.concept_id == field_id,
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    link = DocumentField(document_id=document_id, concept_id=field_id, origin=origin)
    session.add(link)
    session.flush()
    return link


def presence_nodes(session: Session) -> list[Concept]:
    """The single canonical accessor for text-bearing concepts (``kind="concept"``).

    ADR-028 Decision 4's centralised guard: every presence / gap / co-occurrence consumer reads
    concepts *through here*, so abstract ``kind="domain"`` field nodes (which have no text presence
    and would read as a false ``isolated``/``single_source`` gap) are excluded in one place.
    """
    return list(session.execute(select(Concept).where(Concept.kind == "concept")).scalars().all())


def load_taxonomy(session: Session) -> nx.DiGraph:
    """The curated hierarchy as a read-only ``networkx.DiGraph`` (nodes + edges carry attrs).

    Nodes = every ``Concept`` (attrs: ``kind``, ``label``), so isolated and domain nodes are
    present; edges = every ``concept_hierarchy`` row oriented ``source → target`` (attr: ``type``).
    The substrate later increments traverse for coverage rollup; this build never writes.
    """
    graph: nx.DiGraph = nx.DiGraph()
    for concept in session.execute(select(Concept)).scalars().all():
        graph.add_node(concept.id, kind=concept.kind, label=concept.label)
    for row in session.execute(select(ConceptHierarchy)).scalars().all():
        graph.add_edge(row.source_id, row.target_id, type=row.type)
    return graph
