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

from doc_assistant.db.models import Concept, ConceptHierarchy, Document, DocumentField

# The two curated hierarchy edge types (ADR-028 Decision 2). ``related`` (the associative
# Node-A/B layer) is deliberately NOT here — it lives in the derived ``concept_edges``.
HIERARCHY_EDGE_TYPES: frozenset[str] = frozenset({"is_a", "in_field"})
#: Link provenance, shared by both curated tables (ADR-028 D8): ``curated`` = a user edit or the
#: ANZSRC seed (always wins); ``proposed`` = an auto-fill awaiting accept-or-delete.
DOCUMENT_FIELD_ORIGINS: frozenset[str] = frozenset({"curated", "proposed"})
HIERARCHY_ORIGINS: frozenset[str] = DOCUMENT_FIELD_ORIGINS


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
    session: Session, source_id: str, target_id: str, edge_type: str, origin: str = "curated"
) -> ConceptHierarchy:
    """Add one hierarchy edge, rejecting anything that would close a cycle.

    ``source --edge_type--> target``: ``is_a`` = concept → broader concept; ``in_field`` =
    concept/field → broader field. The sole sanctioned writer of ``concept_hierarchy`` — the
    acyclicity invariant (ADR-028 D3) is enforced *here*, so no other path can smuggle a cycle in.

    Idempotent on the unique key ``(source_id, target_id, type)``, with one deliberate exception:
    a **curated** write over an existing ``proposed`` row *promotes* it in place (ADR-028 D8's
    accept primitive — accepting a proposal is the same call the UI already makes to attach). The
    reverse never happens: a ``proposed`` write leaves a ``curated`` row untouched, so an
    auto-propose pass can never quietly overwrite the user's own placement. Flushes so a bad
    foreign key (a source/target that is not a real concept) surfaces as an ``IntegrityError``
    within this call, not later.

    Raises:
        ValueError: ``edge_type``/``origin`` is not one of :data:`HIERARCHY_EDGE_TYPES` /
            :data:`HIERARCHY_ORIGINS`.
        TaxonomyCycleError: the edge would make the hierarchy cyclic (incl. a self-edge).
    """
    if edge_type not in HIERARCHY_EDGE_TYPES:
        raise ValueError(
            f"edge_type must be one of {sorted(HIERARCHY_EDGE_TYPES)}, got {edge_type!r}"
        )
    if origin not in HIERARCHY_ORIGINS:
        raise ValueError(f"origin must be one of {sorted(HIERARCHY_ORIGINS)}, got {origin!r}")

    existing = session.execute(
        select(ConceptHierarchy).where(
            ConceptHierarchy.source_id == source_id,
            ConceptHierarchy.target_id == target_id,
            ConceptHierarchy.type == edge_type,
        )
    ).scalar_one_or_none()
    if existing is not None:
        if origin == "curated" and existing.origin != "curated":
            existing.origin = "curated"  # accept: promote the proposal, don't duplicate it
            session.flush()
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

    edge = ConceptHierarchy(
        source_id=source_id, target_id=target_id, type=edge_type, origin=origin
    )
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


def detach_document_field(session: Session, document_id: str, field_id: str) -> int:
    """Remove a document→field link by its pair. Returns the number of rows removed.

    The counterpart :func:`attach_document_field` shipped without (increment 2a had no caller).
    D8's contract — a proposal the user *accepts or deletes* — makes it required: without a detach,
    a machine-proposed document classification could never be rejected. Origin-agnostic: the user
    may equally undo their own curated attach.
    """
    rows = list(
        session.execute(
            select(DocumentField).where(
                DocumentField.document_id == document_id,
                DocumentField.concept_id == field_id,
            )
        ).scalars()
    )
    for row in rows:
        session.delete(row)
    session.flush()
    return len(rows)


def presence_nodes(session: Session) -> list[Concept]:
    """The single canonical accessor for text-bearing concepts (``kind="concept"``).

    ADR-028 Decision 4's centralised guard: every presence / gap / co-occurrence consumer reads
    concepts *through here*, so abstract ``kind="domain"`` field nodes (which have no text presence
    and would read as a false ``isolated``/``single_source`` gap) are excluded in one place.
    """
    return list(session.execute(select(Concept).where(Concept.kind == "concept")).scalars().all())


def unplaced_concepts(session: Session, *, graph_only: bool = True) -> list[Concept]:
    """Text-bearing concepts with **no** ``in_field`` edge yet — the auto-propose input set.

    Reads through :func:`presence_nodes`' kind guard, so an abstract ``kind="domain"`` field node
    is never returned (a field has no field parent to propose, and the seeded trunk's own
    ``in_field`` edges are not placements). ``graph_only`` (the default) narrows to the curated
    graph vocabulary (``graph_include`` true) — ADR-018's boundary between the concept map and the
    breadth-first keyword families, which the taxonomy augments (ADR-019 D1). Pass
    ``graph_only=False`` to widen to every promoted keyword. Origin is ignored: a concept carrying
    a *proposed* placement is already placed and is not re-proposed.

    Returns ``[]`` on an empty corpus — the honest zero-state, not an error.
    """
    placed = set(
        session.execute(
            select(ConceptHierarchy.source_id).where(ConceptHierarchy.type == "in_field")
        )
        .scalars()
        .all()
    )
    concepts = presence_nodes(session)
    if graph_only:
        concepts = [c for c in concepts if c.graph_include]
    return [c for c in concepts if c.id not in placed]


def unclassified_documents(session: Session) -> list[Document]:
    """Documents with no ``document_field`` link yet — the document half of the propose input.

    The 25-of-47 concept-less documents ADR-028 D6 exists for are in here by construction: this
    asks only about the *explicit* link, never about derived concept presence. ``[]`` at 0 docs.
    """
    classified = set(session.execute(select(DocumentField.document_id)).scalars().all())
    documents = list(session.execute(select(Document)).scalars().all())
    return [d for d in documents if d.id not in classified]


def load_taxonomy(session: Session) -> nx.DiGraph:
    """The curated hierarchy as a read-only ``networkx.DiGraph`` (nodes + edges carry attrs).

    Nodes = every ``Concept`` (attrs: ``kind``, ``label``), so isolated and domain nodes are
    present; edges = every ``concept_hierarchy`` row oriented ``source → target`` (attrs: ``type``,
    ``origin``). The substrate later increments traverse for coverage rollup; this build never
    writes. ``origin`` rides along so a reader can tell a curated placement from a proposed one
    without a second query (ADR-028 D8).
    """
    graph: nx.DiGraph = nx.DiGraph()
    for concept in session.execute(select(Concept)).scalars().all():
        graph.add_node(concept.id, kind=concept.kind, label=concept.label)
    for row in session.execute(select(ConceptHierarchy)).scalars().all():
        graph.add_edge(row.source_id, row.target_id, type=row.type, origin=row.origin)
    return graph
