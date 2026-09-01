"""Concept-graph wire models (PR-G1 — ADR-017 / docs/specs/feature-concept-graph.md).

The read model for one render of the graph view — nodes, edges, Louvain communities, detected
gaps and the staleness report — plus the gap-list surface (E5) and its triage body.

Wire id space: concept **UUIDs** everywhere — `ConceptGraphNodePayload.id`,
`ConceptGraphEdgePayload.source`/`target`, `GapPayload.concept_id` and
`ConceptCommunityPayload.node_ids` are all `Concept.id`. `label` rides **only** on the node;
the client joins by id. Mixing ids and labels across this boundary is the bug that caused
KI-15, so the one id space is a contract, not a convention.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from doc_assistant.knowledge.concept_graph_view import (
        GapListItem,
        GraphStaleness,
        GraphView,
    )
    from doc_assistant.knowledge.concept_skeleton import (
        Community,
        ConceptNode,
        ConceptPresence,
        SkeletonEdge,
    )
    from doc_assistant.knowledge.gaps import Gap


class ConceptGraphNodePayload(BaseModel):
    """One concept node. `degree` and `community` are precomputed layout signal."""

    id: str
    label: str
    doc_ids: list[str]
    degree: int
    community: int

    @classmethod
    def from_node(cls, n: ConceptNode) -> ConceptGraphNodePayload:
        return cls(
            id=n.id,
            label=n.label,
            doc_ids=list(n.doc_ids),
            degree=n.degree,
            community=n.community,
        )


class ConceptGraphEdgePayload(BaseModel):
    """An undirected concept-concept edge, typed by its provenance set.

    `relation`/`stance` are the deferred Node-B annotation and are empty on every edge until
    that pass runs — a renderer must not imply agreement/disagreement it does not have.
    """

    source: str
    target: str
    provenance: list[str]
    weight: float
    n_cooccurrence_chunks: int
    relation: str | None = None

    @classmethod
    def from_edge(cls, e: SkeletonEdge) -> ConceptGraphEdgePayload:
        return cls(
            source=e.source_concept_id,
            target=e.target_concept_id,
            provenance=sorted(e.provenance),
            weight=e.weight,
            n_cooccurrence_chunks=e.n_cooccurrence_chunks,
            relation=e.relation,
        )


class ConceptCommunityPayload(BaseModel):
    """A Louvain community. `id` is POSITIONAL, not identity — it renumbers when the
    vocabulary changes, so a client must never persist a preference against it."""

    id: int
    label: str
    node_ids: list[str]
    size: int

    @classmethod
    def from_community(cls, c: Community) -> ConceptCommunityPayload:
        return cls(id=c.id, label=c.label, node_ids=list(c.node_ids), size=c.size)


class GapPayload(BaseModel):
    """One detected corpus gap (ADR-004), anchored to a concept.

    `rating` is `None` for every deterministic gap (a raw graph fact carries no confidence).
    `status` is the row's own value; per ADR-017 C1 a user's triage lives in its own override
    sidecar (deterministic rows are delete-and-replace), so it is not yet resolved here.
    """

    concept_id: str
    tier: str
    determinism: str
    kind: str
    fact_ids: list[str]
    rating: float | None = None
    status: str

    @classmethod
    def from_gap(cls, g: Gap) -> GapPayload:
        return cls(
            concept_id=g.concept_id,
            tier=g.tier,
            determinism=g.determinism,
            kind=g.kind,
            fact_ids=list(g.evidence.fact_ids),
            rating=g.rating,
            status=g.status,
        )


class GapListItemPayload(BaseModel):
    """One gap for the first-class gap-list surface (ROADMAP E5) — a gap with its concept `label`
    resolved server-side and the **effective** `status` (a user triage override wins; ADR-017 C1).
    Distinct from `GapPayload`, which rides in the graph payload and joins labels by node id."""

    concept_id: str
    label: str
    kind: str
    tier: str
    determinism: str
    fact_ids: list[str]
    rating: float | None
    status: str  # effective: surfaced | promoted | dismissed

    @classmethod
    def from_item(cls, item: GapListItem) -> GapListItemPayload:
        g = item.gap
        return cls(
            concept_id=g.concept_id,
            label=item.label,
            kind=g.kind,
            tier=g.tier,
            determinism=g.determinism,
            fact_ids=list(g.evidence.fact_ids),
            rating=g.rating,
            status=g.status,
        )


class GapTriageRequest(BaseModel):
    """POST body to triage one gap (ADR-017 C1, E5). `status` is the user's verdict; `surfaced`
    resets it to the detector's default (deletes the override). Keyed on `(concept_id, kind)` — the
    stable identity that survives the deterministic gaps' delete-and-rebuild."""

    concept_id: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    status: Literal["surfaced", "promoted", "dismissed"]


class GraphStalenessPayload(BaseModel):
    """How far the built graph has drifted from the live vocabulary.

    The skeleton is a build artifact and the Manage-keywords view writes `Concept` rows live,
    so drift is structural, not a defect: the UI reports it and offers a rebuild.
    """

    stale: bool
    n_concepts_in_db: int
    n_concepts_in_skeleton: int
    added_labels: list[str]
    removed_ids: list[str]
    #: Documents the graph cites that the library can no longer resolve — the corpus half of
    #: staleness, as opposed to the vocabulary half above.
    missing_document_ids: list[str] = []
    n_documents_in_skeleton: int = 0
    #: The library's size, so the client can state coverage rather than guess at it.
    n_documents_in_library: int = 0

    @classmethod
    def from_staleness(cls, s: GraphStaleness) -> GraphStalenessPayload:
        return cls(
            stale=s.stale,
            n_concepts_in_db=s.n_concepts_in_db,
            n_concepts_in_skeleton=s.n_concepts_in_skeleton,
            added_labels=list(s.added_labels),
            removed_ids=list(s.removed_ids),
            missing_document_ids=list(s.missing_document_ids),
            n_documents_in_skeleton=s.n_documents_in_skeleton,
            n_documents_in_library=s.n_documents_in_library,
        )


class ConceptGraphPayload(BaseModel):
    """The whole read model for one render of the concept-graph view."""

    graph_version: str
    nodes: list[ConceptGraphNodePayload]
    edges: list[ConceptGraphEdgePayload]
    communities: list[ConceptCommunityPayload]
    gaps: list[GapPayload]
    staleness: GraphStalenessPayload

    @classmethod
    def from_view(cls, v: GraphView) -> ConceptGraphPayload:
        return cls(
            graph_version=str(v.skeleton.meta.get("graph_version", "")),
            nodes=[ConceptGraphNodePayload.from_node(n) for n in v.skeleton.nodes],
            edges=[ConceptGraphEdgePayload.from_edge(e) for e in v.skeleton.edges],
            communities=[
                ConceptCommunityPayload.from_community(c) for c in v.skeleton.communities
            ],
            gaps=[GapPayload.from_gap(g) for g in v.gaps],
            staleness=GraphStalenessPayload.from_staleness(v.staleness),
        )


class ConceptPresencePayload(BaseModel):
    """Where one concept appears in one document.

    `chunk_keys` are ADR-4 composite `"{document_id}:p{parent_index}"` — the navigation
    payload that takes the ego view from a concept down to the chunks that mention it.
    """

    document_id: str
    chunk_keys: list[str]
    n_mentions: int

    @classmethod
    def from_presence(cls, p: ConceptPresence) -> ConceptPresencePayload:
        return cls(
            document_id=p.document_id,
            chunk_keys=list(p.chunk_keys),
            n_mentions=p.n_mentions,
        )
