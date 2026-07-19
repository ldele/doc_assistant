"""Read model for the desktop concept-graph view (ADR-017 · `docs/specs/feature-concept-graph.md`).

Assembles the regenerable ``skeleton.json`` sidecar + the ``gaps`` sidecar + a staleness verdict
into the one view the thin API shell serves. The API layer maps these dataclasses to wire models;
all of the reasoning lives here (CONTEXT rule 3 — ``apps/`` is a shell).

**Read-only by decision, not by accident (ADR-017 A1).** The graph observes; the Manage-keywords
view edits. Nothing here writes ``Concept``/``ConceptAlias``: the graph renders a *derived*
artifact, so an in-place write would invalidate the very thing being looked at. Rebuilding is
likewise not this module's job — ``build_concept_skeleton`` is the write seam; a route triggers it.

**Wire id space — one, and only one (ADR-017; KI-15).** Concept **UUIDs** everywhere:
``ConceptNode.id``, ``SkeletonEdge.source_concept_id``/``target_concept_id``, ``Gap.concept_id``
and ``Community.node_ids`` are all ``Concept.id``. ``label`` is carried **only** on the node;
consumers join by id. Mixing ids and labels across a boundary is exactly the bug that made KI-15
silently match nothing.

**Staleness is a first-class part of the payload, not an afterthought.** The skeleton is a build
artifact and the Manage-keywords view writes ``Concept`` rows live, so the graph *always* lags user
edits by construction (ADR-015 named the shared-rows boundary). The honest response is to report
the lag and offer a rebuild — never to hide it, and never to auto-rebuild (that would spend the
user's time unasked and destroy the seeded-layout determinism the view is verified with).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from doc_assistant.knowledge.concept_skeleton import (
    ConceptPresence,
    ConceptSkeleton,
    load_concepts,
    load_skeleton,
)
from doc_assistant.knowledge.gaps import Gap, load_gaps

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class GraphStaleness:
    """How far the built skeleton has drifted from the live curated vocabulary.

    Derived at read time from a set comparison of concept ids — nothing is persisted, and
    ``graph_version`` is deliberately **not** the signal: it only ever equals itself, so it can
    tell you *which* build you are looking at but never *whether* it is current.
    """

    stale: bool
    n_concepts_in_db: int
    n_concepts_in_skeleton: int
    added_labels: tuple[str, ...]  # curated since the build — absent from the graph
    removed_ids: tuple[str, ...]  # in the graph but deleted from the vocabulary since


@dataclass(frozen=True)
class GraphView:
    """The whole read model for one render of the concept-graph view."""

    skeleton: ConceptSkeleton
    gaps: tuple[Gap, ...]
    staleness: GraphStaleness


def _staleness(skeleton: ConceptSkeleton) -> GraphStaleness:
    """Compare the skeleton's nodes against the live vocabulary (cheap: two id sets)."""
    concepts, _aliases = load_concepts()
    db_labels = {cid: label for cid, label in concepts}
    db_ids = set(db_labels)
    sk_ids = {n.id for n in skeleton.nodes}
    added = db_ids - sk_ids  # curated since the build
    removed = sk_ids - db_ids  # deleted since the build
    return GraphStaleness(
        stale=bool(added or removed),
        n_concepts_in_db=len(db_ids),
        n_concepts_in_skeleton=len(sk_ids),
        added_labels=tuple(sorted(db_labels[i] for i in added)),
        removed_ids=tuple(sorted(removed)),
    )


def load_graph_view() -> GraphView | None:
    """Assemble the graph read model, or ``None`` when the skeleton has never been built.

    ``None`` is the **normal first run** — ``skeleton.json`` is a gitignored, regenerable sidecar,
    so a fresh clone has none. Callers render an empty state offering a rebuild; they must not
    treat it as an error. (A skeleton that exists but is corrupt raises — see ``load_skeleton``.)

    Gaps are read from their own sidecar rather than recomputed: ``build_gaps`` is the detector's
    write seam, and re-deriving here would make a read route silently depend on the whole detector
    chain. An **empty gap list on a present skeleton is legitimate** — it means ``build_gaps
    --apply`` has not run (or found nothing), not that the graph is broken.
    """
    skeleton = load_skeleton()
    if skeleton is None:
        return None
    gaps = load_gaps()
    staleness = _staleness(skeleton)
    log.info(
        "graph_view_loaded",
        nodes=len(skeleton.nodes),
        edges=len(skeleton.edges),
        communities=len(skeleton.communities),
        gaps=len(gaps),
        stale=staleness.stale,
        graph_version=skeleton.meta.get("graph_version"),
    )
    return GraphView(skeleton=skeleton, gaps=tuple(gaps), staleness=staleness)


def load_concept_presence(concept_id: str) -> list[ConceptPresence]:
    """Every document one concept appears in, with the chunk keys it appears in.

    The navigation payload for the ego view: concept → document → *the chunks where it is
    actually mentioned*. Served **per concept, not in bulk** — the view renders one concept's
    neighbourhood at a time (the ego-first decision), and the corpus carries 1781 chunk keys
    across 222 rows today, which grows with the vocabulary.

    Chunk keys are the ADR-4 composite ``"{document_id}:p{parent_index}"``. Returns ``[]`` for an
    unknown concept — a caller that needs to distinguish "no such concept" from "present nowhere"
    must check the vocabulary itself.
    """
    import json

    from sqlalchemy import select

    from doc_assistant.db.models import ConceptPresenceRow
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = list(
            session.execute(
                select(ConceptPresenceRow)
                .where(ConceptPresenceRow.concept_id == concept_id)
                .order_by(ConceptPresenceRow.document_id)
            ).scalars()
        )
        return [
            ConceptPresence(
                concept_id=r.concept_id,
                document_id=r.document_id,
                chunk_keys=tuple(json.loads(r.chunk_keys_json or "[]")),
                n_mentions=r.n_mentions,
            )
            for r in rows
        ]
