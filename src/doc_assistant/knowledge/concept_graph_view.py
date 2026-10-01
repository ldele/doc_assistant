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
    forms_fingerprints,
    load_broad_forms,
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
    #: Documents the skeleton cites that the library can no longer resolve. A **third** kind of
    #: staleness, and the one that was invisible: the other two watch the vocabulary, which the
    #: user edits constantly, while this watches the corpus the graph was built *over*. A document
    #: whose identity moved (every pre-ADR-047 re-extraction minted a new id) leaves a reference
    #: nothing can resolve, and the view has no honest way to name it. Measured on the reference
    #: library the day this was added: **10 of 10** referenced ids dead, across all 198 nodes.
    missing_document_ids: tuple[str, ...] = ()
    #: Documents the graph covers **that the library still shows** — the cited ids minus deleted
    #: and archived ones, so it can never exceed `n_documents_in_library`.
    n_documents_in_skeleton: int = 0
    #: How many documents the library holds, so the view can state its **coverage**.
    #:
    #: The obvious inverse of `missing_document_ids` — "documents the graph has not seen yet" —
    #: would be a lie dressed as a number. A document appears in the graph once it mentions a
    #: concept in the graph vocabulary; on the reference library that is **30 of 98**, and the
    #: other 68 are not waiting for a rebuild, they mention none of the 13 included concepts (of
    #: 593 curated). Reporting them as pending would send the user to a button that changes
    #: nothing. The honest pair is coverage plus the rule that produces it — which points at
    #: curating vocabulary (ADR-018, ROADMAP 23), the lever that actually moves it.
    n_documents_in_library: int = 0
    #: Concepts on both sides whose name or forms changed since the build (ADR-054): a form added,
    #: removed, or marked exact or broad the other way. Their counts are the old forms' counts
    #: until a rebuild, and no node or edge says so. Ids, like everything on this wire — the view
    #: resolves the name from the node. Empty for a skeleton built before forms were recorded:
    #: there is nothing to compare, and "changed" would be a guess.
    forms_changed_ids: tuple[str, ...] = ()
    #: False for a skeleton built before forms were recorded. Not folded into ``stale``, which
    #: reports differences that are known: here the graph cannot tell, so the view says exactly
    #: that and offers the rebuild that starts the record — without it a form marked on such a
    #: graph would change nothing on screen and offer no way to apply it.
    forms_recorded: bool = True


@dataclass(frozen=True)
class GraphView:
    """The whole read model for one render of the concept-graph view."""

    skeleton: ConceptSkeleton
    gaps: tuple[Gap, ...]
    staleness: GraphStaleness


def _live_document_ids() -> tuple[set[str], set[str]]:
    """``(every document row, the documents the library shows)`` — one two-column read, no join.

    Two sets because they answer two questions. A cited id is *resolvable* while any row holds it,
    archived or not, so staleness asks the first. Coverage is a fraction of the library the user
    sees, so its denominator is the second — the non-archived count ``library.count_documents``
    reports (ROADMAP 54: a coverage number counts the artifact it describes).
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        rows = session.execute(select(Document.id, Document.is_archived)).all()
    every = {str(doc_id) for doc_id, _ in rows}
    shown = {str(doc_id) for doc_id, archived in rows if not archived}
    return every, shown


def _staleness(skeleton: ConceptSkeleton) -> GraphStaleness:
    """Compare the skeleton against the live vocabulary **and** the live corpus (three id sets,
    and the forms each concept is counted through).

    The corpus half is not symmetric with the vocabulary half, on purpose. A document *added*
    since the build is not staleness — the graph simply has not seen it yet, which is true of
    every build the moment it finishes and is already covered by offering a rebuild. A document
    the skeleton *cites* that no longer exists is different: it is a reference the view cannot
    render, and without this it reached the user as a bare UUID in the title slot.
    """
    concepts, aliases = load_concepts()
    db_labels = {cid: label for cid, label in concepts}
    db_ids = set(db_labels)
    sk_ids = {n.id for n in skeleton.nodes}
    added = db_ids - sk_ids  # curated since the build
    removed = sk_ids - db_ids  # deleted since the build

    # The forms half (ADR-054): the same concept, counted through different forms now.
    built_forms = skeleton.meta.get("forms")
    forms_changed: list[str] = []
    if isinstance(built_forms, dict):
        live_forms = forms_fingerprints(concepts, aliases, load_broad_forms())
        forms_changed = sorted(
            cid
            for cid in db_ids & sk_ids
            if cid in built_forms and built_forms[cid] != live_forms[cid]
        )

    live_docs, shown_docs = _live_document_ids()
    cited_docs = {d for n in skeleton.nodes for d in n.doc_ids}
    missing_docs = cited_docs - live_docs
    return GraphStaleness(
        stale=bool(added or removed or missing_docs or forms_changed),
        n_concepts_in_db=len(db_ids),
        n_concepts_in_skeleton=len(sk_ids),
        added_labels=tuple(sorted(db_labels[i] for i in added)),
        removed_ids=tuple(sorted(removed)),
        missing_document_ids=tuple(sorted(missing_docs)),
        # Numerator and denominator over the same set: a deleted document the skeleton still
        # cites is staleness (above), not coverage — counted here it could push "covers N of M"
        # past M and the client hides the line (ROADMAP 54).
        n_documents_in_skeleton=len(cited_docs & shown_docs),
        n_documents_in_library=len(shown_docs),
        forms_changed_ids=tuple(forms_changed),
        forms_recorded=isinstance(built_forms, dict),
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


@dataclass(frozen=True)
class GapListItem:
    """One gap for the first-class gap-list surface (ROADMAP E5): a :class:`Gap` with its concept
    **label resolved server-side**.

    The graph payload carries labels only on nodes (the KI-15 one-id-space contract, and a
    consumer there joins by id). The gap *list* is a standalone surface — not joined to a rendered
    graph — so it needs the human label attached. For a deterministic gap the label comes from the
    curated vocabulary; for a stochastic suggestion whose ``concept_id`` is a candidate not yet a
    ``Concept``, the label falls back to the ``concept_id`` itself (which *is* the candidate
    string). ``status`` is the effective value already resolved by ``load_gaps`` (override wins).
    ``written`` is how the library writes the label when it writes it in a case — what to show
    instead of ``label`` (ADR-054), ``None`` otherwise.
    """

    gap: Gap
    label: str
    written: str | None = None


def load_gap_list() -> list[GapListItem]:
    """The gap list with concept labels resolved (E5). Empty when no gaps are built (0-doc/
    pre-build) — never an error. Ordering is the detector's (kind, concept_id); the UI applies the
    RG-014 presentation order (strong list-shaped kinds first, ``under_connected`` opt-in)."""
    from doc_assistant.knowledge.written_forms import shown_labels

    concepts, _aliases = load_concepts()
    labels = {cid: label for cid, label in concepts}
    shown = shown_labels(concepts)
    return [
        GapListItem(
            gap=g,
            label=labels.get(g.concept_id, g.concept_id),
            written=shown.get(g.concept_id),
        )
        for g in load_gaps()
    ]


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


@dataclass(frozen=True)
class VocabularyMatch:
    """One row a label search found, with whether it is on the graph and has a definition.

    ``on_graph`` is also what separates a *concept* from a *term* (ADR-054): a row the user has
    taken on, or a string the library uses that nobody has. ``written`` is how the library writes
    the label when it writes it in a case — what to show instead of ``label``."""

    id: str
    label: str
    on_graph: bool
    has_definition: bool
    written: str | None = None


def search_vocabulary(query: str, *, limit: int = 20) -> list[VocabularyMatch]:
    """Concepts whose label contains ``query``, across the **whole** vocabulary (ADR-053).

    The Graph tab lists its nodes; the concept panel is where a definition is read and chosen, and
    the words ADR-052 cares most about (``specter``, ``viral``, ``beta``) are not graph nodes. This
    is how the panel reaches them. Case-insensitive; an exact label first, then labels that start
    with the query, then the rest — graph concepts before others within each. Taxonomy fields are
    not concepts and never match. An empty query matches nothing.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope

    q = query.strip()
    if not q:
        return []
    escaped = q.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    with session_scope() as session:
        rows = session.execute(
            select(Concept.id, Concept.label, Concept.graph_include, Concept.definition).where(
                Concept.kind == "concept", Concept.label.ilike(f"%{escaped}%", escape="\\")
            )
        ).all()
    from doc_assistant.knowledge.written_forms import shown_labels

    folded = q.casefold()
    shown = shown_labels([(str(cid), str(label)) for cid, label, _graph, _definition in rows])
    matches = [
        VocabularyMatch(
            id=str(cid),
            label=str(label),
            on_graph=bool(graph),
            has_definition=bool(definition),
            written=shown.get(str(cid)),
        )
        for cid, label, graph, definition in rows
    ]
    matches.sort(
        key=lambda m: (
            m.label.casefold() != folded,
            not m.label.casefold().startswith(folded),
            not m.on_graph,
            m.label.casefold(),
        )
    )
    return matches[:limit]
