"""Concept-graph redesign — the deterministic concept skeleton (Node A).

The curated-vocabulary + deterministic-skeleton redesign of Feature 7
(``docs/archive/concept-graph-redesign.md``; supersedes the open-vocabulary PR-16
``concept_graph.py``, ``.claude/KNOWN_ISSUES.md`` KI-7). The node vocabulary is
**user-curated** (``Concept`` / ``ConceptAlias``); presence and the edge skeleton are
computed with **zero LLM** from that vocabulary plus the library's existing zero-LLM
graphs (``Citation``, ``DocSimilarity``) and chunk-level co-occurrence. Every edge is
auditable to a co-occurrence / citation / similarity fact via a *provenance set*; edges
are kept and ranked by provenance, never dropped for lacking an LLM stance.

Module shape (mirrors ``concept_graph.py`` / ``wiki.py`` / ``epistemics.py``): a pure,
deterministic core (no DB, no LLM, no network) behind a thin impure boundary
(SQLite + Chroma reads, sidecar writes) and an orchestrator. Node A makes **zero** LLM
calls; the confined LLM relation/stance pass (Node B) is a separate, deferred module
(``concept_skeleton_enrich.py``).

The graph is a regenerable sidecar — a ``skeleton.json`` artifact under
``CONCEPT_SKELETON_DIR`` plus the ``concept_edges`` / ``concept_presence`` tables, dropped
and rebuilt on each run (Enrichment-Layer Pattern); it never mutates the chunk store.
"""

from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)


# ============================================================
# Vocabulary (re-homed verbatim out of the now-retired open-vocabulary module — Decision 7)
# ============================================================

POLARITIES: tuple[str, ...] = ("supports", "refines", "contradicts", "supersedes")
SUPPORTING_POLARITIES: frozenset[str] = frozenset({"supports", "refines"})
OPPOSING_POLARITIES: frozenset[str] = frozenset({"contradicts", "supersedes"})

#: The provenance tokens an edge can carry (Decision 5). Node A fills the first three;
#: ``"llm_relation"`` is appended by the deferred Node B.
PROVENANCE_SOURCES: tuple[str, ...] = ("cooccurrence", "citation", "similarity", "llm_relation")

#: Sidecar artifact filename under ``CONCEPT_SKELETON_DIR``.
SKELETON_NAME = "skeleton.json"


# ============================================================
# Data classes (frozen — the pure core is value-in/value-out)
# ============================================================


@dataclass(frozen=True)
class ConceptPresence:
    """Where one curated concept appears in one document (deterministic match)."""

    concept_id: str
    document_id: str
    chunk_keys: tuple[str, ...]  # "{document_id}:p{parent_index}" (ADR-4)
    n_mentions: int


@dataclass(frozen=True)
class SkeletonEdge:
    """An undirected concept-concept edge, typed by a provenance set (Decision 5).

    ``source_concept_id`` < ``target_concept_id`` by construction (canonical order).
    ``stance_by_doc`` / ``relation`` are the deferred Node-B LLM annotation — empty/None
    after the deterministic Node-A build.

    ``provenance_strength`` (R4) is a *graded* strength in ``[0, 1]`` per doc-pair provenance
    token (``citation`` / ``similarity``; ``cooccurrence`` is the base fact and carries none):
    the fraction of candidate endpoint-doc pairs the token's doc graph actually links. Stored
    as a sorted ``(token, strength)`` tuple so the frozen edge stays hashable + byte-stable.
    Presence of a token is unchanged (strength > 0 ⟺ the old boolean "any linked"); the
    strength only refines ``edge_weight``'s tiebreak on a partial doc graph.
    """

    source_concept_id: str
    target_concept_id: str
    provenance: frozenset[str]  # ⊆ PROVENANCE_SOURCES
    weight: float  # derived from provenance (Decision 5); deterministic
    n_cooccurrence_chunks: int  # chunk-level count (Decision 4)
    provenance_strength: tuple[tuple[str, float], ...] = ()  # sorted (token, ratio); R4
    stance_by_doc: tuple[tuple[str, str], ...] = ()  # (document_id, polarity) per asserting doc
    relation: str | None = None


@dataclass(frozen=True)
class ConceptNode:
    """A concept node in the analysed skeleton (degree + community filled by analysis)."""

    id: str
    label: str
    doc_ids: tuple[str, ...]
    degree: int
    community: int


@dataclass(frozen=True)
class Community:
    """A Louvain community: a cluster of related concepts (ADR-1)."""

    id: int
    label: str  # the community's highest-degree node label
    node_ids: tuple[str, ...]
    size: int


@dataclass(frozen=True)
class ConceptSkeleton:
    """The assembled skeleton — the regenerable structural payload."""

    nodes: tuple[ConceptNode, ...]
    edges: tuple[SkeletonEdge, ...]
    communities: tuple[Community, ...]
    meta: dict[str, Any]  # n_documents, n_concepts, n_edges, seed, resolution, graph_version


@dataclass(frozen=True)
class SkeletonResult:
    """What a ``build_concept_skeleton`` run produced (for the CLI report)."""

    skeleton: ConceptSkeleton
    n_documents: int
    n_concepts: int
    n_edges: int
    n_isolated: int
    provenance_counts: dict[str, int]  # token -> number of edges carrying it
    applied: bool


# ============================================================
# Pure core — presence, edges, provenance, weight, communities
# ============================================================


def _surface_forms(label: str, alias_list: list[str]) -> list[str]:
    """Case-folded, de-duplicated surface forms for a concept (label + aliases)."""
    seen: set[str] = set()
    forms: list[str] = []
    for raw in (label, *alias_list):
        form = raw.strip().casefold()
        if form and form not in seen:
            seen.add(form)
            forms.append(form)
    return forms


PRESENCE_BOUNDARY = "boundary"
PRESENCE_SUBSTRING = "substring"
PRESENCE_MODES = (PRESENCE_BOUNDARY, PRESENCE_SUBSTRING)


def compile_boundary_pattern(form: str) -> re.Pattern[str]:
    """Word-boundary regex for one casefolded surface form (R2, shared with ``epistemics``).

    Alnum lookarounds — ``(?<![a-z0-9])form(?![a-z0-9])`` over casefolded text — instead of
    ``\\b`` on purpose: ``\\b`` mishandles forms whose edge characters are non-word (``gpt-4``,
    ``c++``), where a trailing ``\\b`` would demand a following word char. ``form`` must already
    be casefolded (the class is lowercase ``[a-z0-9]``). The single definition every
    boundary-matching caller in the codebase shares (``_presence_matchers`` here,
    ``epistemics.concepts_in_text`` for chunk-level attribution — KI-15) so there is one
    boundary-matching behavior, not a second hand-rolled (and possibly diverging) one."""
    return re.compile(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])")


def _presence_matchers(forms: list[str], mode: str) -> list[tuple[str, re.Pattern[str] | None]]:
    """Precompiled per-form matchers for one concept (compiled once, reused per chunk).

    ``boundary`` mode pairs each form with :func:`compile_boundary_pattern`; ``substring`` mode
    carries ``None`` (the caller falls back to ``str.count``)."""
    if mode == PRESENCE_SUBSTRING:
        return [(form, None) for form in forms]
    if mode != PRESENCE_BOUNDARY:
        raise ValueError(f"unknown presence mode {mode!r} (expected one of {PRESENCE_MODES})")
    return [(form, compile_boundary_pattern(form)) for form in forms]


def match_presence(
    concepts: list[tuple[str, str]],
    aliases: dict[str, list[str]],
    chunk_texts: list[tuple[str, str, str]],
    *,
    mode: str = PRESENCE_BOUNDARY,
) -> list[ConceptPresence]:
    """Deterministic presence — case-folded surface-form match of curated concepts.

    ``concepts`` = ``(concept_id, label)``; ``aliases`` maps ``concept_id`` → surface
    forms; ``chunk_texts`` = ``(chunk_key, document_id, text)`` where ``chunk_key`` is the
    ADR-4 composite ``"{document_id}:p{parent_index}"``. A concept is *present* in a chunk
    iff one of its case-folded surface forms occurs in the chunk text (Decision 2 — the
    LLM never decides presence). Returns one ``ConceptPresence`` per ``(concept, document)``
    with ≥ 1 hit, ``chunk_keys`` sorted, ``n_mentions`` = total surface-form occurrences.

    ``mode`` (R2 / RG-009): ``"boundary"`` (default) counts only whole-word (alnum-bounded)
    occurrences, so ``bert`` does **not** fire inside ``sbert`` / ``colbert`` / ``roberta``
    — the substring-inflation that fabricated co-occurrence edges. ``"substring"`` keeps the
    original raw ``str.count`` behaviour as the A/B lever for the RG-008 comparison run.
    Recall is bounded by alias coverage (the curation burden, RG-009).

    Known accepted looseness (reporting-only today): overlapping alias spans double-count
    ``n_mentions`` — e.g. both ``passage retrieval`` and ``dense passage retrieval`` firing on
    one span. Longest-match span consumption is the upgrade if ``n_mentions`` ever gates.
    """
    # Pre-fold chunk texts once.
    folded = [(key, doc_id, text.casefold()) for key, doc_id, text in chunk_texts]
    # (concept_id, document_id) -> {chunk_key: occurrence_count}
    hits: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for concept_id, label in concepts:
        forms = _surface_forms(label, aliases.get(concept_id, []))
        if not forms:
            continue
        matchers = _presence_matchers(forms, mode)
        for chunk_key, doc_id, low in folded:
            count = sum(
                len(pattern.findall(low)) if pattern is not None else low.count(form)
                for form, pattern in matchers
            )
            if count:
                hits[(concept_id, doc_id)][chunk_key] += count
    presences: list[ConceptPresence] = []
    for (concept_id, doc_id), per_chunk in hits.items():
        chunk_keys = tuple(sorted(per_chunk))
        presences.append(
            ConceptPresence(
                concept_id=concept_id,
                document_id=doc_id,
                chunk_keys=chunk_keys,
                n_mentions=sum(per_chunk.values()),
            )
        )
    presences.sort(key=lambda p: (p.concept_id, p.document_id))
    return presences


def _pair(a: str, b: str) -> tuple[str, str]:
    """Canonical (sorted) undirected concept pair."""
    return (a, b) if a <= b else (b, a)


def cooccurrence_edges(
    presences: list[ConceptPresence], *, min_cooccurrence: int
) -> list[SkeletonEdge]:
    """Chunk-level co-occurrence edges (Decision 4 — the primary precision lever).

    Two concepts get an edge when they are co-present in ≥ ``min_cooccurrence`` **chunks**
    (not documents — doc-level co-occurrence on a same-domain corpus saturates into a dense,
    meaningless graph). Each edge starts with ``provenance={"cooccurrence"}`` and its
    chunk-level count; the deterministic ``weight`` follows from ``edge_weight``.
    """
    # chunk_key -> set of concept_ids present in that chunk.
    chunk_concepts: dict[str, set[str]] = defaultdict(set)
    for p in presences:
        for key in p.chunk_keys:
            chunk_concepts[key].add(p.concept_id)
    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    for concept_ids in chunk_concepts.values():
        ordered = sorted(concept_ids)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pair_counts[_pair(a, b)] += 1
    edges: list[SkeletonEdge] = []
    for (a, b), count in pair_counts.items():
        if count < min_cooccurrence:
            continue
        provenance = frozenset({"cooccurrence"})
        edges.append(
            SkeletonEdge(
                source_concept_id=a,
                target_concept_id=b,
                provenance=provenance,
                weight=edge_weight(provenance, count),
                n_cooccurrence_chunks=count,
            )
        )
    edges.sort(key=lambda e: (e.source_concept_id, e.target_concept_id))
    return edges


def _with_strength(
    existing: tuple[tuple[str, float], ...], token: str, strength: float
) -> tuple[tuple[str, float], ...]:
    """Merge ``token`` → ``strength`` into an edge's sorted strength mapping (replace-or-add)."""
    merged = dict(existing)
    merged[token] = strength
    return tuple(sorted(merged.items()))


def _add_provenance(
    edges: list[SkeletonEdge],
    doc_pairs: list[tuple[str, str]],
    concept_doc_index: dict[str, set[str]],
    token: str,
) -> list[SkeletonEdge]:
    """Annotate existing edges with ``token`` + a graded strength (R4 — ratio, not boolean).

    The no-edge-creation invariant (Decision 5): an edge gains the provenance token only
    when some ``(doc_a, doc_b)`` pair connects a document containing one endpoint to a
    document containing the other (either direction). A pair over concepts that are **not**
    already a co-occurrence edge creates **nothing** — this is the density control that
    prevents "every concept in doc X linked to every concept in doc Y".

    R4 graded strength: over the *candidate* endpoint-doc pairs
    ``{(da, db) : da ∈ docs(A), db ∈ docs(B), da ≠ db}``, ``strength = |linked ∩ candidates| /
    |candidates|`` — the fraction actually connected by this token's doc graph. The token is
    kept iff ``strength > 0`` (identical membership to the old boolean "any linked"); the
    strength is recorded on ``provenance_strength`` and folded into the deterministic weight
    tiebreak. On a saturated graph every strength is ``1.0`` by construction; the signal only
    spreads on a partial graph (the multi-domain regime).
    """
    if not doc_pairs:
        return list(edges)
    # Undirected doc-adjacency for quick "are these two docs linked?" tests.
    linked: set[tuple[str, str]] = set()
    for src, tgt in doc_pairs:
        linked.add((src, tgt))
        linked.add((tgt, src))
    out: list[SkeletonEdge] = []
    for e in edges:
        src_docs = concept_doc_index.get(e.source_concept_id, set())
        tgt_docs = concept_doc_index.get(e.target_concept_id, set())
        candidate = 0
        hit = 0
        for da in src_docs:
            for db in tgt_docs:
                if da == db:
                    continue
                candidate += 1
                if (da, db) in linked:
                    hit += 1
        if candidate and hit:
            strength = round(hit / candidate, 6)
            provenance = e.provenance | {token}
            strengths = _with_strength(e.provenance_strength, token, strength)
            out.append(
                replace(
                    e,
                    provenance=provenance,
                    provenance_strength=strengths,
                    weight=edge_weight(provenance, e.n_cooccurrence_chunks, strengths),
                )
            )
        else:
            out.append(e)
    return out


def add_citation_provenance(
    edges: list[SkeletonEdge],
    citation_pairs: list[tuple[str, str]],
    concept_doc_index: dict[str, set[str]],
) -> list[SkeletonEdge]:
    """Add ``"citation"`` to co-occurrence edges whose concepts span a cited doc pair.

    ``citation_pairs`` = ``(source_doc, target_doc)`` from ``Citation``. Never creates an
    edge (Decision 5 — citation annotates the co-occurrence skeleton)."""
    return _add_provenance(edges, citation_pairs, concept_doc_index, "citation")


def add_similarity_provenance(
    edges: list[SkeletonEdge],
    doc_sim_pairs: list[tuple[str, str]],
    concept_doc_index: dict[str, set[str]],
) -> list[SkeletonEdge]:
    """Add ``"similarity"`` to co-occurrence edges whose concepts span a similar doc pair.

    ``doc_sim_pairs`` = ``(source_doc, target_doc)`` from ``DocSimilarity``. Never creates
    an edge (Decision 5)."""
    return _add_provenance(edges, doc_sim_pairs, concept_doc_index, "similarity")


#: Per-source provenance weight; the count of provenance tokens dominates the score so a
#: multi-provenance edge always outranks a co-occurrence-only edge (Decision 5).
_PROVENANCE_WEIGHT: dict[str, float] = {
    "cooccurrence": 1.0,
    "citation": 1.0,
    "similarity": 1.0,
    "llm_relation": 1.0,
}


def edge_weight(
    provenance: frozenset[str],
    n_cooccurrence_chunks: int,
    provenance_strength: tuple[tuple[str, float], ...] = (),
) -> float:
    """Deterministic edge weight from the provenance set + co-occurrence count (Decision 5).

    The integer part is the number of corroborating provenance sources; a bounded fractional
    term in ``[0, 1)`` breaks ties between equal-provenance edges. So a multi-provenance edge
    (≥ 2) always ranks above a co-occurrence-only edge (< 2) — the **locked invariant**: more
    provenance tokens always outrank fewer.

    R4 splits that fractional tiebreak into two bounded halves: ``0.5*mean(strengths) +
    0.5*(1 - 1/(1 + cooc))``. ``mean(strengths)`` is the average graded provenance strength
    (the ``citation`` / ``similarity`` ratios; ``0.0`` when there are none — a co-occurrence-
    only edge), so on a partial doc graph a better-corroborated edge ranks above an equal-token
    edge with weaker links; ``cooc`` still breaks the remaining ties by co-occurring-chunk
    count. Both halves are in ``[0, 1)`` (the strength half tops out at ``0.5``), so the sum
    stays ``< 1`` and the integer part still dominates.
    """
    base = sum(_PROVENANCE_WEIGHT[p] for p in provenance if p in _PROVENANCE_WEIGHT)
    strengths = [s for _, s in provenance_strength]
    mean_strength = sum(strengths) / len(strengths) if strengths else 0.0
    cooc_term = 1.0 - 1.0 / (1.0 + max(n_cooccurrence_chunks, 0))
    tiebreak = 0.5 * mean_strength + 0.5 * cooc_term
    return round(base + tiebreak, 6)


def contested_edges(edges: list[SkeletonEdge]) -> list[SkeletonEdge]:
    """Edges with cross-source stance disagreement (the 7d ``contested`` signal, Decision 7).

    An edge is contested when ≥ 1 document asserts a supporting polarity and ≥ 1 *other*
    document asserts an opposing polarity. Empty after Node A (no stances yet)."""
    out: list[SkeletonEdge] = []
    for e in edges:
        sup = {doc for doc, pol in e.stance_by_doc if pol in SUPPORTING_POLARITIES}
        opp = {doc for doc, pol in e.stance_by_doc if pol in OPPOSING_POLARITIES}
        if sup and opp:
            out.append(e)
    return out


def detect_communities(
    graph: Any, *, algorithm: str = "louvain", seed: int = 42, resolution: float = 1.0
) -> list[set[str]]:
    """Community detection over the weighted concept graph (ADR-1 — Louvain, seeded).

    A seam (``algorithm=``) reserved for a future numpy-2 Leiden backend; ``"louvain"`` is
    the only implemented algorithm (``networkx.louvain_communities`` — deterministic for a
    fixed ``seed``; graspologic / ``leiden_communities`` are unusable, see ADR-1). Returns
    a list of node-id sets; an empty graph yields ``[]``."""
    if algorithm != "louvain":
        raise ValueError(f"Unsupported community algorithm {algorithm!r} (only 'louvain').")
    from networkx.algorithms.community import louvain_communities

    if graph.number_of_nodes() == 0:
        return []
    raw = louvain_communities(graph, weight="weight", resolution=resolution, seed=seed)
    return [set(c) for c in raw]


def _graph_version(
    nodes: list[ConceptNode],
    edges: list[SkeletonEdge],
    *,
    seed: int,
    resolution: float,
    doc_years: dict[str, int] | None = None,
) -> str:
    """A short, timestamp-free fingerprint of the skeleton's structure (byte-stable rebuild).

    Hashes the sorted node ids + canonical edge signatures (endpoints, sorted provenance,
    weight, co-occurrence count, stance) + the community params + (G3) the ``doc_years`` map,
    so a metadata backfill that changes a doc's year busts the cache even though no node/edge
    changed. Identical inputs → identical version → byte-identical ``skeleton.json``
    (Decision 3)."""
    payload = {
        "nodes": sorted(n.id for n in nodes),
        "edges": sorted(
            [
                e.source_concept_id,
                e.target_concept_id,
                sorted(e.provenance),
                e.weight,
                e.n_cooccurrence_chunks,
                [list(s) for s in e.provenance_strength],  # already sorted by token (R4)
                sorted([list(s) for s in e.stance_by_doc]),
                e.relation,
            ]
            for e in edges
        ),
        "seed": seed,
        "resolution": resolution,
        "doc_years": sorted((doc_years or {}).items()),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def analyze_skeleton(
    nodes: list[ConceptNode],
    edges: list[SkeletonEdge],
    *,
    seed: int = 42,
    resolution: float = 1.0,
    meta_extra: dict[str, Any] | None = None,
) -> ConceptSkeleton:
    """Assemble the skeleton: degree, Louvain communities, meta + ``graph_version`` (pure).

    ``nodes`` carry id/label/doc_ids (degree/community are recomputed here). Degree comes
    from the weighted graph; communities from ``detect_communities`` (ordered by size desc
    then first member, enumerated to ids; label = highest-degree member). Deterministic for
    a fixed ``seed``."""
    import networkx as nx

    by_id = {n.id: n for n in nodes}
    graph = nx.Graph()
    for n in nodes:
        graph.add_node(n.id)
    for e in edges:
        graph.add_edge(e.source_concept_id, e.target_concept_id, weight=e.weight)

    degree: dict[str, int] = {nid: int(deg) for nid, deg in graph.degree()}

    raw = detect_communities(graph, seed=seed, resolution=resolution)
    ordered = sorted((sorted(c) for c in raw), key=lambda c: (-len(c), c[0] if c else ""))
    community_of: dict[str, int] = {}
    communities: list[Community] = []
    for cid, members in enumerate(ordered):
        for nid in members:
            community_of[nid] = cid
        label_node = max(members, key=lambda m: (degree.get(m, 0), m)) if members else ""
        label = by_id[label_node].label if label_node in by_id else label_node
        communities.append(
            Community(id=cid, label=label, node_ids=tuple(members), size=len(members))
        )

    analysed_nodes = tuple(
        ConceptNode(
            id=n.id,
            label=n.label,
            doc_ids=n.doc_ids,
            degree=degree.get(n.id, 0),
            community=community_of.get(n.id, -1),
        )
        for n in sorted(nodes, key=lambda n: n.id)
    )
    sorted_edges = tuple(sorted(edges, key=lambda e: (e.source_concept_id, e.target_concept_id)))
    meta: dict[str, Any] = {
        "n_concepts": len(nodes),
        "n_edges": len(edges),
        "seed": seed,
        "resolution": resolution,
    }
    if meta_extra:
        meta.update(meta_extra)
    doc_years = meta.get("doc_years")
    meta["graph_version"] = _graph_version(
        nodes, edges, seed=seed, resolution=resolution, doc_years=doc_years
    )
    return ConceptSkeleton(
        nodes=analysed_nodes,
        edges=sorted_edges,
        communities=tuple(communities),
        meta=meta,
    )


# ============================================================
# Serialisation — provide BOTH directions (the missing-inverse bit Feature 6)
# ============================================================


def skeleton_to_dict(skeleton: ConceptSkeleton) -> dict[str, Any]:
    """Serialise the skeleton to a JSON-ready dict (deterministic / byte-stable)."""
    return {
        "meta": skeleton.meta,
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "doc_ids": list(n.doc_ids),
                "degree": n.degree,
                "community": n.community,
            }
            for n in skeleton.nodes
        ],
        "edges": [
            {
                "source": e.source_concept_id,
                "target": e.target_concept_id,
                "provenance": sorted(e.provenance),
                "provenance_strength": dict(e.provenance_strength),
                "weight": e.weight,
                "n_cooccurrence_chunks": e.n_cooccurrence_chunks,
                "stance": [list(s) for s in e.stance_by_doc],
                "relation": e.relation,
            }
            for e in skeleton.edges
        ],
        "communities": [
            {"id": c.id, "label": c.label, "node_ids": list(c.node_ids), "size": c.size}
            for c in skeleton.communities
        ],
    }


def skeleton_from_dict(data: dict[str, Any]) -> ConceptSkeleton:
    """Inverse of :func:`skeleton_to_dict` — round-trips the structural payload exactly."""
    nodes = tuple(
        ConceptNode(
            id=n["id"],
            label=n["label"],
            doc_ids=tuple(n.get("doc_ids", [])),
            degree=int(n.get("degree", 0)),
            community=int(n.get("community", -1)),
        )
        for n in data.get("nodes", [])
    )
    edges = tuple(
        SkeletonEdge(
            source_concept_id=e["source"],
            target_concept_id=e["target"],
            provenance=frozenset(e.get("provenance", [])),
            weight=float(e.get("weight", 0.0)),
            n_cooccurrence_chunks=int(e.get("n_cooccurrence_chunks", 0)),
            provenance_strength=tuple(
                sorted(
                    (token, float(strength))
                    for token, strength in e.get("provenance_strength", {}).items()
                )
            ),
            stance_by_doc=tuple((s[0], s[1]) for s in e.get("stance", [])),
            relation=e.get("relation"),
        )
        for e in data.get("edges", [])
    )
    communities = tuple(
        Community(
            id=int(c["id"]),
            label=c["label"],
            node_ids=tuple(c.get("node_ids", [])),
            size=int(c.get("size", 0)),
        )
        for c in data.get("communities", [])
    )
    return ConceptSkeleton(
        nodes=nodes, edges=edges, communities=communities, meta=dict(data.get("meta", {}))
    )


# ============================================================
# 7d seam — node weights (the retired concept_graph.NodeWeight contract shape)
# ============================================================


@dataclass(frozen=True)
class NodeWeight:
    """A concept node's structural epistemic weight (Feature 7d) — corroboration, not
    confidence. Derived from incident stated-claim stances (re-homed verbatim from the
    retired ``concept_graph.NodeWeight``, KI-7 retirement).

    * ``coverage`` — ``corroborated`` (>=2 independent supporting docs, no disputes),
      ``unique`` (<=1 supporting doc, no disputes — the *only source on its topic*,
      held NEUTRAL, never down-weighted; Decision 4), ``contested`` (>=1 disputing doc).
    * ``direction`` — ``stable`` (no disputes), ``superseded_trend`` (disputing docs are
      newer than the supporting ones — currency emerges from polarity over time, never
      from absolute age; Decision 1), ``contested`` (disputed but no clear time order).
    """

    node_id: str
    n_supporting_sources: int
    n_contradicting_sources: int
    agreement_ratio: float
    direction: str  # stable | contested | superseded_trend
    coverage: str  # corroborated | unique | contested


def node_weights_for_epistemics(skeleton: ConceptSkeleton) -> dict[str, NodeWeight]:
    """Per-node corroboration weights in the ``NodeWeight`` contract shape (Decision 7).

    The seam 7d re-founds on (``epistemics.project_chunk`` reads ``.coverage`` /
    ``.direction``). Aggregates incident edges' ``stance_by_doc`` into supporting / opposing
    *document* sets, then applies the unique-source = neutral rule **verbatim** from the
    retired ``concept_graph.compute_node_weights``: coverage is decided contested-FIRST, so a
    sole-source concept (no opposing doc) is ``unique``, never ``contested``. Returns a
    weight for **every** node (a stance-less node → ``unique`` / ``stable``).

    G3 (SPRINT-003): a contested node's ``direction`` is ``superseded_trend`` when its
    contradicting docs are, in aggregate, newer than its supporting docs — see
    :func:`_aggregate_direction`. Reads ``skeleton.meta["doc_years"]`` (absent/empty on a
    pre-G3 skeleton → every contested node stays ``contested``, byte-identical to before).
    """
    doc_years: dict[str, int] = skeleton.meta.get("doc_years") or {}
    incident: dict[str, tuple[set[str], set[str]]] = {n.id: (set(), set()) for n in skeleton.nodes}
    for e in skeleton.edges:
        for doc_id, polarity in e.stance_by_doc:
            for endpoint in (e.source_concept_id, e.target_concept_id):
                bucket = incident.get(endpoint)
                if bucket is None:
                    continue
                if polarity in SUPPORTING_POLARITIES:
                    bucket[0].add(doc_id)
                elif polarity in OPPOSING_POLARITIES:
                    bucket[1].add(doc_id)
    weights: dict[str, NodeWeight] = {}
    for n in skeleton.nodes:
        sup, opp = incident[n.id]
        ns, nc = len(sup), len(opp)
        if nc >= 1:
            coverage = "contested"
            direction = _aggregate_direction(sup, opp, doc_years)
        elif ns <= 1:
            coverage = "unique"
            direction = "stable"
        else:
            coverage = "corroborated"
            direction = "stable"
        agreement = round(ns / (ns + nc), 4) if (ns + nc) else 1.0
        weights[n.id] = NodeWeight(
            node_id=n.id,
            n_supporting_sources=ns,
            n_contradicting_sources=nc,
            agreement_ratio=agreement,
            direction=direction,
            coverage=coverage,
        )
    return weights


#: G6 (SPRINT-006): the definitional smallest per-side count for a median to mean "aggregate"
#: rather than a single opinion. A structural confidence floor, NOT an eval-harness tunable —
#: raising it (e.g. to 3, or adding a year-gap/window) would be a knob and is out of scope; see
#: docs/sprints/SPRINT-006-gate-superseded-confidence.md.
MIN_DATED_DOCS_PER_SIDE = 2


def _aggregate_direction(sup: set[str], opp: set[str], doc_years: dict[str, int]) -> str:
    """``contested`` vs ``superseded_trend`` for a node with >= 1 opposing doc (G3, gated G6).

    Relative polarity-over-time, never absolute age (Decision 1, feature-7d spec): compares
    the **median** publication year of the opposing (contradicting) doc set against the
    median of the supporting doc set. ``superseded_trend`` only when the opposing median is
    strictly newer. Fails safe to ``contested`` whenever the comparison isn't well-formed —
    no supporting docs to compare against, or any doc in either set missing a year (never
    guess a superseded call on incomplete year data).

    G6 (SPRINT-006): also requires >= :data:`MIN_DATED_DOCS_PER_SIDE` dated docs on **each**
    side before treating the comparison as a meaningful aggregate — a median of one document
    is not an aggregate, so the single-doc-per-side case (the thinnest possible evidence) stays
    `contested` instead of firing on what is effectively a coin-flip."""
    if (
        len(sup) < MIN_DATED_DOCS_PER_SIDE
        or len(opp) < MIN_DATED_DOCS_PER_SIDE
        or any(d not in doc_years for d in sup | opp)
    ):
        return "contested"
    sup_years = [doc_years[d] for d in sup]
    opp_years = [doc_years[d] for d in opp]
    sup_median = statistics.median(sup_years)
    opp_median = statistics.median(opp_years)
    return "superseded_trend" if opp_median > sup_median else "contested"


# ============================================================
# Feature 6 seam — document clusters from the skeleton's Louvain communities
# ============================================================


def doc_clusters_from_skeleton(
    skeleton: ConceptSkeleton, document_ids: list[str]
) -> list[list[str]]:
    """Group documents by the concept-community they most belong to (Decision 8).

    The wiki cluster seam (retired ``concept_graph.doc_clusters_from_graph``): every
    node's ``doc_ids`` casts a vote for its community; a document's dominant community
    (most-voted, ties broken by lowest community id) determines its cluster. A document
    with no community vote (no curated concept present, or only isolated/degree-0
    concepts) becomes its own singleton cluster — mirroring the old "solo" bucket.
    Deterministic; reads only the skeleton's own node membership, not a separate per-doc
    cache (the skeleton carries none — it is rebuilt fresh from current DB/Chroma state
    every run, unlike the retired per-doc-hash LLM extraction cache).
    """
    votes: dict[str, Counter[int]] = defaultdict(Counter)
    for n in skeleton.nodes:
        if n.community < 0:
            continue
        for doc_id in n.doc_ids:
            votes[doc_id][n.community] += 1
    by_community: dict[str, list[str]] = defaultdict(list)
    for doc_id in document_ids:
        doc_votes = votes.get(doc_id)
        if not doc_votes:
            by_community[f"solo:{doc_id}"].append(doc_id)
            continue
        dominant = min(doc_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        by_community[f"c{dominant}"].append(doc_id)
    clusters = [sorted(docs) for docs in by_community.values()]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


# ============================================================
# Impure boundary — SQLite + Chroma reads, sidecar I/O
# ============================================================


def load_concepts() -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    """Read the curated vocabulary: ``[(concept_id, label)]`` + ``{concept_id: [alias]}``.

    Materialised into plain tuples inside the session (no detached-ORM access downstream)
    so the pure core stays DB-free. Empty vocabulary → empty graph (the curation prereq)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept, ConceptAlias
    from doc_assistant.db.session import session_scope

    concepts: list[tuple[str, str]] = []
    aliases: dict[str, list[str]] = defaultdict(list)
    with session_scope() as session:
        for row in session.execute(select(Concept)).scalars():
            concepts.append((str(row.id), row.label))
        for arow in session.execute(select(ConceptAlias)).scalars():
            aliases[str(arow.concept_id)].append(arow.alias)
    concepts.sort(key=lambda c: c[0])
    return concepts, dict(aliases)


@dataclass(frozen=True)
class KeywordCandidate:
    """A vocabulary candidate mined from a ``Keyword`` row (Decision 1 — candidate only)."""

    name: str
    promoted: bool  # a Concept with this label already exists


def list_keyword_candidates() -> list[KeywordCandidate]:
    """Existing ``Keyword`` rows as vocabulary candidates + whether each is promoted.

    A Keyword is a *candidate only* — never auto-written as a Concept (Decision 1); the user
    promotes one with :func:`promote_keyword`. ``promoted`` reflects whether a Concept with
    the same label already exists. Free, no LLM (the LLM dedupe/label pass is deferred)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept, Keyword
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        keywords = [k.name for k in session.execute(select(Keyword)).scalars()]
        concept_labels = {c.label for c in session.execute(select(Concept)).scalars()}
    return [
        KeywordCandidate(name=name, promoted=name in concept_labels)
        for name in sorted(set(keywords))
    ]


def promote_keyword(name: str) -> str | None:
    """Promote a ``Keyword`` to a curated ``Concept`` (+ a seed alias). Idempotent.

    Returns the Concept id, or ``None`` if no Keyword has that name. Get-or-create by label
    so promoting twice is a no-op (one Concept, one seed alias). The LLM is never involved
    (Decision 1)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept, ConceptAlias, Keyword
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        keyword = session.execute(select(Keyword).where(Keyword.name == name)).scalar_one_or_none()
        if keyword is None:
            return None
        existing = session.execute(
            select(Concept).where(Concept.label == name)
        ).scalar_one_or_none()
        if existing is not None:
            return str(existing.id)
        concept = Concept(label=name, source="keyword")
        session.add(concept)
        session.flush()
        session.add(ConceptAlias(concept_id=concept.id, alias=name))
        return str(concept.id)


def add_concept(
    label: str,
    *,
    definition: str | None = None,
    aliases: list[str] | None = None,
) -> str:
    """Create or update a **curated** ``Concept`` (a glossary entry). Idempotent.

    The direct-curation counterpart to :func:`promote_keyword` (which requires a mined
    ``Keyword`` row). Get-or-create by ``label`` with ``source="manual"``; on re-add it
    fills/updates the ``definition`` and adds any new ``aliases`` (never removes one). The
    label is always an implicit surface form for presence; ``aliases`` are extra synonyms.
    Returns the Concept id. Zero LLM.
    """
    from sqlalchemy import select

    from doc_assistant.db.models import Concept, ConceptAlias
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        concept = session.execute(
            select(Concept).where(Concept.label == label)
        ).scalar_one_or_none()
        if concept is None:
            concept = Concept(label=label, source="manual", definition=definition)
            session.add(concept)
            session.flush()
        elif definition is not None:
            concept.definition = definition
        existing = {a.alias for a in concept.aliases}
        for alias in aliases or []:
            if alias and alias not in existing:
                session.add(ConceptAlias(concept_id=concept.id, alias=alias))
                existing.add(alias)
        return str(concept.id)


def remove_alias(concept_id: str, alias: str) -> bool:
    """Remove one ``ConceptAlias`` row from a ``Concept``. Returns True if a row was removed.

    Counterpart to :func:`add_concept`'s alias-adding (which never removes one). Case-sensitive
    exact match on ``alias``; a missing concept or a no-match alias both return False."""
    from sqlalchemy import select

    from doc_assistant.db.models import ConceptAlias
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        row = session.execute(
            select(ConceptAlias).where(
                ConceptAlias.concept_id == concept_id, ConceptAlias.alias == alias
            )
        ).scalar_one_or_none()
        if row is None:
            return False
        session.delete(row)
        return True


def delete_concept(concept_id: str) -> bool:
    """Delete a ``Concept`` and its aliases. Returns True if it existed.

    The FK (``ondelete="CASCADE"``) + ORM cascade both drop ``ConceptAlias`` rows; no manual
    cleanup needed."""
    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return False
        session.delete(concept)
        return True


def rename_concept(concept_id: str, new_label: str) -> bool:
    """Rename a ``Concept``'s label in place. Returns True if it existed.

    No uniqueness check against other labels (matches :func:`add_concept`'s own get-or-create,
    which is also unenforced at the DB level) — callers that need "no two families share a
    canonical name" enforce it themselves."""
    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        concept = session.get(Concept, concept_id)
        if concept is None:
            return False
        concept.label = new_label
        return True


@dataclass(frozen=True)
class GlossaryEntry:
    """A curated concept as a glossary row: label + definition + synonyms."""

    label: str
    definition: str | None
    aliases: list[str]
    source: str


def load_glossary() -> list[GlossaryEntry]:
    """All curated concepts as glossary entries (label, definition, aliases), by label."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        concepts = list(session.execute(select(Concept)).scalars())
        entries = [
            GlossaryEntry(
                label=c.label,
                definition=c.definition,
                aliases=sorted(a.alias for a in c.aliases if a.alias != c.label),
                source=c.source,
            )
            for c in concepts
        ]
    entries.sort(key=lambda e: e.label.casefold())
    return entries


def load_presence_inputs(document_ids: list[str] | None = None) -> list[tuple[str, str, str]]:
    """Parent-chunk text for presence matching: ``[(chunk_key, document_id, text)]``.

    Reads the parent-child Chroma store (``PC_CHROMA_PATH``), de-duplicates child rows to
    one entry per parent via ``parent_index`` (the parent text is denormalised onto every
    child), and builds the ADR-4 composite key ``"{document_id}:p{parent_index}"``. Runs on
    the host, not the sandbox (KI-5). Returns ``[]`` if Chroma / the collection is absent."""
    from doc_assistant.config import PC_CHROMA_PATH
    from doc_assistant.embeddings import get_collection_name

    try:
        import chromadb
    except ImportError:  # pragma: no cover - dep present in dev env
        return []

    client = chromadb.PersistentClient(path=PC_CHROMA_PATH)
    try:
        coll = client.get_collection(get_collection_name())
    except Exception:
        log.warning("no_pc_collection", hint="run ingest first; skeleton presence will be empty")
        return []

    where: Any = {"document_id": {"$in": document_ids}} if document_ids else None
    data = coll.get(where=where, include=["metadatas"])
    metadatas = data.get("metadatas") or []
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, str, str]] = []
    for meta in metadatas:
        if not isinstance(meta, dict):
            continue
        document_id = meta.get("document_id")
        parent_index = meta.get("parent_index")
        parent_text = meta.get("parent_text")
        if document_id is None or parent_index is None or not parent_text:
            continue
        key_tuple = (str(document_id), int(parent_index))
        if key_tuple in seen:
            continue
        seen.add(key_tuple)
        chunk_key = f"{document_id}:p{int(parent_index)}"
        out.append((chunk_key, str(document_id), str(parent_text)))
    out.sort(key=lambda t: t[0])
    return out


def load_doc_graphs() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Read the existing zero-LLM doc graphs: ``(citation_pairs, doc_sim_pairs)``.

    ``citation_pairs`` = resolved ``Citation(source → target)`` (target must be a known
    document); ``doc_sim_pairs`` = ``DocSimilarity(source, target)`` edges. Both as
    ``(source_doc_id, target_doc_id)`` tuples for the provenance annotators."""
    from sqlalchemy import select

    from doc_assistant.db.models import Citation, DocSimilarity
    from doc_assistant.db.session import session_scope

    citation_pairs: list[tuple[str, str]] = []
    doc_sim_pairs: list[tuple[str, str]] = []
    with session_scope() as session:
        stmt = select(Citation.source_document_id, Citation.target_document_id).where(
            Citation.target_document_id.is_not(None)
        )
        for src, tgt in session.execute(stmt):
            if src and tgt:
                citation_pairs.append((str(src), str(tgt)))
        sim_stmt = select(DocSimilarity.source_document_id, DocSimilarity.target_document_id)
        for src, tgt in session.execute(sim_stmt):
            if src and tgt:
                doc_sim_pairs.append((str(src), str(tgt)))
    return citation_pairs, doc_sim_pairs


def load_doc_years() -> dict[str, int]:
    """Read ``Document.year`` for every document that has one (zero-LLM, deterministic).

    G3 (SPRINT-003): threaded into the skeleton at the ``meta`` level (``doc_years``), not
    onto ``ConceptNode``, to keep the blast radius to serialisation. Docs with no year are
    simply absent from the map — ``node_weights_for_epistemics`` fails safe to ``contested``
    whenever a required year is missing."""
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    years: dict[str, int] = {}
    with session_scope() as session:
        stmt = select(Document.id, Document.year).where(Document.year.is_not(None))
        for doc_id, year in session.execute(stmt):
            if doc_id is not None and year is not None:
                years[str(doc_id)] = int(year)
    return years


def _concept_doc_index(presences: list[ConceptPresence]) -> dict[str, set[str]]:
    """``concept_id`` → set of documents it is present in (for provenance annotation)."""
    index: dict[str, set[str]] = defaultdict(set)
    for p in presences:
        index[p.concept_id].add(p.document_id)
    return index


def _write_skeleton_rows(
    skeleton: ConceptSkeleton, presences: list[ConceptPresence], version: str
) -> None:
    """Replace the derived sidecar tables with this run's rows (idempotent, two-lifecycle).

    Drops + rebuilds ``concept_edges`` / ``concept_presence`` only — the curated
    ``concepts`` / ``concept_aliases`` are never touched (Decision 8 persistence split)."""
    from sqlalchemy import delete

    from doc_assistant.db.models import ConceptEdge, ConceptPresenceRow
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(ConceptEdge))
        session.execute(delete(ConceptPresenceRow))
        session.add_all(
            ConceptEdge(
                source_concept_id=e.source_concept_id,
                target_concept_id=e.target_concept_id,
                provenance_json=json.dumps(sorted(e.provenance)),
                strength_json=(
                    json.dumps(dict(e.provenance_strength)) if e.provenance_strength else None
                ),
                weight=e.weight,
                n_cooccurrence_chunks=e.n_cooccurrence_chunks,
                relation=e.relation,
                stance_json=(
                    json.dumps([list(s) for s in e.stance_by_doc]) if e.stance_by_doc else None
                ),
                graph_version=version,
            )
            for e in skeleton.edges
        )
        session.add_all(
            ConceptPresenceRow(
                concept_id=p.concept_id,
                document_id=p.document_id,
                chunk_keys_json=json.dumps(list(p.chunk_keys)),
                n_mentions=p.n_mentions,
                graph_version=version,
            )
            for p in presences
        )


def _write_skeleton_json(skeleton: ConceptSkeleton, skeleton_dir: Path) -> None:
    """Write the ``skeleton.json`` sidecar artifact (deterministic / byte-stable)."""
    skeleton_dir.mkdir(parents=True, exist_ok=True)
    path = skeleton_dir / SKELETON_NAME
    path.write_text(
        json.dumps(skeleton_to_dict(skeleton), indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def write_skeleton(
    skeleton: ConceptSkeleton,
    presences: list[ConceptPresence],
    *,
    skeleton_dir: Path | None = None,
) -> None:
    """Persist an (enriched) skeleton to the derived sidecar — the Node B write seam.

    Public wrapper over the two derived-table + ``skeleton.json`` writers so the deferred
    Node-B runner (``concept_skeleton_enrich``) re-writes the *same* sidecar Node A does,
    idempotently and via one code path (Enrichment-Layer Pattern). ``version`` comes from the
    skeleton's own ``graph_version`` so edges/presence rows and ``skeleton.json`` stay in sync.
    """
    from doc_assistant.config import CONCEPT_SKELETON_DIR

    root = skeleton_dir or CONCEPT_SKELETON_DIR
    version = str(skeleton.meta.get("graph_version", ""))
    _write_skeleton_rows(skeleton, presences, version)
    _write_skeleton_json(skeleton, root)


def load_skeleton(*, skeleton_dir: Path | None = None) -> ConceptSkeleton | None:
    """Read the ``skeleton.json`` sidecar back — the read half of :func:`write_skeleton`.

    Returns ``None`` when the artifact does not exist. **Absent is the normal first run, not an
    error**: ``skeleton.json`` is a regenerable sidecar and is gitignored, so a fresh clone has
    none until ``build_concept_skeleton --apply`` runs (Enrichment-Layer Pattern). Callers render
    an empty state and offer a rebuild; they must not treat ``None`` as a failure.

    A file that exists but cannot be read or parsed *is* an error — that means a corrupt or
    truncated artifact, which is not the same thing as "never built", and silently returning
    ``None`` would invite a rebuild that masks the real problem.
    """
    from doc_assistant.config import CONCEPT_SKELETON_DIR

    root = skeleton_dir or CONCEPT_SKELETON_DIR
    path = root / SKELETON_NAME
    if not path.is_file():
        log.info("skeleton_absent", path=str(path))
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RuntimeError(f"concept skeleton at {path} is unreadable") from e
    return skeleton_from_dict(data)


def build_concept_skeleton(
    *,
    apply: bool,
    force: bool = False,
    min_cooccurrence: int | None = None,
    seed: int | None = None,
    resolution: float = 1.0,
    presence_mode: str | None = None,
    document_ids: list[str] | None = None,
    concept_loader: Any = None,
    presence_loader: Any = None,
    doc_graph_loader: Any = None,
    doc_years_loader: Any = None,
    skeleton_dir: Path | None = None,
) -> SkeletonResult:
    """Build the deterministic concept skeleton (Node A) — **zero LLM calls**.

    Two-pass: load curated vocabulary → deterministic presence → chunk-level co-occurrence
    edges → citation / similarity provenance (annotate-never-create) → Louvain analysis →
    (if ``apply``) replace the derived sidecar tables + write ``skeleton.json``. A dry run
    (``apply=False``) computes + reports but writes nothing. Idempotent: same vocabulary +
    same corpus → byte-identical ``skeleton.json`` and identical rows. The curated
    ``concepts`` / ``concept_aliases`` are read-only here; only the derived tables are
    rebuilt. ``force`` is accepted for CLI/Node-B parity (Node A always rebuilds the derived
    tables on ``apply``). The ``*_loader`` seams are DI hooks for testing without a DB/Chroma.

    G3 (SPRINT-003): also loads ``Document.year`` and attaches it to ``skeleton.meta["doc_years"]``
    (once, at build time) so ``node_weights_for_epistemics`` can compare supporting- vs
    contradicting-document years for a contested node. A year-less corpus yields an empty map —
    ``direction`` stays ``contested``/``stable`` exactly as before this sprint (back-compat).
    """
    from doc_assistant.config import (
        CONCEPT_SKELETON_DIR,
        CONCEPT_SKELETON_MIN_COOCCURRENCE,
        CONCEPT_SKELETON_PRESENCE_MODE,
        CONCEPT_SKELETON_SEED,
    )

    min_cooc = CONCEPT_SKELETON_MIN_COOCCURRENCE if min_cooccurrence is None else min_cooccurrence
    seed_val = CONCEPT_SKELETON_SEED if seed is None else seed
    mode = CONCEPT_SKELETON_PRESENCE_MODE if presence_mode is None else presence_mode
    root = skeleton_dir or CONCEPT_SKELETON_DIR
    _ = force  # Node A always rebuilds the derived tables; force is reserved for Node B.

    load_c = concept_loader or load_concepts
    load_p = presence_loader or load_presence_inputs
    load_g = doc_graph_loader or load_doc_graphs
    load_y = doc_years_loader or load_doc_years

    concepts, aliases = load_c()
    chunk_texts = load_p(document_ids)
    citation_pairs, doc_sim_pairs = load_g()
    doc_years = load_y()

    presences = match_presence(concepts, aliases, chunk_texts, mode=mode)
    doc_index = _concept_doc_index(presences)

    edges = cooccurrence_edges(presences, min_cooccurrence=min_cooc)
    edges = add_citation_provenance(edges, citation_pairs, doc_index)
    edges = add_similarity_provenance(edges, doc_sim_pairs, doc_index)

    nodes = [
        ConceptNode(
            id=cid,
            label=label,
            doc_ids=tuple(sorted(doc_index.get(cid, set()))),
            degree=0,
            community=-1,
        )
        for cid, label in concepts
    ]
    n_documents = len({p.document_id for p in presences})
    skeleton = analyze_skeleton(
        nodes,
        edges,
        seed=seed_val,
        resolution=resolution,
        meta_extra={
            "n_documents": n_documents,
            "min_cooccurrence": min_cooc,
            "doc_years": doc_years,
        },
    )

    provenance_counts: dict[str, int] = {token: 0 for token in PROVENANCE_SOURCES}
    for e in skeleton.edges:
        for token in e.provenance:
            provenance_counts[token] = provenance_counts.get(token, 0) + 1
    n_isolated = sum(1 for n in skeleton.nodes if n.degree == 0)

    if apply:
        version = str(skeleton.meta["graph_version"])
        _write_skeleton_rows(skeleton, presences, version)
        _write_skeleton_json(skeleton, root)

    return SkeletonResult(
        skeleton=skeleton,
        n_documents=n_documents,
        n_concepts=len(skeleton.nodes),
        n_edges=len(skeleton.edges),
        n_isolated=n_isolated,
        provenance_counts=provenance_counts,
        applied=apply,
    )
