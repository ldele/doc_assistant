"""Cross-document concept graph (Phase 7 / Feature 7, PR 16).

A concept/entity graph *across* the library — the layer above Feature 6's
document clustering. Where Feature 6 groups whole documents into topics, this
relates the **concepts inside** them: nodes = concepts, edges = relations,
clustered into communities (Louvain) with high-degree "god nodes" surfaced. It
is the substrate real gap detection needs, and the threshold-free clustering
primitive Feature 6's absolute-cosine union-find should eventually re-point at
(see docs/decisions.md → Deferred Improvements).

Architecture (Enrichment-Layer Pattern): post-ingest, idempotent, **sidecar** —
a ``graph.json`` artifact + a per-document extraction cache under
``CONCEPT_GRAPH_DIR``, never the chunk store. **Not** a graph database
(Stonebraker: a graph DBMS is rarely the performant choice; this is build-time
structure, not a query server) — NetworkX in memory + a JSON file.

Integrity, not confidence. Every edge carries one structural tag, derived from
the corpus's own facts — never a model's self-reported confidence:

* ``EXTRACTED``  — at least one document stated this relation, and every document
  that stated it agrees on a single relation phrase.
* ``AMBIGUOUS``  — documents describe the same concept pair with *conflicting*
  relation phrases (>= 2 distinct phrases): the corpus is structurally
  inconsistent about the link.
* ``INFERRED``   — no document stated the link, but the two concepts co-occur in
  >= ``min_cooccurrence`` documents: they travel together though no single paper
  related them.

Local-first by default. Extraction is a per-document LLM batch over the whole
library — exactly the operation that silently burns API credits if it inherits
the analysis provider default. So it defaults to **local Ollama** explicitly
(``CONCEPT_GRAPH_LLM_PROVIDER``), not ``LLM_PROVIDER``; an Anthropic run is
opt-in. Extraction uses the normalized ``llm.LLMClient`` protocol, so the same
code runs on either backend.

Design split mirrors the other enrichment modules: a pure core (canonicalise,
parse, merge into nodes/edges + integrity tagging, Louvain communities, god-node
ranking, graph-gap signals, JSON render) behind a thin impure layer (sample
chunk text, call the model, cache per-doc, orchestrate the build).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from doc_assistant.config import (
    CONCEPT_GRAPH_CHUNK_CHARS,
    CONCEPT_GRAPH_CHUNK_SAMPLE,
    CONCEPT_GRAPH_DIR,
    CONCEPT_GRAPH_GOD_NODES,
    CONCEPT_GRAPH_MAX_TOKENS,
    CONCEPT_GRAPH_MIN_COOCCURRENCE,
    CONCEPT_GRAPH_SEED,
)

log = structlog.get_logger(__name__)

#: The structural edge-integrity vocabulary (reused by Feature 7d's epistemics
#: layer). A tuple, not an enum, to match this project's ``reviewer.FAILURE_TAGS``
#: convention. NOT a confidence score — see the module docstring.
INTEGRITY_TAGS: tuple[str, ...] = ("EXTRACTED", "INFERRED", "AMBIGUOUS")

#: Relation polarity (Feature 7d) — the *epistemic* axis of a stated claim, distinct
#: from the relation phrase (the semantic axis). Captured per supporting document so
#: corroboration vs contradiction is countable. ``supports`` is the neutral default
#: when extraction omits/garbles it (the common, least-surprising case). NOT
#: self-reported confidence — it's "does this doc affirm or dispute the link".
POLARITIES: tuple[str, ...] = ("supports", "refines", "contradicts", "supersedes")
_POLARITY_DEFAULT = "supports"
#: Which polarities corroborate a claim vs dispute it (used by ``compute_node_weights``).
SUPPORTING_POLARITIES: frozenset[str] = frozenset({"supports", "refines"})
OPPOSING_POLARITIES: frozenset[str] = frozenset({"contradicts", "supersedes"})

GRAPH_NAME = "graph.json"
EXTRACTIONS_DIRNAME = "extractions"


# ============================================================
# Data classes
# ============================================================


@dataclass
class Triple:
    """One extracted relation: ``subject`` --(``relation``)--> ``object``.

    ``subject`` / ``object`` are canonical concept keys; ``relation`` is a short
    normalized verb phrase. ``polarity`` (Feature 7d) is the epistemic axis — one
    of ``POLARITIES``, defaulting to ``supports`` when extraction omits it. Self-loops
    and empty endpoints are dropped on parse.
    """

    subject: str
    relation: str
    object: str
    polarity: str = _POLARITY_DEFAULT


@dataclass
class DocExtraction:
    """One document's extracted concepts + relations (the per-doc cache payload).

    ``year`` (Feature 7d) is the document's publication year (from metadata, may be
    ``None``); it is used *only* for relative polarity ordering — never as an
    absolute age input (decisions.md → Feature 7d, Decision 1)."""

    doc_id: str
    doc_hash: str
    filename: str
    concepts: list[str]
    triples: list[Triple]
    year: int | None = None


@dataclass
class ConceptNode:
    """A concept node in the merged corpus graph."""

    id: str  # canonical key, also the node id
    label: str  # display surface form (the most common original casing)
    doc_ids: list[str] = field(default_factory=list)
    mentions: int = 0
    degree: int = 0
    community: int = -1
    god_node: bool = False


@dataclass(frozen=True)
class EdgeSupport:
    """One document's stance on a stated claim (Feature 7d).

    ``(doc_id, polarity, year)`` — the per-source epistemic record corroboration is
    counted from. Frozen + hashable so duplicate (doc, polarity) stances dedupe in a
    set. Only stated edges (EXTRACTED/AMBIGUOUS) carry these; INFERRED edges don't.
    """

    doc_id: str
    polarity: str
    year: int | None


@dataclass
class ConceptEdge:
    """A relation between two concept nodes, with a structural integrity tag."""

    source: str  # node id (the alphabetically-first of the pair, for determinism)
    target: str  # node id
    relations: list[str]  # distinct normalized relation phrases (empty for INFERRED)
    doc_ids: list[str]  # documents supporting the edge
    weight: int  # number of supporting documents
    integrity: str  # one of INTEGRITY_TAGS
    # Feature 7d: per-document (doc_id, polarity, year) stances. Empty for INFERRED
    # edges (co-occurrence is not a stated claim). The epistemic substrate for
    # compute_node_weights; left empty when polarity wasn't extracted (old caches).
    support: list[EdgeSupport] = field(default_factory=list)


@dataclass
class Community:
    """A Louvain community: a cluster of related concepts."""

    id: int
    label: str  # the community's highest-degree node label
    node_ids: list[str]
    size: int


@dataclass
class GraphGaps:
    """Structural graph-gap signals (Feature 7c) — countable, no LLM opinion."""

    isolated_nodes: list[str]  # degree-0 concepts (mentioned, never related)
    thin_bridges: list[tuple[str, str]]  # single edges whose loss disconnects the graph


@dataclass
class ConceptGraph:
    """The merged corpus graph + its analysis (the ``graph.json`` payload)."""

    nodes: list[ConceptNode]
    edges: list[ConceptEdge]
    communities: list[Community]
    god_nodes: list[str]
    gaps: GraphGaps
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def integrity_summary(self) -> dict[str, int]:
        out = {tag: 0 for tag in INTEGRITY_TAGS}
        for e in self.edges:
            out[e.integrity] = out.get(e.integrity, 0) + 1
        return out


@dataclass(frozen=True)
class NodeWeight:
    """A concept node's structural epistemic weight (Feature 7d) — corroboration, not
    confidence. Derived from incident stated-claim ``EdgeSupport`` records.

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


# ============================================================
# Pure core — canonicalisation & parsing
# ============================================================

_WS_RE = re.compile(r"\s+")
_EDGE_PUNCT_RE = re.compile(r"^[^\w]+|[^\w]+$")


def canonical_key(name: str) -> str:
    """Canonical match key for a concept: lowercase, collapse whitespace, trim
    surrounding punctuation. Conservative — no stemming / acronym expansion, so
    distinct concepts never wrongly merge (over-merging is the worse failure)."""
    text = _WS_RE.sub(" ", str(name).strip().lower())
    return _EDGE_PUNCT_RE.sub("", text).strip()


def normalize_relation(relation: str) -> str:
    """Normalize a relation phrase for distinct-phrase counting (the integrity tag)."""
    return _WS_RE.sub(" ", str(relation).strip().lower())


#: Closed relation vocabulary (salvaged from the parallel PR-16 branch). Snapping
#: to it stops benign phrasing variants ("improves" vs "enhances") from inflating
#: a pair's distinct-relation count and manufacturing spurious AMBIGUOUS edges;
#: out-of-vocab verbs fall back to ``related_to`` (never silently dropped).
RELATION_VERBS = ("uses", "extends", "part_of", "compares_to", "defined_as", "related_to")
_RELATION_FALLBACK = "related_to"


def snap_relation(relation: str) -> str:
    """Snap an extracted relation verb to ``RELATION_VERBS`` (fallback ``related_to``)."""
    key = _WS_RE.sub(" ", str(relation).strip().casefold()).replace(" ", "_")
    return key if key in RELATION_VERBS else _RELATION_FALLBACK


def snap_polarity(polarity: object) -> str:
    """Snap an extracted polarity to ``POLARITIES`` (fallback ``supports``) (Feature 7d).

    Tolerant of common synonyms a local model emits (``contradicts``/``disputes``/
    ``conflicts`` → ``contradicts``; ``supersedes``/``replaces``/``obsoletes`` →
    ``supersedes``; ``refines``/``extends``/``improves`` → ``refines``). Anything else
    → ``supports`` — the neutral default, so noise never manufactures a dispute."""
    key = _WS_RE.sub(" ", str(polarity).strip().casefold())
    if key in POLARITIES:
        return key
    if key in {"disputes", "conflicts", "contradict", "refutes", "disagrees"}:
        return "contradicts"
    if key in {"replaces", "obsoletes", "deprecates", "supersede"}:
        return "supersedes"
    if key in {"extends", "improves", "enhances", "refine", "builds_on", "builds on"}:
        return "refines"
    return _POLARITY_DEFAULT


def community_id_for(node_ids: list[str]) -> str:
    """Stable, membership-derived community key (mirrors ``wiki.topic_id_for``).

    Hashes the sorted members only: the same membership yields the same key
    (idempotent / drift-stable), a membership change yields a new key, but
    flipping an edge's integrity tag with identical membership does NOT drift it.
    Salvaged from the parallel PR-16 branch.
    """
    joined = ",".join(sorted(node_ids))
    return "community-" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:10]


def parse_extraction(
    raw: str, *, doc_id: str, doc_hash: str, filename: str, year: int | None = None
) -> DocExtraction:
    """Parse the model's JSON into a ``DocExtraction`` — tolerant by design.

    Two paths: a fast path parses valid JSON; a salvage fallback recovers the
    complete ``concepts`` / ``relations`` elements from truncated or prose-wrapped
    output (common from local models). Concepts and triple endpoints are
    canonicalised, relation verbs snapped to the closed vocab, relation polarity
    (Feature 7d) snapped to ``POLARITIES`` (default ``supports``), self-loops / empty
    endpoints / duplicates dropped. An unsalvageable completion → empty extraction.
    ``year`` is recorded on the extraction for relative polarity ordering only.
    """
    text = _extract_json(raw)
    try:
        data: Any = json.loads(text)
    except Exception:
        data = None

    if isinstance(data, dict):
        concepts_raw: list[Any] = list(data.get("concepts") or [])
        relations_raw: list[Any] = list(data.get("relations") or [])
    else:
        concepts_raw = list(_salvage_strings(text, "concepts"))
        relations_raw = list(_salvage_array(text, "relations"))
        if not concepts_raw and not relations_raw:
            log.warning("concept_extraction_parse_failed", filename=filename)

    concepts: list[str] = []
    seen_c: set[str] = set()
    for c in concepts_raw:
        # Tolerate both bare-string concepts and {"name": ...} objects.
        name = c if isinstance(c, str) else (c.get("name", "") if isinstance(c, dict) else "")
        key = canonical_key(name)
        if key and key not in seen_c:
            seen_c.add(key)
            concepts.append(key)

    triples: list[Triple] = []
    seen_t: set[tuple[str, str, str, str]] = set()
    for t in relations_raw:
        if not isinstance(t, dict):
            continue
        subj, obj = canonical_key(t.get("subject", "")), canonical_key(t.get("object", ""))
        rel = snap_relation(t.get("relation", ""))
        pol = snap_polarity(t.get("polarity", ""))
        if not subj or not obj or subj == obj:
            continue
        sig = (subj, rel, obj, pol)
        if sig in seen_t:
            continue
        seen_t.add(sig)
        triples.append(Triple(subject=subj, relation=rel, object=obj, polarity=pol))
        # A triple's endpoints are concepts even if the model omitted them.
        for key in (subj, obj):
            if key not in seen_c:
                seen_c.add(key)
                concepts.append(key)

    return DocExtraction(doc_id, doc_hash, filename, concepts, triples, year=year)


def _extract_json(text: str) -> str:
    """Best-effort: strip a markdown fence, else take the outermost ``{...}`` span.

    (Same shape as ``wiki._extract_json`` — Ollama's ``format="json"`` returns a
    bare object, but the Anthropic path may wrap it.)"""
    t = text.strip()
    if t.startswith("```"):
        lines = t.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines).strip()
    if not t.startswith("{"):
        start, end = t.find("{"), t.rfind("}")
        if 0 <= start < end:
            t = t[start : end + 1]
    return t


# --- Truncation-tolerant salvage (local models routinely cut JSON short) -----
# Salvaged from the parallel PR-16 branch: recover the complete leading elements
# of a ``"key": [ ... ]`` array even when the array (or the whole completion) is
# truncated mid-way, so a long extraction isn't lost wholesale to one bad tail.
_STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')


def _scan_objects(text: str, array_start: int) -> list[dict[str, Any]]:
    """From the ``[`` at ``array_start``, return each complete top-level ``{...}``
    object until the array closes or the text truncates. String-aware so braces
    inside string values don't miscount."""
    out: list[dict[str, Any]] = []
    i, n = array_start, len(text)
    while i < n:
        while i < n and text[i] not in "{]":
            i += 1
        if i >= n or text[i] == "]":
            break
        depth, j, in_str, esc, complete = 0, i, False, False, False
        while j < n:
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    complete = True
                    break
            j += 1
        if not complete:
            break
        try:
            parsed = json.loads(text[i : j + 1])
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            out.append(parsed)
        i = j + 1
    return out


def _array_segment(text: str, key: str) -> str | None:
    """The substring from the ``[`` of a ``"key": [`` to its (maybe missing) ``]``."""
    k = text.find(f'"{key}"')
    if k == -1:
        return None
    bracket = text.find("[", k)
    if bracket == -1:
        return None
    end = text.find("]", bracket)
    return text[bracket : end + 1] if end != -1 else text[bracket:]


def _salvage_array(text: str, key: str) -> list[dict[str, Any]]:
    """Best-effort recover the object elements of a ``"key": [ {...}, ... ]`` array."""
    k = text.find(f'"{key}"')
    if k == -1:
        return []
    bracket = text.find("[", k)
    return _scan_objects(text, bracket) if bracket != -1 else []


def _salvage_strings(text: str, key: str) -> list[str]:
    """Best-effort recover the quoted-string elements of a ``"key": ["a","b"]`` array."""
    segment = _array_segment(text, key)
    return _STRING_RE.findall(segment) if segment else []


# ============================================================
# Pure core — merge into a corpus graph + integrity tagging
# ============================================================


def _pair(a: str, b: str) -> tuple[str, str]:
    """Undirected edge key: the two endpoints, alphabetically ordered."""
    return (a, b) if a <= b else (b, a)


def build_nodes_edges(
    extractions: list[DocExtraction],
    *,
    min_cooccurrence: int = CONCEPT_GRAPH_MIN_COOCCURRENCE,
) -> tuple[list[ConceptNode], list[ConceptEdge]]:
    """Merge per-document extractions into corpus nodes + integrity-tagged edges.

    Nodes aggregate every concept (and triple endpoint) across documents. Edges
    come from two sources: stated triples (``EXTRACTED`` / ``AMBIGUOUS`` by
    relation-phrase agreement) and cross-document co-occurrence (``INFERRED``).
    Deterministic — everything is sorted.
    """
    # --- Nodes: aggregate doc membership, mention counts, display label votes ---
    node_docs: dict[str, set[str]] = defaultdict(set)
    node_mentions: Counter[str] = Counter()
    label_votes: dict[str, Counter[str]] = defaultdict(Counter)

    # parse_extraction promotes every triple endpoint into ex.concepts, so this
    # one loop already covers triple-participating concepts — no second pass.
    for ex in extractions:
        for key in ex.concepts:
            node_docs[key].add(ex.doc_id)
            node_mentions[key] += 1
            label_votes[key][_display_for(key)] += 1

    nodes = [
        ConceptNode(
            id=key,
            label=label_votes[key].most_common(1)[0][0] if label_votes[key] else key,
            doc_ids=sorted(node_docs[key]),
            mentions=node_mentions[key],
        )
        for key in sorted(node_docs)
    ]
    valid_ids = {n.id for n in nodes}

    # --- EXTRACTED / AMBIGUOUS edges from stated triples ---
    pair_relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    pair_support: dict[tuple[str, str], set[EdgeSupport]] = defaultdict(set)
    for ex in extractions:
        for t in ex.triples:
            if t.subject not in valid_ids or t.object not in valid_ids:
                continue
            pair = _pair(t.subject, t.object)
            pair_relations[pair].add(t.relation)
            pair_docs[pair].add(ex.doc_id)
            # Feature 7d: per-(doc, polarity) stance, deduped within a doc by the set.
            pair_support[pair].add(EdgeSupport(ex.doc_id, t.polarity, ex.year))

    edges: list[ConceptEdge] = []
    stated: set[tuple[str, str]] = set()
    for pair in sorted(pair_relations):
        rels = sorted(pair_relations[pair])
        docs = sorted(pair_docs[pair])
        support = sorted(pair_support[pair], key=lambda s: (s.doc_id, s.polarity))
        integrity = "AMBIGUOUS" if len(rels) >= 2 else "EXTRACTED"
        edges.append(
            ConceptEdge(
                source=pair[0],
                target=pair[1],
                relations=rels,
                doc_ids=docs,
                weight=len(docs),
                integrity=integrity,
                support=support,
            )
        )
        stated.add(pair)

    # --- INFERRED co-occurrence edges (no stated triple, co-mentioned widely) ---
    cooccur: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ex in extractions:
        present = sorted(set(ex.concepts) & valid_ids)
        for i, a in enumerate(present):
            for b in present[i + 1 :]:
                cooccur[_pair(a, b)].add(ex.doc_id)

    for pair in sorted(cooccur):
        if pair in stated:
            continue
        docs = sorted(cooccur[pair])
        if len(docs) < min_cooccurrence:
            continue
        edges.append(
            ConceptEdge(
                source=pair[0],
                target=pair[1],
                relations=[],
                doc_ids=docs,
                weight=len(docs),
                integrity="INFERRED",
            )
        )

    return nodes, edges


def _display_for(key: str) -> str:
    """Readable display label for a canonical key (title-case short acronyms /
    leave multiword phrases as-is). Pure, deterministic."""
    if not key:
        return key
    if " " not in key and len(key) <= 4:
        return key.upper()  # acronym-ish: rag → RAG, bm25 → BM25
    return key


# ============================================================
# Pure core — graph analysis (communities, god nodes, gaps)
# ============================================================


def analyze_graph(
    nodes: list[ConceptNode],
    edges: list[ConceptEdge],
    *,
    god_nodes: int = CONCEPT_GRAPH_GOD_NODES,
    seed: int = CONCEPT_GRAPH_SEED,
) -> tuple[list[Community], list[str], GraphGaps]:
    """Louvain communities + god-node ranking + graph-gap signals over the graph.

    Mutates ``nodes`` in place to fill ``degree`` / ``community`` / ``god_node``.
    Deterministic for a fixed ``seed`` (Louvain is randomized). Pure compute — no
    I/O — though it uses NetworkX.
    """
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

    graph = nx.Graph()
    for n in nodes:
        graph.add_node(n.id)
    for e in edges:
        graph.add_edge(e.source, e.target, weight=e.weight)

    by_id = {n.id: n for n in nodes}

    # Degree.
    for node_id, deg in graph.degree():
        if node_id in by_id:
            by_id[node_id].degree = int(deg)

    # Communities (Louvain, weighted, seeded → reproducible).
    raw_communities: list[set[str]] = (
        louvain_communities(graph, weight="weight", seed=seed) if graph.number_of_nodes() else []
    )
    # Deterministic ordering: by size desc, then first member.
    ordered = sorted(
        (sorted(c) for c in raw_communities), key=lambda c: (-len(c), c[0] if c else "")
    )
    communities: list[Community] = []
    for cid, members in enumerate(ordered):
        for node_id in members:
            if node_id in by_id:
                by_id[node_id].community = cid
        # Label = the community's highest-degree node.
        label_node = max(members, key=lambda m: (by_id[m].degree if m in by_id else 0, m))
        communities.append(
            Community(
                id=cid,
                label=by_id[label_node].label if label_node in by_id else label_node,
                node_ids=members,
                size=len(members),
            )
        )

    # God nodes: highest-degree hubs (degree >= 1), deterministic tie-break by id.
    ranked = sorted((n for n in nodes if n.degree >= 1), key=lambda n: (-n.degree, n.id))
    god = [n.id for n in ranked[:god_nodes]]
    god_set = set(god)
    for n in nodes:
        n.god_node = n.id in god_set

    # Gaps: isolated nodes (degree 0) + thin bridges (cut edges), per component.
    isolated = sorted(n.id for n in nodes if n.degree == 0)
    bridges: list[tuple[str, str]] = []
    for comp in nx.connected_components(graph):
        sub = graph.subgraph(comp)
        if sub.number_of_edges() == 0:
            continue
        bridges.extend(_pair(u, v) for u, v in nx.bridges(sub))
    bridges.sort()

    return communities, god, GraphGaps(isolated_nodes=isolated, thin_bridges=bridges)


def assemble_graph(
    extractions: list[DocExtraction],
    *,
    min_cooccurrence: int = CONCEPT_GRAPH_MIN_COOCCURRENCE,
    god_nodes: int = CONCEPT_GRAPH_GOD_NODES,
    seed: int = CONCEPT_GRAPH_SEED,
    meta: dict[str, Any] | None = None,
) -> ConceptGraph:
    """The full pure pipeline: extractions → merged, analysed ``ConceptGraph``."""
    nodes, edges = build_nodes_edges(extractions, min_cooccurrence=min_cooccurrence)
    communities, god, gaps = analyze_graph(nodes, edges, god_nodes=god_nodes, seed=seed)
    return ConceptGraph(
        nodes=nodes,
        edges=edges,
        communities=communities,
        god_nodes=god,
        gaps=gaps,
        meta=meta or {},
    )


def doc_clusters_from_graph(
    graph: ConceptGraph, extractions: list[DocExtraction]
) -> list[list[str]]:
    """Group documents by the concept-community they most belong to.

    The Feature 6 bridge: a document's concepts vote for a community; documents
    that share a dominant community form a topic. This is the threshold-free
    replacement for Feature 6's absolute-cosine ``cluster_documents`` (wiki.py) —
    exposed here, wired into the wiki in a follow-up PR. Deterministic.
    """
    community_of = {n.id: n.community for n in graph.nodes}
    by_doc_community: dict[str, list[str]] = defaultdict(list)
    for ex in extractions:
        votes = Counter(community_of[c] for c in ex.concepts if community_of.get(c, -1) >= 0)
        if not votes:
            by_doc_community[f"solo:{ex.doc_id}"].append(ex.doc_id)
            continue
        dominant = min(votes.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        by_doc_community[f"c{dominant}"].append(ex.doc_id)
    clusters = [sorted(docs) for docs in by_doc_community.values()]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    return clusters


# ============================================================
# Pure core — claim-corroboration weights (Feature 7d)
# ============================================================


def compute_node_weights(graph: ConceptGraph) -> dict[str, NodeWeight]:
    """Structural epistemic weight per concept node (Feature 7d). Pure, deterministic.

    Aggregates the ``EdgeSupport`` records on every *stated* edge incident to a node
    (INFERRED edges carry no support and don't count — co-occurrence is not a claim),
    counting **distinct documents** that support vs dispute the node's claims:

    * ``coverage``  — ``contested`` if any disputing doc; else ``unique`` if <=1
      supporting doc (the only source on its topic — held NEUTRAL, never penalized:
      Decision 4); else ``corroborated``.
    * ``direction`` — ``stable`` if undisputed; ``superseded_trend`` if the disputing
      docs are newer than the supporting ones (currency from polarity-over-time, not
      age: Decision 1); else ``contested``.

    Every node gets a weight (isolated / claim-less nodes → ``stable``/``unique`` with
    zero counts — neutral, never down-weighted)."""
    incident: dict[str, list[EdgeSupport]] = defaultdict(list)
    for e in graph.edges:
        for endpoint in (e.source, e.target):
            incident[endpoint].extend(e.support)

    weights: dict[str, NodeWeight] = {}
    for n in graph.nodes:
        sup = incident.get(n.id, [])
        supporting_docs = {s.doc_id for s in sup if s.polarity in SUPPORTING_POLARITIES}
        opposing_docs = {s.doc_id for s in sup if s.polarity in OPPOSING_POLARITIES}
        ns, nc = len(supporting_docs), len(opposing_docs)
        denom = ns + nc
        agreement = round(ns / denom, 4) if denom else 1.0

        if nc == 0:
            direction = "stable"
        else:
            opp_years = [s.year for s in sup if s.polarity in OPPOSING_POLARITIES and s.year]
            sup_years = [s.year for s in sup if s.polarity in SUPPORTING_POLARITIES and s.year]
            if opp_years and sup_years and max(opp_years) > max(sup_years):
                direction = "superseded_trend"
            else:
                direction = "contested"

        if nc >= 1:
            coverage = "contested"
        elif ns <= 1:
            coverage = "unique"
        else:
            coverage = "corroborated"

        weights[n.id] = NodeWeight(
            node_id=n.id,
            n_supporting_sources=ns,
            n_contradicting_sources=nc,
            agreement_ratio=agreement,
            direction=direction,
            coverage=coverage,
        )
    return weights


# ============================================================
# Pure core — JSON serialisation
# ============================================================


def graph_to_dict(graph: ConceptGraph) -> dict[str, Any]:
    """Serialize a ``ConceptGraph`` to the ``graph.json`` payload (deterministic)."""
    return {
        "meta": {**graph.meta, "integrity_summary": graph.integrity_summary},
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "doc_ids": n.doc_ids,
                "mentions": n.mentions,
                "degree": n.degree,
                "community": n.community,
                "god_node": n.god_node,
            }
            for n in graph.nodes
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "relations": e.relations,
                "doc_ids": e.doc_ids,
                "weight": e.weight,
                "integrity": e.integrity,
                "support": [
                    {"doc_id": s.doc_id, "polarity": s.polarity, "year": s.year} for s in e.support
                ],
            }
            for e in graph.edges
        ],
        "communities": [
            {
                "id": c.id,
                "key": community_id_for(c.node_ids),
                "label": c.label,
                "node_ids": c.node_ids,
                "size": c.size,
            }
            for c in graph.communities
        ],
        "god_nodes": graph.god_nodes,
        "gaps": {
            "isolated_nodes": graph.gaps.isolated_nodes,
            "thin_bridges": [list(b) for b in graph.gaps.thin_bridges],
        },
    }


def graph_from_dict(data: dict[str, Any]) -> ConceptGraph:
    """Load a ``ConceptGraph`` from a ``graph.json`` payload (inverse of
    ``graph_to_dict``). Tolerant of partials / missing keys.

    The persisted ``communities[].key`` (a ``community_id_for`` membership hash) is
    a render-only field with no ``Community`` attribute, so it is ignored on load;
    ``meta.integrity_summary`` is a derived view and is dropped (the property
    recomputes it). Nodes/edges/communities/god_nodes/gaps round-trip exactly."""
    nodes = [
        ConceptNode(
            id=str(n.get("id", "")),
            label=str(n.get("label", n.get("id", ""))),
            doc_ids=[str(x) for x in n.get("doc_ids") or []],
            mentions=int(n.get("mentions", 0)),
            degree=int(n.get("degree", 0)),
            community=int(n.get("community", -1)),
            god_node=bool(n.get("god_node", False)),
        )
        for n in data.get("nodes") or []
    ]
    edges = [
        ConceptEdge(
            source=str(e.get("source", "")),
            target=str(e.get("target", "")),
            relations=[str(r) for r in e.get("relations") or []],
            doc_ids=[str(x) for x in e.get("doc_ids") or []],
            weight=int(e.get("weight", 0)),
            integrity=str(e.get("integrity", "")),
            support=[
                EdgeSupport(
                    doc_id=str(s.get("doc_id", "")),
                    polarity=snap_polarity(s.get("polarity", "")),
                    year=(int(s["year"]) if s.get("year") is not None else None),
                )
                for s in e.get("support") or []
                if isinstance(s, dict)
            ],
        )
        for e in data.get("edges") or []
    ]
    communities = [
        Community(
            id=int(c.get("id", -1)),
            label=str(c.get("label", "")),
            node_ids=[str(x) for x in c.get("node_ids") or []],
            size=int(c.get("size", 0)),
        )
        for c in data.get("communities") or []
    ]
    gaps_d = data.get("gaps") or {}
    gaps = GraphGaps(
        isolated_nodes=[str(x) for x in gaps_d.get("isolated_nodes") or []],
        thin_bridges=[
            (str(b[0]), str(b[1])) for b in gaps_d.get("thin_bridges") or [] if len(b) == 2
        ],
    )
    meta = {k: v for k, v in (data.get("meta") or {}).items() if k != "integrity_summary"}
    return ConceptGraph(
        nodes=nodes,
        edges=edges,
        communities=communities,
        god_nodes=[str(x) for x in data.get("god_nodes") or []],
        gaps=gaps,
        meta=meta,
    )


def extraction_to_dict(ex: DocExtraction) -> dict[str, Any]:
    """Serialize a ``DocExtraction`` to the per-doc extraction-cache JSON payload."""
    return {
        "doc_id": ex.doc_id,
        "doc_hash": ex.doc_hash,
        "filename": ex.filename,
        "year": ex.year,
        "concepts": ex.concepts,
        "triples": [
            {
                "subject": t.subject,
                "relation": t.relation,
                "object": t.object,
                "polarity": t.polarity,
            }
            for t in ex.triples
        ],
    }


def extraction_from_dict(data: dict[str, Any]) -> DocExtraction:
    """Load a ``DocExtraction`` from a cached JSON payload (tolerant of partials)."""
    raw_year = data.get("year")
    return DocExtraction(
        doc_id=str(data.get("doc_id", "")),
        doc_hash=str(data.get("doc_hash", "")),
        filename=str(data.get("filename", "")),
        concepts=[str(c) for c in data.get("concepts") or []],
        triples=[
            Triple(
                str(t.get("subject", "")),
                str(t.get("relation", "")),
                str(t.get("object", "")),
                polarity=snap_polarity(t.get("polarity", "")),
            )
            for t in (data.get("triples") or [])
            if isinstance(t, dict)
        ],
        year=int(raw_year) if raw_year is not None else None,
    )


# ============================================================
# Impure layer — DB / Chroma / LLM
# ============================================================


@dataclass
class GraphDoc:
    """A document to extract from. ``year`` (Feature 7d) feeds relative polarity order."""

    doc_id: str
    doc_hash: str
    filename: str
    year: int | None = None


def load_documents() -> list[GraphDoc]:
    """Load non-archived documents from SQLite (id / hash / filename / year)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        return [
            GraphDoc(doc_id=str(d.id), doc_hash=d.doc_hash, filename=d.filename, year=d.year)
            for d in session.execute(
                select(Document).where(Document.is_archived.is_(False))
            ).scalars()
        ]


def sample_doc_text(
    doc_id: str,
    *,
    per_doc: int = CONCEPT_GRAPH_CHUNK_SAMPLE,
    max_chars: int = CONCEPT_GRAPH_CHUNK_CHARS,
) -> list[str]:
    """Sample up to ``per_doc`` chunk excerpts for one document from the baseline store.

    Wider than ``wiki.sample_chunks`` (more chunks, longer excerpts) — extraction
    needs more material than a topic summary. Returns ``[]`` if Chroma is absent.
    """
    from doc_assistant.config import CHROMA_PATH
    from doc_assistant.embeddings import get_collection_name

    try:
        import chromadb
    except ImportError:  # pragma: no cover - dep present in dev env
        return []

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        coll = client.get_collection(get_collection_name())
    except Exception:
        log.warning("no_baseline_collection", hint="run ingest first; concept graph will be empty")
        return []

    data = coll.get(where={"document_id": doc_id}, include=["documents"], limit=per_doc)
    return [str(t)[:max_chars] for t in (data.get("documents") or []) if t]


_EXTRACTION_PROMPT = """You are building a concept graph from a research paper. From the \
excerpts below, extract the salient TECHNICAL concepts (methods, models, datasets, \
metrics, tasks) and the relationships stated between them. Use ONLY what the text \
states — do not invent relationships.

For each relation also give its POLARITY — how this paper positions the two concepts:
- "supports": affirms/uses/relates them positively (the default if unsure)
- "refines": extends or improves one with the other
- "contradicts": disputes or conflicts with the relationship
- "supersedes": presents one as replacing/obsoleting the other

EXCERPTS:
{material}

Return JSON only, no prose, no markdown fence:
{{"concepts": ["<5-15 concise concept names>"], \
"relations": [{{"subject": "<concept>", "relation": "<short verb phrase>", \
"object": "<concept>", "polarity": "<supports|refines|contradicts|supersedes>"}}]}}"""


def _format_material(filename: str, excerpts: list[str]) -> str:
    head = f"PAPER: {filename}\n"
    return head + "\n".join(f"- {ex}" for ex in excerpts)


def extract_doc(
    doc: GraphDoc,
    excerpts: list[str],
    client: Any,
    *,
    max_tokens: int = CONCEPT_GRAPH_MAX_TOKENS,
) -> DocExtraction:
    """Run one document's concept extraction through the provider protocol.

    ``client`` is an ``llm.LLMClient`` (Ollama by default — local + free). Raises
    on transport failure so the caller's per-doc isolation records it; a *parse*
    failure degrades to an empty extraction (see ``parse_extraction``).
    """
    if not excerpts:
        log.warning("no_chunk_excerpts", filename=doc.filename, hint="extraction will be sparse")
    prompt = _EXTRACTION_PROMPT.format(material=_format_material(doc.filename, excerpts))
    raw = client.complete(
        [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=max_tokens
    )
    return parse_extraction(
        raw, doc_id=doc.doc_id, doc_hash=doc.doc_hash, filename=doc.filename, year=doc.year
    )


# ============================================================
# Impure layer — sidecar I/O + orchestration
# ============================================================


@dataclass
class ConceptGraphResult:
    """What a ``build_concept_graph`` run produced."""

    graph: ConceptGraph
    extracted: int  # docs freshly extracted this run (LLM calls)
    cached: int  # docs served from the extraction cache
    skipped: int  # docs with no extraction available (dry-run, no cache)
    errors: int  # docs whose extraction raised
    applied: bool


def _extractions_dir(root: Path) -> Path:
    return root / EXTRACTIONS_DIRNAME


def _cache_path(root: Path, doc_hash: str) -> Path:
    return _extractions_dir(root) / f"{doc_hash}.json"


def load_cached_extraction(root: Path, doc_hash: str) -> DocExtraction | None:
    path = _cache_path(root, doc_hash)
    if not path.exists():
        return None
    try:
        return extraction_from_dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return None


def _write_cached_extraction(root: Path, ex: DocExtraction) -> None:
    _extractions_dir(root).mkdir(parents=True, exist_ok=True)
    _cache_path(root, ex.doc_hash).write_text(
        json.dumps(extraction_to_dict(ex), indent=2, sort_keys=True), encoding="utf-8"
    )


def build_concept_graph(
    *,
    apply: bool,
    force: bool = False,
    client: Any | None = None,
    doc_filter: str | None = None,
    min_cooccurrence: int = CONCEPT_GRAPH_MIN_COOCCURRENCE,
    god_nodes: int = CONCEPT_GRAPH_GOD_NODES,
    seed: int = CONCEPT_GRAPH_SEED,
    provider: str = "ollama",
    model: str = "",
    graph_dir: Path | None = None,
) -> ConceptGraphResult:
    """Extract concepts per document, merge into the corpus graph, write the sidecar.

    Idempotent: a document whose extraction is cached (keyed by ``doc_hash``, so a
    content change re-extracts automatically) is reused unless ``force``. Dry-run
    (``apply=False`` or no ``client``) assembles the graph from whatever is already
    cached and writes nothing. ``apply`` extracts missing documents via the model,
    caches each, and writes ``graph.json``. Never touches the chunk store.
    """
    root = graph_dir or CONCEPT_GRAPH_DIR
    docs = load_documents()
    if doc_filter:
        docs = [
            d for d in docs if d.doc_hash.startswith(doc_filter) or d.doc_id.startswith(doc_filter)
        ]

    extractions: list[DocExtraction] = []
    extracted = cached = skipped = errors = 0

    for doc in docs:
        existing = None if force else load_cached_extraction(root, doc.doc_hash)
        if existing is not None:
            extractions.append(existing)
            cached += 1
            continue
        if not (apply and client is not None):
            skipped += 1
            continue
        try:
            excerpts = sample_doc_text(doc.doc_id)
            ex = extract_doc(doc, excerpts, client)
            extractions.append(ex)
            if apply:
                _write_cached_extraction(root, ex)
            extracted += 1
        except Exception as e:
            log.warning("concept_extraction_failed", filename=doc.filename, error=str(e))
            errors += 1

    meta = {
        "provider": provider,
        "model": model,
        "n_documents": len(docs),
        # documents contributing to the graph = freshly extracted + served from cache
        # (NOT the count of LLM calls — see ConceptGraphResult.extracted for that).
        "n_docs_in_graph": len(extractions),
        "min_cooccurrence": min_cooccurrence,
        "seed": seed,
    }
    graph = assemble_graph(
        extractions, min_cooccurrence=min_cooccurrence, god_nodes=god_nodes, seed=seed, meta=meta
    )

    if apply:
        root.mkdir(parents=True, exist_ok=True)
        (root / GRAPH_NAME).write_text(
            json.dumps(graph_to_dict(graph), indent=2, sort_keys=False), encoding="utf-8"
        )

    return ConceptGraphResult(
        graph=graph,
        extracted=extracted,
        cached=cached,
        skipped=skipped,
        errors=errors,
        applied=apply,
    )
