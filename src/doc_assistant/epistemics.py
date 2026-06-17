"""Knowledge-currency / claim-corroboration projection (Phase 7 / Feature 7d).

The epistemic layer on top of Feature 7's concept graph. Feature 7 says *what*
relates to what; ``concept_graph.compute_node_weights`` says *how well corroborated*
each concept's claims are (supporting vs disputing sources, and whether disputes are
newer — ``superseded_trend``). This module **projects** those node-level weights down
onto the retrieval substrate (chunks), so an answer can surface "this evidence sits on
a contested / superseded-trending claim" at read time.

Why a projection, not per-chunk scoring: a ~2000-char chunk is the wrong unit for
epistemics (one parent can hold a stale result and three solid definitions). The
weight lives on the **claim** (a concept-graph node/edge); a chunk inherits the weights
of the concepts that actually appear in it. Attribution is **structural** — a concept
is "in" a chunk if its canonical label occurs in the chunk text (word-boundary match) —
never an LLM judgement, consistent with the project's no-self-reported-confidence rule.

Architecture (Enrichment-Layer Pattern): read-only over the concept-graph sidecar plus
one regenerable projection table (``chunk_epistemics``); never mutates the chunk store,
never calls an LLM (the only LLM cost in Feature 7 is the graph extraction itself). The
unique-source rule (a chunk whose only claim is the corpus's sole source on its topic)
is preserved end-to-end: such a chunk is **neutral**, never marked.

Surfacing v1 is markers + a reviewer failure tag. Wiring the markers into the live
answer's evidence layer (which needs a stable chunk key plumbed through retrieval) is a
documented follow-up; this module ships the deterministic engine + sidecar + the
marker-derivation join, all guard-tested.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from doc_assistant.concept_graph import ConceptGraph, NodeWeight, compute_node_weights

log = logging.getLogger(__name__)

#: Evidence-layer markers (extend the Chunk 2a marker set). Structural, not opinion.
MARKER_CONTESTED = "contested"
MARKER_SUPERSEDED = "superseded_trend"

#: Concepts shorter than this are skipped during structural text attribution — a
#: 1-2 char token (e.g. "ir") word-matches far too much to be a reliable back-pointer.
_MIN_CONCEPT_LEN = 3


# ============================================================
# Data classes
# ============================================================


@dataclass
class ChunkEpistemics:
    """Projected per-chunk epistemic summary (the ``chunk_epistemics`` row payload).

    Keyed by the stable composite ``{document_id}:{chunk_index}`` (Chroma's own ids are
    auto-generated UUIDs and unstable across re-ingest). ``coverage_summary`` counts the
    chunk's claim-nodes by coverage class; ``markers`` is the derived evidence-layer
    surface (empty for a clean chunk → quiet-on-clean).
    """

    document_id: str
    chunk_index: int
    n_claims: int
    n_contested: int
    n_superseded_trend: int
    coverage_summary: dict[str, int] = field(default_factory=dict)
    markers: list[str] = field(default_factory=list)

    @property
    def chunk_key(self) -> str:
        return f"{self.document_id}:{self.chunk_index}"


@dataclass
class EpistemicsResult:
    """What a ``build_epistemics`` run produced."""

    rows: list[ChunkEpistemics]
    graph_version: str
    n_nodes: int
    n_contested_nodes: int
    n_superseded_nodes: int
    n_chunks_marked: int
    applied: bool


# ============================================================
# Pure core — attribution, projection, markers
# ============================================================


def derive_markers(n_contested: int, n_superseded_trend: int) -> list[str]:
    """Evidence-layer markers for a chunk from its claim counts (pure, deterministic).

    A chunk is marked ``contested`` if any of its claims sits on a contested node, and
    ``superseded_trend`` if any sits on a superseded-trending node. A clean chunk (only
    stable / corroborated / unique claims) gets **no** marker — quiet-on-clean, and the
    unique-source rule means a sole-source chunk is never marked."""
    markers: list[str] = []
    if n_contested > 0:
        markers.append(MARKER_CONTESTED)
    if n_superseded_trend > 0:
        markers.append(MARKER_SUPERSEDED)
    return markers


def concepts_in_text(text: str, node_ids: list[str]) -> list[str]:
    """Which concept node ids occur in ``text`` (structural word-boundary match, pure).

    Node ids are canonical (lowercase) keys; matching is case-insensitive on a
    word boundary so "bm25" matches but "ir" (too short) and substrings like the "rag"
    inside "storage" do not. Deterministic order (input order, de-duplicated)."""
    low = text.lower()
    present: list[str] = []
    seen: set[str] = set()
    for nid in node_ids:
        if len(nid) < _MIN_CONCEPT_LEN or nid in seen:
            continue
        if re.search(rf"\b{re.escape(nid)}\b", low):
            seen.add(nid)
            present.append(nid)
    return present


def project_chunk(
    document_id: str,
    chunk_index: int,
    present_node_ids: list[str],
    weights: dict[str, NodeWeight],
) -> ChunkEpistemics:
    """Aggregate the weights of the concepts present in one chunk into a row (pure)."""
    coverage: dict[str, int] = {"corroborated": 0, "unique": 0, "contested": 0}
    n_contested = n_superseded = 0
    seen: set[str] = set()
    for nid in present_node_ids:
        weight = weights.get(nid)
        if weight is None or nid in seen:
            continue
        seen.add(nid)
        coverage[weight.coverage] = coverage.get(weight.coverage, 0) + 1
        if weight.coverage == "contested":
            n_contested += 1
        if weight.direction == "superseded_trend":
            n_superseded += 1
    return ChunkEpistemics(
        document_id=document_id,
        chunk_index=chunk_index,
        n_claims=len(seen),
        n_contested=n_contested,
        n_superseded_trend=n_superseded,
        coverage_summary=coverage,
        markers=derive_markers(n_contested, n_superseded),
    )


def project_chunk_weights(
    graph: ConceptGraph,
    weights: dict[str, NodeWeight],
    doc_chunks: list[tuple[str, int, str]],
) -> list[ChunkEpistemics]:
    """Project node weights onto chunks (pure). ``doc_chunks`` = (document_id,
    chunk_index, text). Only chunks that contain at least one weighted concept get a
    row — a chunk with no claims carries no epistemic signal and is omitted."""
    node_ids = [n.id for n in graph.nodes]
    rows: list[ChunkEpistemics] = []
    for document_id, chunk_index, text in doc_chunks:
        present = concepts_in_text(text, node_ids)
        if not present:
            continue
        rows.append(project_chunk(document_id, chunk_index, present, weights))
    rows.sort(key=lambda r: (r.document_id, r.chunk_index))
    return rows


def markers_for_chunk_keys(
    chunk_keys: list[str], index: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Join retrieved chunk keys against an epistemics marker index (pure).

    ``index`` maps ``{document_id}:{chunk_index}`` → markers. Returns only the keys that
    carry a marker — clean / unknown chunks stay quiet. This is the read-side seam the
    live evidence layer will call once a stable chunk key is plumbed through retrieval
    (deferred; see the module docstring)."""
    out: dict[str, list[str]] = {}
    for key in chunk_keys:
        markers = index.get(key)
        if markers:
            out[key] = list(markers)
    return out


def graph_version(graph: ConceptGraph) -> str:
    """A short, stable fingerprint of the graph's structure (for sidecar staleness).

    Hash of the sorted node ids + edge count — changes when the graph re-extracts,
    stable across a pure re-projection of the same graph."""
    import hashlib

    payload = "|".join(sorted(n.id for n in graph.nodes)) + f"#{len(graph.edges)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# ============================================================
# Impure layer — Chroma read + sidecar I/O + orchestration
# ============================================================


def load_doc_chunks() -> list[tuple[str, int, str]]:
    """Load (document_id, chunk_index, text) for every baseline chunk from Chroma.

    Keyed off the baseline collection (which carries ``chunk_index`` in metadata); the
    parent-child store uses parent/child indices instead and is left for the live-
    surfacing follow-up. Chunks lacking a ``document_id``/``chunk_index`` (e.g. figure
    chunks) are skipped. Returns ``[]`` if Chroma is absent."""
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
        log.warning("No baseline collection — run ingest first; epistemics will be empty")
        return []

    data = coll.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    out: list[tuple[str, int, str]] = []
    for text, meta in zip(documents, metadatas, strict=False):
        if not text or not isinstance(meta, dict):
            continue
        document_id = meta.get("document_id")
        chunk_index = meta.get("chunk_index")
        if document_id is None or chunk_index is None:
            continue
        out.append((str(document_id), int(chunk_index), str(text)))
    return out


def load_epistemics_index() -> dict[str, list[str]]:
    """Read the ``chunk_epistemics`` sidecar into a ``{chunk_key: markers}`` index.

    The read-side counterpart of ``build_epistemics`` — what the live evidence layer
    will consult to mark retrieved chunks (deferred wiring)."""
    from sqlalchemy import select

    from doc_assistant.db.models import ChunkEpistemics as ChunkEpistemicsRow
    from doc_assistant.db.session import session_scope

    index: dict[str, list[str]] = {}
    with session_scope() as session:
        for row in session.execute(select(ChunkEpistemicsRow)).scalars():
            markers = derive_markers(row.n_contested, row.n_superseded_trend)
            if markers:
                index[f"{row.document_id}:{row.chunk_index}"] = markers
    return index


def build_epistemics(*, apply: bool, graph_dir: Path | None = None) -> EpistemicsResult:
    """Compute per-chunk epistemic weights from the concept graph; write the sidecar.

    Read-only + free (no LLM): loads ``graph.json``, computes node weights, projects
    them onto baseline chunks via structural attribution. ``apply`` replaces the
    ``chunk_epistemics`` table (regenerable sidecar — dropped + rebuilt with the graph);
    a dry run computes + reports but writes nothing. Idempotent: same graph + same
    chunks → identical rows. Never touches the chunk store."""
    import json

    from doc_assistant.concept_graph import GRAPH_NAME, graph_from_dict
    from doc_assistant.config import CONCEPT_GRAPH_DIR

    root = graph_dir or CONCEPT_GRAPH_DIR
    graph_path = root / GRAPH_NAME
    if not graph_path.exists():
        raise FileNotFoundError(
            f"No concept graph at {graph_path} — run `python -m scripts.build_concept_graph "
            "--apply` first (Feature 7d projects over Feature 7's graph)."
        )
    graph = graph_from_dict(json.loads(graph_path.read_text(encoding="utf-8")))
    weights = compute_node_weights(graph)
    version = graph_version(graph)

    doc_chunks = load_doc_chunks()
    rows = project_chunk_weights(graph, weights, doc_chunks)

    if apply:
        _write_rows(rows, version)

    n_contested_nodes = sum(1 for w in weights.values() if w.coverage == "contested")
    n_superseded_nodes = sum(1 for w in weights.values() if w.direction == "superseded_trend")
    return EpistemicsResult(
        rows=rows,
        graph_version=version,
        n_nodes=len(weights),
        n_contested_nodes=n_contested_nodes,
        n_superseded_nodes=n_superseded_nodes,
        n_chunks_marked=sum(1 for r in rows if r.markers),
        applied=apply,
    )


def _write_rows(rows: list[ChunkEpistemics], version: str) -> None:
    """Replace the ``chunk_epistemics`` sidecar with this run's rows (idempotent)."""
    import json

    from sqlalchemy import delete

    from doc_assistant.db.models import ChunkEpistemics as ChunkEpistemicsRow
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.execute(delete(ChunkEpistemicsRow))
        session.add_all(
            ChunkEpistemicsRow(
                document_id=r.document_id,
                chunk_index=r.chunk_index,
                n_claims=r.n_claims,
                n_contested=r.n_contested,
                n_superseded_trend=r.n_superseded_trend,
                coverage_summary=json.dumps(r.coverage_summary, sort_keys=True),
                graph_version=version,
            )
            for r in rows
        )
