"""Knowledge-currency / claim-corroboration projection (Phase 7 / Feature 7d).

The epistemic layer on top of the concept skeleton (Node A/B, KI-7 retirement).
``concept_skeleton.node_weights_for_epistemics`` says *how well corroborated* each
concept's claims are (supporting vs disputing sources). This module **projects** those
node-level weights down onto the retrieval substrate (chunks), so an answer can surface
"this evidence sits on a contested claim" at read time. (``superseded_trend`` stays a
valid marker in the vocabulary below, but the skeleton carries no publication years, so
today's Node A/B weights never produce it — see ``node_weights_for_epistemics``.)

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

Surfacing is markers + a reviewer failure tag. The live marker join (E1.1 / KI-8) keys a
retrieved chunk against the sidecar by its ``chunk_key`` — a direct lookup for both the
baseline (flat) and PC-parent (default) segmentations, since ``build_epistemics`` projects
onto both. This replaced PR-M1's coarse PC-parent text-containment, which lost ~40% of
markers at parent boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import structlog

from doc_assistant.knowledge.concept_skeleton import (
    ConceptSkeleton,
    NodeWeight,
    compile_boundary_pattern,
    node_weights_for_epistemics,
)

log = structlog.get_logger(__name__)

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

    ``chunk_key`` is the stable composite the live marker join keys on — ``{doc}:{chunk_index}``
    for a baseline chunk, ``{doc}:p{parent_index}`` for a PC parent (E1.1 / KI-8). Chroma's own
    ids are auto-generated UUIDs and unstable across re-ingest, so the composite is authoritative.
    ``coverage_summary`` counts the chunk's claim-nodes by coverage class; ``markers`` is the
    derived evidence-layer surface (empty for a clean chunk → quiet-on-clean).
    """

    document_id: str
    chunk_index: int
    chunk_key: str
    n_claims: int
    n_contested: int
    n_superseded_trend: int
    coverage_summary: dict[str, int] = field(default_factory=dict)
    markers: list[str] = field(default_factory=list)


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


def concepts_in_text(text: str, labels_by_id: dict[str, str]) -> list[str]:
    """Which concept node ids are attributed to ``text`` (structural word-boundary match, pure).

    Matches on each concept's **label** (not its node id — the curated skeleton's ids are opaque
    ``Concept.id`` UUIDs that never occur in document text; KI-15), casefolded, via the same
    alnum-boundary pattern ``concept_skeleton``'s Node-A presence matcher uses (R2,
    :func:`concept_skeleton.compile_boundary_pattern` — not ``\\b``, which mishandles non-word
    edge chars like "gpt-4"). Labels shorter than ``_MIN_CONCEPT_LEN`` are skipped (too short to
    attribute reliably — "ir" would match far too much). Deterministic order (``labels_by_id``
    iteration order, de-duplicated)."""
    low = text.casefold()
    present: list[str] = []
    seen: set[str] = set()
    for nid, label in labels_by_id.items():
        if nid in seen:
            continue
        form = label.strip().casefold()
        if len(form) < _MIN_CONCEPT_LEN:
            continue
        if compile_boundary_pattern(form).search(low):
            seen.add(nid)
            present.append(nid)
    return present


def project_chunk(
    chunk_key: str,
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
        chunk_key=chunk_key,
        n_claims=len(seen),
        n_contested=n_contested,
        n_superseded_trend=n_superseded,
        coverage_summary=coverage,
        markers=derive_markers(n_contested, n_superseded),
    )


def project_chunk_weights(
    skeleton: ConceptSkeleton,
    weights: dict[str, NodeWeight],
    doc_chunks: list[tuple[str, str, int, str]],
) -> list[ChunkEpistemics]:
    """Project node weights onto chunks (pure). ``doc_chunks`` = (chunk_key, document_id,
    chunk_index, text) — the ``chunk_key`` carries the segmentation (``{doc}:{idx}`` baseline
    or ``{doc}:p{idx}`` parent, E1.1), so the same projection serves both. Only chunks that
    contain at least one weighted concept get a row — a chunk with no claims carries no
    epistemic signal and is omitted."""
    labels_by_id = {n.id: n.label for n in skeleton.nodes}
    rows: list[ChunkEpistemics] = []
    for chunk_key, document_id, chunk_index, text in doc_chunks:
        present = concepts_in_text(text, labels_by_id)
        if not present:
            continue
        rows.append(project_chunk(chunk_key, document_id, chunk_index, present, weights))
    rows.sort(key=lambda r: r.chunk_key)
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


def graph_version(skeleton: ConceptSkeleton) -> str:
    """The skeleton's own structural fingerprint (for sidecar staleness).

    Reuses ``concept_skeleton``'s canonical ``graph_version`` (computed over the full
    node/edge/stance signature, not re-derived here) so there is one definition of
    "did the skeleton change", not two."""
    return str(skeleton.meta.get("graph_version", ""))


# ============================================================
# Impure layer — Chroma read + sidecar I/O + orchestration
# ============================================================


def load_doc_chunks() -> list[tuple[str, str, int, str]]:
    """Load (chunk_key, document_id, chunk_index, text) for every baseline chunk from Chroma.

    Keyed off the baseline collection (which carries ``chunk_index`` in metadata); the
    ``chunk_key`` is the composite ``{doc}:{chunk_index}`` the flat-mode marker join uses.
    Chunks lacking a ``document_id``/``chunk_index`` (e.g. figure chunks) are skipped. Returns
    ``[]`` if Chroma is absent. See :func:`load_pc_parent_chunks` for the PC-parent segment."""
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
        log.warning("no_baseline_collection", hint="run ingest first; epistemics will be empty")
        return []

    data = coll.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    out: list[tuple[str, str, int, str]] = []
    for text, meta in zip(documents, metadatas, strict=False):
        if not text or not isinstance(meta, dict):
            continue
        document_id = meta.get("document_id")
        chunk_index = meta.get("chunk_index")
        if document_id is None or chunk_index is None:
            continue
        out.append(
            (f"{document_id}:{int(chunk_index)}", str(document_id), int(chunk_index), str(text))
        )
    return out


def load_pc_parent_chunks() -> list[tuple[str, str, int, str]]:
    """Load (chunk_key, document_id, parent_index, parent_text) for every PC parent (E1.1 / KI-8).

    The parent-child segmentation the default retrieval mode actually returns. Reads the PC store
    (``PC_CHROMA_PATH``), de-duplicates child rows to one entry per parent via ``parent_index``
    (the parent text is denormalised onto every child), and builds the ADR-4 composite key
    ``{doc}:p{parent_index}`` — so the live marker join for a retrieved parent is a direct key
    lookup, not the coarse text-containment that lost ~40% of markers at parent boundaries. Mirrors
    ``concept_skeleton.load_presence_inputs``; returns ``[]`` if the PC collection is absent."""
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
        log.warning(
            "no_pc_collection", hint="run ingest first; PC-parent epistemics will be empty"
        )
        return []

    data = coll.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []
    seen: set[tuple[str, int]] = set()
    out: list[tuple[str, str, int, str]] = []
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
        out.append(
            (
                f"{document_id}:p{int(parent_index)}",
                str(document_id),
                int(parent_index),
                str(parent_text),
            )
        )
    return out


def load_epistemics_index() -> dict[str, list[str]]:
    """Read the ``chunk_epistemics`` sidecar into a ``{chunk_key: markers}`` index.

    The read-side counterpart of ``build_epistemics`` and the seam the live marker join consults
    (E1.1). Keys on the row's stored ``chunk_key`` — so both segmentations resolve: a baseline
    row (``{doc}:{idx}``, flat mode) and a PC-parent row (``{doc}:p{idx}``, default mode). Falls
    back to ``{document_id}:{chunk_index}`` for a row written before the ``chunk_key`` column
    existed (a migrated-but-not-recomputed DB still joins flat rows; parent rows arrive on the
    next ``compute_epistemics --apply``)."""
    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError

    from doc_assistant.db.models import ChunkEpistemics as ChunkEpistemicsRow
    from doc_assistant.db.session import session_scope

    index: dict[str, list[str]] = {}
    try:
        with session_scope() as session:
            for row in session.execute(select(ChunkEpistemicsRow)).scalars():
                markers = derive_markers(row.n_contested, row.n_superseded_trend)
                if markers:
                    key = row.chunk_key or f"{row.document_id}:{row.chunk_index}"
                    index[key] = markers
    except OperationalError:
        # The chunk_epistemics sidecar table doesn't exist on this DB (the 7d engine never
        # ran) — treat as "no markers", consistent with the quiet-on-absent design.
        return {}
    return index


@dataclass(frozen=True)
class ChunkEval:
    """Per-source epistemic assessment for the always-on D3 strip (ADR-027) — the read-model
    derived from one ``chunk_epistemics`` row. ``coverage`` is the single most-cautionary class
    present; ``superseded`` flags a superseded-trend claim; ``n_claims`` is how many weighted
    concepts the chunk carries. A retrieved source with **no** row is "not assessed" (absent)."""

    coverage: str | None  # "contested" | "corroborated" | "unique" | None
    superseded: bool
    n_claims: int


def _derive_coverage(n_contested: int, coverage_summary: dict[str, int]) -> str | None:
    """The single most-cautionary coverage class of a chunk (contested > corroborated > unique)."""
    if n_contested > 0:
        return "contested"
    if coverage_summary.get("corroborated", 0) > 0:
        return "corroborated"
    if coverage_summary.get("unique", 0) > 0:
        return "unique"
    return None


def load_source_evaluations(chunk_keys: list[str]) -> tuple[dict[str, ChunkEval], str | None]:
    """Per-source evaluations for the D3 strip + the sidecar's ``graph_version`` (ADR-027 D3).

    Scoped, indexed read of ``chunk_epistemics`` for the retrieved sources' ``chunk_key``s (unlike
    the full-scan ``load_epistemics_index`` — KI-18 discipline). A key with no row is simply absent
    (the caller renders "not assessed"). A sidecar predating E1.1's ``chunk_key`` column (NULL key)
    needs a ``compute_epistemics --apply`` to appear here — same transition as the marker join.
    Returns ``({}, None)`` on a never-migrated DB (the 0-doc honest-degradation contract). $0."""
    import json

    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError

    from doc_assistant.db.models import ChunkEpistemics as ChunkEpistemicsRow
    from doc_assistant.db.session import session_scope

    if not chunk_keys:
        return {}, None
    out: dict[str, ChunkEval] = {}
    version: str | None = None
    try:
        with session_scope() as session:
            rows = session.execute(
                select(ChunkEpistemicsRow).where(ChunkEpistemicsRow.chunk_key.in_(chunk_keys))
            ).scalars()
            for row in rows:
                if row.chunk_key is None:  # excluded by the IN filter; guards the dict key type
                    continue
                summary = json.loads(row.coverage_summary or "{}")
                out[row.chunk_key] = ChunkEval(
                    coverage=_derive_coverage(row.n_contested, summary),
                    superseded=row.n_superseded_trend > 0,
                    n_claims=row.n_claims,
                )
                if version is None:
                    version = row.graph_version
    except OperationalError:
        return {}, None
    return out, version


def current_graph_version() -> str | None:
    """The current skeleton's build stamp (from ``concept_presence``) for the D3 freshness compare.

    A single-row read — ``None`` when no skeleton has been built or the table is absent. The D3
    strip flags itself **stale** when the epistemics sidecar's ``graph_version`` differs from this
    (the graph was rebuilt but ``compute_epistemics`` was not re-run)."""
    from sqlalchemy import select
    from sqlalchemy.exc import OperationalError

    from doc_assistant.db.models import ConceptPresenceRow
    from doc_assistant.db.session import session_scope

    try:
        with session_scope() as session:
            return session.execute(
                select(ConceptPresenceRow.graph_version).limit(1)
            ).scalar_one_or_none()
    except OperationalError:
        return None


def build_epistemics(*, apply: bool, skeleton_dir: Path | None = None) -> EpistemicsResult:
    """Compute per-chunk epistemic weights from the concept skeleton; write the sidecar.

    Read-only + free (no LLM): loads ``skeleton.json``, computes node weights, projects
    them onto **both** the baseline and the PC-parent chunk segmentations via structural
    attribution (E1.1 / KI-8 — so the live marker join is a direct key lookup for either
    retrieval mode). ``apply`` replaces the ``chunk_epistemics`` table (regenerable sidecar —
    dropped + rebuilt with the skeleton); a dry run computes + reports but writes nothing.
    Idempotent: same skeleton + same chunks → identical rows. Never touches the chunk store."""
    import json

    from doc_assistant.config import CONCEPT_SKELETON_DIR
    from doc_assistant.knowledge.concept_skeleton import SKELETON_NAME, skeleton_from_dict

    root = skeleton_dir or CONCEPT_SKELETON_DIR
    skeleton_path = root / SKELETON_NAME
    if not skeleton_path.exists():
        raise FileNotFoundError(
            f"No concept skeleton at {skeleton_path} — run `python -m scripts."
            "build_concept_skeleton --apply` first (Feature 7d projects over the skeleton)."
        )
    skeleton = skeleton_from_dict(json.loads(skeleton_path.read_text(encoding="utf-8")))
    weights = node_weights_for_epistemics(skeleton)
    version = graph_version(skeleton)

    # E1.1 (KI-8): project onto BOTH segmentations — baseline chunks (flat-mode join) and PC
    # parents (the default retrieval mode's join). Same weights, same structural attribution;
    # the chunk_key carried on each row distinguishes them, so the live join is a direct key
    # lookup for either mode rather than the text-containment that lost ~40% of parent markers.
    doc_chunks = load_doc_chunks() + load_pc_parent_chunks()
    rows = project_chunk_weights(skeleton, weights, doc_chunks)

    applied = apply
    if apply:
        from sqlalchemy.exc import OperationalError

        try:
            _write_rows(rows, version)
        except OperationalError:
            # E0.4 / WE-9: a never-migrated DB (the 0-doc honest-degradation contract,
            # `.claude/CONTEXT.md`) has no `chunk_epistemics` table, so `_write_rows`' delete-all
            # trips `OperationalError`. Degrade to an honest empty result + hint rather than
            # crashing this build path. A migrated DB never reaches here (the table exists), so
            # `apply` on a real corpus still clears + rewrites the sidecar as before.
            log.warning("epistemics_write_skipped_no_schema", hint="run init_db / ingest first")
            applied = False

    n_contested_nodes = sum(1 for w in weights.values() if w.coverage == "contested")
    n_superseded_nodes = sum(1 for w in weights.values() if w.direction == "superseded_trend")
    return EpistemicsResult(
        rows=rows,
        graph_version=version,
        n_nodes=len(weights),
        n_contested_nodes=n_contested_nodes,
        n_superseded_nodes=n_superseded_nodes,
        n_chunks_marked=sum(1 for r in rows if r.markers),
        applied=applied,
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
                chunk_key=r.chunk_key,
                n_claims=r.n_claims,
                n_contested=r.n_contested,
                n_superseded_trend=r.n_superseded_trend,
                coverage_summary=json.dumps(r.coverage_summary, sort_keys=True),
                graph_version=version,
            )
            for r in rows
        )
