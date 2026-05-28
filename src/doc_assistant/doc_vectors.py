"""Doc-level vector enrichment (Phase 4, close-out).

Reads chunk embeddings from the baseline Chroma store, mean-pools per
document to a single L2-normalised vector, then computes directed top-K
cosine-similarity edges above a score threshold.

The pipeline is intentionally split:

    Chroma -> {doc_id: [chunk_vec, ...]}
            -> mean_pool -> {doc_id: doc_vec}
            -> compute_similarity_edges -> [SimilarityEdge, ...]
            -> caller persists rows (scripts/compute_doc_vectors.py)

Design notes
------------
* Pure-numpy core. No DB writes. Keeps the module testable without
  Chroma or SQLite.
* Edges are stored directed. The cosine relation is symmetric, but the
  top-K trim makes the persisted edge set asymmetric and gives a stable
  "most similar to A" UX. Consumers wanting symmetric edges can union
  the two directions.
* `embedding_model` is tagged on every persisted edge so future
  Phase 5 / Feature 1 swaps don't collide with existing rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select

from doc_assistant.config import CHROMA_PATH
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.embeddings import get_active_model_name, get_collection_name

log = logging.getLogger(__name__)


# ============================================================
# Constants
# ============================================================

# Persisted tag for `DocSimilarity.embedding_model` rows. Defaults to the
# active model's registry key; PR 1 wrote the literal HF id ("bge-base-en-v1.5"),
# so existing rows under that tag stay queryable via the kwarg on
# `library.similar_docs(embedding_model=...)`.
EMBEDDING_MODEL_NAME = get_active_model_name()
DEFAULT_TOP_K = 10
DEFAULT_THRESHOLD = 0.5


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class SimilarityEdge:
    """A directed similarity edge between two documents."""

    source_doc_id: str
    target_doc_id: str
    score: float


# ============================================================
# Pure-numpy core
# ============================================================


def mean_pool(vectors: list[np.ndarray] | np.ndarray) -> np.ndarray:
    """Mean across vectors, then L2-normalise. Returns a 1-D float32 array.

    BGE chunk embeddings are pre-normalised, but the arithmetic mean is
    not. Re-normalising lets downstream cosine similarity be computed
    via dot product. A zero-norm mean (degenerate) is returned as-is.
    """
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("mean_pool: empty input")
    if arr.ndim != 2:
        raise ValueError(f"mean_pool: expected 2-D input, got shape {arr.shape}")
    mean = arr.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm == 0.0:
        return np.asarray(mean, dtype=np.float32)
    return np.asarray(mean / norm, dtype=np.float32)


def compute_doc_vectors(
    chunk_embeddings: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Mean-pool one vector per doc_id, skipping docs with no chunks."""
    out: dict[str, np.ndarray] = {}
    for doc_id, vecs in chunk_embeddings.items():
        if not vecs:
            continue
        out[doc_id] = mean_pool(vecs)
    return out


def compute_similarity_edges(
    doc_vectors: dict[str, np.ndarray],
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[SimilarityEdge]:
    """Directed top-K cosine-similarity edges per source above ``threshold``.

    With N docs this is O(N^2) in similarity and O(N^2 log N) in sort;
    fine for personal-library scale (10s-100s of docs). For larger
    libraries this is the obvious place to swap in an ANN index, but
    that's a Phase 7 problem.
    """
    if len(doc_vectors) < 2:
        return []

    doc_ids = list(doc_vectors.keys())
    matrix = np.stack([doc_vectors[d] for d in doc_ids]).astype(np.float32)
    sims = matrix @ matrix.T
    np.fill_diagonal(sims, -1.0)

    edges: list[SimilarityEdge] = []
    for i, source in enumerate(doc_ids):
        row = sims[i]
        order = np.argsort(-row)
        for j in order[:top_k]:
            score = float(row[j])
            if score < threshold:
                break
            edges.append(
                SimilarityEdge(
                    source_doc_id=source,
                    target_doc_id=doc_ids[j],
                    score=score,
                )
            )
    return edges


# ============================================================
# Chroma adapter — kept thin so the core stays testable
# ============================================================


def _hash_to_doc_id_map() -> dict[str, str]:
    """Return a {doc_hash: Document.id} mapping for all non-archived docs."""
    with session_scope() as session:
        rows = session.execute(
            select(Document.doc_hash, Document.id).where(Document.is_archived.is_(False))
        ).all()
        return {h: did for h, did in rows}


def load_chunk_embeddings_by_document() -> dict[str, list[np.ndarray]]:
    """Group every chunk embedding in the baseline Chroma store by Document.id.

    Prefers the explicit ``document_id`` chunk metadata when present;
    falls back to resolving ``doc_hash`` through the SQLite store for
    chunks that pre-date the metadata field. Chunks that resolve to
    neither are dropped with a warning.
    """
    try:
        import chromadb
    except ImportError as e:  # pragma: no cover - dep present in dev env
        raise RuntimeError("chromadb is required to read embeddings") from e

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection_name = get_collection_name()
    try:
        coll = client.get_collection(collection_name)
    except Exception:
        log.warning(
            "No '%s' collection at %s — run ingest first", collection_name, CHROMA_PATH
        )
        return {}

    data = coll.get(include=["embeddings", "metadatas"])
    raw_embeddings = data.get("embeddings")
    metadatas = data.get("metadatas") or []
    if raw_embeddings is None or len(raw_embeddings) == 0:
        log.warning("Collection has no embeddings")
        return {}

    hash_to_id = _hash_to_doc_id_map()
    grouped: dict[str, list[np.ndarray]] = {}
    dropped_no_id = 0

    for vec, meta in zip(raw_embeddings, metadatas, strict=False):
        meta = meta or {}
        doc_id_raw = meta.get("document_id")
        doc_id: str | None = str(doc_id_raw) if doc_id_raw else None
        if not doc_id:
            doc_hash_raw = meta.get("doc_hash")
            if doc_hash_raw:
                doc_id = hash_to_id.get(str(doc_hash_raw))
        if not doc_id:
            dropped_no_id += 1
            continue
        grouped.setdefault(doc_id, []).append(np.asarray(vec, dtype=np.float32))

    if dropped_no_id:
        log.warning("Dropped %d chunks with no resolvable document_id", dropped_no_id)
    return grouped
