"""Compute doc-level vectors and persist similarity edges.

Reads the baseline Chroma store, mean-pools chunk embeddings per
document, computes pairwise cosine similarity, and writes top-K edges
per source above a threshold to the ``doc_similarities`` table.

Idempotent: skips persistence if edges already exist for this
``embedding_model`` unless ``--force`` is passed (which clears edges
for that model first).

Usage::

    python -m scripts.compute_doc_vectors                  # dry-run
    python -m scripts.compute_doc_vectors --apply          # write edges
    python -m scripts.compute_doc_vectors --apply --force  # recompute
    python -m scripts.compute_doc_vectors --doc <hash>     # report one doc
"""

from __future__ import annotations

import argparse
import sys

from sqlalchemy import delete, func, select

from doc_assistant.db.models import DocSimilarity, Document
from doc_assistant.db.session import session_scope
from doc_assistant.doc_vectors import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_MODEL_NAME,
    SimilarityEdge,
    compute_doc_vectors,
    compute_similarity_edges,
    load_chunk_embeddings_by_document,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _existing_edge_count(embedding_model: str) -> int:
    with session_scope() as session:
        return int(
            session.execute(
                select(func.count())
                .select_from(DocSimilarity)
                .where(DocSimilarity.embedding_model == embedding_model)
            ).scalar_one()
        )


def _persist_edges(edges: list[SimilarityEdge], embedding_model: str, *, force: bool) -> int:
    """Write edges. With ``force``, deletes existing rows for this model first."""
    with session_scope() as session:
        if force:
            session.execute(
                delete(DocSimilarity).where(DocSimilarity.embedding_model == embedding_model)
            )
        else:
            existing = session.execute(
                select(func.count())
                .select_from(DocSimilarity)
                .where(DocSimilarity.embedding_model == embedding_model)
            ).scalar_one()
            if existing:
                return 0

        for e in edges:
            session.add(
                DocSimilarity(
                    source_document_id=e.source_doc_id,
                    target_document_id=e.target_doc_id,
                    embedding_model=embedding_model,
                    score=e.score,
                )
            )
        return len(edges)


def _filename_lookup(doc_ids: list[str]) -> dict[str, str]:
    if not doc_ids:
        return {}
    with session_scope() as session:
        rows = session.execute(
            select(Document.id, Document.filename).where(Document.id.in_(doc_ids))
        ).all()
        return {str(r[0]): str(r[1]) for r in rows}


def _resolve_doc_filter(doc_arg: str) -> str | None:
    """Map ``--doc`` (id-prefix or hash-prefix) to a Document.id."""
    with session_scope() as session:
        by_id = (
            session.execute(select(Document.id).where(Document.id.like(f"{doc_arg}%")))
            .scalars()
            .all()
        )
        if len(by_id) == 1:
            return str(by_id[0])
        by_hash = (
            session.execute(select(Document.id).where(Document.doc_hash.like(f"{doc_arg}%")))
            .scalars()
            .all()
        )
        if len(by_hash) == 1:
            return str(by_hash[0])
        return None


def _format_report(
    docs_count: int,
    vectors_count: int,
    edges: list[SimilarityEdge],
    embedding_model: str,
    *,
    apply: bool,
    inserted: int,
    filter_doc_id: str | None,
) -> str:
    names = _filename_lookup(
        list({e.source_doc_id for e in edges} | {e.target_doc_id for e in edges})
    )
    by_source: dict[str, list[SimilarityEdge]] = {}
    for edge in edges:
        by_source.setdefault(edge.source_doc_id, []).append(edge)

    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Embedding model:           {embedding_model}")
    out.append(f"Documents in library:      {docs_count}")
    out.append(f"Doc-level vectors built:   {vectors_count}")
    out.append(f"Similarity edges produced: {len(edges)}")
    if apply:
        out.append(f"Edges persisted:           {inserted}")
    out.append("=" * 76)
    out.append("")

    if filter_doc_id:
        sources = [filter_doc_id]
    else:
        sources = sorted(by_source.keys(), key=lambda d: names.get(d, d))
    for src in sources:
        src_edges = by_source.get(src, [])
        if not src_edges:
            continue
        out.append(f"  {names.get(src, src[:8])}")
        for edge in sorted(src_edges, key=lambda x: -x.score):
            out.append(
                f"    {edge.score:5.3f}  {names.get(edge.target_doc_id, edge.target_doc_id[:8])}"
            )
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Persist edges to the DB")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear existing edges for this embedding model before writing",
    )
    parser.add_argument(
        "--doc",
        type=str,
        help="Report only this doc's edges (id or hash prefix). Computation is always global.",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_TOP_K, help="Edges per source (default %(default)s)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum cosine score (default %(default)s)",
    )
    args = parser.parse_args()

    filter_doc_id: str | None = None
    if args.doc:
        filter_doc_id = _resolve_doc_filter(args.doc)
        if filter_doc_id is None:
            print(f"--doc '{args.doc}' did not uniquely resolve to one document.")
            return 1

    print("Reading chunk embeddings from Chroma...")
    chunk_embeddings = load_chunk_embeddings_by_document()
    if not chunk_embeddings:
        print("No chunk embeddings found. Run `uv run python -m doc_assistant.ingest` first.")
        return 1

    doc_vectors = compute_doc_vectors(chunk_embeddings)
    edges = compute_similarity_edges(doc_vectors, top_k=args.top_k, threshold=args.threshold)

    inserted = 0
    if args.apply:
        if not args.force and _existing_edge_count(EMBEDDING_MODEL_NAME):
            print(f"Edges already exist for {EMBEDDING_MODEL_NAME}. Pass --force to recompute.")
        else:
            inserted = _persist_edges(edges, EMBEDDING_MODEL_NAME, force=args.force)

    print(
        _format_report(
            docs_count=len(chunk_embeddings),
            vectors_count=len(doc_vectors),
            edges=edges,
            embedding_model=EMBEDDING_MODEL_NAME,
            apply=args.apply,
            inserted=inserted,
            filter_doc_id=filter_doc_id,
        )
    )
    if not args.apply:
        print("Dry run. Pass --apply to persist edges.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
