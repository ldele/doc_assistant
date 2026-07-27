"""Similarity queries (Phase 4 close-out) + the ADR-027 D1 connections bundle."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.library.citations import CitationEdge, cited_by, cites_out

# ============================================================
# Similarity queries (Phase 4 close-out)
# ============================================================


@dataclass
class SimilarDoc:
    """One neighbour returned by `similar_docs`."""

    target_document_id: str
    target_filename: str
    target_title: str | None
    score: float


@dataclass
class CitedByDoc:
    """One in-corpus document that cites the subject doc (deduped; ``n_citations`` = how many
    of its extracted citations resolve to the subject)."""

    document_id: str
    filename: str
    n_citations: int


@dataclass
class DocConnections:
    """The exploration bundle for one document (ADR-027 D1, ROADMAP E4).

    List-shaped on purpose: the v1 panel renders lists, and a depth-1 ego graph is exactly
    ``cites`` + ``cited_by`` — a later graph/navigation iteration reads the same bundle (the
    recorded open gate; see the E4 DEVLOG entry). ``external_refs`` is the *titled* slice of
    the unresolved citations (the showable population), capped; ``external_total`` is the full
    titled count so the UI can say "showing N of M" honestly.
    """

    related: list[SimilarDoc]
    cites: list[CitationEdge]  # resolved, in-corpus only
    cited_by: list[CitedByDoc]
    external_refs: list[CitationEdge]  # unresolved + titled, capped
    external_total: int


# Payload cap for the external-references list — a wire-size bound (raw extraction yields
# ~50-60 refs per paper), not a corpus-tuned threshold: the full count still travels as
# `external_total`, nothing is hidden silently.
EXTERNAL_REFS_CAP = 50


def document_connections(
    doc_id: str,
    *,
    related_limit: int = 10,
    external_cap: int = EXTERNAL_REFS_CAP,
    embedding_model: str | None = None,
) -> DocConnections | None:
    """Assemble one document's exploration bundle (E4): related papers + citation edges.

    Pure read over the existing sidecars (``doc_similarities`` + ``citations``) — no model, no
    network. Returns ``None`` when the document is unknown (the API maps that to 404). Empty
    corpus / empty sidecars degrade to empty lists, never an error (the 0-doc contract).
    ``embedding_model`` scopes the similarity read to the embedder in use — callers should pass
    the active model name so the panel never mixes edges from different embedders.
    """
    with session_scope() as session:
        if session.get(Document, doc_id) is None:
            return None

    related = similar_docs(doc_id, limit=related_limit, embedding_model=embedding_model)
    all_cites = cites_out(doc_id)
    in_corpus = [c for c in all_cites if c.target_document_id is not None]
    titled_external = [
        c for c in all_cites if c.target_document_id is None and c.target_title is not None
    ]

    # Dedupe incoming citations by source document (a doc citing the subject 3 times is one
    # row with n_citations=3, not 3 rows) — preserving cited_by()'s filename ordering.
    by_source: dict[str, CitedByDoc] = {}
    for source_id, filename, _raw in cited_by(doc_id):
        if source_id in by_source:
            by_source[source_id].n_citations += 1
        else:
            by_source[source_id] = CitedByDoc(
                document_id=source_id, filename=filename, n_citations=1
            )

    return DocConnections(
        related=related,
        cites=in_corpus,
        cited_by=list(by_source.values()),
        external_refs=titled_external[:external_cap],
        external_total=len(titled_external),
    )


def similar_docs(
    doc_id: str,
    *,
    limit: int = 10,
    embedding_model: str | None = None,
) -> list[SimilarDoc]:
    """Return the top-N most similar documents to ``doc_id``.

    Reads pre-computed edges from the ``doc_similarities`` sidecar
    table. Empty list if no edges exist (run
    ``scripts/compute_doc_vectors.py --apply``).
    """
    from doc_assistant.db.models import DocSimilarity

    with session_scope() as session:
        stmt = (
            select(
                DocSimilarity.target_document_id,
                Document.filename,
                Document.title,
                DocSimilarity.score,
            )
            .join(Document, Document.id == DocSimilarity.target_document_id)
            .where(DocSimilarity.source_document_id == doc_id)
        )
        if embedding_model is not None:
            stmt = stmt.where(DocSimilarity.embedding_model == embedding_model)
        stmt = stmt.order_by(DocSimilarity.score.desc()).limit(limit)
        return [
            SimilarDoc(
                target_document_id=str(target_id),
                target_filename=str(filename),
                target_title=title,
                score=float(score),
            )
            for target_id, filename, title, score in session.execute(stmt).all()
        ]
