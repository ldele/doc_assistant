"""Library documents — the browser's read + write paths (feature-library-browser.md, L1).

List, drill in to chunks, the E4 connections bundle (ADR-027 D1), the ADR-013 metadata
override + reset, reveal-in-file-manager, and the ADR-014 safe delete.

Reads use lazy ``doc_assistant.library`` imports; chunk/delete reads go through the live Chroma
handle on ``ChatController.rag.db``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.connections import DocConnectionsPayload
from apps.api.models.library import (
    DeleteResultPayload,
    LibraryDocumentChunksPayload,
    LibraryDocumentMetaUpdate,
    LibraryDocumentPayload,
)
from doc_assistant.chat_controller import ChatController
from doc_assistant.embeddings import get_active_model_name

router = APIRouter()


@router.get("/api/library/documents")
def list_library_documents() -> list[LibraryDocumentPayload]:
    """Every ingested (non-archived) document for the Library browser (read-only, no model).

    A read over the SQLite ``Document`` store — feature-library-browser.md (L1)."""
    from doc_assistant.library import list_documents

    return [LibraryDocumentPayload.from_summary(s) for s in list_documents()]


@router.get("/api/library/documents/{doc_id}")
def get_library_document(request: Request, doc_id: str) -> LibraryDocumentChunksPayload:
    """One document's chunks grouped into parent blocks, or 404 if the document is unknown.

    Reads the live Chroma handle (``ChatController.rag.db``) via a metadata filter — no
    embeddings, no generation. A known doc with no stored chunks returns empty parents (not
    a 404)."""
    from doc_assistant.library import get_document_chunks

    controller: ChatController = request.app.state.controller
    view = get_document_chunks(doc_id, controller.rag.db)
    if view is None:
        raise HTTPException(status_code=404, detail="document not found")
    return LibraryDocumentChunksPayload.from_view(view)


@router.get("/api/library/documents/{doc_id}/connections")
def get_document_connections(doc_id: str) -> DocConnectionsPayload:
    """One document's exploration bundle (ADR-027 D1, ROADMAP E4): related papers
    (``doc_similarities``, scoped to the active embedder), resolved in-corpus citation
    edges both directions, and the extracted-but-unresolved external references.

    A pure sidecar read — no model, no Chroma, no LLM. 404 for an unknown document; a
    known document with empty sidecars returns empty lists (honest degrade, 0-doc
    contract). List-shaped by design: a later graph/navigation iteration reads the same
    bundle (recorded open gate, E4 DEVLOG)."""
    from doc_assistant.library import document_connections

    bundle = document_connections(doc_id, embedding_model=get_active_model_name())
    if bundle is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocConnectionsPayload.from_bundle(bundle)


@router.patch("/api/library/documents/{doc_id}")
def patch_library_document(doc_id: str, body: LibraryDocumentMetaUpdate) -> dict[str, bool]:
    """Set a document's user metadata overrides (title/authors/year). ADR-013 — first
    browse-time write path. 404 if the document is unknown; effective values are
    override ?? auto-extracted default."""
    from doc_assistant.library import get_document_details, set_document_meta

    if get_document_details(doc_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    set_document_meta(doc_id, title=body.title, authors=body.authors, year=body.year)
    return {"ok": True}


@router.post("/api/library/documents/{doc_id}/reset-metadata")
def reset_library_document_metadata(doc_id: str) -> dict[str, bool]:
    """Reset a document to its auto-extracted metadata (delete the override row). ADR-013."""
    from doc_assistant.library import clear_document_meta, get_document_details

    if get_document_details(doc_id) is None:
        raise HTTPException(status_code=404, detail="document not found")
    clear_document_meta(doc_id)
    return {"ok": True}


@router.post("/api/library/documents/{doc_id}/reveal")
def reveal_library_document(doc_id: str) -> dict[str, bool]:
    """Reveal the source file in the OS file manager (local desktop action, ADR-013).
    404 if the document is unknown or its source file can't be located (moved/deleted)."""
    from doc_assistant.library import reveal_document_source

    if not reveal_document_source(doc_id):
        raise HTTPException(status_code=404, detail="source file not found")
    return {"ok": True}


@router.delete("/api/library/documents/{doc_id}")
def delete_library_document(doc_id: str, request: Request) -> DeleteResultPayload:
    """Safe-delete a document: source file → Recycle Bin, then drop its DB row + index chunks
    (ADR-014). 404 if unknown; 409 if the source file couldn't be moved to the Recycle Bin."""
    from doc_assistant.library import delete_document

    controller: ChatController = request.app.state.controller
    try:
        result = delete_document(doc_id, controller.rag.db)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if result is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DeleteResultPayload(
        filename=result.filename,
        trashed_file=result.trashed_file,
        chunks_removed=result.chunks_removed,
    )
