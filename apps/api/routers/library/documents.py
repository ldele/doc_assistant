"""Library documents — the browser's read + write paths (feature-library-browser.md, L1).

List, drill in to chunks, the E4 connections bundle (ADR-027 D1), the reference list, the
ADR-013 metadata override + reset, reveal-in-file-manager, and the ADR-014 safe delete.

Reads use lazy ``doc_assistant.library`` imports; chunk/delete reads go through the live Chroma
handle on ``ChatController.rag.db``.
"""

from __future__ import annotations

import threading

import structlog
from fastapi import APIRouter, HTTPException, Request

from apps.api.models.connections import DocConnectionsPayload
from apps.api.models.library import (
    ChunkContextPayload,
    DeleteResultPayload,
    LibraryDocumentChunksPayload,
    LibraryDocumentFiguresPayload,
    LibraryDocumentMetaUpdate,
    LibraryDocumentPayload,
    ReingestOptionsPayload,
    ReingestPartPayload,
    ReingestRequest,
)
from apps.api.models.references import DocReferencesPayload
from doc_assistant.chat_controller import ChatController
from doc_assistant.embeddings import get_active_model_name

log = structlog.get_logger(__name__)
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


@router.get("/api/library/documents/{doc_id}/figures")
def get_document_figures(doc_id: str) -> LibraryDocumentFiguresPayload:
    """One document's figures, addressed **separately from its text chunks** (L1b).

    A figure is a different kind of object from prose — an image with a caption — so it gets
    its own panel rather than being interleaved into the chunk browser.

    A pure sidecar read: the 4b `Figure` rows plus the 4c VLM descriptions. No Chroma, no
    model, no LLM. 404 for an unknown document; a known document with no detectable figures
    returns an empty list (the 0-figure contract, not an error).

    Each figure reports whether it is **retrievable** — a figure enters retrieval only once it
    has a description — and, when it is not, why. Listing figures without that would show
    images the assistant cannot actually see, with nothing to distinguish them."""
    from doc_assistant.library import list_document_figures

    view = list_document_figures(doc_id)
    if view is None:
        raise HTTPException(status_code=404, detail="document not found")
    return LibraryDocumentFiguresPayload.from_view(view)


@router.get("/api/library/documents/{doc_id}/connections")
def get_document_connections(doc_id: str) -> DocConnectionsPayload:
    """One document's exploration bundle (ADR-027 D1, ROADMAP E4): related papers
    (``doc_similarities``, scoped to the active embedder), resolved in-corpus citation
    edges both directions, and the extracted-but-unresolved external references.

    A pure sidecar read — no model, no Chroma, no LLM. 404 for an unknown document; a
    known document with empty sidecars returns empty lists (honest degrade, 0-doc
    contract). List-shaped by design: a later graph/navigation iteration reads the same
    bundle (recorded open gate, E4 DEVLOG).

    The document's *own* reference list is ``/references``, not this bundle."""
    from doc_assistant.library import document_connections

    bundle = document_connections(doc_id, embedding_model=get_active_model_name())
    if bundle is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocConnectionsPayload.from_bundle(bundle)


@router.get("/api/library/documents/{doc_id}/references")
def get_document_references(doc_id: str) -> DocReferencesPayload:
    """One document's reference list — the paper's bibliography, in one list.

    Every extracted reference, including the ones that resolved to nothing: a bibliography
    with the unmatched entries removed would misrepresent what the paper cites. The entries
    that *did* resolve carry a ``document_id``, which is what lets the UI open them.

    A pure sidecar read over the ``citations`` table — no model, no Chroma, no LLM, no
    network. 404 for an unknown document; a document whose references were never extracted
    returns an empty list (the 0-doc contract), not an error."""
    from doc_assistant.library import document_references

    view = document_references(doc_id)
    if view is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocReferencesPayload.from_view(view)


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
def delete_library_document(
    doc_id: str, request: Request, delete_file: bool = False
) -> DeleteResultPayload:
    """Remove a document from the library; bin its source file only if asked (ADR-046 §2).

    ``delete_file`` defaults to **False** — the safe half of the amendment to ADR-014, which used
    to bin the source unconditionally. When true, ADR-014's bin-first-then-remove ordering is
    preserved. 404 if unknown; 409 if the source file could not be moved to the Recycle Bin.

    A client asking for ``delete_file=true`` on a *referenced* document is asserting it showed the
    user the real path first (ADR-046 §2). That is a UI contract; the path ships on every library
    row as ``source_path`` so the client can honour it.
    """
    from doc_assistant.library import delete_document

    controller: ChatController = request.app.state.controller
    try:
        result = delete_document(doc_id, controller.rag.db, delete_file=delete_file)
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    if result is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DeleteResultPayload(
        filename=result.filename,
        trashed_file=result.trashed_file,
        chunks_removed=result.chunks_removed,
    )


@router.get("/api/library/chunk-context")
def chunk_context(request: Request, key: str, window: int = 700) -> ChunkContextPayload:
    """Where a cited chunk sits in its source — the passage plus what surrounds it (ROADMAP 19).

    `key` is the epistemics-format `chunk_key` a citation already carries, so the client needs no
    new identifier. **404 when the chunk cannot be placed** — an unresolved span, a cache that is
    gone, an unknown key — because the alternative is a window centred on the wrong paragraph,
    which is the failure this whole feature refuses.

    `window` is a character radius, clamped: a caller asking for the whole document should read
    the document, not this.
    """
    from doc_assistant.library import get_chunk_context

    controller: ChatController = request.app.state.controller
    ctx = get_chunk_context(key, controller.rag.db, window=max(100, min(window, 4000)))
    if ctx is None:
        raise HTTPException(status_code=404, detail="this chunk cannot be placed in its source")
    return ChunkContextPayload.from_context(ctx)


# --- per-part re-ingest (ADR-048, ROADMAP 20/21) ---------------------------------------------- #


@router.get("/api/library/reingest/options")
def reingest_options() -> ReingestOptionsPayload:
    """What a re-run can do, straight from the registry — the client never hardcodes a cost.

    Also names the passes that have **no** per-document form, because a user who cannot find a
    button deserves to be told there is no button (ADR-048).
    """
    from doc_assistant.reingest import CORPUS_WIDE_PASSES, PARTS

    return ReingestOptionsPayload(
        parts=[
            ReingestPartPayload(
                id=p.id,
                label=p.label,
                blurb=p.blurb,
                cost=p.cost,
                moves_identity=p.moves_identity,
            )
            for p in PARTS
        ],
        corpus_wide=list(CORPUS_WIDE_PASSES),
    )


@router.post("/api/library/documents/reingest", status_code=202)
def reingest_start(request: Request, body: ReingestRequest) -> dict[str, object]:
    """Start a per-part re-run in the background. 202 + poll, like every other job here.

    **409 while an ingest is running.** Both write the same chunk stores and the same `Document`
    rows; letting them overlap would race a re-extract against a corpus scan for the same file.
    The parts are validated *before* the 202 so a bad body is an error rather than a job that
    fails a second later out of sight.
    """
    from doc_assistant.reingest import PART_IDS

    unknown = sorted(set(body.parts) - PART_IDS)
    if unknown:
        raise HTTPException(status_code=400, detail={"error": "unknown parts", "parts": unknown})

    app_ = request.app
    if app_.state.ingest_status.state == "running":
        raise HTTPException(status_code=409, detail="an indexing run is already in progress")

    status = app_.state.reingest_status
    with app_.state.reingest_lock:
        if status.state == "running":
            raise HTTPException(status_code=409, detail="a re-run is already in progress")
        status.state = "running"
        status.total = len(body.document_ids) * len(set(body.parts))
        status.done = 0
        status.current = None
        status.ok = status.skipped = status.errors = 0
        status.message = None
        status.outcomes = None

    def _on_progress(done: int, total: int, current: str | None) -> None:
        with app_.state.reingest_lock:
            status.done, status.total, status.current = done, total, current

    def _worker() -> None:
        from apps.api.services import REINGEST_OUTCOME_CAP

        try:
            result = app_.state.reingest_fn(
                list(body.document_ids), list(body.parts), on_progress=_on_progress
            )
        except Exception as e:  # a crashed job must still leave a readable status
            log.warning("reingest_failed", error=str(e))
            with app_.state.reingest_lock:
                status.state = "error"
                status.message = f"{type(e).__name__}: {e}"
                status.current = None
            return
        with app_.state.reingest_lock:
            status.state = "done"
            status.current = None
            status.ok, status.skipped, status.errors = result.ok, result.skipped, result.errors
            status.outcomes = [
                {
                    "document_id": o.document_id,
                    "filename": o.filename,
                    "part": o.part,
                    "status": o.status,
                    "detail": o.detail,
                }
                for o in result.outcomes[:REINGEST_OUTCOME_CAP]
            ]
            status.message = _reingest_message(result)

    threading.Thread(target=_worker, daemon=True).start()
    return {"started": True, "total": status.total}


def _reingest_message(result: object) -> str:
    """One line for the status bar. Counts, not adjectives — the detail is in `outcomes`."""
    ok = getattr(result, "ok", 0)
    skipped = getattr(result, "skipped", 0)
    errors = getattr(result, "errors", 0)
    parts = [f"{ok} re-run"]
    if skipped:
        parts.append(f"{skipped} skipped")
    if errors:
        parts.append(f"{errors} failed")
    return " · ".join(parts)


@router.get("/api/library/reingest/status")
def reingest_status(request: Request) -> dict[str, object]:
    """Poll the re-run. Same position/outcome split as the ingest status."""
    from apps.api.services import _reingest_status_dict

    return _reingest_status_dict(request.app)
