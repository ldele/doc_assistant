"""Library router — documents, folders, and keyword families.

The Library browser's read + write paths (feature-library-browser.md, feature-corpus-folders.md,
feature-tag-families.md). Reads use lazy ``doc_assistant.library`` imports; chunk/delete reads go
through the live Chroma handle on ``ChatController.rag.db``.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.models import (
    DeleteResultPayload,
    DocConnectionsPayload,
    FolderCreate,
    FolderMembers,
    FolderRename,
    KeywordFamilyCreate,
    KeywordFamilyMember,
    KeywordFamilyPayload,
    KeywordFamilyProposalPayload,
    KeywordFamilyRename,
    LibraryDocumentChunksPayload,
    LibraryDocumentMetaUpdate,
    LibraryDocumentPayload,
    LibraryFolderPayload,
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


@router.get("/api/library/folders")
def list_folders_route() -> list[LibraryFolderPayload]:
    """Every folder with its non-archived member count (feature-corpus-folders.md, ADR-025 F1).

    A folder organises the Library and, since F2, can scope one chat turn's retrieval
    (POST /api/chat `scope_folder_id`)."""
    from doc_assistant.library import list_folders

    return [LibraryFolderPayload.from_folder(f) for f in list_folders()]


@router.post("/api/library/folders")
def create_folder_route(body: FolderCreate) -> LibraryFolderPayload:
    """Create a folder. Idempotent on the case-folded name (returns the existing one)."""
    from doc_assistant.library import create_folder

    try:
        folder = create_folder(body.name, body.description)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return LibraryFolderPayload.from_folder(folder)


@router.patch("/api/library/folders/{folder_id}")
def rename_folder_route(folder_id: str, body: FolderRename) -> LibraryFolderPayload:
    """Rename a folder. 404 if unknown, 400 if the name collides with another folder."""
    from doc_assistant.library import rename_folder

    try:
        folder = rename_folder(folder_id, body.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if folder is None:
        raise HTTPException(status_code=404, detail="folder not found")
    return LibraryFolderPayload.from_folder(folder)


@router.post("/api/library/folders/{folder_id}/documents")
def add_folder_documents_route(folder_id: str, body: FolderMembers) -> LibraryFolderPayload:
    """Add documents to a folder (bulk, idempotent; unknown ids are skipped). 404 if
    the folder is unknown."""
    from doc_assistant.library import add_documents_to_folder

    folder = add_documents_to_folder(folder_id, body.document_ids)
    if folder is None:
        raise HTTPException(status_code=404, detail="folder not found")
    return LibraryFolderPayload.from_folder(folder)


@router.delete("/api/library/folders/{folder_id}/documents/{doc_id}")
def remove_folder_document_route(folder_id: str, doc_id: str) -> LibraryFolderPayload:
    """Remove one document from a folder (a no-op if it isn't a member). The document
    itself is untouched. 404 if the folder is unknown."""
    from doc_assistant.library import remove_documents_from_folder

    folder = remove_documents_from_folder(folder_id, [doc_id])
    if folder is None:
        raise HTTPException(status_code=404, detail="folder not found")
    return LibraryFolderPayload.from_folder(folder)


@router.delete("/api/library/folders/{folder_id}")
def delete_folder_route(folder_id: str) -> dict[str, bool]:
    """Delete a folder. Never deletes documents (spec D6). 404 if unknown."""
    from doc_assistant.library import delete_folder

    if not delete_folder(folder_id):
        raise HTTPException(status_code=404, detail="folder not found")
    return {"ok": True}


@router.get("/api/library/keyword-families")
def list_keyword_families_route() -> list[KeywordFamilyPayload]:
    """Every curated keyword family, each with its union doc_count (feature-tag-families.md,
    PR-1). A family is a curated Concept whose aliases are member Keyword names (ADR-015)."""
    from doc_assistant.library import list_keyword_families

    return [KeywordFamilyPayload.from_family(f) for f in list_keyword_families()]


@router.post("/api/library/keyword-families")
def create_keyword_family_route(body: KeywordFamilyCreate) -> KeywordFamilyPayload:
    """Create a family (canonical label + initial member keywords). Idempotent by canonical
    label; a member keyword already in another family is moved (ADR-015)."""
    from doc_assistant.library import create_keyword_family

    try:
        family = create_keyword_family(body.canonical, body.members)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return KeywordFamilyPayload.from_family(family)


@router.patch("/api/library/keyword-families/{family_id}")
def rename_keyword_family_route(family_id: str, body: KeywordFamilyRename) -> KeywordFamilyPayload:
    """Rename a family's canonical label. 404 if unknown, 409 if the label is taken.

    The uniqueness invariant lives in `library.rename_keyword_family` (PR-2.5 D1) — this shell
    only maps it to a status code. `KeywordFamilyExists` subclasses `ValueError`, so the
    ordering of these two handlers is load-bearing.
    """
    from doc_assistant.library import KeywordFamilyExists, rename_keyword_family

    try:
        family = rename_keyword_family(family_id, body.canonical)
    except KeywordFamilyExists as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if family is None:
        raise HTTPException(status_code=404, detail="keyword family not found")
    return KeywordFamilyPayload.from_family(family)


@router.post("/api/library/keyword-families/{family_id}/members")
def add_keyword_family_member_route(
    family_id: str, body: KeywordFamilyMember
) -> KeywordFamilyPayload:
    """Assign a keyword to a family, moving it off any other family it belonged to
    (ADR-015). 404 if the family is unknown."""
    from doc_assistant.library import add_family_member

    try:
        family = add_family_member(family_id, body.keyword)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if family is None:
        raise HTTPException(status_code=404, detail="keyword family not found")
    return KeywordFamilyPayload.from_family(family)


@router.delete("/api/library/keyword-families/{family_id}/members/{keyword}")
def remove_keyword_family_member_route(family_id: str, keyword: str) -> KeywordFamilyPayload:
    """Remove a keyword from a family's alias set (a no-op if it isn't a member). 404 if the
    family is unknown."""
    from doc_assistant.library import remove_family_member

    family = remove_family_member(family_id, keyword)
    if family is None:
        raise HTTPException(status_code=404, detail="keyword family not found")
    return KeywordFamilyPayload.from_family(family)


@router.delete("/api/library/keyword-families/{family_id}")
def delete_keyword_family_route(family_id: str) -> dict[str, bool]:
    """Delete a family. 404 if unknown."""
    from doc_assistant.library import delete_keyword_family

    if not delete_keyword_family(family_id):
        raise HTTPException(status_code=404, detail="keyword family not found")
    return {"ok": True}


@router.post("/api/library/keyword-families/detect")
def detect_keyword_families_route(request: Request) -> list[KeywordFamilyProposalPayload]:
    """Deterministic, zero-LLM detection pass over every un-familied keyword (PR-2):
    morphological stem-matching (``llm``/``llms``) plus bge-embedding cosine clustering
    (``connectome``/``connectomics``). Nothing here writes to the DB — review a proposal, then
    create/extend a family through the existing CRUD routes above. Reuses the controller's
    already-loaded embedder (no new model load)."""
    from doc_assistant.library import detect_family_candidates

    controller: ChatController = request.app.state.controller

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in controller.rag.embeddings.embed_documents(texts)]

    proposals = detect_family_candidates(embed_fn=embed_fn)
    return [KeywordFamilyProposalPayload.from_proposal(p) for p in proposals]
