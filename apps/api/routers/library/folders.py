"""Library folders — manual corpus organisation (feature-corpus-folders.md, ADR-025 F1).

Flat named folders with bulk membership. Since F2 a folder can also scope one chat turn's
retrieval (``POST /api/chat`` ``scope_folder_id``), so these writes are visible on the answer
path — but only by id: membership is resolved per turn, never cached here.

Deleting a folder never deletes documents (spec D6).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.models.folders import (
    FolderCreate,
    FolderMembers,
    FolderRename,
    LibraryFolderPayload,
)

router = APIRouter()


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
