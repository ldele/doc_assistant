"""Library letter-spacing: the document page's block and the ignore list (ADR-056 5, 9, 10).

Reads one document's recorded runs; writes the user's ignore list, one run, one document or all of
it. Nothing here re-extracts: a change takes effect when the document's text is re-run, which is
the existing re-run route's job. Thin over `doc_assistant.library.spacing`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.models.spacing import (
    DocumentSpacingPayload,
    IgnoreListEntryPayload,
    IgnoreListPayload,
    IgnoreListRemovedPayload,
    IgnoreListRemoveRequest,
    IgnoreListRestoreRequest,
    SpacingEditRequest,
    SpacingKeepAllRequest,
    SpacingRunRequest,
)

router = APIRouter()


def _view(doc_id: str) -> DocumentSpacingPayload:
    from doc_assistant.library import document_spacing

    view = document_spacing(doc_id)
    if view is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentSpacingPayload.from_view(view)


def _refusal(e: Exception) -> HTTPException:
    """An unknown document is a 404; a document with no PDF to key on, or a list that cannot be
    read, is a 409 whose detail is the sentence to show."""
    status = 404 if str(e) == "document not found" else 409
    return HTTPException(status_code=status, detail=str(e))


@router.get("/api/library/documents/{doc_id}/spacing")
def document_spacing(doc_id: str) -> DocumentSpacingPayload:
    """The runs this document's extraction recorded, and what the ignore list asks for now."""
    return _view(doc_id)


@router.post("/api/library/documents/{doc_id}/spacing/keep")
def keep_run(doc_id: str, body: SpacingRunRequest) -> DocumentSpacingPayload:
    from doc_assistant.library import SpacingListError, keep_run

    try:
        keep_run(doc_id, body.spaced)
    except (LookupError, SpacingListError) as e:
        raise _refusal(e) from e
    return _view(doc_id)


@router.post("/api/library/documents/{doc_id}/spacing/edit")
def edit_run(doc_id: str, body: SpacingEditRequest) -> DocumentSpacingPayload:
    from doc_assistant.library import SpacingListError, set_run_spacing

    try:
        set_run_spacing(doc_id, body.spaced, body.chosen)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except (LookupError, SpacingListError) as e:
        raise _refusal(e) from e
    return _view(doc_id)


@router.post("/api/library/documents/{doc_id}/spacing/remove")
def remove_run(doc_id: str, body: SpacingRunRequest) -> DocumentSpacingPayload:
    from doc_assistant.library import SpacingListError, remove_run

    try:
        remove_run(doc_id, body.spaced)
    except (LookupError, SpacingListError) as e:
        raise _refusal(e) from e
    return _view(doc_id)


@router.post("/api/library/documents/{doc_id}/spacing/keep-all")
def keep_all(doc_id: str, body: SpacingKeepAllRequest) -> DocumentSpacingPayload:
    from doc_assistant.library import SpacingListError, set_keep_all

    try:
        set_keep_all(doc_id, body.keep)
    except (LookupError, SpacingListError) as e:
        raise _refusal(e) from e
    return _view(doc_id)


@router.get("/api/library/spacing")
def ignore_list() -> IgnoreListPayload:
    """Every document the ignore list names, for Settings."""
    from doc_assistant.library import SpacingListError, list_entries

    try:
        entries = list_entries()
    except SpacingListError as e:
        raise _refusal(e) from e
    return IgnoreListPayload(entries=[IgnoreListEntryPayload.from_entry(e) for e in entries])


@router.post("/api/library/spacing/remove")
def remove_entries(body: IgnoreListRemoveRequest) -> IgnoreListRemovedPayload:
    """Remove one document's entries, or all; the reply carries the list as it was, for Undo."""
    from doc_assistant.library import SpacingListError, remove_entries

    try:
        previous = remove_entries(body.sha256)
    except SpacingListError as e:
        raise _refusal(e) from e
    gone = [
        e
        for e in previous.get("documents", [])
        if body.sha256 is None or e.get("sha256") == body.sha256
    ]
    return IgnoreListRemovedPayload(
        removed=len(gone), documents=[str(e.get("file", "")) for e in gone], previous=previous
    )


@router.post("/api/library/spacing/restore")
def restore_entries(body: IgnoreListRestoreRequest) -> IgnoreListPayload:
    """Undo: put back the list a removal returned."""
    from doc_assistant.library import SpacingListError, list_entries, restore_entries

    try:
        restore_entries(body.previous)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except SpacingListError as e:
        raise _refusal(e) from e
    return IgnoreListPayload(
        entries=[IgnoreListEntryPayload.from_entry(e) for e in list_entries()]
    )
