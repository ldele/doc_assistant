"""Library text check: what the check found in one document (A2b).

Read-only and thin over `doc_assistant.document_checks`. Running the check again is the re-run
route's job (*Re-run… > Text checks*), so nothing here writes.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.models.text_checks import DocumentTextCheckPayload

router = APIRouter()


@router.get("/api/library/documents/{doc_id}/text-check")
def document_text_check(doc_id: str) -> DocumentTextCheckPayload:
    """The paragraphs of this document that address an AI, and whether that check is current."""
    from doc_assistant.document_checks import document_check

    view = document_check(doc_id)
    if view is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentTextCheckPayload.from_view(doc_id, view)
