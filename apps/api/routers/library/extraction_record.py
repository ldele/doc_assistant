"""Library extraction record: what indexing found in one document (0.9.0).

Read-only and thin over `doc_assistant.ingest.record`. Indexing and *Re-run…* write it; nothing
here does.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from apps.api.models.extraction_record import (
    DocumentExtractionRecordPayload,
    ExtractionRecordPayload,
)

router = APIRouter()


@router.get("/api/library/documents/{doc_id}/extraction-record")
def document_extraction_record(doc_id: str) -> DocumentExtractionRecordPayload:
    """The newest record for this document, its warnings, and whether it is stale."""
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.ingest.record import current_record

    with session_scope() as session:
        if session.get(Document, doc_id) is None:
            raise HTTPException(status_code=404, detail="document not found")
    rec = current_record(doc_id)
    return DocumentExtractionRecordPayload(
        document_id=doc_id,
        record=ExtractionRecordPayload.from_record(rec) if rec is not None else None,
    )
