"""Library summary: the *Summary* layout's counts, for the library or one folder (ROADMAP 112).

Read-only and thin over `doc_assistant.library.summarise_collection`. The one thing added here is
what only the API holds: this session's last index run (`app.state.ingest_run`), whose failures
and their reasons live in memory until the app closes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.summary import CollectionSummaryPayload

router = APIRouter()


def _last_run_failed(request: Request) -> int | None:
    run: Any = getattr(request.app.state, "ingest_run", None)
    if run is None:
        return None
    return int(run.counts().get("failed", 0))


@router.get("/api/library/summary")
def library_summary(request: Request, folder_id: str | None = None) -> CollectionSummaryPayload:
    """The summary of the whole library, or of the folder ``folder_id`` (404 if it is unknown)."""
    from doc_assistant.library import summarise_collection

    summary = summarise_collection(folder_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="folder not found")
    return CollectionSummaryPayload.from_summary(
        summary, last_run_failed=_last_run_failed(request)
    )
