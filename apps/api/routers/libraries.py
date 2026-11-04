"""Libraries router — the read-only check before the app opens another library (ADR-061).

The desktop shell owns which library is open (it starts the backend with ``DOC_DATA_DIR``); this
backend only says what a candidate folder holds. Every judgement is
``doc_assistant.library_home``; the handler maps it to HTTP. A POST, because the folder travels in
the body and the launch token covers writes by method (S-6), although it writes nothing.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from apps.api.models.libraries import LibraryInspectionPayload, LibraryInspectRequest
from doc_assistant.library_home import inspect_library

router = APIRouter()


@router.post("/api/libraries/inspect")
def libraries_inspect(body: LibraryInspectRequest) -> LibraryInspectionPayload:
    """What ``body.path`` holds as a library, and whether the app may open it."""
    found = inspect_library(Path(body.path).expanduser())
    return LibraryInspectionPayload(
        path=found.path,
        is_library=found.is_library,
        documents=found.documents,
        chunks=found.chunks,
        embedding_model=found.embedding_model,
        sources_found=found.sources_found,
        cache_named_folder=found.cache_named_folder,
        is_current=found.is_current,
        problems=found.problems,
        notes=found.notes,
        can_open=found.can_open,
    )
