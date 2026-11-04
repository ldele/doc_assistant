"""Libraries wire models: the check before the app opens another library folder (ADR-061)."""

from __future__ import annotations

from pydantic import BaseModel


class LibraryInspectRequest(BaseModel):
    """POST /api/libraries/inspect body: the folder to read. Nothing is written."""

    path: str


class LibraryInspectionPayload(BaseModel):
    """What the folder holds. Mirrors ``doc_assistant.library_home.LibraryInspection``."""

    path: str
    is_library: bool
    documents: int
    chunks: int
    embedding_model: str
    sources_found: int
    #: The library folder its stored cache paths name, when it is not this one (a copy or a move).
    cache_named_folder: str | None
    is_current: bool
    #: Sentences that stop *Open*; empty means it may open.
    problems: list[str]
    #: Sentences to read before *Open*.
    notes: list[str]
    can_open: bool
