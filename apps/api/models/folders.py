"""Folder wire models (docs/specs/feature-corpus-folders.md — ADR-025 F1).

Manual Library organisation: a flat set of named folders with bulk membership. The chat-side
consumer of a folder is ``ChatRequest.scope_folder_id`` (ADR-025 F2), which crosses the wire as
an id only — membership is resolved per turn, so a Library edit is never stale by answer time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from doc_assistant.library import FolderSummary


class LibraryFolderPayload(BaseModel):
    """One folder plus its non-archived member count (mirrors ``library.FolderSummary``).

    ``parent_id`` is always None in F1 — folders are flat until nesting is decided (spec D1);
    the field is on the wire so adding nesting later is not a contract break."""

    id: str
    name: str
    description: str | None
    parent_id: str | None
    doc_count: int

    @classmethod
    def from_folder(cls, f: FolderSummary) -> LibraryFolderPayload:
        return cls(
            id=f.id,
            name=f.name,
            description=f.description,
            parent_id=f.parent_id,
            doc_count=f.doc_count,
        )


class FolderCreate(BaseModel):
    """POST body to create a folder."""

    name: str = Field(min_length=1)
    description: str | None = None


class FolderRename(BaseModel):
    """PATCH body to rename a folder."""

    name: str = Field(min_length=1)


class FolderMembers(BaseModel):
    """POST body to add documents to a folder (bulk; idempotent)."""

    document_ids: list[str] = Field(min_length=1)
