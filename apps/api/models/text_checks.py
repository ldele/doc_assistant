"""Text-check wire models: what the check found in one document (A2b, docs/security.md S5).

Mirrors `doc_assistant.document_checks`; the TypeScript side is `core/types/textChecks.ts`.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from apps.api.models._common import _as_utc

if TYPE_CHECKING:
    from doc_assistant.document_checks import DocumentCheck


class TextFindingPayload(BaseModel):
    """One paragraph that addresses an AI: its page (``None`` without pages), kinds and start."""

    page: int | None
    labels: list[str]
    excerpt: str


class DocumentTextCheckPayload(BaseModel):
    """``state``: ``current``, ``stale`` (the text or the patterns changed since), or
    ``not_checked``. ``n_marked`` counts every marked paragraph; ``findings`` lists the first."""

    document_id: str
    state: Literal["current", "stale", "not_checked"]
    checked_at: datetime | None
    n_marked: int
    findings: list[TextFindingPayload]

    @classmethod
    def from_view(cls, document_id: str, v: DocumentCheck) -> DocumentTextCheckPayload:
        return cls(
            document_id=document_id,
            state=v.state,
            checked_at=_as_utc(v.checked_at) if v.checked_at is not None else None,
            n_marked=v.n_marked,
            findings=[
                TextFindingPayload(page=f.page, labels=list(f.labels), excerpt=f.excerpt)
                for f in v.findings
            ],
        )
