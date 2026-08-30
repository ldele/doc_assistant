"""Tag-family wire models (feature-tag-families.md — PR-1 CRUD, PR-2 detection).

A family collapses near-duplicate keywords under one canonical label. The detection proposals
(PR-2) are deliberately *not* writes: a proposal is a deterministic, zero-LLM suggestion that
only becomes state when the user accepts it through the same CRUD bodies above it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from doc_assistant.knowledge.keyword_families import FamilyProposal
    from doc_assistant.library import KeywordFamily


class KeywordFamilyPayload(BaseModel):
    """A canonical tag + its member keyword names (mirrors ``library.KeywordFamily``)."""

    id: str
    canonical: str
    aliases: list[str]
    doc_count: int
    #: ADR-018 curation: whether this family's concept is part of the graph vocabulary. Not
    #: nullable on the wire even though the column is — the client renders a two-state control,
    #: and "unset" is not a third thing a user can mean.
    graph_include: bool

    @classmethod
    def from_family(cls, f: KeywordFamily) -> KeywordFamilyPayload:
        return cls(
            id=f.id,
            canonical=f.canonical,
            aliases=list(f.aliases),
            doc_count=f.doc_count,
            graph_include=f.graph_include,
        )


class KeywordFamilyCreate(BaseModel):
    """POST body to create a family: the canonical label + initial member keywords."""

    canonical: str = Field(min_length=1)
    members: list[str] = Field(default_factory=list)


class KeywordFamilyPatch(BaseModel):
    """PATCH body: rename a family, put it on the graph, or both.

    Both fields are optional so each may be sent alone — the rename path predates the graph flag
    and still sends only ``canonical``. A body with neither field is a 400 rather than a silent
    no-op: it is always a caller bug, and returning 200 for it would hide the bug behind a
    correct-looking payload.
    """

    canonical: str | None = Field(default=None, min_length=1)
    graph_include: bool | None = None


class KeywordFamilyMember(BaseModel):
    """POST body to add a member keyword to a family."""

    keyword: str = Field(min_length=1)


# ============================================================
# Detection (feature-tag-families.md — PR-2)
# ============================================================


class KeywordFamilyProposalPayload(BaseModel):
    """One deterministic, zero-LLM family proposal (mirrors ``keyword_families.FamilyProposal``).

    Nothing here has been written to the DB — accepting a proposal calls the existing family CRUD
    above (``POST``/``PATCH .../keyword-families``)."""

    canonical: str
    members: list[str]
    tier: Literal["morphological", "embedding"]
    confidence: float

    @classmethod
    def from_proposal(cls, p: FamilyProposal) -> KeywordFamilyProposalPayload:
        return cls(
            canonical=p.canonical, members=list(p.members), tier=p.tier, confidence=p.confidence
        )
