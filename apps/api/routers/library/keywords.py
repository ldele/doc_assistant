"""Keyword families — curated tag vocabulary (feature-tag-families.md, PR-1 + PR-2).

A family is a curated ``Concept`` whose aliases are member ``Keyword`` names (ADR-015); a
keyword belongs to at most one family, so assigning it moves it.

``POST .../detect`` is the PR-2 detection pass and is deliberately read-only: it proposes
groupings and writes nothing. Accepting a proposal goes back through the CRUD routes above it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.keywords import (
    KeywordFamilyCreate,
    KeywordFamilyMember,
    KeywordFamilyPayload,
    KeywordFamilyProposalPayload,
    KeywordFamilyRename,
)
from doc_assistant.chat_controller import ChatController

router = APIRouter()


@router.get("/api/library/keyword-families")
def list_keyword_families_route() -> list[KeywordFamilyPayload]:
    """Every curated keyword family, each with its union doc_count (feature-tag-families.md,
    PR-1). A family is a curated Concept whose aliases are member Keyword names (ADR-015)."""
    from doc_assistant.library import list_keyword_families

    return [KeywordFamilyPayload.from_family(f) for f in list_keyword_families()]


@router.post("/api/library/keyword-families")
def create_keyword_family_route(body: KeywordFamilyCreate) -> KeywordFamilyPayload:
    """Create a family (canonical label + initial member keywords). Idempotent by canonical
    label; a member keyword already in another family is moved (ADR-015)."""
    from doc_assistant.library import create_keyword_family

    try:
        family = create_keyword_family(body.canonical, body.members)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return KeywordFamilyPayload.from_family(family)


@router.patch("/api/library/keyword-families/{family_id}")
def rename_keyword_family_route(family_id: str, body: KeywordFamilyRename) -> KeywordFamilyPayload:
    """Rename a family's canonical label. 404 if unknown, 409 if the label is taken.

    The uniqueness invariant lives in `library.rename_keyword_family` (PR-2.5 D1) — this shell
    only maps it to a status code. `KeywordFamilyExists` subclasses `ValueError`, so the
    ordering of these two handlers is load-bearing.
    """
    from doc_assistant.library import KeywordFamilyExists, rename_keyword_family

    try:
        family = rename_keyword_family(family_id, body.canonical)
    except KeywordFamilyExists as e:
        raise HTTPException(status_code=409, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if family is None:
        raise HTTPException(status_code=404, detail="keyword family not found")
    return KeywordFamilyPayload.from_family(family)


@router.post("/api/library/keyword-families/{family_id}/members")
def add_keyword_family_member_route(
    family_id: str, body: KeywordFamilyMember
) -> KeywordFamilyPayload:
    """Assign a keyword to a family, moving it off any other family it belonged to
    (ADR-015). 404 if the family is unknown."""
    from doc_assistant.library import add_family_member

    try:
        family = add_family_member(family_id, body.keyword)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    if family is None:
        raise HTTPException(status_code=404, detail="keyword family not found")
    return KeywordFamilyPayload.from_family(family)


@router.delete("/api/library/keyword-families/{family_id}/members/{keyword}")
def remove_keyword_family_member_route(family_id: str, keyword: str) -> KeywordFamilyPayload:
    """Remove a keyword from a family's alias set (a no-op if it isn't a member). 404 if the
    family is unknown."""
    from doc_assistant.library import remove_family_member

    family = remove_family_member(family_id, keyword)
    if family is None:
        raise HTTPException(status_code=404, detail="keyword family not found")
    return KeywordFamilyPayload.from_family(family)


@router.delete("/api/library/keyword-families/{family_id}")
def delete_keyword_family_route(family_id: str) -> dict[str, bool]:
    """Delete a family. 404 if unknown."""
    from doc_assistant.library import delete_keyword_family

    if not delete_keyword_family(family_id):
        raise HTTPException(status_code=404, detail="keyword family not found")
    return {"ok": True}


@router.post("/api/library/keyword-families/detect")
def detect_keyword_families_route(request: Request) -> list[KeywordFamilyProposalPayload]:
    """Deterministic, zero-LLM detection pass over every un-familied keyword (PR-2):
    morphological stem-matching (``llm``/``llms``) plus bge-embedding cosine clustering
    (``connectome``/``connectomics``). Nothing here writes to the DB — review a proposal, then
    create/extend a family through the existing CRUD routes above. Reuses the controller's
    already-loaded embedder (no new model load)."""
    from doc_assistant.library import detect_family_candidates

    controller: ChatController = request.app.state.controller

    def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in controller.rag.embeddings.embed_documents(texts)]

    proposals = detect_family_candidates(embed_fn=embed_fn)
    return [KeywordFamilyProposalPayload.from_proposal(p) for p in proposals]
