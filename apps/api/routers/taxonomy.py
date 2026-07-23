"""Taxonomy router (ADR-028 increment 2a) — serve the field forest, edit the curated hierarchy.

Read model + the write endpoints over the ``knowledge/taxonomy.py`` seam. Edits the *taxonomy*
(hierarchy edges + document→field links), never the concept vocabulary itself (ADR-017 A1 / ADR-019
D11: a dedicated taxonomy surface owns tree edits; concept create/rename/delete stay elsewhere).
$0, zero-LLM — the whole increment is deterministic sidecar reads/writes.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from apps.api.models import (
    FieldDetailPayload,
    HierarchyEdgeRequest,
    TaxonomyViewPayload,
)

router = APIRouter()


@router.get("/api/taxonomy")
def get_taxonomy() -> TaxonomyViewPayload:
    """The curated field forest + per-field coverage + corpus totals.

    Not 404 on an empty forest: the ANZSRC trunk is bundled seed data, so "no fields yet" is a
    legitimate unseeded state (run ``scripts/seed_taxonomy --apply``), not a missing artifact.
    Every rollup is 0 until concepts/documents are attached — the honest zero-state."""
    from doc_assistant.knowledge.taxonomy_view import load_taxonomy_view

    return TaxonomyViewPayload.from_view(load_taxonomy_view())


@router.get("/api/taxonomy/fields/{field_id}")
def get_field_detail(field_id: str) -> FieldDetailPayload:
    """One field's directly-attached concepts + documents + rollup counts.

    404 when ``field_id`` is not a ``kind="domain"`` node — a wrong id or a concept id — as
    distinct from a real-but-empty field, which returns a detail with empty member lists."""
    from doc_assistant.knowledge.taxonomy_view import load_field_detail

    detail = load_field_detail(field_id)
    if detail is None:
        raise HTTPException(status_code=404, detail=f"no taxonomy field with id {field_id!r}")
    return FieldDetailPayload.from_detail(detail)


@router.post("/api/taxonomy/hierarchy", status_code=201)
def add_hierarchy(body: HierarchyEdgeRequest) -> dict[str, Any]:
    """Add one curated hierarchy edge (``source --type--> target``). Idempotent on the unique key.

    Attaching a concept to a field is just an ``in_field`` edge from the concept to the domain
    node — same endpoint. ``409`` if the edge would create a cycle (ADR-028 D3); ``404`` if an id
    is not a concept/domain row; ``400`` on a bad ``type`` (guarded by the request model too)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Concept
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import TaxonomyCycleError, add_hierarchy_edge

    with session_scope() as session:
        present = set(
            session.execute(
                select(Concept.id).where(Concept.id.in_([body.source_id, body.target_id]))
            )
            .scalars()
            .all()
        )
        for cid in (body.source_id, body.target_id):
            if cid not in present:
                raise HTTPException(status_code=404, detail=f"no concept/field with id {cid!r}")
        try:
            add_hierarchy_edge(session, body.source_id, body.target_id, body.type)
        except TaxonomyCycleError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except ValueError as e:  # unknown edge type (the model narrows this, defence in depth)
            raise HTTPException(status_code=400, detail=str(e)) from e
    return {
        "ok": True,
        "source_id": body.source_id,
        "target_id": body.target_id,
        "type": body.type,
    }


@router.delete("/api/taxonomy/hierarchy")
def remove_hierarchy(body: HierarchyEdgeRequest) -> dict[str, int]:
    """Remove a curated hierarchy edge by its unique key. Idempotent (``removed`` 0 if absent)."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import remove_hierarchy_edge

    with session_scope() as session:
        removed = remove_hierarchy_edge(session, body.source_id, body.target_id, body.type)
    return {"removed": removed}


@router.post("/api/taxonomy/documents/{document_id}/fields/{field_id}", status_code=201)
def attach_document(document_id: str, field_id: str) -> dict[str, Any]:
    """Attach a document to a taxonomy field (a ``kind="domain"`` node). Idempotent per pair.

    ``404`` if the document does not exist; ``400`` if the target is not a domain field
    (``NotADomainError``)."""
    from sqlalchemy import select

    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy import NotADomainError, attach_document_field

    with session_scope() as session:
        exists = session.execute(
            select(Document.id).where(Document.id == document_id)
        ).scalar_one_or_none()
        if exists is None:
            raise HTTPException(status_code=404, detail=f"no document with id {document_id!r}")
        try:
            attach_document_field(session, document_id, field_id)
        except NotADomainError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return {"ok": True, "document_id": document_id, "field_id": field_id}
