"""Definitions router — read, write and choose a concept's definition (ADR-053, ROADMAP 93).

The one place the Graph tab *edits* a concept. The concepts router stays read-only over the
vocabulary (ADR-017 A1); a definition is the exception the user chose on 2026-09-21 — it is read
where the concept is, so it is written there too. Every route is a thin call into
``knowledge.definitions``, which owns the rules: candidates are never overwritten, choosing is the
only write to ``Concept.definition``, and every choice can be undone. Each mutation answers with
the concept's refreshed candidates, so the panel never patches its own copy. $0, zero LLM.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from apps.api.models.definitions import (
    ConceptDefinitionsPayload,
    ConceptUsagePayload,
    UserDefinitionRequest,
    VocabularyMatchPayload,
)

router = APIRouter()


def _view(concept_id: str) -> ConceptDefinitionsPayload:
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import load_definitions

    with session_scope() as session:
        view = load_definitions(session, concept_id)
        if view is None:
            raise HTTPException(status_code=404, detail=f"no concept with id {concept_id!r}")
        return ConceptDefinitionsPayload.from_view(view)


def _act(concept_id: str, definition_id: str, action: str) -> ConceptDefinitionsPayload:
    """Run one candidate action, after checking the candidate belongs to this concept."""
    from doc_assistant.db.models import ConceptDefinition
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge import definitions as d

    with session_scope() as session:
        row = session.get(ConceptDefinition, definition_id)
        if row is None or row.concept_id != concept_id:
            raise HTTPException(
                status_code=404,
                detail=f"no definition {definition_id!r} on concept {concept_id!r}",
            )
        try:
            if action == "choose":
                d.choose_definition(session, definition_id)
            elif action == "dismiss":
                d.dismiss_definition(session, definition_id)
            else:
                d.restore_definition(session, definition_id)
        except d.DefinitionError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
    return _view(concept_id)


@router.get("/api/concepts/search")
def search_vocabulary(
    q: str = Query(default="", max_length=200), limit: int = Query(default=20, ge=1, le=100)
) -> list[VocabularyMatchPayload]:
    """Concepts whose label contains ``q`` — the whole vocabulary, not only the graph's nodes.

    Declared before the ``/api/concepts/{concept_id}/…`` routes. An empty query returns ``[]``."""
    from doc_assistant.knowledge.concept_graph_view import search_vocabulary as search

    return [VocabularyMatchPayload.from_match(m) for m in search(q, limit=limit)]


@router.get("/api/concepts/{concept_id}/definitions")
def get_definitions(concept_id: str) -> ConceptDefinitionsPayload:
    """One concept's definition candidates, the chosen one first. 404 for an unknown id or a
    taxonomy field (a field is not a concept and has no definition here)."""
    return _view(concept_id)


@router.get("/api/concepts/{concept_id}/usage")
def get_usage(concept_id: str) -> ConceptUsagePayload:
    """How the library uses this concept — a plain sentence from each of the documents that use it
    most. Read-only and never stored; a few documents are read, whatever the library's size.
    404 for an unknown id or a taxonomy field."""
    from doc_assistant.knowledge.definitions import load_usage

    usage = load_usage(concept_id)
    if usage is None:
        raise HTTPException(status_code=404, detail=f"no concept with id {concept_id!r}")
    return ConceptUsagePayload.from_view(usage)


@router.post("/api/concepts/{concept_id}/definitions", status_code=201)
def add_definition(concept_id: str, body: UserDefinitionRequest) -> ConceptDefinitionsPayload:
    """Store the user's own definition — chosen unless ``choose`` is false. The previous choice
    stays as a candidate, one undo away."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import DefinitionError, add_user_definition

    _view(concept_id)  # 404 before any write
    with session_scope() as session:
        try:
            add_user_definition(session, concept_id, body.text, choose=body.choose)
        except DefinitionError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    return _view(concept_id)


@router.post("/api/concepts/{concept_id}/definitions/extract")
def extract_definitions_for(concept_id: str) -> ConceptDefinitionsPayload:
    """Look in the library for sentences that define this concept, and store them as suggestions.

    Deterministic and free. The keyword index picks the documents that mention it, so this reads
    those documents, not the whole store; without the index it falls back to the full read.
    Idempotent: running it again adds only what is new and never touches a choice."""
    from doc_assistant.knowledge.definitions import chunks_mentioning, extract_definitions

    label = _view(concept_id).label
    extract_definitions(concept_ids=[concept_id], apply=True, chunks=chunks_mentioning([label]))
    return _view(concept_id)


@router.post("/api/concepts/{concept_id}/definitions/undo")
def undo_definition(concept_id: str) -> ConceptDefinitionsPayload:
    """Take back the most recent choose / dismiss / restore on this concept. 409 when there is
    nothing left to undo."""
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.definitions import undo_last

    _view(concept_id)
    with session_scope() as session:
        if undo_last(session, concept_id) is None:
            raise HTTPException(status_code=409, detail="nothing to undo")
    return _view(concept_id)


@router.post("/api/concepts/{concept_id}/definitions/{definition_id}/choose")
def choose(concept_id: str, definition_id: str) -> ConceptDefinitionsPayload:
    """Make this candidate the concept's definition; the previous one goes back to suggested."""
    return _act(concept_id, definition_id, "choose")


@router.post("/api/concepts/{concept_id}/definitions/{definition_id}/dismiss")
def dismiss(concept_id: str, definition_id: str) -> ConceptDefinitionsPayload:
    """Hide a candidate (kept, restorable). Dismissing the chosen one leaves no definition."""
    return _act(concept_id, definition_id, "dismiss")


@router.post("/api/concepts/{concept_id}/definitions/{definition_id}/restore")
def restore(concept_id: str, definition_id: str) -> ConceptDefinitionsPayload:
    """Bring a dismissed candidate back as a suggestion. 409 if it was not dismissed."""
    return _act(concept_id, definition_id, "restore")
