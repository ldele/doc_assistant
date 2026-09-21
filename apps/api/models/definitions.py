"""Concept-definition wire models (ADR-053, ROADMAP 93).

One concept's definition candidates as the Graph tab's concept panel shows them, the body that
writes the user's own definition, and the vocabulary search that makes every concept reachable
from that panel — not only the ones on the graph.

``source`` is ``passage`` (a verbatim sentence from the library; ``chunk_key`` opens it where it
was written), ``user`` or ``model``. ``grade`` is ``strong`` / ``some`` / ``thin`` for a passage
and ``user`` for the user's own words; ``reasons`` are the evidence it was derived from, shown
as they are — a grade is never a model rating itself.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from doc_assistant.knowledge.concept_graph_view import VocabularyMatch
    from doc_assistant.knowledge.definitions import ConceptDefinitions, DefinitionCandidate


class DefinitionCandidatePayload(BaseModel):
    id: str
    text: str
    source: Literal["passage", "user", "model"]
    status: Literal["suggested", "chosen", "dismissed"]
    grade: str
    reasons: list[str]
    document_id: str | None = None
    document_title: str | None = None
    chunk_key: str | None = None

    @classmethod
    def from_candidate(cls, c: DefinitionCandidate) -> DefinitionCandidatePayload:
        return cls(
            id=c.id,
            text=c.text,
            source=c.source,  # type: ignore[arg-type]
            status=c.status,  # type: ignore[arg-type]
            grade=c.grade,
            reasons=list(c.reasons),
            document_id=c.document_id,
            document_title=c.document_title,
            chunk_key=c.chunk_key,
        )


class ConceptDefinitionsPayload(BaseModel):
    """Everything the panel shows about one concept's definition.

    ``thin`` — nothing chosen and no passage shaped like a definition: the panel says the library
    has little to go on. ``extracted`` — whether any passage candidate is on record, so the panel
    can offer to look in the library rather than implying it already did.
    """

    concept_id: str
    label: str
    chosen_id: str | None
    candidates: list[DefinitionCandidatePayload]
    can_undo: bool
    thin: bool
    extracted: bool

    @classmethod
    def from_view(cls, v: ConceptDefinitions) -> ConceptDefinitionsPayload:
        return cls(
            concept_id=v.concept_id,
            label=v.label,
            chosen_id=v.chosen_id,
            candidates=[DefinitionCandidatePayload.from_candidate(c) for c in v.candidates],
            can_undo=v.can_undo,
            thin=v.thin,
            extracted=v.extracted,
        )


class UserDefinitionRequest(BaseModel):
    """The user's own definition, chosen on save unless ``choose`` is false. Replaces nothing."""

    text: str = Field(min_length=1, max_length=4000)
    choose: bool = True


class VocabularyMatchPayload(BaseModel):
    id: str
    label: str
    on_graph: bool
    has_definition: bool

    @classmethod
    def from_match(cls, m: VocabularyMatch) -> VocabularyMatchPayload:
        return cls(id=m.id, label=m.label, on_graph=m.on_graph, has_definition=m.has_definition)
