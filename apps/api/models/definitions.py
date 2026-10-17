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
    from collections.abc import Mapping

    from doc_assistant.knowledge.concept_graph_view import VocabularyMatch
    from doc_assistant.knowledge.definitions import (
        ConceptDefinitions,
        ConceptUsage,
        DefinitionCandidate,
    )
    from doc_assistant.knowledge.term_signals import ConceptSignals, Example


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

    ``is_concept`` — false for a *term*, a row the user has not taken on (ADR-054). The panel then
    shows what the library says and offers no write: every definition route answers 409 for a
    term, and a candidate whose ``id`` starts with ``unsaved:`` was found on request, not stored.
    """

    concept_id: str
    label: str
    chosen_id: str | None
    candidates: list[DefinitionCandidatePayload]
    can_undo: bool
    thin: bool
    extracted: bool
    is_concept: bool = True

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
            is_concept=v.is_concept,
        )


class UsageLinePayload(BaseModel):
    text: str
    document_id: str
    document_title: str | None = None
    chunk_key: str
    doc_mentions: int


class ConceptUsagePayload(BaseModel):
    """How the library uses a concept: one plain sentence from each of the documents that use it
    most. Not candidates — nothing here can be chosen. ``available`` is false when the keyword
    index is not built yet, so an empty list is not read as "never used"."""

    concept_id: str
    available: bool
    examples: list[UsageLinePayload]

    @classmethod
    def from_view(cls, v: ConceptUsage) -> ConceptUsagePayload:
        return cls(
            concept_id=v.concept_id,
            available=v.available,
            examples=[
                UsageLinePayload(
                    text=u.text,
                    document_id=u.document_id,
                    document_title=u.document_title,
                    chunk_key=u.chunk_key,
                    doc_mentions=u.doc_mentions,
                )
                for u in v.examples
            ],
        )


class SignalExamplePayload(BaseModel):
    """A piece of the library's text a signal was read from. ``text`` is verbatim (collapse
    whitespace for display only); ``marks`` are ``[start, end]`` spans of it, as written."""

    text: str
    marks: list[tuple[int, int]]
    document_id: str
    document_title: str | None = None
    chunk_key: str

    @classmethod
    def from_example(cls, e: Example, titles: Mapping[str, str]) -> SignalExamplePayload:
        return cls(
            text=e.text,
            marks=list(e.marks),
            document_id=e.document_id,
            document_title=titles.get(e.document_id),
            chunk_key=e.chunk_key,
        )


class FormOfferPayload(BaseModel):
    """A form the library gives beside one of the row's own, offered and never added.

    ``kind`` ``long``: the row's form is the short one and ``form`` spells it out; ``short``: the
    row's form is the long one and ``form`` abbreviates it. ``beside`` is the row's form as the
    sentence writes it. ``known``: already one of the row's forms."""

    form: str
    beside: str
    kind: Literal["long", "short"]
    documents: int
    known: bool
    initials: bool
    example: SignalExamplePayload


class CitedPayload(BaseModel):
    """The label is mostly a cited author's name: ``et_al`` of its ``mentions`` are followed by
    "et al."."""

    et_al: int
    mentions: int
    example: SignalExamplePayload | None = None


class PlacePayload(BaseModel):
    """Where a label no sentence uses does occur, as the keyword extractor read it. ``sentence``
    is a body sentence that writes it another way (another case, other characters between its
    words), which no count includes; ``elsewhere`` is a line that is not a sentence (a list item,
    a short line, a page header or footer, code)."""

    place: Literal["sentence", "table", "heading", "figure", "references", "elsewhere"]
    mentions: int
    documents: int
    example: SignalExamplePayload


class ConceptSignalsPayload(BaseModel):
    """What the concept panel shows about a row's name — the signals measured on the user's
    answers only (``tests/eval/baselines/term_signals_2026-10-14.md``). Read-only and never
    stored, for a concept and a term alike.

    ``available`` is false when the keyword index is not built, so empty lists mean "not read".
    Every count is over ``documents_read``; ``documents_matched`` is the documents the index
    matched, which the read stops short of above a bound. ``silent``: no sentence of those
    documents uses the label, and ``places`` says where it occurs instead."""

    concept_id: str
    available: bool
    documents_read: int
    documents_matched: int
    mentions: int
    documents: int
    offers: list[FormOfferPayload]
    cited: CitedPayload | None = None
    silent: bool
    places: list[PlacePayload]

    @classmethod
    def from_view(cls, v: ConceptSignals) -> ConceptSignalsPayload:
        from doc_assistant.knowledge.term_signals import offer_example

        s = v.signals
        offers = []
        for o in s.offers if s is not None else ():
            offers.append(
                FormOfferPayload(
                    form=o.form,
                    beside=o.beside_written,
                    kind=o.kind,  # type: ignore[arg-type]
                    documents=o.documents,
                    known=o.known,
                    initials=o.initials,
                    example=SignalExamplePayload.from_example(offer_example(o), v.titles),
                )
            )
        cited = None
        if s is not None and s.cited:
            cited = CitedPayload(
                et_al=s.et_al,
                mentions=s.mentions,
                example=(
                    SignalExamplePayload.from_example(s.et_al_example, v.titles)
                    if s.et_al_example is not None
                    else None
                ),
            )
        return cls(
            concept_id=v.concept_id,
            available=v.available,
            documents_read=v.documents_read,
            documents_matched=v.documents_matched,
            mentions=s.mentions if s is not None else 0,
            documents=s.documents if s is not None else 0,
            offers=offers,
            cited=cited,
            silent=v.silent,
            places=[
                PlacePayload(
                    place=p.place,  # type: ignore[arg-type]
                    mentions=p.mentions,
                    documents=p.documents,
                    example=SignalExamplePayload.from_example(p.example, v.titles),
                )
                for p in v.places
            ],
        )


class UserDefinitionRequest(BaseModel):
    """The user's own definition, chosen on save unless ``choose`` is false. Replaces nothing."""

    text: str = Field(min_length=1, max_length=4000)
    choose: bool = True


class VocabularyMatchPayload(BaseModel):
    """One vocabulary row a search found. ``on_graph`` is also what makes it a *concept* rather
    than a *term* (ADR-054); ``written`` is the label as the library writes it when it writes it
    in a case, shown instead of ``label``."""

    id: str
    label: str
    on_graph: bool
    has_definition: bool
    written: str | None = None

    @classmethod
    def from_match(cls, m: VocabularyMatch) -> VocabularyMatchPayload:
        return cls(
            id=m.id,
            label=m.label,
            on_graph=m.on_graph,
            has_definition=m.has_definition,
            written=m.written,
        )
