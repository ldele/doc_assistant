"""Taxonomy wire models (ADR-028 increment 2a) — the curated field forest + coverage.

The field forest with its direct/rollup coverage counts, one field's drill-in members, and the
single body that adds or removes a curated hierarchy edge.

Every count is 0 until concepts/documents are attached — the honest zero-state, not a missing
value — and ``origin`` keeps an ADR-028 D8 auto-proposal visually distinct from a user edit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

if TYPE_CHECKING:
    from doc_assistant.knowledge.taxonomy_view import FieldDetail, TaxonomyField, TaxonomyView


class TaxonomyFieldPayload(BaseModel):
    """One field (a `kind="domain"` node) with its structure + coverage counts.

    `*_direct` = attached straight to this field; `*_rollup` = the distinct set under this field or
    any narrower descendant (set-semantics, ADR-028 D6). Every count is 0 until concepts/documents
    are attached — the honest zero-state, not a missing value.
    """

    id: str
    label: str
    parent_ids: list[str]
    child_ids: list[str]
    n_concepts_direct: int
    n_documents_direct: int
    n_concepts_rollup: int
    n_documents_rollup: int
    # Of the *direct* members, how many arrived as an auto-proposal (ADR-028 D8) rather than a
    # user edit. The direct/rollup counts stay origin-inclusive; this is the subtractable share.
    n_concepts_proposed: int = 0
    n_documents_proposed: int = 0

    @classmethod
    def from_field(cls, f: TaxonomyField) -> TaxonomyFieldPayload:
        return cls(
            id=f.id,
            label=f.label,
            parent_ids=list(f.parent_ids),
            child_ids=list(f.child_ids),
            n_concepts_direct=f.n_concepts_direct,
            n_documents_direct=f.n_documents_direct,
            n_concepts_rollup=f.n_concepts_rollup,
            n_documents_rollup=f.n_documents_rollup,
            n_concepts_proposed=f.n_concepts_proposed,
            n_documents_proposed=f.n_documents_proposed,
        )


class TaxonomyViewPayload(BaseModel):
    """The whole field forest + corpus-level totals (the classification denominators)."""

    fields: list[TaxonomyFieldPayload]
    roots: list[str]
    n_concepts_total: int
    n_documents_total: int
    n_unassigned_concepts: int

    @classmethod
    def from_view(cls, v: TaxonomyView) -> TaxonomyViewPayload:
        return cls(
            fields=[TaxonomyFieldPayload.from_field(f) for f in v.fields],
            roots=list(v.roots),
            n_concepts_total=v.n_concepts_total,
            n_documents_total=v.n_documents_total,
            n_unassigned_concepts=v.n_unassigned_concepts,
        )


class FieldMemberPayload(BaseModel):
    """A directly-attached member (a concept or a document) of one field.

    `origin` is "curated" (a user edit or the ANZSRC seed) or "proposed" (an ADR-028 D8 auto-fill
    awaiting accept-or-delete) — the UI must not render a machine guess as the user's own edit.
    """

    id: str
    label: str
    origin: str = "curated"


class FieldDetailPayload(BaseModel):
    """One field's directly-attached concepts + documents + rollup counts (a drill-in)."""

    id: str
    label: str
    concepts: list[FieldMemberPayload]
    documents: list[FieldMemberPayload]
    n_concepts_rollup: int
    n_documents_rollup: int

    @classmethod
    def from_detail(cls, d: FieldDetail) -> FieldDetailPayload:
        return cls(
            id=d.id,
            label=d.label,
            concepts=[
                FieldMemberPayload(id=m.id, label=m.label, origin=m.origin) for m in d.concepts
            ],
            documents=[
                FieldMemberPayload(id=m.id, label=m.label, origin=m.origin) for m in d.documents
            ],
            n_concepts_rollup=d.n_concepts_rollup,
            n_documents_rollup=d.n_documents_rollup,
        )


class HierarchyEdgeRequest(BaseModel):
    """Add/remove one curated hierarchy edge (`source --type--> target`)."""

    source_id: str
    target_id: str
    type: Literal["is_a", "in_field"]
