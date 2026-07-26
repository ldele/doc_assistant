"""A/B-compare wire models (feature-ab-compare-sandbox.md — retrieval diff, U6).

Retrieval only, no answer generation: the query runs under the locked defaults (side A) and
under the session override (side B), and the two ranked source sets are diffed. Imports
``RagOverrides`` from ``chat`` — the same ADR-010 governance object the answer path uses, so a
compare cannot drift from the knobs it claims to be testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from apps.api.models.chat import RagOverrides

if TYPE_CHECKING:
    from doc_assistant.compare import CompareResult, CompareRow, CompareSource


class CompareRequest(BaseModel):
    """A/B-compare (U6): retrieve ``text`` under the locked defaults and under ``overrides``, and
    diff the two source sets. ``overrides=None`` = the session matches defaults (a no-op diff)."""

    text: str
    overrides: RagOverrides | None = None
    # ADR-025 F2 — applied to BOTH sides, so the diff isolates the knob rather than the corpus.
    scope_folder_id: str | None = None


class CompareEffPayload(BaseModel):
    """The effective retrieval knobs on one side of a compare."""

    top_k: int
    use_multi_query: bool


class CompareSourcePayload(BaseModel):
    rank: int
    filename: str
    page: int | None
    section: str | None
    score: float
    excerpt: str
    citation: str
    identity: str

    @classmethod
    def from_source(cls, s: CompareSource) -> CompareSourcePayload:
        return cls(
            rank=s.rank,
            filename=s.filename,
            page=s.page,
            section=s.section,
            score=s.score,
            excerpt=s.excerpt,
            citation=s.citation,
            identity=s.identity,
        )


class CompareRowPayload(BaseModel):
    identity: str
    source_a: CompareSourcePayload | None
    source_b: CompareSourcePayload | None
    status: str  # in_both | only_in_a | only_in_b
    rank_delta: int | None

    @classmethod
    def from_row(cls, r: CompareRow) -> CompareRowPayload:
        return cls(
            identity=r.identity,
            source_a=CompareSourcePayload.from_source(r.source_a) if r.source_a else None,
            source_b=CompareSourcePayload.from_source(r.source_b) if r.source_b else None,
            status=r.status,
            rank_delta=r.rank_delta,
        )


class CompareResultPayload(BaseModel):
    """Both ranked source sets (A = defaults, B = session override) + the diff + honesty note."""

    query: str
    sources_a: list[CompareSourcePayload]
    sources_b: list[CompareSourcePayload]
    rows: list[CompareRowPayload]
    eff_a: CompareEffPayload
    eff_b: CompareEffPayload
    note: str
    # ADR-025 F2 — the folder BOTH sides were retrieved under; null = the whole library.
    scope_label: str | None = None

    @classmethod
    def from_result(cls, r: CompareResult) -> CompareResultPayload:
        return cls(
            query=r.query,
            sources_a=[CompareSourcePayload.from_source(s) for s in r.sources_a],
            sources_b=[CompareSourcePayload.from_source(s) for s in r.sources_b],
            rows=[CompareRowPayload.from_row(x) for x in r.rows],
            eff_a=CompareEffPayload(
                top_k=int(r.eff_a["top_k"]),
                use_multi_query=bool(r.eff_a["use_multi_query"]),
            ),
            eff_b=CompareEffPayload(
                top_k=int(r.eff_b["top_k"]),
                use_multi_query=bool(r.eff_b["use_multi_query"]),
            ),
            note=r.note,
            scope_label=r.scope_label,
        )
