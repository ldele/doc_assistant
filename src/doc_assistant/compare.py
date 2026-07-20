"""A/B-compare (retrieval diff) — the pure diff of two retrieved source sets.

Design lock: ``docs/specs/feature-ab-compare-sandbox.md`` (U6, v1). The controller runs retrieval
twice — A = the locked defaults, B = the session ``RagOverrides`` — and hands the two ranked source
lists here; this module classifies the diff and writes the honest note. **Pure**: no LLM, no
retrieval, no I/O. Retrieval itself ($0, no generation) is the controller's job.

Honesty (ADR-010 governance, `benchmark-presentation-tone`): only ``top_k`` and ``use_multi_query``
affect retrieval, and ``top_k`` alone changes only depth (one set nests in the other) — so the note
says plainly when an override can't move membership. A sandbox outcome is indicative on one query,
never a verdict; the eval harness stays the only path to a new default.
"""

from __future__ import annotations

from dataclasses import dataclass

# The effective retrieval knobs on one side of a compare. Values are int (top_k) / bool
# (use_multi_query) — kept int-able so callers can `int(eff["top_k"])` for the retrieve call.
EffKnobs = dict[str, int | bool]


@dataclass(frozen=True)
class CompareSource:
    """One retrieved source on one side of the compare."""

    rank: int  # 1-indexed position in this side's ranked list
    filename: str
    page: int | None
    section: str | None
    score: float  # reranker score
    excerpt: str
    citation: str  # `format_citation` string, "[rank] name · p.N · \"section\""
    identity: str  # sha256(page_content)+"_"+doc_hash — the pipeline's own dedup key


@dataclass(frozen=True)
class CompareRow:
    """One source in the unioned diff, matched (or not) across A and B."""

    identity: str
    source_a: CompareSource | None
    source_b: CompareSource | None
    status: str  # in_both | only_in_a | only_in_b
    rank_delta: int | None  # a_rank - b_rank when in_both (negative = B ranks it higher)


@dataclass(frozen=True)
class CompareResult:
    """The whole comparison: both ranked lists, the diff, effective knobs, the honesty note."""

    query: str
    sources_a: list[CompareSource]
    sources_b: list[CompareSource]
    rows: list[CompareRow]
    eff_a: EffKnobs  # {"top_k": int, "use_multi_query": bool}
    eff_b: EffKnobs
    note: str
    # ADR-025 F2 — the folder both sides were retrieved under, or None for the whole library.
    # BOTH sides always share it: the comparison is about the knob, so the document set has to
    # be held constant, and a diff computed over a different corpus than the next answer will
    # use would be its own quiet lie.
    scope_label: str | None = None


_FAR = 10**9


def diff_sources(a: list[CompareSource], b: list[CompareSource]) -> list[CompareRow]:
    """Union A and B by ``identity``, classify each, and order by best available rank.

    A source present in both is ``in_both`` with ``rank_delta = a_rank - b_rank``; otherwise
    ``only_in_a`` / ``only_in_b``. Rows are ordered by the stronger of the two ranks so the most
    relevant sources lead, regardless of which side they came from.
    """
    by_a = {s.identity: s for s in a}
    by_b = {s.identity: s for s in b}

    ordered_ids: list[str] = []
    seen: set[str] = set()
    for s in [*a, *b]:
        if s.identity not in seen:
            seen.add(s.identity)
            ordered_ids.append(s.identity)

    rows: list[CompareRow] = []
    for ident in ordered_ids:
        sa = by_a.get(ident)
        sb = by_b.get(ident)
        if sa is not None and sb is not None:
            rows.append(CompareRow(ident, sa, sb, "in_both", sa.rank - sb.rank))
        elif sa is not None:
            rows.append(CompareRow(ident, sa, None, "only_in_a", None))
        elif sb is not None:
            rows.append(CompareRow(ident, None, sb, "only_in_b", None))

    def best_rank(r: CompareRow) -> int:
        ra = r.source_a.rank if r.source_a else _FAR
        rb = r.source_b.rank if r.source_b else _FAR
        return min(ra, rb)

    rows.sort(key=best_rank)
    return rows


def compare_note(eff_a: EffKnobs, eff_b: EffKnobs) -> str:
    """The honest note (Decision 6): when the override cannot move retrieval membership, say so.

    Retrieval-affecting knobs are ``top_k`` and ``use_multi_query`` only. If both match the
    defaults, the override changes the *answer*, not the sources. If only ``top_k`` differs, the
    ranking is identical and one set nests in the other (depth, not membership). Otherwise
    (``use_multi_query`` differs) membership genuinely moves and the diff speaks for itself.
    """
    if eff_a == eff_b:
        return (
            "This override doesn't change retrieval — its knobs affect the answer, "
            "not which sources are retrieved."
        )
    if eff_a["use_multi_query"] == eff_b["use_multi_query"]:
        delta = int(eff_b["top_k"]) - int(eff_a["top_k"])
        more = "more" if delta > 0 else "fewer"
        return (
            f"Same ranking, {abs(delta)} {more} source(s) — top_k changes depth only "
            "(one set includes the other)."
        )
    return ""


def build_result(
    query: str,
    sources_a: list[CompareSource],
    sources_b: list[CompareSource],
    eff_a: EffKnobs,
    eff_b: EffKnobs,
    scope_label: str | None = None,
) -> CompareResult:
    """Assemble the :class:`CompareResult` from the two ranked source lists + effective knobs."""
    return CompareResult(
        query=query,
        sources_a=sources_a,
        sources_b=sources_b,
        rows=diff_sources(sources_a, sources_b),
        eff_a=eff_a,
        eff_b=eff_b,
        note=compare_note(eff_a, eff_b),
        scope_label=scope_label,
    )
