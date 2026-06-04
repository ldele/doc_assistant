"""Unit tests for Chunk 2a synthesis logic (pure; no LLM/DB)."""

from __future__ import annotations

from doc_assistant.provenance import ConfidenceSignals, RetrievedChunk
from doc_assistant.synthesis import (
    MARKER_OK,
    MARKER_UNSUPPORTED,
    MARKER_WEAK,
    claim_marker,
    format_banner,
    render_evidence_markdown,
    render_interpretation_markdown,
    segment_claims,
    split_sentences,
)


def _src(num_score: float | None, filename: str = "paper.pdf", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(filename=filename, page=page, reranker_score=num_score)


# --- split_sentences ---------------------------------------------------------


def test_split_sentences_keeps_citations_attached() -> None:
    out = split_sentences("DPR beats BM25 [1]. It uses dual encoders [2]!")
    assert out == ["DPR beats BM25 [1].", "It uses dual encoders [2]!"]


def test_split_sentences_ignores_blank() -> None:
    assert split_sentences("   ") == []


# --- claim_marker ------------------------------------------------------------


def test_marker_uncited_is_unsupported() -> None:
    assert claim_marker([], [_src(0.9)]) == MARKER_UNSUPPORTED


def test_marker_hallucinated_citation_only_is_unsupported() -> None:
    # cites source 5 but only 1 source retrieved -> no real source
    from doc_assistant.synthesis import ClaimCitation

    assert claim_marker([ClaimCitation(source_number=5)], [_src(0.9)]) == MARKER_UNSUPPORTED


def test_marker_cited_strong_is_ok() -> None:
    from doc_assistant.synthesis import ClaimCitation

    assert claim_marker([ClaimCitation(source_number=1)], [_src(0.8)]) == MARKER_OK


def test_marker_cited_weak_score_is_weak() -> None:
    from doc_assistant.synthesis import ClaimCitation

    assert claim_marker([ClaimCitation(source_number=1)], [_src(0.1)]) == MARKER_WEAK


def test_marker_cited_real_but_no_score_is_weak() -> None:
    from doc_assistant.synthesis import ClaimCitation

    assert claim_marker([ClaimCitation(source_number=1)], [_src(None)]) == MARKER_WEAK


# --- segment_claims ----------------------------------------------------------


def test_segment_maps_citations_to_sources() -> None:
    sources = [_src(0.8, "a.pdf", 3), _src(0.9, "b.pdf", 5)]
    claims = segment_claims("Claim one [1]. Claim two [2].", sources)
    assert [c.claim_index for c in claims] == [0, 1]
    assert claims[0].source_numbers == [1]
    assert claims[0].citations[0].filename == "a.pdf"
    assert claims[0].citations[0].page == 3
    assert claims[0].marker == MARKER_OK


def test_segment_uncited_sentence_is_unsupported_claim() -> None:
    sources = [_src(0.8)]
    claims = segment_claims("Grounded claim [1]. Floating claim with no cite.", sources)
    assert claims[1].citations == []
    assert claims[1].marker == MARKER_UNSUPPORTED


def test_segment_dedupes_repeated_citation() -> None:
    claims = segment_claims("Both say it [1][1].", [_src(0.8)])
    assert claims[0].source_numbers == [1]


def test_segment_weak_source_flags_claim() -> None:
    claims = segment_claims("Shaky claim [1].", [_src(0.05)])
    assert claims[0].marker == MARKER_WEAK


# --- rendering ---------------------------------------------------------------


def test_render_evidence_lists_provenance() -> None:
    md = render_evidence_markdown([RetrievedChunk(filename="x.pdf", page=2, reranker_score=0.71)])
    assert "x.pdf" in md and "p.2" in md and "0.71" in md


def test_render_interpretation_quiet_on_ok_claims() -> None:
    md = render_interpretation_markdown(segment_claims("Solid [1].", [_src(0.8)]))
    assert "⚠" not in md  # quiet on clean claims


def test_render_interpretation_flags_weak_and_unsupported() -> None:
    sources = [_src(0.05)]
    md = render_interpretation_markdown(segment_claims("Weak [1]. Floating.", sources))
    assert "weakly grounded" in md and "unsupported" in md


def test_format_banner_none_when_clean() -> None:
    assert format_banner(ConfidenceSignals()) is None


def test_format_banner_fires_with_reasons() -> None:
    banner = format_banner(ConfidenceSignals(weak_retrieval=True, single_source_risk=True))
    assert banner is not None and "weak retrieval" in banner and "single-source" in banner
