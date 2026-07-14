"""Unit tests for Chunk 2a synthesis logic (pure; no LLM/DB)."""

from __future__ import annotations

from doc_assistant.provenance import ConfidenceSignals, RetrievedChunk
from doc_assistant.synthesis import (
    MARKER_OK,
    MARKER_UNSUPPORTED,
    MARKER_WEAK,
    audit_citations,
    cited_source_numbers,
    claim_marker,
    format_banner,
    render_evidence_markdown,
    render_interpretation_markdown,
    segment_claims,
    split_sentences,
)


def _src(num_score: float | None, filename: str = "paper.pdf", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(filename=filename, page=page, reranker_score=num_score)


# --- audit_citations ---------------------------------------------------------


def test_audit_citations_clean_answer() -> None:
    a = audit_citations("DPR uses dual encoders [1]. It beats BM25 [2].", n_sources=5)
    assert a.valid == [1, 2]
    assert a.clean and not a.out_of_range and not a.malformed
    assert a.n_sentences == 2 and a.n_uncited_sentences == 0


def test_audit_citations_flags_out_of_range() -> None:
    a = audit_citations("A claim [1]. Another [11].", n_sources=5)
    assert a.valid == [1] and a.out_of_range == [11]
    assert not a.clean


def test_audit_citations_flags_malformed_key_and_filename() -> None:
    # The two real failure forms: a BibTeX-ish key and an inline filename — both
    # invisible to the [n] parser, so the audit is what surfaces them.
    a = audit_citations("DPR per [karp2020dense]. See (dpr_kumar_2020.pdf) and [1].", n_sources=3)
    assert a.valid == [1]
    assert any("karp2020dense" in m for m in a.malformed)
    assert any("dpr_kumar_2020.pdf" in m for m in a.malformed)
    assert not a.clean
    assert "malformed" in a.note()


def test_audit_citations_counts_uncited_sentences() -> None:
    a = audit_citations("Cited claim [1]. Uncited framing sentence here.", n_sources=2)
    assert a.n_uncited_sentences == 1 and a.clean  # uncited != unclean


# --- cited_source_numbers (robust citation parsing) --------------------------


def test_cited_numbers_canonical() -> None:
    assert cited_source_numbers("DPR [1] beats BM25 [2][3].") == [1, 2, 3]


def test_cited_numbers_source_label() -> None:
    # haiku's dominant non-canonical form: "[Source N]" / "[Sources N, M]".
    assert cited_source_numbers("Per [Source 2] and [Sources 3, 4].") == [2, 3, 4]
    assert cited_source_numbers("case-insensitive [source 5]") == [5]


def test_cited_numbers_bare_lists() -> None:
    assert cited_source_numbers("Combined [2, 4] and [1 and 3] and [5; 6].") == [2, 4, 1, 3, 5, 6]


def test_cited_numbers_ignores_non_citations() -> None:
    # A BibTeX-ish key, a wrapped claim label, and a chemical bracket are NOT citations.
    assert cited_source_numbers("[karp2020dense] [term-based system] [Ca2+] [see 2]") == []


# --- robust forms feed audit + segment ---------------------------------------


def test_audit_source_label_is_valid_not_malformed() -> None:
    a = audit_citations("BM25 is sparse [Source 2]. Dense is latent [Sources 1, 3].", n_sources=5)
    assert a.valid == [1, 2, 3]
    assert a.clean and not a.malformed  # [Source 2] no longer flagged malformed
    assert a.n_uncited_sentences == 0  # both sentences now read as cited


def test_audit_out_of_range_source_label() -> None:
    a = audit_citations("Cites [Source 9] only.", n_sources=3)
    assert a.out_of_range == [9] and not a.malformed and not a.clean


def test_audit_still_flags_genuinely_malformed() -> None:
    a = audit_citations("Per [karp2020dense] and [term-based system].", n_sources=3)
    assert any("karp2020dense" in m for m in a.malformed)
    assert any("term-based system" in m for m in a.malformed)


def test_segment_source_label_claim_is_supported() -> None:
    # "[Source 1]" used to read as uncited → wrongly "unsupported"; now attributed.
    claims = segment_claims("Grounded via [Source 1].", [_src(0.8)])
    assert claims[0].source_numbers == [1]
    assert claims[0].marker == MARKER_OK


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
