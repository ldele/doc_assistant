"""Tests for the document health classifier."""

from doc_assistant.health import classify_document_health


def test_clearly_healthy_document():
    """A typical academic paper should be healthy."""
    report = classify_document_health(
        chunk_count=100,
        avg_chunk_length=800,
        page_count=15,
        section_detection_rate=0.6,
        format="pdf",
    )
    assert report.status == "healthy"
    assert report.score >= 75


def test_single_chunk_is_broken():
    """A document with 1 chunk is broken regardless of other signals."""
    report = classify_document_health(
        chunk_count=1,
        avg_chunk_length=5000,
        page_count=20,
        section_detection_rate=1.0,
        format="pdf",
    )
    assert report.status == "broken"


def test_pdf_with_no_pages_penalty():
    """PDF with no detected pages loses points."""
    healthy_pdf = classify_document_health(
        chunk_count=100,
        avg_chunk_length=800,
        page_count=15,
        section_detection_rate=0.5,
        format="pdf",
    )
    no_pages_pdf = classify_document_health(
        chunk_count=100,
        avg_chunk_length=800,
        page_count=None,
        section_detection_rate=0.5,
        format="pdf",
    )
    assert no_pages_pdf.score < healthy_pdf.score


def test_epub_not_penalized_for_no_pages():
    """EPUB shouldn't lose points for not having pages."""
    report = classify_document_health(
        chunk_count=100,
        avg_chunk_length=800,
        page_count=None,
        section_detection_rate=0.5,
        format="epub",
    )
    # EPUB without pages should still be healthy
    assert report.status == "healthy"


def test_short_chunks_flagged():
    """Documents with very short average chunks are marginal at best."""
    report = classify_document_health(
        chunk_count=50,
        avg_chunk_length=50,
        page_count=10,
        section_detection_rate=0.3,
        format="pdf",
    )
    assert report.status in ("marginal", "broken")


def test_reference_heavy_document():
    """Documents with >40% references chunks lose points."""
    report = classify_document_health(
        chunk_count=200,
        avg_chunk_length=800,
        page_count=20,
        section_detection_rate=0.5,
        format="pdf",
        reference_flagged_ratio=0.6,
    )
    assert "references" in " ".join(report.reasons).lower()


# --- KI-53: "broken" must mean the extractor failed, not that the document is short -----------
#
# Every case below was `broken` under the previous `chunk_count <= 1` / `chunks_per_page < 2`
# rules. They are written against the *verdict a user reads*, not the score, because the verdict
# is what made a working document look like one to delete.


def test_short_html_article_is_not_broken():
    """A 2 KB web article extracts to one full chunk. That is complete, not failed."""
    report = classify_document_health(
        chunk_count=1,
        avg_chunk_length=900,
        page_count=None,
        section_detection_rate=0.0,
        format="html",
    )
    assert report.status == "healthy"


def test_short_epub_chapter_is_not_broken():
    """Two full chunks out of an unpaged container is a short chapter, not a collapse."""
    report = classify_document_health(
        chunk_count=2,
        avg_chunk_length=800,
        page_count=None,
        section_detection_rate=0.5,
        format="epub",
    )
    assert report.status == "healthy"


def test_single_fragment_is_broken():
    """A lone scrap is still broken — too little to be a document, paged or not."""
    report = classify_document_health(
        chunk_count=1,
        avg_chunk_length=80,
        page_count=None,
        section_detection_rate=0.0,
        format="html",
    )
    assert report.status == "broken"


def test_short_plaintext_note_is_never_broken():
    """Reading a short .md file back is not an extraction failure — it is the file."""
    report = classify_document_health(
        chunk_count=1,
        avg_chunk_length=120,
        page_count=None,
        section_detection_rate=0.0,
        format="md",
    )
    assert report.status != "broken"


def test_multipage_pdf_yielding_one_chunk_is_still_broken():
    """The signal that survives: pages that hold text the extraction did not produce."""
    report = classify_document_health(
        chunk_count=1,
        avg_chunk_length=900,
        page_count=12,
        section_detection_rate=0.0,
        format="pdf",
    )
    assert report.status == "broken"
    assert "12 pages" in " ".join(report.reasons)


def test_pages_yielding_almost_no_text_are_broken():
    """A failed text layer shows up as characters-per-page, whatever the chunk count.

    20 pages yielding 1,000 characters between them is 50 a page — the scanned-PDF failure this
    classifier exists to catch, and it must land on `broken`, not on a hedge.
    """
    report = classify_document_health(
        chunk_count=5,
        avg_chunk_length=200,
        page_count=20,
        section_detection_rate=0.0,
        format="pdf",
    )
    assert report.status == "broken"
    assert "the text layer failed" in " ".join(report.reasons)


def test_a_short_paged_document_with_full_chunks_is_not_broken():
    """Three pages, two full chunks — sparse, but the extraction plainly worked.

    This is the case the replaced `chunk_count <= 3` penalty got wrong in the other direction:
    fewer chunks than pages cost 50 points even when the per-page yield was healthy.
    """
    report = classify_document_health(
        chunk_count=2,
        avg_chunk_length=900,
        page_count=3,
        section_detection_rate=0.5,
        format="pdf",
    )
    assert report.status == "healthy"


def test_ordinary_sparse_paper_is_not_penalized():
    """A real paper at ~1.3 chunks/page tripped the old ratio floor and read as marginal."""
    report = classify_document_health(
        chunk_count=20,
        avg_chunk_length=900,
        page_count=15,
        section_detection_rate=0.5,
        format="pdf",
    )
    assert report.status == "healthy"
    assert report.reasons == []
