"""Tests for the document health classifier."""
import pytest
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
        chunk_count=100, avg_chunk_length=800,
        page_count=15, section_detection_rate=0.5, format="pdf"
    )
    no_pages_pdf = classify_document_health(
        chunk_count=100, avg_chunk_length=800,
        page_count=None, section_detection_rate=0.5, format="pdf"
    )
    assert no_pages_pdf.score < healthy_pdf.score


def test_epub_not_penalized_for_no_pages():
    """EPUB shouldn't lose points for not having pages."""
    report = classify_document_health(
        chunk_count=100, avg_chunk_length=800,
        page_count=None, section_detection_rate=0.5, format="epub"
    )
    # EPUB without pages should still be healthy
    assert report.status == "healthy"


def test_short_chunks_flagged():
    """Documents with very short average chunks are marginal at best."""
    report = classify_document_health(
        chunk_count=50, avg_chunk_length=50,
        page_count=10, section_detection_rate=0.3, format="pdf"
    )
    assert report.status in ("marginal", "broken")


def test_reference_heavy_document():
    """Documents with >40% references chunks lose points."""
    report = classify_document_health(
        chunk_count=200, avg_chunk_length=800,
        page_count=20, section_detection_rate=0.5,
        format="pdf", reference_flagged_ratio=0.6,
    )
    assert "references" in " ".join(report.reasons).lower()