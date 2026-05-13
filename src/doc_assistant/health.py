"""Document health classification based on extraction signals.

A document is classified as:
- "healthy": extraction looks good, document is fully usable
- "marginal": something's off but probably usable
- "broken": extraction failed, probably needs re-extraction

The classification is heuristic and based on observable signals at ingest 
time. It's a useful rough sort, not a precise judgment.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class HealthReport:
    """The result of classifying a document's extraction health."""
    status: str  # "healthy" | "marginal" | "broken"
    score: int  # 0-100, higher is healthier
    signals: dict  # raw measurements
    reasons: list[str]  # human-readable explanations of penalties

    def __str__(self) -> str:
        return f"{self.status} (score={self.score}): {', '.join(self.reasons) or 'no issues'}"


def classify_document_health(
    chunk_count: int,
    avg_chunk_length: float,
    page_count: Optional[int],
    section_detection_rate: float,
    format: str,
    reference_flagged_ratio: float = 0.0,
) -> HealthReport:
    """Classify a document's extraction health from observable signals.

    Args:
        chunk_count: total number of chunks produced
        avg_chunk_length: mean length of chunks in characters
        page_count: number of pages, or None if not extractable
        section_detection_rate: 0.0-1.0, fraction of chunks with a section header
        format: file format ("pdf", "epub", etc.)
        reference_flagged_ratio: 0.0-1.0, fraction of chunks marked as references

    Returns:
        HealthReport with status, score, signals, and reasons.
    """
    score = 100
    reasons = []

    # Catastrophic failure: 0 or 1 chunks
    if chunk_count <= 1:
        score -= 100
        reasons.append(f"only {chunk_count} chunk(s) produced")
        # Don't bother with other signals; this is decisively broken
        return _finalize(score, chunk_count, avg_chunk_length, page_count,
                         section_detection_rate, format, reference_flagged_ratio, reasons)

    # Very few chunks for any document
    if chunk_count <= 3:
        score -= 50
        reasons.append(f"only {chunk_count} chunks total")

    # Chunks-per-page check (only if we know page count)
    if page_count and page_count > 0:
        chunks_per_page = chunk_count / page_count
        if chunks_per_page < 2 or chunks_per_page > 15:
            score -= 30
            reasons.append(f"unusual chunks-per-page ratio: {chunks_per_page:.1f}")

    # Average chunk length
    if avg_chunk_length < 100:
        score -= 40
        reasons.append(f"chunks suspiciously short (avg {avg_chunk_length:.0f} chars)")
    elif avg_chunk_length < 300:
        score -= 25
        reasons.append(f"chunks shorter than expected (avg {avg_chunk_length:.0f} chars)")

    # PDF-specific: page markers should be present
    if format == "pdf" and (page_count is None or page_count == 0):
        score -= 25
        reasons.append("no pages detected for PDF")

    # Section detection rate
    if section_detection_rate < 0.05:
        score -= 15
        reasons.append(f"very few sections detected ({section_detection_rate:.0%})")

    # Reference section dominance
    if reference_flagged_ratio > 0.4:
        score -= 30
        reasons.append(f"references make up {reference_flagged_ratio:.0%} of chunks")

    return _finalize(score, chunk_count, avg_chunk_length, page_count,
                     section_detection_rate, format, reference_flagged_ratio, reasons)


def _finalize(score, chunk_count, avg_chunk_length, page_count,
              section_detection_rate, format, reference_flagged_ratio, reasons):
    """Build the final report."""
    score = max(0, score)  # clamp to 0
    if score >= 75:
        status = "healthy"
    elif score >= 40:
        status = "marginal"
    else:
        status = "broken"

    return HealthReport(
        status=status,
        score=score,
        signals={
            "chunk_count": chunk_count,
            "avg_chunk_length": round(avg_chunk_length, 1),
            "page_count": page_count,
            "section_detection_rate": round(section_detection_rate, 3),
            "format": format,
            "reference_flagged_ratio": round(reference_flagged_ratio, 3),
        },
        reasons=reasons,
    )