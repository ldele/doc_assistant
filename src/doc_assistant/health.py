"""Document health classification based on extraction signals.

A document is classified as:
- "healthy": extraction looks good, document is fully usable
- "marginal": something's off but probably usable
- "broken": extraction failed, probably needs re-extraction

The classification is heuristic and based on observable signals at ingest
time. It's a useful rough sort, not a precise judgment.

**"Broken" means the extractor failed, not that the document is small** (KI-53). Every rule that
can return ``broken`` is therefore a statement about text missing *relative to its container* — a
paged document that yielded far less than its pages hold, or a single fragment where a document
should be. Size alone is not evidence: a 2 KB web article that extracts to one full chunk is
complete, and calling it broken tells a user to delete a document that worked. Where there is no
container to compare against (an HTML page, an EPUB chapter) the honest answer is that we cannot
tell, so the classifier does not guess — it withholds the penalty rather than inventing one.
"""

from dataclasses import dataclass
from typing import Any

#: Below this a lone chunk is a fragment (a header, a caption, an error page) rather than a
#: document. Sized against the baseline chunk (``config.BASELINE_CHUNK_SIZE``, 1,000 chars): a
#: fifth of one chunk is not prose that merely stopped early.
SCRAP_CHARS = 200

#: A paged document averaging less than this per page did not give up its text. Half a baseline
#: chunk per page — a real page of prose holds two to four times a chunk, and even a title or
#: figure page beats this once averaged across a document. Deliberately far below "normal" so it
#: fires on failure, not on sparseness.
SPARSE_PAGE_CHARS = 500

#: Formats whose extraction is a read, not a conversion (``extractors.extract_text``). There is no
#: container for text to be lost from, so a short file is short — never broken.
VERBATIM_FORMATS = frozenset({"txt", "md"})


@dataclass
class HealthReport:
    """The result of classifying a document's extraction health."""

    status: str  # "healthy" | "marginal" | "broken"
    score: int  # 0-100, higher is healthier
    signals: dict[str, Any]  # raw measurements
    reasons: list[str]  # human-readable explanations of penalties

    def __str__(self) -> str:
        return f"{self.status} (score={self.score}): {', '.join(self.reasons) or 'no issues'}"


def classify_document_health(
    chunk_count: int,
    avg_chunk_length: float,
    page_count: int | None,
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
    reasons: list[str] = []
    pages = page_count if page_count and page_count > 0 else None
    extracted_chars = chunk_count * avg_chunk_length
    verbatim = format.lower() in VERBATIM_FORMATS

    # Nothing came out. The one size signal that needs no container to interpret.
    if chunk_count == 0:
        score -= 100
        reasons.append("no text extracted")
        return _finalize(
            score,
            chunk_count,
            avg_chunk_length,
            page_count,
            section_detection_rate,
            format,
            reference_flagged_ratio,
            reasons,
        )

    # A single chunk (KI-53). This used to be an unconditional `broken`, which is what filed a
    # clean 2 KB HTML article as a failure. One chunk is only evidence of collapse when something
    # says there was more text to get:
    #   * the document is paged and holds more than one page — two pages of prose cannot fit in
    #     one baseline chunk, so the rest of the text is genuinely missing; or
    #   * the chunk is a scrap, too small to be a document at all.
    # A verbatim format is exempt from the second: reading a short .md file back is not a failure,
    # it is the file. With neither signal present the document is simply short, and short is not a
    # penalty — it is the honest reading of a complete extraction.
    if chunk_count == 1:
        collapsed = pages is not None and pages > 1
        scrap = avg_chunk_length < SCRAP_CHARS and not verbatim
        if collapsed or scrap:
            score -= 100
            reasons.append(
                f"{pages} pages produced a single chunk"
                if collapsed
                else f"a single {avg_chunk_length:.0f}-character fragment"
            )
            return _finalize(
                score,
                chunk_count,
                avg_chunk_length,
                page_count,
                section_detection_rate,
                format,
                reference_flagged_ratio,
                reasons,
            )

    # Text yield per page — the whole paged-format signal, graded, in the unit that carries the
    # meaning. It replaces BOTH of the count-based rules it supersedes: a `chunks_per_page < 2`
    # floor (which fired at ~2,000 chars/page, i.e. on an ordinary sparse paper) and a flat
    # `chunk_count <= 3` penalty. Those measured the same thing worse: with a 1,000-char baseline
    # chunk, "fewer chunks than pages" just *is* "under ~1,000 characters per page", but stated in
    # a unit that made a real 3-page note with two full chunks look damaged.
    #
    # Two tiers, because the failures differ in kind: below a scrap per page the text layer did not
    # work, which is broken; below half a chunk per page the document is thin, which is worth
    # saying but is not a failure. The upper bound stays — many tiny chunks per page is
    # fragmentation, a different failure.
    if pages is not None:
        chars_per_page = extracted_chars / pages
        if chars_per_page < SCRAP_CHARS:
            score -= 70
            reasons.append(
                f"only {chars_per_page:.0f} characters per page — the text layer failed"
            )
        elif chars_per_page < SPARSE_PAGE_CHARS:
            score -= 30
            reasons.append(f"only {chars_per_page:.0f} characters per page")
        if chunk_count / pages > 15:
            score -= 30
            reasons.append(f"unusual chunks-per-page ratio: {chunk_count / pages:.1f}")

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

    return _finalize(
        score,
        chunk_count,
        avg_chunk_length,
        page_count,
        section_detection_rate,
        format,
        reference_flagged_ratio,
        reasons,
    )


def _finalize(
    score: int,
    chunk_count: int,
    avg_chunk_length: float,
    page_count: int | None,
    section_detection_rate: float,
    format: str,
    reference_flagged_ratio: float,
    reasons: list[str],
) -> HealthReport:
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
