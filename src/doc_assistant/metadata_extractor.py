"""Document-level metadata extraction (title / authors / year / DOI).

Reads the first ~3k characters of a doc's extracted markdown and tries to pull
out the four fields that internal citation matching uses.

Heuristics tuned for academic papers (the corpus is neuroscience PDFs).
Books / lectures / slide decks may yield partial or no metadata — the matcher
tolerates NULLs gracefully.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HEAD_CHARS = 3000

_DOI = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
_DOI_URL = re.compile(
    r"(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
    re.IGNORECASE,
)

_YEAR_PUBLISHED = re.compile(
    r"(?:published|received|accepted|copyright|\(c\))[^\n]{0,40}(19\d{2}|20\d{2})",
    re.IGNORECASE,
)
_YEAR_PARENS = re.compile(r"\((19\d{2}|20\d{2})\)")
_YEAR_LOOSE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# arxiv identifier "1707.01836" -> year 2017
_ARXIV_ID = re.compile(r"\b(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?\b")

# Journal-header patterns ("J. Physiol. (1952) 117, 500-544")
_JOURNAL_HEADER = re.compile(
    r"^[A-Z][A-Za-z .&]+\s*\(?\d{4}\)?[,;:\s]*\d+",
)

_SKIP_HEADINGS = {
    "research article",
    "review",
    "reviews",
    "report",
    "letter",
    "letters",
    "perspective",
    "commentary",
    "abstract",
    "introduction",
    "summary",
    "editor's evaluation",
    "competing interests",
    "competing interest",
    "main",
    "results",
    "methods",
    "tools and resources",
}

_SKIP_LINE_PATTERNS = [
    re.compile(r"^\s*<!--"),
    re.compile(r"^\s*\d+\s+of\s+\d+\s*$"),
    re.compile(r"^\s*\*\*=*=>"),
    re.compile(r"^\s*$"),
    re.compile(r"^\s*-----"),
]

_AFFILIATION_KEYWORDS = re.compile(
    r"\b(university|institute|department|laboratory|college|school|hospital|abstract|figure)\b",
    re.IGNORECASE,
)

# Discourse / section leads that a permissive author scan can mistake for a name list
# (e.g. "However, ideas from…", "Additional Key Words and Phrases:"). An author line is a
# list of proper names, never a sentence — so a candidate opening with one of these is not it.
_NON_AUTHOR_LEAD = re.compile(
    r"^(however|moreover|therefore|furthermore|additional(?:ly)?|although|whereas|here\b"
    r"|we\b|our\b|in this|this (?:paper|work|article|study)|index terms|key ?words?"
    r"|abstract|introduction|copyright)",
    re.IGNORECASE,
)

# Publisher boilerplate that a heading scan can mistake for a title, e.g. Springer's
# "The Author(s), under exclusive licence to Springer Nature…". Never a real title.
_COPYRIGHT_HEADING = re.compile(
    r"(the author\(s\)|under (?:exclusive )?licen[cs]e|all rights reserved"
    r"|©|\(c\)\s|copyright|springer nature|creative commons)",
    re.IGNORECASE,
)


@dataclass
class DocMetadata:
    """Best-effort metadata extracted from a document's header region."""

    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None

    @property
    def confidence(self) -> float:
        score = 0.0
        if self.title and len(self.title) >= 10:
            score += 0.35
        if self.authors:
            score += 0.25
        if self.year is not None:
            score += 0.15
        if self.doi is not None:
            score += 0.25
        return min(score, 1.0)


def _clean_markdown(text: str) -> str:
    """Strip markdown markers and affiliation brackets, collapse whitespace."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_+", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.replace("\\", "")  # markdown escape / hard-break artifacts (e.g. "WIESEL\")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _is_skippable_heading(text: str) -> bool:
    normalized = _clean_markdown(text).lower().strip(": ")
    if not normalized:
        return True
    if normalized in _SKIP_HEADINGS:
        return True
    if _COPYRIGHT_HEADING.search(normalized):
        return True
    if re.match(
        r"^\d+(\.\d+)*\.?\s+(introduction|methods?|results?|discussion|abstract)\b",
        normalized,
    ):
        return True
    return bool(_JOURNAL_HEADER.match(normalized))


def _extract_title(head: str) -> str | None:
    """Pick the first non-skippable markdown heading as title.

    Prefers H1 over H2/H3 — some papers have a journal-citation H2
    ("J. Physiol. (1952)...") before the actual H1 title. Title is the
    most semantically prominent heading available.
    """
    lines = head.split("\n")

    by_level: dict[int, str] = {}
    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if m is None:
            continue
        level = len(m.group(1))
        text = _clean_markdown(m.group(2))
        if _is_skippable_heading(text):
            continue
        if len(text) < 10:
            continue
        by_level.setdefault(level, text)

    for level in (1, 2, 3):
        if level in by_level:
            return by_level[level]

    for line in lines:
        if any(p.search(line) for p in _SKIP_LINE_PATTERNS):
            continue
        m = re.match(r"^\s*\*\*([^*]{20,})\*\*\s*$", line)
        if m:
            return _clean_markdown(m.group(1))
    return None


def _extract_doi(head: str) -> str | None:
    m = _DOI_URL.search(head)
    if m is not None:
        return m.group(1).rstrip(".,;)]")
    m = _DOI.search(head)
    if m is not None:
        return m.group(0).rstrip(".,;)]")
    return None


def _extract_year(head: str) -> int | None:
    """Prefer years near publication keywords; arxiv ID; then any year."""
    m = _YEAR_PUBLISHED.search(head)
    if m is not None:
        try:
            return int(m.group(1))
        except (ValueError, IndexError):
            pass

    m_ax = _ARXIV_ID.search(head)
    if m_ax is not None:
        yy = int(m_ax.group(1))
        # arxiv started in 1991; assume 91-99 = 19xx, else 20xx
        return 1900 + yy if yy >= 91 else 2000 + yy

    m = _YEAR_PARENS.search(head)
    if m is None:
        m = _YEAR_LOOSE.search(head)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None


def _looks_like_author_line(line: str) -> tuple[bool, str]:
    """Permissive: is this a list of authors? Returns (verdict, cleaned)."""
    cleaned = re.sub(r"^#{1,6}\s+", "", line)
    cleaned = _clean_markdown(cleaned)

    has_by_prefix = bool(re.match(r"^\s*(?:by|By|BY)\s+", cleaned))
    if has_by_prefix:
        cleaned = re.sub(r"^\s*(?:by|By|BY)\s+", "", cleaned)

    if not cleaned or len(cleaned) > 400:
        return False, cleaned
    if "@" in cleaned:
        return False, cleaned
    if _AFFILIATION_KEYWORDS.search(cleaned):
        return False, cleaned
    if _NON_AUTHOR_LEAD.match(cleaned):
        return False, cleaned
    if not re.match(r"^[A-Z]", cleaned):
        return False, cleaned

    has_commas = cleaned.count(",") >= 1
    has_and = bool(re.search(r"\b(and|&|AND)\b", cleaned))
    has_initials = bool(re.search(r"\b[A-Z]\.\s*[A-Z]", cleaned))
    if not (has_commas or has_and or has_by_prefix or has_initials):
        return False, cleaned
    return True, cleaned


def _extract_authors(head: str, title: str | None) -> str | None:
    """Find an author-like line near the title."""
    lines = head.split("\n")
    title_idx = -1
    if title is not None:
        marker = title[:30]
        for i, line in enumerate(lines):
            if marker in line:
                title_idx = i
                break

    start = title_idx + 1 if title_idx >= 0 else 0
    for line in lines[start : start + 15]:
        if any(p.search(line) for p in _SKIP_LINE_PATTERNS):
            continue
        ok, cleaned = _looks_like_author_line(line)
        if ok:
            return cleaned
    return None


def _arxiv_year_from_filename(filename: str | None) -> int | None:
    """Map filenames like '1707.01836v1.pdf' / '2403.01590v1.md' to a year.

    arXiv IDs use YYMM.NNNNN (post-2007). The two leading digits are the year.
    """
    if not filename:
        return None
    m = _ARXIV_ID.search(filename)
    if m is None:
        return None
    yy = int(m.group(1))
    return 1900 + yy if yy >= 91 else 2000 + yy


def extract_metadata(markdown: str, *, filename: str | None = None) -> DocMetadata:
    """Pull title / authors / year / DOI from a doc's extracted markdown.

    `filename` is optional; used as a hint for arXiv-style year detection
    when the head text doesn't contain a clear publication year.
    """
    head = markdown[:_HEAD_CHARS]
    title = _extract_title(head)
    doi = _extract_doi(head)
    year_from_head = _extract_year(head)
    arxiv_year = _arxiv_year_from_filename(filename)
    # Prefer arXiv year over loose head-year because head can pick up
    # spurious in-text citation years.
    year = arxiv_year if arxiv_year is not None else year_from_head
    authors = _extract_authors(head, title)
    return DocMetadata(title=title, authors=authors, year=year, doi=doi)
