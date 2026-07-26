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

# Two keyword tiers, because they are not equally authoritative (KI-26): *published* and
# *copyright* date the publication, while *received* and *accepted* date the submission — often the
# year before. `41304_2021_Article_335.pdf` says "Accepted: 14 December 2020 / Published online:
# 10 June 2021" and was being recorded as 2020.
#
# The gap may cross a line break: PMC author manuscripts wrap as "Published in final edited form
# as:\nCurr Opin Neurobiol. 2012 February", and a `[^\n]` gap stopped at the newline — so the
# authoritative tier missed and the year fell through to a weaker one (2013, the PMC *availability*
# date). 60 chars with DOTALL keeps the window tight enough that only the keyword's own year is in
# reach.
# `©` is included because it is *the* canonical copyright mark and the word "copyright" often does
# not accompany it — IEEE prints "0018-9294/04$20.00 © 2004 IEEE", which is why
# `chazal_2004-ecg.pdf` was falling through to its 2003 submission date (KI-26).
_YEAR_PUBLISHED = re.compile(
    r"(?:published|copyright|©|\(c\))[\s\S]{0,60}?(19\d{2}|20\d{2})",
    re.IGNORECASE,
)
_YEAR_SUBMITTED = re.compile(
    r"(?:received|accepted|submitted)[\s\S]{0,60}?(19\d{2}|20\d{2})",
    re.IGNORECASE,
)
_YEAR_PARENS = re.compile(r"\((19\d{2}|20\d{2})\)")
_YEAR_LOOSE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# A month adjacent to a year — the shape a journal running header uses for the issue date
# ("IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING, VOL. 51, NO. 7, JULY 2004"). Without it,
# `chazal_2004-ecg.pdf` fell through to its *submission* year, 2003 (KI-26). Both orders occur
# ("JULY 2004", "2009 September"). Front-matter only, and below the tiers where the document
# states its date explicitly.
_MONTH = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?"
    r"|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_YEAR_MONTH = re.compile(
    rf"(?:{_MONTH}\s+((?:19|20)\d{{2}})|((?:19|20)\d{{2}})\s+{_MONTH})",
    re.IGNORECASE,
)

# arxiv identifier "1707.01836" -> year 2017
_ARXIV_ID = re.compile(r"\b(\d{2})(\d{2})\.\d{4,5}(?:v\d+)?\b")

# Journal-header patterns ("J. Physiol. (1952) 117, 500-544").
# IGNORECASE is load-bearing, not cosmetic: `_is_skippable_heading` lowercases before testing, so
# an anchored `^[A-Z]` never matched and this rule was **dead code**. It went unnoticed because the
# old title picker preferred H1 over H2 and the journal line is usually the H2 — remove that
# preference (KI-26) and the dead rule surfaces immediately as a regression.
_JOURNAL_HEADER = re.compile(
    r"^[A-Z][A-Za-z .&]+\s*\(?\d{4}\)?[,;:\s]*\d+",
    re.IGNORECASE,
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
    # Journal front matter and page furniture (KI-26). These are *headings* in the extracted
    # markdown — a Frontiers PDF leads with "## OPEN ACCESS" above the editorial block — so the
    # length filter does not catch them and they were being stored as the document's title on
    # 9 of 97 documents. Add a term here only if it can never be a real title.
    "open access",
    "edited by",
    "reviewed by",
    "correspondence",
    "citation",
    "copyright",
    "disclaimer",
    "graphical abstract",
    "highlights",
    "keywords",
    "key words",
    "funding",
    "acknowledgments",
    "acknowledgements",
    "data availability statement",
    "conflict of interest",
    "publisher's note",
    "special issue article",
    "original research",
    "original article",
    "review article",
    "research paper",
    "brief communication",
    "short communication",
    "letter to the editor",
    "case report",
    "editorial",
    "erratum",
    "correction",
    "systematic review",
    "specialty section",
    "author contributions",
    "supplementary material",
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


#: Bullet / dash glyphs that can open an extracted heading. A title never legitimately starts with
#: one, so a leading run of them is always an extraction artifact. Written as codepoints because
#: ruff's RUF001 rejects ambiguous dash literals in source, and because a reader cannot tell a
#: HYPHEN from a HYPHEN-MINUS by eye: U+002D hyphen-minus, U+2010..U+2015 (hyphen, non-breaking
#: hyphen, figure dash, en dash, em dash, horizontal bar), and the bullet glyphs U+2022 bullet,
#: U+00B7 middle dot, U+2023 triangular bullet, U+25AA small square, U+25CF black circle.
_LEADING_BULLET = re.compile(r"^[-\u2010-\u2015\u2022\u00b7\u2023\u25aa\u25cf]+\s*")


def _clean_markdown(text: str) -> str:
    """Strip markdown markers and affiliation brackets, collapse whitespace."""
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"_+", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = text.replace("\\", "")  # markdown escape / hard-break artifacts (e.g. "WIESEL\")
    text = re.sub(r"\s+", " ", text).strip()
    # A leading bullet or dash is never part of a title — it is a list glyph, or a hyphen the PDF
    # wrapped into the heading. `reranking_bert_nogueira_2019.pdf` extracts as
    # `## - PASSAGE RE RANKING WITH BERT` (the hyphen of "RE-RANKING" landing at the front), and
    # the stored title rode into the library grid, the taxonomy prompt, and every layer keyed on
    # the title. Strip it here, where every candidate path already passes through.
    text = _LEADING_BULLET.sub("", text).strip()
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


def _citation_block_title(head: str) -> str | None:
    """Recover a title from a publisher CITATION block (Frontiers and friends).

    Frontiers front matter puts the access banner and editorial block *above* the title, but it
    also prints a full self-citation:

        CITATION
        Pedrão LFAT, … and Falquetto B (2024) Parkinson's disease models and death
        signaling: what do we know until now? _Front. Neuroanat._ 18:1419108. doi: …

    The title is what sits between the year and the italicised journal name, so it can be read
    back exactly rather than guessed. Deliberately narrow: it needs the ``(YYYY)`` … ``_`` shape,
    so a document without it yields ``None`` rather than a wrong answer (KI-26).
    """
    # The boundary is the italicised journal name, not a particular punctuation mark: titles end in
    # "?", ".", or nothing at all, so requiring a specific terminator only fits one publisher's
    # sample. Match up to the `_Journal_` marker and trim whatever trailing punctuation is there.
    m = re.search(
        r"\((?:19|20)\d{2}\)\s*(?P<title>[^\n]{10,300}?)\s*(?=_[A-Z])",
        head,
    )
    if m is None:
        return None
    title = _clean_markdown(m.group("title")).strip().rstrip(".,;:")
    return title or None


def _title_candidates(head: str) -> list[str]:
    """Ordered title candidates: headings and standalone bold lines, in document order.

    **Position beats markup level** (KI-26). The old rule preferred H1 → H2 → H3 and only fell
    back to bold lines if no heading survived, which loses whenever a publisher marks the title
    bold and the *author list* as a heading — the real shape of `2606.31856v1.pdf`
    (``**Low-dimensional topology…**`` then ``## **Junyu Ren** **Lek-Heng Lim**``) and
    `41304_2021_Article_335.pdf`. That mis-picked the authors as the title on both, and no
    capitalisation heuristic can separate "Junyu Ren Lek-Heng Lim" from "Attention Is All You
    Need" — but *order* separates them cleanly. The level-preference this replaces existed to skip
    a journal-citation H2 before the real H1, which :func:`_is_skippable_heading` already rejects.
    """
    candidates: list[str] = []
    for line in head.split("\n"):
        if any(p.search(line) for p in _SKIP_LINE_PATTERNS):
            continue
        heading = re.match(r"^(#{1,3})\s+(.+?)\s*$", line)
        if heading is not None:
            text = _clean_markdown(heading.group(2))
        else:
            bold = re.match(r"^\s*\*\*([^*]{10,})\*\*\s*$", line)
            if bold is None:
                continue
            text = _clean_markdown(bold.group(1))
        if len(text) < 10 or _is_skippable_heading(text):
            continue
        candidates.append(text)
    return candidates


def _extract_title(head: str) -> str | None:
    """The first non-skippable title candidate, else a publisher CITATION block."""
    candidates = _title_candidates(head)
    if candidates:
        return candidates[0]
    return _citation_block_title(head)


def _extract_doi(head: str) -> str | None:
    m = _DOI_URL.search(head)
    if m is not None:
        return m.group(1).rstrip(".,;)]")
    m = _DOI.search(head)
    if m is not None:
        return m.group(0).rstrip(".,;)]")
    return None


def _front_matter(head: str) -> str:
    """The head up to the abstract/introduction — where publication years live.

    Everything after it is prose, and prose is full of *other papers'* years. Scanning the whole
    head for a loose year is how `dpr_karpukhin_2020.pdf` came to be recorded as **2012**: no
    publication keyword in its header, so the loose scan fell through to a citation year in the
    abstract (KI-26). Cutting at the abstract makes the loose tier structurally unable to read a
    reference as a publication date.
    """
    cut = re.search(r"^\s*#{0,3}\s*\**\s*(abstract|introduction|summary)\b", head, re.I | re.M)
    return head[: cut.start()] if cut else head


def _doi_year(doi: str | None) -> int | None:
    """A 4-digit year embedded in a DOI suffix (Frontiers: ``10.3389/fnana.2024.1419108``).

    Only accepts a year delimited by dots/slashes, so a 4-digit article number cannot pose as one.
    """
    if not doi:
        return None
    m = re.search(r"[./]((?:19|20)\d{2})[./]", doi)
    return int(m.group(1)) if m else None


def _has_authoritative_year(head: str, doi: str | None) -> bool:
    """Did the document *state* its year (publication keyword or DOI), rather than us infer it?

    The dividing line for whether a filename year may override the head (KI-26).
    """
    return (
        _YEAR_PUBLISHED.search(head) is not None
        or _YEAR_SUBMITTED.search(head) is not None
        or _doi_year(doi) is not None
    )


def _extract_year(head: str, *, doi: str | None = None) -> int | None:
    """Publication year, most-trustworthy signal first.

    Order (KI-26): an explicit publication keyword → an arXiv id → a year embedded in the DOI →
    a parenthesised or loose year **restricted to the front matter**. Each tier is a *statement
    about this document*; the loose tier is last and bounded because it is the only one that can
    confuse another paper's year for this one's.
    """
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

    from_doi = _doi_year(doi)
    if from_doi is not None:
        return from_doi

    front = _front_matter(head)
    m = _YEAR_MONTH.search(front)
    if m is not None:
        year = m.group(1) or m.group(2)
        if year:
            return int(year)

    m = _YEAR_SUBMITTED.search(head)
    if m is not None:
        try:
            return int(m.group(1))
        except (ValueError, IndexError):
            pass

    m = _YEAR_PARENS.search(front) or _YEAR_LOOSE.search(front)
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


def _year_from_filename(filename: str | None) -> int | None:
    """A delimited 4-digit year in the filename (``dpr_karpukhin_2020.pdf`` → 2020).

    A weaker claim than a publication keyword — it is whatever the *downloader* named the file —
    but far stronger than a loose year scraped out of prose, and this corpus names files
    ``author_year`` throughout. Requires a `_`/`-`/`.`/space delimiter so a 4-digit id inside a
    longer token (``41304_2021_Article_335`` is fine; ``PIIS0002929724003008`` is not) cannot pose
    as a year.
    """
    if not filename:
        return None
    if _ARXIV_ID.search(filename) is not None:
        return None  # "1904.01169v3" is an arXiv id, not the year 1904 — that tier owns it
    stem = filename.rsplit(".", 1)[0]
    years = [int(y) for y in re.findall(r"(?:^|[ _\-.])((?:19|20)\d{2})(?:$|[ _\-.])", stem)]
    return max(years) if years else None


def extract_metadata(markdown: str, *, filename: str | None = None) -> DocMetadata:
    """Pull title / authors / year / DOI from a doc's extracted markdown.

    `filename` is optional; used as a hint for arXiv-style year detection
    when the head text doesn't contain a clear publication year.
    """
    head = markdown[:_HEAD_CHARS]
    title = _extract_title(head)
    doi = _extract_doi(head)
    year_from_head = _extract_year(head, doi=doi)
    arxiv_year = _arxiv_year_from_filename(filename)
    # Prefer arXiv year over loose head-year because head can pick up
    # spurious in-text citation years.
    year = arxiv_year if arxiv_year is not None else year_from_head
    # A filename year outranks a *loose* head year for the same reason (KI-26): the head tier that
    # can be wrong is the unkeyworded one, and the filename is at least a claim about this file.
    # It never overrides the keyword/arXiv/DOI tiers — `_extract_year` returns those, and they are
    # statements the document makes about itself.
    if year is None or (arxiv_year is None and not _has_authoritative_year(head, doi)):
        from_filename = _year_from_filename(filename)
        if from_filename is not None:
            year = from_filename
    authors = _extract_authors(head, title)
    return DocMetadata(title=title, authors=authors, year=year, doi=doi)
