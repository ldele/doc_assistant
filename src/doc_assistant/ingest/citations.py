"""Citation extraction and internal library matching (Phase 4, tier 1).

Pure stdlib regex extraction. UI-agnostic. The pipeline is:

    raw markdown
        -> find References section
        -> split into refs
        -> parse each ref
        -> match to library Document (DOI / author+year / fuzzy title)
        -> Citation rows persisted by the caller

Design notes
------------
* No new dependencies. SequenceMatcher (stdlib) is used for fuzzy title match.
* Extractor returns dataclasses (`ParsedCitation`). Persistence is the caller's
  job — keeps this module testable without a DB.
* `extraction_method` is recorded on each ParsedCitation so we can later
  distinguish tier-1 (regex) from tier-2 (LLM) and from manual entries.
* Confidence is a float in [0.0, 1.0] computed from which fields were parsed.
  This lets the UI sort/triage low-confidence refs.

This module is the data layer for Phase 4 — UI surfaces are built on top in
`commands.py` and consume these dataclasses unchanged.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

import structlog
from sqlalchemy import select

from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope

log = structlog.get_logger(__name__)


# ============================================================
# Patterns
# ============================================================

_SECTION_HEADING = re.compile(
    r"^#{1,6}\s*\*{0,2}\s*"
    r"(references?(?:\s+cited)?|bibliography|works\s+cited|literature\s+cited)"
    r"\s*\*{0,2}\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_TERMINATING_HEADING = re.compile(
    r"^#{1,6}\s*\*{0,2}\s*"
    r"(acknowledg(e)?ments?|appendix|supplementary|supplemental|"
    r"author\s+contributions?|competing\s+interests?|funding|"
    r"data\s+availability|further\s+information|notes?)"
    r"\s*\*{0,2}\s*\b",
    re.IGNORECASE | re.MULTILINE,
)

_REF_START = re.compile(
    r"^(?:[-*]\s+|\d{1,3}\.\s+|\[\d{1,3}\]\s+)",
    re.MULTILINE,
)

_DOI = re.compile(
    r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
    re.IGNORECASE,
)

_YEAR_IN_PARENS = re.compile(r"\((19\d{2}|20\d{2})[a-z]?\)")
_YEAR_LOOSE = re.compile(r"\b(19\d{2}|20\d{2})\b")

_INLINE_REF_BREAK = re.compile(r"(?<=\s)(\d{1,3}\.\s+)(?=[A-ZÀ-Ÿ])")


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class ParsedCitation:
    """A single citation extracted from a source document's references section.

    Confidence is set by `_score_confidence` based on which fields parsed.
    `extraction_method` distinguishes tier-1 ("regex"), tier-2 ("llm"), etc.
    """

    raw_text: str
    doi: str | None = None
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    extraction_method: str = "regex"
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Result of running the tier-1 extractor on one document."""

    doc_id: str
    citations: list[ParsedCitation] = field(default_factory=list)
    references_section_found: bool = False
    section_char_offset: int | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.citations)

    @property
    def needs_tier2(self) -> bool:
        """Heuristic: refs section detected but parsed <5 refs -> messy format."""
        return self.references_section_found and self.count < 5


# ============================================================
# Section detection
# ============================================================


def detect_references_section(markdown: str) -> tuple[int, int] | None:
    """Return (start, end) character offsets of the References section body.

    Returns None if no References-like heading is found. End offset is exclusive
    and points to the next terminating heading or end-of-document.
    """
    match = _SECTION_HEADING.search(markdown)
    if match is None:
        return None

    body_start = match.end()
    terminator = _TERMINATING_HEADING.search(markdown, pos=body_start)
    next_heading = re.search(r"^#{1,6}\s+\S", markdown[body_start:], re.MULTILINE)

    candidate_ends: list[int] = []
    if terminator is not None:
        candidate_ends.append(terminator.start())
    if next_heading is not None:
        candidate_ends.append(body_start + next_heading.start())

    body_end = min(candidate_ends) if candidate_ends else len(markdown)
    return (body_start, body_end)


# ============================================================
# Splitting refs section into individual entries
# ============================================================


def _split_refs(refs_block: str) -> list[str]:
    """Split a references-section block into individual reference strings."""
    starts = [m.start() for m in _REF_START.finditer(refs_block)]
    if len(starts) >= 3:
        starts.append(len(refs_block))
        chunks = [refs_block[starts[i] : starts[i + 1]] for i in range(len(starts) - 1)]
        return [c.strip() for c in chunks if c.strip()]

    pieces = _INLINE_REF_BREAK.split(refs_block)
    if len(pieces) >= 5:
        out: list[str] = []
        head = pieces[0].strip()
        if head:
            out.append(head)
        for i in range(1, len(pieces) - 1, 2):
            marker = pieces[i]
            body = pieces[i + 1] if i + 1 < len(pieces) else ""
            combined = (marker + body).strip()
            if combined:
                out.append(combined)
        if len(out) >= 3:
            return out

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", refs_block) if p.strip()]
    return paragraphs


# ============================================================
# Parsing one reference string
# ============================================================


def _strip_marker(s: str) -> str:
    return re.sub(r"^(?:[-*]\s+|\d{1,3}\.\s+|\[\d{1,3}\]\s+)", "", s).strip()


def _extract_doi(s: str) -> str | None:
    m = _DOI.search(s)
    if m is None:
        return None
    return m.group(0).rstrip(".,;)]")


def _extract_year(s: str) -> int | None:
    m = _YEAR_IN_PARENS.search(s)
    if m is None:
        m = _YEAR_LOOSE.search(s)
    if m is None:
        return None
    try:
        return int(m.group(1))
    except (ValueError, IndexError):
        return None


def _extract_authors_and_title(text: str, year: int | None) -> tuple[str | None, str | None]:
    """Best-effort authors/title split.

    Two reference formats are common in academic papers:

      A. "Authors (year) Title. Journal, vol, pp."
         Title comes AFTER the year (year is early in string).

      B. "Authors. Title. _Journal_, vol, pp, year." or
         "Authors. Title. _Journal_ pp (year) <citing_pages>"
         Title comes BETWEEN authors and journal italics (year is near end).

    Position alone is the signal — parens don't determine where the title lives.
    """
    if year is None:
        sents = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
        authors = sents[0].strip().rstrip(",.") if sents else None
        return authors, None

    parens_match = _YEAR_IN_PARENS.search(text)
    loose_match = _YEAR_LOOSE.search(text)
    m = parens_match if parens_match is not None else loose_match
    if m is None:
        return None, None

    year_at_end = m.start() > 0.6 * len(text)

    before = text[: m.start()].strip().rstrip(",.")
    after = text[m.end() :].strip().lstrip(".,)( ")

    title: str | None
    if year_at_end:
        # Format B: split `before` into authors + title.
        em = re.search(r"(_[^_]+_|\*[^*]+\*)", before)
        if em is not None:
            authors_title = before[: em.start()].strip().rstrip(",.")
            parts = re.split(r"(?<=[a-z\)])\.\s+", authors_title, maxsplit=1)
            if len(parts) == 2:
                authors = parts[0].strip().rstrip(",.")
                title = parts[1].strip().rstrip(",.")
            else:
                parts2 = authors_title.split(". ", 1)
                authors = parts2[0].strip().rstrip(",.")
                title = parts2[1].strip().rstrip(",.") if len(parts2) == 2 else None
        else:
            parts = before.split(". ", 1)
            authors = parts[0].strip().rstrip(",.")
            title = parts[1].strip().rstrip(",.") if len(parts) == 2 else None
    else:
        authors = before if before else None
        em = re.search(r"(_[^_]+_|\*[^*]+\*)", after)
        if em is not None:
            title = after[: em.start()].strip().rstrip(",.")
        else:
            first = re.split(r"(?<=[.!?])\s+", after, maxsplit=1)
            title = first[0].strip().rstrip(",.") if first else None

    if title and len(title) < 5:
        title = None
    return authors, title


def _score_confidence(c: ParsedCitation) -> float:
    """Confidence in [0, 1] from the fields we parsed."""
    score = 0.0
    if c.year is not None:
        score += 0.30
    if c.authors:
        score += 0.25
    if c.title and len(c.title) >= 10:
        score += 0.25
    if c.doi is not None:
        score += 0.20
    return min(score, 1.0)


def _parse_one(raw: str) -> ParsedCitation | None:
    """Parse a single reference string. Returns None if it's not ref-shaped."""
    body = _strip_marker(raw)
    body = re.sub(r"\s+", " ", body).strip()
    if len(body) < 20:
        return None

    doi = _extract_doi(body)
    year = _extract_year(body)
    authors, title = _extract_authors_and_title(body, year)

    cite = ParsedCitation(
        raw_text=body,
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        extraction_method="regex",
    )
    cite.confidence = _score_confidence(cite)
    if cite.year is None and cite.doi is None and len(body) < 60:
        return None
    return cite


# ============================================================
# Top-level extraction
# ============================================================


def extract_from_markdown(doc_id: str, markdown: str) -> ExtractionResult:
    """Run tier-1 extraction on already-extracted markdown for one document."""
    result = ExtractionResult(doc_id=doc_id)
    section = detect_references_section(markdown)
    if section is None:
        result.notes.append("no references heading detected")
        return result

    start, end = section
    result.references_section_found = True
    result.section_char_offset = start
    refs_block = markdown[start:end]

    chunks = _split_refs(refs_block)
    if not chunks:
        result.notes.append("references heading found but no entries parsed")
        return result

    for chunk in chunks:
        parsed = _parse_one(chunk)
        if parsed is not None:
            result.citations.append(parsed)
    return result


# ============================================================
# Internal matching: parsed citation -> library Document
# ============================================================


def _normalize_for_match(s: str) -> str:
    """Lowercase, strip diacritics, collapse non-alphanum to single spaces."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s).strip().lower()
    return s


def _first_author_surname(authors: str | None) -> str | None:
    """Best-effort first-author surname extraction.

    Handles:
      "Hodgkin, A. L."                   -> hodgkin
      "A. L. Hodgkin"                    -> hodgkin
      "A. L. HODGKIN AND A. F. HUXLEY"   -> hodgkin
      "Suarez LE, Yovel Y, ..."          -> suarez
      "Eric Jonas, Konrad Kording"       -> jonas
    Ambiguous "Firstname Surname Firstname Initial. Surname" formats
    (no separator) will resolve to the LAST surname seen — accepted limitation.
    """
    if not authors:
        return None
    text = authors.replace("\\", "")
    text = re.split(r"\s+(?:and|AND|&)\s+", text, maxsplit=1)[0]

    parts = text.split(",")
    if len(parts) > 1:
        after_comma = parts[1].strip()
        if re.match(r"^[A-Z]\.?\s*[A-Z]?\.?\s*[A-Z]?\.?$", after_comma) or len(after_comma) <= 6:
            text = ",".join(parts[:2])
        else:
            text = parts[0]
    text = text.strip().rstrip(",.")

    m = re.match(r"^([A-Z][a-zA-ZÀ-ÿ\-']+)\s*,\s*[A-Z]", text)
    if m:
        return _normalize_for_match(m.group(1)) or None

    tokens = text.split()
    cands = [t.rstrip(",.") for t in tokens]
    non_init = [
        t for t in cands if not re.match(r"^[A-Z]\.?[A-Z]?\.?[A-Z]?\.?$", t) and len(t) >= 3
    ]
    if non_init:
        return _normalize_for_match(non_init[-1]) or None
    return None


def _title_similarity(a: str, b: str) -> float:
    """Ratio in [0,1] using normalized strings."""
    na, nb = _normalize_for_match(a), _normalize_for_match(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


# Ratio at or above which two normalised titles are taken to name the same paper. Shared with
# `library.citations`, which re-checks a stored resolution before presenting it as a link —
# both sides must agree on what "the same paper" means.
FUZZY_TITLE_THRESHOLD = 0.80


#: Fraction of the SHORTER title's words that must also appear in the longer one before two
#: titles count as the same paper. **Derived from a measured gap, not tuned to a target:** over
#: the 40 links the corrected matcher proposes on this library, every false one scores exactly
#: 0.75 ("Bidirectional recurrent neural networks" vs "Relational recurrent neural networks";
#: "Recurrent neural network grammars" vs "…regularization") and every true one scores 0.88 or
#: above. The floor sits in the empty band between them.
#:
#: It exists because `SequenceMatcher` cannot see that one word carries the meaning: two long
#: titles differing only in their first word score 0.91, comfortably over
#: ``FUZZY_TITLE_THRESHOLD``. Comparing the *sets* of words is what notices a substitution.
MIN_TITLE_WORD_COVERAGE = 0.80

_TITLE_WORD = re.compile(r"[a-z0-9]+")


def _title_words(title: str | None) -> set[str]:
    return set(_TITLE_WORD.findall((title or "").lower()))


def _title_word_coverage(a: str | None, b: str | None) -> float:
    """How much of the shorter title's vocabulary the longer one contains, 0-1.

    Deliberately asymmetric. A reference is often a *prefix* of the real title, or carries an
    author-list tail the regex could not strip, so the longer string is expected to hold extra
    words. What must not happen is a word going *missing* from the shorter one — that is a
    substitution, and a substitution means a different paper.
    """
    wa, wb = _title_words(a), _title_words(b)
    if not wa or not wb:
        return 0.0
    shorter, longer = (wa, wb) if len(wa) <= len(wb) else (wb, wa)
    return len(shorter & longer) / len(shorter)


# A title fragment shorter than this can be contained by coincidence ("a survey" sits inside
# hundreds of titles). Structural, not corpus-tuned: it bounds what "one title contains the
# other" is allowed to mean, and no value of it can admit a title that is merely similar.
MIN_CONTAINED_TITLE_CHARS = 20


def resolution_is_credible(
    *,
    parsed_title: str | None,
    parsed_doi: str | None,
    library_title: str | None,
    library_doi: str | None,
) -> bool:
    """Does a stored ``target_document_id`` actually agree with the document it names?

    **Checked on BOTH sides since 2026-08-26.** It began as a read-side guard, because
    ``match_to_library`` resolved a reference at *extraction* time and one of its three rules —
    first-author surname + publication year — required no title agreement whatsoever. That rule
    now calls this function before accepting a match, so the wrong rows stop being written at
    all; the read-side call remains because the database still holds rows written before that
    (KI-45). On a corpus with many same-year papers by common
    surnames it fires constantly and wrongly: measured on this library 2026-08-10, **13 of the
    16 stored resolutions were false**, e.g. "A review of graph neural networks and pretrained
    language models" resolved to a paper on axonal projections in mouse whisker cortex. A
    reference list that renders those as links would assert, in a research-integrity app, that
    the user owns a paper they do not.

    Two signals, both about the title (the DOI is decisive when both sides carry one):

    * ratio ≥ ``FUZZY_TITLE_THRESHOLD`` **and** word coverage ≥ ``MIN_TITLE_WORD_COVERAGE``.
      The ratio alone is not enough: it reads two long titles differing in one decisive word as
      0.91, so "Bidirectional recurrent neural networks" resolved to "Relational recurrent neural
      networks". The coverage test is what notices the substitution;
    * **containment** — the shorter normalised title appears whole inside the longer. This is
      what a strict ratio misses: the regex often prefixes a title with the tail of the author
      list ("A., Lopes, G., … Real-time, low-latency closed-loop feedback …"), which drags a
      true match to 0.78. On the real corpus containment recovered that one true link and
      admitted **none** of the 12 false ones (they score 0.11-0.37 and contain nothing).

    A rejected resolution does not remove the reference — the paper still cites it. It only
    stops being a link.
    """
    if parsed_doi and library_doi and parsed_doi.strip().lower() == library_doi.strip().lower():
        return True
    if not parsed_title or not library_title:
        # No title on either side leaves nothing to check against. The resolution came from a
        # rule that never compared titles, so "unverifiable" is treated as "not credible".
        return False
    if (
        _title_similarity(parsed_title, library_title) >= FUZZY_TITLE_THRESHOLD
        and _title_word_coverage(parsed_title, library_title) >= MIN_TITLE_WORD_COVERAGE
    ):
        return True
    a, b = _normalize_for_match(parsed_title), _normalize_for_match(library_title)
    shorter, longer = (a, b) if len(a) <= len(b) else (b, a)
    return len(shorter) >= MIN_CONTAINED_TITLE_CHARS and shorter in longer


def match_to_library(
    parsed: ParsedCitation,
    *,
    fuzzy_title_threshold: float = FUZZY_TITLE_THRESHOLD,
) -> str | None:
    """Return Document.id if `parsed` matches a row in the library, else None.

    Strategy, in order:
      1. Exact DOI (case-insensitive)
      2. First-author surname + year **that also agree on the title** (KI-45)
      3. Fuzzy title (SequenceMatcher ratio >= threshold)

    Rule 2 narrows, it does not decide. Requiring `resolution_is_credible` before accepting it is
    what stops the false links: the pair picks the candidates, the title confirms them. It still
    earns its place above rule 3, because within same-surname/same-year candidates a *contained*
    title is safe to accept — which is how a true match the regex prefixed with an author-list
    tail (scoring 0.78, under the ratio threshold) is recovered.
    """
    with session_scope() as session:
        if parsed.doi:
            stmt = select(Document.id).where(
                Document.doi.is_not(None),
                Document.is_archived.is_(False),
                Document.doi.ilike(parsed.doi),
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is not None:
                return str(row)

        surname = _first_author_surname(parsed.authors)
        if surname and parsed.year is not None:
            author_year_stmt = select(
                Document.id, Document.authors, Document.title, Document.doi
            ).where(
                Document.is_archived.is_(False),
                Document.year == parsed.year,
                Document.authors.is_not(None),
            )
            for doc_id, doc_authors, doc_title, doc_doi in session.execute(author_year_stmt).all():
                doc_surname = _first_author_surname(doc_authors)
                if not doc_surname or doc_surname != surname:
                    continue
                # ⚠ KI-45: surname+year alone is not evidence. On a corpus holding many same-year
                # papers by common surnames it fired constantly and wrongly — 13 of 16 stored
                # resolutions were false, and one document had 9 of its 11 links pointing at two
                # unrelated papers. The pair now only *narrows the candidates*; the title still
                # has to agree before the match is accepted.
                if resolution_is_credible(
                    parsed_title=parsed.title,
                    parsed_doi=parsed.doi,
                    library_title=doc_title,
                    library_doi=doc_doi,
                ):
                    return str(doc_id)

        if parsed.title and len(parsed.title) >= 10:
            title_stmt = select(Document.id, Document.title, Document.doi).where(
                Document.is_archived.is_(False),
                Document.title.is_not(None),
            )
            best_id: str | None = None
            best_title: str | None = None
            best_doi: str | None = None
            best_score = 0.0
            for doc_id, doc_title, doc_doi in session.execute(title_stmt).all():
                score = _title_similarity(parsed.title, doc_title)
                if score > best_score:
                    best_score = score
                    best_id = str(doc_id)
                    best_title = doc_title
                    best_doi = doc_doi
            # The best candidate still has to pass the same test rule 2 does. One predicate for
            # "these are the same paper", used by every rule and by the read side, so the three
            # cannot drift apart again — which is how the surname+year rule came to disagree with
            # the reference list's own idea of a credible link (KI-45).
            if (
                best_id is not None
                and best_score >= fuzzy_title_threshold
                and resolution_is_credible(
                    parsed_title=parsed.title,
                    parsed_doi=parsed.doi,
                    library_title=best_title,
                    library_doi=best_doi,
                )
            ):
                return best_id

    return None
