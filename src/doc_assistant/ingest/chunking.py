"""Text → chunk shaping for ingest — pure, store-free.

The splitter factories + import-time singletons, per-chunk metadata extraction, the
health signals, page-marker cleaning, and the table-aware parent/child chunking.
No DB or filesystem: given text in, chunks/metadata out.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from doc_assistant import config

from .tables_marker import TABLE_BLOCK_RE

PAGE_MARKER = re.compile(r"<!--\s*page:(\d+)\s*-->")
HEADING_MARKER = re.compile(r"^(#{1,6})\s+(.+?)$", re.MULTILINE)

# A table caption is short; never pull a large block of prose into a table's parent
# when absorbing the caption attached immediately before a spliced table block.
_MAX_ABSORBED_CAPTION_CHARS = 1000
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")

# Splitter sizes are config-driven (see config.PARENT_CHUNK_SIZE etc.) so a
# chunking sweep can vary them via env without editing source. The factories
# read ``config`` attributes at call time, which keeps them monkeypatch-able
# in tests; the module-level singletons below preserve the original import-time
# construction for the hot path.


def _make_parent_splitter() -> RecursiveCharacterTextSplitter:
    """Large-passage splitter for parent chunks (sent to the LLM)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_CHUNK_SIZE,
        chunk_overlap=config.PARENT_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
    )


def _make_child_splitter() -> RecursiveCharacterTextSplitter:
    """Small-passage splitter for child chunks (embedded for retrieval)."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_CHUNK_SIZE,
        chunk_overlap=config.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " "],
    )


def _make_baseline_splitter() -> RecursiveCharacterTextSplitter:
    """Single-chunk splitter for the baseline (non parent-child) store."""
    return RecursiveCharacterTextSplitter(
        chunk_size=config.BASELINE_CHUNK_SIZE,
        chunk_overlap=config.BASELINE_CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", ". ", " "],
    )


_pc_parent_splitter = _make_parent_splitter()
_pc_child_splitter = _make_child_splitter()


def extract_chunk_metadata(
    chunk_text: str, full_text: str, chunk_start: int
) -> dict[str, int | str | None]:
    """Find the nearest preceding heading and current page number."""
    # Find page number -- last page marker at or before this chunk's start
    text_before = full_text[: chunk_start + len(chunk_text)]
    page_matches = list(PAGE_MARKER.finditer(text_before))
    page: int | None = int(page_matches[-1].group(1)) if page_matches else None

    heading_matches = list(HEADING_MARKER.finditer(text_before))
    section: str | None
    if heading_matches:
        raw_section = heading_matches[-1].group(2).strip()
        section = re.sub(r"[*_`]+", "", raw_section).strip()
        # Empty after stripping = not a real heading
        section = section if section else None
    else:
        section = None

    return {"page": page, "section": section}


def compute_health_signals(documents: list[Document], full_text: str) -> dict[str, int | float]:
    """Compute signals for health classification from a list of chunks."""
    if not documents:
        return {
            "chunk_count": 0,
            "avg_chunk_length": 0.0,
            "section_detection_rate": 0.0,
            "reference_flagged_ratio": 0.0,
        }

    chunk_lengths = [len(d.page_content) for d in documents]
    sections_detected = sum(1 for d in documents if d.metadata.get("section"))

    return {
        "chunk_count": len(documents),
        "avg_chunk_length": sum(chunk_lengths) / len(chunk_lengths),
        "section_detection_rate": sections_detected / len(documents),
        "reference_flagged_ratio": 0.0,
    }


def clean_chunk_text(text: str) -> str:
    """Remove page markers from displayed text (keep them only for metadata)."""
    return PAGE_MARKER.sub("", text).strip()


def _split_trailing_paragraph(text: str) -> tuple[str, str]:
    """Split ``text`` into ``(head, trailing_paragraph)`` at the last blank line.

    The trailing paragraph is everything after the final blank line — the caption the
    splice attaches (single newline) immediately before a table block. ``head`` is the
    rest. With no blank line the whole input is the trailing paragraph.
    """
    matches = list(_BLANK_LINE_RE.finditer(text))
    if not matches:
        return "", text
    boundary = matches[-1]
    return text[: boundary.end()], text[boundary.end() :]


#: How much of a chunk's head and tail is used to locate it when it is not a verbatim substring.
#: Long enough to be distinctive in running prose, short enough to survive the splitter's
#: whitespace handling at both ends.
_LOCATE_PROBE_CHARS = 60

#: Span verification compares raw markdown with cleaned chunk text, so whitespace must not decide.
_WHITESPACE = re.compile(r"\s+")


def _span_holds(haystack: str, span: tuple[int, int], needle: str) -> bool:
    """Does the resolved span actually contain what it claims? Whitespace-insensitively.

    The head/tail probe can land on the wrong occurrence when a document repeats near-identical
    passages — measured on a real paper, a chunk about "Figure S4" resolved onto the "Figure S5"
    caption. Verifying here converts that from a *wrong* answer into *no* answer, which is the
    trade this feature is built on. Normalised because the span is raw markdown while the chunk
    has been through `clean_chunk_text`, so page markers and collapsed blank lines differ.
    """
    window = _WHITESPACE.sub(" ", haystack[span[0] : span[1]])
    return _WHITESPACE.sub(" ", needle).strip() in window


def locate_span(haystack: str, needle: str, cursor: int = 0) -> tuple[int, int] | None:
    """Where ``needle`` sits in ``haystack`` at or after ``cursor``, or ``None`` if unresolvable.

    **The splitter does not emit verbatim substrings.** `RecursiveCharacterTextSplitter` strips
    each piece and rejoins on its separator, so a chunk whose original text held (say) three
    newlines comes back holding two and `str.find` misses it outright. Measured on ordinary
    prose: 2 of 8 children did not match exactly.

    So an exact search is tried first, then the head and tail are probed independently and the
    span is taken between them. When even that fails this returns ``None`` — deliberately, rather
    than falling back to the running cursor. A *missing* offset costs the reader a highlight and
    the caller can still match the text at read time; a *wrong* offset points confidently at the
    wrong paragraph, which is worse than not answering. Searching from ``cursor`` is what keeps a
    passage that repeats verbatim mapped to its own occurrence.
    """
    exact = haystack.find(needle, cursor)
    if exact != -1:
        return exact, exact + len(needle)  # exact by construction; no verification needed

    stripped = needle.strip()
    if not stripped:
        return None
    head = stripped[:_LOCATE_PROBE_CHARS]
    start = haystack.find(head, cursor)
    if start == -1:
        return None
    tail = stripped[-_LOCATE_PROBE_CHARS:]
    tail_at = haystack.find(tail, start)
    if tail_at == -1:
        return None
    span = (start, tail_at + len(tail))
    return span if _span_holds(haystack, span, stripped) else None


def _table_aware_parents(text: str) -> list[str]:
    """Split ``text`` into parent passages, keeping spliced tables retrievable.

    Each spliced table block (``<!-- table:<engine>:page=N:begin -->`` … ``:end -->``)
    is kept **whole** as a single parent and is **co-located with its caption** (the
    caption paragraph the splice attached right before it). A wide table otherwise both
    (a) splits mid-grid across parents and (b) is orphaned from its caption: the
    caption (e.g. "Table 2: Top-20 & Top-100 retrieval accuracy …") is the natural
    query magnet, so retrieval surfaces the caption parent while the grid parent — the
    one holding the numbers — ranks below the candidate pool and never reaches the LLM.
    Binding caption + grid into one atomic parent makes the caption child map straight
    back to the values. Non-table prose is chunked normally. See docs/DEVLOG.md
    2026-06-06.
    """
    parents: list[str] = []
    cursor = 0
    for m in TABLE_BLOCK_RE.finditer(text):
        head, caption = _split_trailing_paragraph(text[cursor : m.start()])
        if len(caption.strip()) > _MAX_ABSORBED_CAPTION_CHARS:
            head, caption = head + caption, ""  # too long to be a caption — leave it
        if head.strip():
            parents.extend(_pc_parent_splitter.split_text(head))
        block = (caption + m.group(0)).strip()
        if block:
            parents.append(block)
        cursor = m.end()
    tail = text[cursor:]
    if tail.strip():
        parents.extend(_pc_parent_splitter.split_text(tail))
    return parents


def build_parent_child_chunks(text: str, base_metadata: dict[str, Any]) -> list[Document]:
    """Produce child chunks each carrying its parent text in metadata.

    Table-aware (see ``_table_aware_parents``): spliced table blocks stay whole and
    travel with their caption, so a wide table's values stay retrievable. Documents
    without spliced tables chunk exactly as before.

    **Page markers are stripped from both the child ``page_content`` and the
    ``parent_text`` metadata** (KI-29). This path is the default retrieval mode, so its
    text is what gets embedded, what the LLM receives as evidence, and what the user
    reads in the source panel — a ``<!-- page:N -->`` left in it leaks into all three.
    The stripping happens here, at assembly, rather than on ``text`` up front, so that
    chunk boundaries and the table-caption binding in ``_table_aware_parents`` are
    computed on exactly the same input as before; only the stored text changes. Page
    *numbers* are unaffected — the parent-child path never derived them from the chunk
    body (the baseline path reads them from the full text before the chunk, see
    ``extract_chunk_metadata``).

    A child that is nothing but a marker cleans down to the empty string; those are
    dropped rather than embedded, and ``child_index`` stays contiguous within a parent.

    **``char_start`` / ``char_end`` locate the chunk in the cached markdown** — the file
    ``source_cache`` names — and deliberately span the *raw* text, before ``clean_chunk_text``
    removed page markers. They are what lets a reader be shown where an answer came from
    (ROADMAP 19) without re-deriving it per query, which is the ingest-once-amortises rule: the
    corpus is read far more often than it is written. Three things they are **not**: an offset
    into ``page_content`` (that is the cleaned text), an offset into the source PDF (use ``page``
    for that), and a promise of exactness when a passage repeats verbatim — the cursor makes
    each occurrence map to itself, but a pathological duplicate can still land on the wrong one.

    **The cursor advances to each span's START, not its end, and that is load-bearing.** Both
    splitters emit *overlapping* chunks (``PARENT_CHUNK_OVERLAP`` 200, ``CHILD_CHUNK_OVERLAP``
    50), so the next chunk begins *before* the previous one ended. Searching onward from the
    previous end therefore starts past the answer: `find` either misses it — no offset at all —
    or lands on a later duplicate, which is a *wrong* offset presented confidently. Measured on
    12 documents of the live corpus: advancing to the end located 2,761 of 3,652 spans and lost
    **122 parents outright** (a lost parent takes all of its children with it); advancing to
    ``start + 1`` locates all 3,652. The ``+ 1`` is what still keeps a verbatim repeat from
    mapping twice onto the same occurrence.
    """
    parents = _table_aware_parents(text)
    children: list[Document] = []
    parent_cursor = 0
    for parent_idx, parent_text in enumerate(parents):
        parent_span = locate_span(text, parent_text, parent_cursor)
        if parent_span is not None:
            parent_cursor = parent_span[0] + 1

        clean_parent = clean_chunk_text(parent_text)
        if not clean_parent:
            continue
        child_idx = 0
        child_cursor = 0
        for child_text in _pc_child_splitter.split_text(parent_text):
            child_span = locate_span(parent_text, child_text, child_cursor)
            if child_span is not None:
                child_cursor = child_span[0] + 1

            clean_child = clean_chunk_text(child_text)
            if not clean_child:
                continue
            meta = {
                **base_metadata,
                "parent_text": clean_parent,
                "parent_index": parent_idx,
                "child_index": child_idx,
            }
            # Spans in the CACHED MARKDOWN — absent when unresolvable rather than guessed.
            if parent_span is not None:
                meta["parent_char_start"] = parent_span[0]
                meta["parent_char_end"] = parent_span[1]
                if child_span is not None:
                    start = parent_span[0] + child_span[0]
                    end = parent_span[0] + child_span[1]
                    # **Verify the COMPOSED span, not just its two halves.** Each `locate_span`
                    # above verified itself, but composition has a gap neither can see: if the
                    # *parent* matched a duplicate occurrence, both finds were exact and the sum
                    # still points at the wrong place. It needs a document that repeats a whole
                    # ~2,000-char parent, so it is rare — and to be clear about the evidence,
                    # **it is not currently observed**: all 39,087 spans on the live corpus hold.
                    # Kept because it costs one comparison and the failure it prevents is the one
                    # this feature is built to avoid — a *wrong* highlight rather than none.
                    if _span_holds(text, (start, end), child_text):
                        meta["char_start"] = start
                        meta["char_end"] = end
            children.append(Document(page_content=clean_child, metadata=meta))
            child_idx += 1
    return children
