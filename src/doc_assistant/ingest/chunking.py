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
    """
    parents = _table_aware_parents(text)
    children: list[Document] = []
    for parent_idx, parent_text in enumerate(parents):
        for child_idx, child_text in enumerate(_pc_child_splitter.split_text(parent_text)):
            meta = {
                **base_metadata,
                "parent_text": parent_text,
                "parent_index": parent_idx,
                "child_index": child_idx,
            }
            children.append(Document(page_content=child_text, metadata=meta))
    return children
