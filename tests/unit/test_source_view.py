"""The source viewer (ROADMAP 18, ADR-050) — page derivation and the honest refusals.

The rendering half needs a real PDF and lives in the integration suite; what is under test here
is the part that decides *which* page and *whether* there is one to show.

The load-bearing case is `page_for_offset`. ADR-050 D2 exists because the row's stated premise
was false: on the live parent-child path only 1.5% of chunks carry a `page`, and all of them are
figures. Everything a reader sees jump to a page is this function, so it is tested against the
marker layout the extractor actually writes (`extractors.py:99`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc_assistant.library import (
    PageUnavailable,
    get_source_view,
    locate_chunk,
    page_for_chunk,
    page_for_offset,
    render_page,
)

# The shape `extract_pdf_pymupdf` writes: a marker, then that page's text.
_CACHE = (
    "\n<!-- page:1 -->\nFirst page body text.\n"
    "\n<!-- page:2 -->\nSecond page body text.\n"
    "\n<!-- page:3 -->\nThird page body text.\n"
)


class _FakeStore:
    """Answers `get(where=..., include=..., limit=...)` the way `_chunk_metadata` calls it."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def get(self, where: dict[str, Any], include: list[str], limit: int = 1) -> dict[str, Any]:
        wanted = {k: v for clause in where["$and"] for k, v in clause.items()}
        hits = [r for r in self.rows if all(r.get(k) == v for k, v in wanted.items())]
        return {"metadatas": hits[:limit]}


# --- page_for_offset: the whole of D2 ---------------------------------------------------------- #


@pytest.mark.parametrize(
    ("needle", "expected"),
    [
        ("First page body", 1),
        ("Second page body", 2),
        ("Third page body", 3),
    ],
)
def test_an_offset_resolves_to_the_page_its_marker_precedes(needle: str, expected: int) -> None:
    assert page_for_offset(_CACHE, _CACHE.index(needle)) == expected


def test_an_offset_before_the_first_marker_has_no_page() -> None:
    """Not page 1 by assumption. Nothing precedes it, so nothing is claimed."""
    assert page_for_offset(_CACHE, 0) is None


def test_text_without_markers_has_no_page() -> None:
    """An unmarked cache is the honest `None` the viewer opens at page 1 on."""
    assert page_for_offset("no markers anywhere in this text", 10) is None
    assert page_for_offset("", 0) is None


def test_an_offset_past_the_end_still_resolves_to_the_last_page() -> None:
    """Clamped rather than refused: the last marker genuinely does precede it."""
    assert page_for_offset(_CACHE, len(_CACHE) * 10) == 3


def test_a_negative_offset_is_refused() -> None:
    assert page_for_offset(_CACHE, -1) is None


def test_the_marker_itself_belongs_to_the_page_it_opens() -> None:
    """An offset landing exactly on `<!-- page:2 -->` is on page 2, not page 1."""
    assert page_for_offset(_CACHE, _CACHE.index("<!-- page:2 -->")) == 2


def test_resolution_is_on_the_passage_start_not_its_end() -> None:
    """A passage straddling a break opens where it *begins* — the D2 divergence from ingest.

    The ingest-time rule (`extract_chunk_metadata`) labels such a chunk with the page it finishes
    on, which would open the reader past the sentence they clicked.
    """
    start = _CACHE.index("First page body")
    end = _CACHE.index("Second page body") + 5
    assert page_for_offset(_CACHE, start) == 1
    assert page_for_offset(_CACHE, end) == 2  # the same text, asked about at its other end


# --- page_for_chunk ---------------------------------------------------------------------------- #


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(_CACHE, encoding="utf-8")
    return p


def test_a_text_parent_gets_its_page_derived(cache: Path) -> None:
    """The 98%: a parent that stored no page still places, from its span plus the cache."""
    store = _FakeStore(
        [
            {
                "document_id": "doc-1",
                "parent_index": 4,
                "source_cache": str(cache),
                "parent_char_start": str(_CACHE.index("Second page body")),
                "parent_char_end": str(_CACHE.index("Second page body") + 10),
            }
        ]
    )
    assert page_for_chunk("doc-1:p4", store) == 2


def test_a_figure_chunk_uses_its_stored_page(cache: Path) -> None:
    """A figure's page is detection output, not a reconstruction — and it has no text span.

    This is the one case where the stored value is the better answer, and the only reason a
    figure citation can be placed at all.
    """
    store = _FakeStore(
        [
            {
                "document_id": "doc-1",
                "parent_index": 9,
                "chunk_type": "figure",
                "page": "7",
                "source_cache": str(cache),
            }
        ]
    )
    assert page_for_chunk("doc-1:p9", store) == 7


def test_an_unparseable_stored_page_falls_through_to_derivation(cache: Path) -> None:
    store = _FakeStore(
        [
            {
                "document_id": "doc-1",
                "parent_index": 4,
                "page": "not-a-number",
                "source_cache": str(cache),
                "parent_char_start": str(_CACHE.index("Third page body")),
            }
        ]
    )
    assert page_for_chunk("doc-1:p4", store) == 3


def test_a_chunk_with_no_span_and_no_stored_page_cannot_be_placed(cache: Path) -> None:
    store = _FakeStore([{"document_id": "doc-1", "parent_index": 4, "source_cache": str(cache)}])
    assert page_for_chunk("doc-1:p4", store) is None


def test_a_missing_cache_file_cannot_be_placed(tmp_path: Path) -> None:
    store = _FakeStore(
        [
            {
                "document_id": "doc-1",
                "parent_index": 4,
                "source_cache": str(tmp_path / "gone.md"),
                "parent_char_start": "20",
            }
        ]
    )
    assert page_for_chunk("doc-1:p4", store) is None


def test_an_unknown_chunk_key_cannot_be_placed(cache: Path) -> None:
    assert page_for_chunk("doc-1:p4", _FakeStore([])) is None
    assert page_for_chunk("nonsense", _FakeStore([])) is None


# --- locate_chunk: the chat citation's entry point ------------------------------------------- #


def test_locating_a_chunk_gives_its_document_and_its_page(cache: Path) -> None:
    """A chat citation carries a `chunk_key` and no document id — this is where it gets one."""
    store = _FakeStore(
        [
            {
                "document_id": "doc-1",
                "parent_index": 4,
                "source_cache": str(cache),
                "parent_char_start": str(_CACHE.index("Second page body")),
            }
        ]
    )
    found = locate_chunk("doc-1:p4", store)
    assert found is not None
    assert (found.document_id, found.page) == ("doc-1", 2)


def test_a_figure_locates_to_a_document_and_its_stored_page(cache: Path) -> None:
    """The case the chat card could not offer before: a figure has no text position at all."""
    store = _FakeStore(
        [
            {
                "document_id": "doc-9",
                "parent_index": 3,
                "chunk_type": "figure",
                "page": "12",
                "source_cache": str(cache),
            }
        ]
    )
    found = locate_chunk("doc-9:p3", store)
    assert found is not None
    assert (found.document_id, found.page) == ("doc-9", 12)


def test_an_unplaceable_chunk_still_names_its_document(cache: Path) -> None:
    """`page` is nullable where `document_id` is not — the document can always be opened."""
    store = _FakeStore([{"document_id": "doc-1", "parent_index": 4, "source_cache": str(cache)}])
    found = locate_chunk("doc-1:p4", store)
    assert found is not None
    assert found.document_id == "doc-1"
    assert found.page is None


def test_an_unknown_key_locates_to_nothing(cache: Path) -> None:
    assert locate_chunk("doc-1:p4", _FakeStore([])) is None
    assert locate_chunk("nonsense", _FakeStore([])) is None


# --- the availability gate (D3/D4) ------------------------------------------------------------- #


def test_an_unknown_document_has_no_view() -> None:
    """`None` is the 404. It is *not* what a missing file returns — see the next test."""
    assert get_source_view("no-such-document") is None


def test_rendering_an_unknown_document_is_refused_with_a_sentence() -> None:
    with pytest.raises(PageUnavailable, match="document not found"):
        render_page("no-such-document", 1)


def _view(monkeypatch: pytest.MonkeyPatch, **over: Any) -> Any:
    """A `SourceDocumentView` as `get_source_view` would build it, without a database."""
    from doc_assistant.library import source_view as mod

    base: dict[str, Any] = {
        "document_id": "doc-1",
        "filename": "paper.pdf",
        "format": "pdf",
        "page_count": 12,
        "available": True,
        "pageable": True,
        "path": "C:/nowhere/paper.pdf",
        "reason": None,
    }
    base.update(over)
    view = mod.SourceDocumentView(**base)
    monkeypatch.setattr(mod, "get_source_view", lambda _id: view)
    return view


def test_a_format_without_pages_is_refused_as_a_property_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ADR-050 D3: an EPUB has no pages. The pane shows its text, not a broken image."""
    _view(monkeypatch, format="epub", pageable=False)
    with pytest.raises(PageUnavailable, match="document in EPUB format has no pages"):
        render_page("doc-1", 1)


def test_an_unreachable_file_is_refused_with_the_reason_the_view_carries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D4: the sentence names the path, and it is the *view's* sentence, not a second one."""
    _view(
        monkeypatch,
        available=False,
        path=None,
        reason="The drive holding this document is not connected (E:\\papers).",
    )
    with pytest.raises(PageUnavailable, match=r"drive holding this document is not connected"):
        render_page("doc-1", 1)


def test_pageable_is_decided_by_format_not_by_trying(monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate is a format check, so a non-PDF never reaches PyMuPDF at all."""
    from doc_assistant.library import source_view as mod

    assert "pdf" in mod.PAGEABLE_FORMATS
    for fmt in ("epub", "html", "docx", "md", ""):
        assert fmt not in mod.PAGEABLE_FORMATS
