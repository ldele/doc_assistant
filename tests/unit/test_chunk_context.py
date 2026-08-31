"""Where a cited chunk sits in its source (ROADMAP 19, the read side).

A citation already says *which* document. This is the other half — the passage shown in place,
with what came before and after it. The rule the whole feature rests on is the one `locate_span`
established at ingest and this inherits: **a window centred on the wrong paragraph is worse than
no window**, so every failure returns `None` and the caller says it cannot place the chunk.

No Chroma here: the store is a dict-backed fake, because what is under test is the key parsing,
the span selection and the windowing — not chromadb's filter syntax.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from doc_assistant.library import get_chunk_context

_DOC = (
    "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron pi rho.\n\n"
    "The passage that is cited sits right here in the middle of the document.\n\n"
    "Sigma tau upsilon phi chi psi omega alpha beta gamma delta epsilon zeta eta theta iota.\n"
)
_CITED = "The passage that is cited sits right here in the middle of the document."


class _FakeStore:
    """Answers `get(where=..., include=..., limit=...)` the way the code calls it."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.raises = False

    def get(self, where: dict[str, Any], include: list[str], limit: int = 1) -> dict[str, Any]:
        if self.raises:
            raise RuntimeError("store unavailable")
        clauses = where["$and"]
        wanted = {k: v for clause in clauses for k, v in clause.items()}
        hits = [r for r in self.rows if all(r.get(k) == v for k, v in wanted.items())]
        return {"metadatas": hits[:limit]}


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    p = tmp_path / "doc.md"
    p.write_text(_DOC, encoding="utf-8")
    return p


def _parent_row(cache: Path, **over: Any) -> dict[str, Any]:
    start = _DOC.index(_CITED)
    row: dict[str, Any] = {
        "document_id": "doc-1",
        "parent_index": 2,
        "filename": "doc.pdf",
        "source_cache": str(cache),
        "parent_char_start": start,
        "parent_char_end": start + len(_CITED),
    }
    row.update(over)
    return row


def test_a_parent_key_resolves_to_the_passage_with_its_surroundings(cache: Path) -> None:
    """A chat citation is a *parent* — the unit the LLM actually read."""
    store = _FakeStore([_parent_row(cache)])
    ctx = get_chunk_context("doc-1:p2", store, window=40)
    assert ctx is not None
    assert ctx.text == _CITED
    assert ctx.before and ctx.before in _DOC
    assert ctx.after and ctx.after in _DOC
    assert ctx.doc_chars == len(_DOC)
    assert ctx.at_document_start is False and ctx.at_document_end is False


def test_a_flat_key_uses_its_own_span_not_a_parents(cache: Path) -> None:
    start = _DOC.index(_CITED)
    store = _FakeStore(
        [
            {
                "document_id": "doc-1",
                "chunk_index": 7,
                "filename": "doc.pdf",
                "source_cache": str(cache),
                "char_start": start,
                "char_end": start + len(_CITED),
                "page": 4,
            }
        ]
    )
    ctx = get_chunk_context("doc-1:7", store)
    assert ctx is not None and ctx.text == _CITED
    assert ctx.page == 4


def test_the_window_never_opens_or_closes_mid_word(cache: Path) -> None:
    """A window cut through a token reads as corruption; it is trimmed to a space."""
    store = _FakeStore([_parent_row(cache)])
    ctx = get_chunk_context("doc-1:p2", store, window=25)
    assert ctx is not None
    assert not ctx.before.startswith(" ")
    # Whatever survives the trim must still be real text from the document.
    assert ctx.before in _DOC and ctx.after in _DOC


def test_a_window_that_reaches_an_edge_says_so(cache: Path) -> None:
    """The reader needs to know a short window is the document ending, not a truncation."""
    store = _FakeStore([_parent_row(cache)])
    ctx = get_chunk_context("doc-1:p2", store, window=10_000)
    assert ctx is not None
    assert ctx.at_document_start and ctx.at_document_end


# --- every way this can fail returns None, and none of them guess ------------------------------ #


def test_a_chunk_whose_span_never_resolved_has_no_context(cache: Path) -> None:
    """The ~3-in-39,000 case from the live corpus: the locator refused, so this refuses too."""
    row = _parent_row(cache)
    del row["parent_char_start"], row["parent_char_end"]
    assert get_chunk_context("doc-1:p2", _FakeStore([row])) is None


def test_a_span_the_cache_no_longer_supports_is_refused(cache: Path) -> None:
    """A cache rewritten shorter than the recorded span must not be sliced at all."""
    store = _FakeStore([_parent_row(cache, parent_char_end=len(_DOC) + 500)])
    assert get_chunk_context("doc-1:p2", store) is None


def test_a_missing_cache_file_has_no_context(tmp_path: Path) -> None:
    store = _FakeStore([_parent_row(tmp_path / "gone.md")])
    assert get_chunk_context("doc-1:p2", store) is None


@pytest.mark.parametrize("key", ["", "garbage", "doc-1:", ":5", "doc-1:pXYZ", "doc-1:notanumber"])
def test_a_malformed_key_is_none_not_an_exception(cache: Path, key: str) -> None:
    assert get_chunk_context(key, _FakeStore([_parent_row(cache)])) is None


def test_an_unknown_chunk_is_none(cache: Path) -> None:
    assert get_chunk_context("doc-1:p99", _FakeStore([_parent_row(cache)])) is None


def test_a_store_that_raises_degrades_to_no_context(cache: Path) -> None:
    """Inform, don't block: a citation still renders, it just cannot be placed."""
    store = _FakeStore([_parent_row(cache)])
    store.raises = True
    assert get_chunk_context("doc-1:p2", store) is None
