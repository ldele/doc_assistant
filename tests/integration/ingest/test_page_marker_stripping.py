"""KI-29 guard — ``<!-- page:N -->`` must not survive into the parent-child path.

The asymmetry that caused it: the baseline chunk path built ``page_content`` through
``clean_chunk_text``, while ``build_parent_child_chunks`` applied it to neither the child
``page_content`` nor the ``parent_text`` metadata. Parent-child is the **default** retrieval mode
(``USE_PARENT_CHILD=true``, locked), so the cleaned path was the one nothing used at answer time
and markers reached the embeddings, the LLM's evidence block and the user's source panel.

The class of bug is KI-26's ``_JOURNAL_HEADER`` again: a stripper that exists, is documented, and
is simply never called on the path that matters. So these tests assert on the *output of the
chunker*, not on the stripper.
"""

from __future__ import annotations

from doc_assistant.ingest import PAGE_MARKER, build_parent_child_chunks
from doc_assistant.ingest.chunking import clean_chunk_text

_MD = """<!-- page:1 -->
# Introduction
Retrieval-augmented generation grounds a language model in retrieved passages.

<!-- page:2 -->
## Method
We combine BM25 with a dense retriever and rerank with a cross-encoder. The ensemble
fuses sparse and dense candidates before the final reranking step.

<!-- page:3 -->
## Results
The hybrid system outperforms either retriever alone on the benchmark.
"""


def _chunks():
    chunks = build_parent_child_chunks(_MD, {"filename": "paper.pdf"})
    assert chunks, "chunker produced no chunks"
    return chunks


def test_the_fixture_actually_contains_markers() -> None:
    """Guard the guard — if the fixture loses its markers these tests prove nothing."""
    assert len(PAGE_MARKER.findall(_MD)) == 3


def test_no_page_marker_in_child_page_content() -> None:
    """What gets embedded, and what the user reads in a cited excerpt."""
    offenders = [c.page_content for c in _chunks() if PAGE_MARKER.search(c.page_content)]
    assert not offenders, f"{len(offenders)} child chunk(s) still carry a page marker: {offenders}"


def test_no_page_marker_in_parent_text() -> None:
    """``parent_text`` is what reaches the LLM as evidence in parent-child mode."""
    offenders = [
        c.metadata["parent_text"]
        for c in _chunks()
        if PAGE_MARKER.search(str(c.metadata.get("parent_text", "")))
    ]
    assert not offenders, f"{len(offenders)} parent(s) still carry a page marker"


def test_no_literal_comment_syntax_survives_anywhere() -> None:
    """Belt and braces: a malformed marker the regex misses would still be visible noise."""
    for chunk in _chunks():
        assert "<!-- page:" not in chunk.page_content
        assert "<!-- page:" not in str(chunk.metadata["parent_text"])


def test_the_prose_itself_is_preserved() -> None:
    """Stripping must remove markers, not text — a silent truncation would also pass the above."""
    joined = " ".join(c.page_content for c in _chunks())
    for phrase in ("Retrieval-augmented generation", "cross-encoder", "outperforms either"):
        assert phrase in joined


def test_no_chunk_is_empty_after_stripping() -> None:
    """A child that was nothing but a marker must be dropped, never embedded as ''."""
    for chunk in _chunks():
        assert chunk.page_content.strip()


def test_a_marker_only_parent_is_dropped_rather_than_stored_blank() -> None:
    chunks = build_parent_child_chunks("<!-- page:1 -->\n", {"filename": "blank.pdf"})
    assert chunks == []


def test_a_scan_with_no_text_layer_yields_no_chunks_at_all() -> None:
    """The real corpus case (`middleton-2001.pdf`): 15 page markers, zero text.

    Before KI-29 those markers were embedded *as if they were content*. Now the document
    correctly yields nothing — which is why ``ingest`` must guard its Chroma upserts against
    an empty list (Chroma raises "Expected Embeddings to be non-empty list"). Pinned here
    because this input is what turns that guard from theoretical into load-bearing.
    """
    scan = "".join(f"\n<!-- page:{n} -->\n\n\n" for n in range(1, 16))
    assert build_parent_child_chunks(scan, {"filename": "scan.pdf"}) == []


def test_child_index_stays_contiguous_within_each_parent() -> None:
    """Dropping cleaned-empty children must not leave gaps downstream grouping relies on."""
    by_parent: dict[int, list[int]] = {}
    for chunk in _chunks():
        by_parent.setdefault(int(chunk.metadata["parent_index"]), []).append(
            int(chunk.metadata["child_index"])
        )
    for parent_index, child_indices in by_parent.items():
        assert child_indices == list(range(len(child_indices))), (
            f"parent {parent_index} has non-contiguous child_index {child_indices}"
        )


def test_every_child_is_a_substring_of_its_cleaned_parent() -> None:
    """The parent-child contract: a child must still locate inside its parent after cleaning.

    If only one side were stripped, this is the invariant that would break — which is exactly
    how a half-applied fix would slip through.
    """
    for chunk in _chunks():
        assert chunk.page_content in str(chunk.metadata["parent_text"])


def test_parent_child_now_agrees_with_the_baseline_path() -> None:
    """The asymmetry itself: both paths must clean, or the default path lies about the other."""
    baseline_clean = clean_chunk_text("Some text\n<!-- page:7 -->\nmore text")
    assert "<!-- page:" not in baseline_clean

    chunks = build_parent_child_chunks(
        "Some text\n<!-- page:7 -->\nmore text", {"filename": "x.pdf"}
    )
    assert chunks
    assert all("<!-- page:" not in c.page_content for c in chunks)
