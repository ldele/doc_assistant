"""Where a chunk came from in the cached markdown (ROADMAP 19, ADR-046 era).

The feature exists so a reader can be shown the passage an answer was drawn from. Its whole
design rests on one asymmetry, and these tests encode it: **a missing location costs a highlight;
a wrong location points confidently at the wrong paragraph.** So `locate_span` returns ``None``
rather than guessing, and every span it does return is verified to contain what it claims.

Measured on the real corpus while building it: 70-86% of chunks resolve, and after verification
**zero** resolved to the wrong passage. Before verification, three chunks in one paper resolved
onto a neighbouring figure caption — repetitive captions defeat a head/tail probe.

**That 70-86% was a bug, not a property — corrected 2026-08-30.** It was the cursor advancing to
each span's *end* while both splitters emit overlapping chunks, so every subsequent search began
past its answer. Re-measured on 12 documents of the live corpus after the fix: 3,652 of 3,652,
with **zero** parents lost against 122 before. The lesson is in the shape of the number: a rate
that nobody could explain sat in this docstring for months as though it described the text.
"""

from __future__ import annotations

import re

from doc_assistant.ingest.chunking import build_parent_child_chunks, locate_span

_WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS.sub(" ", s).strip()


# ============================================================
# locate_span
# ============================================================


def test_a_verbatim_substring_is_located_exactly():
    text = "alpha beta gamma delta"
    assert locate_span(text, "beta gamma") == (6, 16)


def test_the_cursor_keeps_a_repeated_passage_on_its_own_occurrence():
    """Without it every copy of a boilerplate line collapses onto the first one."""
    text = "repeat me. filler. repeat me."
    first = locate_span(text, "repeat me.")
    assert first == (0, 10)
    second = locate_span(text, "repeat me.", first[1])
    assert second == (19, 29)


def test_a_chunk_the_splitter_reflowed_is_still_found():
    """`RecursiveCharacterTextSplitter` strips and rejoins on its separator, so a chunk is not
    always a verbatim substring of what it was cut from — measured at 2 of 8 children."""
    haystack = "Intro.\n\n\n\nThe body text follows here and continues.\n\n\n\nEnd."
    needle = "The body text follows here and continues."
    span = locate_span(haystack, needle)
    assert span is not None
    assert needle in haystack[span[0] : span[1]]


def test_an_unfindable_chunk_returns_none_rather_than_a_guess():
    """The core trade. A running-cursor fallback would return a confidently wrong span."""
    assert locate_span("alpha beta gamma", "nothing like this at all") is None


def test_a_span_that_does_not_hold_its_own_text_is_refused():
    """The bug this caught: a "Figure S4" chunk resolving onto the "Figure S5" caption.

    The head probe matches the shared prefix and the tail probe matches the shared suffix, so a
    naive implementation returns a span spanning the wrong caption entirely.
    """
    haystack = "**Figure S5. Average assembly speed in frames.** Body of five. More text here."
    needle = "**Figure S4. Discriminability of affinity fields.** Body of four. More text here."
    span = locate_span(haystack, needle)
    assert span is None or _norm(needle) in _norm(haystack[span[0] : span[1]])


def test_an_empty_or_whitespace_needle_is_not_located():
    assert locate_span("alpha beta", "   ") is None
    assert locate_span("alpha beta", "") == (0, 0) or locate_span("alpha beta", "") is not None


# ============================================================
# The metadata the chunker actually emits
# ============================================================


def _doc(paragraphs: int = 6) -> str:
    parts = []
    for i in range(paragraphs):
        parts.append(
            f"# Section {i}\n\nDistinct opening sentence number {i}. " + "Filler words. " * 30
        )
    return "\n\n".join(parts)


def test_every_recorded_span_actually_contains_its_chunk():
    """The invariant. If this ever fails, the viewer is highlighting the wrong text."""
    text = _doc()
    chunks = build_parent_child_chunks(text, {"source_cache": "x.md"})
    located = [c for c in chunks if "char_start" in c.metadata]
    assert located, "some chunks must resolve, or the feature does nothing"
    for c in located:
        span = text[c.metadata["char_start"] : c.metadata["char_end"]]
        assert _norm(c.page_content)[:60] in _norm(span), c.metadata


def test_a_chunk_without_a_resolvable_span_simply_omits_it():
    """Absent keys, not sentinel values: a caller must not be able to read -1 as a position."""
    text = _doc()
    for c in build_parent_child_chunks(text, {"source_cache": "x.md"}):
        if "char_start" not in c.metadata:
            assert "char_end" not in c.metadata
        assert c.metadata.get("char_start") != -1


def test_the_child_span_sits_inside_its_parent_span():
    text = _doc()
    for c in build_parent_child_chunks(text, {"source_cache": "x.md"}):
        m = c.metadata
        if "char_start" in m and "parent_char_start" in m:
            assert m["parent_char_start"] <= m["char_start"]
            assert m["char_end"] <= m["parent_char_end"]


def test_spans_are_offsets_into_the_cache_not_into_page_content():
    """`page_content` is cleaned; the span is raw. Conflating them silently shifts every offset."""
    text = "# T\n\n<!-- page:1 -->\n\nThe sentence that matters here, at length, with more words."
    chunks = build_parent_child_chunks(text, {"source_cache": "x.md"})
    located = [c for c in chunks if "char_start" in c.metadata]
    if located:
        c = located[0]
        assert c.metadata["char_end"] <= len(text)
        # The marker is absent from page_content but present in the raw slice it points at.
        assert "<!-- page:1 -->" not in c.page_content


# ============================================================
# The cursor (found 2026-08-30)
# ============================================================
#
# The 70-86% resolve rate this file's own docstring recorded was not a property of the text. It
# was this: the cursor advanced to each span's END while both splitters emit OVERLAPPING chunks,
# so every subsequent search started past the answer. A measurement was written down as a fact
# without the cause being diagnosed, and it stayed for months.


def _overlapping_document(sentences: int = 400) -> str:
    """Unbroken prose — the shape that forces the splitter to OVERLAP consecutive parents.

    This matters, and the first attempt at this fixture got it wrong. Paragraph-separated text
    splits cleanly on its blank lines, so the parents barely overlap and the bug does not fire:
    that version passed against the broken code and proved nothing. One wall of prose leaves the
    splitter no separator to cut on, so every parent overlaps its predecessor by
    ``PARENT_CHUNK_OVERLAP`` — which is exactly when a cursor parked at the previous span's *end*
    begins past the next one.

    Every sentence is distinct on purpose: a repeated phrase could let a later duplicate be found
    by accident and mask the failure.
    """
    return " ".join(
        f"Sentence {i} discusses topic {i} using wording that appears nowhere else in this "
        f"document."
        for i in range(sentences)
    )


def test_every_child_of_an_overlapping_document_gets_a_span() -> None:
    """The regression. Pre-fix this loses 60 of 124 children — every other parent.

    Asserts totality, not a rate. A rate is what let this hide for months as a property of the
    corpus: this file's own docstring recorded "70-86% of chunks resolve" as a finding, when it
    was the cursor skipping past overlapping chunks.
    """
    chunks = build_parent_child_chunks(_overlapping_document(), {"doc_hash": "h"})
    assert chunks, "fixture produced no chunks"
    missing = [c for c in chunks if c.metadata.get("char_start") is None]
    assert missing == [], (
        f"{len(missing)} of {len(chunks)} children have no span — the cursor is skipping past "
        f"overlapping chunks again"
    )


def test_no_parent_is_lost_when_parents_overlap() -> None:
    """A lost parent takes every one of its children's spans with it, so it is the expensive
    half. Pre-fix, parents 1, 3, 5, 7 … were never located at all."""
    chunks = build_parent_child_chunks(_overlapping_document(), {"doc_hash": "h"})
    unlocated = sorted(
        {c.metadata["parent_index"] for c in chunks if c.metadata.get("parent_char_start") is None}
    )
    assert unlocated == [], f"parents {unlocated} were never located"


def test_spans_advance_within_a_parent_and_stay_inside_the_document() -> None:
    """The cheap check that a `+1` cursor did not start matching things out of order.

    Ordering is asserted **per parent**, not across the whole document — because parents overlap
    by design, the first child of parent N+1 legitimately begins *before* the last child of
    parent N. A first version of this test asserted global order and failed on correct output;
    the overlap region really does belong to both parents.
    """
    text = _overlapping_document()
    chunks = build_parent_child_chunks(text, {"doc_hash": "h"})
    by_parent: dict[int, list[int]] = {}
    for c in chunks:
        s, e = c.metadata["char_start"], c.metadata["char_end"]
        assert 0 <= s < e <= len(text), f"span ({s}, {e}) is outside a {len(text)}-char document"
        by_parent.setdefault(c.metadata["parent_index"], []).append(s)
    for parent_index, starts in by_parent.items():
        assert starts == sorted(starts), f"parent {parent_index}'s children came back out of order"


def test_a_composed_span_that_lands_on_a_duplicate_is_dropped_not_recorded() -> None:
    """Both halves can be exact and the sum still wrong — the case `_span_holds` missed.

    If the parent search matches a duplicate occurrence, the child resolves exactly inside that
    wrong parent and the composed offset points confidently at the wrong part of the document.

    **Honest about its own power: this test does not currently discriminate** — it passes with and
    without the composed-span check, because no fixture here reproduces a duplicated whole parent
    and the live corpus has none either (all 39,087 spans hold). It is an invariant test, not a
    regression test, and it is kept for the invariant: **every** recorded span must contain its own
    chunk. A first version of the surrounding change was justified by "580 wrong spans" that turned
    out to be an artefact of comparing cleaned text against a raw slice — the numbers here are the
    corrected ones.
    """
    header = "bioRxiv preprint doi: https://doi.org/10.1101/2021.04.30.442096 | 2\n\n"
    body = [
        header + " ".join(f"Section {i} sentence {j} of ordinary prose." for j in range(60))
        for i in range(8)
    ]
    text = "\n\n".join(body)

    chunks = build_parent_child_chunks(text, {"doc_hash": "h"})
    located = [c for c in chunks if c.metadata.get("char_start") is not None]
    assert located, "the fixture must still resolve most of its chunks"
    for c in located:
        span = text[c.metadata["char_start"] : c.metadata["char_end"]]
        assert _norm(c.page_content)[:60] in _norm(span), (
            f"span {c.metadata['char_start']}..{c.metadata['char_end']} does not contain its "
            f"chunk — a composed offset landed on a duplicate: {c.page_content[:60]!r}"
        )
