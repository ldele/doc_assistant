"""ADR-056 gate 3: the joins on a real library are the ones its owner reviewed (local-only).

Gate 5 put every join the rule made on the user's library in front of the user, who judged each
place: *close it* or *not text*, and chose to close the URL runs as well. The review record keeps,
for each place, what extraction is expected to do (``extraction``: ``close`` or ``keep``). This
test reads the record of runs that a re-extraction wrote beside each cached text and checks the set
of runs that closed against that review, both ways: a join nobody reviewed fails, and so does a
reviewed join that no longer happens. So a later change to the rule, or to PyMuPDF, that moves a
join fails here instead of passing silently.

**Local-only, like the private eval set.** The library and the review name the user's documents, so
neither is in the repository: the test reads them from two paths in the environment and is skipped
without them (CI never has them).

* ``DOC_LETTER_SPACING_LIBRARY``: a data home whose cache was written by this extractor.
* ``DOC_LETTER_SPACING_REVIEW``: the gate-5 review record (JSON, ``items`` with ``file``,
  ``spaced`` and ``extraction``), or a reference derived from it with the ADR's later accepted
  decisions applied. The review keeps what extraction did when it was taken; ADR-056 point 1
  (scanned pages are skipped, accepted after the review) leaves two reviewed places spaced, and
  point 6 pins that, so the 2026-11-04 run read a reference with those two set to ``keep``.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from doc_assistant.ingest.cache import read_spacing_record

LIBRARY = os.environ.get("DOC_LETTER_SPACING_LIBRARY", "")
REVIEW = os.environ.get("DOC_LETTER_SPACING_REVIEW", "")

pytestmark = pytest.mark.skipif(
    not (LIBRARY and REVIEW and Path(LIBRARY, "library.db").is_file() and Path(REVIEW).is_file()),
    reason="local-only: needs a library and its reviewed joins (ADR-056 gate 3)",
)


def _key(filename: str, spaced: str) -> tuple[str, str]:
    """One run, wherever its spaces fall: the review and the record space glyphs the same way, but
    a run's identity is its file and its letters."""
    return filename, spaced.replace(" ", "")


def _closed_on_the_library() -> dict[tuple[str, str], set[int]]:
    """Every run the library's records say closed, with the pages it closed on."""
    db = sqlite3.connect(f"{Path(LIBRARY, 'library.db').resolve().as_uri()}?mode=ro", uri=True)
    try:
        rows = db.execute("SELECT filename, source_cache FROM documents").fetchall()
    finally:
        db.close()
    closed: dict[tuple[str, str], set[int]] = {}
    for filename, cache in rows:
        if not cache:
            continue
        for run in read_spacing_record(Path(cache)):
            if run.get("state") == "closed":
                closed.setdefault(_key(filename, str(run["spaced"])), set()).add(int(run["page"]))
    return closed


def test_the_joins_on_the_library_are_the_reviewed_ones() -> None:
    items = json.loads(Path(REVIEW).read_text(encoding="utf-8"))["items"]
    expected = {_key(i["file"], i["spaced"]) for i in items if i["extraction"] == "close"}
    kept = {_key(i["file"], i["spaced"]) for i in items if i["extraction"] == "keep"}
    closed = _closed_on_the_library()

    unreviewed = sorted(set(closed) - expected - kept)
    against_the_review = sorted(set(closed) & kept)
    missing = sorted(expected - set(closed))
    assert not unreviewed, f"joins nobody reviewed: {unreviewed}"
    assert not against_the_review, f"closed although reviewed as kept: {against_the_review}"
    assert not missing, f"reviewed joins that no longer close: {missing}"


def test_each_reviewed_join_closes_on_its_reviewed_pages() -> None:
    items = json.loads(Path(REVIEW).read_text(encoding="utf-8"))["items"]
    closed = _closed_on_the_library()
    for item in items:
        if item["extraction"] != "close":
            continue
        pages = closed.get(_key(item["file"], item["spaced"]), set())
        assert set(item["pages"]) <= pages, (item["file"], item["spaced"], item["pages"], pages)
