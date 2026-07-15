"""Pure-core tests for the selective-ingestion registry (S1).

Covers the two pure functions with no I/O: `derive_status` (the full 8-combo truth table) and
`validate_selection` (every rejection category + the happy path). The impure scan/resolve
functions are exercised by the integration suite.
"""

import itertools

import pytest

from doc_assistant.ingest.registry import (
    STATUS_CHANGED,
    STATUS_INGESTED,
    STATUS_MISSING,
    STATUS_NEW,
    InvalidSelection,
    derive_status,
    validate_selection,
)

# --- derive_status: the full truth table (2^3 = 8 combos → 4 statuses) -------------------


@pytest.mark.parametrize(
    "file_exists,cache_fresh,has_document,expected",
    [
        # A vanished file is `missing` regardless of the other two.
        (False, False, False, STATUS_MISSING),
        (False, False, True, STATUS_MISSING),
        (False, True, False, STATUS_MISSING),
        (False, True, True, STATUS_MISSING),
        # File present, no Document row → `new` regardless of an incidental cache.
        (True, False, False, STATUS_NEW),
        (True, True, False, STATUS_NEW),
        # File present, Document row → cache freshness splits ingested vs changed.
        (True, True, True, STATUS_INGESTED),
        (True, False, True, STATUS_CHANGED),
    ],
)
def test_derive_status_truth_table(file_exists, cache_fresh, has_document, expected):
    assert derive_status(file_exists, cache_fresh, has_document) == expected


def test_derive_status_is_total():
    """Every one of the 8 input combinations returns one of the 4 known statuses."""
    valid = {STATUS_NEW, STATUS_CHANGED, STATUS_INGESTED, STATUS_MISSING}
    for combo in itertools.product([False, True], repeat=3):
        assert derive_status(*combo) in valid


# --- validate_selection: happy path -----------------------------------------------------


def test_validate_selection_happy_path_normalizes():
    known = {"a/b.pdf", "c.epub"}
    # backslashes, a leading ./, and mixed case suffix all normalize to a known rel_path.
    out = validate_selection(["a\\b.pdf", "./c.epub"], known)
    assert out == ["a/b.pdf", "c.epub"]


def test_validate_selection_dedupes_preserving_first_order():
    known = {"a.pdf", "b.pdf"}
    out = validate_selection(["b.pdf", "a.pdf", "b.pdf"], known)
    assert out == ["b.pdf", "a.pdf"]


def test_validate_selection_empty_is_empty():
    assert validate_selection([], {"a.pdf"}) == []


# --- validate_selection: rejection categories -------------------------------------------


def test_validate_selection_rejects_absolute_posix():
    with pytest.raises(InvalidSelection) as ei:
        validate_selection(["/etc/passwd.pdf"], set())
    assert "/etc/passwd.pdf" in ei.value.offenders["absolute"]


def test_validate_selection_rejects_windows_drive():
    with pytest.raises(InvalidSelection) as ei:
        validate_selection(["C:\\secret\\x.pdf"], set())
    assert ei.value.offenders["absolute"]


def test_validate_selection_rejects_traversal():
    with pytest.raises(InvalidSelection) as ei:
        validate_selection(["../outside.pdf"], set())
    assert "../outside.pdf" in ei.value.offenders["traversal"]


def test_validate_selection_rejects_unsupported_suffix():
    with pytest.raises(InvalidSelection) as ei:
        validate_selection(["a/notes.xyz"], {"a/notes.xyz"})
    assert "a/notes.xyz" in ei.value.offenders["unsupported"]


def test_validate_selection_rejects_unknown_rel_path():
    with pytest.raises(InvalidSelection) as ei:
        validate_selection(["ghost.pdf"], {"real.pdf"})
    assert "ghost.pdf" in ei.value.offenders["unknown"]


def test_validate_selection_lists_all_offenders_together():
    """One raise names every offender, grouped by reason (no fail-fast on the first)."""
    known = {"ok.pdf"}
    with pytest.raises(InvalidSelection) as ei:
        validate_selection(["/abs.pdf", "../up.pdf", "bad.xyz", "missing.pdf", "ok.pdf"], known)
    off = ei.value.offenders
    assert off["absolute"] == ["/abs.pdf"]
    assert off["traversal"] == ["../up.pdf"]
    assert off["unsupported"] == ["bad.xyz"]
    assert off["unknown"] == ["missing.pdf"]


def test_invalid_selection_message_names_offenders():
    err = InvalidSelection({"absolute": ["/x.pdf"], "unknown": ["y.pdf"], "traversal": []})
    msg = str(err)
    assert "/x.pdf" in msg and "y.pdf" in msg
    # empty categories are dropped from both the attribute and the message.
    assert "traversal" not in err.offenders
