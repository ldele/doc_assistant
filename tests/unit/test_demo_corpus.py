"""Unit tests for the demo-corpus pin layer (ADR-025 F3).

The pure half of ``doc_assistant.demo_corpus``: which manifest entries become pins, and what
counts as "this file is a demo file". No database, no network, no model load.

Contract: ``docs/specs/feature-corpus-folders-demo.md`` (M4, M10).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from doc_assistant import demo_corpus
from doc_assistant.library import SourcePin

_MANIFEST = textwrap.dedent(
    """\
    documents:
      - filename: "rag_lewis_2020.pdf"
        sha256: aaaa
        bytes: 100
      - filename: "eval_explicit.pdf"
        collection: eval
        sha256: bbbb
        bytes: 200
      - filename: "alexnet.pdf"
        collection: demo
        sha256: cccc
        bytes: 300
      - filename: "resnet.pdf"
        collection: demo
        sha256: dddd
        bytes: 400
    """
)


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "corpus_manifest.yaml"
    path.write_text(text, encoding="utf-8")
    return path


# --- pin loading ------------------------------------------------------------------------------ #


def test_only_demo_collection_entries_become_pins(tmp_path: Path) -> None:
    """An entry with no `collection` is an EVAL entry (the pre-demo manifest had no such field).

    Getting this backwards would fold the closed benchmark corpus into the demo folder.
    """
    pins = demo_corpus.load_demo_pins(_write(tmp_path, _MANIFEST))

    assert [p.filename for p in pins] == ["alexnet.pdf", "resnet.pdf"]
    assert pins[0] == SourcePin("alexnet.pdf", "cccc", 300)


@pytest.mark.parametrize(
    "text",
    ["", "documents:\n", "documents: []\n", "not-a-mapping\n", "{[unbalanced\n"],
    ids=["empty", "null-documents", "no-documents", "scalar", "malformed-yaml"],
)
def test_unusable_manifest_yields_no_pins_instead_of_raising(tmp_path: Path, text: str) -> None:
    """The caller no-ops on `[]`; a raise here would break an otherwise fine ingest (M10)."""
    assert demo_corpus.load_demo_pins(_write(tmp_path, text)) == []


def test_missing_manifest_yields_no_pins(tmp_path: Path) -> None:
    """The normal state of a frozen build — `tests/` is not bundled and PROJECT_ROOT is a temp
    unpack dir, so this path must be quiet and harmless, not an error (M10)."""
    assert demo_corpus.load_demo_pins(tmp_path / "nope.yaml") == []


def test_incomplete_pin_is_skipped_not_guessed(tmp_path: Path) -> None:
    """A pin with no sha256 can't identify bytes; assigning on the filename alone is the
    name-based matching M4 exists to avoid."""
    manifest = textwrap.dedent(
        """\
        documents:
          - filename: "no_hash.pdf"
            collection: demo
            bytes: 10
          - filename: "good.pdf"
            collection: demo
            sha256: eeee
            bytes: 20
        """
    )
    assert [p.filename for p in demo_corpus.load_demo_pins(_write(tmp_path, manifest))] == [
        "good.pdf"
    ]


# --- byte matching ---------------------------------------------------------------------------- #


def _pin_for(path: Path, name: str = "pinned.pdf") -> SourcePin:
    from doc_assistant.sources_manifest import sha256_file

    return SourcePin(name, sha256_file(path), path.stat().st_size)


def test_matches_by_bytes_even_when_renamed(tmp_path: Path) -> None:
    """M4: the manifest name is display only — a user who renamed a demo PDF still gets it."""
    original = tmp_path / "alexnet.pdf"
    original.write_bytes(b"demo-paper-content")
    by_size = demo_corpus.pins_by_size([_pin_for(original, "alexnet.pdf")])

    renamed = tmp_path / "my favourite paper.pdf"
    renamed.write_bytes(b"demo-paper-content")

    assert demo_corpus.file_matches_demo(renamed, by_size) is True


def test_same_size_different_bytes_is_not_a_match(tmp_path: Path) -> None:
    """The size index is a fast path, not the decision — the sha256 still has to agree."""
    pinned = tmp_path / "pinned.pdf"
    pinned.write_bytes(b"aaaaaaaa")
    by_size = demo_corpus.pins_by_size([_pin_for(pinned)])

    impostor = tmp_path / "impostor.pdf"
    impostor.write_bytes(b"bbbbbbbb")  # same length, different content
    assert impostor.stat().st_size == pinned.stat().st_size

    assert demo_corpus.file_matches_demo(impostor, by_size) is False


def test_unknown_size_is_rejected_without_reading_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the size index: a large non-demo corpus costs stat calls, not reads."""
    pinned = tmp_path / "pinned.pdf"
    pinned.write_bytes(b"aaaaaaaa")
    by_size = demo_corpus.pins_by_size([_pin_for(pinned)])

    other = tmp_path / "other.pdf"
    other.write_bytes(b"a different length entirely")

    def _explode(_: Path) -> str:
        raise AssertionError("sha256 must not be computed for a size that no pin claims")

    monkeypatch.setattr(demo_corpus, "sha256_file", _explode)
    assert demo_corpus.file_matches_demo(other, by_size) is False


def test_vanished_file_is_not_a_match(tmp_path: Path) -> None:
    """The hook runs just after ingest; a file moved mid-run must not raise into the ingest."""
    pinned = tmp_path / "pinned.pdf"
    pinned.write_bytes(b"aaaaaaaa")
    by_size = demo_corpus.pins_by_size([_pin_for(pinned)])

    assert demo_corpus.file_matches_demo(tmp_path / "gone.pdf", by_size) is False


def test_pins_by_size_groups_collisions(tmp_path: Path) -> None:
    """Two pins can share a size; the index must keep both candidates."""
    grouped = demo_corpus.pins_by_size(
        [SourcePin("a.pdf", "aa", 10), SourcePin("b.pdf", "bb", 10), SourcePin("c.pdf", "cc", 20)]
    )
    assert sorted(grouped) == [10, 20]
    assert len(grouped[10]) == 2
