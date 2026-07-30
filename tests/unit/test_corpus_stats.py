"""Guard tests for the Corpus panel's facts (ADR-037).

This surface exists to answer one user question honestly — *will this still work when my library
is much bigger?* — so the tests pin the two ways it could lie:

1. **the keyword-index mode**, which decides the panel's memory sentence. Reporting `on_disk`
   while the process actually runs the legacy in-RAM arm would tell a user their memory is flat
   when it is growing at ~5.9 KB/chunk (KI-32's ceiling);
2. **never blocking** — a settings panel that raises because a directory vanished mid-walk teaches
   the user nothing. Sizes degrade to 0, individually.
"""

from __future__ import annotations

import pytest

from doc_assistant import corpus_stats


@pytest.fixture
def data_home(tmp_path, monkeypatch):
    """Point every artifact path at a temp tree with known, distinguishable sizes."""
    chroma_pc = tmp_path / "chroma_pc"
    chroma = tmp_path / "chroma"
    (chroma_pc / "nested").mkdir(parents=True)
    chroma.mkdir()
    (chroma_pc / "a.bin").write_bytes(b"x" * 1000)
    (chroma_pc / "nested" / "b.bin").write_bytes(b"x" * 500)  # nested files count
    (chroma / "c.bin").write_bytes(b"x" * 200)
    (tmp_path / "library.db").write_bytes(b"x" * 30)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "d.md").write_bytes(b"x" * 7)

    monkeypatch.setattr(corpus_stats, "PC_CHROMA_PATH", str(chroma_pc))
    monkeypatch.setattr(corpus_stats, "CHROMA_PATH", str(chroma))
    monkeypatch.setattr(corpus_stats, "SQLITE_PATH", str(tmp_path / "library.db"))
    monkeypatch.setattr(corpus_stats, "CACHE_PATH", tmp_path / "cache")
    return tmp_path


class TestDiskAccounting:
    def test_sizes_are_per_artifact_and_sum_to_the_total(self, data_home):
        stats = corpus_stats.corpus_stats(documents=3, chunks=100, keyword_index_on_disk=False)

        disk = stats.disk
        assert disk.vector_store_bytes == 1500  # both files, including the nested one
        assert disk.baseline_store_bytes == 200
        assert disk.document_store_bytes == 30
        assert disk.extraction_cache_bytes == 7
        assert disk.total_bytes == 1500 + 200 + 30 + 7 + disk.keyword_index_bytes

    def test_a_missing_artifact_is_zero_not_an_error(self, data_home):
        """A fresh install has no index and no cache yet; the panel must still open."""
        (data_home / "library.db").unlink()

        stats = corpus_stats.corpus_stats(documents=0, chunks=0, keyword_index_on_disk=False)

        assert stats.disk.document_store_bytes == 0
        assert stats.disk.keyword_index_bytes == 0

    def test_an_unreadable_path_does_not_raise(self, data_home, monkeypatch):
        """Errors are swallowed per entry, so one locked file cannot take the panel down."""

        def boom(*a, **k):
            raise OSError("permission denied")

        monkeypatch.setattr(corpus_stats.os, "scandir", boom)

        stats = corpus_stats.corpus_stats(documents=1, chunks=1, keyword_index_on_disk=False)

        assert stats.disk.vector_store_bytes == 0  # degraded, not raised


class TestKeywordIndexMode:
    def test_on_disk_reports_size_and_build_time(self, data_home):
        index = data_home / "sparse_index.sqlite3"
        index.write_bytes(b"x" * 4096)

        state = corpus_stats.corpus_stats(
            documents=2, chunks=50, keyword_index_on_disk=True
        ).keyword_index

        assert state.mode == "on_disk"
        assert state.bytes == 4096
        assert state.built_at is not None and state.built_at.endswith("+00:00")

    def test_the_live_arm_wins_over_a_file_on_disk(self, data_home):
        """**The lie this test exists to prevent.** A stale index file next to a process running
        the legacy in-RAM arm must not report the reassuring answer: memory *is* growing with the
        corpus in that configuration."""
        (data_home / "sparse_index.sqlite3").write_bytes(b"x" * 4096)

        state = corpus_stats.corpus_stats(
            documents=2, chunks=50, keyword_index_on_disk=False
        ).keyword_index

        assert state.mode == "in_memory"
        assert state.bytes is None

    def test_an_empty_corpus_is_disabled_not_in_memory(self, data_home):
        """Robustness contract: 0 documents is supported, and there is no arm to describe."""
        state = corpus_stats.corpus_stats(
            documents=0, chunks=0, keyword_index_on_disk=False
        ).keyword_index

        assert state.mode == "disabled"

    def test_on_disk_without_the_file_reports_no_size_rather_than_lying(self, data_home):
        """The index was built and then deleted underneath a running process: the arm is still
        on-disk (it holds an open handle), but there is no size to report."""
        state = corpus_stats.corpus_stats(
            documents=2, chunks=50, keyword_index_on_disk=True
        ).keyword_index

        assert state.mode == "on_disk"
        assert state.bytes is None
        assert state.built_at is None


def test_as_dict_is_json_shaped(data_home):
    """The payload crosses the wire as-is; nested dataclasses have to flatten to dicts."""
    payload = corpus_stats.corpus_stats(
        documents=7, chunks=99, keyword_index_on_disk=False
    ).as_dict()

    assert payload["documents"] == 7
    assert payload["chunks"] == 99
    assert isinstance(payload["disk"], dict)
    assert isinstance(payload["keyword_index"], dict)
    assert payload["keyword_index"]["mode"] == "in_memory"
