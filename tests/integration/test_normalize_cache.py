"""Integration guard tests for the KI-14 cache-normalization runner.

Exercises ``scripts.normalize_cache.normalize_cache_dir`` against a real temp cache tree:
dry-run writes nothing, ``--apply`` rewrites only the files that change, clean files stay
byte-identical, and a second ``--apply`` reports zero changes (idempotence guard).
"""

from __future__ import annotations

from pathlib import Path

from scripts.normalize_cache import normalize_cache_dir

_PLACEHOLDER = "**==> picture [29 x 29] intentionally omitted <==**"
_DIRTY = f"Intro.\n\n{_PLACEHOLDER}\n\nBody paragraph.\n"
_DIRTY_CLEAN = "Intro.\n\nBody paragraph.\n"
_CLEAN = "# Heading\n\nNo images here.\n"


def _make_cache(tmp_path: Path) -> tuple[Path, Path, Path]:
    cache = tmp_path / "cache"
    (cache / "nested").mkdir(parents=True)
    dirty = cache / "dirty.md"
    clean = cache / "nested" / "clean.md"
    dirty.write_text(_DIRTY, encoding="utf-8")
    clean.write_text(_CLEAN, encoding="utf-8")
    return cache, dirty, clean


def test_dry_run_reports_but_writes_nothing(tmp_path):
    cache, dirty, clean = _make_cache(tmp_path)

    result = normalize_cache_dir(cache, apply=False)

    assert result.scanned == 2
    assert result.n_changed == 1
    assert result.total_placeholders == 1
    assert result.changed[0].path == dirty
    assert not result.applied
    # Nothing on disk changed.
    assert dirty.read_text(encoding="utf-8") == _DIRTY
    assert clean.read_text(encoding="utf-8") == _CLEAN


def test_apply_rewrites_only_changed_files(tmp_path):
    cache, dirty, clean = _make_cache(tmp_path)

    result = normalize_cache_dir(cache, apply=True)

    assert result.applied
    assert result.n_changed == 1
    assert dirty.read_text(encoding="utf-8") == _DIRTY_CLEAN
    # The already-clean file is untouched (byte-identical).
    assert clean.read_text(encoding="utf-8") == _CLEAN


def test_second_apply_reports_zero_changes(tmp_path):
    cache, _dirty, _clean = _make_cache(tmp_path)

    first = normalize_cache_dir(cache, apply=True)
    second = normalize_cache_dir(cache, apply=True)

    assert first.n_changed == 1
    assert second.scanned == 2
    assert second.n_changed == 0
    assert second.total_placeholders == 0


def test_empty_cache_dir_is_no_change(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    result = normalize_cache_dir(cache, apply=True)
    assert result.scanned == 0
    assert result.n_changed == 0
