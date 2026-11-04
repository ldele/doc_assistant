"""`stored_paths`: the rule for whether a path stored in the library may be acted on (ADR-061)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from doc_assistant import app_settings, config, stored_paths
from doc_assistant.ingest import figures as figures_mod


@pytest.fixture
def library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    for sub in ("sources", "cache", "figures"):
        (tmp_path / sub).mkdir()
    monkeypatch.setattr(config, "DOCS_PATH", tmp_path / "sources")
    monkeypatch.setattr(config, "CACHE_PATH", tmp_path / "cache")
    monkeypatch.setattr(figures_mod, "FIGURE_DIR", tmp_path / "figures")
    monkeypatch.setattr(app_settings, "SETTINGS_PATH", tmp_path / "settings.json")
    monkeypatch.delenv("DOC_SOURCE_DIR", raising=False)
    monkeypatch.setattr(stored_paths, "source_bases", lambda: [tmp_path / "sources"])
    return tmp_path


def test_each_kind_of_file_belongs_in_its_own_folder(library: Path) -> None:
    assert stored_paths.confine(library / "cache" / "a" / "x.md", "cache") is not None
    assert stored_paths.confine(library / "figures" / "h" / "p1.png", "figure") is not None
    assert stored_paths.confine(library / "sources" / "x.pdf", "source") is not None
    # The right library, the wrong folder for the purpose.
    assert stored_paths.confine(library / "sources" / "x.md", "cache") is None
    assert stored_paths.confine(library / "cache" / "x.png", "figure") is None


def test_another_folder_and_a_walk_out_are_refused(library: Path, tmp_path_factory) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere")
    assert stored_paths.confine(elsewhere / "x.md", "cache") is None
    assert stored_paths.confine(library / "cache" / ".." / ".." / "x.md", "cache") is None
    # A sibling whose name starts like the folder is not inside it.
    assert stored_paths.confine(library / "cache-old" / "x.md", "cache") is None


def test_nothing_to_act_on_is_none(library: Path) -> None:
    assert stored_paths.confine(None, "cache") is None
    assert stored_paths.confine("", "cache") is None
    assert stored_paths.confine("   ", "source") is None


@pytest.mark.skipif(os.name != "nt", reason="Windows paths compare without case")
def test_case_does_not_matter_on_windows(library: Path) -> None:
    upper = Path(str(library / "cache").upper()) / "x.md"
    assert stored_paths.confine(upper, "cache") is not None


def test_a_registered_root_counts_for_sources(tmp_path: Path, monkeypatch) -> None:
    """A root the library registered (a reference add, ADR-046) is a place its sources may be."""
    root = tmp_path / "papers-elsewhere"
    root.mkdir()
    monkeypatch.setattr(stored_paths, "source_bases", lambda: [tmp_path / "sources", root])
    assert stored_paths.confine(root / "deep" / "x.pdf", "source") is not None
    assert stored_paths.confine(tmp_path / "x.pdf", "source") is None
