"""The suite runs on a data directory of its own (`tests/conftest.py`, KI-60).

A test cannot show that a whole run left the working library alone. That was measured, with a
snapshot of `data/` before and after a full run (DEVLOG 2026-10-01 (6)). What is pinned here is
the mechanism the measurement rests on: every default store resolves inside the run's own
directory, none inside the repository's `data/`, and a subprocess lands in the same place.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import doc_assistant.db.session as session_mod
from doc_assistant import app_settings, config
from tests import conftest as suite

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKING_DATA = (REPO_ROOT / "data").resolve()


def _inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def _configured_paths() -> dict[str, Path]:
    """Every path-valued setting in `config`, by name: the `Path` constants, and the strings
    that name a store (`CHROMA_PATH`, `SQLITE_PATH`, `SQLITE_URL` ...)."""
    found: dict[str, Path] = {}
    for name, value in vars(config).items():
        if name.startswith("_"):
            continue
        if isinstance(value, Path):
            found[name] = value.resolve()
        elif isinstance(value, str) and name.endswith(("_PATH", "_URL", "_DIR")):
            found[name] = Path(value.removeprefix("sqlite:///")).resolve()
    return found


def test_the_data_home_is_the_runs_own_empty_directory() -> None:
    assert suite.DATA_HOME.resolve() == config.DATA_PATH
    assert Path(os.environ["DOC_DATA_DIR"]).resolve() == config.DATA_PATH
    assert suite.DATA_HOME_ROOT.resolve() == config.DATA_PATH.parent
    assert config.DATA_PATH.is_dir()
    assert not _inside(config.DATA_PATH, WORKING_DATA)


def test_no_configured_path_points_into_the_working_data_directory() -> None:
    """Read from `config` itself, so a path added later is covered without being listed here."""
    paths = _configured_paths()
    assert {"SQLITE_PATH", "CACHE_PATH", "DOCS_PATH", "EXPORT_DIR", "CHROMA_PATH"} <= set(paths)
    offenders = {name: str(p) for name, p in paths.items() if _inside(p, WORKING_DATA)}
    assert offenders == {}


@pytest.mark.parametrize(
    "name",
    ["SQLITE_PATH", "CACHE_PATH", "DOCS_PATH", "EXPORT_DIR", "WIKI_DIR", "CONCEPT_SKELETON_DIR"],
)
def test_each_default_store_is_inside_the_data_home(name: str) -> None:
    assert _inside(_configured_paths()[name], config.DATA_PATH)


def test_the_settings_file_is_inside_the_data_home() -> None:
    """The user's own `settings.json` picks the provider and the source folder: read by a test,
    it makes the verdict depend on the machine."""
    assert _inside(Path(app_settings.SETTINGS_PATH).resolve(), config.DATA_PATH)


def test_the_vector_stores_stay_in_the_data_home_when_its_path_is_ascii() -> None:
    """`config` relocates the stores to a machine-wide directory for a non-ASCII data path on
    Windows (KI-11). The suite's directory sits in the repository so that does not happen on an
    ASCII checkout, and everything a run creates is in the one place that gets removed."""
    if sys.platform == "win32" and not str(config.DATA_PATH).isascii():
        pytest.skip("this checkout's path is not ASCII: the vector stores are relocated (KI-11)")
    for name in ("CHROMA_PATH", "PC_CHROMA_PATH"):
        assert Path(getattr(config, name)).resolve().parent == config.DATA_PATH


def test_the_default_engine_writes_inside_the_data_home() -> None:
    """The engine is bound when `db.session` is imported. Two route tests enter the app's
    lifespan without swapping it, and the lifespan migrates whatever it is bound to."""
    database = session_mod.get_engine().url.database
    assert database is not None
    assert _inside(Path(database).resolve(), config.DATA_PATH)
    assert not _inside(Path(database).resolve(), WORKING_DATA)


def test_a_subprocess_started_by_a_test_lands_in_the_same_data_home() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "from doc_assistant import config; print(config.DATA_PATH)"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    assert Path(proc.stdout.strip()) == config.DATA_PATH


def test_a_run_whose_config_was_imported_first_is_refused(tmp_path: Path) -> None:
    """The redirect only works if `config` is imported after it. A plugin that imports the app
    first would leave the suite on whatever `config` resolved, so the run must stop before any
    test, say why, and leave no directory of its own behind."""
    (tmp_path / "imports_config_first.py").write_text(
        "import doc_assistant.config  # noqa: F401\n", encoding="utf-8"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(tmp_path), env.get("PYTHONPATH")]))
    before = {p.name for p in suite.DATA_HOME_ROOT.iterdir()}

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "imports_config_first",
            "-p",
            "no:cacheprovider",
            "-q",
            str(Path(__file__)),
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    report = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "could not give the suite its own data directory" in report
    assert "passed" not in report  # nothing was run
    assert {p.name for p in suite.DATA_HOME_ROOT.iterdir()} == before


def test_a_run_directory_is_swept_only_once_it_is_stale(tmp_path: Path) -> None:
    """A run that was killed leaves its directory behind, and a later run removes it. One that
    is still in use (another run, a session paused in a debugger) is left alone."""
    now = time.time()
    old, recent, other = tmp_path / "run-1-old", tmp_path / "run-2-recent", tmp_path / "keep"
    for directory in (old, recent, other):
        directory.mkdir()
        (directory / "library.db").write_text("x", encoding="utf-8")
    long_ago = now - suite.STALE_AFTER_SECONDS - 60
    os.utime(old, (long_ago, long_ago))
    os.utime(other, (long_ago, long_ago))

    suite.sweep_stale(tmp_path, now)

    assert not old.exists()
    assert recent.exists()
    assert other.exists()  # not a run directory: never touched, however old
