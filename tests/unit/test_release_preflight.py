"""Guards for the release preflight (`scripts/release_preflight.py`).

Two jobs:
1. Run the checks that are cheap and always-true-in-CI as ordinary tests, so a regression fails on
   every push rather than only when someone remembers to run the preflight before a release.
2. Pin the preflight's own logic, because two of its checks were WRONG on their first run and both
   failed in the "looks green" direction — the dangerous one.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from scripts import release_preflight as preflight
from scripts.release_preflight import (
    KI34_BROKEN_MIB,
    KI34_FIXED_MIB,
    SIDECAR_MIN_MIB,
    _just_recipes,
    check_dev_commands,
    check_versions,
    collect_versions,
)

# --- checks worth running on every push, not just at release time -----------------


def test_all_version_strings_agree() -> None:
    """v0.4.0 bumped four of them and missed `uv.lock`.

    CI and the Docker build install with `--locked`, which fails rather than re-resolving, so the
    job died *before* the gates ran — a red build that looked like a broken runner, on `main`, for
    days. Cheap to check, expensive to miss."""
    result = check_versions()
    assert result.status == "PASS", f"{result.detail}: {result.notes}"


def test_every_file_carrying_a_version_is_actually_read() -> None:
    """The list of sources, pinned — because the second version bug was a file nobody opened.

    `apps/desktop/src-tauri/Cargo.toml` and its lock sat at 0.4.1 through v0.4.2, v0.5.0 and
    v0.5.1 while `check_versions` reported green, because it never read them and neither did
    `docs/RELEASE.md` §1. The test above cannot catch that: agreement among the files you *do*
    read says nothing about the one you skip. Adding a version-carrying file to the repo means
    adding it here and to the runbook table."""
    assert set(collect_versions()) == {
        "pyproject.toml",
        "uv.lock",
        "src/doc_assistant/__init__.py",
        "apps/desktop/package.json",
        "apps/desktop/src-tauri/tauri.conf.json",
        "apps/desktop/src-tauri/Cargo.toml",
        "apps/desktop/src-tauri/Cargo.lock",
    }


def test_no_version_source_reads_as_a_placeholder() -> None:
    """A reader that quietly stops finding its value must not be able to look like agreement.

    Every miss returns a sentinel — `(not found)`, `(missing)` — and sentinels compare equal to
    each other, so a broken parse in *all* sources would make `check_versions` pass on seven
    identical placeholders. Requiring each value to look like a version closes that."""
    unparsed = {k: v for k, v in collect_versions().items() if not re.match(r"^\d+\.\d+\.\d+", v)}
    assert not unparsed, f"not version strings — the reader is broken, not the version: {unparsed}"


def test_no_developer_commands_in_shipped_ui() -> None:
    """The app's only failure message once read "backend unreachable. Run `just api`" (KI-39).

    A task runner and a repository recipe, shown to someone who installed an .exe. Anything a user
    is told to run must exist on their machine."""
    result = check_dev_commands()
    assert result.status == "PASS", f"{result.detail}: {result.notes}"


# --- the preflight's own logic (both of these were wrong on the first run) ---------


def test_sidecar_floor_sits_between_the_recorded_KI34_sizes() -> None:
    """The floor only works if it separates the broken build from the fixed one.

    Written first in decimal MB while the recorded numbers are MiB, which put the floor at
    ~1478 MiB — below *both* — so the check passed on a build it was written to reject, while
    looking green. A units bug in a safety check is worse than no check."""
    assert KI34_BROKEN_MIB < SIDECAR_MIN_MIB < KI34_FIXED_MIB, (
        f"floor {SIDECAR_MIN_MIB} must reject the broken build ({KI34_BROKEN_MIB} MiB) "
        f"and accept the fixed one ({KI34_FIXED_MIB} MiB)"
    )


def test_just_recipes_are_read_from_the_justfile() -> None:
    """`just` is an ordinary English word.

    Matching `just \\w+` flagged "just now", "just a" and "just the" — three false positives on the
    first run. Only a real recipe name makes `just X` a command, so the names come from the
    justfile."""
    recipes = _just_recipes()
    assert {"api", "sidecar", "test"} <= recipes, f"parsed recipes look wrong: {sorted(recipes)}"
    for prose_word in ("now", "a", "the", "before"):
        assert prose_word not in recipes


@pytest.mark.parametrize("phrase", ["just now", "just a moment", "just the tags"])
def test_ordinary_english_is_not_a_dev_command(phrase: str) -> None:
    """Pins the false-positive fix: prose containing "just" must never trip the check."""
    word = phrase.split()[1]
    assert word not in _just_recipes()


# --- does `versions` actually CATCH a drift, or only read the files? --------------


VERSION_FILES: dict[str, str] = {
    "pyproject.toml": '[project]\nname = "doc-assistant"\nversion = "{v}"\n',
    "uv.lock": '[[package]]\nname = "doc-assistant"\nversion = "{v}"\n',
    "src/doc_assistant/__init__.py": '__version__ = "{v}"\n',
    "apps/desktop/package.json": '{{"name": "doc-assistant-desktop", "version": "{v}"}}\n',
    "apps/desktop/src-tauri/tauri.conf.json": '{{"version": "{v}"}}\n',
    "apps/desktop/src-tauri/Cargo.toml": (
        '[package]\nname = "doc-assistant-desktop"\nversion = "{v}"\n'
    ),
    "apps/desktop/src-tauri/Cargo.lock": (
        'version = 3\n\n[[package]]\nname = "serde"\nversion = "1.0.0"\n\n'
        '[[package]]\nname = "doc-assistant-desktop"\nversion = "{v}"\n'
    ),
}


def _fake_repo(tmp_path: Path, *, drifted: str | None = None) -> Path:
    """A minimal tree carrying all seven version strings, one optionally left behind at 0.4.1."""
    for rel, template in VERSION_FILES.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(template.format(v="0.4.1" if rel == drifted else "9.9.9"), encoding="utf-8")
    return tmp_path


def test_the_control_passes_when_every_source_agrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this, the drift tests below could pass for the wrong reason (a broken reader)."""
    monkeypatch.setattr(preflight, "ROOT", _fake_repo(tmp_path))
    result = check_versions()
    assert result.status == "PASS", f"{result.detail}: {result.notes}"
    assert "9.9.9" in result.detail


@pytest.mark.parametrize("drifted", sorted(VERSION_FILES))
def test_a_version_left_behind_in_any_single_file_fails(
    drifted: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One stale file out of seven must fail — for **every** file, not just the ones we remember.

    Parametrised over the source list rather than spot-checked, because the failure this guards
    was a file that was never looked at: a hand-written test would have covered the five someone
    already had in mind, which is exactly the set that was not the problem. `Cargo.toml` and
    `Cargo.lock` shipped at 0.4.1 through three tagged releases."""
    monkeypatch.setattr(preflight, "ROOT", _fake_repo(tmp_path, drifted=drifted))
    result = check_versions()
    assert result.status == "FAIL", f"a stale {drifted} went unnoticed"
    assert any(drifted in note and "0.4.1" in note for note in result.notes), (
        f"the report must name the stale file and its value; got {result.notes}"
    )


# --- artifact_fresh: history, not mtimes -----------------------------------------


def test_the_shipped_path_list_is_pinned() -> None:
    """Same lesson as the version-source list: a freshness check is worth its path list.

    A path missing from `SHIPPED_PATHS` is a change that can alter the installer while the
    preflight reports the artifact fresh — invisible, exactly like the Cargo files were to
    `versions`. `Cargo.lock`'s absence is deliberate and argued in the constant's comment."""
    assert set(preflight.SHIPPED_PATHS) == {
        "src/",
        "apps/api/",
        "pyproject.toml",
        "uv.lock",
        "scripts/build_sidecar.py",
        "scripts/doc_assistant_api.spec",
        "apps/desktop/src/",
        "apps/desktop/index.html",
        "apps/desktop/package.json",
        "apps/desktop/vite.config.ts",
        "apps/desktop/src-tauri/src/",
        "apps/desktop/src-tauri/build.rs",
        "apps/desktop/src-tauri/Cargo.toml",
        "apps/desktop/src-tauri/tauri.conf.json",
        "apps/desktop/src-tauri/icons/",
    }
    assert "apps/desktop/src-tauri/Cargo.lock" not in preflight.SHIPPED_PATHS
    for path in preflight.SHIPPED_PATHS:
        assert (Path(__file__).resolve().parents[2] / path).exists(), f"{path} does not exist"


def _git(repo: Path, *args: str) -> None:
    subprocess.run(("git", *args), cwd=repo, check=True, capture_output=True, text=True)


def _write(repo: Path, rel: str, text: str) -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway repo with one shipped file and one that is not, each in its own commit."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "t@example.invalid")
    _git(tmp_path, "config", "user.name", "test")
    _write(tmp_path, "src/doc_assistant/__init__.py", '__version__ = "1.0.0"\n')
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "shipped code")
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    return tmp_path


def test_a_docs_commit_does_not_make_the_artifact_stale(repo: Path) -> None:
    """The everyday case that must stay quiet: writing a DEVLOG entry changes no shipped byte."""
    _write(repo, "docs/DEVLOG.md", "an entry\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "docs: an entry")
    found = preflight._newest_shipped_change()
    assert found is not None
    assert "shipped code" in found[0], f"a docs commit moved the freshness bar: {found[0]}"


def test_a_checkout_that_only_touches_mtimes_does_not_make_the_artifact_stale(repo: Path) -> None:
    """**The regression test.** `git checkout main` re-materialises files with today's date and
    byte-identical content; the old mtime comparison called that a source edit and demanded a
    rebuild that would have changed nothing (2026-09-02, blob `a789456…` on both sides)."""
    stamped = repo / "src/doc_assistant/__init__.py"
    future = datetime.now(tz=UTC).timestamp() + 86_400
    os.utime(stamped, (future, future))
    found = preflight._newest_shipped_change()
    assert found is not None
    # The premise, asserted so this test cannot quietly stop testing anything: under the old
    # comparison that file *was* the newest thing in the tree, which is why it failed the check.
    assert preflight._mtime_aware(stamped) > found[1], "the bumped mtime is not actually newer"
    assert "shipped code" in found[0], f"an mtime bump was read as an edit: {found[0]}"
    assert found[1] < datetime.now(tz=UTC) + timedelta(hours=1), "the future mtime leaked through"


def test_an_uncommitted_edit_to_shipped_code_DOES_make_the_artifact_stale(repo: Path) -> None:
    """mtime is still trusted where it means something: a file git reports as modified."""
    _write(repo, "src/doc_assistant/__init__.py", '__version__ = "2.0.0"\n')
    found = preflight._newest_shipped_change()
    assert found is not None
    assert "uncommitted" in found[0], f"an unsaved edit went unnoticed: {found[0]}"


def test_a_committed_edit_to_shipped_code_moves_the_bar(repo: Path) -> None:
    """The check must still do its job — this is the KI-34 case it exists for."""
    _write(repo, "src/doc_assistant/__init__.py", '__version__ = "2.0.0"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: real change")
    found = preflight._newest_shipped_change()
    assert found is not None
    assert "real change" in found[0], f"a shipped edit did not move the bar: {found[0]}"


def test_the_lock_cargo_rewrites_during_the_build_is_not_a_source_edit(repo: Path) -> None:
    """Cargo rewrites `Cargo.lock` *while building*, so every release ends with a lock newer than
    the artifact it just produced. Counting it would fail this check on every release — which is
    what happened at 0.6.0. See the argument in the `SHIPPED_PATHS` comment."""
    _write(repo, "apps/desktop/src-tauri/Cargo.lock", 'name = "x"\nversion = "0.6.0"\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "chore: pin Cargo.lock")
    found = preflight._newest_shipped_change()
    assert found is not None
    assert "shipped code" in found[0], f"the build's own output read as a source edit: {found[0]}"


@pytest.mark.parametrize(
    ("rel", "shipped"),
    [
        ("src/doc_assistant/rag.py", True),
        ("uv.lock", True),
        ("apps/desktop/src-tauri/icons/icon.ico", True),
        ("uv.lock.bak", False),  # a bare startswith over the tuple would call this shipped
        ("apps/desktop/src-tauri/icons.old/icon.ico", False),
        ("apps/desktop/src-tauri/Cargo.lock", False),  # written by the build being judged
        ("docs/DEVLOG.md", False),
        ("tests/unit/test_release_preflight.py", False),
        ("scripts/release_preflight.py", False),  # this file cannot make the artifact stale
        ("apps/desktop/src-tauri/target/release/x.exe", False),  # build output
    ],
)
def test_shipped_paths_match_exactly_not_by_bare_prefix(rel: str, shipped: bool) -> None:
    """A directory entry matches by prefix; a file entry matches only itself."""
    assert preflight._is_shipped(rel) is shipped
