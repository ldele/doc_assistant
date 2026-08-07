"""Guards for the release preflight (`scripts/release_preflight.py`).

Two jobs:
1. Run the checks that are cheap and always-true-in-CI as ordinary tests, so a regression fails on
   every push rather than only when someone remembers to run the preflight before a release.
2. Pin the preflight's own logic, because two of its checks were WRONG on their first run and both
   failed in the "looks green" direction — the dangerous one.
"""

from __future__ import annotations

import pytest
from scripts.release_preflight import (
    KI34_BROKEN_MIB,
    KI34_FIXED_MIB,
    SIDECAR_MIN_MIB,
    _just_recipes,
    check_dev_commands,
    check_versions,
)

# --- checks worth running on every push, not just at release time -----------------


def test_all_five_version_strings_agree() -> None:
    """v0.4.0 bumped four of them and missed `uv.lock`.

    CI and the Docker build install with `--locked`, which fails rather than re-resolving, so the
    job died *before* the gates ran — a red build that looked like a broken runner, on `main`, for
    days. Cheap to check, expensive to miss."""
    result = check_versions()
    assert result.status == "PASS", f"{result.detail}: {result.notes}"


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
