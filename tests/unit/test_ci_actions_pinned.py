"""Guard that every GitHub Action CI runs is pinned to a commit, and that the pins stay current.

A tag such as ``actions/checkout@v4`` can be moved by whoever controls the action's repository, so
a workflow that names a tag runs whatever the tag points at on the day. A full commit SHA cannot
move (``docs/security.md`` S-12). Adding a step with ``@v5`` is an edit nobody would think of as a
security change, so these tests make it a red CI run. Dependabot rewrites a pin in the same
``@<sha> # vX.Y.Z`` shape, so its PRs keep passing.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
DEPENDABOT = ROOT / ".github" / "dependabot.yml"

USES = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<ref>\S+)(?P<rest>.*)$")
PINNED = re.compile(r"^[\w.-]+/[\w./-]+@[0-9a-f]{40}$")
VERSION_COMMENT = re.compile(r"^\s*#\s*v\d+(?:\.\d+)*\s*$")


def _uses_lines() -> list[tuple[str, int, str, str]]:
    found = []
    for workflow in sorted(WORKFLOWS.glob("*.y*ml")):
        for number, line in enumerate(workflow.read_text(encoding="utf-8").splitlines(), start=1):
            match = USES.match(line)
            if match:
                found.append((workflow.name, number, match["ref"], match["rest"]))
    return found


def test_the_workflows_use_actions_at_all() -> None:
    # Guards the guard: a renamed folder or a changed layout must not turn the next test into a
    # pass over nothing.
    assert len(_uses_lines()) >= 5


def test_every_action_is_pinned_to_a_full_commit_sha_with_its_release() -> None:
    unpinned = [
        f"{name}:{number} {ref}{rest}"
        for name, number, ref, rest in _uses_lines()
        if not ref.startswith("./") and not (PINNED.match(ref) and VERSION_COMMENT.match(rest))
    ]
    assert not unpinned, "pin each to a commit SHA, release in a comment:\n" + "\n".join(unpinned)


def test_dependabot_keeps_the_action_pins_current() -> None:
    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    ecosystems = {update["package-ecosystem"] for update in config["updates"]}
    assert "github-actions" in ecosystems
