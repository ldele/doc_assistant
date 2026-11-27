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


def _workflows() -> list[tuple[str, dict]]:
    return [
        (w.name, yaml.safe_load(w.read_text(encoding="utf-8")))
        for w in sorted(WORKFLOWS.glob("*.y*ml"))
    ]


def test_setup_uv_installs_a_named_release() -> None:
    # S-15: the action is pinned, but `version: "latest"` (or none) still lets a new uv change
    # what CI resolves on the day it ships.
    steps = [
        (name, step)
        for name, workflow in _workflows()
        for job in workflow["jobs"].values()
        for step in job.get("steps", [])
        if str(step.get("uses", "")).startswith("astral-sh/setup-uv@")
    ]
    assert steps, "no setup-uv step found: the guard would pass over nothing"
    loose = [
        name
        for name, step in steps
        if not re.fullmatch(r"\d+\.\d+\.\d+", str((step.get("with") or {}).get("version", "")))
    ]
    assert not loose, f"setup-uv without an exact `version:` in {loose}"


def test_every_workflow_token_reads_only() -> None:
    # S-15: without `permissions:`, the token takes the repository's default, which may write.
    loose = [
        name
        for name, workflow in _workflows()
        if workflow.get("permissions") != {"contents": "read"}
    ]
    assert not loose, f"set top-level `permissions: contents: read` in {loose}"
