"""The release check's two lists that 0.7.2 fixed: the build inputs (review N1), and the
developer commands a user must never be told to run (KI-39, KI-83).

Offline: a fake tree under `tmp_path` stands in for the repository where a check needs planted
files, and the real repository is read where the point is that it passes today.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts import release_preflight as preflight
from scripts.release_preflight import check_dev_commands


def test_every_tracked_app_file_is_shipped_or_named_as_not() -> None:
    """A new build input cannot slip past the freshness check (review N1).

    The DOMPurify fix touched only `package-lock.json`, which was in neither list, so an installer
    built before it read as fresh. Every tracked file under `apps/` must now be in `SHIPPED_PATHS`
    or in `NOT_SHIPPED` with its reason: adding a file means deciding which."""
    assert preflight.unclassified_app_files() == []
    for rel, why in preflight.NOT_SHIPPED.items():
        assert not preflight._is_shipped(rel), f"{rel} is in both lists"
        assert why.strip(), f"{rel} needs its reason"


def _fake_tree(root: Path, files: dict[str, list[str]]) -> None:
    for rel, lines in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{line}{chr(10)}" for line in lines), encoding="utf-8")


def test_a_chat_reply_or_a_ts_string_naming_a_dev_command_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """KI-83: four chat replies told an installed-app user to run `python -m scripts.…`, and the
    check read `.svelte` files only. Docstrings, comments and tests may still explain a command."""
    _fake_tree(
        tmp_path,
        {
            "src/doc_assistant/commands.py": [
                '"""Run `python -m scripts.x` to rebuild (for maintainers)."""',
                "def reply() -> str:",
                '    """A docstring may say `python -m scripts.y`."""',
                '    return "Run `python -m scripts.extract_citations --apply`."',
            ],
            "apps/desktop/src/lib/x.ts": [
                "// npm run dev is fine in a comment",
                "export const hint = 'Try `uv run x`'",
            ],
            "apps/desktop/src/lib/x.test.ts": ["export const t = 'python -m anything'"],
            "apps/desktop/src/App.svelte": ["<p>Indexing keeps running.</p>"],
        },
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    check = check_dev_commands()
    assert check.status == preflight.FAIL
    assert sorted(note.split("  ")[0] for note in check.notes) == [
        "apps/desktop/src/lib/x.ts:2",
        "src/doc_assistant/commands.py:4",
    ]


def test_a_script_path_in_a_reply_fails_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_tree(
        tmp_path,
        {
            "src/doc_assistant/commands.py": [
                "def reply() -> str:",
                '    return "Once `scripts/extract_citations.py --apply` is run, ..."',
            ],
        },
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    assert check_dev_commands().status == preflight.FAIL


def test_clean_replies_and_strings_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tree(
        tmp_path,
        {
            "src/doc_assistant/commands.py": [
                "def reply() -> str:",
                '    return "In the Library, choose Re-run… → References."',
            ],
            "apps/desktop/src/lib/x.ts": ["export const hint = 'Settings → Documents'"],
        },
    )
    monkeypatch.setattr(preflight, "ROOT", tmp_path)
    assert check_dev_commands().status == preflight.OK
