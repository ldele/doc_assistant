"""Guard the window's frame: the app's own title bar on Windows, the native one elsewhere.

ADR-065. Tauri merges ``tauri.windows.conf.json`` over ``tauri.conf.json`` on Windows as a JSON
merge patch, and a merge patch replaces an array whole: the Windows file has to restate the main
window to change one key of it. These tests keep the two copies the same window, so a size or a
minimum changed in one file cannot silently differ on Windows, and keep the Windows file to that
one change.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TAURI = ROOT / "apps" / "desktop" / "src-tauri"


def _load(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((TAURI / name).read_text(encoding="utf-8"))
    return data


def test_windows_restates_the_main_window_with_no_native_frame() -> None:
    [main] = _load("tauri.conf.json")["app"]["windows"]
    [windows] = _load("tauri.windows.conf.json")["app"]["windows"]
    assert windows == {**main, "decorations": False, "shadow": True}


def test_other_platforms_keep_the_native_frame() -> None:
    [main] = _load("tauri.conf.json")["app"]["windows"]
    assert main.get("decorations", True) is True


def test_the_windows_file_changes_nothing_else() -> None:
    conf = _load("tauri.windows.conf.json")
    assert set(conf) <= {"$schema", "app"}
    assert set(conf["app"]) == {"windows"}
