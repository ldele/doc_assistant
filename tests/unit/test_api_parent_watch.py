"""KI-71: the desktop backend ends with the shell that started it.

Closing Provenote left its backend running, holding port 8001 with that launch's token, so the
next launch's window read through it and had every write refused (S-6). The shell now ends the
backend's process tree on exit (``lib.rs``, pinned below) and passes its own process id in
``DOC_PARENT_PID``; the backend stops by itself once that process is gone, which covers a crash or
a forced close where the shell's exit hook never runs. These tests watch a real process end.
"""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest
from apps.api import parent_watch
from apps.api.parent_watch import parent_pid_from_env, watch_parent

LIB_RS = Path(__file__).resolve().parents[2] / "apps" / "desktop" / "src-tauri" / "src" / "lib.rs"


@pytest.mark.parametrize("raw", [None, "", "  ", "abc", "0", "-5"])
def test_no_usable_pid_means_nothing_is_watched(monkeypatch: pytest.MonkeyPatch, raw) -> None:
    """``just api``, the suite and the headless image set none: the server runs as before."""
    if raw is None:
        monkeypatch.delenv("DOC_PARENT_PID", raising=False)
    else:
        monkeypatch.setenv("DOC_PARENT_PID", raw)
    assert parent_pid_from_env() is None


def test_the_pid_is_read_trimmed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DOC_PARENT_PID", " 4321\n")
    assert parent_pid_from_env() == 4321


def test_the_watcher_fires_when_the_process_ends_and_not_before() -> None:
    # A stand-in shell: it lives until its stdin closes.
    shell = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    gone = threading.Event()
    try:
        watch_parent(shell.pid, gone.set, poll=0.05)
        assert not gone.wait(0.5), "fired while the process was still running"
        assert shell.stdin is not None
        shell.stdin.close()
        shell.wait(timeout=10)
        assert gone.wait(10), "did not fire after the process ended"
    finally:
        if shell.poll() is None:
            shell.kill()


def test_main_watches_the_shell_only_when_one_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    from apps.api import __main__ as entry

    watched: list[int] = []
    monkeypatch.setattr(entry, "watch_parent", lambda pid, on_gone: watched.append(pid))
    monkeypatch.setattr(entry.uvicorn.Server, "run", lambda self, sockets=None: None)
    monkeypatch.delenv("DOC_PARENT_PID", raising=False)
    entry.main()
    assert watched == []
    monkeypatch.setenv("DOC_PARENT_PID", "4321")
    entry.main()
    assert watched == [4321]


def test_the_shell_ends_the_sidecar_tree_on_exit_without_a_console_window() -> None:
    """The Rust half has no test harness here, so its contract is pinned in its source: keep
    the child, pass this process's id, end the tree on exit, never show a console window."""
    src = LIB_RS.read_text(encoding="utf-8")
    assert '.env("DOC_PARENT_PID", std::process::id().to_string())' in src
    assert "app.manage(Sidecar(Mutex::new(Some(child))))" in src
    assert "if let RunEvent::Exit = event" in src and "end_sidecar(child)" in src
    assert '["/PID", &child.pid().to_string(), "/T", "/F"]' in src
    assert ".creation_flags(CREATE_NO_WINDOW)" in src
    assert parent_watch.__doc__ is not None and "KI-71" in parent_watch.__doc__
