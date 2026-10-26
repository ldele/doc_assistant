"""The sidecar ends with the desktop shell that started it (KI-71).

The shell passes its own process id in ``DOC_PARENT_PID``. A thread waits for that process to end
and then stops the server, so a shell that crashed or was killed — where its own exit hook never
runs — cannot leave a backend holding port 8001 with that launch's token: the next launch's window
would read through it and have every write refused (security S-6). Unset (``just api``, the
suite, the headless image), nothing is watched.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable


def parent_pid_from_env() -> int | None:
    """The shell's process id from ``DOC_PARENT_PID``; ``None`` when unset, blank or not a pid."""
    try:
        pid = int(os.environ.get("DOC_PARENT_PID", "").strip())
    except ValueError:
        return None
    return pid if pid > 0 else None


def _wait_windows(pid: int) -> None:
    """Block on the process handle itself: no polling, and no mistaking a reused pid later."""
    import ctypes
    from ctypes import wintypes

    synchronize = 0x00100000
    infinite = 0xFFFFFFFF
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel32.OpenProcess(synchronize, False, pid)
    if not handle:  # the process is already gone
        return
    try:
        kernel32.WaitForSingleObject(handle, infinite)
    finally:
        kernel32.CloseHandle(handle)


def _wait_posix(pid: int, poll: float) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:  # alive, owned by another user
            pass
        time.sleep(poll)


def wait_for_exit(pid: int, poll: float = 1.0) -> None:
    """Return once process ``pid`` has ended (at once if it already has)."""
    if sys.platform == "win32":
        _wait_windows(pid)
    else:
        _wait_posix(pid, poll)


def watch_parent(pid: int, on_gone: Callable[[], None], *, poll: float = 1.0) -> threading.Thread:
    """Call ``on_gone`` from a daemon thread once process ``pid`` has ended."""

    def run() -> None:
        wait_for_exit(pid, poll)
        on_gone()

    thread = threading.Thread(target=run, name="parent-watch", daemon=True)
    thread.start()
    return thread
