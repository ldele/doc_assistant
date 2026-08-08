"""Guard tests for KI-42 — the Marker escape hatch is pinned, and its failures are legible.

Two things went wrong at once on 2026-08-08. `uvx --from marker-pdf` was **unpinned**, so
marker-pdf 2.0.0 (surya routes inference through a `docker run` or `llama-server` backend)
silently replaced a working 1.x and killed `extract_tables_marker` on a box with neither. And the
outage was **unreadable**: the availability guard only checked that `uvx` existed, every failure
was swallowed into a per-document row, the note was truncated to 30 characters mid-cause, and
error rows sorted *below* every success in a 97-row report.

These pin the fix: the version is explicit, a non-zero exit is a distinct systemic error that is
not swallowed, and the report puts errors first with their cause intact.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
from scripts.eval_marker_tables import (
    MarkerUnavailableError,
    _marker_command,
    _marker_to_markdown,
)
from scripts.extract_tables_marker import _format_report

from doc_assistant.config import MARKER_VERSION

# ---- the pin ---------------------------------------------------------------


def test_uvx_invocation_pins_the_marker_version(monkeypatch: pytest.MonkeyPatch) -> None:
    # No `marker_single` on PATH -> the uvx path, which must name a version. Unpinned,
    # `uvx` re-resolves to latest on every call and the table path drifts with upstream.
    monkeypatch.setattr(
        "scripts.eval_marker_tables.shutil.which",
        lambda name: None if name == "marker_single" else f"/usr/bin/{name}",
    )
    cmd = _marker_command()
    assert cmd is not None
    assert f"marker-pdf=={MARKER_VERSION}" in cmd
    assert "marker-pdf" not in cmd, "bare 'marker-pdf' means the version is unpinned"


def test_pinned_version_is_a_concrete_release() -> None:
    # A range or a floating spec would reintroduce the drift this pin exists to stop.
    assert MARKER_VERSION[0].isdigit()
    assert not any(ch in MARKER_VERSION for ch in "<>*^~ ")


def test_local_marker_on_path_is_used_as_is(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "scripts.eval_marker_tables.shutil.which", lambda name: "/opt/marker_single"
    )
    assert _marker_command() == ["/opt/marker_single"]


def test_no_launcher_at_all_is_a_systemic_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setattr("scripts.eval_marker_tables.shutil.which", lambda name: None)
    assert _marker_command() is None
    with pytest.raises(MarkerUnavailableError, match="neither"):
        _marker_to_markdown(tmp_path / "x.pdf", [1], tmp_path)


# ---- a non-zero exit is systemic, not a per-document quirk -----------------


def test_nonzero_exit_raises_marker_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # The load-bearing distinction: a PDF with no tables still exits 0, so a non-zero
    # exit is the tool failing. It must be the type the runner refuses to swallow.
    monkeypatch.setattr(
        "scripts.eval_marker_tables.shutil.which", lambda name: "/opt/marker_single"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1, "", "")
    )
    with pytest.raises(MarkerUnavailableError, match="exited with code 1"):
        _marker_to_markdown(tmp_path / "x.pdf", [1], tmp_path)


def test_marker_unavailable_is_a_runtime_error_subclass() -> None:
    # Callers that already handle RuntimeError keep working; only the runner needs to
    # know the difference.
    assert issubclass(MarkerUnavailableError, RuntimeError)


def test_produced_no_markdown_is_NOT_systemic(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    # Exit 0 but nothing written is a per-document outcome — it must stay swallowable,
    # or one odd PDF would abort a 97-document run.
    monkeypatch.setattr(
        "scripts.eval_marker_tables.shutil.which", lambda name: "/opt/marker_single"
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 0, "", "")
    )
    with pytest.raises(RuntimeError) as excinfo:
        _marker_to_markdown(tmp_path / "x.pdf", [1], tmp_path)
    assert not isinstance(excinfo.value, MarkerUnavailableError)


# ---- the report makes a failure visible ------------------------------------


def _row(fn: str, status: str, tables: int, note: str = "") -> dict[str, object]:
    return {"filename": fn, "status": status, "tables": tables, "note": note}


def test_errors_sort_above_successes() -> None:
    rows = [
        _row("a.pdf", "ok", 7),
        _row("b.pdf", "error", 0, "RuntimeError: boom"),
        _row("c.pdf", "ok", 3),
    ]
    body = _format_report(rows).splitlines()
    order = [ln.split()[0] for ln in body if ln.endswith(".pdf") or ".pdf" in ln.split()[0]]
    assert order.index("b.pdf") < order.index("a.pdf"), "an error must not sort below a success"


def test_error_notes_are_not_truncated() -> None:
    cause = (
        "MarkerUnavailableError: marker_single exited with code 1 — its traceback is in "
        "the output above, check that the pinned marker-pdf still resolves"
    )
    report = _format_report([_row("a.pdf", "error", 0, cause)])
    assert cause in report, "the cause was cut off mid-word by the 30-char note clamp"


def test_non_error_notes_stay_clamped() -> None:
    # The clamp is still right for routine notes — it only hid causes.
    report = _format_report([_row("a.pdf", "skipped", 0, "x" * 80)])
    assert "x" * 80 not in report
