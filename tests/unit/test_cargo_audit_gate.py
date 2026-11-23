"""Guards for the blocking Rust-tree gate (`scripts/cargo_audit_gate.py`, security S-16).

The gate's job is to fail, so these pin the failing directions: a vulnerability or a warning nobody
reviewed, a review for another crate, and a gate that cannot see (no report, a broken ignore file)
must never read as a pass. All offline: reports are fixtures, never a live cargo audit run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scripts import cargo_audit_gate as gate
from scripts.pip_audit_gate import load_ignores

ROOT = Path(__file__).resolve().parents[2]

ENTRY = """
[[ignore]]
id = "RUSTSEC-2024-0429"
package = "glib"
reviewed = "2026-11-23"
why = "Linux only; the installer is Windows."
reverse_if = "The app ships for Linux."
"""


def _entry(
    crate: str, version: str, rid: str | None, patched: tuple[str, ...] = ()
) -> dict[str, Any]:
    advisory = None if rid is None else {"id": rid, "aliases": [], "title": "x"}
    return {
        "advisory": advisory,
        "package": {"name": crate, "version": version},
        "versions": {"patched": list(patched)},
    }


def _report(
    vulns: list[dict[str, Any]] | None = None,
    warnings: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    vulns = vulns or []
    return {
        "lockfile": {"dependency-count": 453},
        "vulnerabilities": {"found": bool(vulns), "count": len(vulns), "list": vulns},
        "warnings": warnings or {},
    }


def test_a_clean_report_passes() -> None:
    v = gate.evaluate(_report(), [])
    assert v.unreviewed == [] and v.crates == 453


def test_an_unreviewed_vulnerability_fails() -> None:
    v = gate.evaluate(_report([_entry("url", "2.5.8", "RUSTSEC-2026-0592", (">=2.5.9",))]), [])
    assert [(f.package, f.kind, f.patched) for f in v.unreviewed] == [
        ("url", "vulnerability", (">=2.5.9",))
    ]


def test_a_warning_is_a_finding_too() -> None:
    # cargo audit exits 0 on warnings; the gate must not.
    report = _report(warnings={"unsound": [_entry("glib", "0.18.5", "RUSTSEC-2024-0429")]})
    v = gate.evaluate(report, [])
    assert [(f.package, f.kind) for f in v.unreviewed] == [("glib", "unsound")]


def test_a_yanked_crate_without_an_advisory_is_a_finding_named_yanked() -> None:
    v = gate.evaluate(_report(warnings={"yanked": [_entry("foo", "1.0.0", None)]}), [])
    assert [(f.package, f.id) for f in v.unreviewed] == [("foo", "yanked")]


def test_a_reviewed_finding_passes_and_is_listed() -> None:
    report = _report(warnings={"unsound": [_entry("glib", "0.18.5", "RUSTSEC-2024-0429")]})
    v = gate.evaluate(report, load_ignores(ENTRY))
    assert v.unreviewed == [] and len(v.ignored) == 1 and v.stale == []


def test_a_review_covers_only_its_own_crate() -> None:
    report = _report([_entry("gtk", "0.18.2", "RUSTSEC-2024-0429")])
    v = gate.evaluate(report, load_ignores(ENTRY))
    assert [f.package for f in v.unreviewed] == ["gtk"]
    assert [ig.package for ig in v.stale] == ["glib"]


def test_a_finding_reported_twice_counts_once() -> None:
    e = _entry("url", "2.5.8", "RUSTSEC-2026-0592")
    assert len(gate.evaluate(_report([e, e]), []).unreviewed) == 1


def test_main_exit_codes(tmp_path: Path) -> None:
    ignores = tmp_path / "ignore.toml"
    ignores.write_text(ENTRY, encoding="utf-8")
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_report([_entry("url", "2.5.8", "RUSTSEC-2026-0592")])), "utf-8")
    assert gate.main(["--report", str(report), "--ignore-file", str(ignores)]) == 1
    report.write_text(json.dumps(_report()), encoding="utf-8")
    assert gate.main(["--report", str(report), "--ignore-file", str(ignores)]) == 0


def test_a_broken_ignore_file_fails_closed(tmp_path: Path) -> None:
    ignores = tmp_path / "ignore.toml"
    ignores.write_text(ENTRY.replace('reverse_if = "The app ships for Linux."', ""), "utf-8")
    report = tmp_path / "r.json"
    report.write_text(json.dumps(_report()), encoding="utf-8")
    assert gate.main(["--report", str(report), "--ignore-file", str(ignores)]) == 2


def test_no_report_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, stdout="", stderr="error: no lockfile")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        gate.run_cargo_audit(Path("Cargo.lock"))
    assert exc.value.code == 2


def test_no_cargo_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("cargo")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        gate.run_cargo_audit(Path("Cargo.lock"))
    assert exc.value.code == 2


def test_the_reviewed_list_parses_and_every_entry_names_a_crate_in_the_lock() -> None:
    # A typo in `package` would make an entry match nothing and read as stale, not as an error.
    entries = load_ignores((ROOT / "cargo-audit-ignore.toml").read_text(encoding="utf-8"))
    lock = (ROOT / "apps" / "desktop" / "src-tauri" / "Cargo.lock").read_text(encoding="utf-8")
    assert entries
    for ig in entries:
        assert f'name = "{ig.package}"' in lock, ig.package


def test_ci_runs_the_gate_on_a_pinned_cargo_audit() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "python3 -m scripts.cargo_audit_gate" in ci
    assert "cargo install cargo-audit --locked --version" in ci
