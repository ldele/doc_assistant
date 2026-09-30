"""Guards for the blocking dependency-advisory gate (`scripts/pip_audit_gate.py`, security S-8).

The gate's only job is to fail, so these tests pin the failing directions: an advisory nobody
reviewed, an ignore that lacks a reason, and a gate that cannot see (no report, a broken ignore
file) must never read as a pass. All offline — reports are fixtures, never a live pip-audit run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from scripts import pip_audit_gate as gate

WHY_LINE = 'why = "Server-side only; the app embeds it."'
ENTRY = f"""
[[ignore]]
id = "PYSEC-2026-1"
package = "Examplepkg"
reviewed = "2026-09-30"
{WHY_LINE}
reverse_if = "Anything starts its server."
"""


def _report(*deps: dict[str, Any]) -> dict[str, Any]:
    return {"dependencies": list(deps), "fixes": []}


def _dep(name: str, version: str, *vulns: dict[str, Any]) -> dict[str, Any]:
    return {"name": name, "version": version, "vulns": list(vulns)}


def _vuln(vid: str, aliases: tuple[str, ...] = (), fix: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"id": vid, "aliases": list(aliases), "fix_versions": list(fix), "description": "x"}


def test_the_committed_ignore_file_is_complete_and_unique() -> None:
    ignores = gate.load_ignores(gate.IGNORE_FILE.read_text(encoding="utf-8"))
    assert ignores, "the ignore file parsed to nothing"
    assert len({ig.id for ig in ignores}) == len(ignores)
    for ig in ignores:  # load_ignores refuses empty fields; this states the contract
        assert all(getattr(ig, f) for f in gate.FIELDS), ig.id


@pytest.mark.parametrize("field", gate.FIELDS)
def test_an_entry_missing_any_field_is_refused(field: str) -> None:
    text = "\n".join(ln for ln in ENTRY.splitlines() if not ln.startswith(f"{field} ="))
    with pytest.raises(ValueError, match=f"missing {field}"):
        gate.load_ignores(text)


def test_an_empty_reason_is_refused() -> None:
    with pytest.raises(ValueError, match="missing why"):
        gate.load_ignores(ENTRY.replace(WHY_LINE, 'why = "  "'))


def test_a_misspelt_field_is_refused_not_ignored() -> None:
    text = ENTRY.replace("reverse_if", "reverse-if")
    with pytest.raises(ValueError, match="unknown field reverse-if"):
        gate.load_ignores(text)


def test_a_duplicate_id_is_refused() -> None:
    with pytest.raises(ValueError, match="duplicate id"):
        gate.load_ignores(ENTRY + ENTRY)


def test_an_unreviewed_advisory_fails() -> None:
    ignores = gate.load_ignores(ENTRY)
    report = _report(_dep("pillow", "12.2.0", _vuln("PYSEC-2026-9", fix=("12.3.0",))))
    verdict = gate.evaluate(report, ignores)
    assert [f.id for f in verdict.unreviewed] == ["PYSEC-2026-9"]
    assert verdict.stale == ignores  # the one ignore matched nothing here


def test_a_reviewed_advisory_matches_by_alias_and_normalised_name() -> None:
    ignores = gate.load_ignores(ENTRY)
    report = _report(_dep("examplepkg", "1.0", _vuln("GHSA-aaaa", aliases=("PYSEC-2026-1",))))
    verdict = gate.evaluate(report, ignores)
    assert not verdict.unreviewed
    assert [(f.id, ig.id) for f, ig in verdict.ignored] == [("GHSA-aaaa", "PYSEC-2026-1")]
    assert not verdict.stale


def test_an_ignore_never_covers_another_package() -> None:
    ignores = gate.load_ignores(ENTRY)
    report = _report(_dep("otherpkg", "1.0", _vuln("PYSEC-2026-1")))
    assert [f.package for f in gate.evaluate(report, ignores).unreviewed] == ["otherpkg"]


def test_one_advisory_reported_twice_counts_once() -> None:
    report = _report(_dep("pkg", "1.0", _vuln("PYSEC-2026-7"), _vuln("PYSEC-2026-7")))
    assert len(gate.evaluate(report, []).unreviewed) == 1


def test_skipped_dependencies_are_listed_not_audited() -> None:
    report = _report(
        {"name": "torch", "version": "2.12.0+cpu", "skip_reason": "not on PyPI"},
        _dep("pkg", "1.0"),
    )
    verdict = gate.evaluate(report, [])
    assert verdict.audited == 1
    assert verdict.skipped == [("torch", "not on PyPI")]


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_main_exit_codes(tmp_path: Path) -> None:
    ignore_file = _write(tmp_path, "ignore.toml", ENTRY)
    clean = _write(tmp_path, "clean.json", json.dumps(_report(_dep("pkg", "1.0"))))
    dirty = _write(tmp_path, "dirty.json", json.dumps(_report(_dep("pkg", "1.0", _vuln("X-1")))))
    ignore_args = ["--ignore-file", str(ignore_file)]
    assert gate.main(["--report", str(clean), *ignore_args]) == 0  # a stale ignore is not fatal
    assert gate.main(["--report", str(dirty), *ignore_args]) == 1


def test_a_broken_ignore_file_fails_closed(tmp_path: Path) -> None:
    report = _write(tmp_path, "clean.json", json.dumps(_report(_dep("pkg", "1.0"))))
    broken = _write(tmp_path, "ignore.toml", ENTRY.replace(WHY_LINE, ""))
    absent = tmp_path / "absent.toml"
    assert gate.main(["--report", str(report), "--ignore-file", str(broken)]) == 2
    assert gate.main(["--report", str(report), "--ignore-file", str(absent)]) == 2


def test_no_report_from_pip_audit_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="network down")

    monkeypatch.setattr(gate.subprocess, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        gate.run_pip_audit()
    assert exc.value.code == 2
