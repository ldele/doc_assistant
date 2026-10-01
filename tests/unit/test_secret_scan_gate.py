"""Guards for the secret scan CI can fail (`scripts/secret_scan_gate.py`, security S-13).

The gate's only job is to fail, so these pin the failing directions on a scratch repository with
the real scanner: a key the baseline does not record fails; one it records passes; and a gate that
could not look (no baseline, no file list, a hook that broke) exits 2, never 0.

The planted key is the published documentation example, assembled at run time so this file holds
nothing the scanner matches. No network: the gate runs the hook with `--no-verify`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from scripts import secret_scan_gate as gate

FAKE_KEY = "AKIA" + "IOSFODNN7" + "EXAMPLE"
LEAK = f'access_key = "{FAKE_KEY}"\n'


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _scan_cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """`detect-secrets scan` — the command that writes a baseline (and that CI used to run)."""
    return subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )


def _record_baseline(root: Path) -> Path:
    baseline = root / gate.BASELINE_NAME
    baseline.write_text(_scan_cli(root).stdout, encoding="utf-8")
    _git(root, "add", gate.BASELINE_NAME)
    return baseline


def _track(root: Path, name: str, text: str) -> None:
    (root / name).write_text(text, encoding="utf-8")
    _git(root, "add", name)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A scratch repository: one clean tracked file and a baseline the scanner wrote for it."""
    _git(tmp_path, "init", "-q")
    _track(tmp_path, "clean.py", "x = 1\n")
    _record_baseline(tmp_path)
    return tmp_path


# ---- the gate, end to end -------------------------------------------------------------------


def test_a_clean_tree_passes(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gate.main(["--root", str(repo)]) == 0
    out = capsys.readouterr().out
    assert "UNRECORDED secrets: 0" in out and "-> OK" in out


def test_a_key_the_baseline_does_not_record_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _track(repo, "leak.py", LEAK)
    assert gate.main(["--root", str(repo)]) == 1
    out = capsys.readouterr().out
    assert "leak.py:1" in out and "AWS Access Key" in out and "-> FAIL" in out
    assert FAKE_KEY not in out  # the type and the place, never the value


def test_the_command_ci_used_to_run_passes_the_same_key(repo: Path) -> None:
    """Why the step was replaced: `scan --baseline` records the finding and exits 0."""
    _track(repo, "leak.py", LEAK)
    before = (repo / gate.BASELINE_NAME).read_text(encoding="utf-8")
    old = _scan_cli(repo, "--baseline", gate.BASELINE_NAME)
    assert old.returncode == 0
    assert "leak.py" in (repo / gate.BASELINE_NAME).read_text(encoding="utf-8")
    assert "leak.py" not in before


def test_a_recorded_secret_passes(repo: Path) -> None:
    _track(repo, "fixture.py", LEAK)
    baseline = _record_baseline(repo)
    assert "fixture.py" in baseline.read_text(encoding="utf-8")
    assert gate.main(["--root", str(repo)]) == 0


def test_a_moved_line_passes_and_the_baseline_is_never_rewritten(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook rewrites the baseline it is given when a recorded line moves, and exits 3. The
    gate hands it a copy: the tracked file stays byte for byte, and out of date is no failure."""
    _track(repo, "fixture.py", LEAK)
    baseline = _record_baseline(repo)
    recorded = baseline.read_bytes()
    _track(repo, "fixture.py", "# a line above\n" + LEAK)

    assert gate.main(["--root", str(repo)]) == 0
    assert "out of date" in capsys.readouterr().out
    assert baseline.read_bytes() == recorded


def test_named_files_are_scanned_instead_of_the_tracked_list(repo: Path) -> None:
    (repo / "untracked.py").write_text(LEAK, encoding="utf-8")
    assert gate.main(["--root", str(repo)]) == 0  # git does not track it
    assert gate.main(["--root", str(repo), "untracked.py"]) == 1


# ---- a gate that cannot look must not pass --------------------------------------------------


def test_no_baseline_is_a_failure_to_look(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    assert gate.main(["--root", str(tmp_path), "a.py"]) == 2
    assert "the scan did not run" in capsys.readouterr().out


def test_outside_a_repository_there_is_no_file_list(tmp_path: Path) -> None:
    (tmp_path / gate.BASELINE_NAME).write_text("{}", encoding="utf-8")
    assert gate.main(["--root", str(tmp_path)]) == 2


def _fake_hook(returncode: int, stdout: str = "", stderr: str = "") -> Any:
    def run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], returncode, stdout, stderr)

    return run


def test_a_hook_that_exits_1_without_a_report_is_not_read_as_a_finding(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The hook also returns 1 when it cannot parse its arguments. That is a scan that did not
    run (exit 2), and must not be reported as a secret somebody has to go and find."""
    monkeypatch.setattr(gate.subprocess, "run", _fake_hook(1, "Your baseline file is unstaged."))
    assert gate.main(["--root", str(repo), "clean.py"]) == 2
    assert "UNRECORDED" not in capsys.readouterr().out


def test_a_hook_that_crashes_fails_closed(repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(gate.subprocess, "run", _fake_hook(2, "", "Traceback ..."))
    assert gate.main(["--root", str(repo), "clean.py"]) == 2


def test_the_hook_runs_offline_against_a_copy_of_the_baseline(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(gate.subprocess, "run", run)
    assert gate.main(["--root", str(repo), "clean.py", gate.BASELINE_NAME]) == 0
    (command,) = seen
    assert "--no-verify" in command
    given = Path(command[command.index("--baseline") + 1])
    assert given.name == gate.BASELINE_NAME and given.parent != repo
    assert command[-1] == "clean.py"  # the baseline itself is not scanned as a source file


def test_no_verify_removes_the_network_check_even_with_a_baseline(repo: Path) -> None:
    """ "No network call" rests on the hook honouring `--no-verify` when the baseline it is given
    lists the verification filter itself. Read from the hook's own argument parser, so an upgrade
    that changes this fails here and not silently in CI."""
    from detect_secrets.core.usage import ParserBuilder
    from detect_secrets.settings import default_settings, get_settings

    verification = "detect_secrets.filters.common.is_ignored_due_to_verification_policies"
    baseline = str(repo / gate.BASELINE_NAME)
    with default_settings():
        ParserBuilder().add_pre_commit_arguments().parse_args(["--baseline", baseline])
        assert verification in get_settings().filters
    with default_settings():
        ParserBuilder().add_pre_commit_arguments().parse_args(
            ["--no-verify", "--baseline", baseline]
        )
        assert verification not in get_settings().filters


# ---- pure helpers ---------------------------------------------------------------------------


def test_batches_keep_every_file_in_order_under_the_budget() -> None:
    files = [f"dir/file_{n:03d}.py" for n in range(100)]
    split = gate.batches(files, max_chars=200)
    assert [name for batch in split for name in batch] == files
    assert len(split) > 1 and all(batch for batch in split)
    assert all(sum(len(name) + 1 for name in batch) <= 200 for batch in split)


def test_a_path_longer_than_the_budget_is_scanned_alone_not_dropped() -> None:
    long = "x" * 50
    assert gate.batches(["a.py", long, "b.py"], max_chars=10) == [["a.py"], [long], ["b.py"]]


def test_no_files_is_no_batch() -> None:
    assert gate.batches([]) == []


def test_this_repository_needs_more_than_one_command_line_on_windows() -> None:
    """The reason batching exists: the tracked paths alone pass what one Windows command line
    holds, so a single call would fail on the machine the gate is written on."""
    files = gate.tracked_files(gate.ROOT)
    assert files, "git lists no tracked file"
    assert all(sum(len(n) + 1 for n in b) <= gate.MAX_BATCH_CHARS for b in gate.batches(files))


def test_findings_are_read_from_the_report_and_carry_no_value() -> None:
    # The hook's report names each hit's hash under a key the scanner itself reads as a keyword
    # when a value follows it. The key is assembled away from its value, so this file stays clean
    # under its own gate.
    hash_field = "hashed_" + "secret"
    hit = {
        "type": "AWS Access Key",
        "filename": "a.py",
        hash_field: "0123abcd",
        "is_verified": False,
        "line_number": 4,
    }
    report = json.dumps({"results": {"a.py": [hit]}})
    assert gate.parse_findings(report) == [gate.Finding("AWS Access Key", "a.py", 4)]
    assert "0123abcd" not in gate.render(gate.Verdict(1, gate.parse_findings(report), False))


@pytest.mark.parametrize("report", ["", "not json", "{}", '{"results": {}}', '{"results": []}'])
def test_anything_that_is_not_a_report_of_findings_is_a_scan_error(report: str) -> None:
    with pytest.raises(gate.ScanError):
        gate.parse_findings(report)


# ---- the wiring -----------------------------------------------------------------------------


def test_ci_runs_the_gate_and_not_the_command_that_writes_a_baseline() -> None:
    workflow = (gate.ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    commands = [line.strip() for line in workflow.splitlines() if line.strip().startswith("run:")]
    assert any("scripts.secret_scan_gate" in command for command in commands)
    assert not any("detect-secrets scan" in command for command in commands)


def test_the_baseline_the_gate_reads_is_tracked() -> None:
    assert gate.BASELINE_NAME in gate.tracked_files(gate.ROOT)
