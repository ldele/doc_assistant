"""Security step S-13 — a secret scan that can fail.

    uv run --no-sync python -m scripts.secret_scan_gate            # every tracked file
    uv run --no-sync python -m scripts.secret_scan_gate a.py b.md  # the named files

Scans the tracked files with the `detect-secrets` pre-commit hook and **fails on a secret that
`.secrets.baseline` does not record**. Until 2026-10-01 CI ran `detect-secrets scan --baseline
.secrets.baseline`, which is the command that *writes* a baseline: it rescans, records a new
finding in the file and exits 0, so a key pushed past the local hook turned the check green
(`docs/security.md` S13) — a check that cannot fail is not a control. The hook is the command that
compares, and it is the one pre-commit already runs on a commit.

Three differences from the hook as pre-commit runs it, all on purpose:

* **It never rewrites the baseline.** When recorded line numbers have moved, the hook rewrites the
  file it was given and exits 3. Here it is given a temporary copy, and a baseline that is only out
  of date is reported and passes: a gate that edits a tracked file is a second source of diffs.
* **It makes no network call** (`--no-verify`). The hook can ask a provider whether a candidate key
  is live and drop the ones that are not; a gate that fails closed keeps them, and stays offline.
* **It reads every file as UTF-8** (`python -X utf8`). The scanner opens a file in the platform's
  default encoding and treats a decode error as "binary, skip". On Windows that default is cp1252,
  and a UTF-8 file with one byte cp1252 does not define — a curly closing quote is enough — is
  skipped whole: 47 of this repository's 838 tracked files on 2026-10-01, source files among
  them. The first CI run of this gate (Linux, UTF-8) reported three findings the Windows run had
  never seen. UTF-8 mode makes the two platforms scan the same text.

Exit codes: 0 — nothing unrecorded · 1 — an unrecorded secret · 2 — the scan could not run (no
baseline, no file list, or the hook failed). The last fails closed: a gate that cannot look must
not pass.

A finding is printed as its type and location, never its value.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
BASELINE_NAME = ".secrets.baseline"

#: Windows refuses a command line past 32,767 characters, and the tracked paths of this repository
#: already add up to more than that. Well under the limit, so the interpreter path and the options
#: fit beside the file names.
MAX_BATCH_CHARS = 20_000

# The hook's exit codes (`detect_secrets.pre_commit_hook.main`). 1 is also what it returns when it
# cannot parse its arguments, which is why a finding is read from the JSON report and never from
# the code alone.
HOOK_CLEAN = 0
HOOK_FOUND = 1
HOOK_BASELINE_REWRITTEN = 3

#: Interpreter flags for the scanner's process: UTF-8 mode, so `open()` without an encoding reads
#: UTF-8 on every platform. Without it the scanner skips, on Windows, each file cp1252 cannot
#: decode — silently, as if it were binary (see the module docstring).
UTF8_MODE = ("-X", "utf8")


@dataclass(frozen=True)
class Finding:
    """One secret the baseline does not record: what kind, and where. Never the value."""

    type: str
    filename: str
    line: int


@dataclass(frozen=True)
class Verdict:
    scanned: int
    findings: list[Finding]
    baseline_out_of_date: bool


class ScanError(Exception):
    """The scan could not run. The gate exits 2: it did not look, so it must not pass."""


def batches(files: Sequence[str], max_chars: int = MAX_BATCH_CHARS) -> list[list[str]]:
    """Split the file list so no one command line outgrows ``max_chars`` (pure).

    Order is kept and no batch is empty. A path longer than the budget gets a batch of its own
    rather than being dropped: a file left out is a file not scanned."""
    out: list[list[str]] = []
    current: list[str] = []
    size = 0
    for name in files:
        cost = len(name) + 1
        if current and size + cost > max_chars:
            out.append(current)
            current, size = [], 0
        current.append(name)
        size += cost
    if current:
        out.append(current)
    return out


def parse_findings(report: str) -> list[Finding]:
    """The findings in the hook's ``--json`` report; ``ScanError`` when it is not such a report.

    The hook exits 1 both for a finding and for arguments it could not parse. Only a report that
    lists results counts as a finding, so a hook that broke is never read as one that looked."""
    try:
        results = json.loads(report)["results"]
        found = [
            Finding(
                type=str(hit["type"]),
                # The scanner reports in the local separator; one form, so a Windows run and
                # CI's name the same file the same way.
                filename=str(filename).replace("\\", "/"),
                line=int(hit["line_number"]),
            )
            for filename, hits in results.items()
            for hit in hits
        ]
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ScanError("detect-secrets-hook exited 1 without a report of findings") from exc
    if not found:
        raise ScanError("detect-secrets-hook exited 1 and its report lists no finding")
    return found


def tracked_files(root: Path) -> list[str]:
    """Every file git tracks under ``root``. ``ScanError`` when git cannot say."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise ScanError(f"could not run git: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise ScanError(f"`git ls-files` failed (exit {proc.returncode}): {detail}")
    return [name for name in proc.stdout.decode("utf-8").split("\0") if name]


def scan(files: Sequence[str], *, root: Path, baseline: Path) -> Verdict:
    """Run the hook over ``files`` (paths relative to ``root``) against a copy of ``baseline``."""
    if not baseline.is_file():
        raise ScanError(f"no baseline at {baseline}")
    # The baseline holds hashes that read as high-entropy strings; the hook skips the file it was
    # given, and it is given a copy, so the tracked one is left out by name.
    names = [name for name in files if Path(name).name != baseline.name]
    findings: list[Finding] = []
    out_of_date = False
    with tempfile.TemporaryDirectory() as scratch:
        copy = Path(scratch) / baseline.name
        shutil.copyfile(baseline, copy)
        for batch in batches(names):
            proc = subprocess.run(
                [
                    sys.executable,
                    *UTF8_MODE,
                    "-m",
                    "detect_secrets.pre_commit_hook",
                    "--no-verify",
                    "--json",
                    "--baseline",
                    str(copy),
                    *batch,
                ],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if proc.returncode == HOOK_FOUND:
                try:
                    findings.extend(parse_findings(proc.stdout))
                except ScanError as exc:
                    detail = (proc.stderr or proc.stdout).strip()
                    raise ScanError(f"{exc}: {detail}") from exc
            elif proc.returncode == HOOK_BASELINE_REWRITTEN:
                out_of_date = True
            elif proc.returncode != HOOK_CLEAN:
                raise ScanError(
                    f"detect-secrets-hook failed (exit {proc.returncode}): {proc.stderr.strip()}"
                )
    return Verdict(scanned=len(names), findings=findings, baseline_out_of_date=out_of_date)


def render(verdict: Verdict, baseline_name: str = BASELINE_NAME) -> str:
    lines = [f"secret scan gate: {verdict.scanned} files scanned against {baseline_name}"]
    if verdict.baseline_out_of_date:
        lines.append(
            f"  {baseline_name} is out of date (a recorded line moved, or an entry no longer "
            "matches). Not a failure; `pre-commit run detect-secrets --all-files` refreshes it."
        )
    lines.append(f"  UNRECORDED secrets: {len(verdict.findings)}")
    for f in sorted(verdict.findings, key=lambda f: (f.filename, f.line, f.type)):
        lines.append(f"    {f.filename}:{f.line}  {f.type}")
    if verdict.findings:
        lines.append(
            "-> FAIL: remove the secret and rotate it. A false positive is recorded on its line "
            "(`# pragma: allowlist secret`) or in the baseline, by the pre-commit hook."
        )
    else:
        lines.append("-> OK")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument("files", nargs="*", help="files to scan (default: every file git tracks)")
    ap.add_argument("--root", type=Path, default=ROOT, help="the repository to scan")
    ap.add_argument("--baseline", type=Path, help=f"default: <root>/{BASELINE_NAME}")
    args = ap.parse_args(argv)
    root: Path = args.root
    baseline: Path = args.baseline or root / BASELINE_NAME
    try:
        files = list(args.files) or tracked_files(root)
        verdict = scan(files, root=root, baseline=baseline)
    except ScanError as exc:
        print(f"secret scan gate: {exc}")
        print("-> FAIL: the scan did not run, so nothing was checked")
        return 2
    print(render(verdict, baseline.name))
    return 1 if verdict.findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
