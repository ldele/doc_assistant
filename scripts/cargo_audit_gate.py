"""Security step S-16 — `cargo audit` over the desktop shell's Rust tree, blocking, reviewed list.

    python -m scripts.cargo_audit_gate                  # audit apps/desktop/src-tauri/Cargo.lock
    python -m scripts.cargo_audit_gate --report r.json  # judge a saved `cargo audit --json` report

Runs `cargo audit --json` over the lockfile and **fails on any finding that is not reviewed in
`cargo-audit-ignore.toml`**: a vulnerability, and also a warning (unmaintained, unsound, yanked,
notice). `cargo audit` itself exits 0 on warnings, and the Rust tree went into every installer
with no check at all until 2026-11-23 (`docs/security.md` S16), so a warning is a finding here
too: it is cleared by an update, or written down with why it does not apply and what would
reverse that.

The ignore file has the same fields and the same parser as `pip-audit-ignore.toml` (S-8). A yanked
crate has no advisory id; its entry uses the id `yanked`.

Exit codes: 0 — nothing unreviewed · 1 — an unreviewed finding · 2 — the ignore file is malformed,
or cargo audit produced no report. The last two fail closed: a gate that cannot look must not
pass. A **stale** ignore (listed, but no longer reported) is printed and never fatal; delete it.

Standard library only (it runs in CI's Rust job, which installs no Python packages).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

from scripts.pip_audit_gate import Ignore, load_ignores

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / "cargo-audit-ignore.toml"
LOCKFILE = ROOT / "apps" / "desktop" / "src-tauri" / "Cargo.lock"


@dataclass(frozen=True)
class Finding:
    """One advisory or warning reported against one locked crate."""

    kind: str  # "vulnerability", or the warning kind: "unmaintained", "unsound", "yanked", ...
    package: str
    version: str
    id: str
    aliases: tuple[str, ...]
    patched: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    crates: int
    unreviewed: list[Finding]
    ignored: list[tuple[Finding, Ignore]]
    stale: list[Ignore]


def _finding(kind: str, entry: dict[str, Any]) -> Finding:
    advisory = entry.get("advisory") or {}
    package = entry.get("package") or {}
    versions = entry.get("versions") or {}
    return Finding(
        kind=kind,
        package=str(package.get("name", "?")),
        version=str(package.get("version", "?")),
        id=str(advisory.get("id") or kind),
        aliases=tuple(advisory.get("aliases") or ()),
        patched=tuple(versions.get("patched") or ()),
    )


def findings(report: dict[str, Any]) -> tuple[list[Finding], int]:
    """Every vulnerability and warning in the report, deduplicated, and the number of crates."""
    found: dict[tuple[str, str, str], Finding] = {}
    for entry in (report.get("vulnerabilities") or {}).get("list", []):
        f = _finding("vulnerability", entry)
        found.setdefault((f.package, f.version, f.id), f)
    for kind, entries in (report.get("warnings") or {}).items():
        for entry in entries:
            f = _finding(kind, entry)
            found.setdefault((f.package, f.version, f.id), f)
    crates = int((report.get("lockfile") or {}).get("dependency-count", 0))
    return list(found.values()), crates


def evaluate(report: dict[str, Any], ignores: list[Ignore]) -> Verdict:
    """Match each finding to a reviewed ignore by id or any alias, for its own crate only."""
    found, crates = findings(report)
    by_key = {(ig.package, ig.id): ig for ig in ignores}
    used: set[tuple[str, str]] = set()
    unreviewed: list[Finding] = []
    ignored: list[tuple[Finding, Ignore]] = []
    for f in found:
        match = next(
            (by_key[(f.package, k)] for k in (f.id, *f.aliases) if (f.package, k) in by_key), None
        )
        if match is None:
            unreviewed.append(f)
        else:
            ignored.append((f, match))
            used.add((match.package, match.id))
    stale = [ig for ig in ignores if (ig.package, ig.id) not in used]
    return Verdict(crates, unreviewed, ignored, stale)


def run_cargo_audit(lockfile: Path) -> dict[str, Any]:
    """`cargo audit --json` over the lockfile; exit 2 when it produced no report."""
    try:
        proc = subprocess.run(
            ["cargo", "audit", "--json", "--file", str(lockfile)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError as exc:
        print("cargo-audit gate: `cargo` is not on PATH (install cargo-audit)")
        raise SystemExit(2) from exc
    try:
        report: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(proc.stderr)
        print(f"cargo-audit gate: cargo audit produced no JSON report (exit {proc.returncode})")
        raise SystemExit(2) from exc
    return report


def render(v: Verdict) -> str:
    lines = [f"cargo-audit gate: {v.crates} crates in the lockfile"]
    lines.append(f"  reviewed ignores that matched: {len(v.ignored)}")
    for f, ig in sorted(v.ignored, key=lambda p: (p[0].package, p[0].id)):
        lines.append(f"    {f.package} {f.version}  {f.id} ({f.kind})  (reviewed {ig.reviewed})")
    lines.append(f"  stale ignores (no longer reported): {len(v.stale)}")
    for ig in v.stale:
        lines.append(f"    {ig.package}  {ig.id}")
    lines.append(f"  UNREVIEWED findings: {len(v.unreviewed)}")
    for f in sorted(v.unreviewed, key=lambda f: (f.package, f.id)):
        fix = ", ".join(f.patched) or "no fixed release"
        lines.append(f"    {f.package} {f.version}  {f.id} ({f.kind})  fix: {fix}")
    if v.unreviewed:
        lines.append("-> FAIL: `cargo update -p <crate>`, or review it in cargo-audit-ignore.toml")
    else:
        lines.append("-> OK")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument(
        "--report", type=Path, help="judge a saved `cargo audit --json` report instead of a run"
    )
    ap.add_argument("--ignore-file", type=Path, default=IGNORE_FILE)
    ap.add_argument("--lockfile", type=Path, default=LOCKFILE)
    args = ap.parse_args(argv)
    try:
        ignores = load_ignores(args.ignore_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"cargo-audit gate: cannot use {args.ignore_file.name}: {exc}")
        return 2
    if args.report:
        report: dict[str, Any] = json.loads(args.report.read_text(encoding="utf-8-sig"))
    else:
        report = run_cargo_audit(args.lockfile)
    verdict = evaluate(report, ignores)
    print(render(verdict))
    return 1 if verdict.unreviewed else 0


if __name__ == "__main__":
    raise SystemExit(main())
