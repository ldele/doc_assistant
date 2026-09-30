"""Security step S-8 — `pip-audit`, blocking, with a reviewed ignore list.

    uv run --no-sync python -m scripts.pip_audit_gate                  # audit this venv
    uv run --no-sync python -m scripts.pip_audit_gate --report r.json  # judge a saved report

Runs `pip-audit --format json` over the current environment and **fails on any advisory that is
not reviewed in `pip-audit-ignore.toml`**. pip-audit reads no ignore file of its own and has no
severity filter, so this is the whole policy: an advisory is either fixed by an upgrade or written
down with why it does not apply here and what would reverse that. Until 2026-09-30 CI ran
pip-audit with `continue-on-error`, and 59 advisories across 18 packages sat behind a green check
(`docs/security.md` S6) — a check that cannot fail is not a control.

Exit codes: 0 — nothing unreviewed · 1 — an unreviewed advisory · 2 — the ignore file is malformed,
or pip-audit produced no report. The last two fail closed: a gate that cannot look must not pass.

A **stale** ignore (listed, but its advisory no longer appears) is reported and never fatal: CI's
CPU venv and a CUDA dev venv install different package sets, so an entry can be live in one and
absent from the other. Delete an entry once it is stale in both.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tomllib

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
IGNORE_FILE = ROOT / "pip-audit-ignore.toml"
FIELDS = ("id", "package", "reviewed", "why", "reverse_if")


@dataclass(frozen=True)
class Ignore:
    """One reviewed advisory: every field is required and non-empty."""

    id: str
    package: str
    reviewed: str
    why: str
    reverse_if: str


@dataclass(frozen=True)
class Finding:
    """One advisory reported against one installed package."""

    package: str
    version: str
    id: str
    aliases: tuple[str, ...]
    fix_versions: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    audited: int
    unreviewed: list[Finding]
    ignored: list[tuple[Finding, Ignore]]
    stale: list[Ignore]
    skipped: list[tuple[str, str]]


def normalize(name: str) -> str:
    """PEP 503 name normalisation, so `Pillow` and `pillow` are one package."""
    return re.sub(r"[-_.]+", "-", name).lower()


def load_ignores(text: str) -> list[Ignore]:
    """Parse the ignore file; raise ValueError naming every malformed entry at once."""
    entries = tomllib.loads(text).get("ignore", [])
    problems: list[str] = []
    out: list[Ignore] = []
    seen: set[str] = set()
    for n, raw in enumerate(entries, start=1):
        label = f"entry {n} ({raw.get('id', '?')})"
        missing = [k for k in FIELDS if not str(raw.get(k, "")).strip()]
        unknown = sorted(set(raw) - set(FIELDS))
        if missing:
            problems.append(f"{label}: missing {', '.join(missing)}")
        if unknown:
            problems.append(f"{label}: unknown field {', '.join(unknown)}")
        if missing or unknown:
            continue
        if raw["id"] in seen:
            problems.append(f"{label}: duplicate id")
            continue
        seen.add(raw["id"])
        out.append(Ignore(**{k: str(raw[k]).strip() for k in FIELDS}))
    if problems:
        raise ValueError("; ".join(problems))
    return out


def findings(report: dict[str, Any]) -> tuple[list[Finding], list[tuple[str, str]], int]:
    """The advisories, the skipped dependencies, and how many packages were audited.

    One advisory can arrive twice for one package (pip-audit merges two sources and keeps both
    texts), so findings are deduplicated on (package, version, id).
    """
    found: dict[tuple[str, str, str], Finding] = {}
    skipped: list[tuple[str, str]] = []
    audited = 0
    for dep in report.get("dependencies", []):
        if "skip_reason" in dep:
            skipped.append((dep["name"], dep["skip_reason"]))
            continue
        audited += 1
        for v in dep.get("vulns", []):
            key = (normalize(dep["name"]), str(dep.get("version", "?")), v["id"])
            found.setdefault(
                key,
                Finding(
                    package=dep["name"],
                    version=str(dep.get("version", "?")),
                    id=v["id"],
                    aliases=tuple(v.get("aliases") or ()),
                    fix_versions=tuple(v.get("fix_versions") or ()),
                ),
            )
    return list(found.values()), skipped, audited


def evaluate(report: dict[str, Any], ignores: list[Ignore]) -> Verdict:
    """Match each advisory to a reviewed ignore by id or any alias, for its own package only."""
    found, skipped, audited = findings(report)
    by_key = {(normalize(ig.package), ig.id): ig for ig in ignores}
    used: set[str] = set()
    unreviewed: list[Finding] = []
    ignored: list[tuple[Finding, Ignore]] = []
    for f in found:
        pkg = normalize(f.package)
        match = next((by_key[(pkg, k)] for k in (f.id, *f.aliases) if (pkg, k) in by_key), None)
        if match is None:
            unreviewed.append(f)
        else:
            ignored.append((f, match))
            used.add(match.id)
    stale = [ig for ig in ignores if ig.id not in used]
    return Verdict(audited, unreviewed, ignored, stale, skipped)


def run_pip_audit() -> dict[str, Any]:
    """pip-audit over the current environment as JSON; exit 2 when it produced no report."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip_audit", "--format", "json", "--progress-spinner", "off"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    try:
        report: dict[str, Any] = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        sys.stderr.write(proc.stderr)
        print(f"pip-audit gate: pip-audit produced no JSON report (exit {proc.returncode})")
        raise SystemExit(2) from exc
    return report


def render(v: Verdict) -> str:
    skipped = ", ".join(name for name, _ in v.skipped) or "none"
    lines = [f"pip-audit gate: {v.audited} packages audited; skipped (not on PyPI): {skipped}"]
    lines.append(f"  reviewed ignores that matched: {len(v.ignored)}")
    for f, ig in sorted(v.ignored, key=lambda p: (p[0].package, p[0].id)):
        lines.append(f"    {f.package} {f.version}  {f.id}  (reviewed {ig.reviewed})")
    lines.append(f"  stale ignores (advisory no longer reported here): {len(v.stale)}")
    for ig in v.stale:
        lines.append(f"    {ig.package}  {ig.id}")
    lines.append(f"  UNREVIEWED advisories: {len(v.unreviewed)}")
    for f in sorted(v.unreviewed, key=lambda f: (f.package, f.id)):
        fix = ", ".join(f.fix_versions) or "no fixed release"
        lines.append(f"    {f.package} {f.version}  {f.id}  fix: {fix}")
    if v.unreviewed:
        lines.append("-> FAIL: upgrade the package, or review it into pip-audit-ignore.toml")
    else:
        lines.append("-> OK")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    ap.add_argument(
        "--report",
        type=Path,
        help="judge a saved `pip-audit --format json` report instead of running pip-audit",
    )
    ap.add_argument("--ignore-file", type=Path, default=IGNORE_FILE)
    args = ap.parse_args(argv)
    try:
        ignores = load_ignores(args.ignore_file.read_text(encoding="utf-8"))
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"pip-audit gate: cannot use {args.ignore_file.name}: {exc}")
        return 2
    if args.report:
        report: dict[str, Any] = json.loads(args.report.read_text(encoding="utf-8-sig"))
    else:
        report = run_pip_audit()
    verdict = evaluate(report, ignores)
    print(render(verdict))
    return 1 if verdict.unreviewed else 0


if __name__ == "__main__":
    raise SystemExit(main())
