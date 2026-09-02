"""Release preflight — mechanically check the things that have actually gone wrong.

    uv run --no-sync python -m scripts.release_preflight
    uv run --no-sync python -m scripts.release_preflight --json      # machine-readable

Read-only. Exits 1 if any check FAILS, 0 otherwise (WARNs never fail the run).

**Every check here is a bug that shipped or nearly shipped.** This file is not a generic
best-practice list; each entry cites the incident that put it here, so nobody deletes one for
looking redundant:

* ``versions``  — v0.4.0 bumped five version strings and missed ``uv.lock``. CI and the Docker
  build install with ``--locked``, which fails rather than re-resolving, so every gate after
  dependency-install was skipped on ``main`` for days before anyone noticed. The two Cargo files
  joined later, and from the *opposite* failure: this check never opened them, so they held 0.4.1
  through v0.4.2, v0.5.0 and v0.5.1 while it reported green. An agreement check is worth exactly
  as much as its file list, which is why that list is now pinned by a test of its own.
* ``artifact_fresh`` — the whole point of 2026-08-06. Source-green says **nothing** about a frozen
  binary (KI-34: the shipped build could not read a single PDF while every test passed). If the
  installer predates the code, the thing tested is not the thing shipped.
* ``sidecar_size`` — KI-34 is detectable as a size cliff: 1545.5 MB broken vs 1562.1 MB fixed,
  because ``collect_all("fitz")`` silently dropped ~17 MB of PyMuPDF data files. The cheapest
  possible regression check on a packaging bug that is invisible from source.
* ``rg012`` — ties "the clean-machine gate passed" to **this exact artifact**, by matching the
  installer build timestamp the harness logged against the installer on disk. A PASS from a
  previous build is worse than no PASS, because it reads as evidence.
* ``dev_commands`` — the app told users to run ``just api`` (KI-39): a task runner and a repo
  recipe that someone who installed an .exe does not have.

What this CANNOT check, and why the checklist in ``docs/RELEASE.md`` still exists: whether the
CHANGELOG is *true*, whether a known limit is still a limit, and whether the answers the app gives
are any good. Those need judgment and measurement, not a script.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent

SIDECAR = ROOT / "apps/desktop/src-tauri/binaries/doc-assistant-api-x86_64-pc-windows-msvc.exe"
BUNDLE = ROOT / "apps/desktop/src-tauri/target/release/bundle/nsis"
RG012_ARCHIVES = Path("C:/rg012-host")

# The Rust half of the version bump. The crate name differs from the Python package's, and the
# lock records it under that name — so both have to be spelled out rather than derived.
CARGO_TOML = "apps/desktop/src-tauri/Cargo.toml"
CARGO_LOCK = "apps/desktop/src-tauri/Cargo.lock"
CARGO_CRATE = "doc-assistant-desktop"

# The sidecar carries bundled model weights + PyMuPDF data. A build that comes out materially
# smaller has dropped something (KI-34).
#
# **MiB, not decimal MB** — deliberately, because the recorded KI-34 reference numbers are what
# Windows reports: 1545.5 MiB broken vs 1562.1 MiB fixed. The floor sits BETWEEN them, so the exact
# regression that shipped would fail this check. Getting the unit wrong makes the floor ~1478 MiB
# and the check useless while still looking green (caught writing this file).
# Re-baseline deliberately when dependencies change size — and record the new numbers here.
SIDECAR_MIN_MIB = 1555.0
KI34_BROKEN_MIB, KI34_FIXED_MIB = 1545.5, 1562.1

# Source trees whose mtime the artifact must beat. `apps/desktop/src-tauri/target` is excluded by
# construction (it is build output, not source).
SOURCE_GLOBS = ("src/**/*.py", "apps/api/**/*.py", "apps/desktop/src/**/*")

OK, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    notes: list[str] = field(default_factory=list)


def _mib(p: Path) -> float:
    """MiB — what Windows reports, and the unit every recorded reference number is in."""
    return p.stat().st_size / 1024 / 1024


def _mtime(p: Path) -> datetime:
    return datetime.fromtimestamp(p.stat().st_mtime)


def _describe(p: Path) -> str:
    return f"{_mib(p):,.1f} MiB  {_mtime(p):%Y-%m-%d %H:%M}"


def _run(*args: str) -> str:
    return subprocess.run(
        args, cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def _newest_source() -> tuple[Path | None, datetime | None]:
    """The most recently edited tracked source file (git-tracked only, so build output and
    scratch files cannot mask a stale artifact)."""
    tracked = set(_run("git", "ls-files").splitlines())
    newest_p: Path | None = None
    newest_t: datetime | None = None
    for rel in tracked:
        if not rel.startswith(("src/", "apps/api/", "apps/desktop/src/")):
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        t = datetime.fromtimestamp(p.stat().st_mtime)
        if newest_t is None or t > newest_t:
            newest_p, newest_t = p, t
    return newest_p, newest_t


def collect_versions() -> dict[str, str]:
    """Every file that carries the project's own version, read. Keys are repo-relative paths.

    Split out from `check_versions` so a test can pin **which files are read**, because the bug
    that added the two Cargo entries was a *missing source*, not a wrong comparison: a check that
    asks only "do the ones I open agree?" stays green forever while a file it never opens drifts.
    `Cargo.toml` and `Cargo.lock` sat at 0.4.1 through v0.4.2, v0.5.0 and v0.5.1 — three tagged
    releases — because neither this function nor `docs/RELEASE.md` §1 listed them.
    """
    found: dict[str, str] = {}
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    found["pyproject.toml"] = pyproject["project"]["version"]

    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    m = re.search(r'name = "doc-assistant"\s*\nversion = "([^"]+)"', lock)
    found["uv.lock"] = m.group(1) if m else "(not found)"

    init = (ROOT / "src/doc_assistant/__init__.py").read_text(encoding="utf-8")
    mv = re.search(r'^__version__ = "([^"]+)"', init, re.MULTILINE)
    found["src/doc_assistant/__init__.py"] = mv.group(1) if mv else "(not found)"

    for rel in ("apps/desktop/package.json", "apps/desktop/src-tauri/tauri.conf.json"):
        data = json.loads((ROOT / rel).read_text(encoding="utf-8"))
        found[rel] = data.get("version", "(missing)")

    manifest = tomllib.loads((ROOT / CARGO_TOML).read_text(encoding="utf-8"))
    found[CARGO_TOML] = str(manifest.get("package", {}).get("version", "(missing)"))

    # The lock is the one source that cannot be bumped ahead of time and stay bumped: cargo
    # rewrites it only when cargo *runs*, which on a release is during the build — after the
    # release commit. It is a list of package tables, so the crate has to be found by name.
    locked = tomllib.loads((ROOT / CARGO_LOCK).read_text(encoding="utf-8"))
    crate = next((p for p in locked.get("package", []) if p.get("name") == CARGO_CRATE), None)
    found[CARGO_LOCK] = str(crate.get("version", "(missing)")) if crate else "(not found)"

    return found


def check_versions() -> Check:
    """All seven version strings must agree — including uv.lock (the v0.4.0 CI break), the two
    Cargo files (silently 0.4.1 for three releases), and the `__version__` constant the update
    check compares against (ADR-044: a stale constant makes the app compare itself to a lie)."""
    found = collect_versions()
    distinct = set(found.values())
    if len(distinct) == 1:
        return Check("versions", OK, f"all {len(found)} agree on {distinct.pop()}")
    return Check(
        "versions",
        FAIL,
        "version strings disagree",
        [f"{k} = {v}" for k, v in found.items()],
    )


def check_tree_clean() -> Check:
    dirty = _run("git", "status", "--porcelain")
    if not dirty:
        return Check("tree_clean", OK, "no uncommitted tracked changes")
    return Check(
        "tree_clean",
        FAIL,
        "uncommitted changes — the tag would not match the build",
        dirty.splitlines()[:10],
    )


def check_changelog(version: str) -> Check:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## [{version}]" not in text:
        return Check("changelog", FAIL, f"no '## [{version}]' section in CHANGELOG.md")
    line = next(ln for ln in text.splitlines() if ln.startswith(f"## [{version}]"))
    if "unreleased" in line.lower():
        return Check("changelog", FAIL, f"section still marked Unreleased: {line.strip()}")
    return Check("changelog", OK, line.strip())


def check_artifacts() -> tuple[Check, Path | None]:
    if not SIDECAR.is_file():
        return Check("artifacts", FAIL, f"no frozen sidecar at {SIDECAR}"), None
    installers = sorted(
        BUNDLE.glob("Provenote*setup.exe"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not installers:
        return Check("artifacts", FAIL, f"no Provenote*setup.exe under {BUNDLE}"), None
    inst = installers[0]
    notes = [f"sidecar   {_describe(SIDECAR)}", f"installer {_describe(inst)}"]
    if len(installers) > 1:
        notes.append(f"NOTE {len(installers)} installers here — never pick one incidentally")
    return Check("artifacts", OK, f"{inst.name}", notes), inst


def check_artifact_fresh(installer: Path | None) -> Check:
    """The artifact must be newer than the newest source edit. This is the KI-34 lesson as code."""
    if installer is None:
        return Check("artifact_fresh", SKIP, "no artifact to compare")
    newest_p, newest_t = _newest_source()
    if newest_t is None:
        return Check("artifact_fresh", WARN, "could not determine newest source file")
    sidecar_t = datetime.fromtimestamp(SIDECAR.stat().st_mtime)
    inst_t = datetime.fromtimestamp(installer.stat().st_mtime)
    stale = [n for n, t in (("sidecar", sidecar_t), ("installer", inst_t)) if t < newest_t]
    rel = newest_p.relative_to(ROOT) if newest_p else "?"
    detail = f"newest source: {rel} @ {newest_t:%Y-%m-%d %H:%M}"
    if stale:
        return Check(
            "artifact_fresh",
            FAIL,
            f"{' and '.join(stale)} predate(s) the newest source edit — REBUILD",
            [detail, f"sidecar {sidecar_t:%Y-%m-%d %H:%M}", f"installer {inst_t:%Y-%m-%d %H:%M}"],
        )
    return Check("artifact_fresh", OK, "newer than every tracked source file", [detail])


def check_sidecar_size() -> Check:
    if not SIDECAR.is_file():
        return Check("sidecar_size", SKIP, "no sidecar")
    mib = SIDECAR.stat().st_size / 1024 / 1024
    if mib < SIDECAR_MIN_MIB:
        return Check(
            "sidecar_size",
            FAIL,
            f"{mib:,.1f} MiB is below the {SIDECAR_MIN_MIB:,.1f} MiB floor — a bundle was dropped",
            [
                f"KI-34 reference: {KI34_BROKEN_MIB} MiB broken vs {KI34_FIXED_MIB} MiB fixed",
                "collect_all('fitz') without 'pymupdf' lost the data files and broke every PDF",
            ],
        )
    return Check(
        "sidecar_size",
        OK,
        f"{mib:,.1f} MiB (floor {SIDECAR_MIN_MIB:,.1f}, KI-34 fixed = {KI34_FIXED_MIB})",
    )


_CHOSEN = re.compile(r"installer chosen: (\S+) \(([\d,]+) MB, built ([^)]+)\)")


def check_rg012(installer: Path | None) -> Check:
    """Did the clean-machine gate pass **on this artifact**?

    Matches the build timestamp the harness recorded against the installer on disk. A PASS from a
    previous build is worse than no PASS at all — it reads as evidence for something never tested.
    """
    if installer is None:
        return Check("rg012", SKIP, "no artifact to match")
    if not RG012_ARCHIVES.is_dir():
        return Check("rg012", WARN, f"no harness at {RG012_ARCHIVES} (run on the build box)")
    logs = list(RG012_ARCHIVES.glob("out*/rg012.log"))
    if not logs:
        return Check("rg012", FAIL, "no RG-012 run recorded — the clean-machine gate has not run")
    inst_t = datetime.fromtimestamp(installer.stat().st_mtime).replace(second=0, microsecond=0)
    matches: list[str] = []
    for log in sorted(logs, key=lambda p: p.stat().st_mtime, reverse=True):
        text = log.read_text(encoding="utf-8", errors="replace")
        m = _CHOSEN.search(text)
        if not m:
            continue
        try:
            built = datetime.strptime(m.group(3).strip(), "%m/%d/%Y %I:%M:%S %p")
        except ValueError:
            continue
        if built.replace(second=0, microsecond=0) != inst_t:
            continue
        verdict = "PASS" if "TIER-2: PASS" in text else "FAIL"
        matches.append(f"{log.parent.name}: {verdict}")
        if verdict == "PASS":
            return Check("rg012", OK, f"PASS on this artifact ({log.parent.name})")
    if matches:
        return Check("rg012", FAIL, "the run against this artifact did NOT pass", matches)
    return Check(
        "rg012",
        FAIL,
        "no RG-012 run matches this installer — the gate ran against a DIFFERENT build",
        [f"installer built {inst_t:%Y-%m-%d %H:%M}", f"{len(logs)} archived run(s) found"],
    )


def _just_recipes() -> set[str]:
    """Recipe names from the justfile.

    `just` is an ordinary English word, so matching ``just \\w+`` flags "just now", "just a" and
    "just the" — three false positives on the first run of this check. Only a real recipe name
    makes ``just X`` a command, so read them from the justfile rather than guessing. This also
    keeps the check honest as recipes come and go.
    """
    jf = ROOT / "justfile"
    if not jf.is_file():
        return set()
    text = jf.read_text(encoding="utf-8")
    return {m.group(1) for m in re.finditer(r"^([a-z][\w-]*)(?: [^:\n]*)?:(?!=)", text, re.M)}


def check_dev_commands() -> Check:
    """No developer command may appear in a user-facing frontend string (KI-39).

    The app's only failure message used to read "backend unreachable. Run ``just api``" — a task
    runner and a repo recipe that someone who installed an .exe does not have. Comments are
    skipped: explaining a dev command to the next maintainer is fine, printing it at a user is not.
    """
    recipes = _just_recipes()
    patterns = [
        r"npm run [\w:-]+",
        r"\buv run\b",
        r"\bpip install\b",
        r"\bcargo (build|run|test)\b",
    ]
    if recipes:
        patterns.append(r"\bjust (" + "|".join(sorted(map(re.escape, recipes))) + r")\b")
    dev_cmd = re.compile("(" + "|".join(patterns) + ")")

    offenders: list[str] = []
    for p in sorted((ROOT / "apps/desktop/src").rglob("*.svelte")):
        in_block_comment = False
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if in_block_comment:
                if "-->" in stripped or "*/" in stripped:
                    in_block_comment = False
                continue
            if stripped.startswith(("<!--", "/*")) and not ("-->" in stripped or "*/" in stripped):
                in_block_comment = True
                continue
            if stripped.startswith(("//", "*", "<!--", "/*")):
                continue
            m = dev_cmd.search(line)
            if m:
                offenders.append(f"{p.relative_to(ROOT)}:{i}  {m.group(0)!r}")
    if offenders:
        return Check(
            "dev_commands", FAIL, "developer command in shipped UI text (KI-39)", offenders[:8]
        )
    return Check(
        "dev_commands",
        OK,
        f"no developer commands in shipped UI text ({len(recipes)} just recipes checked)",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "version"
    ]
    artifacts, installer = check_artifacts()
    checks = [
        check_versions(),
        check_changelog(version),
        check_tree_clean(),
        artifacts,
        check_artifact_fresh(installer),
        check_sidecar_size(),
        check_rg012(installer),
        check_dev_commands(),
    ]

    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=1))
    else:
        print(f"\nRelease preflight — version {version}\n" + "=" * 60)
        for c in checks:
            mark = {OK: "[ok]  ", FAIL: "[FAIL]", WARN: "[warn]", SKIP: "[skip]"}[c.status]
            print(f"{mark} {c.name:<16} {c.detail}")
            for n in c.notes:
                print(f"           {n}")
        failed = [c.name for c in checks if c.status == FAIL]
        print("=" * 60)
        if failed:
            print(f"NOT READY — {len(failed)} check(s) failed: {', '.join(failed)}")
            print("The judgment steps are in docs/RELEASE.md; this script only covers the")
            print("mechanical ones. A green run here is necessary, not sufficient.")
        else:
            print("Mechanical checks pass. Now do the judgment steps in docs/RELEASE.md.")
    return 1 if any(c.status == FAIL for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
