"""Guard tests for the Windows text-encoding hazard (non-negotiable #9) reaching tracked docs.

On 2026-08-11 a survey of all 166 tracked markdown files found **four already committed
double-encoded** — `docs/ROADMAP.md` (228 occurrences), `docs/architecture.md` (113),
`docs/knowledge-layer.md` (60), `docs/decisions.md` (12) — and they were exactly the four carrying
a UTF-8 BOM, the signature of one tool having read UTF-8 as the ANSI codepage and re-saved.
`architecture.md`'s flow diagram showed three garbage characters in place of each `↓` and `→`, and
`knowledge-layer.md`'s trust table — the one AGENTS.md tells every agent to read before believing a
marker — had the same in place of every ⚠️ and ❌ in its severity column, which is the whole point
of that column. Two of the four are linked from the README, so it was public.

(This docstring deliberately does NOT quote the damaged characters: they are exactly the
"ambiguous unicode" ruff's RUF002 exists to reject, so pasting an example here fails the linter.)

Every one was committed broken and **no gate noticed**, which is the actual defect these tests fix:
CONTEXT.md §9 already wrote the rule down, and a written rule caught nothing. pytest captures
stdout through its own UTF-8 buffer and CI is Linux, so neither can observe the hazard indirectly —
it has to be asserted against the file bytes, which is what these do.
"""

from __future__ import annotations

import subprocess
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# UTF-8 BOM. Harmless to a UTF-8 reader, but it is the tell that an ANSI-defaulting tool wrote the
# file, and PowerShell 5.1 needs BOM-less files kept ASCII for the opposite reason (CONTEXT.md §9).
BOM = "﻿"

# Mojibake signatures: what a UTF-8 lead byte looks like after being re-read as cp1252/latin-1.
# 'Â' (C2), 'Ã' (C3) and 'â€' (E2 80) cover the punctuation and accents this corpus actually uses.
MOJIBAKE = ("Â", "Ã", "â€")

TEXT_SUFFIXES = {".md", ".yaml", ".yml", ".toml"}


def _tracked_text_files() -> list[Path]:
    """Files git tracks, filtered to text formats we author. Empty list if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
            timeout=60,
        ).stdout
    except (subprocess.SubprocessError, OSError):  # pragma: no cover - git always present in CI
        return []
    paths = []
    for rel in out.decode("utf-8").split("\0"):
        if not rel:
            continue
        p = REPO_ROOT / rel
        if p.suffix.lower() in TEXT_SUFFIXES and p.is_file():
            paths.append(p)
    return paths


TRACKED = _tracked_text_files()


@pytest.mark.skipif(not TRACKED, reason="git unavailable; nothing to scan")
def test_no_tracked_text_file_carries_a_utf8_bom() -> None:
    """A BOM on a file we author means an ANSI-defaulting tool wrote it — the upstream cause."""
    offenders = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in TRACKED
        if p.read_bytes().startswith(BOM.encode("utf-8"))
    ]
    assert not offenders, (
        "UTF-8 BOM in tracked text files: "
        + ", ".join(offenders)
        + ". Write these as UTF-8 without a BOM (CONTEXT.md non-negotiable #9)."
    )


@pytest.mark.skipif(not TRACKED, reason="git unavailable; nothing to scan")
def test_no_tracked_text_file_is_double_encoded() -> None:
    """`·` must not be stored as `Â·`. Catches the 2026-08-11 damage class at the commit."""
    offenders: list[str] = []
    for p in TRACKED:
        text = p.read_text(encoding="utf-8")
        hits = sum(text.count(sig) for sig in MOJIBAKE)
        if hits:
            offenders.append(f"{p.relative_to(REPO_ROOT).as_posix()} ({hits})")
    assert not offenders, (
        "Double-encoded (mojibake) text in: "
        + ", ".join(offenders)
        + ". A tool read UTF-8 as the ANSI codepage and re-saved; repair by mapping each "
        "character back to its byte (cp1252 first, then latin-1) and decoding as UTF-8."
    )


# One control character in the repo is CONTENT, not damage: a DEVLOG entry whose subject is a
# separator character quotes it literally, inside backticks. Exempting it by (file, codepoint)
# rather than globally keeps every other stray control a failure — and the companion test below
# asserts the exemption is still earned, so it expires by itself if that prose ever goes.
#
# **Moved 2026-08-15**: was `docs/DEVLOG.md`, until the log was rotated and the entry holding it
# went to `DEVLOG-archive-002.md` (line ~1612). The companion test caught the move by failing —
# which is the whole point of it, so re-point the key rather than deleting the exemption.
# Rotating a doc moves its exemptions too; expect this key to follow the content again.
ALLOWED_CONTROLS: dict[str, set[str]] = {"docs/archive/DEVLOG-archive-002.md": {"U+001F"}}


@pytest.mark.skipif(not TRACKED, reason="git unavailable; nothing to scan")
def test_no_tracked_text_file_carries_stray_control_characters() -> None:
    """Leftovers of a partial repair: `❌` half-decodes to `â` + U+009D + `Œ`.

    Newline/tab/carriage-return are legitimate; any other C0/C1 control in prose is damage,
    except where ALLOWED_CONTROLS records a deliberate literal.
    """
    offenders: list[str] = []
    for p in TRACKED:
        rel = p.relative_to(REPO_ROOT).as_posix()
        text = p.read_text(encoding="utf-8")
        bad = {
            f"U+{ord(ch):04X}"
            for ch in text
            if unicodedata.category(ch) == "Cc" and ch not in "\n\r\t"
        } - ALLOWED_CONTROLS.get(rel, set())
        if bad:
            offenders.append(f"{rel} ({', '.join(sorted(bad))})")
    assert not offenders, "Stray control characters in: " + ", ".join(offenders)


def test_control_character_exemptions_are_still_earned() -> None:
    """An allowlist nobody revisits becomes a blind spot — fail when an entry stops applying."""
    if not TRACKED:
        pytest.skip("git unavailable")
    for rel, codepoints in ALLOWED_CONTROLS.items():
        text = (REPO_ROOT / rel).read_text(encoding="utf-8")
        present = {f"U+{ord(ch):04X}" for ch in text if unicodedata.category(ch) == "Cc"}
        stale = codepoints - present
        assert not stale, (
            f"{rel}: exemption for {', '.join(sorted(stale))} is no longer needed — "
            "remove it from ALLOWED_CONTROLS."
        )


def test_the_scan_actually_covers_the_files_that_were_damaged() -> None:
    """A guard that silently scans nothing passes forever — pin the four known-damaged files in."""
    if not TRACKED:
        pytest.skip("git unavailable")
    covered = {p.relative_to(REPO_ROOT).as_posix() for p in TRACKED}
    for rel in (
        "docs/ROADMAP.md",
        "docs/architecture.md",
        "docs/knowledge-layer.md",
        "docs/decisions.md",
    ):
        assert rel in covered, f"{rel} fell out of the scan set"
    assert len(TRACKED) > 100, f"scan set implausibly small ({len(TRACKED)})"
