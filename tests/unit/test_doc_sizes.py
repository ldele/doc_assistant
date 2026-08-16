"""Caps on the append-only coordination docs, so they get rotated instead of growing forever.

**The failure this comes from (2026-08-15).** `docs/DEVLOG.md` reached **8,244 lines / 623 KB** —
6.4x the next largest doc in the repo — before anyone noticed. It had been rotated once, on
2026-07-21, and nothing made the next rotation happen; the working log had quietly become an
archive that also happened to hold this week's work. An append-only doc with no cap has no natural
moment to rotate: every individual entry is small and correct, so the growth is invisible per
commit and only visible in aggregate.

**Why this is a pytest guard and not a cpc gate.** `scripts/conventions.toml` caps the entry file
(`entry_max_lines`), each module `CLAUDE.md`, and the baton (`session_max_entries`) — but
`docs_check` implements no DEVLOG rule, and the cpc tooling under `tools/conventions/` is vendored,
gitignored and owned by another project (ADR-001/ADR-007), so a cap added there would be lost on
the next refresh. The durable place for a project-specific rule is this repo's own test suite,
which is where `test_docs_encoding.py` already lives for the same reason.

**Rotating is the fix, never raising the cap.** Move the OLDEST entries verbatim into
`docs/archive/<NAME>-archive-NNN.md`, newest-first, unedited — the pattern
`DEVLOG-archive-001/002` and `SESSION-archive-001/002` already follow. Raising a number here to
make a red test green re-creates exactly the condition this guard exists to catch.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Cap, and the headroom rationale. Caps sit ~40-60% above the post-rotation size so a rotation
# buys weeks of work rather than days — a cap that trips immediately just gets raised.
CAPS: dict[str, int] = {
    # Rotated 2026-08-15 at 2,550 lines (8,244 -> archive-002). Busiest days add ~250 lines.
    "docs/DEVLOG.md": 4000,
    # Local-only (ADR-029). Rotates resolved issues to docs/archive/KNOWN_ISSUES-resolved-NNN.md.
    ".claude/KNOWN_ISSUES.md": 1800,
    # Local-only. No archive yet; closed RG entries are the ones to move when this trips.
    ".claude/RIGOR_TODO.md": 1500,
}

# `.claude/SESSION.md` is deliberately absent: it is already capped at 10 ENTRIES by
# `docs_check` rule 11b (`session_max_entries`, cpc ADR-018). A second cap in a different unit
# could fail a baton that is entry-compliant, which would make the two rules contradict.


@pytest.mark.parametrize(("rel", "cap"), sorted(CAPS.items()))
def test_append_only_doc_stays_under_its_cap(rel: str, cap: int) -> None:
    path = REPO_ROOT / rel
    if not path.exists():
        # `.claude/` is gitignored working state — absent in a fresh clone and in CI.
        pytest.skip(f"{rel} not present (local-only working state)")

    lines = len(path.read_text(encoding="utf-8").splitlines())

    assert lines <= cap, (
        f"{rel} is {lines:,} lines, over its {cap:,}-line cap.\n"
        f"Rotate the OLDEST entries verbatim into docs/archive/ (newest-first, unedited), "
        f"then update the live doc's header to say where they went and from which date it "
        f"continues.\n"
        f"Do NOT raise the cap to make this pass — unbounded growth is the defect being guarded."
    )


def test_archives_are_exempt_by_construction() -> None:
    """The archive is the destination, so it must never be capped.

    Pins the intent rather than the sizes: a future edit that adds `docs/archive/...` to CAPS
    would make rotation impossible — the very act of fixing a breach would cause another one.
    """
    assert not any(rel.startswith("docs/archive/") for rel in CAPS), (
        "An archive file is capped in CAPS. Archives absorb rotations and must stay uncapped, "
        "or rotating a doc simply moves the breach."
    )
