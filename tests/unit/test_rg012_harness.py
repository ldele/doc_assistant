"""Guards for the RG-012 clean-machine harness (`scripts/rg012/`).

The harness runs inside Windows Sandbox, where no test can reach it, and its verdict is read back
on the host by `scripts/release_preflight.py`. So the two halves share a text contract — log lines
and file names — that nothing but these tests holds together. Each test below is a way the gate has
already failed, or would fail silently: a non-ASCII byte that stops PowerShell 5.1 parsing before
anything logs (two lost runs, 2026-08-05/06), a reader that no longer finds what the writer writes
(0.6.0's size line), and a sandbox that runs some other copy of the script.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripts import release_preflight as preflight

HARNESS = Path(__file__).resolve().parents[2] / "scripts" / "rg012"
RUN = HARNESS / "rg012-run.ps1"
WSB = HARNESS / "rg012-tier2.wsb"


def test_the_harness_files_are_ascii_only() -> None:
    """Windows PowerShell 5.1 reads a BOM-less UTF-8 file as ANSI: one em-dash breaks the parse
    before the script can log a line — what both "LogonCommand never fires" reports were."""
    for path in (RUN, WSB):
        bad = [i for i, b in enumerate(path.read_bytes()) if b > 127]
        assert not bad, f"{path.name} has non-ASCII bytes at offsets {bad[:5]}"


def test_the_harness_writes_every_line_and_file_the_preflight_reads() -> None:
    """Pin the writer against the reader, not either against itself."""
    script = RUN.read_text(encoding="ascii")
    for literal in (
        preflight._RUN_START,
        '"python on PATH? "',
        '"chunk_count after ingest: "',
        '"turns planned: "',
        "'turn-{0}-result.json'",
        "installer chosen: {0} ({1} MB, built {2})",
    ):
        assert literal in script, f"the harness no longer writes {literal!r}"

    # And the reader's patterns accept the lines those literals render to.
    assert preflight._PYTHON_ON_PATH.search("python on PATH? False   (must be False)")
    assert preflight._CHUNKS_AFTER_INGEST.search("chunk_count after ingest: 322")
    assert preflight._TURNS_PLANNED.search("turns planned: 3")


def test_the_harness_asks_enough_questions_for_a_citation_verdict() -> None:
    """Fewer turns than the reader requires would make every run fail the citation half."""
    script = RUN.read_text(encoding="ascii")
    block = re.search(r"\$questions = @\((.*?)\n\)", script, re.S)
    assert block is not None, "the question list moved; update this test with it"
    questions = re.findall(r"^\s*'([^']+)'", block.group(1), re.M)
    assert len(questions) >= preflight.RG012_MIN_TURNS
    assert len(set(questions)) == len(questions), "a repeated question is not an independent turn"


def test_each_turn_gets_its_own_session() -> None:
    """A shared session lets one answer's history shape the next, so the turns stop being
    independent."""
    script = RUN.read_text(encoding="ascii")
    assert re.search(r"session_id = \('rg012-\{0\}-turn\{1\}' -f \$stamp, \$k\)", script)


def test_the_sandbox_runs_the_tracked_script() -> None:
    """The .wsb must map this folder and launch this file — not a host copy that can drift."""
    wsb = WSB.read_text(encoding="ascii")
    mapping = re.search(
        r"<HostFolder>([^<]*)</HostFolder><SandboxFolder>C:\\rg012\\script</SandboxFolder>", wsb
    )
    assert mapping is not None
    assert mapping.group(1).replace("\\", "/").lower().endswith("scripts/rg012")
    assert r"-File C:\rg012\script\rg012-run.ps1" in wsb
    assert RUN.is_file()
