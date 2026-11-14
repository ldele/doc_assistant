"""The invented hostile documents match their pins (tests/eval/hostile/README.md).

A hostile case's check is written against the exact text of its document, so an edit that is not
re-pinned would leave a case testing an instruction the document no longer carries. Line endings
are normalised before hashing: git may check a file out with CRLF on Windows.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import yaml  # type: ignore[import-untyped]

HOSTILE = Path(__file__).resolve().parents[2] / "tests" / "eval" / "hostile"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def test_every_hostile_document_matches_its_pin() -> None:
    manifest = yaml.safe_load((HOSTILE / "hostile_manifest.yaml").read_text(encoding="utf-8"))
    pinned = {d["filename"]: d["sha256"] for d in manifest["documents"]}
    on_disk = {p.name for p in HOSTILE.glob("h*.md")}
    assert set(pinned) == on_disk
    for name, sha in pinned.items():
        assert _digest(HOSTILE / name) == sha, f"{name} changed without a new pin"
