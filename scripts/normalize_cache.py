"""Normalize cached extraction markdown — strip PyMuPDF4LLM image placeholders (KI-14).

The cached ``.md`` is the ingest source-of-truth and ``--rebuild`` does NOT re-extract
(``ingest.load_or_extract`` trusts mtime freshness), so documents cached before the R1
extraction fix still carry ``**==> picture […] intentionally omitted <==**`` placeholder
lines in the RAG chunk store. This idempotent runner rewrites each cached ``.md`` through
``extractors.strip_image_placeholders`` via the atomic cache writer — **only when stripping
changes the content**. Because ``doc_hash`` is computed over the cached markdown, a
normalized doc gets a new hash, so a subsequent plain ``python -m doc_assistant.ingest``
(no ``--rebuild``) re-chunks / re-embeds only the affected documents.

Additive + idempotent (Enrichment-Layer Pattern); dry-run by default; never touches the
chunk store directly. $0 (deterministic, no LLM). Run on the host (KI-5). The cache dir is
``config.CACHE_PATH`` (honours ``DOC_DATA_DIR`` — point it at another data home to
normalize that corpus).

Usage:
    python -m scripts.normalize_cache            # dry-run: report what would change
    python -m scripts.normalize_cache --apply    # rewrite changed caches atomically
    # then re-index the affected documents:
    python -m doc_assistant.ingest
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from doc_assistant import config
from doc_assistant.extractors import count_image_placeholders, strip_image_placeholders
from doc_assistant.fsutil import atomic_write_text

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


@dataclass(frozen=True)
class FileChange:
    """A cached ``.md`` whose normalized content differs from what is on disk."""

    path: Path
    placeholders: int


@dataclass(frozen=True)
class NormalizeResult:
    scanned: int
    applied: bool
    changed: list[FileChange] = field(default_factory=list)

    @property
    def n_changed(self) -> int:
        return len(self.changed)

    @property
    def total_placeholders(self) -> int:
        return sum(c.placeholders for c in self.changed)


def normalize_cache_dir(cache_dir: Path, *, apply: bool) -> NormalizeResult:
    """Strip image placeholders from every cached ``.md`` under ``cache_dir``.

    Dry-run by default (``apply=False``) — reports which files *would* change and writes
    nothing. With ``apply=True`` each changed file is rewritten atomically. Idempotent: a
    second ``apply`` run reports zero changes.
    """
    changed: list[FileChange] = []
    scanned = 0
    for md_path in sorted(cache_dir.rglob("*.md")):
        if not md_path.is_file():
            continue
        scanned += 1
        text = md_path.read_text(encoding="utf-8")
        cleaned = strip_image_placeholders(text)
        if cleaned == text:
            continue
        changed.append(FileChange(path=md_path, placeholders=count_image_placeholders(text)))
        if apply:
            atomic_write_text(md_path, cleaned)
    return NormalizeResult(scanned=scanned, applied=apply, changed=changed)


def _format_report(result: NormalizeResult, cache_dir: Path) -> str:
    out: list[str] = []
    out.append("=" * 76)
    out.append(f"Cache dir:               {cache_dir}")
    out.append(f"Files scanned:           {result.scanned}")
    out.append(f"Files needing changes:   {result.n_changed}")
    out.append(f"Placeholder lines:       {result.total_placeholders}")
    out.append(f"Mode:                    {'APPLIED' if result.applied else 'DRY-RUN'}")
    out.append("=" * 76)
    for change in result.changed:
        out.append(f"    -{change.placeholders:<4d}  {change.path.name}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true", help="Rewrite changed caches (default: dry-run)"
    )
    args = parser.parse_args()

    cache_dir = config.CACHE_PATH
    if not cache_dir.exists():
        print(f"No cache directory at {cache_dir} — nothing to normalize.")
        return 0

    result = normalize_cache_dir(cache_dir, apply=args.apply)
    print(_format_report(result, cache_dir))
    if not args.apply:
        print("\nDry run (nothing written). Pass --apply to rewrite the changed caches.")
    elif result.n_changed:
        print("\nDone. Next: python -m doc_assistant.ingest  (re-index the affected docs).")
    else:
        print("\nNothing to change — caches already normalized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
