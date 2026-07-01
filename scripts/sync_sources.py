"""Build + re-download a private manifest of your ingested library.

The companion to ``scripts/download_corpus.py``, but for your OWN documents rather
than the public arXiv corpus. ``data/sources_manifest.yaml`` records each file in
``data/sources/`` by ``sha256`` + size and the URL it was downloaded from, so the
library can be reconstituted on another machine. The manifest is **gitignored**
(your library is mostly copyrighted) — share it out-of-band.

URLs are curated by you (ingest does not capture them), except that any file
matching the public corpus by checksum is auto-filled.

Usage::

    python -m scripts.sync_sources                # build/update the manifest from data/sources/
    python -m scripts.sync_sources --download     # re-fetch missing files from their urls
    python -m scripts.sync_sources --verify-only  # checksum what's on disk against the manifest
    python -m scripts.sync_sources --dry-run      # print the plan, write/fetch nothing
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from doc_assistant import config
from doc_assistant.sources_manifest import (
    build_manifest,
    download_missing,
    load_manifest,
    verify_present,
    write_manifest,
)

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def _build(manifest_path: Path, sources_dir: Path, *, dry_run: bool) -> int:
    result = build_manifest(sources_dir=sources_dir, manifest_path=manifest_path)
    print(f"[build] scanned {sources_dir}")
    print(f"    total entries    : {result.total}")
    print(f"    new this run     : {result.new}")
    print(f"    url auto-filled  : {result.enriched} (matched the public corpus)")
    print(f"    still missing url: {result.missing_url}")
    if dry_run:
        print(f"    (dry-run) would write {manifest_path}")
        return 0
    write_manifest(manifest_path, result.entries)
    print(f"    wrote {manifest_path}")
    if result.missing_url:
        print(
            f"    !!  fill in the `url:` for the {result.missing_url} private file(s), "
            "then re-run to share the manifest."
        )
    return 0


def _download(manifest_path: Path, dest_dir: Path, *, dry_run: bool) -> int:
    entries = load_manifest(manifest_path)
    if not entries:
        print(f"[download] no manifest at {manifest_path} — run a plain build first")
        return 1
    outcomes = download_missing(entries, dest_dir, dry_run=dry_run)
    tally: dict[str, int] = {}
    for outcome in outcomes:
        tally[outcome.status] = tally.get(outcome.status, 0) + 1
        if outcome.status in {"downloaded", "would_download", "mismatch", "failed"}:
            suffix = f" — {outcome.detail}" if outcome.detail else ""
            print(f"    {outcome.status:14s} {outcome.filename}{suffix}")
    print("\n--- summary ---")
    for status in ("downloaded", "would_download", "present", "no_url", "mismatch", "failed"):
        if status in tally:
            print(f"  {status:14s}: {tally[status]}")
    return 1 if tally.get("failed") else 0


def _verify(manifest_path: Path, dest_dir: Path) -> int:
    entries = load_manifest(manifest_path)
    if not entries:
        print(f"[verify] no manifest at {manifest_path}")
        return 1
    mismatched = 0
    for outcome in verify_present(entries, dest_dir):
        if outcome.status == "mismatch":
            mismatched += 1
            print(f"    !!  sha256 MISMATCH {outcome.filename}")
    print(f"[verify] checked {dest_dir}: {mismatched} mismatch(es)")
    return 1 if mismatched else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(config.SOURCES_MANIFEST),
        help="Manifest path (default %(default)s)",
    )
    parser.add_argument(
        "--dest",
        type=str,
        default=str(config.DOCS_PATH),
        help="Sources directory to scan / download into (default %(default)s)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Re-download missing files listed in the manifest (instead of building)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Checksum files already on disk against the manifest; build/fetch nothing",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without writing or fetching"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    dest_dir = Path(args.dest)

    if args.verify_only:
        return _verify(manifest_path, dest_dir)
    if args.download:
        return _download(manifest_path, dest_dir, dry_run=args.dry_run)
    return _build(manifest_path, dest_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
