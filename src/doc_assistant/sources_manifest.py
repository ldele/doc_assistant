"""Private sources manifest — record + re-download your ingested library.

A private mirror of the public-corpus reproducibility flow
(``scripts/download_corpus.py`` + ``tests/eval/corpus_manifest.yaml``): the
manifest maps each file in ``data/sources/`` to the URL it was downloaded from,
plus a ``sha256`` / ``bytes`` pin, so the library can be reconstituted on another
machine.

Ingest never captures where a file came from — the ``Document`` row only knows the
*local* path — so the URLs are curated by the user. The one shortcut: any file
whose bytes match the public corpus (by ``sha256``, falling back to filename) gets
its URL filled in automatically.

The manifest is **gitignored** — the library is mostly copyrighted and not
redistributable, so it is shared out-of-band, never committed (the repo is
public). Typical flow::

    # machine A (has the files)
    python -m scripts.sync_sources             # build/update data/sources_manifest.yaml
    # ...fill in `url:` for any private file the auto-match couldn't resolve...
    # copy data/sources_manifest.yaml to machine B out-of-band

    # machine B (fresh checkout)
    python -m scripts.sync_sources --download  # re-fetch into data/sources/
    python -m doc_assistant.ingest             # then ingest as usual

Pure core (``merge_entries`` / ``enrich_with_public_corpus`` / (de)serialisers) is
separated from the filesystem + network boundary so the merge/auto-fill logic is
unit-testable without touching disk or the wire.
"""

from __future__ import annotations

import hashlib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
import yaml  # type: ignore[import-untyped]

from doc_assistant import config
from doc_assistant.extractors import is_supported

log = structlog.get_logger(__name__)

PUBLIC_CORPUS_MANIFEST = config.PROJECT_ROOT / "tests" / "eval" / "corpus_manifest.yaml"
_USER_AGENT = {"User-Agent": "doc_assistant-sources-sync/1.0 (reproducibility script)"}
_DOWNLOAD_TIMEOUT_S = 60
_READ_BLOCK = 1 << 20

_MANIFEST_HEADER = """\
# Private sources manifest — a re-downloadable index of THIS machine's data/sources/.
#
# GITIGNORED ON PURPOSE. Your library is mostly copyrighted and not redistributable,
# so this file is never committed (the repo is public) — share it out-of-band.
#
# Each entry pins a file by sha256 + size and records the URL it came from. `url` is
# null until you fill it in (ingest does not capture download URLs); files that match
# the public corpus by checksum are auto-filled. Rebuild after adding files:
#     python -m scripts.sync_sources
# Re-download on another machine (fetches every entry that has a url):
#     python -m scripts.sync_sources --download
"""


@dataclass
class SourceEntry:
    """One library file: its relative name, origin URL, and content pin."""

    filename: str
    url: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None


@dataclass
class BuildResult:
    """Outcome of (re)building the manifest from disk — does not include the write."""

    entries: list[SourceEntry]
    total: int
    new: int
    enriched: int
    missing_url: int


@dataclass
class SyncOutcome:
    """Per-file result of a download / verify pass."""

    filename: str
    status: str  # downloaded | present | no_url | mismatch | failed | would_download
    detail: str = ""


# --------------------------------------------------------------------------- #
# Pure core
# --------------------------------------------------------------------------- #
def entry_to_dict(entry: SourceEntry) -> dict[str, Any]:
    """Serialise one entry to the manifest's on-disk shape (key order preserved)."""
    return {
        "filename": entry.filename,
        "url": entry.url,
        "sha256": entry.sha256,
        "bytes": entry.size_bytes,
    }


def entry_from_dict(raw: dict[str, Any]) -> SourceEntry:
    """Parse one manifest mapping into a ``SourceEntry`` (tolerates extra keys)."""
    return SourceEntry(
        filename=str(raw["filename"]),
        url=(str(raw["url"]) if raw.get("url") else None),
        sha256=(str(raw["sha256"]) if raw.get("sha256") else None),
        size_bytes=(int(raw["bytes"]) if raw.get("bytes") is not None else None),
    )


def parse_manifest(text: str) -> list[SourceEntry]:
    """Parse manifest YAML text into entries (empty list if blank / no documents)."""
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        return []
    docs = data.get("documents") or []
    return [entry_from_dict(d) for d in docs]


def render_manifest(entries: list[SourceEntry]) -> str:
    """Serialise entries to YAML text (block style, key order preserved) + a header."""
    body = yaml.safe_dump(
        {"documents": [entry_to_dict(e) for e in entries]},
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    return f"{_MANIFEST_HEADER}\n{body}"


def merge_entries(existing: list[SourceEntry], scanned: list[SourceEntry]) -> list[SourceEntry]:
    """Merge freshly-scanned files into the existing manifest.

    Preserves the user-curated ``url`` for files still present, refreshes their
    ``sha256`` / ``size_bytes``, appends newly-seen files (``url=None``), and keeps
    existing entries whose file is absent on this machine (they stay
    re-downloadable). Order: existing entries first (stable), then new files.
    """
    by_name = {e.filename: e for e in scanned}
    seen: set[str] = set()
    merged: list[SourceEntry] = []
    for prior in existing:
        seen.add(prior.filename)
        fresh = by_name.get(prior.filename)
        if fresh is None:
            merged.append(prior)  # not on this box; keep as-is
        else:
            merged.append(
                SourceEntry(
                    filename=prior.filename,
                    url=prior.url,
                    sha256=fresh.sha256,
                    size_bytes=fresh.size_bytes,
                )
            )
    merged.extend(fresh for fresh in scanned if fresh.filename not in seen)
    return merged


def enrich_with_public_corpus(entries: list[SourceEntry], public: list[SourceEntry]) -> int:
    """Fill ``url`` in-place for url-less entries that match the public corpus.

    Matches by ``sha256`` first (exact bytes), then by filename. Returns how many
    entries were enriched.
    """
    by_sha = {e.sha256: e for e in public if e.sha256}
    by_name = {e.filename: e for e in public}
    filled = 0
    for entry in entries:
        if entry.url:
            continue
        match: SourceEntry | None = None
        if entry.sha256 is not None:
            match = by_sha.get(entry.sha256)
        if match is None:
            match = by_name.get(entry.filename)
        if match is not None and match.url:
            entry.url = match.url
            filled += 1
    return filled


# --------------------------------------------------------------------------- #
# Impure boundary (filesystem + network)
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    """Stream a file's SHA-256 (constant memory)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_READ_BLOCK), b""):
            h.update(block)
    return h.hexdigest()


def scan_sources(sources_dir: Path) -> list[SourceEntry]:
    """Scan ``sources_dir`` for ingestable files, pinned by sha256 + size.

    Recurses; only files ingest would pick up (``is_supported``) are included.
    Filenames are stored relative to ``sources_dir`` (POSIX) so nested files keep
    distinct keys and round-trip on re-download. Sorted for a stable manifest.
    """
    if not sources_dir.exists():
        return []
    entries: list[SourceEntry] = []
    for path in sorted(sources_dir.rglob("*")):
        if path.is_file() and is_supported(path):
            entries.append(
                SourceEntry(
                    filename=path.relative_to(sources_dir).as_posix(),
                    url=None,
                    sha256=sha256_file(path),
                    size_bytes=path.stat().st_size,
                )
            )
    return entries


def load_manifest(path: Path) -> list[SourceEntry]:
    """Load the manifest, or an empty list if it does not exist yet."""
    if not path.exists():
        return []
    return parse_manifest(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, entries: list[SourceEntry]) -> None:
    """Write the manifest (creating parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_manifest(entries), encoding="utf-8")


def load_public_corpus(path: Path = PUBLIC_CORPUS_MANIFEST) -> list[SourceEntry]:
    """Load the committed public-corpus manifest as entries (for URL auto-fill)."""
    if not path.exists():
        return []
    return parse_manifest(path.read_text(encoding="utf-8"))


def build_manifest(
    *,
    sources_dir: Path | None = None,
    manifest_path: Path | None = None,
    public_corpus_path: Path = PUBLIC_CORPUS_MANIFEST,
) -> BuildResult:
    """Scan sources, merge into the existing manifest, auto-fill public-corpus URLs.

    Reads only — the caller persists the result with ``write_manifest`` (so a
    ``--dry-run`` can preview without writing).
    """
    sources_dir = sources_dir if sources_dir is not None else config.DOCS_PATH
    manifest_path = manifest_path if manifest_path is not None else config.SOURCES_MANIFEST
    existing = load_manifest(manifest_path)
    prior_names = {e.filename for e in existing}
    scanned = scan_sources(sources_dir)
    merged = merge_entries(existing, scanned)
    enriched = enrich_with_public_corpus(merged, load_public_corpus(public_corpus_path))
    new = sum(1 for e in merged if e.filename not in prior_names)
    missing = sum(1 for e in merged if not e.url)
    return BuildResult(
        entries=merged,
        total=len(merged),
        new=new,
        enriched=enriched,
        missing_url=missing,
    )


def _http_get(url: str) -> bytes:
    """Fetch ``url`` over http(s). Raises ``ValueError`` for any other scheme."""
    if not url.lower().startswith(("http://", "https://")):
        raise ValueError(f"unsupported URL scheme (http/https only): {url}")
    request = urllib.request.Request(url, headers=_USER_AGENT)
    # Scheme restricted to http/https above, so bandit's B310 url-open audit is satisfied.
    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT_S) as resp:  # nosec B310
        return bytes(resp.read())


def download_entry(entry: SourceEntry, dest_dir: Path) -> SyncOutcome:
    """Download one entry into ``dest_dir`` and checksum-verify it if pinned."""
    if not entry.url:
        return SyncOutcome(entry.filename, "no_url")
    try:
        data = _http_get(entry.url)
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return SyncOutcome(entry.filename, "failed", str(exc))
    target = dest_dir / entry.filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if entry.sha256:
        actual = hashlib.sha256(data).hexdigest()
        if actual != entry.sha256:
            return SyncOutcome(
                entry.filename,
                "mismatch",
                f"expected {entry.sha256[:12]}…, got {actual[:12]}…",
            )
    return SyncOutcome(entry.filename, "downloaded")


def download_missing(
    entries: list[SourceEntry], dest_dir: Path, *, dry_run: bool = False
) -> list[SyncOutcome]:
    """Fetch every entry whose file is absent from ``dest_dir`` and has a URL."""
    outcomes: list[SyncOutcome] = []
    for entry in entries:
        target = dest_dir / entry.filename
        if target.exists():
            outcomes.append(SyncOutcome(entry.filename, "present"))
        elif not entry.url:
            outcomes.append(SyncOutcome(entry.filename, "no_url"))
        elif dry_run:
            outcomes.append(SyncOutcome(entry.filename, "would_download", entry.url))
        else:
            outcomes.append(download_entry(entry, dest_dir))
    return outcomes


def verify_present(entries: list[SourceEntry], dest_dir: Path) -> list[SyncOutcome]:
    """Checksum every entry already on disk against its manifest ``sha256``."""
    outcomes: list[SyncOutcome] = []
    for entry in entries:
        target = dest_dir / entry.filename
        if not target.exists():
            continue
        if not entry.sha256:
            outcomes.append(SyncOutcome(entry.filename, "present", "no sha256 to check"))
            continue
        status = "present" if sha256_file(target) == entry.sha256 else "mismatch"
        outcomes.append(SyncOutcome(entry.filename, status))
    return outcomes
