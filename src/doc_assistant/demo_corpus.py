"""Demo-corpus folder membership — automatic at ingest, plus a one-time backfill (ADR-025 F3).

The public corpus manifest (``tests/eval/corpus_manifest.yaml``) carries two collections: the
closed **eval** set behind the committed benchmarks, and a **demo** set (classic DL papers) fetched
with ``download_corpus --demo``. This module puts the demo set into an ordinary folder — the same
``Folder`` rows F1 created and F2 taught to scope retrieval — so "chat with just the demo papers"
works without a second organizing concept (ADR-025 fork 1, the ADR-015 reuse pattern).

**The manifest is an origin signal, never an ongoing authority.** A document is considered exactly
once, on the ingest run that first created its ``Document`` row; after that the folder is the
user's. Remove a demo paper from it and nothing ever puts it back (ADR-013 user-wins; spec M2).
The single honest exception is ``ingest --rebuild``, which deletes every row so every document
looks new — that is logged, not hidden (M3).

Matching is by file **bytes** (size fast-path, then SHA-256), never by name, so a renamed demo file
is still recognised — the same rule ``library.match_pinned_sources`` uses for ``--remove-demo``, so
one definition of "is a demo file" serves both directions (M4).

Degrades to a silent no-op when the manifest is absent — the normal state of a PyInstaller-frozen
build, where ``PROJECT_ROOT`` points into a temp unpack dir and ``tests/`` is not bundled. That is
coherent rather than broken: the demo corpus is a repo-clone flow end to end, so a packaged
install has no demo files to assign either (M10).

Contract: ``docs/specs/feature-corpus-folders-demo.md``.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import structlog
import yaml  # type: ignore[import-untyped]
from sqlalchemy import select

from doc_assistant import app_settings, config
from doc_assistant.db.models import Document
from doc_assistant.db.session import session_scope
from doc_assistant.library import (
    FolderSummary,
    SourceMatch,
    SourcePin,
    add_documents_to_folder,
    create_folder,
    folder_document_ids,
    get_folder,
    match_pinned_sources,
)
from doc_assistant.sources_manifest import sha256_file

log = structlog.get_logger(__name__)

MANIFEST_PATH = config.PROJECT_ROOT / "tests" / "eval" / "corpus_manifest.yaml"
DEMO_COLLECTION = "demo"
DEFAULT_FOLDER_NAME = "Demo corpus"
DEFAULT_FOLDER_DESCRIPTION = (
    "Classic deep-learning papers from the public demo collection "
    "(scripts/download_corpus.py --demo). An ordinary folder — rename it, "
    "edit it, or delete it; nothing will put a document you removed back."
)


@dataclass(frozen=True)
class AssignResult:
    """Outcome of one assignment pass — nothing happened when ``folder_id`` is None."""

    folder_id: str | None = None
    folder_name: str | None = None
    added: list[str] = field(default_factory=list)
    already_member: int = 0


# --------------------------------------------------------------------------- #
# Manifest pins
# --------------------------------------------------------------------------- #
def load_demo_pins(manifest_path: Path | None = None) -> list[SourcePin]:
    """Content pins for the manifest's **demo** collection ( ``[]`` when unavailable).

    An entry with no ``collection`` field is an eval-corpus entry (the pre-demo manifest carried
    no such field), matching ``download_corpus._selected``. A missing, blank, malformed, or
    unreadable manifest yields ``[]`` — the caller then no-ops (M10), so this never raises.
    """
    path = manifest_path if manifest_path is not None else MANIFEST_PATH
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.debug("demo_manifest_absent", path=str(path))
        return []
    except (OSError, yaml.YAMLError) as e:
        log.warning("demo_manifest_unreadable", path=str(path), error=str(e))
        return []
    documents = (raw or {}).get("documents") if isinstance(raw, dict) else None
    if not documents:
        return []
    pins: list[SourcePin] = []
    for entry in documents:
        if not isinstance(entry, dict) or entry.get("collection", "eval") != DEMO_COLLECTION:
            continue
        filename, digest, size = entry.get("filename"), entry.get("sha256"), entry.get("bytes")
        if not filename or not digest or size is None:
            log.warning("demo_pin_incomplete", filename=filename)
            continue
        pins.append(SourcePin(str(filename), str(digest), int(size)))
    return pins


def pins_by_size(pins: Sequence[SourcePin]) -> dict[int, list[SourcePin]]:
    """Group pins by byte size — the fast path that keeps a big corpus at stat-call cost."""
    grouped: dict[int, list[SourcePin]] = {}
    for pin in pins:
        grouped.setdefault(pin.size_bytes, []).append(pin)
    return grouped


def file_matches_demo(path: Path, by_size: Mapping[int, Sequence[SourcePin]]) -> bool:
    """Whether ``path``'s exact bytes match a demo pin (name-independent — M4).

    Only a size collision costs a read; an unreadable/absent file is simply not a match (this runs
    right after a successful ingest, so that would mean the file moved mid-run).
    """
    try:
        candidates = by_size.get(path.stat().st_size)
        if not candidates:
            return False
        digest = sha256_file(path)
    except OSError as e:
        log.debug("demo_match_unreadable", path=str(path), error=str(e))
        return False
    return any(pin.sha256 == digest for pin in candidates)


# --------------------------------------------------------------------------- #
# The folder
# --------------------------------------------------------------------------- #
def resolve_demo_folder(*, create: bool) -> FolderSummary | None:
    """The folder holding the demo corpus, optionally creating it.

    Resolution is by the **persisted id**, not by name, so renaming the folder is respected —
    ADR-025 promises an ordinary, renamable folder, and a name-keyed lookup would silently create a
    second "Demo corpus" the first time someone renamed theirs (M5). If the id is unset or points
    at a folder that no longer exists, ``create=True`` makes one (``create_folder`` is an
    idempotent get-or-create, so a hand-made folder of the same name is adopted, not duplicated);
    ``create=False`` returns None.
    """
    stored = app_settings.get_demo_folder_id()
    if stored is not None:
        existing = get_folder(stored)
        if existing is not None:
            return existing
    if not create:
        return None
    folder = create_folder(DEFAULT_FOLDER_NAME, description=DEFAULT_FOLDER_DESCRIPTION)
    app_settings.set_demo_folder_id(folder.id)
    log.info("demo_folder_ready", folder_id=folder.id, name=folder.name)
    return folder


def apply_assignments(document_ids: Sequence[str]) -> AssignResult:
    """Add ``document_ids`` to the demo folder, creating it only if there is something to add.

    Idempotent: documents already in the folder are counted, not re-added. An empty list creates
    **no** folder — an empty "Demo corpus" nobody asked for would be exactly the honest-empty rule
    F1 follows, inverted (M7).
    """
    if not document_ids:
        return AssignResult()
    folder = resolve_demo_folder(create=True)
    if folder is None:  # pragma: no cover - create=True always returns a folder
        return AssignResult()
    current = set(folder_document_ids(folder.id))
    fresh = [doc_id for doc_id in document_ids if doc_id not in current]
    if fresh:
        add_documents_to_folder(folder.id, fresh)
    log.info(
        "demo_folder_assigned",
        folder_id=folder.id,
        folder=folder.name,
        added=len(fresh),
        already_member=len(document_ids) - len(fresh),
    )
    return AssignResult(
        folder_id=folder.id,
        folder_name=folder.name,
        added=fresh,
        already_member=len(document_ids) - len(fresh),
    )


# --------------------------------------------------------------------------- #
# The two entry points
# --------------------------------------------------------------------------- #
def assign_new_documents(doc_hashes: Collection[str]) -> AssignResult:
    """The ingest hook: assign whichever **newly created** document rows are demo files.

    ``doc_hashes`` is the set of ``Document.doc_hash`` values whose rows did not exist before this
    ingest run (``ingest.main`` computes it as a set-difference around the processing loop). Keying
    on *new rows* rather than on "was processed" is what stops a re-ingest — the inverse-orphan
    repair, or a ``--path`` rerun — from re-adding a document the user removed by hand (M1/M2).
    """
    if not doc_hashes:
        return AssignResult()
    pins = load_demo_pins()
    if not pins:
        return AssignResult()
    by_size = pins_by_size(pins)
    with session_scope() as session:
        rows = session.execute(
            select(Document.id, Document.source_original).where(
                Document.doc_hash.in_(list(doc_hashes))
            )
        ).all()
    matched = [
        str(doc_id)
        for doc_id, source in rows
        if source and file_matches_demo(Path(str(source)), by_size)
    ]
    if not matched:
        return AssignResult()
    return apply_assignments(matched)


def backfill_matches(sources_dir: Path | None = None) -> list[SourceMatch]:
    """Demo files on disk with their library rows — the one-time backfill's plan.

    Reuses ``match_pinned_sources`` (the same content-hash scan ``--remove-demo`` runs), so the
    files a backfill assigns are exactly the files a removal would take away. A match with no
    ``document_id`` is a file that was never ingested (or whose row is ambiguous) — reported, not
    assigned.
    """
    pins = load_demo_pins()
    if not pins:
        return []
    root = sources_dir if sources_dir is not None else app_settings.get_source_dir()
    return match_pinned_sources(pins, root)
