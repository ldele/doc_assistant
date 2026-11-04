"""What a Re-index of the library folder would do, and roughly how long it would take.

The Re-index confirmation (the user, 2026-11-04) asks two things before a run starts: how many
files it will read, and how long that takes. Both come from here.

**The counts** are the registry's stat-only scan (`registry.scan_sources`), no extraction and no
model: a file is *new* (never indexed), *changed* (indexed, but its source is newer than its cache,
so it is read again) or *unchanged* (skipped at ~0.4 ms each, `docs/performance.md`). Only the
library folder counts, because that is what a Re-index walks; files referenced from elsewhere are
indexed when they are added. Excluded files are skipped by a Re-index and counted apart.

**The time** is measured, never assumed. Every run that indexes at least one document records its
seconds per document on this computer (:func:`record_speed`), and the estimate is that figure times
the documents to read. No run yet means no estimate, and the screen says so: a constant typed in
here would be a corpus-tuned number about someone else's machine (the robustness contract). It is
rough by nature: a cold PDF took 5.7 to 36.7 s to extract in the measurements behind
`docs/performance.md`, so a per-document mean is honest about a batch, never about one file.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from doc_assistant import app_settings
from doc_assistant.db.models import LIBRARY_ROOT_ID

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

log = structlog.get_logger(__name__)

_SPEED_KEY = "ingest_speed"


@dataclass(frozen=True)
class IngestPlan:
    """A Re-index of the library folder, before it runs."""

    new: int
    new_bytes: int
    changed: int
    changed_bytes: int
    unchanged: int
    excluded: int
    #: Seconds per document indexed on this computer's last run, or ``None`` before any run.
    seconds_per_document: float | None
    #: How many documents that figure was measured on (0 when there is none).
    measured_documents: int

    def to_read(self, only_new: bool) -> int:
        """How many files the run reads: new ones, plus changed ones unless ``only_new``."""
        return self.new if only_new else self.new + self.changed

    def seconds(self, only_new: bool) -> float | None:
        """The rough duration, or ``None`` when nothing has been measured yet."""
        if self.seconds_per_document is None:
            return None
        return self.to_read(only_new) * self.seconds_per_document


def plan_library(session: Session, source_dir: Path) -> IngestPlan:
    """Count what a Re-index of ``source_dir`` would read, from the registry's stat-only scan."""
    from doc_assistant.ingest import registry

    counts = {"new": 0, "changed": 0, "unchanged": 0, "excluded": 0}
    sizes = {"new": 0, "changed": 0}
    for view in registry.scan_sources(session, source_dir):
        if view.root_id != LIBRARY_ROOT_ID or view.status == registry.STATUS_MISSING:
            continue
        if view.excluded:
            counts["excluded"] += 1
        elif view.status == registry.STATUS_NEW:
            counts["new"] += 1
            sizes["new"] += view.size
        elif view.status == registry.STATUS_CHANGED:
            counts["changed"] += 1
            sizes["changed"] += view.size
        else:
            counts["unchanged"] += 1
    per_doc, measured = measured_speed()
    return IngestPlan(
        new=counts["new"],
        new_bytes=sizes["new"],
        changed=counts["changed"],
        changed_bytes=sizes["changed"],
        unchanged=counts["unchanged"],
        excluded=counts["excluded"],
        seconds_per_document=per_doc,
        measured_documents=measured,
    )


def new_file_keys(session: Session, source_dir: Path) -> list[str]:
    """The library folder's never-indexed files, as keys for `registry.resolve_selection`.

    What *Only index new files* runs: changed files keep their current version, excluded files stay
    skipped. A file that failed every time is still *new* (it is not in the library), so it is
    tried again, which is what the person asked for.
    """
    from doc_assistant.ingest import registry

    return [
        registry.source_key(v.root_id, v.rel_path)
        for v in registry.scan_sources(session, source_dir)
        if v.root_id == LIBRARY_ROOT_ID and v.status == registry.STATUS_NEW and not v.excluded
    ]


def measured_speed() -> tuple[float | None, int]:
    """``(seconds per document, documents measured)`` from the last run, or ``(None, 0)``."""
    raw = app_settings.load_user_settings().get(_SPEED_KEY)
    if not isinstance(raw, dict):
        return None, 0
    seconds, documents = raw.get("seconds_per_document"), raw.get("documents")
    if not isinstance(seconds, int | float) or not isinstance(documents, int):
        return None, 0
    if seconds <= 0 or documents <= 0:
        return None, 0
    return float(seconds), documents


def record_speed(seconds: float, documents: int) -> None:
    """Keep this run's seconds per document for the next estimate. Never fails a run.

    ``seconds`` is the work time only: extraction warm-up plus each indexed document, never the
    time a run spent paused. The last run replaces the one before, so a new machine or a new
    extractor shows up in the very next estimate.
    """
    if documents <= 0 or seconds <= 0:
        return
    try:
        settings = app_settings.load_user_settings()
        settings[_SPEED_KEY] = {
            "seconds_per_document": round(seconds / documents, 3),
            "documents": documents,
            "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        app_settings.save_user_settings(settings)
    except Exception as e:  # an estimate for next time must never cost this run
        log.warning("ingest_speed_not_recorded", error=str(e))
