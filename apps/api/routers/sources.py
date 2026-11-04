"""Sources router — selective ingestion: the registry scan/patch + the background ingest job."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request

from apps.api.models.sources import (
    AcceptsPayload,
    AddOutcomePayload,
    AddRequest,
    AddResultPayload,
    CatalogueScanRequest,
    CatalogueScanResponse,
    FileVerdictPayload,
    IngestFilePayload,
    IngestFilesPayload,
    IngestPlanPayload,
    IngestRequest,
    InspectRequest,
    InspectResponse,
    SourceFilePayload,
    SourcePatch,
    UndoAddRequest,
)
from apps.api.services import _ingest_status_dict, _IngestStatus, log
from doc_assistant import app_settings
from doc_assistant.db.session import session_scope

if TYPE_CHECKING:  # the ingest package pulls the torch chain; imported where a run starts
    from doc_assistant.ingest.run import IngestRun

#: The indexing screen's filters and the follow-up runs, spelled as `doc_assistant.ingest.run`
#: spells them, so the route signatures need no import of the ingest package at module load.
FileGroup = Literal["all", "in_progress", "waiting", "not_started", "finished", "failed"]
RerunScope = Literal["failed", "remaining"]

router = APIRouter()


@router.post("/api/ingest", status_code=202)
def ingest_start(request: Request, body: IngestRequest | None = None) -> dict[str, Any]:
    app_ = request.app
    # Resolve once so the whole endpoint speaks the canonical path. The registry resolves
    # source_dir everywhere it scans (scan_sources / resolve_selection / view_for), so the
    # `files=` selection is always resolved — the `scope=` branch and `status.source_dir` must
    # agree, or they diverge on Windows (case-normalized username, 8.3 short paths) and any
    # symlinked source dir. `get_source_dir` already resolves in prod; this is idempotent.
    source = app_settings.get_source_dir().resolve()

    # Selective ingestion (S1): an explicit `paths` selection is resolved + validated up front
    # so a bad path is a 400 *before* anything starts running (nothing partial). No body (or
    # null paths) keeps today's behavior — the whole source dir, now minus standing exclusions.
    selection: list[Path] | None = None
    if body is not None and body.only_new:
        selection = _new_files_selection(source, body)
    elif body is not None and body.rerun is not None:
        selection = _rerun_selection(app_, body.rerun, body.paths)
    elif body is not None and body.paths is not None:
        from doc_assistant.ingest import registry

        with session_scope() as session:
            try:
                selection = registry.resolve_selection(session, source, body.paths)
            except registry.InvalidSelection as e:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid selection", "offenders": e.offenders},
                ) from e

    from doc_assistant.ingest.run import IngestRun

    run = IngestRun()
    status: _IngestStatus = app_.state.ingest_status
    with app_.state.ingest_lock:
        if status.state == "running":
            raise HTTPException(status_code=409, detail="ingest already running")
        app_.state.ingest_run = run
        status.state = "running"
        status.source_dir = str(source)
        status.added = status.skipped = status.errors = 0
        status.message = None
        # Cleared with the counters, or the bar would open showing the *previous* run's position.
        status.total = status.done = 0
        status.current = None

    def _on_progress(done: int, total: int, current: str | None) -> None:
        """Copy one `ingest.main` progress tick onto the shared status.

        Runs on the ingest thread, so it takes the same lock every reader takes. It only ever
        *writes position* — never `added`/`skipped`/`errors`, which stay end-of-run outcomes.
        """
        with app_.state.ingest_lock:
            status.done, status.total, status.current = done, total, current

    def _worker() -> None:
        try:
            if selection is not None:
                stats = app_.state.ingest_fn(files=selection, on_progress=_on_progress, run=run)
            else:
                stats = app_.state.ingest_fn(scope=str(source), on_progress=_on_progress, run=run)
        except Exception as e:  # surface any ingest failure to the status view
            log.exception("ingest_failed", source=str(source))
            with app_.state.ingest_lock:
                status.state = "error"
                status.message = str(e)
                status.current = None  # nothing is in flight any more; the bar must not say it is
            return
        # Reload the controller so the new corpus is live BEFORE reporting "done" — a client
        # that sees "done" then reads chunk_count must get the updated count (BM25 + Chroma
        # handles are built at construction). A reload failure is logged but doesn't fail the
        # ingest: the documents are persisted, a restart picks them up, so a successful ingest
        # never flips to "error".
        try:
            app_.state.controller.release_keyword_index()  # KI-77: Windows keeps an open file
            app_.state.controller = app_.state.controller_factory()
        except Exception:
            log.exception("controller_reload_failed_after_ingest")
        with app_.state.ingest_lock:
            status.added = int(stats.get("added", 0))
            status.skipped = int(stats.get("skipped", 0))
            status.errors = int(stats.get("error", 0))
            status.message = (
                f"indexed {status.added} new, {status.skipped} unchanged, {status.errors} errors"
            )
            status.current = None
            # A stopped run ends like a finished one (its documents are live), and says so.
            status.state = "stopped" if run.stopped else "done"

    threading.Thread(target=_worker, name="ingest", daemon=True).start()
    return _ingest_status_dict(app_)


@router.get("/api/ingest/status")
def ingest_status(request: Request) -> dict[str, Any]:
    return _ingest_status_dict(request.app)


def _new_files_selection(source: Path, body: IngestRequest) -> list[Path]:
    """*Only index new files*: the library folder's never-indexed files, or a 400 saying why."""
    from doc_assistant.ingest import registry
    from doc_assistant.ingest.estimate import new_file_keys

    if body.paths is not None or body.rerun is not None:
        raise HTTPException(status_code=400, detail="only_new takes no paths and no rerun")
    with session_scope() as session:
        keys = new_file_keys(session, source)
        if not keys:
            raise HTTPException(status_code=400, detail="the library folder has no new files")
        return registry.resolve_selection(session, source, keys)


@router.get("/api/ingest/plan")
def ingest_plan() -> IngestPlanPayload:
    """What a Re-index of the library folder would read, and roughly how long it would take.

    Stat-only: no extraction, no model, safe to call each time the confirmation opens.
    """
    from doc_assistant.ingest.estimate import plan_library

    with session_scope() as session:
        plan = plan_library(session, app_settings.get_source_dir().resolve())
    return IngestPlanPayload(
        new=plan.new,
        new_bytes=plan.new_bytes,
        changed=plan.changed,
        changed_bytes=plan.changed_bytes,
        unchanged=plan.unchanged,
        excluded=plan.excluded,
        seconds_all=plan.seconds(only_new=False),
        seconds_new=plan.seconds(only_new=True),
        seconds_per_document=plan.seconds_per_document,
        measured_documents=plan.measured_documents,
    )


def _rerun_selection(app_: Any, rerun: RerunScope, paths: list[str] | None) -> list[Path]:
    """The files ``rerun`` names from the last run, or a 400 saying why there are none."""
    if paths is not None:
        raise HTTPException(status_code=400, detail="send paths or rerun, not both")
    last: IngestRun | None = app_.state.ingest_run
    if last is None:
        raise HTTPException(status_code=400, detail="there is no earlier run to take files from")
    selection = last.paths(rerun)
    if not selection:
        raise HTTPException(status_code=400, detail=f"the last run has no {rerun} files")
    return selection


def _live_run(request: Request) -> IngestRun:
    """The run in flight, or a 409: pause and stop act on a running run only."""
    status: _IngestStatus = request.app.state.ingest_status
    with request.app.state.ingest_lock:
        run: IngestRun | None = request.app.state.ingest_run
        if status.state != "running" or run is None:
            raise HTTPException(status_code=409, detail="no indexing run is in progress")
        return run


@router.post("/api/ingest/pause")
def ingest_pause(request: Request) -> dict[str, Any]:
    """Pause after the document in flight (ROADMAP 31). Searching and chatting go on as usual."""
    _live_run(request).pause()
    return _ingest_status_dict(request.app)


@router.post("/api/ingest/resume")
def ingest_resume(request: Request) -> dict[str, Any]:
    _live_run(request).resume()
    return _ingest_status_dict(request.app)


@router.post("/api/ingest/stop")
def ingest_stop(request: Request) -> dict[str, Any]:
    """End the run after the document in flight. What finished stays; the next run skips it."""
    _live_run(request).stop()
    return _ingest_status_dict(request.app)


@router.get("/api/ingest/files")
def ingest_files(
    request: Request,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    group: FileGroup = "all",
) -> IngestFilesPayload:
    """One page of the current or last run's files, in the screen's order (ROADMAP 31).

    Empty, not 404, before any run: a screen opened on a fresh app has nothing to list.
    """
    run: IngestRun | None = request.app.state.ingest_run
    if run is None:
        return IngestFilesPayload(total=0, offset=offset, files=[])
    rows, total = run.page(offset, limit, group)
    return IngestFilesPayload(
        total=total,
        offset=offset,
        files=[
            IngestFilePayload(path=f.path, name=f.name, state=f.state, detail=f.detail)
            for f in rows
        ],
    )


@router.get("/api/sources")
def list_sources(request: Request) -> list[SourceFilePayload]:
    """Selective ingestion (S1): stat-only scan of the source dir → each file with a
    derived ingest status (new / changed / ingested / missing) + its ``excluded`` flag."""
    from doc_assistant.ingest import registry

    source = app_settings.get_source_dir()
    with session_scope() as session:
        return [SourceFilePayload.from_view(v) for v in registry.scan_sources(session, source)]


@router.patch("/api/sources")
def patch_source(body: SourcePatch, request: Request) -> SourceFilePayload:
    """Update one registry row's intent (v1: ``excluded``). 404 if rel_path is unknown."""
    from doc_assistant.ingest import registry

    source = app_settings.get_source_dir()
    with session_scope() as session:
        try:
            registry.set_source_meta(
                session, body.rel_path, excluded=body.excluded, root_id=body.root_id
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"unknown source: {body.rel_path}") from e
        view = registry.view_for(session, source, body.rel_path, root_id=body.root_id)
        if view is None:  # unreachable (set_source_meta would have raised) — narrows for mypy
            raise HTTPException(status_code=404, detail=f"unknown source: {body.rel_path}")
        return SourceFilePayload.from_view(view)


@router.post("/api/catalogue/zotero/scan")
def scan_zotero(body: CatalogueScanRequest) -> CatalogueScanResponse:
    """Read a Zotero library and return the files it holds — **staging nothing** (ROADMAP 17).

    The deliberate shape of this route is that it stops where the existing add flow begins. It
    returns absolute paths; the client stages them; the AD2 review sheet then says what would
    happen to each one, including which are already in the library, and AD3 applies the user's
    choice of copy or reference. Zotero gets no second add path, no second duplicate rule and no
    second progress bar — which is also why importing a library you have already imported is
    simply an add where everything reads as a duplicate.

    The one thing it *does* write is `ExternalMetadata`: what the catalogue says about each file,
    recorded now because this is when the catalogue is open. It is applied to documents at the
    end of the next ingest (`ingest._apply_imported_metadata`), and it is keyed by path, so
    recording it for a file the user then declines to add costs one unused row.

    A missing or unreadable library is a **404 with a sentence in it**, not a 500: not having
    Zotero installed is an ordinary state of the world, not a server fault.
    """
    from doc_assistant.adapters import zotero
    from doc_assistant.adapters.catalogue import CatalogueUnavailable, record_external

    try:
        scan = zotero.read_library(
            Path(body.data_dir) if body.data_dir else None,
            base_dir=Path(body.base_dir) if body.base_dir else None,
            include_snapshots=body.include_snapshots,
        )
    except CatalogueUnavailable as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    if scan.documents:
        with session_scope() as session:
            record_external(session, scan.documents, source=zotero.SOURCE)

    return CatalogueScanResponse(
        label=scan.label,
        root=str(scan.root),
        paths=[str(d.path) for d in scan.documents],
        found=len(scan.documents),
        skipped=dict(scan.skipped),
        with_metadata=sum(1 for d in scan.documents if d.title),
    )


@router.get("/api/documents/accepts")
def document_accepts() -> AcceptsPayload:
    """What an add accepts — formats and the limits — so the uploader can state them up front.

    Served, not hard-coded in the frontend: the limits are env-configurable (security S-1/S-2), and
    a stated number the backend does not enforce would be a lie in the one place users read first.
    """
    from doc_assistant.library.add import accepted_input

    return AcceptsPayload(**accepted_input())


@router.post("/api/documents/inspect")
def inspect_documents(body: InspectRequest) -> InspectResponse:
    """AD2 — say what would happen to each candidate path. **Mutates nothing the user can see.**

    This is the review sheet's only data source, and it is deliberately a separate call from the
    one that applies anything (AD3): spec constraint 2 is that nothing is copied, registered or
    indexed before the sheet has been shown and confirmed, and two endpoints is how that stays
    true rather than remembered.

    Directories expand recursively server-side, matching `registry.scan_sources`. An empty
    `paths` is a valid request with an empty answer — a drop that yielded nothing is not an error.
    """
    from doc_assistant.library.add import (
        AddBatchTooLargeError,
        inspect,
        sort_for_review,
        summarise,
    )

    try:
        verdicts = inspect([Path(p) for p in body.paths])
    except AddBatchTooLargeError as e:  # security S-2: a pick of a whole drive, say so
        raise HTTPException(status_code=400, detail=str(e)) from e
    return InspectResponse(
        files=[FileVerdictPayload.from_verdict(v) for v in sort_for_review(verdicts)],
        counts=summarise(verdicts),
    )


def _running(request: Request) -> bool:
    """True while an ingest is in flight. Adding during one would race the registry scan."""
    status: _IngestStatus = request.app.state.ingest_status
    with request.app.state.ingest_lock:
        return bool(status.state == "running")


@router.post("/api/documents/add")
def add_documents(request: Request, body: AddRequest) -> AddResultPayload:
    """AD3 — copy the chosen files into the library folder and register them.

    **Does not index.** Indexing is the caller's next step through the existing `POST /api/ingest`
    with an explicit `paths` list (spec constraint 4): one ingest path in the system, not two.

    Both placement modes work since AD3b: ``copy`` copies into the library folder, ``reference``
    registers the file where it already lives under its own root. 409 while an ingest is running,
    mirroring `/api/ingest` itself.
    """
    from doc_assistant.library.add import AddBatchTooLargeError, AddOutcome, apply_add

    if _running(request):
        raise HTTPException(status_code=409, detail="ingest already running")

    try:
        result = apply_add(
            [Path(p) for p in body.paths],
            mode=body.mode,
            reference_root=Path(body.reference_root) if body.reference_root else None,
        )
    except AddBatchTooLargeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    def out(o: AddOutcome) -> AddOutcomePayload:
        return AddOutcomePayload(
            path=o.path, name=o.name, ok=o.ok, rel_path=o.rel_path, key=o.key, error=o.error
        )

    return AddResultPayload(
        added=[out(o) for o in result.added],
        failed=out(result.failed) if result.failed else None,
        not_attempted=result.not_attempted,
        stopped_early=result.stopped_early,
    )


@router.post("/api/documents/undo-add")
def undo_add_documents(request: Request, body: UndoAddRequest) -> dict[str, int]:
    """Reverse a just-completed add — the *Undo all* half of grill branch 6.

    Only removes rows whose `origin` is `copied`, so it can never delete a file the app does not
    own (the ADR-014 amendment, enforced in the library rather than trusted to the caller).

    The controller's Chroma handle is passed through so undo can also remove the *document* the
    add produced (KI-51 part 1) — without it, undoing an already-indexed add left the row and its
    chunks behind, and the library went on listing a document whose file undo had just deleted.
    """
    from doc_assistant.library.add import undo_add

    if _running(request):
        raise HTTPException(status_code=409, detail="ingest already running")
    controller = request.app.state.controller
    undone = undo_add(
        body.rel_paths, chroma_db=controller.rag.db, forget=controller.forget_document
    )
    return {"undone": undone}
