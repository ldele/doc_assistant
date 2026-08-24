"""Sources router — selective ingestion: the registry scan/patch + the background ingest job."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models.sources import (
    AddOutcomePayload,
    AddRequest,
    AddResultPayload,
    FileVerdictPayload,
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
    if body is not None and body.paths is not None:
        from doc_assistant.ingest import registry

        with session_scope() as session:
            try:
                selection = registry.resolve_selection(session, source, body.paths)
            except registry.InvalidSelection as e:
                raise HTTPException(
                    status_code=400,
                    detail={"error": "invalid selection", "offenders": e.offenders},
                ) from e

    status: _IngestStatus = app_.state.ingest_status
    with app_.state.ingest_lock:
        if status.state == "running":
            raise HTTPException(status_code=409, detail="ingest already running")
        status.state = "running"
        status.source_dir = str(source)
        status.added = status.skipped = status.errors = 0
        status.message = None

    def _worker() -> None:
        try:
            if selection is not None:
                stats = app_.state.ingest_fn(files=selection)
            else:
                stats = app_.state.ingest_fn(scope=str(source))
        except Exception as e:  # surface any ingest failure to the status view
            log.exception("ingest_failed", source=str(source))
            with app_.state.ingest_lock:
                status.state = "error"
                status.message = str(e)
            return
        # Reload the controller so the new corpus is live BEFORE reporting "done" — a client
        # that sees "done" then reads chunk_count must get the updated count (BM25 + Chroma
        # handles are built at construction). A reload failure is logged but doesn't fail the
        # ingest: the documents are persisted, a restart picks them up, so a successful ingest
        # never flips to "error".
        try:
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
            status.state = "done"

    threading.Thread(target=_worker, name="ingest", daemon=True).start()
    return _ingest_status_dict(app_)


@router.get("/api/ingest/status")
def ingest_status(request: Request) -> dict[str, Any]:
    return _ingest_status_dict(request.app)


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
            registry.set_source_meta(session, body.rel_path, excluded=body.excluded)
        except KeyError as e:
            raise HTTPException(status_code=404, detail=f"unknown source: {body.rel_path}") from e
        view = registry.view_for(session, source, body.rel_path)
        if view is None:  # unreachable (set_source_meta would have raised) — narrows for mypy
            raise HTTPException(status_code=404, detail=f"unknown source: {body.rel_path}")
        return SourceFilePayload.from_view(view)


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
    from doc_assistant.library.add import inspect, sort_for_review, summarise

    verdicts = inspect([Path(p) for p in body.paths])
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

    409 while an ingest is running, mirroring `/api/ingest` itself. 501 for `mode="reference"`,
    which ADR-046 decided and AD3b will build — refusing beats silently copying to a location the
    user did not choose.
    """
    from doc_assistant.library.add import apply_add

    if body.mode not in ("copy", "reference"):
        raise HTTPException(status_code=422, detail=f"unknown mode {body.mode!r}")
    if _running(request):
        raise HTTPException(status_code=409, detail="ingest already running")

    try:
        result = apply_add([Path(p) for p in body.paths], mode=body.mode)  # type: ignore[arg-type]
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e

    def out(o: object) -> AddOutcomePayload:
        return AddOutcomePayload(
            path=o.path,
            name=o.name,
            ok=o.ok,
            rel_path=o.rel_path,
            error=o.error,  # type: ignore[attr-defined]
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
    """
    from doc_assistant.library.add import undo_add

    if _running(request):
        raise HTTPException(status_code=409, detail="ingest already running")
    return {"undone": undo_add(body.rel_paths)}
