"""Shared app-state helpers for the FastAPI shell — the cross-router glue (APIRouter split).

Everything here is **router-agnostic**: the background-job status dataclasses that ride on
``app.state``, the two ``202 + poll`` status serializers, the settings read view, and the lazy
default job runners. Kept in one module (imported by ``main`` and by the routers, importing
neither back) so the split has no circular imports — ``main`` wires ``app.state`` from these and
each router reads them via ``request.app.state``.

No route lives here. ``apps/`` stays a thin shell (CONTEXT rule 3): these functions only read
``ChatController``/``app_settings``/config and format a dict — they own no retrieval, provenance,
or ingest logic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from fastapi import FastAPI

from doc_assistant import app_settings
from doc_assistant.chat_controller import ChatController
from doc_assistant.config import DATA_PATH
from doc_assistant.embeddings import get_active_model_name

if TYPE_CHECKING:  # the ingest package pulls the torch chain; the run is only typed here
    from doc_assistant.ingest.queue import IngestQueue
    from doc_assistant.ingest.run import RunSink

log = structlog.get_logger(__name__)


def _supported_note() -> str:
    """The formats line Settings shows, from the extractor registry rather than a second list.

    It was a hand-written string that had drifted: it left out ODT and RTF while the uploader's
    own hard-coded line left out TXT (found 2026-09-16).
    """
    from doc_assistant.library.add import accepted_input

    return " · ".join(e.lstrip(".") for e in accepted_input()["extensions"] if e != ".htm")


def _settings_view() -> dict[str, Any]:
    """A read-only view of the locked + env-toggleable knobs (Phase-8 settings UI is later)."""
    from doc_assistant.config import (
        BM25_WEIGHT,
        CANDIDATE_K,
        CHILD_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        PARENT_CHUNK_OVERLAP,
        PARENT_CHUNK_SIZE,
        REVIEWER_EVIDENCE_CHARS,
        SYNTHESIS_MODE,
        TOP_K,
        USE_MULTI_QUERY,
        USE_PARENT_CHILD,
    )
    from doc_assistant.llm import provider_available
    from doc_assistant.providers import answer_provider_ids, describe

    # ADR-011 (U1c): the *effective* provider/model — the persisted selection if the user has
    # switched, else the config default — never the import-time LLM_PROVIDER/LLM_MODEL constants
    # (those would go stale the moment a switch happens).
    eff_provider, eff_model = app_settings.effective_llm()

    return {
        "provider": eff_provider,
        "model": eff_model,
        "embedding_model": get_active_model_name(),
        "top_k": TOP_K,
        "candidate_k": CANDIDATE_K,
        "use_parent_child": USE_PARENT_CHILD,
        "synthesis_mode": SYNTHESIS_MODE,
        # The locked defaults for the RAG-sandbox toggles (ADR-010 + the U1b amendment) — the
        # sandbox section needs these to render each control's un-overridden state correctly.
        "use_multi_query": USE_MULTI_QUERY,
        # ADR-027 D2 (E3): the *effective* answer-layer default (persisted choice if set, else
        # the config default) — never the raw constant, which would go stale the moment the
        # user toggles it (the same rule as provider/model above). Doubles as the sandbox
        # baseline: U1b's un-overridden state IS the persisted default now.
        "epistemics_markers_enabled": app_settings.effective_markers_enabled(),
        "reviewer_evidence_chars": REVIEWER_EVIDENCE_CHARS,
        "parent_chunk": [PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP],
        "child_chunk": [CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP],
        # ADR-010 "fix in passing": sourced from config, not a hardcoded literal — the
        # read-only display can no longer silently drift from the real BM25_WEIGHT.
        "retrieval_weights": {"bm25": BM25_WEIGHT, "vector": round(1 - BM25_WEIGHT, 3)},
        # ADR-011 (U1c) — the provider picker's option list: availability (fork E) + paid/local
        # labeling (fork D/KI-4), so a keyless provider renders disabled with its reason. Since
        # ADR-060 `paid` is the guard's test, billed or not known, from local state only.
        "providers": [
            {
                "id": p,
                "available": provider_available(p),
                "paid": describe(p).billing.guard_trips,
            }
            for p in answer_provider_ids()
        ],
    }


def _full_settings(app: FastAPI) -> dict[str, Any]:
    """Read view: the locked knobs plus the data home, the user's source folder, corpus facts."""
    controller: ChatController = app.state.controller
    source = app_settings.get_source_dir()
    chunks = controller.chunk_count()
    return {
        **_settings_view(),
        "data_home": str(DATA_PATH),
        "source_dir": str(source),
        "source_dir_exists": source.is_dir(),
        "supported_formats": _supported_note(),
        "chunk_count": chunks,
        # ADR-037: what this corpus costs on this machine. Assembled by the controller (it needs
        # the live pipeline's arm, not the presence of a file); the shell only serializes it.
        "corpus": controller.corpus_stats().as_dict(),
    }


def setup_state_dict(app: FastAPI, *, probe: bool = True) -> dict[str, Any]:
    """Serialize ``readiness.setup_state`` for ``GET /api/setup`` (ADR-034).

    The counts come from here because the shell is what holds the controller and the library
    session; the *judgement* (what is ready, what to do next) is entirely
    ``doc_assistant.readiness`` — this function adds no rules of its own.

    ``document_count`` degrades to 0 if the library cannot be read: a first-run install may have no
    schema yet, and a setup view that 500s is worse than one reporting "nothing indexed" — which is
    the truth in that state anyway.
    """
    from dataclasses import asdict

    from doc_assistant.readiness import setup_state

    controller: ChatController = app.state.controller
    try:
        from doc_assistant.library import count_documents

        documents = count_documents()
    except Exception as e:
        log.warning("setup_document_count_failed", error=str(e))
        documents = 0
    state = setup_state(
        chunk_count=controller.chunk_count(), document_count=documents, probe=probe
    )
    return asdict(state)


@dataclass
class _IngestStatus:
    """Background-ingest progress, read by GET /api/ingest/status (guarded by a lock)."""

    state: str = "idle"  # idle | running | done | error
    source_dir: str | None = None
    added: int = 0
    skipped: int = 0
    errors: int = 0
    message: str | None = None
    #: Progress, written per document by the `on_progress` sink the route installs. `total` is
    #: known before the first document, so this is a real fraction rather than a spinner. `added`
    #: / `skipped` / `errors` stay end-of-run totals — they are the *outcome*, not the position,
    #: and reporting a partial outcome as if it were final is the failure this pair avoids.
    total: int = 0
    done: int = 0
    #: The file in flight, or None when nothing is (before the first document, and after the last).
    current: str | None = None


@dataclass
class _ReingestStatus:
    """Background per-part re-ingest progress (ADR-048), read by the re-ingest status route.

    Mirrors ``_IngestStatus`` for the same reason ``_GraphRebuildStatus`` does — this repo's shape
    for app-triggered background work is 202 + poll. It carries the same position/outcome split:
    ``total``/``done``/``current`` are *where the run is* and are written per (document, part),
    while ``ok``/``skipped``/``errors`` are end-of-run totals. Reporting a partial outcome as a
    final one is the failure that pair exists to prevent.

    ``outcomes`` is the per-(document, part) report the panel renders when the run finishes — a
    part that could not run says so, with its reason. It is capped because a row-21 selection can
    be large and this dict is serialized on every poll.
    """

    state: str = "idle"  # idle | running | done | error
    total: int = 0
    done: int = 0
    current: str | None = None
    ok: int = 0
    skipped: int = 0
    errors: int = 0
    message: str | None = None
    outcomes: list[dict[str, str]] | None = None


#: How many per-(document, part) outcomes the status route will carry. A row-21 run over the whole
#: library would otherwise put thousands of rows on every poll; the counts above stay exact.
REINGEST_OUTCOME_CAP = 200


@dataclass
class _GraphRebuildStatus:
    """Background concept-skeleton rebuild progress, read by the rebuild status route.

    Mirrors ``_IngestStatus`` deliberately (ADR-017 B1): the rebuild is ~7s of deterministic,
    zero-LLM work, and this repo's established shape for a derived-data build triggered from the
    app is 202 + poll, not a blocking request.
    """

    state: str = "idle"  # idle | running | done | error
    graph_version: str | None = None
    message: str | None = None


def ingest_queue(app: FastAPI) -> IngestQueue:
    """The engine's one list (A8″, `docs/specs/ingestion-queue.md`), made on first use.

    Lazily, because importing anything under `doc_assistant.ingest` pulls the torch chain, which
    this module keeps out of an import. ``ingest_fn`` and ``reingest_fn`` are read from
    ``app.state`` at call time, so a test that swaps either after `create_app` is honoured."""
    queue: IngestQueue | None = getattr(app.state, "ingest_queue", None)
    if queue is None:
        from doc_assistant.ingest.queue import IngestQueue

        def drained(superseded: list[str], changed: bool) -> None:
            # The run's end, once, when the list is empty: forget what the re-runs replaced, at
            # once, then reload so the new corpus is live (as after an ingest; S18).
            controller = app.state.controller
            for doc_hash in superseded:
                forget = getattr(controller, "forget_document", None)
                if forget is not None:
                    forget(doc_hash)
            if not (changed or superseded):
                return
            try:
                release = getattr(controller, "release_keyword_index", None)
                if release is not None:
                    release()  # KI-77: Windows keeps an open file
                app.state.controller = app.state.controller_factory()
            except Exception:
                log.exception("controller_reload_failed_after_ingest")

        queue = IngestQueue(
            ingest_fn=lambda **kw: app.state.ingest_fn(**kw),
            rerun_fn=lambda ids, parts: app.state.reingest_fn(ids, parts),
            on_drained=drained,
        )
        app.state.ingest_queue = queue
    return queue


def _ingest_status_dict(app: FastAPI) -> dict[str, Any]:
    """The one list's status (A8″): its state, outcome totals, position and counts.

    The same fields as before the list was one, so the status bar and the indexing screen read it
    unchanged; ``total`` grows as items are added."""
    st: _IngestStatus = app.state.ingest_status
    queue: IngestQueue | None = getattr(app.state, "ingest_queue", None)
    run = queue.run if queue is not None else None
    if queue is None or run is None:
        return {
            "state": "idle",
            "source_dir": st.source_dir,
            "added": 0,
            "skipped": 0,
            "errors": 0,
            "message": None,
            "total": 0,
            "done": 0,
            "current": None,
            "control": "none",
            "counts": None,
        }
    counts = run.counts()
    finished = counts["added"] + counts["unchanged"] + counts["done"] + counts["failed"]
    tally = dict(queue.tally)
    message: str | None = None
    if queue.state == "error":
        message = queue.error
    elif queue.state in ("done", "stopped"):
        message = (
            f"indexed {tally['added']} new, {tally['skipped']} unchanged, {tally['error']} errors"
        )
    return {
        "state": queue.state,
        "source_dir": st.source_dir,
        "added": tally["added"],
        "skipped": tally["skipped"],
        "errors": tally["error"],
        "message": message,
        "total": counts["total"],
        "done": finished,
        "current": run.current(),
        # ROADMAP 31: what pause or stop has been asked, and each state's count of files.
        # Counts only, never the files: at 10,000 documents a poll cannot carry the list, so
        # the screen reads it a page at a time from GET /api/ingest/files.
        "control": run.control() if queue.state == "running" else "none",
        "counts": counts,
    }


def _reingest_status_dict(app: FastAPI) -> dict[str, Any]:
    """The latest re-run request's status, read from its rows in the one list (A8″).

    The same fields the re-run dialog reads, plus ``waiting_behind``: how many rows wait or run
    ahead of it, so a re-run asked during a long index says what it is waiting for."""
    ticket: list[str] = list(getattr(app.state, "reingest_ticket", []) or [])
    queue: IngestQueue | None = getattr(app.state, "ingest_queue", None)
    run = queue.run if queue is not None else None
    if not ticket or queue is None or run is None:
        return {
            "state": "idle",
            "total": 0,
            "done": 0,
            "current": None,
            "ok": 0,
            "skipped": 0,
            "errors": 0,
            "message": None,
            "outcomes": [],
            "waiting_behind": 0,
        }
    rows = []
    for item_id in ticket:
        try:
            rows.append(run.item(item_id))
        except KeyError:
            continue
    live = [r for r in rows if r.state in ("waiting", "reading")]
    outcomes = [o for r in rows for o in queue.outcomes.get(r.id, [])]
    ok = sum(1 for o in outcomes if o.status == "ok")
    skipped = sum(1 for o in outcomes if o.status == "skipped")
    errors = sum(1 for o in outcomes if o.status == "error")
    state = "running" if live else "done"
    message = None
    crashed = [r for r in rows if r.state == "failed" and not queue.outcomes.get(r.id)]
    if state == "done" and crashed and len(crashed) == len(rows):
        state, message = "error", crashed[0].detail  # the re-run raised: say why, as before
    elif state == "done":
        bits = [f"{ok} re-run"] + ([f"{skipped} skipped"] if skipped else [])
        message = " · ".join(bits + ([f"{errors} failed"] if errors else []))
    waiting = [r for r in rows if r.state == "waiting"]
    return {
        "state": state,
        "total": len(rows),
        "done": len(rows) - len(live),
        "current": next((r.name for r in rows if r.state == "reading"), None),
        "ok": ok,
        "skipped": skipped,
        "errors": errors,
        "message": message,
        "outcomes": [
            {
                "document_id": o.document_id,
                "filename": o.filename,
                "part": o.part,
                "status": o.status,
                "detail": o.detail,
            }
            for o in outcomes[:REINGEST_OUTCOME_CAP]
        ],
        "waiting_behind": run.ahead_of(waiting[0].id) if waiting else 0,
    }


def _graph_rebuild_status_dict(app: FastAPI) -> dict[str, Any]:
    st: _GraphRebuildStatus = app.state.graph_rebuild_status
    with app.state.graph_rebuild_lock:
        return {"state": st.state, "graph_version": st.graph_version, "message": st.message}


def _default_reingest(
    document_ids: list[str],
    parts: list[str],
    *,
    on_progress: Callable[[int, int, str | None], None] | None = None,
) -> Any:
    """Lazy wrapper over ``reingest.rerun`` — same reason as ``_default_ingest``: importing this
    module must not pull the ingest -> torch chain, and only the `text` part ever needs it."""
    from doc_assistant.reingest import rerun

    return rerun(document_ids, parts, on_progress=on_progress)


def _default_ingest(
    *,
    scope: str | None = None,
    files: list[Path] | None = None,
    on_progress: Callable[[int, int, str | None], None] | None = None,
    run: RunSink | None = None,
) -> dict[str, int]:
    """Lazy wrapper so importing this module doesn't pull the heavy ingest -> torch chain.

    ``scope`` = the whole source dir (honoring exclusions); ``files`` = an explicit, pre-resolved
    selection (selective ingestion, S1). The two are mutually exclusive at the ``main()`` layer.
    ``on_progress`` is forwarded verbatim; the type is spelled out here rather than imported so
    this module keeps its lazy-import property. ``run`` carries each file's outcome and the
    pause and stop controls (ROADMAP 31).
    """
    from doc_assistant.ingest import main as ingest_main

    return ingest_main(scope=scope, files=files, on_progress=on_progress, run=run)


def _default_rebuild_graph() -> str:
    """Lazy wrapper so importing this module doesn't pull the heavy concept-skeleton chain.

    Returns the new ``graph_version``. The CLI runners (``build_concept_skeleton`` then
    ``build_gaps``) stay the canonical seam — this is a second caller of the *same* idempotent
    functions, which is what ADR-017 B1 decided the Enrichment-Layer Pattern permits (it constrains
    what derived data is, not who triggers it).

    E0.3 / KI-21: the skeleton rebuild is chained into ``build_gaps`` so the acquire loop the
    button exists to close (gap → ingest → rebuild → gap closes) actually closes in-app — otherwise
    ``load_graph_view`` keeps serving gaps computed against the *previous* skeleton, including a
    gap the user just closed. ``min_degree`` is derived from the rebuilt skeleton's own degree
    distribution (E0.3 — no hardcoded literal). Both passes are deterministic + zero-LLM; the plain
    ``build_concept_skeleton(apply=True)`` preserves Node-B stance (E0.5b), so a rebuild does not
    silently darken epistemics.
    """
    from doc_assistant.knowledge.concept_skeleton import build_concept_skeleton
    from doc_assistant.knowledge.gaps import build_gaps, derive_min_degree

    result = build_concept_skeleton(apply=True)
    build_gaps(apply=True, min_degree=derive_min_degree(result.skeleton))
    return str(result.skeleton.meta.get("graph_version", ""))
