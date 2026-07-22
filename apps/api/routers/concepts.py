"""Concepts router — the concept-graph read model, the gap list + triage, and the rebuild trigger.

Read-only over the vocabulary by decision (ADR-017 A1): the graph observes, the Manage-keywords
routes (in the library router) edit. The one write here is the rebuild *trigger*, which runs the
same idempotent runner the CLI does (202 + poll, ADR-017 B1). Route order is preserved from the
pre-split module — ``/api/concepts/gaps`` stays declared before ``/api/concepts/{concept_id}/…``.
"""

from __future__ import annotations

import threading
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from apps.api.models import (
    ConceptGraphPayload,
    ConceptPresencePayload,
    GapListItemPayload,
    GapTriageRequest,
)
from apps.api.services import _graph_rebuild_status_dict, _GraphRebuildStatus, log

router = APIRouter()


@router.get("/api/concepts/graph")
def get_concept_graph() -> ConceptGraphPayload:
    """The concept-graph read model: nodes + edges + communities + gaps + a staleness verdict.

    **404 when the skeleton has never been built** — ``skeleton.json`` is a gitignored,
    regenerable sidecar, so a fresh clone legitimately has none. That is an empty state the UI
    answers with the rebuild button below, not a server fault; the detail says so.

    Ids are concept **UUIDs** throughout (nodes, edge endpoints, gap anchors, community
    members); ``label`` rides only on the node. One id space, deliberately — see KI-15.
    """
    from doc_assistant.knowledge.concept_graph_view import load_graph_view

    view = load_graph_view()
    if view is None:
        raise HTTPException(
            status_code=404,
            detail="concept graph not built yet — run a rebuild to create it",
        )
    return ConceptGraphPayload.from_view(view)


@router.get("/api/concepts/gaps")
def get_gap_list() -> list[GapListItemPayload]:
    """The first-class gap list (ROADMAP E5, ADR-004/ADR-017 C1): every detected corpus gap
    with its concept label resolved and its **effective** triage status (a user override wins).

    A pure sidecar read — no model, no LLM. Empty list when no gaps are built (0-doc /
    pre-``build_gaps`` — the normal first-run state), never an error. Presentation order (the
    RG-014 strong-kinds-first lens) is the client's; this serves the detector's order."""
    from doc_assistant.knowledge.concept_graph_view import load_gap_list

    return [GapListItemPayload.from_item(item) for item in load_gap_list()]


@router.post("/api/concepts/gaps/triage")
def triage_gap(body: GapTriageRequest) -> dict[str, bool]:
    """Record (or reset) a user's triage verdict on one gap (ADR-017 C1, E5).

    The verdict lives in the ``gap_triage`` override sidecar keyed on ``(concept_id, kind)``,
    so it survives ``build_gaps``'s delete-and-rebuild of the deterministic rows — a dismissal
    means something across the acquire loop. ``surfaced`` resets to the detector's default
    (deletes the override). Idempotent; no validation beyond the enum (the gap need not still
    exist — triaging a gap the next rebuild will drop is harmless and self-cleaning)."""
    from doc_assistant.knowledge.gaps import set_gap_status

    set_gap_status(body.concept_id, body.kind, body.status)
    return {"ok": True}


@router.get("/api/concepts/{concept_id}/presence")
def get_concept_presence(concept_id: str) -> list[ConceptPresencePayload]:
    """Where one concept appears: its documents + the chunk keys that mention it.

    Per-concept, not bulk — the view renders one neighbourhood at a time (ego-first), and the
    chunk-key set grows with the vocabulary. An unknown concept returns ``[]`` (a concept that
    is present nowhere is indistinguishable here, by design — the vocabulary is the authority
    on existence)."""
    from doc_assistant.knowledge.concept_graph_view import load_concept_presence

    return [ConceptPresencePayload.from_presence(p) for p in load_concept_presence(concept_id)]


@router.post("/api/concepts/graph/rebuild", status_code=202)
def rebuild_concept_graph(request: Request) -> dict[str, Any]:
    """Rebuild the concept skeleton (Node A: ~7s, zero-LLM, deterministic, idempotent).

    202 + poll, mirroring ``POST /api/ingest`` — the repo's established shape for a
    derived-data build triggered from the app (ADR-017 B1). 409 while one is already running.
    """
    app_ = request.app
    status: _GraphRebuildStatus = app_.state.graph_rebuild_status
    with app_.state.graph_rebuild_lock:
        if status.state == "running":
            raise HTTPException(status_code=409, detail="graph rebuild already running")
        status.state = "running"
        status.message = None

    def _worker() -> None:
        try:
            version = app_.state.rebuild_graph_fn()
        except Exception as e:  # a failed rebuild must not kill the thread silently
            log.exception("graph_rebuild_failed")
            with app_.state.graph_rebuild_lock:
                status.state = "error"
                status.message = str(e)
            return
        with app_.state.graph_rebuild_lock:
            status.graph_version = version
            status.message = "graph rebuilt"
            status.state = "done"

    threading.Thread(target=_worker, name="graph-rebuild", daemon=True).start()
    return _graph_rebuild_status_dict(app_)


@router.get("/api/concepts/graph/rebuild/status")
def concept_graph_rebuild_status(request: Request) -> dict[str, Any]:
    return _graph_rebuild_status_dict(request.app)
