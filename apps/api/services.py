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

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI

from doc_assistant import app_settings
from doc_assistant.chat_controller import ChatController
from doc_assistant.config import DATA_PATH
from doc_assistant.embeddings import get_active_model_name

log = structlog.get_logger(__name__)

SUPPORTED_NOTE = "pdf · epub · html · docx · md (and similar text formats)"


def _settings_view() -> dict[str, Any]:
    """A read-only view of the locked + env-toggleable knobs (Phase-8 settings UI is later)."""
    from doc_assistant.config import (
        BM25_WEIGHT,
        CANDIDATE_K,
        CHILD_CHUNK_OVERLAP,
        CHILD_CHUNK_SIZE,
        PAID_PROVIDERS,
        PARENT_CHUNK_OVERLAP,
        PARENT_CHUNK_SIZE,
        REVIEWER_EVIDENCE_CHARS,
        SYNTHESIS_MODE,
        TOP_K,
        USE_MULTI_QUERY,
        USE_PARENT_CHILD,
    )
    from doc_assistant.llm import provider_available

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
        # labeling (fork D/KI-4), so a keyless provider renders disabled with its reason.
        "providers": [
            {
                "id": p,
                "available": provider_available(p),
                "paid": p in PAID_PROVIDERS,
            }
            for p in ("anthropic", "ollama")
        ],
    }


def _full_settings(app: FastAPI) -> dict[str, Any]:
    """Read view: the locked knobs plus the data home, the user's source folder, chunk_count."""
    controller: ChatController = app.state.controller
    source = app_settings.get_source_dir()
    return {
        **_settings_view(),
        "data_home": str(DATA_PATH),
        "source_dir": str(source),
        "source_dir_exists": source.is_dir(),
        "supported_formats": SUPPORTED_NOTE,
        "chunk_count": controller.chunk_count(),
    }


@dataclass
class _IngestStatus:
    """Background-ingest progress, read by GET /api/ingest/status (guarded by a lock)."""

    state: str = "idle"  # idle | running | done | error
    source_dir: str | None = None
    added: int = 0
    skipped: int = 0
    errors: int = 0
    message: str | None = None


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


def _ingest_status_dict(app: FastAPI) -> dict[str, Any]:
    st: _IngestStatus = app.state.ingest_status
    with app.state.ingest_lock:
        return {
            "state": st.state,
            "source_dir": st.source_dir,
            "added": st.added,
            "skipped": st.skipped,
            "errors": st.errors,
            "message": st.message,
        }


def _graph_rebuild_status_dict(app: FastAPI) -> dict[str, Any]:
    st: _GraphRebuildStatus = app.state.graph_rebuild_status
    with app.state.graph_rebuild_lock:
        return {"state": st.state, "graph_version": st.graph_version, "message": st.message}


def _default_ingest(
    *, scope: str | None = None, files: list[Path] | None = None
) -> dict[str, int]:
    """Lazy wrapper so importing this module doesn't pull the heavy ingest -> torch chain.

    ``scope`` = the whole source dir (honoring exclusions); ``files`` = an explicit, pre-resolved
    selection (selective ingestion, S1). The two are mutually exclusive at the ``main()`` layer.
    """
    from doc_assistant.ingest import main as ingest_main

    return ingest_main(scope=scope, files=files)


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
