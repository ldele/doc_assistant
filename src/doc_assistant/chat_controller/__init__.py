"""UI-agnostic turn orchestration (PR-M0 — Tauri desktop-shell migration).

The whole RAG/integrity turn — slash-command dispatch, pending claim-edit handling, library-query
routing, history-aware rewrite, retrieval, figure lookup, source assembly, ``SYNTHESIS_MODE=human``
branch, answer streaming, provenance capture, confidence-signal gating + (flagged-only) reviewer
call, claim segmentation + eager persistence, citation audit, usage accounting, and export
stashing — used to live inside the original web-UI ``on_message`` handler, interleaved with UI
rendering.

This package lifts that orchestration out of the UI so any frontend renders the same value object.
``ChatController.handle_message`` yields a stream of :class:`TurnEvent` (streamed ``Token``s +
``Step`` status updates) terminating in a :class:`Result` wrapping a :class:`TurnResult`.

**No UI-framework import here.** Behaviour is frozen and guarded by
``tests/integration/test_turn_parity.py``.

**Layout.** ``session`` (caller-owned state, ADR-3) · ``views`` (the pure render payload that
``apps/api/models/chat.py`` mirrors) · ``events`` (the ``TurnEvent`` union) · ``helpers`` (pure
formatters + turn-knob resolution) · ``controller`` (``ChatController``). The dependency direction
is strictly session/views → events/helpers → controller.

Prefer importing the sub-module when you know it; the flat re-export below keeps
``from doc_assistant.chat_controller import X`` working for the existing call sites — including
the tests that reach for ``_resolve_turn_knobs``.

**Monkeypatching:** patch a helper on the module that *owns* it (``…chat_controller.helpers``),
not on this package — a re-exported name is a separate binding (see `src/doc_assistant/CLAUDE.md`).

See ``docs/archive/pr-m0-chat-controller.md`` and
``docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md``.
"""

from __future__ import annotations

# Re-exported so `chat_controller.app_settings.X` resolves for callers/tests. Patching an
# ATTRIBUTE on this shared module object is fine through any binding; REPLACING a name
# (e.g. is_library_query) must target `chat_controller.controller` instead.
from doc_assistant import app_settings
from doc_assistant.chat_controller.controller import (
    ChatController,
)
from doc_assistant.chat_controller.events import (
    Result,
    Step,
    Token,
    TurnEvent,
)
from doc_assistant.chat_controller.helpers import (
    _build_claim_review,
    _build_claims_block,
    _build_retrieved_chunks,
    _build_source_views,
    _chunk_key,
    _export_sources,
    _format_provenance_card,
    _format_review_block,
    _is_local,
    _marker_chip,
    _overrides_note,
    _ProvenanceInputs,
    _ProvenanceOutcome,
    _resolve_scope,
    _resolve_turn_knobs,
    _scope_dict,
    _scope_label,
    _scope_note,
    _sources_block,
    _token_suffix,
)
from doc_assistant.chat_controller.session import (
    RagOverrides,
    Session,
    _TurnKnobs,
)
from doc_assistant.chat_controller.views import (
    ClaimView,
    ScopeView,
    SourceEpistemics,
    SourceEvalSummary,
    SourceView,
    TurnResult,
    UsageView,
)

__all__ = [
    "ChatController",
    "ClaimView",
    "RagOverrides",
    "Result",
    "ScopeView",
    "Session",
    "SourceEpistemics",
    "SourceEvalSummary",
    "SourceView",
    "Step",
    "Token",
    "TurnEvent",
    "TurnResult",
    "UsageView",
    "_ProvenanceInputs",
    "_ProvenanceOutcome",
    "_TurnKnobs",
    "_build_claim_review",
    "_build_claims_block",
    "_build_retrieved_chunks",
    "_build_source_views",
    "_chunk_key",
    "_export_sources",
    "_format_provenance_card",
    "_format_review_block",
    "_is_local",
    "_marker_chip",
    "_overrides_note",
    "_resolve_scope",
    "_resolve_turn_knobs",
    "_scope_dict",
    "_scope_label",
    "_scope_note",
    "_sources_block",
    "_token_suffix",
    "app_settings",
]
