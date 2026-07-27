"""Per-conversation session state (ADR-3).

Caller-owned and injected into every call, so two frontends sharing one ``ChatController``
singleton cannot leak overrides into each other."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from doc_assistant import export
from doc_assistant.tracking import TokenCounter

# ============================================================
# Session state (ADR-3) — caller-owned, injected into every call
# ============================================================


@dataclass
class Session:
    """Per-conversation state. Caller-owned; injected into every ``ChatController``
    call. The controller holds no per-turn state in globals — it is stateless across
    turns except for this injected object (multi-session later is non-breaking)."""

    history: list[dict[str, str]] = field(default_factory=list)
    counter: TokenCounter = field(default_factory=TokenCounter)
    export_turns: list[export.ExportTurn] = field(default_factory=list)
    awaiting_edit: dict[str, Any] | None = None
    session_id: str = field(default_factory=lambda: time.strftime("%Y%m%d-%H%M%S"))


@dataclass(frozen=True)
class RagOverrides:
    """Session-scoped, per-turn RAG knob overrides (ADR-010 / feature-rag-sandbox.md).
    ``None`` (a field or the whole object) = use the locked default. Non-persistent: never
    written to config/.env/app_settings, and never assigned to a module global — threaded
    as an explicit request-scoped parameter so concurrent turns on the shared
    ``ChatController`` singleton cannot leak overrides into each other.

    ``epistemics_markers_enabled``/``reviewer_evidence_chars`` (U1b, SPRINT-011, ADR-010's
    2026-07-10 amendment) are the two "must revisit" niche knobs — same non-persistent,
    request-scoped mechanics as the original three."""

    top_k: int | None = None
    synthesis_mode: str | None = None  # "ai" | "human"
    use_multi_query: bool | None = None
    epistemics_markers_enabled: bool | None = None
    reviewer_evidence_chars: int | None = None


@dataclass(frozen=True)
class _TurnKnobs:
    """The effective per-turn RAG knobs (ADR-010), resolved once from a ``RagOverrides`` plus the
    locked config defaults, together with the provenance ``overrides_note`` derived from them. A
    ``None`` field (or ``overrides=None``) = the locked default. See ``_resolve_turn_knobs``."""

    top_k: int
    synthesis_mode: str
    multi_query: bool
    markers_enabled: bool
    reviewer_evidence_chars: int
    overrides_note: str
