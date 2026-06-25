"""In-memory session store for the desktop API (PR-M2, ADR-3).

Single-user, **process-scoped**, no persistence, no eviction in v1 — an app restart
clears it, consistent with the original per-chat reset. One ``SessionStore`` lives
per app instance (held on ``app.state``); the ``session_id`` is a caller-supplied key, so
multi-user / persisted sessions later is a non-breaking change (same rationale as the
provenance UUIDs).

NOTE: not guarded against two concurrent in-flight turns on the *same* ``session_id`` —
the frontend (PR-M3) must prevent overlapping sends for one conversation.
"""

from __future__ import annotations

from doc_assistant.chat_controller import Session


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def get_or_create(self, session_id: str) -> Session:
        """Return the session for ``session_id``, creating it on first use."""
        session = self._sessions.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        """Return the session for ``session_id`` or ``None`` (→ a 404 for claims/export)."""
        return self._sessions.get(session_id)
