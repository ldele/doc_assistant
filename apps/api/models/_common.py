"""Helpers shared by more than one wire-model module.

Deliberately tiny: anything that grows a domain of its own belongs in that domain's module,
not here. ``_common`` exists so ``conversations`` and ``library`` can share one timestamp
coercion without importing each other.
"""

from __future__ import annotations

from datetime import datetime, timezone


def _as_utc(dt: datetime) -> datetime:
    """Tag a naive DB timestamp (``AnswerRecord.created_at`` is naive UTC) as UTC so the ISO wire
    value carries an offset — otherwise a browser ``new Date()`` reads it as *local* time."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
