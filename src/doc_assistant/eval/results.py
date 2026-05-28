"""Shared result dataclasses for the eval harness (generic).

These are the wire format between the system-under-test, the scorers,
the runner, and the store. Plain dataclasses; no third-party deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class EvalOutput:
    """What the system-under-test returns for one query.

    ``citations`` is a list of opaque identifiers (filenames, doc IDs,
    URLs — whatever the system retrieves). Scorers that compare against
    ``EvalCase.expected_citations`` use string equality.
    """

    answer: str
    citations: list[str] = field(default_factory=list)
    token_input: int | None = None
    token_output: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScoreResult:
    """One scorer's verdict on one case.

    Convention: a scorer that couldn't grade the case (missing
    ``expected_answer`` / ``expected_citations`` / etc.) returns
    ``value=0.0`` with an ``"error"`` key in ``details``. The
    ``is_skipped`` property surfaces this distinction so aggregations
    can separate "scored zero" from "didn't run".
    """

    scorer_name: str
    value: float
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_skipped(self) -> bool:
        """True when the scorer didn't actually grade — used to filter aggregates."""
        return "error" in self.details


@dataclass
class EvalResult:
    """One full run-output for one case: latency, optional error, all scorer verdicts."""

    case_id: str
    output: EvalOutput | None
    scores: list[ScoreResult] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None
    timestamp: datetime = field(default_factory=_utcnow)
