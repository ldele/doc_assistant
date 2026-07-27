"""``TurnEvent`` — the tagged union ``handle_message`` streams.

Streamed ``Token``s and ``Step`` status updates, terminating in a ``Result`` that wraps the
finished ``TurnResult``."""

from __future__ import annotations

from dataclasses import dataclass

from doc_assistant.chat_controller.views import TurnResult

# ============================================================
# TurnEvent — a tagged union streamed by handle_message
# ============================================================


@dataclass
class Token:
    """One streamed answer-token delta."""

    text: str


@dataclass
class Step:
    """A progress status update (retrieval / rewrite). Advisory; renderers may show it."""

    name: str
    status: str


@dataclass
class Result:
    """The terminal event: the finished TurnResult."""

    result: TurnResult


TurnEvent = Token | Step | Result
