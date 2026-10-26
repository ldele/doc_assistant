"""``TurnEvent`` — the tagged union ``handle_message`` streams.

Streamed ``Token``s and ``Step`` status updates, terminating in a ``Result`` that wraps the
finished ``TurnResult`` — or in a ``Failure`` when the answer could not be produced."""

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


@dataclass
class Failure:
    """The terminal event of a turn that stopped before it finished: one sentence the reader can
    act on, naming the engine that failed. Never a traceback; the log has that."""

    message: str


TurnEvent = Token | Step | Result | Failure
