"""The single logging-configuration seam (ADR-003 / `docs/specs/structlog-observability.md`).

``configure_logging`` is the **only** place that configures logging. It wires
``structlog`` on top of the standard library (``structlog.stdlib.LoggerFactory`` +
``ProcessorFormatter``) so the library's own ``structlog.get_logger`` events and any
third-party stdlib logs (chromadb, httpx, transformers, …) flow through one renderer.

Pure setup only — **no business logic and no app imports** (this module must stay
importable without pulling in ``doc_assistant.config`` or any ``apps/`` code). Each app
entrypoint (``apps/cli.py``, ``apps/api``) and the ``python -m``
program entrypoints (``doc_assistant.ingest``, ``doc_assistant.db.migrations``) call it
once, early. ``src/`` *library* code never calls it: importing a library module must have
no logging side effect, and if ``configure_logging`` never runs, structlog falls back to
its own defaults — loggers still work, just unconfigured.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

# Third-party loggers that emit chatty INFO/DEBUG noise (model downloads, HTTP
# round-trips, telemetry). Pin them to WARNING so the app's own events stay
# readable. They still flow through the same renderer when they do fire.
_NOISY_LIBRARIES = (
    "chromadb",
    "httpx",
    "httpcore",
    "urllib3",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "filelock",
)

# Marks the handler this module installs, so a re-call removes only our own handler
# (idempotent) without disturbing handlers another framework (e.g. pytest) added.
_HANDLER_FLAG = "_doc_assistant_handler"


def _shared_processors() -> list[Any]:
    """Processors applied to BOTH structlog events and foreign stdlib records, so a
    log line looks the same whoever emitted it."""
    return [
        structlog.contextvars.merge_contextvars,  # bound context (bind_contextvars)
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]


def configure_logging(*, json: bool = False, level: str = "INFO") -> None:
    """Configure structlog + stdlib logging for one process. Idempotent (last call wins).

    Args:
        json: ``True`` renders newline-delimited JSON (machine consumption / a deployed,
            observed FastAPI context); ``False`` renders human-readable console lines
            (dev/CLI default) so converted ``print()`` progress stays visible.
        level: Root log level name (e.g. ``"INFO"``, ``"WARNING"``). ``"INFO"`` keeps
            the migrated progress lines visible.
    """
    shared = _shared_processors()

    # structlog's chain ends by handing the event dict to the stdlib formatter; the
    # actual rendering happens in that formatter (below). So the json/console choice
    # lives entirely in the handler's formatter — re-calling with a different `json`
    # just swaps the formatter, and cached structlog loggers keep working.
    structlog.configure(
        processors=[*shared, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    if json:
        render_processors: list[Any] = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ]
    else:
        render_processors = [
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            # ConsoleRenderer renders exc_info itself; only colour a real TTY so piped
            # output / the frozen sidecar's captured stderr stays clean ASCII.
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ]

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared,
        processors=render_processors,
    )

    handler = logging.StreamHandler()  # stderr by default
    handler.setFormatter(formatter)
    setattr(handler, _HANDLER_FLAG, True)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if getattr(existing, _HANDLER_FLAG, False):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level.upper())

    for name in _NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
