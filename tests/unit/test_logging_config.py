"""Tests for the logging seam (ADR-003 / `docs/specs/structlog-observability.md`).

Covers the four behaviours the spec's DoD calls out: renderer selection (console vs
JSON), level filtering, idempotency, and that the module configures logging without
importing any app code. Output is captured by swapping the installed handler's stream
to a buffer — deterministic, independent of pytest's stdout/stderr capture.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest
import structlog

from doc_assistant.logging_config import _HANDLER_FLAG, configure_logging


def _installed_handler() -> logging.Handler:
    root = logging.getLogger()
    return next(h for h in root.handlers if getattr(h, _HANDLER_FLAG, False))


def _capture(*, json: bool, level: str = "INFO") -> io.StringIO:
    """Configure logging then redirect the installed handler at a buffer."""
    configure_logging(json=json, level=level)
    buf = io.StringIO()
    handler = _installed_handler()
    assert isinstance(handler, logging.StreamHandler)
    handler.setStream(buf)
    return buf


def test_console_renderer_is_human_readable() -> None:
    buf = _capture(json=False)
    structlog.get_logger("test.console").warning("collection_missing", collection="bge")
    out = buf.getvalue()
    assert "collection_missing" in out
    assert "collection" in out and "bge" in out
    # Console output is not JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip().splitlines()[-1])


def test_json_renderer_emits_structured_event() -> None:
    buf = _capture(json=True)
    structlog.get_logger("test.json").warning("collection_missing", collection="bge", count=3)
    line = buf.getvalue().strip().splitlines()[-1]
    payload = json.loads(line)
    assert payload["event"] == "collection_missing"
    assert payload["collection"] == "bge"
    assert payload["count"] == 3
    assert payload["level"] == "warning"


def test_level_filters_below_threshold() -> None:
    buf = _capture(json=True, level="WARNING")
    log = structlog.get_logger("test.level")
    log.info("suppressed_event")
    log.warning("kept_event")
    out = buf.getvalue()
    assert "kept_event" in out
    assert "suppressed_event" not in out


def test_configure_is_idempotent() -> None:
    configure_logging(json=False)
    configure_logging(json=True)
    configure_logging(json=False)
    root = logging.getLogger()
    flagged = [h for h in root.handlers if getattr(h, _HANDLER_FLAG, False)]
    assert len(flagged) == 1


def test_exc_info_is_rendered_in_json() -> None:
    buf = _capture(json=True)
    log = structlog.get_logger("test.exc")
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("operation_failed", op="x")
    payload = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert payload["event"] == "operation_failed"
    assert payload["level"] == "error"  # .exception() logs at ERROR
    assert "boom" in payload.get("exception", "")


def test_module_imports_no_app_code() -> None:
    """Decision 1: the seam is pure setup — no business logic, no app imports. Importing
    it must never pull in config or app code (so it stays a trivially-importable
    wiring module)."""
    import doc_assistant.logging_config as mod

    text = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = (
        "from doc_assistant",
        "import doc_assistant",
        "from apps",
        "import apps",
    )
    offending = [line.strip() for line in text.splitlines() if line.strip().startswith(forbidden)]
    assert offending == [], f"logging_config must not import app code, found: {offending}"
