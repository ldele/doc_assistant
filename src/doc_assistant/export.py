"""Conversation + debug export (markdown transcripts, dev bundles, per-turn log).

Two audiences over one substrate:

* **User** — a clean markdown transcript of the chat (Q/A turns, optional source
  list). What an end-user downloads.
* **Dev** — a verbose bundle per turn: the answer, retrieved sources *with reranker
  scores*, embedded figures, the reviewer verdict, and telemetry (model, tokens,
  latency). Plus a structured per-turn JSONL event appended to a session log, so a
  dev can grep what the pipeline actually did across runs and iterate quickly.

This module owns the rendering + file I/O so ``apps/`` stays a thin shell: the UI
collects an ``ExportTurn`` per answer from data it already has (retrieved docs +
scores, figure paths, reviewer, token counts) and calls these functions. Pure
renderers (``render_*``, ``log_event``) are unit-testable with plain data; the
``write_*`` / ``append_*`` helpers are the thin impure boundary. Sidecar by the
Enrichment-Layer rule — never touches the chunk store.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from doc_assistant.config import EXPORT_DIR

log = logging.getLogger(__name__)


# ============================================================
# View models (the export's own shape — decoupled from the DB)
# ============================================================


@dataclass
class ExportSource:
    """One retrieved source as the export sees it."""

    n: int  # 1-based source number, matches the in-answer [n] citation
    filename: str | None = None
    page: int | None = None
    section: str | None = None
    reranker_score: float | None = None
    is_figure: bool = False
    image_path: str | None = None  # on-disk PNG for a figure source (dev export embeds it)
    excerpt: str | None = None


@dataclass
class ExportTurn:
    """One Q/A turn, with everything both export flavours need."""

    question: str
    answer: str
    standalone_query: str | None = None  # the rewritten/standalone query, if it differed
    sources: list[ExportSource] = field(default_factory=list)
    reviewer_summary: str | None = None  # e.g. "faithfulness 4/5 · hedging 3/5"
    failure_tag: str | None = None
    verdict: str | None = None  # self-eval roll-up, e.g. "pass — faithfulness 4/5"
    token_input: int | None = None
    token_output: int | None = None
    latency_ms: float | None = None
    model_name: str | None = None
    embedding_model: str | None = None
    record_id: str | None = None


# ============================================================
# Pure renderers
# ============================================================


def _score(s: float | None) -> str:
    return f"{s:.3f}" if s is not None else "—"


def _source_line(src: ExportSource) -> str:
    """A compact one-line source citation (user transcript: no scores)."""
    name = src.filename or "?"
    page = f", p.{src.page}" if src.page else ""
    fig = " 🖼" if src.is_figure else ""
    return f"- [{src.n}] {name}{page}{fig}"


def render_turn_markdown(turn: ExportTurn, *, index: int | None = None, dev: bool = False) -> str:
    """Render one turn. ``dev=False`` → clean Q/A (+ a source list); ``dev=True`` →
    rewritten query, per-source reranker scores, embedded figures, reviewer, telemetry."""
    n = f" {index}" if index is not None else ""
    parts: list[str] = []

    if not dev:
        parts.append(f"## Turn{n}")
        parts.append(f"**You:** {turn.question}")
        parts.append("")
        parts.append(f"**Assistant:** {turn.answer.strip() or '_(no answer)_'}")
        if turn.sources:
            parts.append("")
            parts.append("**Sources:**")
            parts += [_source_line(s) for s in turn.sources]
        return "\n".join(parts) + "\n"

    # --- dev bundle ---
    rid = f" · record `{turn.record_id[:8]}`" if turn.record_id else ""
    parts.append(f"## Turn{n}{rid}")
    if turn.verdict:
        parts.append(f"**Verdict:** {turn.verdict}")
        parts.append("")
    parts.append(f"**You asked:** {turn.question}")
    if turn.standalone_query and turn.standalone_query != turn.question:
        parts.append(f"_rewritten →_ `{turn.standalone_query}`")
    parts.append("")
    parts.append("**Answer:**")
    parts.append("")
    parts.append(turn.answer.strip() or "_(no answer)_")

    if turn.sources:
        parts.append("")
        parts.append("**Retrieved sources** (reranker score):")
        parts.append("")
        parts.append("| # | source | page | score | figure |")
        parts.append("|---|--------|------|-------|--------|")
        for s in turn.sources:
            parts.append(
                f"| {s.n} | {s.filename or '?'} | {s.page or '—'} "
                f"| {_score(s.reranker_score)} | {'yes' if s.is_figure else ''} |"
            )

    figs = [s for s in turn.sources if s.is_figure and s.image_path]
    if figs:
        parts.append("")
        parts.append("**Figures:**")
        for s in figs:
            cap = (s.excerpt or "").splitlines()[0][:120] if s.excerpt else f"figure [{s.n}]"
            parts.append("")
            parts.append(f"![{cap}]({s.image_path})")

    if turn.reviewer_summary or turn.failure_tag:
        tag = f" · failure_tag `{turn.failure_tag}`" if turn.failure_tag else ""
        parts.append("")
        parts.append(f"**Reviewer:** {turn.reviewer_summary or '—'}{tag}")

    parts.append("")
    parts.append(
        "**Telemetry:** "
        f"model `{turn.model_name or '?'}` · embed `{turn.embedding_model or '?'}` · "
        f"{(turn.token_input or 0)} in + {(turn.token_output or 0)} out tok · "
        f"{(turn.latency_ms or 0.0):.0f} ms"
    )
    return "\n".join(parts) + "\n"


def render_conversation_markdown(
    turns: list[ExportTurn], *, title: str = "Conversation export", dev: bool = False
) -> str:
    """Render a whole conversation. Header + each turn, separated by rules."""
    flavour = "developer bundle" if dev else "transcript"
    header = [f"# {title}", "", f"_{len(turns)} turn(s) · {flavour}._", ""]
    if not turns:
        header.append("_No turns to export yet._")
        return "\n".join(header) + "\n"

    # Verdict roll-up (dev only) — the at-a-glance judgement across the run.
    if dev and any(t.verdict for t in turns):
        header += [
            "## Verdict summary",
            "",
            "| # | verdict | question |",
            "|---|---------|----------|",
        ]
        for i, t in enumerate(turns):
            q = t.question if len(t.question) <= 60 else t.question[:57] + "..."
            header.append(f"| {i + 1} | {t.verdict or '—'} | {q} |")
        header.append("")

    body = "\n\n---\n\n".join(
        render_turn_markdown(t, index=i + 1, dev=dev) for i, t in enumerate(turns)
    )
    return "\n".join(header) + "\n" + body


def log_event(turn: ExportTurn) -> dict[str, Any]:
    """A flat, grep-able structured event for one turn (the per-turn JSONL line)."""
    return {
        "record_id": turn.record_id,
        "question": turn.question,
        "standalone_query": turn.standalone_query,
        "n_sources": len(turn.sources),
        "n_figures": sum(1 for s in turn.sources if s.is_figure),
        "top_score": max(
            (s.reranker_score for s in turn.sources if s.reranker_score is not None), default=None
        ),
        "reviewer_summary": turn.reviewer_summary,
        "failure_tag": turn.failure_tag,
        "token_input": turn.token_input,
        "token_output": turn.token_output,
        "latency_ms": turn.latency_ms,
        "model_name": turn.model_name,
        "embedding_model": turn.embedding_model,
    }


# ============================================================
# Impure boundary — file I/O
# ============================================================


def _ensure_dir(export_dir: Path | None = None) -> Path:
    root = export_dir or EXPORT_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root


def write_markdown(filename: str, content: str, *, export_dir: Path | None = None) -> Path:
    """Write a markdown export to ``EXPORT_DIR/filename``; return the path."""
    path = _ensure_dir(export_dir) / filename
    path.write_text(content, encoding="utf-8")
    return path


def append_log_event(
    session_id: str, event: dict[str, Any], *, export_dir: Path | None = None
) -> Path:
    """Append one turn's event as a JSONL line to ``session-{session_id}.jsonl``.

    Stamps an ISO-8601 ``ts`` here (the impure boundary) so ``log_event`` stays pure.
    Append-only — the session log is the chronological debug trail."""
    path = _ensure_dir(export_dir) / f"session-{session_id}.jsonl"
    stamped = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
    return path
