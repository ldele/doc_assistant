"""Tests for ``doc_assistant.export`` — the conversation/dev export + per-turn log.

Pure renderers (user vs dev markdown, the structured log event) on plain data, plus
the thin file-I/O boundary against a tmp dir. No DB / Chroma / LLM / network.
"""

from __future__ import annotations

import json
from pathlib import Path

from doc_assistant.export import (
    ExportSource,
    ExportTurn,
    append_log_event,
    log_event,
    render_conversation_markdown,
    render_turn_markdown,
    write_markdown,
)


def _turn() -> ExportTurn:
    return ExportTurn(
        question="What is BM25?",
        answer="BM25 is a ranking function [1].",
        standalone_query="define BM25 ranking function",
        sources=[
            ExportSource(
                n=1, filename="bm25.pdf", page=3, reranker_score=0.81, excerpt="BM25 ..."
            ),
            ExportSource(
                n=2,
                filename="fig.pdf",
                page=5,
                reranker_score=0.40,
                is_figure=True,
                image_path="/data/figures/x/page5_fig0.png",
                excerpt="Figure 2: the curve.",
            ),
        ],
        reviewer_summary="faithfulness 4/5 · hedging 3/5",
        failure_tag="none",
        token_input=1200,
        token_output=180,
        latency_ms=842.0,
        model_name="claude-haiku-4-5",
        embedding_model="bge-base",
        record_id="abcdef1234567890",
    )


# ============================================================
# User transcript (clean)
# ============================================================


def test_render_turn_user_is_clean_no_scores():
    md = render_turn_markdown(_turn(), index=1, dev=False)
    assert "**You:** What is BM25?" in md
    assert "BM25 is a ranking function" in md
    assert "- [1] bm25.pdf, p.3" in md
    assert "- [2] fig.pdf, p.5 🖼" in md  # figure marker
    # The clean transcript hides dev internals.
    assert "0.81" not in md
    assert "Telemetry" not in md
    assert "rewritten" not in md


# ============================================================
# Dev bundle (verbose)
# ============================================================


def test_render_turn_dev_has_scores_figures_reviewer_telemetry():
    md = render_turn_markdown(_turn(), index=1, dev=True)
    assert "record `abcdef12`" in md  # short record id
    assert "_rewritten →_ `define BM25 ranking function`" in md
    assert "| 1 | bm25.pdf | 3 | 0.810 |" in md  # per-source reranker score
    assert "![Figure 2: the curve.](/data/figures/x/page5_fig0.png)" in md  # embedded figure
    assert "**Reviewer:** faithfulness 4/5 · hedging 3/5 · failure_tag `none`" in md
    assert "model `claude-haiku-4-5`" in md and "1200 in + 180 out tok" in md


def test_render_conversation_header_and_empty():
    md = render_conversation_markdown([_turn(), _turn()], title="Sess", dev=False)
    assert md.startswith("# Sess")
    assert "2 turn(s) · transcript." in md
    empty = render_conversation_markdown([], dev=True)
    assert "No turns to export yet" in empty


def test_verdict_renders_in_dev_and_rolls_up_in_summary():
    t = _turn()
    t.verdict = "pass — faithfulness 4/5"
    # per-turn dev render shows the verdict line
    assert "**Verdict:** pass — faithfulness 4/5" in render_turn_markdown(t, dev=True)
    # the conversation header carries a verdict roll-up table (dev only)
    convo = render_conversation_markdown([t], title="S", dev=True)
    assert "## Verdict summary" in convo and "pass — faithfulness 4/5" in convo
    # ...but the clean user transcript stays verdict-free
    assert "Verdict summary" not in render_conversation_markdown([t], dev=False)


# ============================================================
# Structured log event
# ============================================================


def test_log_event_shape():
    ev = log_event(_turn())
    assert ev["n_sources"] == 2
    assert ev["n_figures"] == 1
    assert ev["top_score"] == 0.81
    assert ev["record_id"] == "abcdef1234567890"
    assert ev["failure_tag"] == "none"


def test_log_event_no_sources_top_score_none():
    ev = log_event(ExportTurn(question="q", answer="a"))
    assert ev["n_sources"] == 0 and ev["top_score"] is None


# ============================================================
# File I/O boundary (tmp dir)
# ============================================================


def test_write_markdown(tmp_path: Path):
    p = write_markdown("out.md", "# hi\n", export_dir=tmp_path)
    assert p == tmp_path / "out.md"
    assert p.read_text(encoding="utf-8") == "# hi\n"


def test_append_log_event_jsonl_appends_and_stamps(tmp_path: Path):
    append_log_event("S1", {"a": 1}, export_dir=tmp_path)
    append_log_event("S1", {"a": 2}, export_dir=tmp_path)
    lines = (tmp_path / "session-S1.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2  # appended, not overwritten
    first = json.loads(lines[0])
    assert first["a"] == 1 and "ts" in first  # event preserved + timestamped
