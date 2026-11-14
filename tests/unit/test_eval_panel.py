"""Panel files (spec ``docs/specs/eval-harness-chat.md`` § Columns and panels)."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.eval.panel import Column, load_panel, turns_per_trial

REPO = Path(__file__).resolve().parents[2]


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "panel.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_app_and_replay_columns(tmp_path: Path) -> None:
    cols = load_panel(
        _write(
            tmp_path,
            """
columns:
  - {name: a, provider: ollama, model: llama3.1:8b, adapter: controller}
  - {name: b, provider: ollama, model: qwen2.5:7b}
  - {name: t, kind: replay, model: some-model}
""",
        )
    )
    assert cols[0] == Column("a", "ollama", "llama3.1:8b", adapter="controller")
    assert cols[1].adapter == "pipeline" and cols[1].runs_here
    assert cols[2].kind == "replay" and not cols[2].runs_here


@pytest.mark.parametrize(
    "body, msg",
    [
        ("columns: []", "no columns"),
        ("other: 1", "'columns' list"),
        ("columns:\n  - {provider: ollama, model: m}", "'name' is required"),
        ("columns:\n  - {name: a, provider: ollama}", "'model' is required"),
        ("columns:\n  - {name: a, model: m}", "'provider' is required"),
        ("columns:\n  - {name: a, provider: p, model: m, adapter: http}", "adapter must be"),
        ("columns:\n  - {name: a, provider: p, model: m, kind: judge}", "kind must be"),
        (
            "columns:\n  - {name: a, provider: p, model: m}\n  - {name: a, provider: p, model: n}",
            "duplicate",
        ),
    ],
)
def test_invalid_panels_are_refused(tmp_path: Path, body: str, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        load_panel(_write(tmp_path, body))


def test_turns_per_trial_counts_replayed_turns() -> None:
    assert turns_per_trial([0, 0, 2]) == 5
    assert turns_per_trial([]) == 0


def test_the_shipped_example_panel_loads() -> None:
    cols = load_panel(REPO / "tests" / "eval" / "panel.example.yaml")
    assert cols and any(c.runs_here for c in cols)
