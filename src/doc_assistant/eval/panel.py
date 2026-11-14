"""Panels: which models a run answers with (generic).

Spec ``docs/specs/eval-harness-chat.md`` § Columns and panels. A **column** is one way of
producing answers: a provider, a model and an adapter. A **panel** is a YAML file listing
columns, so the models a run uses are a setting of that run, recorded with it, and never fixed
in code or in the spec. Comparisons with an interval are made within one column only.

    columns:
      - name: local-llama
        provider: ollama
        model: llama3.1:8b
        adapter: controller        # pipeline | controller (default: pipeline)
      - name: haiku-replay
        kind: replay               # filled outside the app; a run lists it and skips it
        model: claude-haiku-4-5

``kind: app`` (the default) runs through the app and is the only kind a gate reads. ``kind:
replay`` is a model given the prompts an ``app`` column recorded, outside the app: not a run of
the app, so this harness lists it and runs nothing for it. ``provider`` is any provider id the
app knows (Settings lists them); this module does not check it, the app does at run time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

ADAPTERS = ("pipeline", "controller")
KINDS = ("app", "replay")


@dataclass(frozen=True)
class Column:
    name: str
    provider: str
    model: str
    adapter: str = "pipeline"
    kind: str = "app"

    @property
    def runs_here(self) -> bool:
        """Whether this harness produces the column's answers (``app``) or only lists it."""
        return self.kind == "app"


def _column(raw: Any, index: int) -> Column:
    if not isinstance(raw, dict):
        raise ValueError(f"panel column {index}: must be a mapping, got {type(raw).__name__}")
    name = str(raw.get("name") or "").strip()
    if not name:
        raise ValueError(f"panel column {index}: 'name' is required")
    kind = str(raw.get("kind") or "app")
    if kind not in KINDS:
        raise ValueError(f"panel column {name!r}: kind must be one of {KINDS}, got {kind!r}")
    adapter = str(raw.get("adapter") or "pipeline")
    if adapter not in ADAPTERS:
        raise ValueError(
            f"panel column {name!r}: adapter must be one of {ADAPTERS}, got {adapter!r}"
        )
    model = str(raw.get("model") or "").strip()
    if not model:
        raise ValueError(f"panel column {name!r}: 'model' is required")
    provider = str(raw.get("provider") or "").strip()
    if kind == "app" and not provider:
        raise ValueError(f"panel column {name!r}: 'provider' is required for an app column")
    return Column(name=name, provider=provider, model=model, adapter=adapter, kind=kind)


def load_panel(path: str | Path) -> list[Column]:
    """Load a panel file. Column names must be unique; an empty panel is refused."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict) or not isinstance(raw.get("columns"), list):
        raise ValueError(f"panel {path}: expected a mapping with a 'columns' list")
    columns = [_column(c, i) for i, c in enumerate(raw["columns"])]
    if not columns:
        raise ValueError(f"panel {path}: no columns")
    names = [c.name for c in columns]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"panel {path}: duplicate column names {dupes}")
    return columns


def turns_per_trial(prior_turns: list[int]) -> int:
    """Model turns one trial asks for: each case's earlier turns plus the scored one.

    ``prior_turns`` is each case's count of earlier turns. A turn the app answers without a
    model (a records question, an empty retrieval) still counts: the statement before a run is
    an upper bound, never an estimate that can come in under."""
    return sum(n + 1 for n in prior_turns)
