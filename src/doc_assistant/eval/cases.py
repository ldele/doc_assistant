"""Eval case definition + YAML loader (generic).

A case is one input the system-under-test should handle, plus enough
expected metadata that the scorers can grade the output. The shape is
deliberately permissive — every expected-* field is optional so the
same case file can drive multiple scorer mixes.

Locked design choices:

* ``id`` is required and must be unique within a file.
* ``query`` is required.
* All ``expected_*`` fields are optional. A scorer that needs one
  raises if it's missing — fail fast at scoring, not at load.
* ``tags`` lets eval runs filter or aggregate. No semantics enforced.

The loader returns plain dataclasses, not Pydantic models, so this
module has zero runtime deps beyond stdlib + PyYAML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


@dataclass
class EvalCase:
    """One row of an eval set."""

    id: str
    query: str
    expected_answer: str | None = None
    expected_substrings: list[str] = field(default_factory=list)
    expected_citations: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _coerce_str_list(value: Any, field_name: str, case_id: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"Case {case_id!r}: field {field_name!r} must be a list of strings, "
            f"got {type(value).__name__}"
        )
    out: list[str] = []
    for i, v in enumerate(value):
        if not isinstance(v, str):
            raise ValueError(
                f"Case {case_id!r}: {field_name}[{i}] must be a string, got {type(v).__name__}"
            )
        out.append(v)
    return out


def _parse_case(raw: dict[str, Any]) -> EvalCase:
    if "id" not in raw:
        raise ValueError(f"Case missing required field 'id': {raw!r}")
    case_id = str(raw["id"])
    if "query" not in raw:
        raise ValueError(f"Case {case_id!r}: missing required field 'query'")
    return EvalCase(
        id=case_id,
        query=str(raw["query"]),
        expected_answer=raw.get("expected_answer"),
        expected_substrings=_coerce_str_list(
            raw.get("expected_substrings"), "expected_substrings", case_id
        ),
        expected_citations=_coerce_str_list(
            raw.get("expected_citations"), "expected_citations", case_id
        ),
        tags=_coerce_str_list(raw.get("tags"), "tags", case_id),
        metadata=dict(raw.get("metadata") or {}),
    )


def load_cases_yaml(path: str | Path) -> list[EvalCase]:
    """Load an eval set from a YAML file.

    The YAML must be a list of case dicts. Duplicate ``id`` values raise
    ``ValueError``.
    """
    text = Path(path).read_text(encoding="utf-8")
    raw = yaml.safe_load(text) or []
    if not isinstance(raw, list):
        raise ValueError(f"Eval file {path}: top-level must be a list, got {type(raw).__name__}")

    cases = [_parse_case(r) for r in raw]
    seen: set[str] = set()
    for c in cases:
        if c.id in seen:
            raise ValueError(f"Eval file {path}: duplicate case id {c.id!r}")
        seen.add(c.id)
    return cases
