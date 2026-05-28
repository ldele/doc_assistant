"""Tests for the eval case YAML loader (Phase 5 / Feature 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.eval.cases import EvalCase, load_cases_yaml


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "cases.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_load_minimal_case(tmp_path: Path):
    f = _write(tmp_path, "- id: c1\n  query: 'what?'\n")
    cases = load_cases_yaml(f)
    assert len(cases) == 1
    c = cases[0]
    assert isinstance(c, EvalCase)
    assert c.id == "c1"
    assert c.query == "what?"
    assert c.expected_answer is None
    assert c.expected_substrings == []
    assert c.expected_citations == []
    assert c.tags == []
    assert c.metadata == {}


def test_load_full_case(tmp_path: Path):
    f = _write(
        tmp_path,
        """
- id: c1
  query: 'what?'
  expected_answer: 'the thing'
  expected_substrings: [foo, bar]
  expected_citations: [paper.pdf]
  tags: [neuroscience]
  metadata:
    difficulty: easy
""",
    )
    c = load_cases_yaml(f)[0]
    assert c.expected_answer == "the thing"
    assert c.expected_substrings == ["foo", "bar"]
    assert c.expected_citations == ["paper.pdf"]
    assert c.tags == ["neuroscience"]
    assert c.metadata == {"difficulty": "easy"}


def test_load_empty_file_returns_empty_list(tmp_path: Path):
    f = _write(tmp_path, "")
    assert load_cases_yaml(f) == []


def test_missing_id_raises(tmp_path: Path):
    f = _write(tmp_path, "- query: 'what?'\n")
    with pytest.raises(ValueError, match="missing required field 'id'"):
        load_cases_yaml(f)


def test_missing_query_raises(tmp_path: Path):
    f = _write(tmp_path, "- id: c1\n")
    with pytest.raises(ValueError, match="missing required field 'query'"):
        load_cases_yaml(f)


def test_duplicate_id_raises(tmp_path: Path):
    f = _write(
        tmp_path,
        "- id: c1\n  query: 'a'\n- id: c1\n  query: 'b'\n",
    )
    with pytest.raises(ValueError, match="duplicate case id 'c1'"):
        load_cases_yaml(f)


def test_non_list_top_level_raises(tmp_path: Path):
    f = _write(tmp_path, "id: c1\nquery: what?\n")
    with pytest.raises(ValueError, match="top-level must be a list"):
        load_cases_yaml(f)


def test_non_string_substring_raises(tmp_path: Path):
    f = _write(
        tmp_path,
        "- id: c1\n  query: x\n  expected_substrings: [foo, 42]\n",
    )
    with pytest.raises(ValueError, match="must be a string"):
        load_cases_yaml(f)


def test_non_list_substrings_raises(tmp_path: Path):
    f = _write(
        tmp_path,
        "- id: c1\n  query: x\n  expected_substrings: foo\n",
    )
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_cases_yaml(f)
