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


# ------------------------------------------------------------ chat case shapes (eval-harness-chat)


def _load(tmp_path, lines):
    p = tmp_path / "c.yaml"
    p.write_text(NL.join(lines) + NL, encoding="utf-8")
    return load_cases_yaml(p)


NL = chr(10)


def test_turns_make_the_last_one_the_query(tmp_path):
    (c,) = _load(tmp_path, ["- id: t", "  family: conversation", "  turns: ['a?', 'b?', 'c?']"])
    assert c.query == "c?" and c.prior_turns == ["a?", "b?"] and c.family == "conversation"


def test_family_defaults_to_single_and_expectations_parse(tmp_path):
    (c,) = _load(
        tmp_path,
        [
            "- id: x",
            "  query: q",
            "  expect_refusal: true",
            "  expected_route: library",
            "  must_not_contain: [A]",
            "  must_not_cite: [b.md]",
        ],
    )
    assert c.family == "single" and c.expect_refusal and c.expected_route == "library"
    assert c.must_not_contain == ["A"] and c.must_not_cite == ["b.md"]


def test_query_and_turns_together_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        _load(tmp_path, ["- id: x", "  query: q", "  turns: [a]"])


def test_expect_refusal_must_be_a_bool(tmp_path):
    with pytest.raises(ValueError, match="true or false"):
        _load(tmp_path, ["- id: x", "  query: q", "  expect_refusal: maybe"])


def test_missing_query_and_turns_is_refused(tmp_path):
    with pytest.raises(ValueError, match="query"):
        _load(tmp_path, ["- id: x", "  family: single"])
