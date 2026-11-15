"""The public case files keep their shape (spec docs/specs/eval-harness-chat.md, H3).

Every public case counts only because a person checked it against its source, so each one must
say so (`author_verified`) and keep what it rests on. A family's file holds that family only, and
the expectations a family's scorers read are present: a conversation case without turns or a
records case without a route would be scored as something else.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from doc_assistant.eval.cases import load_cases_yaml

EVAL = Path(__file__).resolve().parents[2] / "tests" / "eval"
FILES = {
    "cases.public.yaml": "single",
    "cases.conversation.yaml": "conversation",
    "cases.records.yaml": "records",
    "cases.absent.yaml": "absent",
    "cases.hostile.yaml": "hostile",
}
HOSTILE = EVAL / "hostile"


@pytest.mark.parametrize("name, family", FILES.items())
def test_every_public_case_is_verified_and_in_its_family(name: str, family: str) -> None:
    cases = load_cases_yaml(EVAL / name)
    assert cases
    for c in cases:
        assert c.family == family, c.id
        assert c.metadata.get("author_verified") is True, c.id


def test_conversation_cases_replay_earlier_turns() -> None:
    for c in load_cases_yaml(EVAL / "cases.conversation.yaml"):
        assert c.prior_turns, f"{c.id}: a conversation case needs at least two turns"
        assert c.expected_substrings or c.expected_answer, c.id


def test_records_cases_name_their_route() -> None:
    for c in load_cases_yaml(EVAL / "cases.records.yaml"):
        assert c.expected_route == "library", c.id
        assert c.expected_substrings, c.id


def test_absent_cases_name_the_facts_an_invention_would_carry() -> None:
    for c in load_cases_yaml(EVAL / "cases.absent.yaml"):
        assert c.not_in_library, c.id
        assert c.must_not_contain, c.id


def test_absent_fingerprints_are_not_the_questions_own_words() -> None:
    # An honest answer repeats the question's words; a fingerprint taken from them would fail it.
    for c in load_cases_yaml(EVAL / "cases.absent.yaml"):
        for s in c.must_not_contain:
            assert s.lower() not in c.query.lower(), f"{c.id}: {s!r} is in the question"


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_hostile_cases_quote_an_attack_that_is_in_a_pinned_document() -> None:
    pinned = {p.name for p in HOSTILE.glob("h*.md")}
    for c in load_cases_yaml(EVAL / "cases.hostile.yaml"):
        doc = c.metadata.get("evidence_file")
        assert doc in pinned, c.id
        body = _flat((HOSTILE / doc).read_text(encoding="utf-8"))
        assert _flat(c.metadata["planted"]) in body, f"{c.id}: the quoted attack is not in {doc}"
        assert c.must_not_contain, c.id
        for s in c.must_not_contain:
            assert s.lower() in body.lower(), f"{c.id}: fingerprint {s!r} is not from the attack"


def test_cases_added_in_h3_keep_their_evidence() -> None:
    for name in FILES:
        for c in load_cases_yaml(EVAL / name):
            if c.metadata.get("reviewed"):
                assert c.metadata.get("evidence"), c.id


def test_case_ids_are_unique_across_the_public_files() -> None:
    ids = [c.id for name in FILES for c in load_cases_yaml(EVAL / name)]
    assert len(ids) == len(set(ids))
