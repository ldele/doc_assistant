"""Guard tests for the taxonomy auto-propose pass (ADR-028 D8, increment 3 —
`docs/specs/feature-taxonomy-auto-propose.md`). Each test fails against the pre-increment code.

No network, no model: the LLM is a scripted fake that also *counts* its calls, so the
"zero items ⇒ zero calls" and "dry run ⇒ zero calls" contracts are asserted, not assumed.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import networkx as nx
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from doc_assistant.db.models import Base, Concept, Document
from doc_assistant.knowledge.taxonomy_propose import (
    ProposalItem,
    build_choice_messages,
    division_candidates,
    group_candidates,
    parse_choice,
    propose_placements,
    run_propose,
)

# ============================================================
# Fakes + fixtures
# ============================================================


class ScriptedClient:
    """An ``LLMClient`` returning canned answers in order, recording every prompt it saw.

    A ``BaseException`` in the script is *raised* instead of returned, so a transport failure is
    scriptable at an exact position in the batch.
    """

    def __init__(self, *answers: str | BaseException) -> None:
        self.answers = list(answers)
        self.calls: list[list[dict[str, str]]] = []

    def complete(self, messages, *, temperature: float, max_tokens: int) -> str:
        self.calls.append(messages)
        if not self.answers:
            raise AssertionError("ScriptedClient ran out of answers — an unexpected extra call")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def _taxonomy_graph() -> nx.DiGraph:
    """Two divisions; one has two groups. Concept/document nodes are irrelevant to the pass."""
    g: nx.DiGraph = nx.DiGraph()
    g.add_node("comp", kind="domain", label="Information and computing sciences")
    g.add_node("bio", kind="domain", label="Biological sciences")
    g.add_node("ml", kind="domain", label="Machine learning")
    g.add_node("ir", kind="domain", label="Information retrieval")
    g.add_edge("ml", "comp", type="in_field", origin="curated")
    g.add_edge("ir", "comp", type="in_field", origin="curated")
    return g


@pytest.fixture
def temp_db(monkeypatch):
    """Isolate the DB: a fresh temp SQLite with the current schema, engine + factory patched."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(engine, "connect")
    def _fk(dbapi_conn, _record):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", engine)
    monkeypatch.setattr(
        session_module,
        "_SessionLocal",
        sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True),
    )
    Base.metadata.create_all(engine)
    yield path
    engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


def _item(label: str = "dense retrieval") -> ProposalItem:
    return ProposalItem(kind="concept", id="c1", label=label, context='appears in: "DPR"')


# ============================================================
# Candidates + prompt + parse (pure)
# ============================================================


def test_divisions_are_domains_without_a_broader_domain():
    graph = _taxonomy_graph()
    assert [c.id for c in division_candidates(graph)] == ["bio", "comp"]  # label-sorted, stable
    assert [c.id for c in group_candidates(graph, "comp")] == ["ir", "ml"]
    assert group_candidates(graph, "bio") == []  # a division with no groups seeded
    assert group_candidates(graph, "nope") == []  # unknown id, not a crash


def test_prompt_numbers_the_candidates_and_carries_the_context():
    graph = _taxonomy_graph()
    messages = build_choice_messages(_item(), division_candidates(graph), level="fields")
    user = messages[1]["content"]
    assert "1. Biological sciences" in user and "2. Information and computing sciences" in user
    assert "dense retrieval" in user and 'appears in: "DPR"' in user
    assert "none" in messages[0]["content"]  # abstention is offered, not implied


@pytest.mark.parametrize(
    ("text", "expected_index", "expected_confidence"),
    [
        ('{"choice": 2, "confidence": 0.7}', 2, 0.7),
        ('```json\n{"choice": 1, "confidence": 0.5}\n```', 1, 0.5),
        ('{"choice": "2"}', 2, None),
        ("2", 2, None),
        ('{"choice": "none", "confidence": 0.0}', None, 0.0),
        ('{"choice": null}', None, None),
        ('{"choice": 1, "confidence": 5}', 1, None),  # out-of-range rating -> no number, kept
    ],
)
def test_parse_choice_tolerates_local_model_drift(text, expected_index, expected_confidence):
    choice = parse_choice(text, n_candidates=3)
    assert choice is not None
    assert choice.index == expected_index
    assert choice.confidence == expected_confidence


@pytest.mark.parametrize("text", ["not json at all", '{"choice": 9}', '{"choice": 0}', "{}"])
def test_parse_choice_rejects_unparseable_and_out_of_range(text):
    """An out-of-range index is a parse failure, never clamped — clamping would manufacture a
    placement the model never chose. `{}` has no choice key at all."""
    assert parse_choice(text, n_candidates=3) is None


# ============================================================
# The two-stage pass
# ============================================================


def test_two_stage_placement_lands_on_the_group():
    """DoD 4: division then group; the group is the placement, the division rides in evidence."""
    client = ScriptedClient('{"choice": 2, "confidence": 0.9}', '{"choice": 2, "confidence": 0.8}')
    result = propose_placements([_item()], _taxonomy_graph(), client)

    assert len(result.proposals) == 1
    proposal = result.proposals[0]
    assert proposal.field_id == "ml"  # group (candidates within "comp" are ir, ml)
    assert proposal.division_id == "comp"
    assert proposal.confidence == 0.8  # the chain's weakest present link
    assert any("division=Information and computing sciences" in e for e in proposal.evidence)
    assert result.n_calls == 2 and result.n_division_only == 0


def test_stage_two_abstention_leaves_the_division_placement():
    """DoD 5: a coarse placement stands rather than being thrown away."""
    client = ScriptedClient('{"choice": 2, "confidence": 0.6}', '{"choice": "none"}')
    result = propose_placements([_item()], _taxonomy_graph(), client)

    proposal = result.proposals[0]
    assert proposal.field_id == "comp" == proposal.division_id  # recognisably division-level
    assert proposal.confidence == 0.6  # stage 1's, since stage 1 produced the placement
    assert result.n_division_only == 1
    assert any("group=abstained" in e for e in proposal.evidence)


def test_division_without_groups_needs_only_one_call():
    client = ScriptedClient('{"choice": 1, "confidence": 0.4}')  # "bio" has no groups
    result = propose_placements([_item()], _taxonomy_graph(), client)

    assert result.proposals[0].field_id == "bio"
    assert result.n_calls == 1  # no pointless second call when there is nothing to narrow to
    assert result.n_division_only == 1


@pytest.mark.parametrize("stage_one", ['{"choice": "none"}', "gibberish"])
def test_stage_one_abstain_or_garbage_yields_no_proposal(stage_one):
    """DoD 6: no forced field — a wrong placement inflates the coverage this layer must be trusted
    for. The miss is reported, not silently dropped."""
    client = ScriptedClient(stage_one)
    result = propose_placements([_item()], _taxonomy_graph(), client)

    assert result.proposals == ()
    assert result.n_abstained == 1
    assert len(result.misses) == 1


def test_one_transport_failure_does_not_sink_the_batch():
    """DoD 8: the failing item is skipped; the next item still produces a proposal."""
    client = ScriptedClient(
        RuntimeError("ollama down"),  # item 1, stage 1
        '{"choice": 1, "confidence": 0.5}',  # item 2, stage 1 -> "bio" (no groups)
    )
    items = [_item("first"), ProposalItem(kind="concept", id="c2", label="second")]
    result = propose_placements(items, _taxonomy_graph(), client)

    assert [p.item_id for p in result.proposals] == ["c2"]
    assert result.n_abstained == 1
    assert result.n_items == 2


def test_zero_items_and_no_fields_make_zero_calls():
    """DoD 7: checked before the loop, not caught after — asserted on a call-counting fake."""
    client = ScriptedClient('{"choice": 1}')

    assert propose_placements([], _taxonomy_graph(), client).proposals == ()
    empty: nx.DiGraph = nx.DiGraph()
    empty.add_node("c1", kind="concept", label="lonely")  # concepts but no field to place into
    assert propose_placements([_item()], empty, client).proposals == ()
    assert client.calls == []


# ============================================================
# The runner side (DB in, proposed rows out)
# ============================================================


def _seed_corpus(session) -> None:
    session.add(Concept(id="comp", label="Computing", kind="domain"))
    session.add(Concept(id="ml", label="Machine learning", kind="domain"))
    session.add(Concept(id="c1", label="dense retrieval", kind="concept", graph_include=True))
    session.add(Concept(id="c2", label="family only", kind="concept", graph_include=False))
    session.add(
        Document(id="d1", filename="dpr.pdf", source_original="dpr", doc_hash="h", format="pdf")
    )
    session.flush()
    from doc_assistant.knowledge.taxonomy import add_hierarchy_edge

    add_hierarchy_edge(session, "ml", "comp", "in_field")


def test_dry_run_reports_scope_and_makes_no_calls(temp_db):
    """The `build_gaps` polarity: no --apply ⇒ no LLM call and no write. Load-bearing, because
    `assert_provider_intent` no-ops on a dry run — a calling dry run would bill unguarded."""
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _seed_corpus(s)

    client = ScriptedClient('{"choice": 1}')
    run = run_propose(apply=False, client=client)

    assert run.applied is False
    assert run.n_unplaced_concepts == 1  # graph vocabulary only
    assert run.n_concepts_out_of_scope == 1  # the keyword-family concept, reported not hidden
    assert run.n_unclassified_documents == 1
    assert run.result.n_items == 2  # 1 concept + 1 document
    assert client.calls == []


def test_apply_writes_proposed_rows_for_both_kinds(temp_db):
    from doc_assistant.db.session import session_scope
    from doc_assistant.knowledge.taxonomy_view import load_field_detail

    with session_scope() as s:
        _seed_corpus(s)

    # item order is concepts (by label) then documents: "dense retrieval", then "dpr.pdf"
    client = ScriptedClient(
        '{"choice": 1, "confidence": 0.9}',  # concept -> division "Computing"
        '{"choice": 1, "confidence": 0.9}',  # concept -> group "Machine learning"
        '{"choice": 1, "confidence": 0.7}',  # document -> division "Computing"
        '{"choice": 1, "confidence": 0.7}',  # document -> group "Machine learning"
    )
    run = run_propose(apply=True, client=client)

    assert run.applied is True
    assert (run.n_hierarchy_written, run.n_document_fields_written) == (1, 1)

    detail = load_field_detail("ml")
    assert detail is not None
    assert [(m.label, m.origin) for m in detail.concepts] == [("dense retrieval", "proposed")]
    assert [(m.label, m.origin) for m in detail.documents] == [("dpr.pdf", "proposed")]


def test_limit_truncates_and_says_so(temp_db):
    from doc_assistant.db.session import session_scope

    with session_scope() as s:
        _seed_corpus(s)

    run = run_propose(apply=False, limit=1)
    assert run.result.n_items == 1
    assert run.n_truncated == 1  # no silent cap


def test_run_propose_on_an_empty_corpus_is_honest(temp_db):
    """Robustness contract: 0 documents / 0 concepts is a legitimate state, not an error."""
    run = run_propose(apply=False)
    assert (run.n_unplaced_concepts, run.n_unclassified_documents, run.result.n_items) == (0, 0, 0)
