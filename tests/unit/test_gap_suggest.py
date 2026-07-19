"""Guard tests for the Tier-2a stochastic ceiling (SPRINT-005 / ADR-004 Decision 4).

The LLM is behind the ``LLMClient`` seam, so every test drives ``suggest_for_thin`` with a
fake client returning canned JSON — no network, no Ollama, no model load. The invariants under
test are the sprint's confinement guarantees: suggestion-only output shape (rating + inputs in
evidence), the skeleton is never mutated (quarantine), one bad concept doesn't sink the batch,
and zero ``under_connected`` gaps means zero LLM calls.
"""

from __future__ import annotations

import copy

from doc_assistant.knowledge.concept_skeleton import (
    Community,
    ConceptNode,
    ConceptSkeleton,
    SkeletonEdge,
)
from doc_assistant.knowledge.gap_suggest import build_messages, parse_suggestion, suggest_for_thin
from doc_assistant.knowledge.gaps import Gap


class FakeClient:
    """An ``LLMClient`` that replays canned per-call responses (the Node-B test pattern)."""

    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    def complete(
        self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int
    ) -> str:
        self.calls.append(messages)
        resp = self.responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return str(resp)


def _node(cid: str, label: str, degree: int = 1) -> ConceptNode:
    return ConceptNode(id=cid, label=label, doc_ids=(), degree=degree, community=0)


def _edge(a: str, b: str) -> SkeletonEdge:
    return SkeletonEdge(
        source_concept_id=a,
        target_concept_id=b,
        provenance=frozenset({"cooccurrence"}),
        weight=1.0,
        n_cooccurrence_chunks=2,
    )


def _skeleton(nodes: list[ConceptNode], edges: list[SkeletonEdge]) -> ConceptSkeleton:
    return ConceptSkeleton(
        nodes=tuple(nodes),
        edges=tuple(edges),
        communities=(
            Community(
                id=0, label=nodes[0].label, node_ids=tuple(n.id for n in nodes), size=len(nodes)
            ),
        ),
        meta={"graph_version": "v1"},
    )


def _under_connected_gap(concept_id: str) -> Gap:
    return Gap(
        concept_id=concept_id, tier="t1", determinism="deterministic", kind="under_connected"
    )


# ============================================================
# parse_suggestion / build_messages (pure)
# ============================================================


def test_build_messages_includes_concept_and_neighbours() -> None:
    messages = build_messages("BERT", ["SBERT", "RoBERTa"])
    user = messages[1]["content"]
    assert "BERT" in user
    assert "SBERT" in user and "RoBERTa" in user


def test_parse_suggestion_valid() -> None:
    parsed = parse_suggestion('{"kind": "suggested_link", "target": "HyDE", "rating": 0.7}')
    assert parsed == ("suggested_link", "HyDE", 0.7)


def test_parse_suggestion_strips_code_fence() -> None:
    text = '```json\n{"kind": "thin_area", "target": "retrieval fusion", "rating": 0.4}\n```'
    assert parse_suggestion(text) == ("thin_area", "retrieval fusion", 0.4)


def test_parse_suggestion_rejects_unknown_kind() -> None:
    assert parse_suggestion('{"kind": "invented", "target": "X", "rating": 0.5}') is None


def test_parse_suggestion_rejects_out_of_range_rating() -> None:
    assert parse_suggestion('{"kind": "thin_area", "target": "X", "rating": 1.5}') is None


def test_parse_suggestion_rejects_malformed_json() -> None:
    assert parse_suggestion("not json at all") is None


def test_parse_suggestion_rejects_empty_target() -> None:
    assert parse_suggestion('{"kind": "thin_area", "target": "", "rating": 0.5}') is None


# ============================================================
# suggest_for_thin — the confinement guarantees
# ============================================================


def test_stochastic_label_rating_and_inputs() -> None:
    skeleton = _skeleton(
        [_node("bert", "BERT"), _node("sbert", "SBERT")],
        [_edge("bert", "sbert")],
    )
    client = FakeClient(['{"kind": "suggested_link", "target": "HyDE", "rating": 0.8}'])

    result = suggest_for_thin([_under_connected_gap("bert")], skeleton, client)

    assert len(result) == 1
    gap = result[0]
    assert gap.tier == "t2a"
    assert gap.determinism == "stochastic"
    assert gap.kind == "suggested_link"
    assert gap.status == "surfaced"
    assert gap.rating == 0.8
    assert any("concept=BERT" in f for f in gap.evidence.fact_ids)
    assert any("neighbours=SBERT" in f for f in gap.evidence.fact_ids)
    assert any("target=HyDE" in f for f in gap.evidence.fact_ids)


def test_quarantine_never_mutates_the_skeleton() -> None:
    skeleton = _skeleton(
        [_node("bert", "BERT"), _node("sbert", "SBERT")],
        [_edge("bert", "sbert")],
    )
    before = copy.deepcopy(skeleton)
    client = FakeClient(['{"kind": "suggested_concept", "target": "ELMo", "rating": 0.3}'])

    suggest_for_thin([_under_connected_gap("bert")], skeleton, client)

    assert skeleton == before  # frozen dataclasses, but assert the value is untouched


def test_one_bad_concept_does_not_sink_the_batch() -> None:
    skeleton = _skeleton(
        [_node("a", "A"), _node("b", "B"), _node("c", "C")],
        [_edge("a", "b"), _edge("b", "c")],
    )
    client = FakeClient(
        [
            RuntimeError("transport down"),  # concept "a" — transport failure
            "not json",  # concept "b" — unparseable
            '{"kind": "thin_area", "target": "fusion methods", "rating": 0.5}',  # concept "c"
        ]
    )

    result = suggest_for_thin(
        [_under_connected_gap("a"), _under_connected_gap("b"), _under_connected_gap("c")],
        skeleton,
        client,
    )

    assert len(result) == 1
    assert result[0].concept_id == "c"
    assert len(client.calls) == 3  # every concept still got its own call


def test_zero_under_connected_makes_zero_calls() -> None:
    skeleton = _skeleton([_node("a", "A")], [])
    client = FakeClient([RuntimeError("must not be called")])

    result = suggest_for_thin(
        [Gap(concept_id="a", tier="t1", determinism="deterministic", kind="isolated")],
        skeleton,
        client,
    )

    assert result == []
    assert client.calls == []


def test_non_under_connected_gaps_are_ignored() -> None:
    skeleton = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    client = FakeClient(['{"kind": "thin_area", "target": "X", "rating": 0.5}'])

    result = suggest_for_thin(
        [
            Gap(concept_id="a", tier="t1", determinism="deterministic", kind="isolated"),
            _under_connected_gap("b"),
        ],
        skeleton,
        client,
    )

    assert len(result) == 1
    assert result[0].concept_id == "b"
