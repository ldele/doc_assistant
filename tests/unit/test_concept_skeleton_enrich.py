"""Guard tests for Node B — the confined LLM relation/stance pass.

The LLM is behind the ``LLMClient`` seam, so every test drives ``annotate_relations`` with a
fake client returning canned JSON — no network, no Ollama, no model load. The invariants under
test are the spec's confinement guarantees (Decision 6): annotate existing edges only, never
create a node/edge, degrade gracefully on a bad document, and stay idempotent + deterministic.
"""

from __future__ import annotations

import json

import pytest

from doc_assistant.knowledge.concept_skeleton import (
    Community,
    ConceptNode,
    ConceptPresence,
    ConceptSkeleton,
    SkeletonEdge,
    contested_edges,
)
from doc_assistant.knowledge.concept_skeleton_enrich import (
    annotate_relations,
    build_messages,
    parse_annotations,
    present_by_doc,
)

# ============================================================
# Helpers
# ============================================================


class FakeClient:
    """An ``LLMClient`` that replays canned per-call responses.

    A response that is an ``Exception`` instance is raised (to exercise the transport-failure
    path); anything else is returned as the completion text.
    """

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


def _node(cid: str, label: str) -> ConceptNode:
    return ConceptNode(id=cid, label=label, doc_ids=(), degree=1, community=0)


def _edge(a: str, b: str) -> SkeletonEdge:
    return SkeletonEdge(
        source_concept_id=a,
        target_concept_id=b,
        provenance=frozenset({"cooccurrence"}),
        weight=1.0,
        n_cooccurrence_chunks=2,
    )


def _skeleton(nodes: list[ConceptNode], edges: list[SkeletonEdge]) -> ConceptSkeleton:
    community = Community(id=0, label="c", node_ids=tuple(n.id for n in nodes), size=len(nodes))
    return ConceptSkeleton(
        nodes=tuple(nodes),
        edges=tuple(edges),
        communities=(community,),
        meta={"seed": 42, "resolution": 1.0},
    )


def _ann(pair: int, relation: str, stance: str) -> str:
    return json.dumps({"annotations": [{"pair": pair, "relation": relation, "stance": stance}]})


# ============================================================
# annotate_relations — the happy path
# ============================================================


def test_annotate_sets_relation_stance_and_provenance():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    client = FakeClient([_ann(0, "improves on", "supports")])
    out = annotate_relations(sk, {"doc1": ["a", "b"]}, client)
    e = out.edges[0]
    assert "llm_relation" in e.provenance
    assert e.relation == "improves on"
    assert e.stance_by_doc == (("doc1", "supports"),)
    assert e.weight > 2.0  # llm_relation adds a full provenance point over the cooc-only 1.x
    assert out.meta["graph_version"] != sk.meta.get("graph_version")
    assert out.meta["n_llm_annotated_edges"] == 1
    assert out.meta["node_b_calls"] == 1


def test_prompt_lists_only_existing_edge_pairs():
    # a,b,c all present, but only (a,b) is an edge — c must never reach the model.
    sk = _skeleton([_node("a", "A"), _node("b", "B"), _node("c", "C")], [_edge("a", "b")])
    client = FakeClient([_ann(0, "r", "supports")])
    annotate_relations(sk, {"doc1": ["a", "b", "c"]}, client)
    user_msg = client.calls[0][1]["content"]
    assert "[0] A <-> B" in user_msg
    assert "C" not in user_msg.split("Pairs to annotate:")[1]  # no C in the pair list


def test_never_creates_edge_for_non_edge_pair():
    # Response references a valid pair AND an out-of-range pair; neither may add an edge.
    sk = _skeleton([_node("a", "A"), _node("b", "B"), _node("c", "C")], [_edge("a", "b")])
    resp = (
        '{"annotations":[{"pair":0,"relation":"r","stance":"supports"},'
        '{"pair":9,"relation":"x","stance":"contradicts"}]}'
    )
    client = FakeClient([resp])
    out = annotate_relations(sk, {"doc1": ["a", "b", "c"]}, client)
    assert len(out.edges) == 1
    assert (out.edges[0].source_concept_id, out.edges[0].target_concept_id) == ("a", "b")


def test_require_provenance_skips_uncorroborated_edges():
    # Only the citation+similarity-backed edge (a,b) is eligible; the co-occurrence-only
    # edge (c,d) is never sent to the model and stays unannotated.
    ab = SkeletonEdge("a", "b", frozenset({"cooccurrence", "citation", "similarity"}), 3.0, 2)
    cd = SkeletonEdge("c", "d", frozenset({"cooccurrence"}), 1.0, 2)
    sk = _skeleton([_node("a", "A"), _node("b", "B"), _node("c", "C"), _node("d", "D")], [ab, cd])
    client = FakeClient([_ann(0, "r", "supports")])
    out = annotate_relations(
        sk,
        {"doc1": ["a", "b", "c", "d"]},
        client,
        require_provenance=frozenset({"citation", "similarity"}),
    )
    user_msg = client.calls[0][1]["content"]
    assert "A <-> B" in user_msg and "C <-> D" not in user_msg
    by_pair = {(e.source_concept_id, e.target_concept_id): e for e in out.edges}
    assert "llm_relation" in by_pair[("a", "b")].provenance
    assert "llm_relation" not in by_pair[("c", "d")].provenance


# ============================================================
# Stance aggregation + contested detection
# ============================================================


def test_two_docs_opposing_stances_are_contested():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    client = FakeClient([_ann(0, "r", "supports"), _ann(0, "r", "contradicts")])
    out = annotate_relations(sk, {"doc1": ["a", "b"], "doc2": ["a", "b"]}, client)
    e = out.edges[0]
    assert set(e.stance_by_doc) == {("doc1", "supports"), ("doc2", "contradicts")}
    assert len(contested_edges(list(out.edges))) == 1


# ============================================================
# Graceful degradation
# ============================================================


def test_malformed_json_leaves_edge_unannotated():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    out = annotate_relations(sk, {"doc1": ["a", "b"]}, FakeClient(["not json at all"]))
    e = out.edges[0]
    assert "llm_relation" not in e.provenance
    assert e.relation is None
    assert e.stance_by_doc == ()


def test_invalid_stance_value_is_dropped():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    out = annotate_relations(sk, {"doc1": ["a", "b"]}, FakeClient([_ann(0, "r", "maybe")]))
    assert "llm_relation" not in out.edges[0].provenance


def test_one_failed_doc_does_not_sink_the_run():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    client = FakeClient([RuntimeError("ollama down"), _ann(0, "r", "supports")])
    out = annotate_relations(sk, {"doc1": ["a", "b"], "doc2": ["a", "b"]}, client)
    e = out.edges[0]
    assert e.stance_by_doc == (("doc2", "supports"),)  # only the healthy doc contributed
    assert out.meta["node_b_calls"] == 1


# ============================================================
# Idempotency + determinism
# ============================================================


def test_reannotation_is_idempotent():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    pbd = {"doc1": ["a", "b"]}
    first = annotate_relations(sk, pbd, FakeClient([_ann(0, "r", "supports")]))
    # Re-run Node B on the already-enriched skeleton with the same response.
    second = annotate_relations(first, pbd, FakeClient([_ann(0, "r", "supports")]))
    assert second.edges == first.edges  # no doubled stance, same provenance/weight
    assert second.meta["graph_version"] == first.meta["graph_version"]


def test_a_doc_with_fewer_than_two_present_concepts_makes_no_call():
    sk = _skeleton([_node("a", "A"), _node("b", "B")], [_edge("a", "b")])
    client = FakeClient([])  # no responses available → a call would IndexError
    out = annotate_relations(sk, {"doc1": ["a"]}, client)
    assert client.calls == []
    assert out.meta["node_b_calls"] == 0
    assert "llm_relation" not in out.edges[0].provenance


# ============================================================
# Pure parsers
# ============================================================


def test_parse_accepts_bare_list_and_fenced_object():
    bare = '[{"pair":0,"relation":"r","stance":"refines"}]'
    assert parse_annotations(bare, 1) == [(0, "r", "refines")]
    fenced = '```json\n{"annotations":[{"pair":0,"relation":"r","stance":"supports"}]}\n```'
    assert parse_annotations(fenced, 1) == [(0, "r", "supports")]


def test_parse_drops_out_of_range_and_dedupes():
    resp = (
        '{"annotations":[{"pair":0,"relation":"r","stance":"supports"},'
        '{"pair":0,"relation":"dup","stance":"refines"},'
        '{"pair":7,"relation":"x","stance":"supports"}]}'
    )
    assert parse_annotations(resp, 2) == [(0, "r", "supports")]  # first-wins, 7 out of range


def test_parse_returns_empty_on_garbage():
    assert parse_annotations("the model refused", 3) == []


def test_present_by_doc_inverts_and_dedupes():
    presences = [
        ConceptPresence(concept_id="a", document_id="d1", chunk_keys=("d1:p0",), n_mentions=1),
        ConceptPresence(concept_id="b", document_id="d1", chunk_keys=("d1:p1",), n_mentions=1),
        ConceptPresence(concept_id="a", document_id="d2", chunk_keys=("d2:p0",), n_mentions=1),
    ]
    assert present_by_doc(presences) == {"d1": ["a", "b"], "d2": ["a"]}


def test_build_messages_shape():
    msgs = build_messages(["A", "B"], [("A", "B")])
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "[0] A <-> B" in msgs[1]["content"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
