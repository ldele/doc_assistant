"""Tests for the pure Tier-1 gap detectors (``doc_assistant.knowledge.gaps``) over a fixed
toy concept skeleton — no DB, no LLM, no Chroma.
"""

from __future__ import annotations

from doc_assistant.knowledge.concept_skeleton import (
    ConceptNode,
    SkeletonEdge,
    analyze_skeleton,
    edge_weight,
)
from doc_assistant.knowledge.gaps import (
    detect_isolated,
    detect_single_source,
    detect_thin_bridges,
    detect_under_connected,
)

_COOC = frozenset({"cooccurrence"})


def _edge(a: str, b: str, n: int = 2) -> SkeletonEdge:
    return SkeletonEdge(a, b, _COOC, edge_weight(_COOC, n), n)


def test_degree_zero_concept_is_isolated():
    nodes = [
        ConceptNode("iso", "Iso", ("d1",), 0, -1),
        ConceptNode("a", "A", ("d1", "d2"), 0, -1),
        ConceptNode("b", "B", ("d1", "d2"), 0, -1),
    ]
    skeleton = analyze_skeleton(nodes, [_edge("a", "b")], seed=42)
    gaps = detect_isolated(skeleton)
    assert [g.concept_id for g in gaps] == ["iso"]
    gap = gaps[0]
    assert gap.tier == "t1"
    assert gap.determinism == "deterministic"
    assert gap.kind == "isolated"
    assert gap.rating is None


def test_single_source_flagged_not_penalized():
    nodes = [
        ConceptNode("sole", "Sole", ("d1",), 0, -1),  # one doc: single-source
        ConceptNode("shared", "Shared", ("d1", "d2"), 0, -1),  # two docs: not flagged
    ]
    skeleton = analyze_skeleton(nodes, [_edge("sole", "shared")], seed=42)
    gaps = detect_single_source(skeleton)
    assert [g.concept_id for g in gaps] == ["sole"]
    gap = gaps[0]
    assert gap.tier == "t1"
    assert gap.determinism == "deterministic"
    assert gap.kind == "single_source"
    # Flagged for attention, never penalized (7d Decision 4, carried over) — no rating.
    assert gap.rating is None
    assert gap.evidence.fact_ids == ("d1",)


def _nodes(*ids: str) -> list[ConceptNode]:
    return [ConceptNode(i, i.upper(), ("d1",), 0, -1) for i in ids]


def test_a_dead_end_edge_is_not_a_thin_bridge():
    """KL1: p-q-r is two bridges, but each cuts off one lone concept, not a group. A concept on one
    edge is `under_connected`, not a bridge; before KL1 all three were flagged, and every "thin
    bridge" on the working library was exactly this shape."""
    skeleton = analyze_skeleton(_nodes("p", "q", "r"), [_edge("p", "q"), _edge("q", "r")], seed=42)
    assert detect_thin_bridges(skeleton) == []


def test_a_two_concept_island_is_not_a_thin_bridge():
    skeleton = analyze_skeleton(_nodes("cre", "ntsr1"), [_edge("cre", "ntsr1")], seed=42)
    assert detect_thin_bridges(skeleton) == []


def test_the_smaller_group_is_flagged_never_the_hub_it_hangs_from():
    """RG-014 found the most-connected concept reported as a thin bridge. A triangle around a hub,
    and a pair hanging off the hub by one edge: the pair's end of the bridge is the gap."""
    nodes = _nodes("hub", "a", "b", "s1", "s2")
    edges = [
        _edge("hub", "a"),
        _edge("a", "b"),
        _edge("hub", "b"),
        _edge("hub", "s1"),
        _edge("s1", "s2"),
    ]
    gaps = detect_thin_bridges(analyze_skeleton(nodes, edges, seed=42))
    assert [g.concept_id for g in gaps] == ["s1"]
    (gap,) = gaps
    assert set(gap.evidence.fact_ids) == {"hub", "s1"}
    assert gap.tier == "t1" and gap.determinism == "deterministic" and gap.kind == "thin_bridge"


def test_two_equal_groups_flag_both_ends_of_the_bridge():
    nodes = _nodes("a", "b", "c", "d", "e", "f")
    edges = [
        _edge("a", "b"),
        _edge("b", "c"),
        _edge("a", "c"),
        _edge("d", "e"),
        _edge("e", "f"),
        _edge("d", "f"),
        _edge("c", "d"),
    ]
    gaps = detect_thin_bridges(analyze_skeleton(nodes, edges, seed=42))
    assert {g.concept_id for g in gaps} == {"c", "d"}


def test_a_chain_of_groups_judges_every_bridge_by_its_own_sides():
    """Three triangles in a row, joined by two bridges: each bridge splits 3 | 6, so each flags
    only the end on its three-concept side — the tree pass must size sides per bridge."""
    nodes = _nodes("a1", "a2", "a3", "b1", "b2", "b3", "c1", "c2", "c3")
    edges = []
    for g in ("a", "b", "c"):
        edges += [_edge(f"{g}1", f"{g}2"), _edge(f"{g}2", f"{g}3"), _edge(f"{g}1", f"{g}3")]
    edges += [_edge("a3", "b1"), _edge("b3", "c1")]
    gaps = detect_thin_bridges(analyze_skeleton(nodes, edges, seed=42))
    assert [g.concept_id for g in gaps] == ["a3", "c1"]


def test_thin_bridge_absent_in_a_triangle():
    # A triangle has no bridges (every edge sits on a cycle).
    nodes = [
        ConceptNode("x", "X", ("d1",), 0, -1),
        ConceptNode("y", "Y", ("d1",), 0, -1),
        ConceptNode("z", "Z", ("d1",), 0, -1),
    ]
    edges = [_edge("x", "y"), _edge("y", "z"), _edge("x", "z")]
    skeleton = analyze_skeleton(nodes, edges, seed=42)
    assert detect_thin_bridges(skeleton) == []


def test_below_min_degree_is_under_connected():
    nodes = [
        ConceptNode("hub", "Hub", ("d1", "d2", "d3"), 0, -1),
        ConceptNode("leaf1", "L1", ("d1",), 0, -1),
        ConceptNode("leaf2", "L2", ("d2",), 0, -1),
    ]
    edges = [_edge("hub", "leaf1"), _edge("hub", "leaf2")]
    skeleton = analyze_skeleton(nodes, edges, seed=42)
    gaps = detect_under_connected(skeleton, min_degree=2)
    assert {g.concept_id for g in gaps} == {"leaf1", "leaf2"}
    assert all(g.kind == "under_connected" and g.determinism == "deterministic" for g in gaps)
    assert "hub" not in {g.concept_id for g in gaps}  # degree 2 meets the floor


def test_under_connected_excludes_isolated_degree_zero():
    # A degree-0 concept is `isolated`, not double-reported as `under_connected`
    # (both "hub" and "leaf" sit below the floor here — the point is "lonely" doesn't).
    nodes = [
        ConceptNode("hub", "Hub", ("d1", "d2"), 0, -1),
        ConceptNode("leaf", "Leaf", ("d1",), 0, -1),
        ConceptNode("lonely", "Lonely", (), 0, -1),
    ]
    skeleton = analyze_skeleton(nodes, [_edge("hub", "leaf")], seed=42)
    gaps = detect_under_connected(skeleton, min_degree=2)
    assert "lonely" not in {g.concept_id for g in gaps}
