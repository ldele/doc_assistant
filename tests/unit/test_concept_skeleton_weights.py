"""Guard tests for ``node_weights_for_epistemics`` — the 7d seam.

The unique-source = neutral rule is the 7d regression that matters: a sole-source
concept is ``unique`` / ``stable``, never ``contested``. Stances are injected directly
into the fixture edges (no LLM).
"""

from __future__ import annotations

from doc_assistant.concept_skeleton import (
    ConceptNode,
    ConceptSkeleton,
    SkeletonEdge,
    node_weights_for_epistemics,
)


def _skeleton(nodes: list[ConceptNode], edges: list[SkeletonEdge]) -> ConceptSkeleton:
    return ConceptSkeleton(nodes=tuple(nodes), edges=tuple(edges), communities=(), meta={})


def test_sole_source_is_unique_never_contested() -> None:
    nodes = [ConceptNode("a", "A", ("d1",), 1, 0), ConceptNode("b", "B", ("d1",), 1, 0)]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("d1", "supports"),),
            relation="uses",
        )
    ]
    w = node_weights_for_epistemics(_skeleton(nodes, edges))
    assert w["a"].coverage == "unique"
    assert w["a"].direction == "stable"
    assert w["a"].n_contradicting_sources == 0
    assert w["a"].agreement_ratio == 1.0


def test_two_opposing_docs_make_contested() -> None:
    nodes = [ConceptNode("a", "A", ("d1", "d2"), 1, 0), ConceptNode("b", "B", ("d1", "d2"), 1, 0)]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("d1", "supports"), ("d2", "contradicts")),
        )
    ]
    w = node_weights_for_epistemics(_skeleton(nodes, edges))
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "contested"
    assert w["a"].n_supporting_sources == 1
    assert w["a"].n_contradicting_sources == 1


def test_multiple_supporting_no_opposing_is_corroborated() -> None:
    nodes = [ConceptNode("a", "A", ("d1", "d2"), 1, 0), ConceptNode("b", "B", ("d1", "d2"), 1, 0)]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("d1", "supports"), ("d2", "refines")),
        )
    ]
    w = node_weights_for_epistemics(_skeleton(nodes, edges))
    assert w["a"].coverage == "corroborated"  # 2 distinct supporting docs, 0 opposing
    assert w["a"].direction == "stable"
    assert w["a"].n_supporting_sources == 2


def test_node_a_has_no_stances_every_node_unique_stable() -> None:
    # The deterministic Node-A reality: edges carry no stance → all nodes neutral.
    nodes = [ConceptNode("a", "A", ("d1",), 1, 0), ConceptNode("b", "B", ("d1",), 0, 0)]
    edges = [SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.5, 2)]
    w = node_weights_for_epistemics(_skeleton(nodes, edges))
    assert set(w) == {"a", "b"}  # a weight for EVERY node, including the edge-less one
    for nw in w.values():
        assert nw.coverage == "unique"
        assert nw.direction == "stable"
        assert nw.agreement_ratio == 1.0


# --------------------------------------------------------------------------- #
# G3 (SPRINT-003) — year-aware superseded_trend
# --------------------------------------------------------------------------- #


def _skeleton_with_years(
    nodes: list[ConceptNode], edges: list[SkeletonEdge], doc_years: dict[str, int]
) -> ConceptSkeleton:
    return ConceptSkeleton(
        nodes=tuple(nodes), edges=tuple(edges), communities=(), meta={"doc_years": doc_years}
    )


def test_newer_opposing_makes_superseded() -> None:
    nodes = [
        ConceptNode("a", "A", ("old", "new"), 1, 0),
        ConceptNode("b", "B", ("old", "new"), 1, 0),
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("old", "supports"), ("new", "contradicts")),
        )
    ]
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, {"old": 2018, "new": 2024}))
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "superseded_trend"


def test_older_or_equal_opposing_stays_contested() -> None:
    nodes = [
        ConceptNode("a", "A", ("old", "new"), 1, 0),
        ConceptNode("b", "B", ("old", "new"), 1, 0),
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            # the OPPOSING doc ("old") is not newer than the SUPPORTING doc ("new")
            stance_by_doc=(("new", "supports"), ("old", "contradicts")),
        )
    ]
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, {"old": 2018, "new": 2024}))
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "contested"


def test_equal_year_opposing_stays_contested() -> None:
    nodes = [ConceptNode("a", "A", ("d1", "d2"), 1, 0), ConceptNode("b", "B", ("d1", "d2"), 1, 0)]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("d1", "supports"), ("d2", "contradicts")),
        )
    ]
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, {"d1": 2020, "d2": 2020}))
    assert w["a"].direction == "contested"  # strictly newer required, not >=


def test_missing_year_stays_contested_failsafe() -> None:
    nodes = [
        ConceptNode("a", "A", ("old", "new"), 1, 0),
        ConceptNode("b", "B", ("old", "new"), 1, 0),
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("old", "supports"), ("new", "contradicts")),
        )
    ]
    # "new" (the opposing/contradicting doc) has no year on file — never guess superseded.
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, {"old": 2018}))
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "contested"


def test_sole_disputer_with_years_stays_contested() -> None:
    # nc >= 1 but ns == 0 (no supporting doc at all) — nothing to compare against.
    nodes = [ConceptNode("a", "A", ("new",), 1, 0), ConceptNode("b", "B", ("new",), 1, 0)]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("new", "contradicts"),),
        )
    ]
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, {"new": 2024}))
    assert w["a"].direction == "contested"


def test_pre_g3_skeleton_with_no_doc_years_key_is_byte_identical_behaviour() -> None:
    # A skeleton.json written before this sprint has no "doc_years" key in meta at all.
    nodes = [ConceptNode("a", "A", ("d1", "d2"), 1, 0), ConceptNode("b", "B", ("d1", "d2"), 1, 0)]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(("d1", "supports"), ("d2", "contradicts")),
        )
    ]
    w = node_weights_for_epistemics(_skeleton(nodes, edges))  # meta={} — no "doc_years" key
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "contested"
