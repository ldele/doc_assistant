"""Guard tests for ``node_weights_for_epistemics`` — the 7d seam.

The unique-source = neutral rule is the 7d regression that matters: a sole-source
concept is ``unique`` / ``stable``, never ``contested``. Stances are injected directly
into the fixture edges (no LLM).
"""

from __future__ import annotations

from doc_assistant.knowledge.concept_skeleton import (
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


def test_two_dated_per_side_newer_opposing_fires_superseded() -> None:
    # G6: >= 2 dated docs per side is the confidence floor for treating median-vs-median as a
    # genuine aggregate, not a coin-flip — this is the smallest fixture that still clears it.
    nodes = [
        ConceptNode("a", "A", ("old1", "old2", "new1", "new2"), 1, 0),
        ConceptNode("b", "B", ("old1", "old2", "new1", "new2"), 1, 0),
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(
                ("old1", "supports"),
                ("old2", "supports"),
                ("new1", "contradicts"),
                ("new2", "contradicts"),
            ),
        )
    ]
    doc_years = {"old1": 2017, "old2": 2018, "new1": 2023, "new2": 2024}
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, doc_years))
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "superseded_trend"


def test_single_disputer_one_supporter_now_stays_contested() -> None:
    # G6 (SPRINT-006): the exact 1-v-1 fixture that fired `superseded_trend` under G3 alone is
    # now demoted — median-of-one is not an aggregate. This is the sprint's headline behavior
    # change; the old assertion (`superseded_trend`) is deliberately gone.
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
    assert w["a"].direction == "contested"


def test_thin_side_two_vs_one_stays_contested() -> None:
    # 2 dated supporters clear the floor, but only 1 dated disputer does not — the thin side
    # still gates the whole comparison to `contested`, even though the median test alone would
    # have fired (opp median 2024 > sup median 2018.5).
    nodes = [
        ConceptNode("a", "A", ("old1", "old2", "new"), 1, 0),
        ConceptNode("b", "B", ("old1", "old2", "new"), 1, 0),
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "llm_relation"}),
            2.0,
            3,
            stance_by_doc=(
                ("old1", "supports"),
                ("old2", "supports"),
                ("new", "contradicts"),
            ),
        )
    ]
    doc_years = {"old1": 2018, "old2": 2019, "new": 2024}
    w = node_weights_for_epistemics(_skeleton_with_years(nodes, edges, doc_years))
    assert w["a"].coverage == "contested"
    assert w["a"].direction == "contested"


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
