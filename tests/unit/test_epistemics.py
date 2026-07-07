"""Tests for the pure core of ``doc_assistant.epistemics`` (Feature 7d).

Structural concept→chunk attribution, projection of node weights onto chunks, marker
derivation, and the read-side marker join — all with plain data, no DB / Chroma / LLM.
The impure build path is covered by ``tests/integration/test_compute_epistemics.py``.
"""

from __future__ import annotations

from doc_assistant.concept_skeleton import (
    ConceptNode,
    NodeWeight,
    SkeletonEdge,
    analyze_skeleton,
    edge_weight,
    node_weights_for_epistemics,
)
from doc_assistant.epistemics import (
    MARKER_CONTESTED,
    MARKER_SUPERSEDED,
    MarkedChunk,
    concepts_in_text,
    derive_markers,
    markers_for_chunk_keys,
    markers_for_parent,
    project_chunk,
    project_chunk_weights,
)

# ============================================================
# Structural attribution
# ============================================================


def test_concepts_in_text_word_boundary_match():
    present = concepts_in_text(
        "RAG combines BM25 with dense retrieval.",
        ["rag", "bm25", "dense retrieval", "missing"],
    )
    assert present == ["rag", "bm25", "dense retrieval"]


def test_concepts_in_text_skips_short_and_avoids_substring_false_positive():
    # "ir" is too short to attribute; "rag" must NOT match inside "storage".
    assert concepts_in_text("cloud storage for ir systems", ["rag", "ir"]) == []


def test_concepts_in_text_dedupes_repeats():
    assert concepts_in_text("bm25 bm25 bm25", ["bm25", "bm25"]) == ["bm25"]


# ============================================================
# Marker derivation
# ============================================================


def test_derive_markers():
    assert derive_markers(0, 0) == []
    assert derive_markers(2, 0) == [MARKER_CONTESTED]
    assert derive_markers(0, 1) == [MARKER_SUPERSEDED]
    assert derive_markers(1, 1) == [MARKER_CONTESTED, MARKER_SUPERSEDED]


# ============================================================
# Projection
# ============================================================


def test_project_chunk_aggregates_coverage_and_marks_contested():
    weights = {
        "bm25": NodeWeight("bm25", 1, 1, 0.5, "contested", "contested"),
        "rag": NodeWeight("rag", 3, 0, 1.0, "stable", "corroborated"),
    }
    row = project_chunk("doc1", 2, ["bm25", "rag"], weights)
    assert row.chunk_key == "doc1:2"
    assert row.n_claims == 2
    assert row.n_contested == 1
    assert row.coverage_summary == {"corroborated": 1, "unique": 0, "contested": 1}
    assert row.markers == [MARKER_CONTESTED]


def test_project_chunk_unique_source_stays_quiet():
    weights = {"hyde": NodeWeight("hyde", 1, 0, 1.0, "stable", "unique")}
    row = project_chunk("doc1", 0, ["hyde"], weights)
    assert row.n_claims == 1
    assert row.coverage_summary["unique"] == 1
    assert row.markers == []  # the only source on its topic is never marked


def test_project_chunk_weights_maps_to_right_chunks_and_omits_empty():
    # colbert<->ranking: "old" supports, "new" contradicts → contested (the skeleton
    # carries no publication years, so superseded_trend isn't reachable here — see
    # concept_skeleton.node_weights_for_epistemics).
    # hyde<->prompting: a single supporting source → unique → never marked.
    cooc = frozenset({"cooccurrence"})
    nodes = [
        ConceptNode("colbert", "colbert", ("old", "new"), 0, -1),
        ConceptNode("ranking", "ranking", ("old", "new"), 0, -1),
        ConceptNode("hyde", "hyde", ("z",), 0, -1),
        ConceptNode("prompting", "prompting", ("z",), 0, -1),
    ]
    edges = [
        SkeletonEdge(
            "colbert",
            "ranking",
            cooc,
            edge_weight(cooc, 2),
            2,
            stance_by_doc=(("new", "contradicts"), ("old", "supports")),
        ),
        SkeletonEdge(
            "hyde", "prompting", cooc, edge_weight(cooc, 1), 1, stance_by_doc=(("z", "supports"),)
        ),
    ]
    skeleton = analyze_skeleton(nodes, edges, seed=42)
    weights = node_weights_for_epistemics(skeleton)
    doc_chunks = [
        ("doc-colbert", 0, "this chunk explains colbert and ranking together"),
        ("doc-hyde", 0, "a hyde prompting trick that nobody else covers"),
        ("doc-empty", 0, "totally unrelated prose with none of the concepts"),
    ]
    rows = project_chunk_weights(skeleton, weights, doc_chunks)
    by_key = {r.chunk_key: r for r in rows}

    # colbert/ranking is disputed → contested → its chunk is marked.
    assert MARKER_CONTESTED in by_key["doc-colbert:0"].markers
    # hyde is a sole source → its chunk has a claim but stays quiet.
    assert by_key["doc-hyde:0"].n_claims >= 1
    assert by_key["doc-hyde:0"].markers == []
    # a chunk with no weighted concept gets no row at all.
    assert "doc-empty:0" not in by_key


# ============================================================
# Read-side marker join (the deferred live-surfacing seam)
# ============================================================


def test_markers_for_chunk_keys_returns_only_marked():
    index = {"d:1": [MARKER_CONTESTED], "d:2": []}
    out = markers_for_chunk_keys(["d:1", "d:2", "d:3"], index)
    assert out == {"d:1": [MARKER_CONTESTED]}  # clean + unknown keys stay quiet


# ============================================================
# PR-M1 — PC→baseline containment mapping (markers_for_parent)
# ============================================================


def test_markers_for_parent_contained_chunk_surfaces_markers():
    parent = "Long parent passage. Synapses are discrete junctions. More text."
    marked = [
        MarkedChunk(
            chunk_index=2, text="Synapses are discrete junctions.", markers=[MARKER_CONTESTED]
        )
    ]
    assert markers_for_parent(parent, marked) == [MARKER_CONTESTED]


def test_markers_for_parent_uncontained_chunk_is_quiet():
    parent = "A passage about ion channels and gating."
    marked = [
        MarkedChunk(
            chunk_index=0, text="Reticular continuity of the net.", markers=[MARKER_CONTESTED]
        )
    ]
    assert markers_for_parent(parent, marked) == []


def test_markers_for_parent_unions_multiple_chunks_deduped():
    parent = "alpha beta gamma delta"
    marked = [
        MarkedChunk(chunk_index=0, text="alpha beta", markers=[MARKER_CONTESTED]),
        MarkedChunk(
            chunk_index=1, text="gamma delta", markers=[MARKER_SUPERSEDED, MARKER_CONTESTED]
        ),
    ]
    # First-seen order, de-duplicated across chunks.
    assert markers_for_parent(parent, marked) == [MARKER_CONTESTED, MARKER_SUPERSEDED]


def test_markers_for_parent_empty_inputs():
    assert markers_for_parent("", [MarkedChunk(0, "x", [MARKER_CONTESTED])]) == []
    assert markers_for_parent("some text", []) == []


def test_epistemics_reads_tolerate_missing_table(tmp_path, monkeypatch):
    # An older library.db has no chunk_epistemics table (the 7d engine never ran). The read
    # side must return empty, not raise — else every answer crashes the marker join (PR-M1).
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.epistemics import load_epistemics_index, load_marked_chunks

    engine = create_engine(f"sqlite:///{tmp_path / 'no_tables.db'}", future=True)  # empty schema
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    try:
        assert load_epistemics_index() == {}
        assert load_marked_chunks(["doc1", "doc2"]) == {}
    finally:
        engine.dispose()
