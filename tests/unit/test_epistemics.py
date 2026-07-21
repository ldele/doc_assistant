"""Tests for the pure core of ``doc_assistant.knowledge.epistemics`` (Feature 7d).

Structural concept→chunk attribution, projection of node weights onto chunks, marker
derivation, and the read-side marker join — all with plain data, no DB / Chroma / LLM.
The impure build path is covered by ``tests/integration/test_compute_epistemics.py``.
"""

from __future__ import annotations

from doc_assistant.knowledge.concept_skeleton import (
    ConceptNode,
    NodeWeight,
    SkeletonEdge,
    analyze_skeleton,
    edge_weight,
    node_weights_for_epistemics,
)
from doc_assistant.knowledge.epistemics import (
    MARKER_CONTESTED,
    MARKER_SUPERSEDED,
    concepts_in_text,
    derive_markers,
    markers_for_chunk_keys,
    project_chunk,
    project_chunk_weights,
)

# ============================================================
# Structural attribution
# ============================================================


def test_concepts_in_text_word_boundary_match():
    labels_by_id = {"n1": "rag", "n2": "bm25", "n3": "dense retrieval", "n4": "missing"}
    present = concepts_in_text("RAG combines BM25 with dense retrieval.", labels_by_id)
    assert present == ["n1", "n2", "n3"]


def test_concepts_in_text_skips_short_and_avoids_substring_false_positive():
    # "ir" is too short to attribute; "rag" must NOT match inside "storage".
    labels_by_id = {"n1": "rag", "n2": "ir"}
    assert concepts_in_text("cloud storage for ir systems", labels_by_id) == []


def test_concepts_in_text_dedupes_repeated_ids():
    # Same id appearing twice in the input (shouldn't happen from a real skeleton, but the
    # de-dup guard is cheap and load-bearing for `seen`) is only attributed once.
    assert concepts_in_text("bm25 bm25 bm25", {"n1": "bm25"}) == ["n1"]


def test_concepts_in_text_matches_label_not_uuid_id():
    # KI-15: the curated skeleton's node id is an opaque Concept.id UUID, not a readable
    # string — attribution must match the LABEL, never the id itself. The id below deliberately
    # never appears anywhere in the text.
    uuid_id = "00688507-0351-442b-b156-00521129a344"
    labels_by_id = {uuid_id: "sentence encoder"}
    assert concepts_in_text("A sentence encoder maps text to a dense vector.", labels_by_id) == [
        uuid_id
    ]
    # The id itself is never present in any chunk — searching for it directly must find nothing.
    assert concepts_in_text(uuid_id, labels_by_id) == []


def test_concepts_in_text_boundary_handles_nonword_edge_chars():
    # R2's rationale for alnum lookarounds over `\b`: a trailing `\b` after a non-word char
    # (the "4" in "gpt-4") would wrongly demand a following word character. Shared via
    # concept_skeleton.compile_boundary_pattern — this guards epistemics.py doesn't regress to
    # `\b` (which it did until KI-15's fix).
    labels_by_id = {"n1": "gpt-4"}
    assert concepts_in_text("we benchmarked gpt-4 on the task.", labels_by_id) == ["n1"]
    assert concepts_in_text("gpt-4o is a different model.", labels_by_id) == []


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
    row = project_chunk("doc1:2", "doc1", 2, ["bm25", "rag"], weights)
    assert row.chunk_key == "doc1:2"
    assert row.n_claims == 2
    assert row.n_contested == 1
    assert row.coverage_summary == {"corroborated": 1, "unique": 0, "contested": 1}
    assert row.markers == [MARKER_CONTESTED]


def test_project_chunk_carries_pc_parent_key():
    # E1.1 (KI-8): the projection is segmentation-agnostic — a parent chunk_key is stored verbatim,
    # so build_epistemics can project onto PC parents ({doc}:p{idx}) exactly like baseline chunks.
    weights = {"bm25": NodeWeight("bm25", 1, 1, 0.5, "contested", "contested")}
    row = project_chunk("doc1:p4", "doc1", 4, ["bm25"], weights)
    assert row.chunk_key == "doc1:p4"
    assert row.markers == [MARKER_CONTESTED]


def test_project_chunk_unique_source_stays_quiet():
    weights = {"hyde": NodeWeight("hyde", 1, 0, 1.0, "stable", "unique")}
    row = project_chunk("doc1:0", "doc1", 0, ["hyde"], weights)
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
        ("doc-colbert:0", "doc-colbert", 0, "this chunk explains colbert and ranking together"),
        ("doc-hyde:0", "doc-hyde", 0, "a hyde prompting trick that nobody else covers"),
        ("doc-empty:0", "doc-empty", 0, "totally unrelated prose with none of the concepts"),
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


def test_epistemics_reads_tolerate_missing_table(tmp_path, monkeypatch):
    # An older library.db has no chunk_epistemics table (the 7d engine never ran). The read
    # side must return empty, not raise — else every answer crashes the marker join.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    from doc_assistant.db import session as session_mod
    from doc_assistant.knowledge.epistemics import load_epistemics_index

    engine = create_engine(f"sqlite:///{tmp_path / 'no_tables.db'}", future=True)  # empty schema
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
    monkeypatch.setattr(session_mod, "_engine", engine)
    monkeypatch.setattr(session_mod, "_SessionLocal", factory)
    try:
        assert load_epistemics_index() == {}
    finally:
        engine.dispose()
