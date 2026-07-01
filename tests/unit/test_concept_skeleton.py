"""Pure-core guard tests for the deterministic concept skeleton (Node A).

Fixed toy inputs, no DB / LLM / network. The no-edge-creation guard
(citation/similarity annotate-never-create) is the density-control invariant.
"""

from __future__ import annotations

from doc_assistant.concept_skeleton import (
    ConceptNode,
    ConceptPresence,
    SkeletonEdge,
    add_citation_provenance,
    add_similarity_provenance,
    analyze_skeleton,
    cooccurrence_edges,
    edge_weight,
    match_presence,
    skeleton_from_dict,
    skeleton_to_dict,
)

# ---- presence (Decision 2) -------------------------------------------------


def test_presence_matches_label_and_alias_and_skips_absent() -> None:
    concepts = [
        ("c_rag", "RAG"),
        ("c_dpr", "Dense Passage Retrieval"),
        ("c_bm25", "BM25"),  # never appears → no presence
    ]
    aliases = {"c_dpr": ["DPR"]}
    chunks = [
        ("d1:p0", "d1", "We use RAG for grounding."),
        ("d1:p1", "d1", "DPR is a dense retriever."),  # matched via the alias
        ("d1:p2", "d1", "Nothing relevant here."),
    ]
    presences = match_presence(concepts, aliases, chunks)
    by_concept = {p.concept_id: p for p in presences}

    assert set(by_concept) == {"c_rag", "c_dpr"}  # c_bm25 absent → no row
    assert by_concept["c_rag"].chunk_keys == ("d1:p0",)
    assert by_concept["c_dpr"].chunk_keys == ("d1:p1",)
    assert by_concept["c_rag"].document_id == "d1"


def test_presence_counts_mentions_across_chunks() -> None:
    concepts = [("c", "rerank")]
    chunks = [
        ("d1:p0", "d1", "rerank rerank twice"),
        ("d1:p1", "d1", "rerank once"),
        ("d2:p0", "d2", "rerank in another doc"),
    ]
    presences = match_presence(concepts, {}, chunks)
    by_doc = {p.document_id: p for p in presences}
    assert by_doc["d1"].chunk_keys == ("d1:p0", "d1:p1")
    assert by_doc["d1"].n_mentions == 3  # two + one occurrences
    assert by_doc["d2"].chunk_keys == ("d2:p0",)


# ---- co-occurrence (Decision 4) --------------------------------------------


def test_cooccurrence_threshold() -> None:
    presences = [
        ConceptPresence("a", "d1", ("d1:p0", "d1:p1"), 2),
        ConceptPresence("b", "d1", ("d1:p0", "d1:p1"), 2),
        ConceptPresence("c", "d1", ("d1:p0",), 1),
    ]
    edges2 = cooccurrence_edges(presences, min_cooccurrence=2)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges2} == {("a", "b")}
    edge = edges2[0]
    assert edge.provenance == frozenset({"cooccurrence"})
    assert edge.n_cooccurrence_chunks == 2

    edges1 = cooccurrence_edges(presences, min_cooccurrence=1)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges1} == {
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    }


# ---- the no-edge-creation invariant (Decision 5) ---------------------------


def test_citation_similarity_never_create_edges() -> None:
    # a,b co-occur in d1; x only in d2 and never co-occurs with a or b.
    presences = [
        ConceptPresence("a", "d1", ("d1:p0",), 1),
        ConceptPresence("b", "d1", ("d1:p0",), 1),
        ConceptPresence("x", "d2", ("d2:p0",), 1),
    ]
    edges = cooccurrence_edges(presences, min_cooccurrence=1)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges} == {("a", "b")}
    doc_index = {"a": {"d1"}, "b": {"d1"}, "x": {"d2"}}

    # A d1->d2 citation could "link" a/b to x — but (a,x)/(b,x) are not co-occurrence
    # edges, so NOTHING is created (the density-control invariant).
    cited = add_citation_provenance(edges, [("d1", "d2")], doc_index)
    simd = add_similarity_provenance(cited, [("d1", "d2")], doc_index)
    assert len(cited) == 1 and len(simd) == 1
    assert {(e.source_concept_id, e.target_concept_id) for e in simd} == {("a", "b")}


def test_provenance_added_to_an_existing_edge_only() -> None:
    # a,b co-occur in d1 (the edge); a also in d2, b also in d3.
    presences = [
        ConceptPresence("a", "d1", ("d1:p0",), 1),
        ConceptPresence("b", "d1", ("d1:p0",), 1),
        ConceptPresence("a", "d2", ("d2:p0",), 1),
        ConceptPresence("b", "d3", ("d3:p0",), 1),
    ]
    edges = cooccurrence_edges(presences, min_cooccurrence=1)
    assert {(e.source_concept_id, e.target_concept_id) for e in edges} == {("a", "b")}
    doc_index = {"a": {"d1", "d2"}, "b": {"d1", "d3"}}

    cited = add_citation_provenance(edges, [("d2", "d3")], doc_index)
    assert cited[0].provenance == frozenset({"cooccurrence", "citation"})
    simd = add_similarity_provenance(cited, [("d2", "d3")], doc_index)
    assert simd[0].provenance == frozenset({"cooccurrence", "citation", "similarity"})
    # weight grew with provenance.
    assert simd[0].weight > edges[0].weight


# ---- edge weight (Decision 5) ----------------------------------------------


def test_edge_weight_deterministic_and_ranks_multiprovenance_higher() -> None:
    single = edge_weight(frozenset({"cooccurrence"}), 5)
    assert single == edge_weight(frozenset({"cooccurrence"}), 5)  # deterministic
    multi = edge_weight(frozenset({"cooccurrence", "citation"}), 1)
    assert multi > single  # provenance count dominates co-occurrence count
    assert single < 2.0 <= multi
    # equal provenance → more co-occurrence chunks ranks higher
    assert edge_weight(frozenset({"cooccurrence"}), 10) > edge_weight(
        frozenset({"cooccurrence"}), 1
    )


# ---- serialisation round-trip (Decision 7 carry-over) ----------------------


def test_skeleton_dict_roundtrip_is_exact() -> None:
    nodes = [
        ConceptNode("a", "Alpha", ("d1",), 0, -1),
        ConceptNode("b", "Beta", ("d1",), 0, -1),
        ConceptNode("c", "Gamma", (), 0, -1),  # isolated curated concept
    ]
    edges = [
        SkeletonEdge(
            "a",
            "b",
            frozenset({"cooccurrence", "citation"}),
            2.5,
            3,
            stance_by_doc=(("d1", "supports"),),
            relation="uses",
        )
    ]
    sk = analyze_skeleton(nodes, edges, seed=42)
    back = skeleton_from_dict(skeleton_to_dict(sk))

    assert skeleton_to_dict(back) == skeleton_to_dict(sk)
    assert back.edges[0].provenance == frozenset({"cooccurrence", "citation"})
    assert back.edges[0].stance_by_doc == (("d1", "supports"),)
    assert back.edges[0].relation == "uses"


# ---- Louvain determinism (ADR-1) -------------------------------------------


def test_louvain_communities_deterministic_for_seed() -> None:
    nodes = [ConceptNode(c, c.upper(), ("d1",), 0, -1) for c in ("a", "b", "c", "d", "e", "f")]
    tri = [("a", "b"), ("a", "c"), ("b", "c"), ("d", "e"), ("d", "f"), ("e", "f")]
    edges = [SkeletonEdge(s, t, frozenset({"cooccurrence"}), 1.5, 3) for s, t in tri]
    s1 = analyze_skeleton(nodes, edges, seed=42)
    s2 = analyze_skeleton(nodes, edges, seed=42)

    assert skeleton_to_dict(s1) == skeleton_to_dict(s2)
    assert s1.meta["graph_version"] == s2.meta["graph_version"]
    assert len({n.community for n in s1.nodes}) == 2  # two disjoint triangles


def test_isolated_concept_is_a_zero_degree_node() -> None:
    nodes = [
        ConceptNode("a", "A", ("d1",), 0, -1),
        ConceptNode("b", "B", ("d1",), 0, -1),
        ConceptNode("lonely", "Lonely", (), 0, -1),
    ]
    edges = [SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.5, 2)]
    sk = analyze_skeleton(nodes, edges, seed=42)
    by_id = {n.id: n for n in sk.nodes}
    assert by_id["lonely"].degree == 0
    assert by_id["a"].degree == 1
