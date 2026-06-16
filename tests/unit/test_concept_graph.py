"""Tests for the pure core of ``doc_assistant.concept_graph`` (Feature 7).

Canonicalisation, parsing, the merge into integrity-tagged nodes/edges, Louvain
communities + god-node ranking + graph-gap signals, the Feature-6 bridge, and
JSON (de)serialisation — all with plain data, no DB and no real LLM. The impure
build path is covered by ``tests/integration/test_build_concept_graph.py``.
"""

from __future__ import annotations

from doc_assistant.concept_graph import (
    INTEGRITY_TAGS,
    ConceptEdge,
    ConceptNode,
    DocExtraction,
    Triple,
    analyze_graph,
    assemble_graph,
    build_nodes_edges,
    canonical_key,
    community_id_for,
    doc_clusters_from_graph,
    extraction_from_dict,
    extraction_to_dict,
    graph_to_dict,
    normalize_relation,
    parse_extraction,
    snap_relation,
)


def _ex(
    doc: str, concepts: list[str], triples: list[tuple[str, str, str]] | None = None
) -> DocExtraction:
    # Snap relations like the real extraction path (parse_extraction) does.
    return DocExtraction(
        doc_id=doc,
        doc_hash=f"h{doc}",
        filename=f"{doc}.pdf",
        concepts=[canonical_key(c) for c in concepts],
        triples=[
            Triple(canonical_key(s), snap_relation(r), canonical_key(o))
            for s, r, o in (triples or [])
        ],
    )


# ============================================================
# Canonicalisation
# ============================================================


def test_canonical_key_lowercases_collapses_and_trims_punct():
    assert canonical_key("  RAG! ") == "rag"
    assert canonical_key("Dense   Passage  Retrieval") == "dense passage retrieval"
    assert canonical_key("Retrieval-Augmented Generation") == "retrieval-augmented generation"
    assert canonical_key("(BM25).") == "bm25"


def test_normalize_relation():
    assert normalize_relation("  Is Improved   By ") == "is improved by"


# ============================================================
# Parsing (tolerant)
# ============================================================


def test_parse_extraction_canonicalises_and_adds_triple_endpoints():
    raw = (
        '{"concepts": ["RAG", "Dense Passage Retrieval"], '
        '"relations": [{"subject": "RAG", "relation": "uses", "object": "DPR"}]}'
    )
    ex = parse_extraction(raw, doc_id="d", doc_hash="hd", filename="d.pdf")
    assert "rag" in ex.concepts and "dense passage retrieval" in ex.concepts
    assert "dpr" in ex.concepts  # triple endpoint promoted to a concept
    assert ex.triples == [Triple("rag", "uses", "dpr")]


def test_parse_extraction_bad_json_is_empty():
    ex = parse_extraction("not json at all", doc_id="d", doc_hash="hd", filename="d.pdf")
    assert ex.concepts == [] and ex.triples == []


def test_parse_extraction_drops_self_loops_and_dupes_and_handles_fence():
    raw = (
        "```json\n"
        '{"concepts": ["X", "X"], "relations": ['
        '{"subject": "X", "relation": "r", "object": "X"},'
        '{"subject": "A", "relation": "r", "object": "B"},'
        '{"subject": "A", "relation": "r", "object": "B"}]}'
        "\n```"
    )
    ex = parse_extraction(raw, doc_id="d", doc_hash="hd", filename="d.pdf")
    assert ex.concepts.count("x") == 1  # de-duped
    # self-loop dropped, dupe collapsed; "r" snapped to the closed-vocab fallback
    assert ex.triples == [Triple("a", "related_to", "b")]


# ============================================================
# Salvaged from the parallel PR-16 branch (quality bundle)
# ============================================================


def test_snap_relation_closed_vocab():
    assert snap_relation("uses") == "uses"
    assert snap_relation("Compares To") == "compares_to"  # casefold + space→underscore
    assert snap_relation("improves") == "related_to"  # out-of-vocab → fallback
    assert snap_relation("") == "related_to"


def test_parse_extraction_salvages_truncated_json():
    # A completion cut off mid-array: the first relation is complete, the second isn't.
    raw = (
        '{"concepts": ["RAG", "DPR"], "relations": ['
        '{"subject": "RAG", "relation": "uses", "object": "DPR"},'
        '{"subject": "RAG", "relation": "ext'
    )
    ex = parse_extraction(raw, doc_id="d", doc_hash="hd", filename="d.pdf")
    assert ex.triples == [Triple("rag", "uses", "dpr")]  # complete leading object recovered
    assert "rag" in ex.concepts and "dpr" in ex.concepts


def test_community_id_for_is_stable_and_membership_derived():
    assert community_id_for(["a", "b"]) == community_id_for(["b", "a"])  # order-independent
    assert community_id_for(["a", "b"]) != community_id_for(["a", "c"])  # membership change
    assert community_id_for(["a"]).startswith("community-")


# ============================================================
# Merge → integrity-tagged nodes + edges
# ============================================================


def test_build_nodes_edges_extracted_inferred_ambiguous():
    exs = [
        _ex(
            "a",
            ["rag", "dpr", "reranking"],
            [("rag", "uses", "dpr"), ("rag", "extends", "reranking")],
        ),
        _ex("b", ["rag", "dpr"], [("rag", "uses", "dpr")]),
        _ex("c", ["rag", "dpr", "reranking"], [("rag", "uses", "reranking")]),
    ]
    nodes, edges = build_nodes_edges(exs, min_cooccurrence=2)

    by_id = {n.id: n for n in nodes}
    assert set(by_id) == {"rag", "dpr", "reranking"}
    assert by_id["rag"].doc_ids == ["a", "b", "c"]
    assert by_id["reranking"].doc_ids == ["a", "c"]

    tag = {(e.source, e.target): e for e in edges}
    # rag-dpr stated identically by a + b → EXTRACTED, weight 2
    assert tag[("dpr", "rag")].integrity == "EXTRACTED"
    assert tag[("dpr", "rag")].weight == 2
    assert tag[("dpr", "rag")].relations == ["uses"]
    # rag-reranking stated with two distinct in-vocab verbs (extends vs uses) → AMBIGUOUS
    assert tag[("rag", "reranking")].integrity == "AMBIGUOUS"
    assert sorted(tag[("rag", "reranking")].relations) == ["extends", "uses"]
    # dpr-reranking never stated but co-occur in 2 docs → INFERRED
    assert tag[("dpr", "reranking")].integrity == "INFERRED"
    assert tag[("dpr", "reranking")].relations == []


def test_inferred_requires_min_cooccurrence():
    # dpr-reranking co-occur in only ONE doc → no INFERRED edge at min_cooccurrence=2
    exs = [_ex("a", ["dpr", "reranking"]), _ex("b", ["dpr"])]
    _, edges = build_nodes_edges(exs, min_cooccurrence=2)
    assert edges == []


def test_merge_is_deterministic_regardless_of_doc_order():
    exs = [_ex("a", ["x", "y"], [("x", "r", "y")]), _ex("b", ["y", "z"], [("y", "r", "z")])]
    n1, e1 = build_nodes_edges(list(reversed(exs)))
    n2, e2 = build_nodes_edges(exs)
    assert [n.id for n in n1] == [n.id for n in n2]
    assert [(e.source, e.target, e.integrity) for e in e1] == [
        (e.source, e.target, e.integrity) for e in e2
    ]


# ============================================================
# Graph analysis — communities, god nodes, gaps
# ============================================================


def test_analyze_graph_degree_god_nodes_and_no_bridges_in_triangle():
    nodes = [ConceptNode(id=i, label=i) for i in ("a", "b", "c")]
    edges = [
        ConceptEdge("a", "b", ["r"], ["d1"], 1, "EXTRACTED"),
        ConceptEdge("b", "c", ["r"], ["d1"], 1, "EXTRACTED"),
        ConceptEdge("a", "c", ["r"], ["d1"], 1, "EXTRACTED"),
    ]
    communities, god, gaps = analyze_graph(nodes, edges, god_nodes=2, seed=1)
    assert all(n.degree == 2 for n in nodes)
    assert len(god) == 2  # top-2 hubs
    assert gaps.isolated_nodes == []
    assert gaps.thin_bridges == []  # every edge is in a cycle
    assert len(communities) == 1


def test_analyze_graph_isolated_node_and_thin_bridge():
    # Path x-y-z (both edges are bridges) plus an isolated node q.
    nodes = [ConceptNode(id=i, label=i) for i in ("x", "y", "z", "q")]
    edges = [
        ConceptEdge("x", "y", ["r"], ["d1"], 1, "EXTRACTED"),
        ConceptEdge("y", "z", ["r"], ["d1"], 1, "EXTRACTED"),
    ]
    _, _, gaps = analyze_graph(nodes, edges, seed=1)
    assert gaps.isolated_nodes == ["q"]
    assert gaps.thin_bridges == [("x", "y"), ("y", "z")]
    assert next(n for n in nodes if n.id == "q").degree == 0


# ============================================================
# Full assembly + the Feature-6 bridge
# ============================================================


def test_assemble_graph_and_integrity_summary():
    exs = [
        _ex("a", ["rag", "dpr"], [("rag", "uses", "dpr")]),
        _ex("b", ["rag", "dpr"], [("rag", "uses", "dpr")]),
    ]
    g = assemble_graph(exs, meta={"provider": "ollama"})
    assert g.integrity_summary["EXTRACTED"] == 1
    assert sum(g.integrity_summary.values()) == len(g.edges)
    assert set(g.integrity_summary) == set(INTEGRITY_TAGS)
    assert g.meta["provider"] == "ollama"


def test_doc_clusters_group_by_concept_community():
    exs = [
        _ex("a", ["rag", "dpr"], [("rag", "uses", "dpr")]),
        _ex("b", ["rag", "dpr"], [("rag", "uses", "dpr")]),
        _ex("c", ["rag", "dpr"], [("rag", "uses", "dpr")]),
        _ex("d", ["standalone"]),  # isolated concept → its own cluster
    ]
    g = assemble_graph(exs)
    clusters = doc_clusters_from_graph(g, exs)
    biggest = clusters[0]
    assert set(biggest) == {"a", "b", "c"}
    assert ["d"] in clusters


# ============================================================
# JSON (de)serialisation
# ============================================================


def test_graph_to_dict_shape():
    g = assemble_graph([_ex("a", ["rag", "dpr"], [("rag", "uses", "dpr")])])
    d = graph_to_dict(g)
    assert set(d) == {"meta", "nodes", "edges", "communities", "god_nodes", "gaps"}
    assert d["meta"]["integrity_summary"]["EXTRACTED"] == 1
    assert {"isolated_nodes", "thin_bridges"} == set(d["gaps"])
    assert all({"source", "target", "integrity", "weight"} <= set(e) for e in d["edges"])


def test_extraction_dict_round_trip():
    ex = _ex("a", ["rag", "dpr"], [("rag", "uses", "dpr")])
    back = extraction_from_dict(extraction_to_dict(ex))
    assert back.doc_id == ex.doc_id and back.doc_hash == ex.doc_hash
    assert back.concepts == ex.concepts
    assert back.triples == ex.triples
