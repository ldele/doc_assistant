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
    compute_node_weights,
    doc_clusters_from_graph,
    extraction_from_dict,
    extraction_to_dict,
    graph_from_dict,
    graph_to_dict,
    normalize_relation,
    parse_extraction,
    snap_polarity,
    snap_relation,
)


def _exp(
    doc: str,
    concepts: list[str],
    triples: list[tuple[str, str, str, str]],
    *,
    year: int | None = None,
) -> DocExtraction:
    """An extraction with explicit (subject, relation, object, polarity) + year (7d)."""
    return DocExtraction(
        doc_id=doc,
        doc_hash=f"h{doc}",
        filename=f"{doc}.pdf",
        concepts=[canonical_key(c) for c in concepts],
        triples=[
            Triple(canonical_key(s), snap_relation(r), canonical_key(o), polarity=snap_polarity(p))
            for s, r, o, p in triples
        ],
        year=year,
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


def test_graph_dict_round_trip():
    """graph_from_dict is the inverse of graph_to_dict for the structural payload.

    (meta gains a derived integrity_summary on serialize; the property recomputes
    it, so it's dropped on load — everything else round-trips exactly.)"""
    g = assemble_graph(
        [
            _ex("a", ["rag", "dpr"], [("rag", "uses", "dpr")]),
            _ex("b", ["rag", "bm25"], [("rag", "compares to", "bm25")]),
            _ex("c", ["standalone"]),
        ],
        meta={"provider": "ollama", "model": "llama3.1:8b"},
    )
    back = graph_from_dict(graph_to_dict(g))

    assert [(n.id, n.label, n.community, n.degree, n.god_node) for n in back.nodes] == [
        (n.id, n.label, n.community, n.degree, n.god_node) for n in g.nodes
    ]
    assert [(n.doc_ids, n.mentions) for n in back.nodes] == [
        (n.doc_ids, n.mentions) for n in g.nodes
    ]
    assert [(e.source, e.target, e.relations, e.integrity, e.weight) for e in back.edges] == [
        (e.source, e.target, e.relations, e.integrity, e.weight) for e in g.edges
    ]
    assert [(c.id, c.node_ids, c.size) for c in back.communities] == [
        (c.id, c.node_ids, c.size) for c in g.communities
    ]
    assert back.god_nodes == g.god_nodes
    assert back.gaps.isolated_nodes == g.gaps.isolated_nodes
    assert back.gaps.thin_bridges == g.gaps.thin_bridges
    assert back.meta == {"provider": "ollama", "model": "llama3.1:8b"}
    # Re-serialising the reloaded graph reproduces the persisted dict exactly.
    assert graph_to_dict(back) == graph_to_dict(g)


# ============================================================
# Claim-corroboration node weights (Feature 7d)
# ============================================================


def test_snap_polarity_vocab_and_synonyms():
    assert snap_polarity("supports") == "supports"
    assert snap_polarity("CONTRADICTS") == "contradicts"
    assert snap_polarity("disputes") == "contradicts"
    assert snap_polarity("replaces") == "supersedes"
    assert snap_polarity("improves") == "refines"
    assert snap_polarity("") == "supports"  # neutral default — noise never invents a dispute
    assert snap_polarity("frobnicates") == "supports"


def test_edge_support_records_carry_polarity_and_year():
    exs = [
        _exp("a", ["rag", "dpr"], [("rag", "uses", "dpr", "supports")], year=2020),
        _exp("b", ["rag", "dpr"], [("rag", "uses", "dpr", "contradicts")], year=2021),
    ]
    _, edges = build_nodes_edges(exs)
    edge = next(e for e in edges if {e.source, e.target} == {"rag", "dpr"})
    stances = {(s.doc_id, s.polarity, s.year) for s in edge.support}
    assert stances == {("a", "supports", 2020), ("b", "contradicts", 2021)}


def test_node_weights_corroborated_stable():
    exs = [
        _exp("a", ["rag", "dpr"], [("rag", "uses", "dpr", "supports")], year=2020),
        _exp("b", ["rag", "dpr"], [("rag", "uses", "dpr", "supports")], year=2021),
    ]
    w = compute_node_weights(assemble_graph(exs))
    assert w["rag"].coverage == "corroborated"
    assert w["rag"].direction == "stable"
    assert (w["rag"].n_supporting_sources, w["rag"].n_contradicting_sources) == (2, 0)


def test_node_weights_contested_when_disputed_not_newer():
    exs = [
        _exp("a", ["bm25", "dense"], [("bm25", "compares to", "dense", "supports")], year=2019),
        _exp("b", ["bm25", "dense"], [("bm25", "compares to", "dense", "contradicts")], year=2018),
    ]
    w = compute_node_weights(assemble_graph(exs))
    assert w["bm25"].coverage == "contested"
    assert w["bm25"].direction == "contested"  # the disputing source is older, not newer
    assert w["bm25"].n_contradicting_sources == 1


def test_node_weights_superseded_trend_when_dispute_is_newer():
    exs = [
        _exp(
            "old", ["colbert", "ranking"], [("colbert", "uses", "ranking", "supports")], year=2005
        ),
        _exp(
            "new",
            ["colbert", "ranking"],
            [("colbert", "uses", "ranking", "supersedes")],
            year=2022,
        ),
    ]
    w = compute_node_weights(assemble_graph(exs))
    assert w["colbert"].coverage == "contested"
    assert w["colbert"].direction == "superseded_trend"  # newer source disputes the older


def test_node_weights_unique_source_is_neutral():
    """The regression that matters most: a sole-source claim is unique, never penalized."""
    exs = [
        _exp("z", ["hyde", "prompting"], [("hyde", "uses", "prompting", "supports")], year=2020)
    ]
    w = compute_node_weights(assemble_graph(exs))
    assert w["hyde"].coverage == "unique"
    assert w["hyde"].direction == "stable"
    assert w["hyde"].n_contradicting_sources == 0
    assert w["hyde"].agreement_ratio == 1.0


def test_node_weights_isolated_node_is_neutral():
    """A concept with no stated claim (only mentioned) is unique/stable, not down-weighted."""
    exs = [_exp("a", ["loner"], [])]
    w = compute_node_weights(assemble_graph(exs))
    assert w["loner"].coverage == "unique"
    assert w["loner"].direction == "stable"
    assert w["loner"].n_supporting_sources == 0


def test_node_weights_age_alone_does_not_penalize():
    """Decision 1: an old but uncontradicted claim keeps full weight — age is not an input."""
    exs = [
        _exp(
            "ancient", ["tfidf", "ranking"], [("tfidf", "uses", "ranking", "supports")], year=1975
        ),
        _exp(
            "recent", ["tfidf", "ranking"], [("tfidf", "uses", "ranking", "supports")], year=2024
        ),
    ]
    w = compute_node_weights(assemble_graph(exs))
    assert w["tfidf"].coverage == "corroborated"
    assert w["tfidf"].direction == "stable"  # old, but never contradicted → stable
