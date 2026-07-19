"""Pure-core guard tests for the deterministic concept skeleton (Node A).

Fixed toy inputs, no DB / LLM / network. The no-edge-creation guard
(citation/similarity annotate-never-create) is the density-control invariant.
"""

from __future__ import annotations

import pytest

from doc_assistant.knowledge.concept_skeleton import (
    PRESENCE_BOUNDARY,
    PRESENCE_SUBSTRING,
    ConceptNode,
    ConceptPresence,
    SkeletonEdge,
    add_citation_provenance,
    add_similarity_provenance,
    analyze_skeleton,
    compile_boundary_pattern,
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


# ---- presence match mode (R2 / RG-009 — word-boundary vs substring) --------


def test_presence_boundary_rejects_substring_hits() -> None:
    # The confound R2 fixes: "bert" must NOT fire inside sbert / colbert / roberta.
    concepts = [("c_bert", "BERT")]
    chunks = [
        ("d1:p0", "d1", "SBERT and ColBERT build on RoBERTa."),  # no standalone token
        ("d2:p0", "d2", "BERT is the base model."),  # standalone → the only real hit
    ]
    presences = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    by_doc = {p.document_id: p for p in presences}
    assert set(by_doc) == {"d2"}  # d1 contributes nothing under boundary
    assert by_doc["d2"].chunk_keys == ("d2:p0",)
    assert by_doc["d2"].n_mentions == 1


def test_presence_boundary_matches_at_punctuation_and_string_edges() -> None:
    concepts = [("c_bert", "BERT")]
    chunks = [
        ("d1:p0", "d1", "We use BERT."),  # trailing period
        ("d2:p0", "d2", "(BERT) is strong"),  # wrapped in parens
        ("d3:p0", "d3", "bert"),  # whole string, no surrounding chars
    ]
    presences = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    assert {p.document_id for p in presences} == {"d1", "d2", "d3"}
    assert all(p.n_mentions == 1 for p in presences)


def test_compile_boundary_pattern_matches_presence_matchers_output() -> None:
    # KI-15: epistemics.concepts_in_text shares this exact pattern builder so chunk-level
    # attribution and Node-A presence matching can never diverge on boundary semantics.
    pattern = compile_boundary_pattern("gpt-4")
    assert pattern.search("we benchmarked gpt-4 on the task.")
    assert not pattern.search("gpt-4o is a different model.")  # non-word edge char, not \b
    # Same behavior as match_presence's own boundary mode on the identical form.
    presences = match_presence(
        [("c", "gpt-4")], {}, [("d1:p0", "d1", "gpt-4 was released."), ("d1:p1", "d1", "gpt-4o.")]
    )
    assert {p.document_id: p.chunk_keys for p in presences} == {"d1": ("d1:p0",)}


def test_presence_boundary_handles_hyphen_and_plus_forms() -> None:
    # \b would mishandle these edge chars; alnum lookarounds get them right.
    concepts = [("c_gpt4", "GPT-4"), ("c_cpp", "C++")]
    chunks = [
        ("d1:p0", "d1", "GPT-4 is used, not GPT-4o."),  # matches gpt-4, not inside gpt-4o
        ("d2:p0", "d2", "Written in C++ here."),
    ]
    presences = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    by_concept = {p.concept_id: p for p in presences}
    assert by_concept["c_gpt4"].n_mentions == 1  # the gpt-4o occurrence is excluded
    assert by_concept["c_gpt4"].document_id == "d1"
    assert by_concept["c_cpp"].document_id == "d2"  # c++ matched despite '+' edges
    # Deterministic, sorted by (concept_id, document_id).
    assert [(p.concept_id, p.document_id) for p in presences] == sorted(
        (p.concept_id, p.document_id) for p in presences
    )


def test_presence_substring_mode_reproduces_raw_count() -> None:
    concepts = [("c_bert", "BERT")]
    chunks = [("d1:p0", "d1", "SBERT and ColBERT and BERT.")]
    substring = match_presence(concepts, {}, chunks, mode=PRESENCE_SUBSTRING)
    assert substring[0].n_mentions == 3  # sbert + colbert + bert (old behaviour)
    boundary = match_presence(concepts, {}, chunks, mode=PRESENCE_BOUNDARY)
    assert boundary[0].n_mentions == 1  # only the standalone token


def test_presence_default_mode_is_boundary() -> None:
    concepts = [("c_bert", "BERT")]
    chunks = [("d1:p0", "d1", "SBERT only, no standalone.")]
    # No mode passed → boundary → sbert does not count → no presence row at all.
    assert match_presence(concepts, {}, chunks) == []


def test_presence_invalid_mode_raises() -> None:
    with pytest.raises(ValueError, match="presence mode"):
        match_presence([("c", "bert")], {}, [("d:p0", "d", "bert")], mode="bogus")


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


# ---- graded provenance strength (R4 — ratio, not boolean) ------------------


def test_provenance_strength_is_ratio_on_a_partial_graph() -> None:
    # a,b co-occur; both present in d1,d2,d3. Only d1<->d2 are similar → a partial graph.
    edge = SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.0, 4)
    doc_index = {"a": {"d1", "d2", "d3"}, "b": {"d1", "d2", "d3"}}
    out = add_similarity_provenance([edge], [("d1", "d2")], doc_index)
    e = out[0]
    assert e.provenance == frozenset({"cooccurrence", "similarity"})  # token kept (strength>0)
    # 6 ordered candidate pairs among {d1,d2,d3}; 2 linked (d1->d2, d2->d1) → 2/6.
    assert dict(e.provenance_strength)["similarity"] == pytest.approx(1 / 3, abs=1e-6)


def test_provenance_strength_saturated_graph_is_one() -> None:
    # Every candidate endpoint-doc pair is linked → strength pins at 1.0 (the honest
    # "no discrimination on a saturated graph" case; R4 payoff is on partial graphs).
    edge = SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.0, 2)
    doc_index = {"a": {"d1", "d2"}, "b": {"d1", "d2"}}
    out = add_citation_provenance([edge], [("d1", "d2")], doc_index)
    assert dict(out[0].provenance_strength)["citation"] == 1.0


def test_provenance_strength_absent_when_token_not_added() -> None:
    # Concepts share only one doc → no da≠db candidate pair → no token, no strength.
    edge = SkeletonEdge("a", "b", frozenset({"cooccurrence"}), 1.0, 1)
    doc_index = {"a": {"d1"}, "b": {"d1"}}
    out = add_citation_provenance([edge], [("d1", "d2")], doc_index)
    assert out[0].provenance == frozenset({"cooccurrence"})  # unchanged
    assert out[0].provenance_strength == ()  # co-occurrence carries no strength


# ---- edge weight (Decision 5 + R4 graded tiebreak) -------------------------


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


def test_edge_weight_strength_refines_tiebreak_within_band() -> None:
    # Same tokens + same co-occurrence count → the graded strength breaks the tie...
    prov = frozenset({"cooccurrence", "citation"})
    strong = edge_weight(prov, 3, (("citation", 1.0),))
    weak = edge_weight(prov, 3, (("citation", 0.1),))
    assert strong > weak
    assert weak >= 2.0 and strong < 3.0  # ...but both stay inside the 2-token band


def test_edge_weight_token_count_dominates_strength() -> None:
    # The locked invariant: a co-occurrence-only edge, even with a huge co-occurrence count,
    # never outranks a 2-token edge with the weakest possible strength.
    single = edge_weight(frozenset({"cooccurrence"}), 10_000)
    multi = edge_weight(frozenset({"cooccurrence", "citation"}), 1, (("citation", 0.0),))
    assert single < 2.0 <= multi


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
            provenance_strength=(("citation", 0.4),),
            stance_by_doc=(("d1", "supports"),),
            relation="uses",
        )
    ]
    sk = analyze_skeleton(nodes, edges, seed=42)
    back = skeleton_from_dict(skeleton_to_dict(sk))

    assert skeleton_to_dict(back) == skeleton_to_dict(sk)
    assert back.edges[0].provenance == frozenset({"cooccurrence", "citation"})
    assert back.edges[0].provenance_strength == (("citation", 0.4),)  # R4 round-trips exactly
    assert back.edges[0].stance_by_doc == (("d1", "supports"),)
    assert back.edges[0].relation == "uses"


# ---- G3 (SPRINT-003) — doc_years threading, round-trip, back-compat --------


def test_doc_years_round_trips_via_skeleton_meta() -> None:
    # Threaded at the skeleton/meta level (not a new ConceptNode field — see the sprint's
    # blast-radius note); skeleton_to_dict/from_dict already carry `meta` verbatim.
    nodes = [ConceptNode("a", "Alpha", ("d1", "d2"), 0, -1)]
    edges: list[SkeletonEdge] = []
    sk = analyze_skeleton(
        nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2020, "d2": 2022}}
    )
    back = skeleton_from_dict(skeleton_to_dict(sk))
    assert back.meta["doc_years"] == {"d1": 2020, "d2": 2022}


def test_year_less_skeleton_json_still_loads_and_has_no_doc_years_key() -> None:
    # A pre-G3 skeleton.json has no "doc_years" key in meta at all — back-compat invariant.
    nodes = [ConceptNode("a", "Alpha", ("d1",), 0, -1)]
    sk = analyze_skeleton(nodes, [], seed=42)  # no meta_extra at all
    back = skeleton_from_dict(skeleton_to_dict(sk))
    assert "doc_years" not in back.meta


def test_graph_version_changes_when_doc_years_change() -> None:
    nodes = [
        ConceptNode("a", "Alpha", ("d1", "d2"), 0, -1),
        ConceptNode("b", "Beta", ("d1", "d2"), 0, -1),
    ]
    edges = [
        SkeletonEdge(
            "a", "b", frozenset({"cooccurrence"}), 1.5, 2, stance_by_doc=(("d1", "supports"),)
        )
    ]
    s1 = analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2020}})
    s2 = analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2021}})
    s3 = analyze_skeleton(nodes, edges, seed=42, meta_extra={"doc_years": {"d1": 2020}})
    assert s1.meta["graph_version"] != s2.meta["graph_version"]  # a year backfill busts the cache
    assert (
        s1.meta["graph_version"] == s3.meta["graph_version"]
    )  # identical inputs -> identical hash


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
