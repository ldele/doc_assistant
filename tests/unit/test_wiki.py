"""Tests for the pure core of ``doc_assistant.wiki`` (Feature 6).

Clustering, gap signals, link derivation, markdown rendering, manifest diff,
and the (LLM-injected) note assembly — all exercised with plain data, no DB and
no real LLM. The impure build path is covered by
``tests/integration/test_build_wiki.py``.
"""

from __future__ import annotations

from doc_assistant.wiki import (
    DocRef,
    SimEdge,
    TopicNote,
    _assemble_notes,
    build_manifest,
    cluster_documents,
    compute_gap_signals,
    compute_links,
    diff_manifests,
    fallback_title,
    render_note_markdown,
    slugify,
    topic_id_for,
)


def _doc(i: str, *, keywords: list[str] | None = None) -> DocRef:
    return DocRef(
        doc_id=i, doc_hash=f"h{i}", filename=f"{i}.pdf", title=i.upper(), keywords=keywords or []
    )


# ============================================================
# Clustering
# ============================================================


def test_cluster_links_above_threshold_and_keeps_singletons():
    edges = [SimEdge("a", "b", 0.7), SimEdge("d", "e", 0.6)]
    clusters = cluster_documents(["a", "b", "c", "d", "e"], edges, min_similarity=0.55)
    # a-b together, d-e together, c alone; sorted size-desc then by first member
    assert ["a", "b"] in clusters
    assert ["d", "e"] in clusters
    assert ["c"] in clusters
    assert len(clusters) == 3


def test_cluster_ignores_below_threshold_edges():
    edges = [SimEdge("a", "b", 0.40)]
    clusters = cluster_documents(["a", "b"], edges, min_similarity=0.55)
    assert clusters == [["a"], ["b"]]


def test_cluster_transitive_merge():
    edges = [SimEdge("a", "b", 0.8), SimEdge("b", "c", 0.8)]
    clusters = cluster_documents(["a", "b", "c"], edges, min_similarity=0.55)
    assert clusters == [["a", "b", "c"]]


def test_cluster_is_deterministic():
    edges = [SimEdge("x", "y", 0.9)]
    a = cluster_documents(["z", "y", "x"], edges, min_similarity=0.55)
    b = cluster_documents(["x", "y", "z"], edges, min_similarity=0.55)
    assert a == b


# ============================================================
# topic_id stability
# ============================================================


def test_topic_id_is_order_independent_and_stable():
    assert topic_id_for(["h1", "h2"]) == topic_id_for(["h2", "h1"])
    assert topic_id_for(["h1"]) != topic_id_for(["h2"])
    assert topic_id_for(["h1", "h2"]).startswith("topic-")


# ============================================================
# Gap signals (6b)
# ============================================================


def test_gap_single_source_and_isolated():
    g = compute_gap_signals(1, 0, min_citations=3)
    assert g.single_source and g.no_links and g.citation_thin
    assert "single-source" in g.reasons and "isolated (no links)" in g.reasons
    # single-source supersedes the generic citation-thin label
    assert "citation-thin" not in g.reasons


def test_gap_thin_but_not_single():
    g = compute_gap_signals(2, 1, min_citations=3)
    assert g.citation_thin and not g.single_source and not g.no_links
    assert g.reasons == ["citation-thin"]


def test_gap_healthy_topic_has_no_signals():
    g = compute_gap_signals(5, 3, min_citations=3)
    assert not g.any()
    assert g.reasons == []


# ============================================================
# Links (cross-cluster edges)
# ============================================================


def test_links_are_cross_cluster_edges():
    clusters = [["a", "b"], ["c"]]
    edges = [SimEdge("a", "b", 0.7), SimEdge("b", "c", 0.52)]  # a-b internal, b-c crosses
    links = compute_links(clusters, edges)
    assert links[0] == {1}
    assert links[1] == {0}


def test_no_cross_edges_means_no_links():
    clusters = [["a"], ["b"]]
    links = compute_links(clusters, [])
    assert links == {}


# ============================================================
# slug / fallback title
# ============================================================


def test_slugify():
    assert slugify("Dense Retrieval!") == "dense-retrieval"
    assert slugify("  ") == "untitled"


def test_fallback_title_prefers_dominant_keyword():
    docs = [_doc("a", keywords=["retrieval", "rag"]), _doc("b", keywords=["retrieval"])]
    assert fallback_title(docs) == "Retrieval"


def test_fallback_title_uses_doc_label_without_keywords():
    assert fallback_title([_doc("a")]) == "A"


# ============================================================
# Markdown rendering (6d Obsidian)
# ============================================================


def test_render_note_has_frontmatter_links_sources_and_gap():
    note = TopicNote(
        topic_id="topic-abc",
        title="Dense Retrieval",
        docs=[DocRef("a", "h1", "dpr.pdf", "DPR", 2020, ["retrieval"])],
        summary="A topic about dense retrieval.",
        tags=["rag", "retrieval"],
        links=[("topic-xyz", "Reranking")],
        gap=compute_gap_signals(1, 1),
    )
    md = render_note_markdown(note)
    assert md.startswith("---\n")
    assert "topic_id: topic-abc" in md
    assert 'aliases: ["Dense Retrieval"]' in md
    assert 'tags: ["rag", "retrieval"]' in md
    assert "# Dense Retrieval" in md
    assert "[[topic-xyz|Reranking]]" in md  # Obsidian wikilink with alias
    assert "dpr.pdf" in md
    assert "Knowledge-gap signals" in md  # single-source note flagged


def test_render_note_without_links_or_gap_is_clean():
    note = TopicNote(
        topic_id="topic-q",
        title="Big Topic",
        docs=[DocRef("a", "h1", "a.pdf"), DocRef("b", "h2", "b.pdf"), DocRef("c", "h3", "c.pdf")],
        summary="s",
        tags=[],
        links=[],
        gap=compute_gap_signals(3, 2),
    )
    md = render_note_markdown(note)
    assert "## Related" not in md
    assert "Knowledge-gap signals" not in md
    # No extracted title → render the filename once, not "a.pdf — `a.pdf`".
    assert "- `a.pdf`" in md
    assert "a.pdf — `a.pdf`" not in md


# ============================================================
# Manifest + drift (6c)
# ============================================================


def test_build_manifest_maps_topic_to_sorted_hashes():
    note = TopicNote(
        "topic-1",
        "T",
        [DocRef("a", "h2", "a"), DocRef("b", "h1", "b")],
        "",
        [],
        [],
        compute_gap_signals(2, 0),
    )
    assert build_manifest([note]) == {"topic-1": ["h1", "h2"]}


def test_diff_manifests_reports_added_and_removed():
    drift = diff_manifests({"t1": ["h1"], "t2": ["h2"]}, {"t2": ["h2"], "t3": ["h3"]})
    assert drift.added == ["t3"]
    assert drift.removed == ["t1"]
    assert drift.any()


def test_diff_manifests_stable_is_no_drift():
    drift = diff_manifests({"t1": ["h1"]}, {"t1": ["h1"]})
    assert not drift.any()


# ============================================================
# Note assembly with an injected summarizer (no real LLM)
# ============================================================


def test_assemble_notes_uses_summarizer_and_wires_links():
    docs = [_doc("a"), _doc("b"), _doc("c")]
    edges = [SimEdge("a", "b", 0.8), SimEdge("b", "c", 0.52)]  # a-b cluster, c links to it

    def fake_summarize(members, samples):
        return ("Topic " + "+".join(d.doc_id for d in members), "summary text", ["tag1"])

    notes = _assemble_notes(
        docs,
        edges,
        min_similarity=0.55,
        min_citations=3,
        summarize=fake_summarize,
        per_doc_chunks=2,
    )
    by_size = {len(n.docs): n for n in notes}
    ab, c = by_size[2], by_size[1]
    assert ab.summary == "summary text"
    assert ab.tags == ["tag1"]
    # a-b note links to the c note and vice-versa (the cross-cluster 0.52 edge)
    assert any(tid == c.topic_id for tid, _ in ab.links)
    assert c.gap.single_source  # c is a 1-doc topic


def test_assemble_notes_dry_run_has_no_summary():
    docs = [_doc("a", keywords=["retrieval"])]
    notes = _assemble_notes(
        docs, [], min_similarity=0.55, min_citations=3, summarize=None, per_doc_chunks=2
    )
    assert notes[0].summary == ""
    assert notes[0].title == "Retrieval"  # fallback from keyword


def test_assemble_notes_concept_clusters_override_cosine():
    """When concept_clusters is given, it replaces cosine union-find clustering."""
    docs = [_doc("a"), _doc("b"), _doc("c")]
    # A strong a-b cosine edge would group {a,b} via cluster_documents...
    edges = [SimEdge("a", "b", 0.95)]
    # ...but the concept communities say {a,c} and {b} instead. The override wins.
    notes = _assemble_notes(
        docs,
        edges,
        min_similarity=0.55,
        min_citations=3,
        summarize=None,
        per_doc_chunks=2,
        concept_clusters=[["a", "c"], ["b"]],
    )
    members = sorted(sorted(d.doc_id for d in n.docs) for n in notes)
    assert members == [["a", "c"], ["b"]]
