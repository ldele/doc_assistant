"""The slash-command formatters, which were 16% covered (2026-08-20).

`commands.py` had **215 uncovered lines** — the largest single block in `src/` — and almost all of
it is pure: dataclasses in, markdown out, no I/O. `test_commands.py` covers `parse_command` and a
few `execute_command` branches; every formatter below had nothing.

These are the CLI's entire user-facing surface, and the interesting behaviour in them is
**truncation**. Six independent caps (authors at 50, titles at 80, keywords at 10, citation
snippets at 120, external references at 30, graph nodes at 25) each decide how much of the truth
the user is shown. A cap that silently drops the remainder reads as *this is all of it*, which is
the failure the project keeps writing down — so where the code reports what it hid, that report is
asserted, and where a cap changes the output shape entirely (the graph) both sides are asserted.

Nothing here asserts on the badge emoji: `health_badge` owns those and `test_query_router.py`
pins them. These tests assert the structure each formatter is itself responsible for.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from doc_assistant.commands import (
    format_cited_by,
    format_cites_out,
    format_document_details,
    format_graph,
    format_similar,
    format_summary_message,
    help_message,
)
from doc_assistant.library import (
    CitationEdge,
    CitationGraph,
    DocumentDetails,
    DocumentSummary,
    LibrarySummary,
)
from doc_assistant.library.citations import GraphEdge, GraphNode
from doc_assistant.library.similarity import SimilarDoc

# ============================================================
# Builders
# ============================================================


def _summary(
    *, documents: int = 3, chunks: int = 1234, health: dict[str, int] | None = None
) -> LibrarySummary:
    return LibrarySummary(
        total_documents=documents,
        total_chunks=chunks,
        by_health=health if health is not None else {"healthy": documents},
        by_format={"pdf": documents},
    )


def _doc(
    name: str,
    *,
    health: str | None = "healthy",
    chunks: int = 10,
    tags: list[str] | None = None,
    folders: list[str] | None = None,
) -> DocumentSummary:
    return DocumentSummary(
        id=f"{name}aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        filename=f"{name}.pdf",
        title=name,
        format="pdf",
        health=health,
        chunk_count=chunks,
        page_count=3,
        tags=tags or [],
        folders=folders or [],
    )


def _details(**over: object) -> DocumentDetails:
    base: dict[str, object] = {
        # Non-hex on purpose: a 32-char hex id here reads as a "Hex High Entropy String" to
        # detect-secrets and lands a false positive in `.secrets.baseline` — which then carries a
        # line number that goes stale the moment anything above it moves.
        "id": "document-under-test-000000000000",
        "filename": "paper.pdf",
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
        "notes": None,
        "format": "pdf",
        "doc_hash": "hash-abc",
        "source_original": "/data/sources/paper.pdf",
        "source_cache": None,
        "extractor_used": None,
        "extraction_health": "healthy",
        "chunk_count": 12,
        "page_count": None,
        "extracted_at": None,
        "added_at": None,
        "updated_at": None,
    }
    base.update(over)
    return DocumentDetails(**base)  # type: ignore[arg-type]


def _edge(
    *,
    doc_id: str | None = None,
    title: str | None = "A Paper",
    authors: str | None = "Smith, J.",
    year: int | None = 2020,
    doi: str | None = None,
    raw: str | None = None,
) -> CitationEdge:
    return CitationEdge(
        raw_text=raw,
        target_title=title,
        target_authors=authors,
        target_year=year,
        target_doi=doi,
        target_document_id=doc_id,
        target_filename=None,
        extraction_method="regex",
        confidence=0.9,
    )


# ============================================================
# format_summary_message
# ============================================================


def test_summary_reports_totals_with_a_thousands_separator() -> None:
    out = format_summary_message(_summary(documents=97, chunks=36717), [], None)
    assert "**97 documents**" in out
    assert "36,717 chunks" in out


def test_summary_groups_documents_worst_health_first() -> None:
    """Ordering is the whole point of the grouping: a broken document must not sort below a
    healthy one just because of insertion order."""
    docs = [_doc("good"), _doc("bad", health="broken"), _doc("meh", health="marginal")]
    out = format_summary_message(_summary(), docs, None)
    assert out.index("bad.pdf") < out.index("meh.pdf") < out.index("good.pdf")


def test_summary_puts_a_document_with_no_health_in_the_unknown_group() -> None:
    """`groups[d.health or "unknown"]` — a null health must not KeyError the whole overview."""
    out = format_summary_message(_summary(), [_doc("mystery", health=None)], None)
    assert "mystery.pdf" in out
    assert "unknown" in out.lower()


def test_summary_omits_health_groups_that_have_no_documents() -> None:
    out = format_summary_message(_summary(), [_doc("good")], None)
    assert "broken" not in out.lower()


def test_summary_states_the_filter_and_how_much_it_hid() -> None:
    """A filtered view that showed only its own count would read as the whole library."""
    out = format_summary_message(_summary(documents=97), [_doc("a")], "broken")
    assert "Filtered: broken" in out
    assert "1 of 97" in out


def test_summary_shows_tags_and_folders_only_when_present() -> None:
    with_meta = format_summary_message(
        _summary(), [_doc("a", tags=["seminal"], folders=["Reading"])], None
    )
    assert "tags: seminal" in with_meta
    assert "folders: Reading" in with_meta

    without = format_summary_message(_summary(), [_doc("a")], None)
    assert "tags:" not in without
    assert "folders:" not in without


def test_summary_addresses_documents_by_short_id() -> None:
    out = format_summary_message(_summary(), [_doc("a")], None)
    assert "`aaaaaaaa`" in out, "the CLI's own /document argument is the 8-char prefix"


def test_summary_of_an_empty_library_is_still_a_valid_overview() -> None:
    out = format_summary_message(_summary(documents=0, chunks=0, health={}), [], None)
    assert "Library overview" in out
    assert "**0 documents**" in out


# ============================================================
# format_document_details
# ============================================================


def test_details_of_a_missing_document_says_so_rather_than_rendering_a_blank_card() -> None:
    assert format_document_details(None) == "Document not found."


def test_details_omits_every_optional_field_that_is_absent() -> None:
    """A bare document must not render empty labels — `**Title:**` with nothing after it reads
    as data loss rather than as data never extracted."""
    out = format_document_details(_details())
    for label in (
        "**Title:**",
        "**Authors:**",
        "**Year:**",
        "**DOI:**",
        "**Pages:**",
        "### Notes",
    ):
        assert label not in out


def test_details_shows_optional_fields_that_are_present() -> None:
    out = format_document_details(
        _details(
            title="On Circuits", authors="Middleton, F.", year=2001, doi="10.1/x", page_count=15
        )
    )
    assert "**Title:** On Circuits" in out
    assert "**Authors:** Middleton, F." in out
    assert "**Year:** 2001" in out
    assert "**DOI:** 10.1/x" in out
    assert "**Pages:** 15" in out


def test_details_always_carries_the_identity_fields() -> None:
    """id / source / hash are what make a report reproducible — they are never optional."""
    out = format_document_details(_details())
    assert "document-under-test-000000000000" in out
    assert "/data/sources/paper.pdf" in out
    assert "hash-abc" in out


def test_details_caps_keywords_at_ten_and_says_how_many_it_hid() -> None:
    out = format_document_details(_details(keywords=[f"kw{i}" for i in range(14)]))
    assert "kw9" in out
    assert "kw10" not in out
    assert "and 4 more" in out


def test_details_does_not_announce_a_remainder_when_there_is_none() -> None:
    out = format_document_details(_details(keywords=["kw0", "kw1"]))
    assert "more" not in out.split("**Keywords:**")[1].splitlines()[0]


def test_details_renders_ingestion_history_with_a_readable_timestamp() -> None:
    out = format_document_details(
        _details(
            ingestion_history=[
                {
                    "timestamp": datetime(2026, 8, 19, 14, 5),
                    "event_type": "reextracted",
                    "extractor": "pymupdf",
                    "chunks_produced": 52,
                    "health_status": "healthy",
                    "notes": "cache cleared by hand",
                }
            ]
        )
    )
    assert "2026-08-19 14:05" in out
    assert "**reextracted**" in out
    assert "notes: cache cleared by hand" in out


def test_details_survives_an_ingestion_event_with_no_timestamp() -> None:
    """History rows predate the column; a null must render as `?`, not crash the whole card."""
    out = format_document_details(
        _details(ingestion_history=[{"timestamp": None, "event_type": "ingested"}])
    )
    assert "?: **ingested**" in out


def test_details_caps_ingestion_history_at_ten_events() -> None:
    events = [{"timestamp": None, "event_type": f"event{i}", "extractor": None} for i in range(13)]
    out = format_document_details(_details(ingestion_history=events))
    assert "event9" in out
    assert "event10" not in out


# ============================================================
# format_cites_out
# ============================================================


def test_cites_out_with_nothing_extracted_names_the_command_that_would_fix_it() -> None:
    out = format_cites_out("paper.pdf", [])
    assert "no citations extracted" in out
    assert "extract_citations" in out


def test_cites_out_splits_resolved_from_external_and_counts_both() -> None:
    """The split is the honest part: 'cites 3 works' alone would imply the library holds them."""
    edges = [_edge(doc_id="1" * 32), _edge(), _edge()]
    out = format_cites_out("paper.pdf", edges)
    assert "cites 3 works" in out
    assert "1 resolved to library docs" in out
    assert "2 external" in out


def test_cites_out_caps_the_external_list_and_says_how_many_it_hid() -> None:
    out = format_cites_out("paper.pdf", [_edge(title=f"Paper {i}") for i in range(35)])
    assert "Paper 29" in out
    assert "Paper 30" not in out
    assert "and 5 more" in out


def test_cites_out_never_truncates_the_resolved_list() -> None:
    """The in-library references are the actionable ones — the cap is for the external tail."""
    edges = [_edge(doc_id=f"{i:032d}", title=f"Owned {i}") for i in range(35)]
    out = format_cites_out("paper.pdf", edges)
    assert "Owned 34" in out


# ============================================================
# _ref_one_line, through format_cites_out
# ============================================================


def test_a_resolved_reference_is_addressable_by_short_id() -> None:
    out = format_cites_out("p.pdf", [_edge(doc_id="deadbeef" + "0" * 24)])
    assert "`deadbeef`" in out


def test_a_reference_with_no_parsed_fields_falls_back_to_its_raw_text() -> None:
    out = format_cites_out("p.pdf", [_edge(title=None, authors=None, year=None, raw="[7] Ibid.")])
    assert "[7] Ibid." in out


def test_a_reference_with_nothing_at_all_is_labelled_unparsed() -> None:
    """Silently emitting an empty bullet would look like a rendering bug to the user."""
    out = format_cites_out("p.pdf", [_edge(title=None, authors=None, year=None, raw=None)])
    assert "(unparsed)" in out


def test_long_author_and_title_strings_are_truncated() -> None:
    out = format_cites_out("p.pdf", [_edge(authors="A" * 80, title="T" * 120)])
    assert "A" * 50 in out
    assert "A" * 51 not in out
    assert "T" * 80 in out
    assert "T" * 81 not in out


def test_a_doi_is_shown_when_the_reference_carries_one() -> None:
    out = format_cites_out("p.pdf", [_edge(doi="10.1234/abc")])
    assert "doi:10.1234/abc" in out


# ============================================================
# format_cited_by
# ============================================================


def test_cited_by_with_no_rows_explains_both_reasons_it_could_be_empty() -> None:
    """Nothing cites it, or extraction never ran — the user cannot tell those apart alone."""
    out = format_cited_by("paper.pdf", [])
    assert "no library documents cite this one" in out
    assert "extract_citations" in out


def test_cited_by_lists_each_citing_document_by_short_id() -> None:
    rows = [("f" * 32, "citing.pdf", "See Smith 2020 for details")]
    out = format_cited_by("paper.pdf", rows)
    assert "1 library document(s) cite paper.pdf" in out
    assert "`ffffffff`" in out
    assert "citing.pdf" in out


def test_cited_by_truncates_a_long_citation_snippet() -> None:
    out = format_cited_by("p.pdf", [("a" * 32, "c.pdf", "x" * 200)])
    assert "x" * 120 in out
    assert "x" * 121 not in out


def test_cited_by_handles_a_row_with_no_raw_text() -> None:
    out = format_cited_by("p.pdf", [("a" * 32, "c.pdf", None)])
    assert "c.pdf" in out


# ============================================================
# format_similar
# ============================================================


def test_similar_with_no_edges_names_the_command_that_would_build_them() -> None:
    out = format_similar("paper.pdf", [])
    assert "no similarity edges" in out
    assert "compute_doc_vectors" in out


def test_similar_shows_the_cosine_score_to_three_decimals() -> None:
    """The score is the only thing distinguishing a strong neighbour from a weak one; rounding
    it to a whole number would flatten the ranking it exists to convey."""
    out = format_similar(
        "p.pdf",
        [
            SimilarDoc(
                target_document_id="b" * 32,
                target_filename="n.pdf",
                target_title=None,
                score=0.91234,
            )
        ],
    )
    assert "cosine 0.912" in out


def test_similar_includes_a_neighbour_title_when_there_is_one() -> None:
    with_title = format_similar(
        "p.pdf",
        [
            SimilarDoc(
                target_document_id="b" * 32,
                target_filename="n.pdf",
                target_title="On Cortex",
                score=0.9,
            )
        ],
    )
    assert "On Cortex" in with_title


# ============================================================
# format_graph
# ============================================================


def _graph(n: int, *, edges: int = 0) -> CitationGraph:
    nodes = [
        GraphNode(id=f"{i:032d}", filename=f"doc{i}.pdf", title=None, is_center=(i == 0))
        for i in range(n)
    ]
    return CitationGraph(
        nodes=nodes,
        edges=[GraphEdge(source=nodes[0].id, target=nodes[i + 1].id) for i in range(edges)],
    )


@pytest.mark.parametrize("n", [0, 1])
def test_a_graph_with_no_edges_explains_itself_instead_of_drawing_nothing(n: int) -> None:
    """A lone node is not a graph; an empty mermaid block would look broken rather than empty."""
    out = format_graph("paper.pdf", _graph(n))
    assert "no internal citation edges" in out
    assert "mermaid" not in out


def test_a_small_graph_renders_a_mermaid_block() -> None:
    out = format_graph("paper.pdf", _graph(3, edges=2))
    assert "```mermaid" in out
    assert "graph LR" in out
    assert out.rstrip().endswith("```")


def test_the_centre_node_is_marked_so_the_reader_can_find_the_subject() -> None:
    out = format_graph("paper.pdf", _graph(3, edges=2))
    assert ":::center" in out
    assert out.count(":::center") == 1
    assert "classDef center" in out


def test_every_edge_reaches_the_diagram() -> None:
    out = format_graph("paper.pdf", _graph(4, edges=3))
    assert out.count(" --> ") == 3


def test_a_graph_too_large_to_draw_reports_its_size_instead_of_rendering() -> None:
    """26 nodes is the documented ceiling; past it the answer is a count and a pointer, never a
    mermaid block the renderer would choke on."""
    out = format_graph("paper.pdf", _graph(30, edges=5))
    assert "30 nodes, 5 edges" in out
    assert "mermaid" not in out
    assert "graph_subgraph" in out


def test_a_node_label_with_a_double_quote_cannot_break_the_mermaid_syntax() -> None:
    """Mermaid labels are double-quoted; an unescaped quote in a filename ends the label early
    and corrupts the whole diagram."""
    nodes = [
        GraphNode(id="0" * 32, filename='a"quoted".pdf', title=None, is_center=True),
        GraphNode(id="1" * 32, filename="b.pdf", title=None, is_center=False),
    ]
    out = format_graph("p.pdf", CitationGraph(nodes=nodes, edges=[GraphEdge("0" * 32, "1" * 32)]))
    assert "00000000[\"**a'quoted'.pdf**\"]:::center" in out
    assert '"quoted"' not in out


def test_a_long_node_label_is_truncated_to_keep_the_diagram_readable() -> None:
    nodes = [
        GraphNode(id="0" * 32, filename="z" * 60, title=None, is_center=True),
        GraphNode(id="1" * 32, filename="b.pdf", title=None, is_center=False),
    ]
    out = format_graph("p.pdf", CitationGraph(nodes=nodes, edges=[GraphEdge("0" * 32, "1" * 32)]))
    assert "z" * 30 in out
    assert "z" * 31 not in out


# ============================================================
# help_message
# ============================================================


def test_help_lists_every_command_the_dispatcher_implements() -> None:
    """A command the user cannot discover is a command that does not exist."""
    text = help_message()
    for cmd in ("/library", "/document", "/cites", "/cited-by", "/similar", "/graph", "/help"):
        assert cmd in text, f"{cmd} is dispatched but undocumented"
