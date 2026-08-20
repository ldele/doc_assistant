"""Tests for the library data access layer."""

import contextlib
import os
import tempfile

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker


@pytest.fixture
def temp_database(monkeypatch):
    """Replace the global engine with one pointing to a temp database.

    This is the correct way to isolate database tests — patch the engine,
    not just the config string.
    """
    # Create temp DB file
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    # Build a fresh engine pointing at the temp file
    test_engine = create_engine(f"sqlite:///{path}", future=True)

    @event.listens_for(test_engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Patch both the engine and the session factory in db.session
    from doc_assistant.db import session as session_module

    monkeypatch.setattr(session_module, "_engine", test_engine)
    test_session_factory = sessionmaker(
        bind=test_engine, autoflush=False, autocommit=False, future=True
    )
    monkeypatch.setattr(session_module, "_SessionLocal", test_session_factory)

    # Create tables in the temp DB
    from doc_assistant.db.models import Base

    Base.metadata.create_all(test_engine)

    yield path

    # Teardown
    test_engine.dispose()
    with contextlib.suppress(OSError):
        os.unlink(path)


# Now all the test functions take temp_database as a parameter
# to opt into the isolated DB:


def test_empty_library_summary(temp_database):
    from doc_assistant.library import library_summary

    s = library_summary()
    assert s.total_documents == 0
    assert s.total_chunks == 0


def test_add_and_list_document(temp_database):
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import list_documents

    with session_scope() as session:
        doc = Document(
            filename="test.pdf",
            source_original="/tmp/test.pdf",
            doc_hash="abcd1234",
            format="pdf",
            extraction_health="healthy",
            chunk_count=10,
        )
        session.add(doc)

    docs = list_documents()
    assert len(docs) == 1
    assert docs[0].filename == "test.pdf"


def test_filter_by_health(temp_database):
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import list_documents

    with session_scope() as session:
        for i, h in enumerate(["healthy", "broken", "healthy"]):
            session.add(
                Document(
                    filename=f"doc{i}.pdf",
                    source_original=f"/tmp/doc{i}.pdf",
                    doc_hash=f"hash{i}",
                    format="pdf",
                    extraction_health=h,
                    chunk_count=10,
                )
            )

    healthy = list_documents(health="healthy")
    broken = list_documents(health="broken")
    assert len(healthy) == 2
    assert len(broken) == 1


def test_short_id_lookup(temp_database):
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import find_document_by_short_id

    with session_scope() as session:
        doc = Document(
            filename="test.pdf",
            source_original="/tmp/test.pdf",
            doc_hash="hashxyz",
            format="pdf",
        )
        session.add(doc)
        session.flush()
        full_id = doc.id

    found = find_document_by_short_id(full_id[:8])
    assert found == full_id


# ============================================================
# Chunk browser (Library L1 — feature-library-browser.md)
# ============================================================
# `group_children` is pure — flat Chroma child chunks -> ordered parent blocks. No DB, no Chroma.
# The impure `get_document_chunks` (SQLite + live handle) is covered by the API integration test.


def _chunk(p, c, text, parent_text="P", keep=None):
    return {
        "parent_index": p,
        "child_index": c,
        "parent_text": parent_text,
        "text": text,
        "keep_for_retrieval": keep,
    }


def test_group_children_orders_and_groups():
    from doc_assistant.library import ChunkChild, ParentBlock, group_children

    # Deliberately out of order across two parents; child 0 of parent 0 is not retrievable.
    blocks = group_children(
        [
            _chunk(1, 1, "p1c1", parent_text="P1"),
            _chunk(0, 1, "p0c1", parent_text="P0"),
            _chunk(0, 0, "p0c0", parent_text="P0", keep=False),
            _chunk(1, 0, "p1c0", parent_text="P1"),
        ]
    )

    assert [b.parent_index for b in blocks] == [0, 1]  # parents ordered by parent_index
    assert isinstance(blocks[0], ParentBlock)
    assert blocks[0].parent_text == "P0"  # from the first-seen child of the parent
    assert [(c.child_index, c.text, c.retrievable) for c in blocks[0].children] == [
        (0, "p0c0", False),  # keep_for_retrieval=False -> retrievable=False
        (1, "p0c1", True),  # None -> retrievable (kept)
    ]
    assert isinstance(blocks[0].children[0], ChunkChild)
    assert [c.text for c in blocks[1].children] == ["p1c0", "p1c1"]


def test_group_children_drops_rows_missing_index():
    from doc_assistant.library import group_children

    blocks = group_children(
        [_chunk(0, 0, "keep"), _chunk(None, 0, "drop-no-parent"), _chunk(0, None, "drop-no-child")]
    )
    assert len(blocks) == 1
    assert [c.text for c in blocks[0].children] == ["keep"]


def test_group_children_parent_index_zero_is_not_dropped():
    from doc_assistant.library import group_children

    # Falsy-but-valid indices (0) must survive — the guard is `is None`, not truthiness.
    blocks = group_children([_chunk(0, 0, "zero")])
    assert [b.parent_index for b in blocks] == [0]
    assert blocks[0].children[0].child_index == 0


def test_group_children_empty():
    from doc_assistant.library import group_children

    assert group_children([]) == []


# ============================================================
# ADR-027 D1 (E4) — document_connections (the exploration bundle)
# ============================================================


def _seed_document(filename: str, *, title: str | None = None) -> str:
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = Document(
            filename=filename,
            source_original=f"/tmp/{filename}",
            doc_hash=f"hash-{filename}",
            format="pdf",
            title=title,
        )
        session.add(doc)
        session.flush()
        return str(doc.id)


def _seed_similarity(
    source_id: str, target_id: str, score: float, model: str = "bge-base"
) -> None:
    from doc_assistant.db.models import DocSimilarity
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            DocSimilarity(
                source_document_id=source_id,
                target_document_id=target_id,
                embedding_model=model,
                score=score,
            )
        )


def _seed_citation(
    source_id: str,
    *,
    target_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
    authors: str | None = None,
    doi: str | None = None,
    raw: str | None = None,
) -> None:
    from doc_assistant.db.models import Citation
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        session.add(
            Citation(
                source_document_id=source_id,
                target_document_id=target_id,
                target_title=title,
                target_year=year,
                target_authors=authors,
                target_doi=doi,
                raw_citation_text=raw,
                extraction_method="regex",
            )
        )


def test_document_connections_unknown_doc_is_none(temp_database):
    from doc_assistant.library import document_connections

    assert document_connections("no-such-id") is None


def test_document_connections_empty_sidecars_degrade_to_empty_lists(temp_database):
    # The 0-doc/0-sidecar contract: a known doc with nothing computed returns an all-empty
    # bundle, never an error — the panel renders an honest empty state from this.
    from doc_assistant.library import document_connections

    doc = _seed_document("a.pdf")
    bundle = document_connections(doc)
    assert bundle is not None
    assert bundle.related == []
    assert bundle.cited_by == []


def test_document_connections_related_scoped_to_embedding_model(temp_database):
    # The similarity read is scoped to the embedder in use — edges computed under another
    # model must not leak into the panel (they describe a different geometry).
    from doc_assistant.library import document_connections

    a, b = _seed_document("a.pdf"), _seed_document("b.pdf")
    _seed_similarity(a, b, 0.91, model="bge-base")
    scoped = document_connections(a, embedding_model="bge-base")
    other = document_connections(a, embedding_model="specter2")
    assert scoped is not None and [r.target_document_id for r in scoped.related] == [b]
    assert other is not None and other.related == []


def test_document_connections_carries_no_outgoing_citations(temp_database):
    # The bundle is this document's *neighbourhood*, not its bibliography (2026-08-10): what it
    # cites belongs to `document_references`, in one list. A guard, not a tautology — putting
    # the resolved half back here is exactly how the two blocks would start disagreeing.
    from doc_assistant.library import document_connections

    a, b = _seed_document("a.pdf"), _seed_document("b.pdf")
    _seed_citation(a, target_id=b, title="Resolved in-corpus paper")
    _seed_citation(a, title="External titled ref", year=2019)

    bundle = document_connections(a)
    assert bundle is not None
    assert not hasattr(bundle, "cites")
    assert not hasattr(bundle, "external_refs")
    assert bundle.cited_by == []  # a cites b, so b is cited_by a — a itself is cited by nobody


def test_document_connections_dedupes_cited_by_with_count(temp_database):
    # A doc citing the subject 3 times is ONE row with n_citations=3 — the panel lists
    # documents, not raw citation rows.
    from doc_assistant.library import document_connections

    a, c = _seed_document("subject.pdf"), _seed_document("citer.pdf")
    for _ in range(3):
        _seed_citation(c, target_id=a, title="Subject paper")

    bundle = document_connections(a)
    assert bundle is not None
    assert len(bundle.cited_by) == 1
    assert bundle.cited_by[0].document_id == c
    assert bundle.cited_by[0].n_citations == 3


# ============================================================
# document_references — the Library document view's References block
# ============================================================


def test_resolution_is_credible_rejects_the_surname_year_false_positive():
    # The real shape of the defect (measured 2026-08-10): `match_to_library`'s rule 2 needs
    # only first-author surname + year, so a 2024 knowledge-graph reference resolved to a 2024
    # paper on mouse whisker cortex. 13 of this library's 16 stored resolutions look like this.
    from doc_assistant.library.citations import resolution_is_credible

    assert not resolution_is_credible(
        parsed_title="A review of graph neural networks and pretrained language models",
        parsed_doi=None,
        library_title="Cell class-specific long-range axonal projections of neurons in mouse",
        library_doi=None,
    )


def test_resolution_is_credible_accepts_a_matching_title_and_an_exact_doi():
    from doc_assistant.library.citations import resolution_is_credible

    assert resolution_is_credible(
        parsed_title="From local to global: A graph rag approach to query-focused summarization",
        parsed_doi=None,
        library_title="From Local to Global: A GraphRAG Approach to Query-Focused Summarization",
        library_doi=None,
    )
    # A DOI both sides carry is decisive on its own — titles need not agree at all.
    assert resolution_is_credible(
        parsed_title=None,
        parsed_doi="10.1038/S41592-022-01443-0",
        library_title="Something else entirely",
        library_doi="10.1038/s41592-022-01443-0",
    )


def test_resolution_is_credible_accepts_a_title_buried_in_an_author_prefix():
    # The case a strict ratio loses: the regex prefixes the title with the tail of the author
    # list, scoring 0.78 on a *true* match. Containment recovers it — and on the real corpus
    # admitted none of the 12 false links, which contain nothing and score 0.11-0.37.
    from doc_assistant.library.citations import resolution_is_credible

    assert resolution_is_credible(
        parsed_title=(
            "A., Lopes, G., Saunders, J. L., Mathis, A. & Mathis, M. W. Real-time, "
            "low-latency closed-loop feedback using markerless posture tracking"
        ),
        parsed_doi=None,
        library_title=(
            "Real-time, low-latency closed-loop feedback using markerless posture tracking"
        ),
        library_doi=None,
    )


def test_resolution_is_credible_needs_a_title_on_both_sides():
    # A resolution that cannot be checked is not credible: it came from a rule that never
    # compared titles in the first place.
    from doc_assistant.library.citations import resolution_is_credible

    assert not resolution_is_credible(
        parsed_title=None, parsed_doi=None, library_title="A Title", library_doi=None
    )
    assert not resolution_is_credible(
        parsed_title="A Title", parsed_doi=None, library_title=None, library_doi=None
    )


def test_resolution_is_credible_ignores_a_short_coincidental_containment():
    # "a survey" sits inside hundreds of titles — containment only counts for a fragment long
    # enough that sharing it is evidence rather than coincidence.
    from doc_assistant.library.citations import resolution_is_credible

    assert not resolution_is_credible(
        parsed_title="A survey",
        parsed_doi=None,
        library_title="A survey of techniques for constructing chinese knowledge graphs",
        library_doi=None,
    )


def test_plausible_year_drops_what_cannot_be_a_publication_year():
    from datetime import date

    from doc_assistant.library.citations import plausible_year

    today = date(2026, 8, 10)
    assert plausible_year(2024, today=today) == 2024
    assert plausible_year(1901, today=today) == 1901
    assert plausible_year(2027, today=today) == 2027  # in press, carries next year's date
    assert plausible_year(2089, today=today) is None  # an identifier the regex lifted
    assert plausible_year(1799, today=today) is None
    assert plausible_year(None, today=today) is None


def test_document_references_unlinks_a_resolution_that_does_not_check_out(temp_database):
    # End to end: the reference stays in the list (the paper does cite it), the link does not.
    from doc_assistant.db.models import Document
    from doc_assistant.db.session import session_scope
    from doc_assistant.library import document_references

    a, b = _seed_document("a.pdf"), _seed_document("b.pdf")
    with session_scope() as session:
        session.get(Document, b).title = "Mamba-UNet: pure visual mamba for medical images"
    _seed_citation(a, target_id=b, title="QAGCN: answering multi-relation questions", year=2024)

    view = document_references(a)
    assert view is not None
    assert view.total == 1 and view.shown == 1
    assert view.in_library == 0
    assert view.references[0].target_document_id is None
    assert view.references[0].title == "QAGCN: answering multi-relation questions"


def test_document_references_sinks_an_impossible_year_instead_of_heading_the_list(temp_database):
    # Sorting newest-first on the raw field put "(2089)" at the top of the block — the first
    # thing the reader sees, and a number no reference in this corpus actually carries.
    from doc_assistant.library import document_references

    a = _seed_document("a.pdf")
    _seed_citation(a, title="Junk year", year=2089)
    _seed_citation(a, title="Real paper", year=2015)

    view = document_references(a)
    assert view is not None
    assert [r.title for r in view.references] == ["Real paper", "Junk year"]
    assert [r.year for r in view.references] == [2015, None]


def test_document_references_unknown_doc_is_none(temp_database):
    from doc_assistant.library import document_references

    assert document_references("no-such-id") is None


def test_document_references_no_citations_is_an_empty_list_not_none(temp_database):
    # The 0-doc contract: "this paper's bibliography was never extracted" is an ordinary
    # state the panel renders as an honest empty, not a 404.
    from doc_assistant.library import document_references

    view = document_references(_seed_document("a.pdf"))
    assert view is not None
    assert view.references == []
    assert (view.total, view.in_library, view.shown) == (0, 0, 0)


def test_document_references_keeps_unresolved_and_untitled_rows(temp_database):
    # The whole point of a *bibliography*: a reference that matched nothing, and one that
    # parsed no title at all, are still references this paper makes. 243 of this corpus's
    # 4,374 rows are untitled — dropping them would misstate what the paper cites (the
    # connections bundle drops them by design, which is why it is not this block's source).
    from doc_assistant.library import document_references

    a = _seed_document("a.pdf")
    b = _seed_document("b.pdf", title="Owned paper")
    _seed_citation(a, target_id=b, title="Owned paper", year=2021)
    _seed_citation(a, title="Not in the library", year=2020)
    _seed_citation(a, raw="[3] a line the regex could not parse", year=2019)

    view = document_references(a)
    assert view is not None
    assert view.total == 3
    assert view.shown == 3
    assert view.in_library == 1
    assert [r.target_document_id for r in view.references] == [b, None, None]


def test_document_references_marks_the_owned_row_with_the_library_title(temp_database):
    # The parsed title is extraction output; the owned document's own title is the one the
    # library vouches for, so the link renders from that when the two disagree.
    from doc_assistant.library import document_references

    a = _seed_document("a.pdf")
    b = _seed_document("b.pdf", title="Attention Is All You Need")
    _seed_citation(a, target_id=b, title="attention is all you need. In")

    view = document_references(a)
    assert view is not None
    ref = view.references[0]
    assert ref.target_document_id == b
    assert ref.library_title == "Attention Is All You Need"
    assert ref.title == "attention is all you need. In"
    assert ref.target_filename == "b.pdf"


def test_document_references_cap_never_drops_a_reference_you_own(temp_database):
    # The cap is a wire-size bound, and the rows worth the budget are the ones the reader can
    # actually open. A resolved reference sitting at position 300 of 346 must survive a cap of
    # 2 — otherwise the block's one interactive feature disappears on exactly the long
    # bibliographies where it matters.
    from doc_assistant.library import document_references

    a = _seed_document("a.pdf")
    owned = _seed_document("owned.pdf", title="Owned paper")
    for i in range(5):
        _seed_citation(a, title=f"External ref {i}", year=2020 - i)
    _seed_citation(a, target_id=owned, title="Owned paper", year=1990)  # sorts last by year

    view = document_references(a, cap=2)
    assert view is not None
    assert view.total == 6
    assert view.shown == 2
    assert view.in_library == 1  # counted over ALL references, not the shown slice
    assert owned in [r.target_document_id for r in view.references]


def test_document_references_are_ordered_newest_first(temp_database):
    # The paper's own numbering is not recorded (Citation has no ordinal and its id is a
    # uuid4), so the order is year-descending — and the panel says so rather than implying
    # the list is the paper's own.
    from doc_assistant.library import document_references

    a = _seed_document("a.pdf")
    for year in (1998, 2024, 2011):
        _seed_citation(a, title=f"Ref {year}", year=year)

    view = document_references(a)
    assert view is not None
    assert [r.year for r in view.references] == [2024, 2011, 1998]


# ============================================================
# ADR-013 metadata overrides — the list and the detail view must agree
# ============================================================


def _doc_with_override(title_override=None, authors_override=None, year_override=None):
    """One document with extracted values, plus whatever override the test wants."""
    from doc_assistant.db.models import Document, DocumentMeta
    from doc_assistant.db.session import session_scope

    with session_scope() as session:
        doc = Document(
            id="doc-1",
            filename="scan.pdf",
            doc_hash="hash-1",
            format="pdf",
            source_original="/docs/scan.pdf",
            title="A Revised Neuroanatom of Cireuits",  # as OCR read it
            authors="FRANK A. MIDDLETON PETER L. STRICK",
            year=2001,
            chunk_count=52,
            extraction_health="healthy",
        )
        session.add(doc)
        if any(v is not None for v in (title_override, authors_override, year_override)):
            session.add(
                DocumentMeta(
                    document_id="doc-1",
                    title_override=title_override,
                    authors_override=authors_override,
                    year_override=year_override,
                )
            )
    return "doc-1"


def test_the_detail_view_applies_a_title_override(temp_database):
    """The bug this pins (2026-08-19): the grid showed the corrected title and the document's own
    page showed the raw extracted one, because the merge lived in `list_documents` alone."""
    from doc_assistant.library import get_document_details

    doc_id = _doc_with_override(title_override="A Revised Neuroanatomy of Circuits")

    details = get_document_details(doc_id)

    assert details is not None
    assert details.title == "A Revised Neuroanatomy of Circuits"


def test_the_list_and_the_detail_view_agree(temp_database):
    """The property that keeps them from drifting again: one document, one answer."""
    from doc_assistant.library import get_document_details, list_documents

    doc_id = _doc_with_override(
        title_override="A Revised Neuroanatomy of Circuits",
        authors_override="Frank A. Middleton, Peter L. Strick",
    )

    listed = next(d for d in list_documents() if d.id == doc_id)
    details = get_document_details(doc_id)

    assert details is not None
    assert (details.title, details.authors, details.year) == (
        listed.title,
        listed.authors,
        listed.year,
    )


def test_without_an_override_both_report_what_extraction_found(temp_database):
    """The override is additive: with none stored, nothing is invented or hidden."""
    from doc_assistant.library import get_document_details, list_documents

    doc_id = _doc_with_override()

    listed = next(d for d in list_documents() if d.id == doc_id)
    details = get_document_details(doc_id)

    assert details is not None
    assert details.title == "A Revised Neuroanatom of Cireuits" == listed.title
    assert details.year == 2001


def test_a_partial_override_leaves_the_other_fields_extracted(temp_database):
    """Overriding the title must not blank the authors — each field stands alone (ADR-013)."""
    from doc_assistant.library import get_document_details

    doc_id = _doc_with_override(title_override="A Revised Neuroanatomy of Circuits")

    details = get_document_details(doc_id)

    assert details is not None
    assert details.authors == "FRANK A. MIDDLETON PETER L. STRICK"
    assert details.year == 2001


class _EmptyChroma:
    """Chroma-shaped `get()` with no chunks — these tests are about the header, not the body."""

    def get(self, *, where=None, include=None):
        return {"documents": [], "metadatas": []}


def test_the_document_page_header_applies_the_override(temp_database):
    """The surface the user actually reported: the page title stayed as extraction read it while
    the grid beside it showed the correction (2026-08-19). `get_document_chunks` is a separate
    read path from both `list_documents` and `get_document_details`."""
    from doc_assistant.library import get_document_chunks

    doc_id = _doc_with_override(
        title_override="A Revised Neuroanatomy of Circuits",
        authors_override="Frank A. Middleton, Peter L. Strick",
    )

    view = get_document_chunks(doc_id, _EmptyChroma())

    assert view is not None
    assert view.title == "A Revised Neuroanatomy of Circuits"
    assert view.authors == "Frank A. Middleton, Peter L. Strick"


def test_the_figures_block_applies_the_override(temp_database):
    """Otherwise one document is named two different things on one screen."""
    from doc_assistant.library.figures import list_document_figures

    doc_id = _doc_with_override(title_override="A Revised Neuroanatomy of Circuits")

    view = list_document_figures(doc_id)

    assert view is not None
    assert view.title == "A Revised Neuroanatomy of Circuits"


def test_every_display_surface_agrees_on_the_title(temp_database):
    """The property, stated once over all four read paths. A fifth surface added later without
    the merge fails here rather than in a screenshot."""
    from doc_assistant.library import get_document_chunks, get_document_details, list_documents
    from doc_assistant.library.figures import list_document_figures

    doc_id = _doc_with_override(title_override="A Revised Neuroanatomy of Circuits")

    titles = {
        "list": next(d for d in list_documents() if d.id == doc_id).title,
        "details": get_document_details(doc_id).title,
        "chunks": get_document_chunks(doc_id, _EmptyChroma()).title,
        "figures": list_document_figures(doc_id).title,
    }

    assert set(titles.values()) == {"A Revised Neuroanatomy of Circuits"}, titles


def test_the_year_used_for_analysis_stays_the_extracted_one(temp_database):
    """Deliberate asymmetry: `document_years` feeds the year-aware epistemics rule, which is an
    analysis of what the corpus says — not a display. A metadata edit must not silently move a
    knowledge-layer verdict; that is a decision with an eval behind it, not a consistency fix.
    """
    from doc_assistant.library.documents import document_years

    doc_id = _doc_with_override(year_override=1999)

    assert document_years([doc_id]) == {doc_id: 2001}
