"""The library-metadata router, which sits on the chat hot path and had almost no coverage.

`query_router` was at **29%** in the 2026-08-20 coverage pass — the lowest of any module that runs
on a user request. `ChatController._stream` consults it on **every** message
(`chat_controller/controller.py:305`): a match short-circuits the whole RAG pipeline and
`answer_library_query`'s return value is shown to the user verbatim. Both halves of that decision
were untested.

Two things make this worth real tests rather than a smoke check:

* **The negative lookahead is the load-bearing part.** `_NOT_TOPICAL` is what keeps *"show my
  papers about RAG"* out of the metadata branch. If it stops firing, topical questions silently
  stop reaching retrieval and get a document count instead — a total answer failure that raises no
  error and logs nothing.
* **`answer_library_query` is a formatter over live SQLite.** Its zero-document branches are the
  robustness contract in miniature, and they are exactly the branches a populated dev box never
  executes.

The library reads are monkeypatched at `doc_assistant.query_router`, not at
`doc_assistant.library`: `query_router` binds both names at import time, so patching the source
module would leave this module's references untouched (the same trap `chat_controller/__init__.py`
documents for `is_library_query`).
"""

from __future__ import annotations

from datetime import datetime

import pytest

from doc_assistant.library.models import DocumentSummary, LibrarySummary
from doc_assistant.query_router import answer_library_query, health_badge, is_library_query

# ============================================================
# Fixtures
# ============================================================


def _doc(
    name: str,
    *,
    health: str = "healthy",
    chunks: int = 10,
    added: datetime | None = None,
) -> DocumentSummary:
    return DocumentSummary(
        id=f"{name}-0123456789abcdef",
        filename=f"{name}.pdf",
        title=name.title(),
        format="pdf",
        health=health,
        chunk_count=chunks,
        page_count=5,
        added_at=added,
    )


def _summary(*, documents: int = 3, chunks: int = 120) -> LibrarySummary:
    return LibrarySummary(
        total_documents=documents,
        total_chunks=chunks,
        by_health={"healthy": documents, "broken": 0},
        by_format={"pdf": documents},
    )


@pytest.fixture
def library(monkeypatch: pytest.MonkeyPatch):
    """Patch the two library reads `query_router` bound at import time.

    Returns a setter so each test states the library it is asking about, including the empty one.
    """

    def configure(docs: list[DocumentSummary], summary: LibrarySummary | None = None) -> None:
        monkeypatch.setattr(
            "doc_assistant.query_router.list_documents",
            lambda health=None, **kw: (
                [d for d in docs if d.health == health] if health else list(docs)
            ),
        )
        monkeypatch.setattr(
            "doc_assistant.query_router.library_summary",
            lambda: summary if summary is not None else _summary(documents=len(docs)),
        )

    return configure


# ============================================================
# is_library_query — the routing decision
# ============================================================


@pytest.mark.parametrize(
    "text",
    [
        "what is my latest document?",
        "show me the newest paper",
        "how many documents do I have",
        "how many chunks are indexed",
        "list all my documents",
        "show my papers",
        "what's in my library",
        "library stats",
        "collection summary",
        "do I have any broken documents",
        "any marginal files?",
        "document count",
        "paper total",
    ],
)
def test_metadata_questions_route_to_the_library(text: str) -> None:
    assert is_library_query(text)


@pytest.mark.parametrize(
    ("base", "topical"),
    [
        ("show my papers", "show my papers about RAG"),
        ("list all my documents", "list all my documents on hippocampal replay"),
        ("show my papers", "show my papers regarding transformers"),
        ("what's in my library", "what's in my library related to connectomics"),
        ("list my documents", "list my documents that mention Cajal"),
        ("list all my papers", "list all my papers discussing attention"),
    ],
)
def test_a_topical_qualifier_sends_the_question_to_retrieval_instead(
    base: str, topical: str
) -> None:
    """`_NOT_TOPICAL` is the difference between a document count and a real answer.

    If the lookahead regresses, the user asks about their documents' *content* and is told how many
    they own instead, with no error anywhere to show that retrieval was skipped.

    **The `base` half is what stops this being vacuous.** Asserting only that the topical phrasing
    is rejected would keep passing if the underlying pattern stopped matching altogether — the test
    would then be guarding nothing while looking green. Both halves are asserted together so the
    lookahead is proven to be the thing making the difference.
    """
    assert is_library_query(base), "the pattern must match before the qualifier is added"
    assert not is_library_query(topical)


@pytest.mark.parametrize(
    "text",
    [
        "what does the hippocampus do",
        "summarise the Cajal paper",
        "how does BM25 scoring work",
        "",
    ],
)
def test_ordinary_content_questions_do_not_route_to_the_library(text: str) -> None:
    assert not is_library_query(text)


def test_routing_is_case_insensitive() -> None:
    assert is_library_query("HOW MANY DOCUMENTS DO I HAVE")


# ============================================================
# health_badge
# ============================================================


@pytest.mark.parametrize("health", ["healthy", "marginal", "broken"])
def test_health_badge_names_the_state_it_was_given(health: str) -> None:
    assert health in health_badge(health)


def test_unknown_and_missing_health_both_read_as_unknown() -> None:
    """A document with no recorded health must not borrow another state's badge."""
    assert health_badge(None) == health_badge("something-unrecognised")
    assert "unknown" in health_badge(None)


def test_every_health_state_gets_a_distinct_badge() -> None:
    badges = {health_badge(h) for h in ("healthy", "marginal", "broken", None)}
    assert len(badges) == 4


# ============================================================
# answer_library_query — latest / newest
# ============================================================


def test_latest_document_reports_name_size_and_health(library) -> None:
    library(
        [
            _doc("older", added=datetime(2026, 1, 1)),
            _doc("newest", chunks=42, added=datetime(2026, 8, 1)),
        ]
    )
    answer = answer_library_query("what is my latest document?")
    assert "newest.pdf" in answer
    assert "42 chunks" in answer
    assert "2026-08-01" in answer


def test_latest_document_on_an_empty_library_says_so_and_says_what_to_do(library) -> None:
    """The 0-document robustness contract: degrade honestly, and name the next action."""
    library([], _summary(documents=0, chunks=0))
    answer = answer_library_query("what is my latest document?")
    assert "empty" in answer.lower()
    assert "data/sources/" in answer


def test_documents_without_add_dates_do_not_claim_a_latest(library) -> None:
    """Every document predates the `added_at` column; picking one arbitrarily would be a guess."""
    library([_doc("undated_a"), _doc("undated_b")])
    answer = answer_library_query("show me the newest paper")
    assert "no recorded add dates" in answer.lower()
    assert "undated_a.pdf" not in answer


# ============================================================
# answer_library_query — health
# ============================================================


def test_broken_documents_are_listed_by_name(library) -> None:
    library([_doc("good"), _doc("bad", health="broken")])
    answer = answer_library_query("do I have any broken documents")
    assert "bad.pdf" in answer
    assert "good.pdf" not in answer


def test_no_broken_documents_is_stated_plainly(library) -> None:
    library([_doc("good")])
    assert "No broken documents" in answer_library_query("any broken documents?")


def test_a_long_broken_list_is_truncated_and_says_how_many_it_hid(library) -> None:
    """A silent cap would read as "these are all of them" — the count is what keeps it honest."""
    library([_doc(f"bad{i}", health="broken") for i in range(13)])
    answer = answer_library_query("list broken documents")
    assert "13 broken" in answer
    assert "and 3 more" in answer


def test_marginal_is_routed_separately_from_broken(library) -> None:
    library([_doc("shaky", health="marginal"), _doc("bad", health="broken")])
    answer = answer_library_query("any marginal files?")
    assert "shaky.pdf" in answer
    assert "bad.pdf" not in answer


# ============================================================
# answer_library_query — counts and the fallback
# ============================================================


def test_how_many_reports_totals_and_the_health_breakdown(library) -> None:
    library([_doc("a")], _summary(documents=97, chunks=36717))
    answer = answer_library_query("how many documents do I have")
    assert "97 documents" in answer
    assert "36,717 chunks" in answer, "thousands separator keeps large corpora readable"
    assert "healthy" in answer


def test_a_generic_library_question_falls_back_to_an_overview(library) -> None:
    library([_doc("a")], _summary(documents=97, chunks=36717))
    answer = answer_library_query("what's in my library")
    assert "97 documents" in answer
    assert "/library" in answer


# ============================================================
# The contract between the two halves
# ============================================================


@pytest.mark.parametrize(
    "text",
    [
        "what is my latest document?",
        "how many documents do I have",
        "list all my documents",
        "what's in my library",
        "library stats",
        "do I have any broken documents",
        "document count",
    ],
)
def test_everything_the_router_claims_it_can_answer_produces_an_answer(library, text: str) -> None:
    """The two functions are used as a pair, so a gap between them is a blank reply to the user.

    Run against an empty library, because that is where a formatter is most likely to reach for a
    value that is not there.
    """
    library([], _summary(documents=0, chunks=0))
    answer = answer_library_query(text)
    assert isinstance(answer, str)
    assert answer.strip()
