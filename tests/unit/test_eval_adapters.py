"""The eval harness's one adapter onto `RAGPipeline`, at 26% coverage (2026-08-20).

`eval/adapters.py` is small but load-bearing and structurally special: its own docstring calls it
**the only module in `doc_assistant.eval` that depends on the rest of `doc_assistant`**, and
extracting the harness to a standalone repo (Feature 5) means deleting this file and rewriting it
against a new system-under-test. That makes its contract — *what shape must a pipeline present, and
what must come back* — the interface the whole harness rests on, and it was unasserted.

The two things it is responsible for are the two that quietly corrupt a run rather than failing it:

* **Per-case token accounting.** A fresh `TokenCounter` per query is what makes cost per-case
  rather than per-run. If the counter leaked across calls, every eval row after the first would
  overstate its cost, and the numbers would still look plausible.
* **Citation extraction.** `citation_overlap` is the harness's zero-variance retrieval scorer
  (`tests/eval/TESTING.md`), and it reads exactly the list this adapter builds. Duplicate or
  mis-ordered filenames here move a scorer that is supposed to have no variance at all.

The pipeline is a hand-rolled stub rather than a mock: the adapter only needs `retrieve`,
`stream_answer` and `embeddings`, and stating that in ~20 lines documents the required shape better
than a `Mock` that would accept any call at all — including the ones a future refactor breaks.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from doc_assistant.eval.adapters import embedding_callable, rag_pipeline_adapter

# ============================================================
# Stub pipeline
# ============================================================


class _Doc:
    """The one thing the adapter reads off a retrieved chunk: `.metadata`."""

    def __init__(self, **metadata: Any) -> None:
        self.metadata = metadata


class _Embeddings:
    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector if vector is not None else [1, 2, 3]
        self.calls: list[str] = []

    def embed_query(self, text: str) -> list[Any]:
        self.calls.append(text)
        return list(self.vector)


class _StubPipeline:
    """Minimal stand-in for `RAGPipeline` — exactly the surface the adapter touches."""

    def __init__(
        self,
        docs: list[_Doc] | None = None,
        answer_chunks: list[str] | None = None,
        *,
        tokens: tuple[int, int] = (11, 7),
    ) -> None:
        self.docs = docs if docs is not None else [_Doc(filename="a.pdf")]
        self.answer_chunks = answer_chunks if answer_chunks is not None else ["Hello", " world"]
        self.tokens = tokens
        self.embeddings = _Embeddings()
        self.retrieve_calls: list[str] = []
        self.stream_calls: list[tuple[str, int]] = []

    def retrieve(self, query: str) -> list[_Doc]:
        self.retrieve_calls.append(query)
        return list(self.docs)

    def stream_answer(self, query: str, docs: list[_Doc], counter: Any = None) -> Iterator[str]:
        self.stream_calls.append((query, len(docs)))
        if counter is not None:
            counter.input_tokens += self.tokens[0]
            counter.output_tokens += self.tokens[1]
        yield from self.answer_chunks


@pytest.fixture
def pipeline() -> _StubPipeline:
    return _StubPipeline()


# ============================================================
# rag_pipeline_adapter — the answer
# ============================================================


def test_the_streamed_chunks_are_joined_into_one_answer(pipeline: _StubPipeline) -> None:
    pipeline.answer_chunks = ["The ", "hippo", "campus."]
    assert rag_pipeline_adapter(pipeline)("q").answer == "The hippocampus."


def test_the_answer_is_stripped(pipeline: _StubPipeline) -> None:
    """Leading whitespace from a first token would defeat `exact_match` on every case."""
    pipeline.answer_chunks = ["\n  ", "Answer.", "  \n"]
    assert rag_pipeline_adapter(pipeline)("q").answer == "Answer."


def test_a_stream_that_yields_nothing_gives_an_empty_answer_not_a_crash(
    pipeline: _StubPipeline,
) -> None:
    """A model returning no content is a real, seen failure (KI-28) — the run must record it as
    an empty answer and score it, not abort the whole eval."""
    pipeline.answer_chunks = []
    assert rag_pipeline_adapter(pipeline)("q").answer == ""


def test_the_query_reaches_both_retrieval_and_generation(pipeline: _StubPipeline) -> None:
    rag_pipeline_adapter(pipeline)("what is a cortex?")
    assert pipeline.retrieve_calls == ["what is a cortex?"]
    assert pipeline.stream_calls == [("what is a cortex?", 1)]


def test_generation_is_given_the_documents_retrieval_returned(pipeline: _StubPipeline) -> None:
    """Retrieving one set and generating from another would make every score meaningless."""
    pipeline.docs = [_Doc(filename="a.pdf"), _Doc(filename="b.pdf"), _Doc(filename="c.pdf")]
    rag_pipeline_adapter(pipeline)("q")
    assert pipeline.stream_calls[0][1] == 3


# ============================================================
# rag_pipeline_adapter — citations
# ============================================================


def test_citations_are_the_filenames_of_the_retrieved_chunks(pipeline: _StubPipeline) -> None:
    pipeline.docs = [_Doc(filename="a.pdf"), _Doc(filename="b.pdf")]
    assert rag_pipeline_adapter(pipeline)("q").citations == ["a.pdf", "b.pdf"]


def test_repeated_filenames_are_deduplicated(pipeline: _StubPipeline) -> None:
    """Several chunks from one paper is the normal case, and `citation_overlap` is a set
    comparison — a duplicated filename inflates nothing but misreports what was retrieved."""
    pipeline.docs = [_Doc(filename="a.pdf"), _Doc(filename="a.pdf"), _Doc(filename="b.pdf")]
    assert rag_pipeline_adapter(pipeline)("q").citations == ["a.pdf", "b.pdf"]


def test_deduplication_keeps_first_appearance_order(pipeline: _StubPipeline) -> None:
    """Order is rank order. The scorer ignores it today, but a rank-aware scorer is the documented
    next step for `citation_overlap` (TESTING.md), and it would read this list."""
    pipeline.docs = [_Doc(filename="b.pdf"), _Doc(filename="a.pdf"), _Doc(filename="b.pdf")]
    assert rag_pipeline_adapter(pipeline)("q").citations == ["b.pdf", "a.pdf"]


def test_a_chunk_with_no_filename_is_skipped_rather_than_cited_as_none(
    pipeline: _StubPipeline,
) -> None:
    """Figure chunks and other synthetic units can lack a filename; `None` in the citation list
    would compare against nothing and silently depress the retrieval score."""
    pipeline.docs = [_Doc(filename="a.pdf"), _Doc(page=3), _Doc(filename=None)]
    assert rag_pipeline_adapter(pipeline)("q").citations == ["a.pdf"]


def test_retrieving_nothing_yields_no_citations_and_still_returns_a_result(
    pipeline: _StubPipeline,
) -> None:
    pipeline.docs = []
    out = rag_pipeline_adapter(pipeline)("q")
    assert out.citations == []
    assert out.raw["n_retrieved"] == 0


# ============================================================
# rag_pipeline_adapter — the raw payload the store reads
# ============================================================


def test_the_raw_payload_carries_the_original_query(pipeline: _StubPipeline) -> None:
    assert rag_pipeline_adapter(pipeline)("my question").raw["query"] == "my question"


def test_every_retrieved_chunk_gets_a_descriptor_even_when_deduplicated(
    pipeline: _StubPipeline,
) -> None:
    """`retrieved` is per-*chunk*, `citations` is per-*document*. A retrieval-shape scorer needs
    the chunk count, so collapsing these two would hide duplicate hits from the same paper."""
    pipeline.docs = [_Doc(filename="a.pdf"), _Doc(filename="a.pdf")]
    out = rag_pipeline_adapter(pipeline)("q")
    assert len(out.raw["retrieved"]) == 2
    assert out.citations == ["a.pdf"]


def test_the_descriptor_exposes_chunk_kind_not_just_filename(pipeline: _StubPipeline) -> None:
    """The whole point of `retrieved` (Feature 4c): a figure-retrieval scorer must be able to see
    *what kind* of chunk came back."""
    pipeline.docs = [_Doc(filename="a.pdf", page=4, chunk_type="figure", figure_id="fig-1")]
    d = rag_pipeline_adapter(pipeline)("q").raw["retrieved"][0]
    assert d == {"filename": "a.pdf", "page": 4, "chunk_type": "figure", "figure_id": "fig-1"}


def test_descriptor_keys_are_present_even_when_the_metadata_lacks_them(
    pipeline: _StubPipeline,
) -> None:
    """A missing key must read as `None`, not raise — chunk metadata is not uniform across a
    library ingested over months."""
    pipeline.docs = [_Doc(filename="a.pdf")]
    d = rag_pipeline_adapter(pipeline)("q").raw["retrieved"][0]
    assert set(d) == {"filename", "page", "chunk_type", "figure_id"}
    assert d["chunk_type"] is None


def test_the_descriptors_stay_plain_dicts(pipeline: _StubPipeline) -> None:
    """Deliberate: no `doc_assistant` types cross into the generic harness, so Feature 5 can lift
    the harness out without dragging the library with it."""
    for d in rag_pipeline_adapter(pipeline)("q").raw["retrieved"]:
        assert type(d) is dict


# ============================================================
# rag_pipeline_adapter — token accounting
# ============================================================


def test_token_counts_come_back_on_the_result(pipeline: _StubPipeline) -> None:
    out = rag_pipeline_adapter(pipeline)("q")
    assert (out.token_input, out.token_output) == (11, 7)


def test_each_query_gets_a_fresh_counter(pipeline: _StubPipeline) -> None:
    """Per-case, not per-run. A leaked counter would make every row after the first overstate its
    cost while still looking like a plausible number."""
    call = rag_pipeline_adapter(pipeline)
    first = call("q1")
    second = call("q2")
    assert first.token_input == second.token_input == 11
    assert first.token_output == second.token_output == 7


def test_zero_tokens_are_reported_as_none_rather_than_zero(pipeline: _StubPipeline) -> None:
    """`or None` in the adapter: a provider that reports no usage is *unknown* cost, not free.
    Recording 0 would silently drag a run's mean spend down."""
    pipeline.tokens = (0, 0)
    out = rag_pipeline_adapter(pipeline)("q")
    assert out.token_input is None
    assert out.token_output is None


# ============================================================
# embedding_callable
# ============================================================


def test_the_embedder_is_the_pipeline_s_own(pipeline: _StubPipeline) -> None:
    """The reason this adapter exists: scoring must not load a second HuggingFace model, and an
    `embedding_similarity` score is only comparable within one embedder anyway (TESTING.md)."""
    embedding_callable(pipeline)("some text")
    assert pipeline.embeddings.calls == ["some text"]


def test_the_embedding_is_a_list_of_plain_floats(pipeline: _StubPipeline) -> None:
    """Numpy scalars survive arithmetic but not JSON, and the run store serialises what it is
    given — so the conversion is a storage contract, not a style preference."""
    pipeline.embeddings = _Embeddings([1, 2, 3])
    vector = embedding_callable(pipeline)("text")
    assert vector == [1.0, 2.0, 3.0]
    assert all(type(x) is float for x in vector)


def test_the_returned_callable_is_reusable(pipeline: _StubPipeline) -> None:
    embed = embedding_callable(pipeline)
    embed("one")
    embed("two")
    assert pipeline.embeddings.calls == ["one", "two"]
