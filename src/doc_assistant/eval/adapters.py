"""doc_assistant-specific adapters for the generic eval harness.

**This is the only module in ``doc_assistant.eval`` that depends on
the rest of ``doc_assistant``.** Extracting the harness into a
standalone repo (Feature 5) means dropping this file and writing a new
one in the consumer project.

The adapter wraps ``RAGPipeline`` so the generic ``Runner`` can call
it like any other system-under-test.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from doc_assistant.eval.results import EvalOutput
from doc_assistant.tracking import TokenCounter

if TYPE_CHECKING:
    from doc_assistant.pipeline import RAGPipeline


def rag_pipeline_adapter(pipeline: RAGPipeline) -> Callable[[str], EvalOutput]:
    """Return a callable ``(query: str) -> EvalOutput`` over ``pipeline``.

    Per-query lifecycle:

    1. Spin up a fresh ``TokenCounter`` so token cost is per-case, not
       per-run.
    2. Run retrieval, collect filenames of returned chunks as citations.
    3. Stream the answer to exhaustion, accumulating the text.
    4. Return ``EvalOutput`` with answer, citations, token counts, and
       a ``raw`` dict carrying the original query (the store uses it).
    """

    def _call(query: str) -> EvalOutput:
        counter = TokenCounter()
        docs = pipeline.retrieve(query)
        citations: list[str] = []
        seen: set[str] = set()
        for doc in docs:
            filename = doc.metadata.get("filename")
            if filename and filename not in seen:
                seen.add(filename)
                citations.append(str(filename))

        # Per-chunk descriptors so a retrieval-shape scorer (e.g. the Feature 4c
        # FigureRetrievalScorer) can see *what kind* of chunk came back, not just
        # the filenames. Kept generic — plain dicts, no doc_assistant types.
        retrieved = [
            {
                "filename": doc.metadata.get("filename"),
                "page": doc.metadata.get("page"),
                "chunk_type": doc.metadata.get("chunk_type"),
                "figure_id": doc.metadata.get("figure_id"),
            }
            for doc in docs
        ]

        chunks: list[str] = []
        for chunk in pipeline.stream_answer(query, docs, counter=counter):
            chunks.append(chunk)
        answer = "".join(chunks).strip()

        return EvalOutput(
            answer=answer,
            citations=citations,
            token_input=counter.input_tokens or None,
            token_output=counter.output_tokens or None,
            raw={"query": query, "n_retrieved": len(docs), "retrieved": retrieved},
        )

    return _call


def embedding_callable(pipeline: RAGPipeline) -> Callable[[str], list[float]]:
    """Adapt the pipeline's loaded embedder for ``EmbeddingSimilarityScorer``.

    Avoids loading a second HuggingFace model just for scoring. Returns
    a callable ``(text: str) -> list[float]``.
    """

    def _embed(text: str) -> list[float]:
        return [float(x) for x in pipeline.embeddings.embed_query(text)]

    return _embed
