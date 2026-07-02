"""R6 fix 4 probe — every parent-child chunk carries ``parent_text``.

In parent-child retrieval mode ``pipeline.retrieve_with_scores`` only returns candidates
whose metadata carries ``parent_text`` (it becomes the LLM context). A PC chunk written
without it would be silently unreturnable. This guards the invariant at its source — the
deterministic chunker — with no Chroma / embeddings.
"""

from __future__ import annotations

from doc_assistant.ingest import build_parent_child_chunks

_MD = """<!-- page:1 -->
# Introduction
Retrieval-augmented generation grounds a language model in retrieved passages.

<!-- page:2 -->
## Method
We combine BM25 with a dense retriever and rerank with a cross-encoder. The ensemble
fuses sparse and dense candidates before the final reranking step.

<!-- page:3 -->
## Results
The hybrid system outperforms either retriever alone on the benchmark.
"""


def test_every_parent_child_chunk_carries_parent_text() -> None:
    chunks = build_parent_child_chunks(_MD, {"filename": "paper.pdf"})
    assert chunks, "chunker produced no chunks"
    missing = [c for c in chunks if not c.metadata.get("parent_text")]
    assert not missing, f"{len(missing)}/{len(chunks)} PC chunks lack parent_text (unreturnable)"
