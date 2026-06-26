"""Feature 4a CI gate: a spliced Marker table survives ingest chunking and is findable.

Deterministic — no Marker subprocess, no Chroma/embeddings. Exercises the real
cross-module path: ``tables_marker`` splice → ``ingest`` parent-child chunking. Proves
the spliced high-fidelity table reaches a retrievable chunk (the splice→re-ingest→
retrieval mechanism). Live end-to-end quality on real PDFs is the opt-in
``tests/eval/cases.tables.yaml`` (run after a real Marker pass on the RTX box).
"""

from __future__ import annotations

from doc_assistant.ingest import build_parent_child_chunks
from doc_assistant.ingest.tables_marker import parse_marker_tables, splice_tables_inline

_CACHE_MD = """<!-- page:1 -->
# Intro
Background on page one.

<!-- page:2 -->
## Results
Table 1 reports accuracy.

| Model | Acc |
| --- | --- |
| BM25 | 42 |
| DPR | 79 |

<!-- page:3 -->
## Conclusion
No table here.
"""

# Marker's faithful render of page 2 (a value the lossy pymupdf table never had).
_MARKER_MD = """## Results

| Model | Top-20 | Top-100 |
| --- | --- | --- |
| DPR | 78.4 | 85.4 |
"""


def _haystack(chunks: list) -> str:
    return "\n".join(c.page_content + " " + str(c.metadata.get("parent_text", "")) for c in chunks)


def test_spliced_marker_table_reaches_a_chunk() -> None:
    spliced = splice_tables_inline(_CACHE_MD, parse_marker_tables(_MARKER_MD, [2]))
    chunks = build_parent_child_chunks(spliced, {"filename": "dpr.pdf"})
    hay = _haystack(chunks)
    # The Marker table's distinctive value reached the retrievable chunk stream...
    assert "85.4" in hay
    assert "Top-100" in hay
    # ...and the lossy pymupdf twin was de-duped out of it.
    assert "| BM25 | 42 |" not in hay


def test_pages_marker_skipped_keep_their_content() -> None:
    spliced = splice_tables_inline(_CACHE_MD, parse_marker_tables(_MARKER_MD, [2]))
    chunks = build_parent_child_chunks(spliced, {"filename": "dpr.pdf"})
    hay = _haystack(chunks)
    assert "Background on page one" in hay  # p1 untouched
    assert "No table here" in hay  # p3 untouched


# Regression for the DPR Table-2 retrieval gap (#4a, 2026-06-06): a wide table whose
# caption is far above the grid in the source. The caption is the query magnet; the
# grid holds the values. They MUST end up in the same parent, else retrieval surfaces
# the caption parent and the values never reach the LLM.
_CACHE_MD_WIDE = """<!-- page:1 -->
# Intro

<!-- page:5 -->
Table 2: Top-20 & Top-100 retrieval accuracy on test sets, measured as the percentage
of top 20/100 retrieved passages that contain the answer.

| Model | Acc |
| --- | --- |
| BM25 | 42 |

Several sentences of intervening prose about training schemes and run-time efficiency
that would otherwise separate the caption from the grid in the page span.

## 5.2 Ablation Study on Model Training
To understand further how different model training options affect the results.

<!-- page:6 -->
## Conclusion
"""

_MARKER_MD_WIDE = """| Training | Top-20 | Top-100 |
| --- | --- | --- |
| DPR | 78.4 | 85.4 |
"""


def test_caption_and_values_land_in_one_parent() -> None:
    spliced = splice_tables_inline(_CACHE_MD_WIDE, parse_marker_tables(_MARKER_MD_WIDE, [5]))
    chunks = build_parent_child_chunks(spliced, {"filename": "dpr.pdf"})
    # Some parent holds BOTH the caption phrase (the magnet) and the grid value (85.4).
    parents = {c.metadata["parent_index"]: c.metadata["parent_text"] for c in chunks}
    co_located = [
        pt
        for pt in parents.values()
        if "retrieval accuracy" in pt and "85.4" in pt and "Top-100" in pt
    ]
    assert co_located, "caption and grid value must share a parent"
