"""Guard tests for the deduplicated parent-text map (KI-32 step 1).

**What the change is.** Chroma denormalises a parent block's full text into every one of its
children (5.5 on the live corpus), so the in-RAM BM25 corpus held that many copies of every parent:
a measured 80 MB of its 265 MB (`tests/eval/baselines/memory_and_lazy_reranker_2026-07-30.md`).
`_split_parent_texts` lifts the text out into one entry per parent and `_parent_text_for`
re-attaches it for the <= TOP_K parents a turn actually returns.

**What these tests have to pin.** The saving is worthless if it changes an answer, and the failure
mode is silent: a candidate whose parent text cannot be resolved is *skipped* by the parent-child
branch, so a broken lookup shows up as slightly worse retrieval, never as an error. So:

1. the split actually deduplicates (the memory win, which a later "simplification" could undo);
2. retrieval output is **identical**, document for document and score for score, to the pre-split
   form (the equivalence half);
3. a BM25-arm candidate with no `parent_text` in its metadata still expands (the regression this
   change could have introduced), and the map is *load-bearing* for it: emptied, that candidate
   disappears. A test that passes with the map removed would be pinning nothing.

Model-free: a bare pipeline (``__new__``) with a fake ensemble and reranker, like its sibling
retrieval tests.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from doc_assistant.pipeline import RAGPipeline, _split_parent_texts

_PARENT_A = "Parent A. " + "Dense retrieval encodes questions and passages jointly. " * 4
_PARENT_B = "Parent B. " + "A cross-encoder reranks the candidate passages. " * 4


def _child(
    text: str, *, doc_hash: str, parent_index: int, child_index: int, parent_text: str | None
) -> Document:
    """A parent-child chunk shaped like Chroma's, i.e. carrying its parent's full text."""
    meta: dict[str, object] = {
        "doc_hash": doc_hash,
        "filename": f"{doc_hash}.pdf",
        "parent_index": parent_index,
        "child_index": child_index,
        "page": 1,
    }
    if parent_text is not None:
        meta["parent_text"] = parent_text
    return Document(page_content=text, metadata=meta)


def _corpus() -> list[Document]:
    """Two parents: three children under A, two under B. Five chunks, two distinct parents."""
    return [
        _child(
            "dense retrieval one",
            doc_hash="a",
            parent_index=0,
            child_index=0,
            parent_text=_PARENT_A,
        ),
        _child(
            "dense retrieval two",
            doc_hash="a",
            parent_index=0,
            child_index=1,
            parent_text=_PARENT_A,
        ),
        _child(
            "dense retrieval three",
            doc_hash="a",
            parent_index=0,
            child_index=2,
            parent_text=_PARENT_A,
        ),
        _child(
            "cross-encoder rerank one",
            doc_hash="b",
            parent_index=0,
            child_index=0,
            parent_text=_PARENT_B,
        ),
        _child(
            "cross-encoder rerank two",
            doc_hash="b",
            parent_index=0,
            child_index=1,
            parent_text=_PARENT_B,
        ),
    ]


class _FakeReranker:
    def predict(self, pairs: list) -> list[float]:
        # Descending and deterministic, so ranking never depends on candidate count.
        return [1.0 - 0.01 * i for i in range(len(pairs))]


def _rig(monkeypatch: pytest.MonkeyPatch, candidates: list[Document], parents) -> RAGPipeline:
    monkeypatch.setattr("doc_assistant.pipeline.USE_PARENT_CHILD", True)
    monkeypatch.setattr("doc_assistant.pipeline.USE_MULTI_QUERY", False)
    rag = RAGPipeline.__new__(RAGPipeline)
    # The in-RAM map is what this file pins (KI-32 step 1); `None` selects that arm over the
    # on-disk index, which resolves parents from its own table (test_pipeline_sparse_arm.py).
    rag._sparse = None
    rag.ensemble = RunnableLambda(lambda _q: list(candidates))
    rag.reranker = _FakeReranker()
    rag._parent_texts = parents
    return rag


# ============================================================
# The split: fewer copies, same information
# ============================================================


class TestSplit:
    def test_five_chunks_two_parents_becomes_two_entries(self):
        docs = _corpus()

        parents = _split_parent_texts(docs)

        assert parents == {("a", 0): _PARENT_A, ("b", 0): _PARENT_B}

    def test_the_text_is_removed_from_every_chunk_metadata(self):
        """The memory win itself. If this ever passes with `parent_text` still present, the
        duplicates are back and the 3x is gone."""
        docs = _corpus()

        _split_parent_texts(docs)

        assert all("parent_text" not in d.metadata for d in docs)
        # Nothing else about the chunk moved.
        assert [d.page_content for d in docs] == [
            "dense retrieval one",
            "dense retrieval two",
            "dense retrieval three",
            "cross-encoder rerank one",
            "cross-encoder rerank two",
        ]
        assert docs[0].metadata == {
            "doc_hash": "a",
            "filename": "a.pdf",
            "parent_index": 0,
            "child_index": 0,
            "page": 1,
        }

    def test_flat_chunks_are_untouched_and_yield_an_empty_map(self):
        """Baseline (non-PC) chunks have no parent text; the map is legitimately empty."""
        meta = {"doc_hash": "a", "filename": "a.pdf"}
        docs = [Document(page_content="flat chunk", metadata=meta)]

        assert _split_parent_texts(docs) == {}
        assert docs[0].metadata == {"doc_hash": "a", "filename": "a.pdf"}

    def test_a_chunk_with_no_identity_keeps_its_text_rather_than_losing_it(self):
        """No `doc_hash`/`parent_index` means nothing to key on. Keep the text where it is —
        `_parent_text_for` reads metadata first, so such a chunk still expands."""
        orphan = Document(page_content="child", metadata={"parent_text": _PARENT_A})

        assert _split_parent_texts([orphan]) == {}
        assert orphan.metadata["parent_text"] == _PARENT_A

    def test_the_map_is_empty_for_an_empty_corpus(self):
        """Robustness contract: 0 documents is a supported state."""
        assert _split_parent_texts([]) == {}


# ============================================================
# The lookup: metadata first, map second
# ============================================================


class TestLookup:
    def test_metadata_wins_over_the_map(self, monkeypatch):
        """The vector arm returns Chroma documents that still carry the text, and a document
        ingested *after* construction can only be expanded that way — the map is a snapshot."""
        rag = _rig(monkeypatch, [], {("a", 0): "the map's version"})
        doc = _child("child", doc_hash="a", parent_index=0, child_index=0, parent_text="its own")

        assert rag._parent_text_for(doc) == "its own"

    def test_the_map_answers_for_a_stripped_chunk(self, monkeypatch):
        rag = _rig(monkeypatch, [], {("a", 0): _PARENT_A})
        doc = _child("child", doc_hash="a", parent_index=0, child_index=0, parent_text=None)

        assert rag._parent_text_for(doc) == _PARENT_A

    def test_none_when_neither_has_it(self, monkeypatch):
        rag = _rig(monkeypatch, [], {})
        doc = _child("child", doc_hash="a", parent_index=0, child_index=0, parent_text=None)

        assert rag._parent_text_for(doc) is None

    def test_a_string_parent_index_still_resolves(self, monkeypatch):
        """Metadata types are not guaranteed across a store round-trip; the key is coerced."""
        rag = _rig(monkeypatch, [], {("a", 3): _PARENT_A})
        doc = Document(page_content="child", metadata={"doc_hash": "a", "parent_index": "3"})

        assert rag._parent_text_for(doc) == _PARENT_A


# ============================================================
# Equivalence + the regression, end to end through retrieve_with_scores
# ============================================================


class TestRetrievalIsUnchanged:
    def test_output_is_identical_before_and_after_the_split(self, monkeypatch):
        """The whole justification: same documents, same metadata, same scores, less memory."""
        before = _rig(monkeypatch, _corpus(), {})  # metadata still carries parent_text

        split_docs = _corpus()
        parents = _split_parent_texts(split_docs)
        after = _rig(monkeypatch, split_docs, parents)

        out_before = before.retrieve_with_scores("retrieval", top_k=10)
        out_after = after.retrieve_with_scores("retrieval", top_k=10)

        assert [(d.page_content, d.metadata, s) for d, s in out_before] == [
            (d.page_content, d.metadata, s) for d, s in out_after
        ]

    def test_a_bm25_arm_candidate_still_expands_to_its_parent(self, monkeypatch):
        """The regression this change could have introduced: `_bm25_docs` no longer carries the
        text, so without the map every keyword-only hit would silently vanish from the answer."""
        docs = _corpus()
        parents = _split_parent_texts(docs)
        rag = _rig(monkeypatch, docs, parents)

        out = rag.retrieve_with_scores("retrieval", top_k=10)

        # Two parents, deduped from five children, expanded to the parent text.
        assert [d.page_content for d, _ in out] == [_PARENT_A, _PARENT_B]
        # The returned parent never carries the text twice (unchanged contract).
        assert all("parent_text" not in d.metadata for d, _ in out)
        # The score is the winning *child*'s score.
        assert out[0][1] == pytest.approx(1.0)

    def test_without_the_map_those_candidates_disappear(self, monkeypatch):
        """Non-vacuousness check for the test above: if the map were not consulted, this is what
        the user would get — a quietly shorter answer, no error anywhere."""
        docs = _corpus()
        _split_parent_texts(docs)
        rag = _rig(monkeypatch, docs, {})

        assert rag.retrieve_with_scores("retrieval", top_k=10) == []

    def test_dedup_still_collapses_siblings_of_one_parent(self, monkeypatch):
        docs = _corpus()
        parents = _split_parent_texts(docs)
        rag = _rig(monkeypatch, docs, parents)

        out = rag.retrieve_with_scores("retrieval", top_k=1)

        assert len(out) == 1
        assert out[0][0].page_content == _PARENT_A

    def test_a_folder_scoped_turn_also_resolves_its_parents(self, monkeypatch):
        """ADR-025 F2 is *why* the corpus is resident at all, so it is where a lookup regression
        would hide: the scoped arm is rebuilt from `_bm25_docs`, which no longer holds the text."""
        docs = _corpus()
        parents = _split_parent_texts(docs)
        rag = _rig(monkeypatch, docs, parents)
        rag._bm25_docs = docs
        rag._scoped = OrderedDict()
        rag._weights = [0.4, 0.6]

        class _Db:
            def as_retriever(self, *, search_kwargs):
                return RunnableLambda(lambda _q: [])  # vector arm contributes nothing here

        rag.db = _Db()

        out = rag.retrieve_with_scores("retrieval", top_k=10, scope=frozenset({"a"}))

        # Only document "a" is in scope, and its parent text came from the map.
        assert [d.page_content for d, _ in out] == [_PARENT_A]
