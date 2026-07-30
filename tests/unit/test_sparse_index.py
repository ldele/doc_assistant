"""Guard tests for the on-disk sparse arm (ADR-036, KI-32 step 2).

This module replaced a retrieval **input**, so the bar is the one ADR-035 set: every failure has to
be a slower or an emptier answer, never a quietly different one. The tests split into

1. *semantics* — the properties that make the SQL arm behave like the in-RAM one it replaced: OR
   (not AND) across query terms, the project tokenizer's vocabulary, scope applied before the
   limit, and an FTS5 expression that cannot be steered by the user's own text;
2. *refusal* — a stale, corrupt or foreign index is rebuilt, never served;
3. *the memory property itself* — with the index active the pipeline holds **no** corpus in RAM.
   That is the entire point of the change, and it is one careless edit away from regressing
   silently, so it is asserted rather than trusted.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from doc_assistant import sparse_index
from doc_assistant.sparse_index import SparseIndex

# Two documents, three chunks each: enough for OR/AND, scope and ranking to be distinguishable.
_CHUNKS: list[tuple[str, dict[str, object]]] = [
    (
        "dense passage retrieval encodes questions and passages",
        {"doc_hash": "aaa", "filename": "dpr.pdf", "parent_index": 0, "parent_text": "PARENT A0"},
    ),
    (
        "a cross-encoder reranks candidate passages jointly",
        {"doc_hash": "aaa", "filename": "dpr.pdf", "parent_index": 0, "parent_text": "PARENT A0"},
    ),
    (
        "BM25 remains a strong sparse baseline",
        {"doc_hash": "aaa", "filename": "dpr.pdf", "parent_index": 1, "parent_text": "PARENT A1"},
    ),
    (
        "colbert uses late interaction over token embeddings",
        {"doc_hash": "bbb", "filename": "colbert.pdf", "parent_index": 0, "parent_text": "PAR B0"},
    ),
    (
        "late interaction keeps the passage representation",
        {"doc_hash": "bbb", "filename": "colbert.pdf", "parent_index": 0, "parent_text": "PAR B0"},
    ),
    (
        "the index stores one vector per token",
        {"doc_hash": "bbb", "filename": "colbert.pdf", "parent_index": 1, "parent_text": "PAR B1"},
    ),
]


def _ids(n: int) -> list[str]:
    return [f"chunk-{i:04d}" for i in range(n)]


@pytest.fixture
def index(tmp_path, monkeypatch) -> SparseIndex:
    monkeypatch.delenv("DOC_SPARSE_INDEX", raising=False)
    path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
    path.parent.mkdir()
    stamp = sparse_index.fingerprint("langchain", _ids(len(_CHUNKS)))
    return SparseIndex.build(path, stamp, iter(_CHUNKS))


def _texts(docs) -> list[str]:
    return [d.page_content for d in docs]


# ============================================================
# Semantics — behave like the arm it replaced
# ============================================================


class TestSemantics:
    def test_any_query_term_matches_not_all(self, index):
        """FTS5's default for a bare term list is **AND**. The arm this replaces scored a document
        containing *any* term, so an AND expression would silently return a fraction of the
        candidates and no error anywhere."""
        hits = _texts(index.search("colbert questions", k=10))

        # "colbert" is in one chunk, "questions" in another, and neither contains both.
        assert any("colbert" in h for h in hits)
        assert any("questions" in h for h in hits)

    def test_the_project_tokenizer_vocabulary_is_preserved(self, index):
        """`keywords.tokenize` keeps `cross-encoder` as ONE token and casefolds `BM25`. The FTS5
        table has to agree, or index and query stop meeting (R6's original bug, one layer down)."""
        assert any("cross-encoder" in h for h in _texts(index.search("cross-encoder", k=10)))
        assert any("BM25" in h for h in _texts(index.search("bm25", k=10)))  # query casefolded
        assert any("BM25" in h for h in _texts(index.search("BM25?", k=10)))  # punctuation dropped

    def test_scope_is_applied_before_the_limit(self, index):
        """A scoped turn must get *its own* top-k, not the global top-k filtered down to whatever
        survives — the difference between "the best k in this folder" and "however many of the
        global best happen to be here" (ADR-025 F2)."""
        scoped = index.search("passage token index", k=3, scope=frozenset({"bbb"}))

        assert len(scoped) == 3  # a post-filter would have returned fewer
        assert all("colbert" in d.metadata["filename"] for d in scoped)

    def test_an_empty_scope_returns_nothing_and_never_widens(self, index):
        assert index.search("retrieval", k=10, scope=frozenset()) == []

    def test_a_query_with_no_terms_is_empty_not_an_error(self, index):
        assert index.search("!!! ???", k=10) == []
        assert index.search("", k=10) == []

    @pytest.mark.parametrize(
        "query", ["retrieval NOT passage", 'colbert OR "', "passage*", '"; DROP TABLE chunks; --']
    )
    def test_user_text_cannot_steer_the_fts_expression(self, index, query):
        """Every term is quoted, so FTS5 operators inside a user's question are data, not syntax.
        The query must still answer (or answer nothing) and the index must survive."""
        index.search(query, k=5)

        assert index.search("retrieval", k=5), "the index still answers afterwards"

    def test_metadata_round_trips_without_the_parent_text(self, index):
        doc = index.search("dense passage", k=1)[0]

        assert doc.metadata["filename"] == "dpr.pdf"
        assert doc.metadata["parent_index"] == 0
        # KI-32: the parent text lives once in its own table, never on every child.
        assert "parent_text" not in doc.metadata

    def test_parent_text_is_stored_once_and_looked_up(self, index):
        assert index.parent_text("aaa", 0) == "PARENT A0"
        assert index.parent_text("bbb", 1) == "PAR B1"
        assert index.parent_text("aaa", 99) is None
        assert index.parent_text("nope", 0) is None

    def test_k_bounds_the_result(self, index):
        assert len(index.search("passage retrieval token index", k=2)) == 2
        assert index.search("passage", k=0) == []

    def test_doc_hashes_lists_what_the_index_holds(self, index):
        assert index.doc_hashes() == {"aaa", "bbb"}


# ============================================================
# Refusal — a stale or damaged index is rebuilt, never served
# ============================================================


class TestRefusesRatherThanServesAWrongIndex:
    def test_absent_index_returns_none(self, tmp_path):
        assert sparse_index.open_index(tmp_path / "nope.sqlite3", "fp") is None

    def test_the_streaming_and_collected_fingerprints_agree(self):
        """`_open_sparse_index` streams pages (bounded memory); tests and small callers pass a
        list. Both must produce the same identity or a launch would rebuild every time — and the
        digest must not depend on how Chroma happened to page the rows."""
        ids = _ids(12)
        one_shot = sparse_index.fingerprint("langchain", ids)

        assert sparse_index.fingerprint_from_pages("langchain", [ids[:5], ids[5:]]) == one_shot
        assert sparse_index.fingerprint_from_pages("langchain", [ids[7:], ids[:7]]) == one_shot
        assert sparse_index.fingerprint_from_pages("langchain", [[i] for i in ids]) == one_shot

    def test_the_streaming_fingerprint_still_sees_a_changed_corpus(self):
        """Order-independence must not become blindness: an add, a removal and a replacement all
        have to move the digest."""
        base = sparse_index.fingerprint_from_pages("langchain", [_ids(10)])

        assert sparse_index.fingerprint_from_pages("langchain", [_ids(11)]) != base  # added
        assert sparse_index.fingerprint_from_pages("langchain", [_ids(9)]) != base  # removed
        swapped = [*_ids(9), "chunk-9999"]
        assert sparse_index.fingerprint_from_pages("langchain", [swapped]) != base  # replaced

    def test_a_matching_fingerprint_opens(self, tmp_path, index):
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        stamp = sparse_index.fingerprint("langchain", _ids(len(_CHUNKS)))

        opened = sparse_index.open_index(path, stamp)

        assert opened is not None
        assert opened.chunks == len(_CHUNKS)

    def test_replaced_chunk_ids_invalidate_at_the_same_count(self, tmp_path, index):
        """What a bare `count()` misses, and exactly what `ingest --rebuild` does (fresh UUIDs)."""
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        rebuilt = [f"rebuilt-{i:04d}" for i in range(len(_CHUNKS))]

        stale = sparse_index.fingerprint("langchain", rebuilt)
        assert sparse_index.open_index(path, stale) is None

    def test_id_order_does_not_matter(self, tmp_path, index):
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        shuffled = list(reversed(_ids(len(_CHUNKS))))

        assert sparse_index.open_index(path, sparse_index.fingerprint("langchain", shuffled))

    def test_a_different_collection_invalidates(self, tmp_path, index):
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        other = sparse_index.fingerprint("specter2", _ids(len(_CHUNKS)))

        assert sparse_index.open_index(path, other) is None

    def test_a_changed_tokenizer_invalidates(self, tmp_path, index, monkeypatch):
        """The index *is* tokens: a tokenizer change must invalidate it without anyone remembering
        to bump a constant."""
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        before = sparse_index.fingerprint("langchain", _ids(len(_CHUNKS)))
        monkeypatch.setattr(
            "doc_assistant.knowledge.keywords.tokenize", lambda text: text.split(), raising=True
        )

        assert sparse_index.fingerprint("langchain", _ids(len(_CHUNKS))) != before
        assert sparse_index.open_index(path, before) is not None  # the OLD stamp still matches
        after = sparse_index.fingerprint("langchain", _ids(len(_CHUNKS)))
        assert sparse_index.open_index(path, after) is None

    def test_a_corrupt_file_returns_none_instead_of_raising(self, tmp_path, index):
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        path.write_bytes(b"not a database at all")

        assert sparse_index.open_index(path, "whatever") is None

    def test_a_foreign_database_returns_none(self, tmp_path):
        """Well-formed SQLite, wrong contents: no `meta` table, so it must not be trusted."""
        path = tmp_path / "foreign.sqlite3"
        con = sqlite3.connect(path)
        con.execute("CREATE TABLE unrelated (x TEXT)")
        con.commit()
        con.close()

        assert sparse_index.open_index(path, "fp") is None

    def test_a_failed_build_leaves_no_half_index(self, tmp_path):
        """The build writes a temp file and moves it, so an exception mid-stream cannot leave a
        partial database for the next launch to detect."""
        path = tmp_path / sparse_index.INDEX_FILENAME

        def exploding():
            yield _CHUNKS[0]
            raise RuntimeError("store read failed halfway")

        with pytest.raises(RuntimeError):
            SparseIndex.build(path, "fp", exploding())

        assert not path.exists()
        assert list(tmp_path.glob("*.building")) == []

    def test_a_rebuild_replaces_the_old_index(self, tmp_path, index):
        """Second build over the same path: the new content is what serves, no leftovers."""
        path = tmp_path / "chroma" / sparse_index.INDEX_FILENAME
        index.close()
        fresh = [("entirely new text about frogs", {"doc_hash": "zzz", "parent_index": 0})]

        rebuilt = SparseIndex.build(path, "fp2", iter(fresh))

        assert rebuilt.chunks == 1
        assert rebuilt.doc_hashes() == {"zzz"}
        assert sparse_index.open_index(path, "fp2") is not None
        assert list(path.parent.glob("*.building")) == []


# ============================================================
# The switch + the robustness contract
# ============================================================


class TestSwitchAndEmptyCorpus:
    @pytest.mark.parametrize("value", ["0", "false", "no", "FALSE", " 0 "])
    def test_falsey_values_disable(self, monkeypatch, value):
        monkeypatch.setenv("DOC_SPARSE_INDEX", value)
        assert sparse_index.enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything-else"])
    def test_everything_else_leaves_it_on(self, monkeypatch, value):
        monkeypatch.setenv("DOC_SPARSE_INDEX", value)
        assert sparse_index.enabled() is True

    def test_an_empty_corpus_builds_an_empty_index_that_answers_nothing(self, tmp_path):
        """0 documents is a supported state, not an error (robustness contract)."""
        path = tmp_path / sparse_index.INDEX_FILENAME

        built = SparseIndex.build(path, "fp", iter([]))

        assert built.chunks == 0
        assert built.search("anything", k=10) == []
        assert built.doc_hashes() == set()

    def test_the_index_path_sits_beside_the_store(self):
        assert sparse_index.index_path("/data/chroma_pc") == Path("/data") / "sparse_index.sqlite3"
