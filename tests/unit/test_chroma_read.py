"""Guard tests for `chroma_read.get_all` — the paged whole-collection read.

The bug these pin (2026-07-25): Chroma binds one SQL parameter per returned row, so an unpaged
`coll.get()` raises `too many SQL variables` once a collection passes SQLite's ceiling. It took the
corpus from 47 to 97 documents to hit it (33,163 parent-child chunks) and it broke
`RAGPipeline.__init__` — i.e. the answer path could not construct at all. These tests use a fake
collection that *counts pages*, so "it is actually paged" is asserted rather than assumed.
"""

from __future__ import annotations

import pytest

from doc_assistant.chroma_read import PAGE_SIZE, get_all


class FakeCollection:
    """A Chroma-shaped `get()` over an in-memory row list, recording the pages requested."""

    def __init__(self, n_rows: int, *, keys: tuple[str, ...] = ("metadatas",)) -> None:
        self.rows = [{"i": i} for i in range(n_rows)]
        self.keys = keys
        self.calls: list[tuple[int, int]] = []  # (limit, offset)
        self.max_rows_per_call = 0

    def get(self, *, limit=None, offset=0, where=None, include=None):
        rows = (
            self.rows if where is None else [r for r in self.rows if r["i"] in where["i"]["$in"]]
        )
        page = rows[offset : offset + limit] if limit is not None else rows
        self.calls.append((limit, offset))
        self.max_rows_per_call = max(self.max_rows_per_call, len(page))
        out = {"ids": [str(r["i"]) for r in page]}
        for key in include if include is not None else self.keys:
            out[key] = list(page)
        return out


def test_reads_every_row_in_bounded_pages():
    """More rows than one page: all of them come back, and no single call exceeds the page size."""
    coll = FakeCollection(PAGE_SIZE * 2 + 37)

    out = get_all(coll, include=["metadatas"])

    assert len(out["ids"]) == PAGE_SIZE * 2 + 37
    assert len(out["metadatas"]) == PAGE_SIZE * 2 + 37
    assert coll.max_rows_per_call <= PAGE_SIZE  # the whole point — bounded per statement
    assert len(coll.calls) == 3  # two full pages + the short one that ends the walk
    assert [c[1] for c in coll.calls] == [0, PAGE_SIZE, PAGE_SIZE * 2]  # offsets advance


def test_row_order_and_content_match_an_unpaged_read():
    """Paging must be invisible to the caller: same rows, same order, concatenated per key."""
    coll = FakeCollection(PAGE_SIZE + 5)

    out = get_all(coll, include=["metadatas"])

    assert [m["i"] for m in out["metadatas"]] == list(range(PAGE_SIZE + 5))


def test_short_collection_is_one_call():
    coll = FakeCollection(3)

    out = get_all(coll, include=["metadatas"])

    assert len(out["ids"]) == 3
    assert len(coll.calls) == 1


def test_empty_collection_returns_the_requested_keys_not_a_crash():
    """0 documents is a legitimate state (robustness contract) — the keys must still exist."""
    coll = FakeCollection(0)

    out = get_all(coll, include=["documents", "metadatas"])

    assert out["ids"] == []
    assert out["documents"] == []
    assert out["metadatas"] == []


def test_where_filter_is_passed_through_on_every_page():
    coll = FakeCollection(PAGE_SIZE + 10)
    ids = list(range(PAGE_SIZE + 10))

    out = get_all(coll, where={"i": {"$in": ids}}, include=["metadatas"])

    assert len(out["metadatas"]) == PAGE_SIZE + 10
    assert len(coll.calls) == 2


@pytest.mark.parametrize("page_size", [1, 7, 100])
def test_page_size_is_a_bound_not_a_behaviour_change(page_size):
    """Any page size returns the identical result — the number is structural, not tuned."""
    coll = FakeCollection(50)

    out = get_all(coll, include=["metadatas"], page_size=page_size)

    assert [m["i"] for m in out["metadatas"]] == list(range(50))
