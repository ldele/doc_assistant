"""Guard tests for the BM25-weight sweep scorer (scripts/sweep_bm25_weight.py).

Pure helpers only — no pipeline, models, or Chroma. Importing the module is cheap
(it sets HF offline env + imports config/eval; ``RAGPipeline`` is imported lazily
inside the sweep functions, not at module load).
"""

from __future__ import annotations

import pytest
from scripts.sweep_bm25_weight import _filenames, _parse_grid, _recall_at_k


class _Doc:
    def __init__(self, filename: str | None) -> None:
        self.metadata = {"filename": filename}


# ---- _filenames: ordered, de-duplicated, non-empty -------------------------


def test_filenames_dedup_preserving_order() -> None:
    docs = [_Doc("a.pdf"), _Doc("b.pdf"), _Doc("a.pdf"), _Doc("c.pdf")]
    assert _filenames(docs) == ["a.pdf", "b.pdf", "c.pdf"]


def test_filenames_drops_empty_and_missing() -> None:
    # A "" or missing filename must not survive: the bidirectional-substring matcher
    # would treat "" as a substring of every expected fragment and fabricate hits.
    docs = [_Doc("a.pdf"), _Doc(""), _Doc(None), _Doc("b.pdf")]
    assert _filenames(docs) == ["a.pdf", "b.pdf"]


def test_empty_filename_cannot_fabricate_a_hit_end_to_end() -> None:
    # Through the real path (_filenames → _recall_at_k), a stray blank filename
    # scores 0, not a false 1.0.
    files = _filenames([_Doc("")])
    assert _recall_at_k(["anything"], files, 5) == 0.0


# ---- _recall_at_k: bidirectional substring, truncated at k ------------------


def test_recall_bidirectional_substring() -> None:
    # cases.yaml contract: a fragment matches the full filename and vice versa.
    assert _recall_at_k(["hodgkin_huxley_1952"], ["hodgkin_huxley_1952.pdf"], 5) == 1.0
    assert _recall_at_k(["cajal_1911.pdf"], ["cajal_1911"], 5) == 1.0


def test_recall_truncates_at_k() -> None:
    files = ["x.pdf", "y.pdf", "z.pdf"]
    assert _recall_at_k(["z"], files, 2) == 0.0  # z sits at index 2, outside top-2
    assert _recall_at_k(["z"], files, 3) == 1.0


def test_recall_partial_and_full() -> None:
    assert _recall_at_k(["a", "b"], ["a.pdf"], 5) == 0.5
    assert _recall_at_k(["a", "b"], ["a.pdf", "b.pdf"], 5) == 1.0


def test_recall_none_when_no_expected_citations() -> None:
    assert _recall_at_k([], ["a.pdf"], 5) is None


# ---- _parse_grid -----------------------------------------------------------


def test_parse_grid_default() -> None:
    assert _parse_grid(None) == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def test_parse_grid_custom() -> None:
    assert _parse_grid("0.0, 0.5 ,1.0") == [0.0, 0.5, 1.0]


@pytest.mark.parametrize("bad", ["0.5,1.5", "-0.1,0.4", "2.0"])
def test_parse_grid_rejects_out_of_range(bad: str) -> None:
    with pytest.raises(ValueError, match="outside"):
        _parse_grid(bad)
