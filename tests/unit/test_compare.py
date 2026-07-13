"""Unit tests for the A/B-compare pure diff + honesty note (feature-ab-compare-sandbox.md, U6).

``diff_sources`` and ``compare_note`` are pure — no retrieval, no LLM, no I/O. The impure
``ChatController.compare_retrieval`` (which runs retrieval twice) is covered by the API integration
test with a monkeypatched retriever.
"""

from __future__ import annotations

from doc_assistant.compare import CompareSource, build_result, compare_note, diff_sources


def _src(rank: int, identity: str, score: float = 1.0) -> CompareSource:
    return CompareSource(
        rank=rank,
        filename="f.pdf",
        page=None,
        section=None,
        score=score,
        excerpt="ex",
        citation=f"[{rank}] f.pdf",
        identity=identity,
    )


def test_diff_sources_classifies_and_ranks() -> None:
    # Identical lists -> every source in both, rank unchanged.
    rows = diff_sources([_src(1, "x"), _src(2, "y")], [_src(1, "x"), _src(2, "y")])
    assert [(r.status, r.rank_delta) for r in rows] == [("in_both", 0), ("in_both", 0)]

    # Reorder -> in_both with a rank delta (a_rank - b_rank).
    rows = diff_sources([_src(1, "x"), _src(2, "y")], [_src(1, "y"), _src(2, "x")])
    by_id = {r.identity: r for r in rows}
    assert by_id["x"].status == "in_both" and by_id["x"].rank_delta == -1  # a=1, b=2
    assert by_id["y"].status == "in_both" and by_id["y"].rank_delta == 1  # a=2, b=1

    # Disjoint -> all only_in_a / only_in_b.
    rows = diff_sources([_src(1, "x")], [_src(1, "z")])
    assert {r.status for r in rows} == {"only_in_a", "only_in_b"}
    assert all(r.rank_delta is None for r in rows)


def test_diff_sources_superset_is_depth_only() -> None:
    # B = A plus a deeper source (top_k depth): shared in_both + a only_in_b tail, rank-ordered.
    rows = diff_sources([_src(1, "x")], [_src(1, "x"), _src(2, "z")])
    assert [(r.identity, r.status) for r in rows] == [("x", "in_both"), ("z", "only_in_b")]


def test_compare_note_no_retrieval_change() -> None:
    eff = {"top_k": 5, "use_multi_query": True}
    note = compare_note(eff, dict(eff))
    assert "doesn't change retrieval" in note


def test_compare_note_top_k_depth_only() -> None:
    note = compare_note(
        {"top_k": 5, "use_multi_query": True}, {"top_k": 10, "use_multi_query": True}
    )
    assert "5 more source(s)" in note and "depth only" in note


def test_compare_note_multi_query_moves_membership_is_silent() -> None:
    # When use_multi_query differs, membership genuinely moves — no note; the diff speaks.
    note = compare_note(
        {"top_k": 5, "use_multi_query": False}, {"top_k": 5, "use_multi_query": True}
    )
    assert note == ""


def test_build_result_assembles() -> None:
    a = [_src(1, "x")]
    b = [_src(1, "x"), _src(2, "z")]
    result = build_result(
        "q", a, b, {"top_k": 5, "use_multi_query": True}, {"top_k": 6, "use_multi_query": True}
    )
    assert result.query == "q"
    assert len(result.rows) == 2
    assert "depth only" in result.note
