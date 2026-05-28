"""Tests for the doc-level vector enrichment module (Phase 4 close-out).

Pure-logic tests over the numpy core. Chroma/SQLite adapters are not
exercised here — they're covered by the CLI runner end-to-end against a
real store.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from doc_assistant.doc_vectors import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    SimilarityEdge,
    compute_doc_vectors,
    compute_similarity_edges,
    mean_pool,
)

# ============================================================
# mean_pool
# ============================================================


def test_mean_pool_basic_normalized():
    vecs = [np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])]
    out = mean_pool(vecs)
    assert out.shape == (3,)
    assert math.isclose(float(np.linalg.norm(out)), 1.0, abs_tol=1e-6)
    assert math.isclose(float(out[0]), 1.0, abs_tol=1e-6)


def test_mean_pool_renormalises_after_mean():
    vecs = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
    out = mean_pool(vecs)
    assert math.isclose(float(np.linalg.norm(out)), 1.0, abs_tol=1e-6)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    assert math.isclose(float(out[0]), inv_sqrt2, abs_tol=1e-6)
    assert math.isclose(float(out[1]), inv_sqrt2, abs_tol=1e-6)


def test_mean_pool_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        mean_pool([])


def test_mean_pool_rejects_1d_input():
    with pytest.raises(ValueError, match="2-D"):
        mean_pool(np.array([1.0, 0.0, 0.0]))


def test_mean_pool_handles_degenerate_zero_mean():
    vecs = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
    out = mean_pool(vecs)
    assert math.isclose(float(np.linalg.norm(out)), 0.0, abs_tol=1e-6)


# ============================================================
# compute_doc_vectors
# ============================================================


def test_compute_doc_vectors_skips_empty_entries():
    chunks = {
        "doc-a": [np.array([1.0, 0.0])],
        "doc-b": [],
        "doc-c": [np.array([0.0, 1.0]), np.array([0.0, 1.0])],
    }
    out = compute_doc_vectors(chunks)
    assert set(out.keys()) == {"doc-a", "doc-c"}
    assert math.isclose(float(np.linalg.norm(out["doc-a"])), 1.0, abs_tol=1e-6)
    assert math.isclose(float(np.linalg.norm(out["doc-c"])), 1.0, abs_tol=1e-6)


# ============================================================
# compute_similarity_edges
# ============================================================


def test_similarity_edges_empty_input():
    assert compute_similarity_edges({}) == []


def test_similarity_edges_single_doc_returns_nothing():
    vecs = {"only": np.array([1.0, 0.0], dtype=np.float32)}
    assert compute_similarity_edges(vecs) == []


def test_similarity_edges_identical_vectors_score_1():
    vecs = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
    }
    edges = compute_similarity_edges(vecs, threshold=0.5)
    assert len(edges) == 2  # a->b and b->a
    for e in edges:
        assert math.isclose(e.score, 1.0, abs_tol=1e-6)
        assert e.source_doc_id != e.target_doc_id


def test_similarity_edges_orthogonal_filtered_by_threshold():
    vecs = {
        "x": np.array([1.0, 0.0], dtype=np.float32),
        "y": np.array([0.0, 1.0], dtype=np.float32),
    }
    edges = compute_similarity_edges(vecs, threshold=0.5)
    assert edges == []


def test_similarity_edges_no_self_links():
    vecs = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
        "c": np.array([0.9, 0.1], dtype=np.float32) / np.linalg.norm([0.9, 0.1]),
    }
    edges = compute_similarity_edges(vecs, threshold=0.0)
    for e in edges:
        assert e.source_doc_id != e.target_doc_id


def test_similarity_edges_top_k_trims():
    vecs = {f"d{i}": np.array([1.0, 0.0], dtype=np.float32) for i in range(5)}
    edges = compute_similarity_edges(vecs, top_k=2, threshold=0.0)
    by_source: dict[str, list[SimilarityEdge]] = {}
    for e in edges:
        by_source.setdefault(e.source_doc_id, []).append(e)
    for _src, src_edges in by_source.items():
        assert len(src_edges) <= 2


def test_similarity_edges_sorted_desc_per_source():
    vecs = {
        "anchor": np.array([1.0, 0.0], dtype=np.float32),
        "near": np.array([0.95, 0.05], dtype=np.float32) / np.linalg.norm([0.95, 0.05]),
        "far": np.array([0.6, 0.4], dtype=np.float32) / np.linalg.norm([0.6, 0.4]),
    }
    edges = compute_similarity_edges(vecs, top_k=5, threshold=0.0)
    anchor_edges = [e for e in edges if e.source_doc_id == "anchor"]
    scores = [e.score for e in anchor_edges]
    assert scores == sorted(scores, reverse=True)


def test_similarity_edges_respects_custom_threshold():
    vecs = {
        "a": np.array([1.0, 0.0], dtype=np.float32),
        "b": np.array([1.0, 0.0], dtype=np.float32),
        "c": np.array([0.5, 0.5], dtype=np.float32) / np.linalg.norm([0.5, 0.5]),
    }
    # cos(a,c) ~= 0.707 — keep at threshold 0.7, drop at 0.9
    keep_edges = compute_similarity_edges(vecs, threshold=0.7)
    drop_edges = compute_similarity_edges(vecs, threshold=0.9)
    keep_pairs = {(e.source_doc_id, e.target_doc_id) for e in keep_edges}
    drop_pairs = {(e.source_doc_id, e.target_doc_id) for e in drop_edges}
    assert ("a", "c") in keep_pairs
    assert ("a", "c") not in drop_pairs


def test_default_constants_are_sane():
    assert DEFAULT_TOP_K >= 1
    assert 0.0 <= DEFAULT_THRESHOLD <= 1.0
