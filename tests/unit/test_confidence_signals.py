"""Tests for confidence signals (PR 5.1).

Pure-function tests against synthetic AnswerProvenance objects.
Thresholds match the constants in provenance.py:
  WEAK_RETRIEVAL_THRESHOLD = 0.3
  SCORE_CLUSTER_SPAN = 0.05
  SCORE_CLUSTER_MAX = 0.7
  SINGLE_SOURCE_MAX_DOCS = 2
"""

from __future__ import annotations

from doc_assistant.provenance import (
    AnswerProvenance,
    RetrievedChunk,
    compute_confidence_signals,
)


def _prov(chunks: list[RetrievedChunk]) -> AnswerProvenance:
    return AnswerProvenance(id="x", query="q", answer="a", retrieved_chunks=chunks)


def _chunk(score: float | None, filename: str = "paper.pdf") -> RetrievedChunk:
    return RetrievedChunk(filename=filename, reranker_score=score)


# ============================================================
# No-flag baselines
# ============================================================


def test_no_chunks_returns_all_false():
    sig = compute_confidence_signals(_prov([]))
    assert not sig.any()
    assert sig.reasons == []


def test_high_confidence_diverse_sources_no_flags():
    chunks = [
        _chunk(0.92, "a.pdf"),
        _chunk(0.88, "b.pdf"),
        _chunk(0.85, "c.pdf"),
        _chunk(0.82, "d.pdf"),
        _chunk(0.79, "e.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert not sig.any()
    assert sig.max_score == 0.92
    assert sig.unique_sources == 5


def test_high_confidence_clustered_at_top_is_not_a_concern():
    """High-and-clustered = consensus, not ambiguity. Should not fire."""
    chunks = [
        _chunk(0.90, "a.pdf"),
        _chunk(0.89, "b.pdf"),
        _chunk(0.88, "c.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert not sig.score_cluster_concern
    assert not sig.weak_retrieval


# ============================================================
# weak_retrieval
# ============================================================


def test_weak_retrieval_fires_when_max_below_threshold():
    chunks = [_chunk(0.25, "a.pdf"), _chunk(0.20, "b.pdf"), _chunk(0.15, "c.pdf")]
    sig = compute_confidence_signals(_prov(chunks))
    assert sig.weak_retrieval
    assert "weak retrieval" in sig.reasons


def test_weak_retrieval_does_not_fire_at_exact_threshold():
    chunks = [_chunk(0.30, "a.pdf"), _chunk(0.25, "b.pdf"), _chunk(0.20, "c.pdf")]
    sig = compute_confidence_signals(_prov(chunks))
    # max_score (0.30) is NOT < 0.30
    assert not sig.weak_retrieval


# ============================================================
# score_cluster_concern
# ============================================================


def test_cluster_concern_fires_when_top3_tight_and_mid_range():
    chunks = [
        _chunk(0.55, "a.pdf"),
        _chunk(0.54, "b.pdf"),
        _chunk(0.53, "c.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert sig.score_cluster_concern
    assert "ambiguous top matches" in sig.reasons


def test_cluster_concern_does_not_fire_when_max_is_high():
    """Tight cluster at high score = consensus, not concern."""
    chunks = [
        _chunk(0.95, "a.pdf"),
        _chunk(0.94, "b.pdf"),
        _chunk(0.93, "c.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert not sig.score_cluster_concern


def test_cluster_concern_does_not_fire_when_span_is_wide():
    chunks = [
        _chunk(0.65, "a.pdf"),
        _chunk(0.55, "b.pdf"),
        _chunk(0.40, "c.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert not sig.score_cluster_concern


def test_cluster_concern_does_not_fire_when_also_weak():
    """When weak_retrieval fires (max < 0.3), cluster_concern shouldn't double-fire."""
    chunks = [
        _chunk(0.20, "a.pdf"),
        _chunk(0.19, "b.pdf"),
        _chunk(0.18, "c.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert sig.weak_retrieval
    assert not sig.score_cluster_concern  # only weak fires; not both


# ============================================================
# single_source_risk
# ============================================================


def test_single_source_fires_when_one_filename():
    chunks = [_chunk(0.9, "only.pdf") for _ in range(5)]
    sig = compute_confidence_signals(_prov(chunks))
    assert sig.single_source_risk
    assert sig.unique_sources == 1


def test_single_source_fires_at_two_unique_sources():
    chunks = [
        _chunk(0.9, "a.pdf"),
        _chunk(0.9, "a.pdf"),
        _chunk(0.9, "b.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert sig.single_source_risk  # 2 unique, threshold is <= 2
    assert sig.unique_sources == 2


def test_single_source_does_not_fire_at_three_unique_sources():
    chunks = [
        _chunk(0.9, "a.pdf"),
        _chunk(0.9, "b.pdf"),
        _chunk(0.9, "c.pdf"),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    assert not sig.single_source_risk


def test_single_source_does_not_fire_when_no_filenames():
    """Empty/None filenames shouldn't accidentally count as 'one source'."""
    chunks = [RetrievedChunk(filename=None, reranker_score=0.9)]
    sig = compute_confidence_signals(_prov(chunks))
    assert not sig.single_source_risk


# ============================================================
# Multi-flag and reasons ordering
# ============================================================


def test_multiple_flags_combine():
    chunks = [_chunk(0.20, "only.pdf") for _ in range(3)]
    sig = compute_confidence_signals(_prov(chunks))
    assert sig.weak_retrieval and sig.single_source_risk
    assert sig.any()
    assert len(sig.reasons) >= 2


def test_threshold_overrides_work():
    """Caller can pass stricter thresholds."""
    chunks = [_chunk(0.50, "a.pdf"), _chunk(0.49, "b.pdf"), _chunk(0.48, "c.pdf")]
    sig_default = compute_confidence_signals(_prov(chunks))
    sig_strict = compute_confidence_signals(_prov(chunks), weak_threshold=0.6)
    assert not sig_default.weak_retrieval
    assert sig_strict.weak_retrieval


def test_chunks_with_missing_scores_handled_gracefully():
    """No reranker scores present — only single_source check fires."""
    chunks = [
        RetrievedChunk(filename="a.pdf", reranker_score=None),
        RetrievedChunk(filename="a.pdf", reranker_score=None),
    ]
    sig = compute_confidence_signals(_prov(chunks))
    # No scores → can't compute weak/cluster
    assert not sig.weak_retrieval
    assert not sig.score_cluster_concern
    # But the single-source check is still meaningful
    assert sig.single_source_risk
    assert sig.unique_sources == 1
