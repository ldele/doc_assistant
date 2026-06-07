"""Regression guard: the cross-encoder reranker must emit sigmoid-bounded scores.

The integrity layer (``provenance`` thresholds, ``synthesis`` Chunk 2a markers)
assumes reranker scores live in [0, 1]. bge-reranker-base currently defaults to
sigmoid, but a sentence-transformers upgrade could switch it to raw logits and
silently miscalibrate every confidence marker (see pipeline._sigmoid_activation_kwarg).
This test fails loudly if that ever happens.

Loads the real reranker (~1 GB download on first run), so it lives in
integration, not unit. Skips cleanly if the model can't be fetched offline.
"""

from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("torch")


def test_reranker_scores_are_sigmoid_bounded():
    from sentence_transformers import CrossEncoder

    from doc_assistant.pipeline import _sigmoid_activation_kwarg

    try:
        ce = CrossEncoder("BAAI/bge-reranker-base", **_sigmoid_activation_kwarg())
    except Exception as exc:  # offline / no model cache — don't fail CI on network
        pytest.skip(f"reranker unavailable: {exc}")

    scores = ce.predict(
        [
            ["vector embeddings for retrieval", "dense vector embeddings power semantic search"],
            ["vector embeddings for retrieval", "a recipe for chocolate sponge cake"],
        ]
    )

    assert all(0.0 <= float(s) <= 1.0 for s in scores), f"scores not in [0,1]: {scores}"
    # Relevant pair must outrank the irrelevant one (sanity, not just bounds).
    assert float(scores[0]) > float(scores[1])
