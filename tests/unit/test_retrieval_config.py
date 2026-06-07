"""Guards for retrieval-pool configuration (CANDIDATE_K vs TOP_K).

The candidate pool fetched per retriever must be at least as large as the final
rerank cut, or the reranker has nothing to choose from. config.py enforces this
at import; these tests pin the invariant and the import-time guard.
"""

from __future__ import annotations

import importlib

import pytest


def test_candidate_k_at_least_top_k():
    from doc_assistant import config

    assert config.CANDIDATE_K >= config.TOP_K


def test_default_candidate_k_widens_pool():
    """Default must give the reranker real headroom (was hardcoded 10 == TOP_K)."""
    from doc_assistant import config

    assert config.CANDIDATE_K > config.TOP_K


def test_misconfigured_candidate_k_raises(monkeypatch):
    monkeypatch.setenv("CANDIDATE_K", "3")
    monkeypatch.setenv("TOP_K", "10")
    import doc_assistant.config as config

    with pytest.raises(ValueError, match="CANDIDATE_K"):
        importlib.reload(config)

    # Restore module to a clean state for subsequent tests.
    monkeypatch.delenv("CANDIDATE_K", raising=False)
    monkeypatch.delenv("TOP_K", raising=False)
    importlib.reload(config)
