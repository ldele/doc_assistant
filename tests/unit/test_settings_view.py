"""ADR-010 "fix in passing": `_settings_view()`'s `retrieval_weights` must be sourced from
`config.BM25_WEIGHT`, not a hardcoded literal — the read-only display cannot silently drift
from the real value. No FastAPI app, no network: this hits the module function directly.
"""

from __future__ import annotations

import pytest
from apps.api.main import _settings_view


def test_retrieval_weights_sourced_from_config(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("doc_assistant.config.BM25_WEIGHT", 0.7)
    weights = _settings_view()["retrieval_weights"]
    assert weights == {"bm25": 0.7, "vector": 0.3}


def test_retrieval_weights_moves_with_the_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("doc_assistant.config.BM25_WEIGHT", 0.1)
    first = _settings_view()["retrieval_weights"]
    monkeypatch.setattr("doc_assistant.config.BM25_WEIGHT", 0.9)
    second = _settings_view()["retrieval_weights"]
    assert first != second
    assert first == {"bm25": 0.1, "vector": 0.9}
    assert second == {"bm25": 0.9, "vector": 0.1}


def test_sandbox_toggle_baselines_move_with_config(monkeypatch: pytest.MonkeyPatch) -> None:
    # U1b (SPRINT-011): the sandbox switch/input need the locked default to render their
    # un-overridden state correctly — these must reflect config, not a stale literal.
    monkeypatch.setattr("doc_assistant.config.EPISTEMICS_MARKERS_ENABLED", False)
    monkeypatch.setattr("doc_assistant.config.REVIEWER_EVIDENCE_CHARS", 777)
    view = _settings_view()
    assert view["epistemics_markers_enabled"] is False
    assert view["reviewer_evidence_chars"] == 777
