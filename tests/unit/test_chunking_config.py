"""Chunking is config-driven (Phase 6 chunking experiment).

These guard tests pin two things:

1. **Defaults are behaviour-preserving.** The env-var defaults must
   reproduce the historical hardcoded splitter sizes, so making chunking
   configurable changed nothing until an experiment justifies new values.
2. **Factories actually read config.** A sweep sets the sizes via env →
   ``config`` reads them at import → the factories must reflect them. If a
   factory ever hardcodes a size again, the monkeypatch test fails.

We assert on ``RecursiveCharacterTextSplitter._chunk_size`` /
``_chunk_overlap`` — langchain's stable internal storage for these.
"""

from __future__ import annotations

import pytest

from doc_assistant import config, ingest


def test_defaults_match_locked_historical_sizes() -> None:
    """Env-var defaults reproduce the pre-refactor hardcoded values."""
    assert config.PARENT_CHUNK_SIZE == 2000
    assert config.PARENT_CHUNK_OVERLAP == 200
    assert config.CHILD_CHUNK_SIZE == 400
    assert config.CHILD_CHUNK_OVERLAP == 50
    assert config.BASELINE_CHUNK_SIZE == 1000
    assert config.BASELINE_CHUNK_OVERLAP == 200


def test_factories_use_default_config() -> None:
    parent = ingest._make_parent_splitter()
    child = ingest._make_child_splitter()
    baseline = ingest._make_baseline_splitter()

    assert (parent._chunk_size, parent._chunk_overlap) == (2000, 200)
    assert (child._chunk_size, child._chunk_overlap) == (400, 50)
    assert (baseline._chunk_size, baseline._chunk_overlap) == (1000, 200)


@pytest.mark.parametrize(
    ("factory_name", "size_attr", "overlap_attr"),
    [
        ("_make_parent_splitter", "PARENT_CHUNK_SIZE", "PARENT_CHUNK_OVERLAP"),
        ("_make_child_splitter", "CHILD_CHUNK_SIZE", "CHILD_CHUNK_OVERLAP"),
        ("_make_baseline_splitter", "BASELINE_CHUNK_SIZE", "BASELINE_CHUNK_OVERLAP"),
    ],
)
def test_factories_read_config_at_call_time(
    monkeypatch: pytest.MonkeyPatch,
    factory_name: str,
    size_attr: str,
    overlap_attr: str,
) -> None:
    """Each factory reflects monkeypatched config values, proving no hardcoding."""
    monkeypatch.setattr(config, size_attr, 1234)
    monkeypatch.setattr(config, overlap_attr, 56)

    splitter = getattr(ingest, factory_name)()

    assert splitter._chunk_size == 1234
    assert splitter._chunk_overlap == 56


def test_child_smaller_than_parent_by_default() -> None:
    """Sanity: the parent-child contract requires child < parent."""
    assert config.CHILD_CHUNK_SIZE < config.PARENT_CHUNK_SIZE
