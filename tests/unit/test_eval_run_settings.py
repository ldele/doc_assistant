"""Guards that an eval run records the settings that define what it measured (KI-41 / RG-026).

The 2026-06-06 chunking sweep swept one configuration six times and nothing in the run record
could show it, because `config_json` held no chunk sizes. These tests pin the three properties
that would have surfaced it: the settings are recorded, they are the *live* ones, and a caller's
explicit override wins over the snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from doc_assistant import config
from doc_assistant.eval.results import EvalResult
from doc_assistant.eval.run_settings import run_defining_settings
from doc_assistant.eval.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    """Wired the way a real runner must wire it (`scripts/run_eval.py`)."""
    return Store(tmp_path / "eval.duckdb", settings_provider=run_defining_settings)


@pytest.fixture
def results() -> list[EvalResult]:
    """One minimal result — these tests are about the run row, not the case rows."""
    return []


def test_run_records_the_live_chunk_sizes(
    store: Store, results: list[EvalResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The headline: a sweep's grid point must be visible in its own output."""
    monkeypatch.setattr(config, "PARENT_CHUNK_SIZE", 3000)
    monkeypatch.setattr(config, "PARENT_CHUNK_OVERLAP", 300)
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 256)
    monkeypatch.setattr(config, "CHILD_CHUNK_OVERLAP", 32)

    run_id = store.persist_run(results, system_name="t")
    recorded = store.run_config(run_id)

    assert recorded["parent_chunk_size"] == 3000
    assert recorded["parent_chunk_overlap"] == 300
    assert recorded["child_chunk_size"] == 256
    assert recorded["child_chunk_overlap"] == 32


def test_two_runs_at_different_chunk_sizes_are_distinguishable(
    store: Store, results: list[EvalResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property KI-41 needed and did not have: two arms must differ in the record.

    Had this held in June, six identical `child_chunk_size` values under six different notes
    would have been visible in the sweep's own output.
    """
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 400)
    control = store.persist_run(results, system_name="t", note="control")
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", 600)
    treatment = store.persist_run(results, system_name="t", note="treatment")

    assert store.run_config(control)["child_chunk_size"] == 400
    assert store.run_config(treatment)["child_chunk_size"] == 600


def test_run_records_the_locked_retrieval_settings(
    store: Store, results: list[EvalResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "TOP_K", 8)
    monkeypatch.setattr(config, "CANDIDATE_K", 16)
    monkeypatch.setattr(config, "USE_PARENT_CHILD", False)
    monkeypatch.setattr(config, "USE_MULTI_QUERY", True)

    recorded = store.run_config(store.persist_run(results, system_name="t"))

    assert recorded["top_k"] == 8
    assert recorded["candidate_k"] == 16
    assert recorded["use_parent_child"] is False
    assert recorded["use_multi_query"] is True
    assert recorded["embedding_model"] == config.EMBEDDING_MODEL


def test_run_records_the_generator(
    store: Store, results: list[EvalResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which LLM wrote the answers is part of what the run measured (found 2026-08-15).

    `contains_all` scores the generated text, so a run that does not name its generator cannot
    be compared to one that used a different model — and every run in the store did exactly that
    until this key existed.
    """
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LLM_MODEL", "llama3.1:8b")

    recorded = store.run_config(store.persist_run(results, system_name="t"))

    assert recorded["llm_provider"] == "ollama"
    assert recorded["llm_model"] == "llama3.1:8b"


def test_two_runs_on_different_generators_are_distinguishable(
    store: Store, results: list[EvalResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-08-15 property: same retrieval settings, different model, visible in the record.

    A local-model control and a Haiku re-run are different experiments. Without this they are two
    identical-looking rows whose answer scores differ, which reads as a pipeline improvement.
    """
    monkeypatch.setattr(config, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(config, "LLM_MODEL", "llama3.1:8b")
    local = store.persist_run(results, system_name="t", note="control")
    monkeypatch.setattr(config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(config, "LLM_MODEL", "claude-haiku-4-5-20251001")
    paid = store.persist_run(results, system_name="t", note="re-run")

    assert store.run_config(local)["llm_model"] != store.run_config(paid)["llm_model"]


def test_caller_config_wins_over_the_snapshot(
    store: Store, results: list[EvalResult], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`run_eval --bm25-weight 0.7` runs at 0.7 while `config.BM25_WEIGHT` still reads 0.4.

    The value that actually ran is the one worth recording, so an explicit key wins.
    """
    monkeypatch.setattr(config, "BM25_WEIGHT", 0.4)
    run_id = store.persist_run(results, system_name="t", config={"bm25_weight": 0.7})
    assert store.run_config(run_id)["bm25_weight"] == 0.7


def test_caller_keys_outside_the_snapshot_are_kept(
    store: Store, results: list[EvalResult]
) -> None:
    run_id = store.persist_run(
        results, system_name="t", config={"n_cases": 10, "scorers": ["contains_all"]}
    )
    recorded = store.run_config(run_id)
    assert recorded["n_cases"] == 10
    assert recorded["scorers"] == ["contains_all"]
    assert "child_chunk_size" in recorded  # and the snapshot is still there alongside


def test_rows_written_before_this_change_still_read(store: Store) -> None:
    """Backward compatibility: an old row carries only its caller's keys, and must not crash.

    Written straight to the table the way the pre-2026-08-07 code did — the point is that
    `run_config` returns what is there and does *not* substitute today's values for settings the
    run never recorded. An old run's geometry is unknown, and saying so is the honest answer.
    """
    store.conn.execute(
        "INSERT INTO runs (id, started_at, finished_at, system_name, config_json, n_cases, note) "
        "VALUES ('old-run', now(), now(), 'doc_assistant/bge-base', ?, 10, 'chunk-sweep | ...')",
        [json.dumps({"embedding_model": "bge-base", "n_cases": 10, "scorers": ["contains_all"]})],
    )

    recorded = store.run_config("old-run")

    assert recorded["embedding_model"] == "bge-base"
    assert recorded.get("child_chunk_size") is None
    assert "child_chunk_size" not in recorded


def test_unknown_run_returns_empty(store: Store) -> None:
    assert store.run_config("no-such-run") == {}


def test_snapshot_reads_config_at_call_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Not frozen at import — otherwise a per-run override would never be visible."""
    before = run_defining_settings()["child_chunk_size"]
    monkeypatch.setattr(config, "CHILD_CHUNK_SIZE", before + 111)
    assert run_defining_settings()["child_chunk_size"] == before + 111


def test_store_without_a_provider_records_only_the_caller_config(
    tmp_path: Path, results: list[EvalResult]
) -> None:
    """The default stays inert, which is what keeps the harness liftable.

    `doc_assistant.eval` must not import app config (ADR-003 D8, pinned by
    `test_eval_harness_isolation.py`), so the settings arrive by injection and a bare Store —
    the shape a standalone copy of the harness would use — records exactly what it is given.
    """
    with Store(tmp_path / "bare.duckdb") as bare:
        recorded = bare.run_config(bare.persist_run(results, system_name="t", config={"k": 1}))

    assert recorded == {"k": 1}
