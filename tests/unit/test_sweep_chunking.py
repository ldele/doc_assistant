"""Guard tests for the chunking sweep's preflight (scripts/sweep_chunking.py).

The preflight exists because the 2026-06-06 sweep swept nothing: ``.env`` overwrote the grid
before ingest read it (KI-38), so all six configs re-embedded the same corpus and the result
read as "no config beats the default" (KI-41 / RG-026). These tests pin the two claims the
preflight makes — *each arm gets what it asked for* and *no two arms are the same experiment* —
and the last one pins the channel itself, end to end, in a real subprocess.

Pure helpers plus one subprocess probe; no pipeline, models, or Chroma.
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import pytest
from scripts.sweep_chunking import (
    DEFAULT_GRID,
    ChunkConfig,
    duplicate_arms,
    ineffective_settings,
    preflight,
    probe_settings,
)

CONTROL = ChunkConfig(2000, 200, 400, 50)
LARGE_PARENT = ChunkConfig(3000, 300, 400, 50)

# A full run_defining_settings() snapshot, as the probe returns it.
BASE_SETTINGS: dict[str, Any] = {
    "parent_chunk_size": 2000,
    "parent_chunk_overlap": 200,
    "child_chunk_size": 400,
    "child_chunk_overlap": 50,
    "baseline_chunk_size": 1000,
    "baseline_chunk_overlap": 200,
    "embedding_model": "bge-base",
    "use_parent_child": True,
    "use_multi_query": False,
    "top_k": 10,
    "candidate_k": 20,
    "bm25_weight": 0.4,
    "rerank_candidate_cap": 60,
}


def settings(**overrides: Any) -> dict[str, Any]:
    return {**BASE_SETTINGS, **overrides}


# ---- ChunkConfig: env carries exactly what the arm asked for ----------------


def test_env_is_the_asked_settings_upper_cased() -> None:
    # config.py reads the upper-cased setting names; one source of truth, no translation table.
    assert LARGE_PARENT.env == {
        "PARENT_CHUNK_SIZE": "3000",
        "PARENT_CHUNK_OVERLAP": "300",
        "CHILD_CHUNK_SIZE": "400",
        "CHILD_CHUNK_OVERLAP": "50",
    }
    assert LARGE_PARENT.asked == {
        "parent_chunk_size": 3000,
        "parent_chunk_overlap": 300,
        "child_chunk_size": 400,
        "child_chunk_overlap": 50,
    }


def test_asked_keys_are_recorded_by_the_run() -> None:
    # The preflight compares `asked` against the run record, so every asked key must be one the
    # record actually carries — otherwise the check silently compares against <not recorded>.
    from doc_assistant.eval.run_settings import run_defining_settings

    assert set(CONTROL.asked) <= set(run_defining_settings())


# ---- ineffective_settings: did the arm get what it asked for? --------------


def test_no_mismatch_when_the_arm_gets_what_it_asked_for() -> None:
    effective = settings(parent_chunk_size=3000, parent_chunk_overlap=300)
    assert ineffective_settings(LARGE_PARENT, effective) == []


def test_reports_the_overwritten_key_with_both_values() -> None:
    # KI-41 exactly: the arm asks for parent 3000, .env puts 2000 back.
    problems = ineffective_settings(LARGE_PARENT, settings())
    assert len(problems) == 2  # size and overlap both lost
    assert "parent_chunk_size: asked 3000, effective 2000" in problems[0]


def test_reports_a_setting_the_run_does_not_record() -> None:
    # A key dropped from run_defining_settings() must fail loudly, not pass by absence.
    absent = settings()
    del absent["child_chunk_size"]
    problems = ineffective_settings(CONTROL, absent)
    assert problems == ["child_chunk_size: asked 400, effective '<not recorded>'"]


# ---- duplicate_arms: is any pair the same experiment twice? ----------------


def test_distinct_arms_are_not_duplicates() -> None:
    resolved = [
        ("control", settings()),
        ("large parent", settings(parent_chunk_size=3000, parent_chunk_overlap=300)),
    ]
    assert duplicate_arms(resolved) == []


def test_ki41_shape_every_arm_collapsing_onto_the_control() -> None:
    # The 2026-06-06 sweep: six notes, one configuration. Every arm pairs with the control.
    resolved = [(cfg.note, settings()) for cfg in DEFAULT_GRID]
    duplicates = duplicate_arms(resolved)
    assert len(duplicates) == len(DEFAULT_GRID) - 1
    assert {first for first, _ in duplicates} == {DEFAULT_GRID[0].note}


def test_duplicates_compare_the_whole_snapshot_not_just_chunk_sizes() -> None:
    # Same chunk geometry, different embedder = a real difference, so not a duplicate.
    resolved = [
        ("a", settings()),
        ("b", settings(embedding_model="specter2")),
        ("c", settings()),
    ]
    assert duplicate_arms(resolved) == [("a", "c")]


def test_default_grid_has_no_duplicate_points() -> None:
    # A grid-authoring check that costs nothing: two identical rows are two identical re-embeds.
    assert duplicate_arms([(cfg.note, cfg.asked) for cfg in DEFAULT_GRID]) == []


# ---- probe_settings: never answers with a guess ----------------------------


def test_probe_raises_when_the_subprocess_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "Traceback...\nImportError: no config")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="ImportError: no config"):
        probe_settings({})


def test_probe_raises_on_output_that_is_not_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, "warning: something\n", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="no settings JSON"):
        probe_settings({})


def test_preflight_reports_an_unresolvable_arm_rather_than_passing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 1, "", "boom")

    monkeypatch.setattr(subprocess, "run", fake_run)
    problems = preflight([CONTROL])
    assert len(problems) == 1
    assert "could not resolve its settings" in problems[0]


def test_preflight_passes_when_every_arm_resolves_to_what_it_asked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(
        _cmd: list[str], *, env: dict[str, str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[str]:
        resolved = settings(**{key.lower(): int(value) for key, value in env.items()})
        return subprocess.CompletedProcess([], 0, json.dumps(resolved), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(os, "environ", {})  # only the arm's own variables reach the fake
    assert preflight([CONTROL, LARGE_PARENT]) == []


def test_preflight_fails_the_whole_grid_when_env_is_overridden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The regression that matters: every arm resolves to the control, whatever it asked for.
    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess([], 0, json.dumps(settings()), "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    problems = preflight(DEFAULT_GRID)
    assert any("asked" in p and "effective" in p for p in problems)
    assert any("same experiment twice" in p for p in problems)


# ---- the channel itself, end to end ---------------------------------------


def test_the_grid_actually_reaches_config_in_a_real_subprocess() -> None:
    """The one test that would have caught KI-41: a real probe under a real environment.

    Everything above is arithmetic over a fake snapshot. This spawns the interpreter the sweep
    spawns, with the arm's variables set, and asserts the settings come back changed — which is
    false under ``load_dotenv(override=True)`` whenever ``.env`` defines the chunk sizes, as
    ``.env.example`` ships them.
    """
    effective = probe_settings({**os.environ, **LARGE_PARENT.env})
    assert ineffective_settings(LARGE_PARENT, effective) == []
