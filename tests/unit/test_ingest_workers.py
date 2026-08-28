"""The ingest politeness budget (`ingest.workers`).

The budget answers "how much of this machine may Provenote use while indexing" — a question
engineering cannot answer for the user, because it depends on what else they are doing. It is
**output-neutral** (worker count changes duration, never results), which is the test ADR-037 used
to decide whether a knob is safe to expose, and it is resolved **per run** so a server can pass its
own number rather than inheriting a desktop preference.

Measured before choosing the default (16 documents, 28 logical cores): serial 367.7 s, 2 workers
250.1 s (1.47x), 4 → 1.68x, 7 → 1.73x, 14 → 1.74x. Two workers buy 85% of the achievable gain for
2 cores instead of 14, which is why `light` is the default rather than a compromise.
"""

from __future__ import annotations

import pytest

from doc_assistant.ingest.workers import (
    BUDGETS,
    DEFAULT_BUDGET,
    resolve_workers,
    warm_extraction_cache,
)


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    monkeypatch.delenv("DOC_INGEST_WORKERS", raising=False)
    monkeypatch.delenv("DOC_INGEST_PARALLEL_FROZEN", raising=False)


# ============================================================
# Resolving the budget
# ============================================================


def test_off_is_exactly_one_worker():
    """`off` must be the old serial path, not "a small number of workers"."""
    assert resolve_workers("off", cpu_count=28) == 1


def test_the_default_is_polite():
    """The whole point of the control. It must never be most of the machine."""
    assert DEFAULT_BUDGET == "light"
    assert resolve_workers(None, cpu_count=28) == 2


def test_no_budget_ever_takes_the_whole_machine():
    """Even `full` leaves half the cores — the machine belongs to its user."""
    for budget in BUDGETS:
        assert resolve_workers(budget, cpu_count=28) <= 14


def test_budgets_are_ordered():
    """At **every** core count, not just a generous one.

    This assertion used to be pinned to `cpu_count=32`, which is exactly where the ladder was
    still monotonic: `cores // 4` gave `balanced` one worker at 4 cores and `cores // 2` gave
    `full` one at 2, both *below* `light`'s two, so moving up a rung on an ordinary laptop
    silently bought serial extraction. A ladder is a claim about the whole range, and pinning it
    to one point tested the claim where it could not fail.
    """
    for cores in range(1, 65):
        order = [resolve_workers(b, cpu_count=cores) for b in ("off", "light", "balanced", "full")]
        assert order == sorted(order), f"cpu_count={cores}: {order}"


def test_a_small_machine_never_gets_a_silly_number():
    """A 2-core laptop must not be told to run 8 extractors."""
    for budget in BUDGETS:
        assert 1 <= resolve_workers(budget, cpu_count=2) <= 2


def test_no_budget_is_ever_below_the_default():
    """`light` is the floor for everything above it — the inversion, stated directly.

    Ranges are what let the original small-machine test walk past this: `1 <= n <= 2` passes for
    both the right answer and the wrong one. This compares the rungs to each other instead.
    """
    for cores in range(1, 65):
        floor = resolve_workers("light", cpu_count=cores)
        for budget in ("balanced", "full"):
            got = resolve_workers(budget, cpu_count=cores)
            assert got >= floor, f"{budget} gave {got} < light's {floor} at cpu_count={cores}"


def test_a_single_core_machine_is_always_serial():
    for budget in BUDGETS:
        assert resolve_workers(budget, cpu_count=1) == 1


def test_an_explicit_integer_is_honoured():
    """A server knows its own capacity; it should not have to pick a named tier."""
    assert resolve_workers(6, cpu_count=28) == 6
    assert resolve_workers(0, cpu_count=28) == 1, "never zero — that would mean no work"


def test_an_unknown_budget_falls_back_instead_of_raising():
    """A stale settings file must not stop an ingest."""
    assert resolve_workers("turbo", cpu_count=28) == resolve_workers(DEFAULT_BUDGET, cpu_count=28)


def test_the_env_override_wins(monkeypatch):
    """The seam a container uses to say "you have two cores" without a settings file."""
    monkeypatch.setenv("DOC_INGEST_WORKERS", "3")
    assert resolve_workers("full", cpu_count=28) == 3


def test_an_unparsable_override_is_ignored_not_fatal(monkeypatch):
    monkeypatch.setenv("DOC_INGEST_WORKERS", "lots")
    assert resolve_workers("off", cpu_count=28) == 1


# ============================================================
# The warm-up itself
# ============================================================


def test_one_worker_does_nothing_at_all(tmp_path):
    """`off` must not merely be slow-parallel — it must not start a pool.

    The serial loop behind this does the identical work, so a budget of 1 has to be a true no-op
    or the run would extract everything twice.
    """
    result = warm_extraction_cache([tmp_path / "a.pdf", tmp_path / "b.pdf"], workers=1)
    assert result == {"extracted": 0, "failed": 0, "workers": 1}


def test_a_single_document_does_not_pay_for_a_pool(tmp_path):
    """Spawning a process to do one document costs more than doing it."""
    assert warm_extraction_cache([tmp_path / "only.pdf"], workers=8)["workers"] == 1


def test_a_frozen_build_stays_serial(monkeypatch, tmp_path):
    """⚠ The fork-bomb guard.

    Windows spawns rather than forks, so a child re-imports `__main__` — and in a PyInstaller
    bundle that is the whole application, so each worker would start another API server instead of
    extracting a PDF. `freeze_support()` at the entry point is the fix, but until a packaged build
    has been observed doing this correctly the safe answer is to decline.
    """
    monkeypatch.setattr("sys.frozen", True, raising=False)
    result = warm_extraction_cache([tmp_path / f"{i}.pdf" for i in range(4)], workers=4)
    assert result["workers"] == 1, "a frozen build must not spawn a pool"


def test_the_frozen_guard_can_be_lifted_deliberately(monkeypatch, tmp_path):
    """So verifying it in a packaged build does not need a code change."""
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setenv("DOC_INGEST_PARALLEL_FROZEN", "1")
    files = [tmp_path / f"{i}.pdf" for i in range(3)]
    for f in files:
        f.write_bytes(b"not a real pdf")
    result = warm_extraction_cache(files, workers=2)
    # They all fail to extract — the point is that a pool was attempted, not that it succeeded.
    assert result["workers"] == 2


def test_an_unreadable_file_is_reported_not_raised(tmp_path):
    """One bad PDF must not take down the pool; the serial pass reports it in its own terms."""
    files = [tmp_path / f"{i}.pdf" for i in range(3)]
    for f in files:
        f.write_bytes(b"definitely not a pdf")
    result = warm_extraction_cache(files, workers=2)
    assert result["failed"] == 3
    assert result["extracted"] == 0
