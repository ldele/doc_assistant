"""How much of this machine an ingest may use, and the parallel extraction that spends it.

**Extraction is 89% of ingest cost and was entirely serial** (measured 2026-08-25: a cold pass over
97 documents took 61 min, the same run with warm caches 6.5 min). It is also per-document and
independent, which makes it the one stage worth parallelising. Chunking, embedding and the store
writes stay serial on purpose: embedding is GPU-bound and already batched, and two processes
writing one Chroma collection is a corruption risk, not a speed-up.

So this module does exactly one thing — **warm the extraction cache in parallel, then get out of
the way.** The existing serial loop runs afterwards and finds every cache fresh. Nothing
downstream knows this happened, and at a budget of 1 the behaviour is byte-identical to before.

**The budget is a politeness control, not a performance knob.** ADR-037 keeps cost knobs out of the
UI because they are not *output-neutral* — they change what an answer says. This one cannot:
worker count changes how long extraction takes and nothing else. What it trades is the user's
machine while they are trying to use it, which is a question engineering cannot answer on their
behalf. Even ``full`` deliberately leaves half the cores alone.

**Resolved per run, never stored as a global.** The desktop reads a persisted preference; a server
passes its own number per request. Baking it into module state would hand the eventual multi-user
build a single-machine assumption it would then have to unpick.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)

#: Named budgets, as fractions of the machine. Deliberately conservative: the point of the control
#: is that Provenote does not take the whole box while someone is working on it.
#:
#: * ``off``      — one worker. Identical to the pre-2026-08-25 serial path.
#: * ``light``    — two. **The default, and the measurement is why** (16 documents, 28 cores):
#:   serial 367.7 s · 2 workers 250.1 s (1.47x) · 4 → 1.68x · 7 → 1.73x · 14 → 1.74x. Two workers
#:   buy **85% of the entire achievable gain for 2 cores instead of 14**, so the polite setting is
#:   very nearly the fast one and there is no reason to make anyone choose.
#: * ``balanced`` — a quarter of the cores. Worth ~0.2x more than ``light`` here; offered because
#:   the curve is corpus- and machine-dependent and someone ingesting overnight may want it.
#: * ``full``     — half. **Never all of them** on a machine with cores to spare, on purpose: OCR
#:   pages are memory-hungry and the machine still belongs to its user.
#:
#: **The ladder is monotonic, which the fractions alone were not.** Every rung is floored at the
#: one below it, so ``balanced`` and ``full`` can never resolve to fewer workers than ``light`` —
#: `cores // 4` used to hand ``balanced`` a single worker on a 4-core laptop. The floor is what
#: makes the small-machine end of the table degenerate rather than inverted: below 5 cores every
#: budget above ``off`` lands on ``light``'s two, which is the whole box at 2 cores. That is the
#: default's own behaviour (``light`` is ``min(2, cores)``), not an escalation — the honest read
#: is that a 2-core machine has one meaningful choice, ``off`` or not.
#:
#: **Why it plateaus is not known.** Two hypotheses were tested and both failed: the long-tail
#: document (the slowest here is 50 s, far under the 211 s floor observed at 14 workers) and
#: OpenMP thread contention (``OMP_THREAD_LIMIT=1`` made it marginally *worse*, 225 s vs 217 s).
#: Recorded as measured-but-unexplained rather than guessed at.
BUDGETS = ("off", "light", "balanced", "full")
DEFAULT_BUDGET = "light"


def resolve_workers(budget: str | int | None = None, *, cpu_count: int | None = None) -> int:
    """How many extraction workers this run may use. Always ≥ 1.

    Accepts a named budget, an explicit integer (a server knows its own capacity), or ``None`` for
    the default. An unrecognised name falls back to the default with a warning rather than raising:
    a stale settings file must not stop an ingest.

    ``DOC_INGEST_WORKERS`` overrides everything — the developer/CI escape hatch, and the seam a
    container uses to say "you have two cores" without a settings file.
    """
    override = os.getenv("DOC_INGEST_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            log.warning("ingest_workers_override_unparsable", value=override)

    cores = cpu_count if cpu_count is not None else (os.cpu_count() or 1)

    if isinstance(budget, int):
        return max(1, budget)
    name = (budget or DEFAULT_BUDGET).strip().lower()
    if name not in BUDGETS:
        log.warning("ingest_budget_unknown", value=budget, using=DEFAULT_BUDGET)
        name = DEFAULT_BUDGET

    if name == "off":
        return 1
    light = min(2, cores)
    if name == "light":
        return light
    # ⚠ Each rung is floored at the one below it, and that floor is the fix rather than a
    # nicety. The fractions alone were **not monotonic on a small machine**: `cores // 4` gave
    # `balanced` 1 worker at 4 cores and `cores // 2` gave `full` 1 at 2 cores, while `light`
    # gave 2 — so a user on an ordinary 4-core laptop who moved *up* from the default to make
    # ingest faster silently got serial extraction. A budget is a ceiling on politeness; it must
    # never ask for less work than a politer one.
    if name == "full":
        return max(light, cores // 2)
    return max(light, cores // 4)  # balanced


def _extract_one(path_str: str) -> tuple[str, str | None]:
    """Warm one document's cache in a worker process. Returns ``(path, error or None)``.

    Module-level and picklable because Windows spawns rather than forks. It calls the ordinary
    `load_or_extract`, so a fresh cache is a no-op and the fingerprint rules apply unchanged — this
    is the same work the serial loop would have done, moved earlier.

    Errors are **returned, not raised**: one unreadable PDF must not take down the pool, and the
    serial pass behind this will meet the same file and report it in its own terms.
    """
    from doc_assistant.ingest.cache import load_or_extract

    try:
        load_or_extract(Path(path_str))
    except Exception as e:
        return path_str, f"{type(e).__name__}: {e}"
    return path_str, None


def warm_extraction_cache(paths: list[Path], workers: int) -> dict[str, int]:
    """Extract ``paths`` into the markdown cache using ``workers`` processes.

    A pure warm-up: it touches the cache and nothing else — no database, no Chroma, no figures. The
    caller's serial loop then runs exactly as it always did and finds the work already done. That
    separation is what keeps this safe to add: if it fails wholesale, ingest is merely as slow as
    it was yesterday.

    Returns ``{"extracted", "failed", "workers"}``. Processes, not threads: PyMuPDF's work is in C
    and the GIL would serialise most of it back.
    """
    if workers <= 1 or len(paths) <= 1:
        return {"extracted": 0, "failed": 0, "workers": 1}

    # ⚠ A frozen build stays serial until someone proves otherwise. Windows spawns rather than
    # forks, so every child re-imports `__main__` — and in a PyInstaller bundle that is the whole
    # application, so each worker would start another API server instead of extracting a PDF.
    # `multiprocessing.freeze_support()` at the entry point is the fix and it is now called there,
    # but it has NOT been verified in a real frozen build, and the failure mode is bad enough
    # (a fork bomb of sidecars) that the safe default is to decline. Delete this branch once a
    # packaged build has been observed extracting in parallel.
    if getattr(sys, "frozen", False) and not os.getenv("DOC_INGEST_PARALLEL_FROZEN"):
        log.info(
            "extraction_warmup_skipped_frozen",
            hint="parallel extraction is unverified in a packaged build; running serially",
        )
        return {"extracted": 0, "failed": 0, "workers": 1}

    workers = min(workers, len(paths))
    log.info("extraction_warmup_start", documents=len(paths), workers=workers)
    done = failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, str(p)): p for p in paths}
        for fut in as_completed(futures):
            try:
                _path, error = fut.result()
            except Exception as e:
                failed += 1
                log.warning("extraction_warmup_worker_died", error=str(e))
                continue
            if error:
                failed += 1
                log.warning("extraction_warmup_failed", file=Path(_path).name, error=error)
            else:
                done += 1
    log.info("extraction_warmup_done", extracted=done, failed=failed, workers=workers)
    return {"extracted": done, "failed": failed, "workers": workers}
