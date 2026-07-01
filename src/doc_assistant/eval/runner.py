"""Eval runner — drives cases through a system-under-test (generic).

Caller provides:

* a list of ``EvalCase``,
* a ``SystemUnderTest`` callable mapping a query to ``EvalOutput``,
* a list of ``Scorer`` instances.

The runner times each case, catches exceptions (one bad case shouldn't
abort the run), and returns a list of ``EvalResult``. Persistence is
the caller's concern (see ``Store``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeAlias

import structlog

from doc_assistant.eval.cases import EvalCase
from doc_assistant.eval.results import EvalOutput, EvalResult
from doc_assistant.eval.scorers import Scorer

log = structlog.get_logger(__name__)

SystemUnderTest: TypeAlias = Callable[[str], EvalOutput]


class Runner:
    """Run an eval set against a system-under-test with a fixed scorer mix."""

    def __init__(self, scorers: list[Scorer]) -> None:
        if not scorers:
            raise ValueError("Runner needs at least one scorer")
        self.scorers = scorers

    def run(
        self,
        cases: list[EvalCase],
        system_under_test: SystemUnderTest,
        *,
        progress: Callable[[int, int, EvalCase], None] | None = None,
    ) -> list[EvalResult]:
        """Run every case once, score with every scorer.

        ``progress`` is invoked as ``progress(index, total, case)`` before
        each case — wire it to ``tqdm`` or a logger at the call site.
        Exceptions raised by the system-under-test or by any scorer are
        captured on the corresponding ``EvalResult`` and do not abort
        the run.
        """
        results: list[EvalResult] = []
        total = len(cases)
        for i, case in enumerate(cases):
            if progress is not None:
                progress(i, total, case)

            start = time.monotonic()
            output: EvalOutput | None
            error: str | None
            try:
                output = system_under_test(case.query)
                error = None
            except Exception as e:
                log.exception("system_under_test_raised", case_id=case.id)
                output, error = None, f"{type(e).__name__}: {e}"
            latency_ms = (time.monotonic() - start) * 1000.0

            scores = []
            for scorer in self.scorers:
                try:
                    scores.append(scorer(case, output))
                except Exception as e:
                    log.exception("scorer_raised", scorer=scorer.name, case_id=case.id)
                    from doc_assistant.eval.results import ScoreResult

                    scores.append(
                        ScoreResult(scorer.name, 0.0, {"error": f"{type(e).__name__}: {e}"})
                    )

            results.append(
                EvalResult(
                    case_id=case.id,
                    output=output,
                    scores=scores,
                    latency_ms=latency_ms,
                    error=error,
                )
            )
        return results
