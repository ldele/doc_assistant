"""Chunking experiment driver — Phase 6 (reopens Phase 2.4).

Sweeps a grid of parent/child chunk sizes through the *real* pipeline so
the best chunking strategy is **measured**, not assumed. For each config it:

1. Re-ingests the corpus with ``ingest --rebuild`` under the config's chunk
   sizes (a chunk-size change invalidates the embedding cache, so a full
   re-embed per config is mandatory — this is the slow part).
2. Runs ``scripts.run_eval`` and tags the run with a ``--note`` that encodes
   the exact config, so the runs are identifiable in ``data/eval.duckdb``.

It does **not** invent its own scoring or aggregation — it reuses the Phase 5
eval harness end to end. After the sweep, compare configs with the harness's
own aggregate report (filter the runs by the printed notes).

Cost & safety
-------------
* This rebuilds your vector stores repeatedly. Run it against a corpus you can
  afford to re-embed, ideally a representative subset (point ``DOCS_PATH`` at a
  sample, or stage a smaller library).
* ``--with-llm-judge`` calls the Anthropic API once per case per config —
  budget before enabling.
* ``--dry-run`` prints the plan (configs + commands) without ingesting or evaluating. It
  **does** run the preflight below, which is the cheap way to prove a sweep is wired before
  paying a GPU-day for it.

Preflight — why this sweep refuses to start (KI-41 / RG-026)
------------------------------------------------------------
The grid travels to the ingest subprocess through the environment, and until 2026-08-07
``config.load_dotenv(override=True)`` overwrote all four variables from ``.env`` — which
``.env.example`` ships uncommented — before ingest read them. The 2026-06-06 sweep therefore
re-embedded the **same** configuration six times and compared it with itself. Nothing raised,
nothing logged, and the run record held no chunk sizes to contradict the note; it cost ~6 full
corpus re-embeds and stood as the evidence for a locked setting for two months.

So before the first re-embed, every arm is asked what it would *actually* run under, and the
sweep stops unless each arm gets the settings it asked for and no two arms are the same
experiment. A driver whose variable is silently ignored fails in the "no effect" direction,
which reads as a confirmed default — the one failure a negative result cannot survive.

Usage::

    uv run python -m scripts.sweep_chunking --dry-run
    uv run python -m scripts.sweep_chunking --with-embedding --repeat 3
    uv run python -m scripts.sweep_chunking --with-llm-judge --repeat 5
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChunkConfig:
    """One point in the chunking grid. Overlaps default to ~10-12% of size."""

    parent_size: int
    parent_overlap: int
    child_size: int
    child_overlap: int

    @property
    def note(self) -> str:
        return (
            f"chunk-sweep | parent={self.parent_size}/{self.parent_overlap} "
            f"child={self.child_size}/{self.child_overlap}"
        )

    @property
    def asked(self) -> dict[str, int]:
        """What this arm intends to vary, keyed as ``run_defining_settings`` keys.

        Named in the recorded vocabulary rather than the environment's so the preflight can
        compare intent against the run record directly, with no translation table to drift.
        """
        return {
            "parent_chunk_size": self.parent_size,
            "parent_chunk_overlap": self.parent_overlap,
            "child_chunk_size": self.child_size,
            "child_chunk_overlap": self.child_overlap,
        }

    @property
    def env(self) -> dict[str, str]:
        """The subprocess environment carrying :attr:`asked` into ingest and eval.

        ``config.py`` reads these as the upper-cased setting names, so that is how they are
        derived — one source of truth. If the two naming schemes ever diverge, the arm sets a
        variable nothing reads and the preflight says so on the next run: its ``effective``
        comes back at the default while ``asked`` did not change.
        """
        return {key.upper(): str(value) for key, value in self.asked.items()}


# Default grid. Index 0 is the current locked default (the baseline to beat).
# Keep the grid small — every row is a full corpus re-embed plus an eval pass.
DEFAULT_GRID: list[ChunkConfig] = [
    ChunkConfig(2000, 200, 400, 50),  # current default (control)
    ChunkConfig(2000, 200, 256, 32),  # smaller child — finer retrieval
    ChunkConfig(2000, 200, 600, 75),  # larger child — more context per hit
    ChunkConfig(1500, 150, 400, 50),  # smaller parent — tighter LLM context
    ChunkConfig(3000, 300, 400, 50),  # larger parent — broader LLM context
    ChunkConfig(1000, 100, 256, 32),  # small/small — precision regime
]


# The preflight probe. It calls ``run_defining_settings`` — the very function that writes
# ``config_json`` on every eval run — rather than reading the config constants itself. A
# verification gate must call the contract, never restate it: a restated gate can disagree with
# the thing it guards and be believed anyway (the RG-012 false failure, 2026-08-07).
_PROBE = (
    "import json;"
    "from doc_assistant.eval.run_settings import run_defining_settings;"
    "print(json.dumps(run_defining_settings()))"
)


def probe_settings(env: Mapping[str, str]) -> dict[str, Any]:
    """Ask a subprocess under ``env`` which run-defining settings it would use.

    A subprocess, not an in-process read, because that *is* the channel under test: ``config``
    resolves the environment once at import, and the sweep's ingest and eval are subprocesses.
    Reading the parent's own already-imported config would test nothing.

    Raises ``RuntimeError`` if the probe cannot answer — never a default, never a guess. A
    preflight that cannot resolve the settings has not shown they are right.
    """
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=dict(env),
        capture_output=True,
        text=True,
        encoding="utf-8",  # Windows: pipes decode as cp1252 otherwise (non-negotiable #9)
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        raise RuntimeError(detail[-1] if detail else f"probe exited {proc.returncode}")
    try:
        settings = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"probe printed no settings JSON: {proc.stdout.strip()[:120]!r}") from e
    if not isinstance(settings, dict):
        raise RuntimeError(f"probe returned {type(settings).__name__}, expected an object")
    return settings


def ineffective_settings(cfg: ChunkConfig, effective: Mapping[str, Any]) -> list[str]:
    """The settings this arm asked for that its run would not actually use.

    Empty means the grid reaches the code. Non-empty is KI-41 exactly: the arm sets the
    variable, and something between the variable and ``config`` wins.
    """
    return [
        f"{key}: asked {asked}, effective {effective.get(key, '<not recorded>')!r}"
        for key, asked in cfg.asked.items()
        if effective.get(key) != asked
    ]


def duplicate_arms(resolved: Sequence[tuple[str, Mapping[str, Any]]]) -> list[tuple[str, str]]:
    """Pairs of arms that would record identical run-defining settings.

    Compares the **whole** snapshot rather than the chunk sizes alone: two arms recording the
    same settings are indistinguishable in ``data/eval.duckdb`` whatever made them so, and a
    difference the record cannot show is a difference the comparison cannot use.
    """
    first_seen: dict[str, str] = {}
    duplicates: list[tuple[str, str]] = []
    for note, settings in resolved:
        fingerprint = json.dumps(settings, sort_keys=True, default=str)
        first = first_seen.setdefault(fingerprint, note)
        if first != note:
            duplicates.append((first, note))
    return duplicates


def preflight(grid: Sequence[ChunkConfig]) -> list[str]:
    """Check the grid reaches the code, before a single corpus is re-embedded.

    Returns the problems found; empty means the sweep is safe to run.
    """
    print(f"Preflight: resolving what each of the {len(grid)} configs would actually run under.")
    problems: list[str] = []
    resolved: list[tuple[str, Mapping[str, Any]]] = []
    for cfg in grid:
        try:
            effective = probe_settings({**os.environ, **cfg.env})
        except RuntimeError as e:
            problems.append(f"{cfg.note}: could not resolve its settings - {e}")
            print(f"  [ERROR ] {cfg.note}")
            continue
        resolved.append((cfg.note, effective))
        mismatches = ineffective_settings(cfg, effective)
        problems.extend(f"{cfg.note} -> {m}" for m in mismatches)
        print(
            f"  [{'OK' if not mismatches else 'IGNORED':6}] effective "
            f"parent={effective.get('parent_chunk_size')}/"
            f"{effective.get('parent_chunk_overlap')} "
            f"child={effective.get('child_chunk_size')}/{effective.get('child_chunk_overlap')}"
        )
    problems.extend(
        f"same experiment twice: {first!r} and {second!r} record identical settings"
        for first, second in duplicate_arms(resolved)
    )
    return problems


def _run(cmd: list[str], env: dict[str, str], *, dry_run: bool) -> int:
    printable = " ".join(cmd)
    if dry_run:
        print(f"    [dry-run] {printable}")
        return 0
    print(f"    $ {printable}")
    proc = subprocess.run(cmd, env=env)
    return proc.returncode


def _eval_cmd(
    note: str, *, cases: str | None, with_embedding: bool, with_llm_judge: bool, repeat: int
) -> list[str]:
    cmd = [sys.executable, "-m", "scripts.run_eval", "--note", note, "--repeat", str(repeat)]
    if cases is not None:
        cmd.extend(["--cases", cases])
    if with_embedding:
        cmd.append("--with-embedding")
    if with_llm_judge:
        cmd.append("--with-llm-judge")
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-embedding", action="store_true", help="Add embedding scorer")
    parser.add_argument(
        "--with-llm-judge", action="store_true", help="Add LLM judge (Anthropic API — costs money)"
    )
    parser.add_argument(
        "--repeat", type=int, default=1, help="Eval trials per config (variance). Default 1."
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=None,
        help="Cases YAML passed to run_eval (default: run_eval's own default). "
        "Use tests/eval/cases.public.yaml to keep the sweep in the verified-10 public regime.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without ingesting or evaluating"
    )
    args = parser.parse_args()
    if args.repeat < 1:
        print("--repeat must be >= 1")
        return 1

    grid = DEFAULT_GRID
    print(f"Chunking sweep: {len(grid)} configs, --repeat {args.repeat}")
    print("Each config = full re-ingest (re-embed) + eval. This is slow by design.\n")

    # Before anything is re-embedded, and in --dry-run too: a sweep whose grid does not reach
    # the code produces a confident negative result about a configuration it never ran (KI-41).
    problems = preflight(grid)
    if problems:
        print("\nPreflight FAILED. Nothing was ingested or evaluated.")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nAn arm whose setting is overwritten measures the control, so the sweep would\n"
            "report 'no effect' for a configuration it never ran. Check the variables above\n"
            "against .env, which takes effect for any the environment leaves empty and which\n"
            ".env.example ships with the chunk sizes uncommented (KI-38/KI-41)."
        )
        return 1
    print("Preflight OK: every config reaches the code, and no two are the same experiment.\n")

    failures: list[str] = []
    for i, cfg in enumerate(grid, start=1):
        tag = "(control)" if i == 1 else ""
        print(f"[{i}/{len(grid)}] {cfg.note} {tag}")

        run_env = {**os.environ, **cfg.env}

        ingest_cmd = [sys.executable, "-m", "doc_assistant.ingest", "--rebuild"]
        rc = _run(ingest_cmd, run_env, dry_run=args.dry_run)
        if rc != 0:
            print(f"    ! ingest failed (rc={rc}); skipping eval for this config")
            failures.append(cfg.note)
            continue

        eval_cmd = _eval_cmd(
            cfg.note,
            cases=args.cases,
            with_embedding=args.with_embedding,
            with_llm_judge=args.with_llm_judge,
            repeat=args.repeat,
        )
        rc = _run(eval_cmd, run_env, dry_run=args.dry_run)
        if rc != 0:
            print(f"    ! eval failed (rc={rc})")
            failures.append(cfg.note)
        print()

    print(
        "Sweep complete." if not failures else f"Sweep finished with {len(failures)} failure(s):"
    )
    for f in failures:
        print(f"  - {f}")
    # The grid's own keys, so this hint cannot drift from what the sweep actually varied.
    varied = " ".join(sorted(DEFAULT_GRID[0].asked))
    print("\nEach config's runs are tagged with its 'chunk-sweep | ...' note in data/eval.duckdb.")
    print("Read two arms against each other with the comparability check, declaring the grid")
    print("as the independent variable -- the arms are then not flagged for the thing they")
    print("vary, while anything ELSE that moved (the corpus above all) still blocks:")
    print("")
    print("    python -m scripts.compare_runs --list")
    print(f"    python -m scripts.compare_runs <control-run> <arm-run> --varying {varied}")
    print("")
    print("If it reports that a declared variable did NOT change, the preflight passed and the")
    print("arms still measured one configuration twice -- that is KI-41, read from the record.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
