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
* ``--dry-run`` prints the plan (configs + commands) without touching anything.

Usage::

    uv run python -m scripts.sweep_chunking --dry-run
    uv run python -m scripts.sweep_chunking --with-embedding --repeat 3
    uv run python -m scripts.sweep_chunking --with-llm-judge --repeat 5
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass


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
    def env(self) -> dict[str, str]:
        return {
            "PARENT_CHUNK_SIZE": str(self.parent_size),
            "PARENT_CHUNK_OVERLAP": str(self.parent_overlap),
            "CHILD_CHUNK_SIZE": str(self.child_size),
            "CHILD_CHUNK_OVERLAP": str(self.child_overlap),
        }


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
    print("\nCompare configs via the eval harness aggregate, filtering on the notes above")
    print("(each config's runs are tagged with its 'chunk-sweep | ...' note in data/eval.duckdb).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
