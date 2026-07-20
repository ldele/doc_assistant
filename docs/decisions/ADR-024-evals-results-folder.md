<!-- status: active · updated: 2026-07-20 · class: append-only -->

# ADR-024 — Docs restructure: a top-level `evals/` folder for benchmark results

- **Status:** accepted
- **Date:** 2026-07-20
- **Deciders:** Lucas (directive: "split some of the benchmarks from readme and rest of the
  project and have a dedicated folder for it … or another name, benchmarks might be for
  performance not the evaluation of the AI models"), executed with Claude Code

## Context

`README.md` had grown to 413 lines, ~95 of them the Benchmarks section (headline table, embedder
comparison, chunking sweep, BM25-weight sweep, two reproduction guides). That depth serves the
full-read audience but sits in the document the 30-second/5-minute readers scan (readme-writer:
the README is the door, not the archive). The eval story also had no front door of its own: the
harness lives in `src/doc_assistant/eval/`, the strategy + cases + baselines in `tests/eval/`,
and the results narrative was only in the README.

## Options

1. **Top-level `evals/` results folder; harness/cases/baselines stay put.** Chosen.
2. **Name it `benchmarks/`** — rejected on vocabulary (the user's own instinct): "benchmarks"
   reads as performance (latency/throughput); this project's term everywhere is *eval*
   (`src/doc_assistant/eval/`, `run_eval`, "the eval harness"), and "evals" is the term of art
   for AI-quality measurement.
3. **Move `tests/eval/` wholesale into the new folder** — rejected: 61 files reference
   `tests/eval/` paths (script defaults like `sweep_bm25_weight.DEFAULT_CASES`, the CI
   `--ignore=tests/eval/`, code comments in `config.py`/`llm.py`, and append-only records —
   archived sprints, the decisions monolith — that must not be edited). All churn, no
   functional gain.
4. **`docs/evals/`** — rejected: the folder is the recruiter/engineer-facing benchmark record;
   top-level visibility in the GitHub file listing is the point, same audience logic as the
   README itself.

## Decision

New top-level **`evals/`** containing `README.md`: the full benchmark write-ups (moved verbatim
from the README's Benchmarks section, links re-based) plus a "where the eval pieces live" map and
the public-vs-private question-set split. The root README keeps a compact **Benchmarks** section —
the headline 3-scorer table, one interpretation paragraph, links out. Deliberately **not** moved:
harness code (`src/doc_assistant/eval/`), strategy (`tests/eval/TESTING.md` — cited from code
comments and `.env.example`), cases/manifests, and `tests/eval/baselines/` (the locked-settings
rule in `.claude/CONTEXT.md` names that path).

## Consequences

- **Easier:** the README is door-length again; one URL answers "how good is it and how do I
  check"; a future result gets a narrative home in `evals/` without re-inflating the README.
- **Commits us to:** new benchmark write-ups land in `evals/README.md` (or a sibling file
  there); the underlying run data still goes to `tests/eval/baselines/` per the locked-settings
  rule — `evals/` is the narrative record, `baselines/` is the data.
- **Costs:** one more top-level directory; two-hop links (`evals/README.md` →
  `tests/eval/baselines/*`).
- **Reverses if:** `evals/` never grows past the one README and the split reads as ceremony —
  fold it back into the README.

## Links

- [ADR-021](ADR-021-adopt-cpc-big-project-layout.md) / [ADR-022](ADR-022-docs-system-rationalization.md) / [ADR-023](ADR-023-knowledge-subpackage.md) — the layout/docs-system line this extends.
- `evals/README.md` — the folder's content.
- `tests/eval/TESTING.md` — strategy (unmoved, linked).
