# CLAUDE.md — doc_assistant

Entry-point file for Claude sessions on this project; both Cowork and Claude Code read it on session start. Session status lives in the baton, not in this file.

## What this is

A local-first personal research assistant: ingests PDF/EPUB/HTML/DOCX/MD, hybrid RAG (BM25 + vector + cross-encoder rerank), answers with inline citations plus a research-integrity layer (provenance, dual interpretation, reviewer). Not a general-purpose chatbot — reliable answers grounded in *your* documents, with measurable quality.

## Coordination triad

Read on session start, in this order:

1. `.claude/SESSION.md` — handoff baton: who worked last, what's done/uncommitted, what's next, which tool picks up.
2. `.claude/CONTEXT.md` — stable facts: stack, locked settings, provider config, phase map, open questions.
3. `docs/DEVLOG.md` (tail) — per-change history. (Predates the `.claude/DEVLOG.md` convention; stays in `docs/` — do not move or duplicate.)

Also: `.claude/KNOWN_ISSUES.md` — open weaknesses, recurring failures, workarounds.

Note: `.claude/` is gitignored local working state — these files are absent in a fresh clone.

Human-destined docs live in `docs/` and the repo root (README). Don't put LLM-coordination state there.

## Key documents

- `docs/architecture.md` — pipeline flow, module responsibilities, public contracts, full engineering standards.
- `docs/decisions.md` — every design decision with rationale; this project's ADR home (no `.claude/ADRs/`). **Always check before suggesting architectural changes.**
- `docs/doc-assistant-roadmap.md` — Phase 5+ source of intent; PR-by-PR implementation order at the bottom.
- `docs/figures-and-tables.md` — Feature 4 detection layer (regions, tables, the Marker decision).
- `docs/specs/` — code-level build specs: `feature-4b-figure-detection.md` (✅ built — PR 8, `figures.py` + `extract_figures.py` + `Figure` table, 2026-06-14) · `llm-provider-isolation.md` · `feature-4a-marker-table-ingest.md` · `chunk-2a-dual-interpretation.md` · `feature-7d-knowledge-currency.md` (design-locked, blocked on PR 13 + PR 16) · `torch-backend-per-machine.md` (per-machine CUDA/CPU torch via conflicting extras; ✅ implemented `423cbfa`). **Next PR: 9 — Feature 4c (VLM figure description, gated).**

## Tool split

| Work | Tool |
|------|------|
| Code edits, git, tests, refactors, measurement runs | **Claude Code** |
| Planning, specs, docs, research, connectors/MCP, artifacts | **Cowork** |

When both could do it: stay in the tool that's already open.

## Handoff protocol

End of any non-trivial session: **append** a baton entry to `.claude/SESSION.md` — active tool, what's done, which tool picks up, next action by file (file:line where possible). Append-only; correct with a new entry, never rewrite old ones.

## Build protocol (Claude Code)

1. Pick the next PR from the roadmap's PR-by-PR table. One PR per session — never bundle.
2. Read that PR's `docs/decisions.md` section before starting.
3. Where a spec exists in `docs/specs/`, it is the code-level contract (files, contracts, guard tests, DoD).
4. Append one `docs/DEVLOG.md` entry per logical change (what / why / rejected / opens).

## Engineering standards (non-negotiable)

- No secrets in code. `.env` gitignored; `.env.example` committed.
- CI green before merge; `bandit` HIGH blocks merge.
- Coverage floor CI-enforced: 40% today (`ci.yml`); raise toward 45%+ as integration tests land.
- Structured logging via `structlog`; no `print()` in `src/`. *(Currently violated — see `.claude/KNOWN_ISSUES.md`; fix or descope consciously, don't ignore.)*
- Exceptions chain (`raise X from e`); user-facing messages translated at the UI boundary.
- `apps/` = thin shells only; all business logic in `src/doc_assistant/`.
- **Enrichment-Layer Pattern** — derived data ships as a separate module + CLI runner, idempotent, sidecar by default, never mutates the primary chunk store.
- **Locked settings** (table in `.claude/CONTEXT.md`) change only via an experiment through the eval harness: `--repeat`, beat the control beyond its variance, record a baseline in `tests/eval/baselines/`.
- **Never `git commit`/`git push`, nor open or merge a PR, without explicit user review.** Stage, summarize the diff, ask.
- Refresh docs at every checkpoint (PR, merge, milestone): README, DEVLOG, CONTEXT/architecture as warranted — docs and code land together.

## Skills in play

| When | Use |
|------|-----|
| Session start/end, Cowork↔Code switch | `session-baton` / `handoff` |
| Any change session (log format) | `dev-log` |
| Claiming a result / touching a locked setting | `rigor-gate` |
| Pre-merge review of a diff | `review-router` |
| Recurring bug or design weakness | `known-issues` |
| Non-obvious design choice | `architecture-decision` (record in `docs/decisions.md`) |
| Stress-testing a spec before locking | `grill-me` |
| README work | `readme-writer` |

## Setup

See `README.md` (Python 3.12 + uv: `uv sync` → `.env` → ingest → Chainlit or CLI). Runtime quirks (3.14/Chainlit; the former win32 cu130 segfault, now resolved via `torch-backend = "auto"`; sandbox sync) live in `.claude/KNOWN_ISSUES.md`.
