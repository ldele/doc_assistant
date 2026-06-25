# CLAUDE.md — doc_assistant

Entry point for Claude sessions (Code + Cowork read this on session start). It **points** at the
coordination files; it does not restate them. Session status lives in the baton, not here. Project
conventions follow an internal **CONVENTIONS** standard (cpc); see `docs/decisions/ADR-001-adopt-cpc-standard.md`. ADRs in `docs/decisions/`.

## What this is

A local-first personal research assistant: ingests PDF/EPUB/HTML/DOCX/MD, hybrid RAG (BM25 + vector
+ cross-encoder rerank), answers with inline citations plus a research-integrity layer (provenance,
dual interpretation, reviewer). Not a general-purpose chatbot — reliable answers grounded in *your*
documents, with measurable quality.

**State (2026-06-20):** Phase 6 + Phase 7 in progress; core RAG, eval harness, integrity layer,
provider-agnostic LLM, figures/tables, wiki, and concept graph all shipped. Detail: `.claude/CONTEXT.md`.
**Stack:** Python 3.12 + uv; Chroma + SQLite; Claude API or local Ollama. Full stack + locked
settings: `.claude/CONTEXT.md`.

## Coordination files (read in this order)

1. `.claude/SESSION.md` — handoff baton: who worked last, what's done/uncommitted, what's next, which tool picks up.
2. `.claude/CONTEXT.md` — canonical facts: stack, locked settings, provider config, phase map, open questions.
3. `docs/DEVLOG.md` (tail) — per-change history. *(Lives in `docs/`, not `.claude/` — do not move or duplicate.)*
4. `.claude/KNOWN_ISSUES.md` — open weaknesses, recurring failures, workarounds.

Reference: `docs/ROADMAP.md` · `docs/architecture.md` · `docs/decisions.md` (ADR home; split into `docs/decisions/` in progress, ADR-001) · `docs/specs/`.

Tracking: `.claude/CONTEXT.md` + `.claude/KNOWN_ISSUES.md` are committed; `.claude/SESSION.md` stays local (gitignored).

## Non-negotiables (digest — full text in `.claude/CONTEXT.md`)

- **Never `git commit`/`push`, nor open/merge a PR, without explicit user review.** Stage, summarize, stop.
- `apps/` are thin shells; all logic in `src/doc_assistant/`. **Enrichment-Layer Pattern** — derived
  data is an idempotent sidecar module + CLI runner, never mutates the chunk store.
- **Locked settings** (TOP_K, CANDIDATE_K, chunk sizes, retrieval weights, …) change only via an
  eval-harness experiment (`--repeat`, beat the control, record a baseline). Table: `.claude/CONTEXT.md`.
- `structlog` only, no `print()` in `src/` *(currently violated — `.claude/KNOWN_ISSUES.md`)*;
  exceptions chain (`raise X from e`); no secrets in code (`.env` gitignored).
- Engineering preferences (design principles + working protocol) live in the cpc **CONVENTIONS**
  §12/§13 — read there, don't restate. Three are gate-enforced (`cpc-push-guard`, `cpc-coupling-check`,
  `cpc-test-api-check`); plus the doc gates (`cpc-docs-check`, `cpc-init-check`).

## Tool split

| Work | Tool |
|------|------|
| Code edits, git, tests, refactors, measurement runs | **Claude Code** |
| Planning, specs, docs, research, connectors/MCP, artifacts | **Cowork** |

When both could do it: stay in the tool that's already open.

## Handoff protocol

End of any non-trivial session: **append** a baton entry to `.claude/SESSION.md` — active tool, what's
done, which tool picks up, next action by file (file:line where possible). Append-only; correct with a
new entry, never rewrite old ones.

## Build protocol (Claude Code)

1. Pick the next PR from the `docs/ROADMAP.md` PR table. One PR per session — never bundle.
2. Read that PR's `docs/decisions.md` section (or its ADR) before starting.
3. Where a spec exists in `docs/specs/`, it is the code-level contract (files, contracts, guard tests, DoD).
4. Append one `docs/DEVLOG.md` entry per logical change (what / why / rejected / opens).

## Skills in play

| When | Use |
|------|-----|
| Session start/end, Cowork↔Code switch | `session-baton` / `handoff` |
| Any change session (log format) | `dev-log` |
| Claiming a result / touching a locked setting | `rigor-gate` |
| Pre-merge review of a diff | `review-router` |
| Recurring bug or design weakness | `known-issues` |
| Non-obvious design choice | `architecture-decision` (record in `docs/decisions/`) |
| Stress-testing a spec before locking | `grill-me` |
| README work | `readme-writer` |

## Setup

See `README.md` (Python 3.12 + uv: `uv sync` → `.env` → ingest → Tauri desktop or CLI). Runtime quirks
(the 3.12 pin — native deps not yet 3.14-stable, KI-2; the resolved win32 cu130 segfault; sandbox sync;
the Anthropic credit-leak) live in `.claude/KNOWN_ISSUES.md`.

> Rules changed? Edit `.claude/CONTEXT.md` (the canonical text). This file only points.
> Cowork project settings only point here — never restate rules there.
