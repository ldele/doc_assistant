# AGENTS.md — doc_assistant

Canonical entry file for agent sessions — tool-neutral; any coding/planning agent reads this on
session start. The sibling one-line `CLAUDE.md` is an `@AGENTS.md` import stub so Claude Code,
which reads `CLAUDE.md`, picks this up too (cpc ADR-014; adopted here by ADR-021). This file
**points** at the coordination files; it does not restate them. Session status lives in the baton,
not here. Conventions follow the internal **cpc** standard (`docs/decisions/ADR-001-adopt-cpc-standard.md`);
the big-project layout (this entry file, module `CLAUDE.md` files) is `docs/decisions/ADR-021-adopt-cpc-big-project-layout.md`.

## What this is

A local-first personal research assistant (product name **Provenote**, code identity
`doc_assistant` — ADR-012): ingests PDF/EPUB/HTML/DOCX/MD, hybrid RAG (BM25 + vector +
cross-encoder rerank), answers with inline citations plus a research-integrity layer (provenance,
dual interpretation, reviewer). Not a general-purpose chatbot — reliable answers grounded in *your*
documents, with measurable quality. **Robustness contract:** every feature must degrade honestly at
**0 documents** and scale to **~10,000 documents** — no corpus-tuned magic numbers (`.claude/CONTEXT.md`).

**State (2026-07-19):** Phase 6 + Phase 7 in progress; core RAG, eval harness, integrity layer,
provider-agnostic LLM, figures/tables, wiki, and concept graph all shipped. Detail: `.claude/CONTEXT.md`.
**Stack:** Python 3.12 + uv; Chroma + SQLite; Svelte 5/Tauri + FastAPI; Claude API or local Ollama.
Full stack + locked settings: `.claude/CONTEXT.md`.

## Coordination files (read in this order)

1. `.claude/SESSION.md` — handoff baton: who worked last, what's done/uncommitted, what's next, which tool picks up.
2. `.claude/CONTEXT.md` — canonical facts: stack, locked settings, provider config, phase map, open questions.
3. `docs/DEVLOG.md` (top entries — newest first) — per-change history. *(Lives in `docs/`, not `.claude/` — do not move or duplicate.)*
4. `.claude/KNOWN_ISSUES.md` — open weaknesses, recurring failures, workarounds, plus a one-line
   **Resolved** index; closed issues in full at `docs/archive/KNOWN_ISSUES-resolved-001.md`.

Reference: `docs/ROADMAP.md` · `docs/architecture.md` · `docs/decisions/` (ADRs — living index
`docs/decisions.md`, ADR-022) · `docs/specs/` · `GLOSSARY.md` (pinned vocabulary).

## Sub-module focus (big-project layout, ADR-021)

Each real module boundary carries its own `CLAUDE.md` (≤40 lines: purpose, key files, the rules
that bite there — local only, globals referenced by code, never restated). When a task names a
module, read that module's file and stay inside its boundary:

- `src/doc_assistant/CLAUDE.md` — backend library: **all** business logic lives here.
- `apps/desktop/CLAUDE.md` — Svelte 5 / Tauri frontend (thin renderer).
- `apps/api/CLAUDE.md` — FastAPI/SSE boundary (thin shell over `ChatController`).
- `scripts/CLAUDE.md` — enrichment/eval CLI runners (dry-run defaults, provider guards).

## Non-negotiables (digest — full text in `.claude/CONTEXT.md`)

- **Never `git commit`/`push`, nor open/merge a PR, without explicit user review.** Stage, summarize, stop.
- `apps/` are thin shells; all logic in `src/doc_assistant/`. **Enrichment-Layer Pattern** — derived
  data is an idempotent sidecar module + CLI runner, never mutates the chunk store.
- **Locked settings** (TOP_K, CANDIDATE_K, chunk sizes, retrieval weights, …) change only via an
  eval-harness experiment (`--repeat`, beat the control, record a baseline). Table: `.claude/CONTEXT.md`.
- `structlog` only, no `print()` in `src/` (ADR-003); exceptions chain (`raise X from e`); no
  secrets in code (`.env` gitignored).
- Engineering preferences (design principles + working protocol) live in cpc **CONVENTIONS**
  §12/§13 — read there, don't restate. The cpc gates run **locally only** from the vendored,
  gitignored `tools/conventions/` — never in CI (cpc is private, this repo is public;
  ADR-001/ADR-007). Wiring + commands: `.claude/CONTEXT.md`.

## Work split (by agent role)

| Work | Agent role (current tool) |
|------|---------------------------|
| Code edits, git, tests, refactors, measurement runs | code-execution agent (**Claude Code**) |
| Planning, specs, docs, research, connectors/MCP, artifacts | planning agent (**Cowork**) |

When both could do it: stay in the tool that's already open.

## Handoff protocol

**Trigger (non-negotiable gate).** Any user signal to stop, pause, wrap up, hand off, or switch tools
— "let's stop", "that's it for today", "hand off", "switch to Cowork/Code", "pick this up later" — is
a **MUST-run session-close**: write the baton entry **before yielding the turn**, in the same response
that acknowledges the stop. Do not end such a turn without it. If unsure whether the session did
anything worth recording, write the entry anyway (a one-line "no code change; next: X" baton is cheap;
a lost handoff is not). This is the one step that keeps the next session — or the next tool — able to
resume; skipping it is a protocol violation, not a judgment call. Run
`python tools/conventions/rungate.py keypoint session-close` to get the full close checklist.

End of any non-trivial session: add a baton entry to `.claude/SESSION.md` — **newest on top**, heading
`## YYYY-MM-DD — <Code|Cowork> — <topic>` — active tool, what's done, which tool picks up, next action
by file (file:line where possible). Append-only; correct with a new entry, never rewrite old ones.
Cap 10 entries (cpc ADR-018): rotate older entries verbatim to docs/archive/SESSION-archive-NNN.md
(local-only, like the baton).

## Build protocol (code-execution agent)

1. Pick the next PR from the `docs/ROADMAP.md` PR table. One PR per session — never bundle.
2. Read that PR's ADR in `docs/decisions/` before starting.
3. Where a spec exists in `docs/specs/`, it is the code-level contract (files, contracts, guard tests, DoD).
4. Append one `docs/DEVLOG.md` entry per logical change (what / why / rejected / opens).

## Keypoints (cpc ADR-020 — run the named command at the named moment)

| When | Run |
|------|-----|
| Plan / increment start | `python tools/conventions/rungate.py keypoint plan-start` |
| Session start | `python tools/conventions/rungate.py keypoint session-start` |
| Sprint activation | `python tools/conventions/rungate.py keypoint sprint-start` |
| Sprint checkpoint / closeout | `python tools/conventions/rungate.py keypoint sprint-close` |
| Session end / handoff | `python tools/conventions/rungate.py keypoint session-close` |

Deterministic floor + judgment checklist (cpc ADR-020); per-project extras register in
`scripts/conventions.toml` `[keypoints.<name>]`. Skills-based agents route each checklist line to
the named skill below.

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
> An agent's project settings (e.g. Cowork's) only point here — never restate rules there.
