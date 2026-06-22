<!-- status: active · updated: 2026-06-20 · class: living -->

# CONTEXT — doc_assistant

Canonical facts: stack, locked settings, provider config, phase map, open questions. This is the
**single source** for the rules the root `CLAUDE.md` only digests and points at (cpc CONVENTIONS §5).
Conventions: this project follows an internal project-conventions standard (cpc); the decision
and its contract are recorded in `docs/decisions/ADR-001-adopt-cpc-standard.md`.

**Goal:** A local-first personal research assistant. Ingest PDF/EPUB/HTML/DOCX/MD → hybrid RAG
(BM25 + vector + cross-encoder rerank) → answers with inline page-level citations, plus a
research-integrity layer (provenance, evidence/interpretation split, separate-context reviewer).
Not a chatbot wrapper — reliable, grounded, *measurable* answers over **your** documents.

**Current phase (2026-06-20):** Phase 6 in progress; Phase 7 (gap detection) underway. Core RAG,
eval harness, document store + library UI, citation graph, the integrity layer, the provider-agnostic
LLM layer, figures/tables, and the wiki/synthesis layer are shipped. The cross-document concept graph
(PR 16) + the 7d engine shipped too, **but their open-vocabulary core was superseded by a 2026-06-18
redesign that is not yet built — do not build on `data/graph/graph.json` (`.claude/KNOWN_ISSUES.md`
KI-7).** ~555 tests; ruff / mypy --strict / bandit clean.
Desktop-shell migration underway (ADR-002): PR-M0 (`ChatController`) + PR-M1 (live 7d marker chips)
+ PR-M2 (FastAPI + SSE, `apps/api/`) built; next is PR-M3 (Tauri frontend). Other candidates: PR 17
(Zotero ingest), the remaining 7d `query_router` seam, or the wiki `[[links]]`-from-concept-edges
refinement. Full plan: `docs/ROADMAP.md`.

## Stack

| Layer | Choice |
|---|---|
| Language / runtime | Python 3.12 (dev works on 3.14; Chainlit needs 3.12 at runtime). Package manager: **uv**. |
| Embeddings | `bge-base-en-v1.5` (default; swappable via `EMBEDDING_MODEL`, `specter2` also registered) |
| Reranker | `bge-reranker-base` (local cross-encoder) |
| Vector store | Chroma (local, persistent) — `data/chroma/` |
| Keyword search | BM25 |
| Document store | SQLite via SQLAlchemy — `data/library.db` |
| LLM (generation/reviewer/judge) | Claude API **or** local Ollama (provider-agnostic) |
| Orchestration | LangChain |
| UI | Chainlit (web) + CLI + **FastAPI desktop API** (`apps/api/`, PR-M2) — all thin **renderers** over `chat_controller.ChatController` (PR-M0). Migrating to a Tauri desktop shell over the FastAPI/SSE boundary; see `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`. Chainlit stays until PR-M5. |
| PDF / tables | PyMuPDF4LLM (full-text default); Marker for high-fidelity tables, isolated out-of-process post-ingest pass |
| torch backend | per-machine, chosen by a mutually-exclusive uv extra (`cu130` GPU / `cpu`) — see `docs/specs/torch-backend-per-machine.md` |

Pipeline flow + module contracts: `docs/architecture.md`. Design rationale for every choice:
`docs/decisions.md`.

## Locked settings

Change **only** via an experiment through the eval harness: `--repeat`, beat the control beyond its
variance, record a baseline in `tests/eval/baselines/` (and update the decision). Full rationale per
row: `docs/decisions.md`.

| Setting | Locked value | Where enforced / note |
|---|---|---|
| Hybrid retrieval weights | BM25 `0.4` / vector `0.6` | `pipeline.py`. ⚠ vibes-locked — weights never measured; architecture is justified, the split is not. |
| Final rerank cut | `TOP_K = 10` | `config.py`. Measured peak over [3,5,8,10,12,15]; strong shape, weak error bars. |
| Candidate pool / retriever | `CANDIDATE_K = 20` | `config.py`; guard `CANDIDATE_K >= TOP_K`. Split from TOP_K on 2026-06-07 (`09115c8`). Public-confirmed (no regression); **verdict unvalidated** — see Open questions + `tests/eval/baselines/candidate_k_public_2026-06-13.md`. |
| Parent-child retrieval | default ON (`USE_PARENT_CHILD`) | `pipeline.py`; env-toggleable. Measured lift; magnitude weak. |
| Chunk sizes | `2000/200 · 400/50` (parent/child) | `config.py`. 6-config sweep on public corpus — none beats it. |
| Primary ingest path | extract → markdown → chunk → embed → store | Locked (`docs/decisions.md`); enrichment is additive sidecars only. |
| Self-improvement gate | `MIN_FAILURE_TAG_COUNT` / `MIN_FAILURE_TAG_DOCS` | `config.py`. Below the gate: instrumentation, not action. |
| Coverage floor | 40% | CI `--cov-fail-under=40` (`ci.yml`); raise toward 45%+ as integration tests land. |

## Provider config

- LLM provider/model resolved at import time (`LLM_PROVIDER`, `LLM_MODEL`, + reviewer/judge knobs).
  Local default for fully-offline runs: an 8B Ollama model (e.g. `llama3.1:8b`).
- **Credit-leak hazard:** `.env` defaults are all-Anthropic and every generator/reviewer/judge
  inherits it, so a "local" enrichment run can silently bill the API. **Force `--provider ollama`
  on enrichment/self-eval runs.** See `.claude/KNOWN_ISSUES.md`.
- Provider isolation contract: `docs/specs/llm-provider-isolation.md`. No live paid API calls in
  tests (cpc CONVENTIONS §13 — gate-enforced).

## Non-negotiable rules

Engineering preferences (design principles + the four working-protocol rules — never commit/push
without approval, name cross-module coupling, no API credits in tests, copy-pasteable commands) live
in cpc CONVENTIONS **§12 / §13** — read them there, do not restate. Project-specific rules:

1. **Never `git commit`/`push`, nor open/merge a PR, without explicit user review.** Stage, summarize
   the diff, stop. (cpc §13; gate `cpc-push-guard`.)
2. **No secrets in code.** `.env` gitignored; `.env.example` committed with placeholders.
3. **`apps/` are thin shells.** All business logic in `src/doc_assistant/`; UI → library, never the reverse.
4. **Enrichment-Layer Pattern.** Derived data ships as a separate module + idempotent CLI runner,
   sidecar by default, never mutates the primary chunk store.
5. **Structured logging via `structlog`; no `print()` in `src/`.** *(Currently violated — see
   `.claude/KNOWN_ISSUES.md`.)*
6. **Exceptions chain** (`raise X from e`); user-facing messages translated at the UI boundary.
7. **bandit HIGH blocks merge; CI green before merge.** Docs land with the code at every checkpoint.

## Phase map

| Phase | Content | Status |
|---|---|---|
| 1–3 | Core RAG · measured quality + eval harness · document store + library UI | done |
| 4 | Citation graph + doc-similarity edges | done |
| 5 | Embedding & eval foundation (config-driven embedder, golden set, provenance) | done |
| 6 | Figures & tables, dual-layer interpretation, reviewer + self-improvement loop | in progress |
| 7 | Gap detection — wiki/synthesis layer + cross-document concept graph (incl. 7d engine) | in progress |
| 8 | UI polish (settings page exposing the RAG sandbox knobs) | planned |
| 9 | Literature-review generation (PRISMA-trAIce export) | planned |
| — | Extract eval harness to a standalone repo (Feature 5) | planned |

## Open questions

- **`CANDIDATE_K=20` verdict is unvalidated.** The 10-paper public corpus (one-paper-per-topic) is
  too small to rank `CANDIDATE_K`; retest on the larger private set (`cases.yaml`, multi-paper-per-topic)
  with `--repeat` before locking it as a measured win. `CANDIDATE_K=10` reproduces pre-split behaviour.
- **Per-project embedder routing (Feature 1b) deferred** — no model beats `bge-base` on an identifiable
  sub-corpus yet (SPECTER2 lost on every retrieval signal). Re-run SPECTER2 `--repeat 5` first.
- **Concept graph redesign (2026-06-18) decided, not built** — the shipped open-vocabulary core is
  superseded (KI-7); the curated-vocabulary + deterministic-skeleton design has unvalidated edge
  precision + presence recall (flagged for RIGOR_TODO before locking the edge model).
- **BM25/vector `0.4/0.6` weights never measured** — sweep when a `--bm25-weight` flag exists.
