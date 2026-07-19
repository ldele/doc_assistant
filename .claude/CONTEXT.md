<!-- status: active · updated: 2026-07-19 (ADR-021 big-project layout: AGENTS.md entry + module CLAUDE.md files + cpc 1.2.3 vendored) · class: living -->

# CONTEXT — doc_assistant

Canonical facts: stack, locked settings, provider config, phase map, open questions. This is the
**single source** for the rules the root entry file (`AGENTS.md`; `CLAUDE.md` is its one-line
`@AGENTS.md` stub — ADR-021) only digests and points at (cpc CONVENTIONS §5).
Conventions: this project follows an internal project-conventions standard (cpc); the decision
and its contract are recorded in `docs/decisions/ADR-001-adopt-cpc-standard.md`.

**Goal:** A local-first personal research assistant. Ingest PDF/EPUB/HTML/DOCX/MD → hybrid RAG
(BM25 + vector + cross-encoder rerank) → answers with inline page-level citations, plus a
research-integrity layer (provenance, evidence/interpretation split, separate-context reviewer).
Not a chatbot wrapper — reliable, grounded, *measurable* answers over **your** documents.

**Product name: Provenote** (V3a/V3b, committed `181046c`/`487f2df`, 2026-07-14) — the product/
installer identity only (wordmark, window title, app icon, bundle id `com.provenote.desktop`).
The **code identity stays `doc_assistant`** (Python package, commands, npm name, `doc-assistant-api`
sidecar) — intentional split, do not "finish" the rename into code. ADR-012.

**Current phase (2026-07-07):** Phase 6 done; Phase 7 (concept skeleton + gap detection) well
underway. Core RAG, eval harness, document store + library UI, citation graph, the integrity layer,
the provider-agnostic LLM layer, figures/tables, and the wiki/synthesis layer are shipped. The
concept-graph redesign — curated vocabulary + deterministic skeleton (Node A, 2026-06-30) +
confined LLM relation/stance enrichment (Node B, PR #6 `6679540`) — is **BUILT**
(`concept_skeleton.py` + `concept_skeleton_enrich.py` + `scripts/{seed_concepts,
build_concept_skeleton}.py` + the `concept_*` tables + `CONCEPT_SKELETON_*` config), validated on
the real corpus (RG-001/008/009, R5 PASS, ADR-008). **The retired open-vocabulary
`concept_graph.py` is DELETED (2026-07-07, KI-7 RESOLVED, ROADMAP row G1)** — `epistemics.py` /
`wiki.py` now read the skeleton directly; `EPISTEMICS_MARKERS_ENABLED` defaults on. **The
gap-detection layer's deterministic Tier-1 + Tier-2a floor is BUILT (2026-07-07, ROADMAP row G2)**,
and **the Tier-2a stochastic ceiling is BUILT (2026-07-08, ROADMAP row G5)** — `gaps.py` + `GapRow` +
`scripts/build_gaps.py` + `gap_suggest.py`, ADR-004; real-validated on the RTX/Ollama box
(`tests/eval/baselines/gap_suggest_ollama_2026-07-08.md`). **G4**
(`SPRINT-004-ki10-frozen-os-trust.md`, still active — genuinely un-landed, not just un-archived) is
the one remaining planned-contract sprint: the KI-10 frozen-build OS-trust fix, runnable only on a
TLS-MITM box (on-proxy paid verification user-approved), which this RTX box is not. **G3**
(`SPRINT-003-year-aware-superseded.md`) — un-parked 2026-07-08 (the `extract_doc_metadata --apply`
backfill gave 45/47 docs a year, 96%, disproving the "coverage too thin" park premise) and **code
built same day** (`load_doc_years` + `_aggregate_direction`, median-vs-median, parameter-free,
fail-safe to `contested` on missing years; `epistemics.py` unchanged); **host `--apply` run DONE
2026-07-08 (same session as G6, below)** — real corpus: 226 contested / 26 superseded_trend nodes
(ungated). **G6** (`SPRINT-006-gate-superseded-confidence.md`) — **built + real-validated
2026-07-08, staged, awaiting review**: gates `superseded_trend` to >= 2 dated docs per side
(`MIN_DATED_DOCS_PER_SIDE`, named constant not a `config.py` tunable); real corpus after-gate:
**9 superseded_trend nodes** (17/26 pre-gate fires, 65%, were the demoted single-doc case — the
review finding that motivated the sprint, confirmed at similar magnitude). Found and worked around
a real footgun in the process: `build_concept_skeleton --apply` **alone** wipes existing Node-B
stance annotations (rebuilds edges with no `relation`/`stance_by_doc`) — the correct host command
is `--apply --enrich` together; see `tests/eval/baselines/superseded_year_rule_2026-07.md`. **G7**
(`SPRINT-007-fix-epistemics-label-attribution.md`) — **built + real-validated same day**: fixed
KI-15, the bug G6 surfaced — `epistemics.concepts_in_text` was matching concept **UUIDs** (the
curated skeleton's node id), not labels, against chunk text, so live answer-time marker surfacing
(PR-M1) had been silently dark on the real corpus since the G1 re-point, independent of G3/G6's
node-level correctness. Now matches on label via a shared
`concept_skeleton.compile_boundary_pattern` (R2's alnum-boundary regex, reused not reimplemented).
Real corpus: **0 → 4008 chunks with a claim, 0 → 3334 marked** (of 6215) — spot-checked correct,
not just non-zero. KI-15 RESOLVED. Still deferred: Tier 2b (external reach); S1/S2 selective
ingestion; a live-UI smoke test that the desktop chips actually render now (PR-M1's read side was
never the broken part). 790 tests; ruff / mypy --strict / bandit clean.
Desktop-shell migration (ADR-002): **M0–M5 all shipped (2026-06-25).** M0 (`ChatController`) · M1 (live 7d
marker chips) · M2 (FastAPI + SSE, `apps/api/`) · M3 (Svelte/Tauri frontend, `apps/desktop/`) · **M4** —
frozen 1.6 GB onefile bundling model weights (KI-9) + OS trust store (KI-10) + the ASCII-Chroma fix (KI-11);
RG-010 (~30 s cold-start) / RG-011 (no SSE first-token penalty) / RG-012 Tier-1 (clean-machine freeze +
installer smoke, in Windows Sandbox) / RG-013 all closed. The **data-home / first-run-ingest flow is now
built** (backend `77eb5f9`: `/api/settings` + `/api/ingest`; frontend `apps/desktop` settings panel +
empty-corpus banner) — RG-012 **Tier-2** (a cited turn on a clean box) pends a re-freeze bundling it + the
clean-box run. Runbook: `docs/desktop-packaging.md`. · **M5** — Chainlit removed; the 3.12-pin lift
was verified-and-deferred (KI-2: native deps crash on 3.14, not Chainlit). UI is now Tauri desktop + CLI.
Other candidates: PR 17 (Zotero ingest), the 7d `query_router` seam, the wiki `[[links]]` refinement.
Full plan: `docs/ROADMAP.md`.

## Stack

| Layer | Choice |
|---|---|
| Language / runtime | Python 3.12 (the pinned runtime; native deps not yet cp314-stable — KI-2; Chainlit removed in PR-M5). Package manager: **uv**. |
| Embeddings | `bge-base-en-v1.5` (default; swappable via `EMBEDDING_MODEL`, `specter2` also registered) |
| Reranker | `bge-reranker-base` (local cross-encoder) |
| Vector store | Chroma (local, persistent) — `data/chroma/` |
| Keyword search | BM25 |
| Document store | SQLite via SQLAlchemy — `data/library.db` |
| LLM (generation/reviewer/judge) | Claude API **or** local Ollama (provider-agnostic) |
| Orchestration | LangChain |
| UI | **Tauri desktop app** (`apps/desktop/`, Svelte 5 + Vite, PR-M3) over the **FastAPI/SSE** boundary (`apps/api/`, PR-M2); CLI remains — both thin **renderers** over `chat_controller.ChatController` (PR-M0). See `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md`. Chainlit removed at PR-M5; native Tauri packaging is PR-M4. |
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
| Hybrid retrieval weights | BM25 `0.4` / vector `0.6` | `config.BM25_WEIGHT` (vector = `1 - w`), wired in `pipeline.py`. **Swept 2026-07-03** (`tests/eval/baselines/bm25_weight_sweep_2026-07-03.md`): post-rerank recall is FLAT across `[0,1]` — the cross-encoder re-scores the full candidate union, so the weight is inert on the shipped top-K by construction. Kept (negative result); the split is on the better *pre*-rerank side. Sweep: `scripts/sweep_bm25_weight.py` / `--bm25-weight`. |
| Final rerank cut | `TOP_K = 10` | `config.py`. Measured peak over [3,5,8,10,12,15]; strong shape, weak error bars. |
| Candidate pool / retriever | `CANDIDATE_K = 20` | `config.py`; guard `CANDIDATE_K >= TOP_K`. Split from TOP_K on 2026-06-07 (`09115c8`). Public-confirmed (no regression); **verdict unvalidated** — see Open questions + `tests/eval/baselines/candidate_k_public_2026-06-13.md`. |
| Parent-child retrieval | default ON (`USE_PARENT_CHILD`) | `pipeline.py`; env-toggleable. Measured lift; magnitude weak. |
| Chunk sizes | `2000/200 · 400/50` (parent/child) | `config.py`. 6-config sweep on public corpus — none beats it. |
| Primary ingest path | extract → markdown → chunk → embed → store | Locked (`docs/decisions.md`); enrichment is additive sidecars only. |
| Self-improvement gate | `MIN_FAILURE_TAG_COUNT` / `MIN_FAILURE_TAG_DOCS` | `config.py`. Below the gate: instrumentation, not action. |
| Coverage floor | 40% | CI `--cov-fail-under=40` (`ci.yml`); raise toward 45%+ as integration tests land. |

## Provider config

- LLM provider/model resolved at import time (`LLM_PROVIDER`, `LLM_MODEL`, + reviewer/judge knobs)
  for every CLI/enrichment path. Local default for fully-offline runs: an 8B Ollama model (e.g.
  `llama3.1:8b`).
- **Desktop chat generation is the one exception (ADR-011, U1c, 2026-07-11):** the Settings UI can
  live-switch the chat provider/model between already-configured providers (key stays in `.env`,
  no restart) — `app_settings.effective_llm()` is the source of truth there, not the raw
  `LLM_PROVIDER`/`LLM_MODEL` constants. The reviewer follows an unpinned switch; an explicit
  `REVIEWER_PROVIDER` in `.env` still wins. In-app key entry (a v2 north-star) is not built.
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
5. **Structured logging via `structlog`; no `print()` in `src/`.** Configured once per entrypoint
   via `logging_config.configure_logging` (console renderer for dev/CLI, JSON when `LOG_JSON=true`);
   `src/` library code never configures logging. Enforced as of ADR-003 (KI-1 closed). The one
   sanctioned `stderr` write is the paid-run abort-window box in `llm.py` (an interactive CLI prompt,
   not a log event).
6. **Exceptions chain** (`raise X from e`); user-facing messages translated at the UI boundary.
7. **bandit HIGH blocks merge; CI green before merge.** Docs land with the code at every checkpoint.

**cpc gate wiring (ADR-007 — canonical text).** The cpc gates are vendored at `tools/conventions/`
(cpc **1.2.3**; re-vendor by re-running `cpc-init` from the cpc checkout at the release *tag*,
never an unreleased HEAD — this box re-vendored 2026-07-19 with the ADR-021 layout switch) and
wired via `.pre-commit-config.cpc.yaml`, whose hooks run the **vendored copy** through the
`tools/conventions/rungate.py` shim (no pip-install from the private remote, no network at hook
time). **Both are gitignored — local-only, never in the shared `.pre-commit-config.yaml` or CI:**
cpc is a private tooling repo, this repo is public (ADR-001).
Install (no clash with the main config's pre-commit stage):
`pre-commit install -c .pre-commit-config.cpc.yaml -t pre-push -t commit-msg` — docs/test-api
checks + `cpc-push-guard` at pre-push, `cpc-coupling-check` at commit-msg. **The big-project
layout is adopted (ADR-021, 2026-07-19):** `AGENTS.md` is the canonical entry file (`CLAUDE.md` a
bare `@AGENTS.md` stub, `[entry] enforce_stub = true`), module `CLAUDE.md` files cover
`src/doc_assistant/` · `apps/desktop/` · `apps/api/` · `scripts/` (≤40 lines each), and
`GLOSSARY.md` is laid + filled. `cpc-init-check` therefore now **passes** — still run it on-call,
not as a hook: `python tools/conventions/rungate.py init_check --root . --profile standard`.
On-call any time:
`python tools/conventions/rungate.py docs_check --root . --strict`;
`python tools/conventions/rungate.py keypoint <session-start|plan-start|sprint-start|sprint-close|session-close>`
(gate battery + judgment checklist per workflow boundary, cpc ADR-020; per-project extras go in
`[keypoints.<name>]` in `scripts/conventions.toml` — none registered yet). `just check` / `just
lint` / `just keypoint <name>` alias these (facade only). Baton hygiene is gate-read
(rule 11): newest-on-top, cap 10 entries (`scripts/conventions.toml`), rotate older entries verbatim
to docs/archive/SESSION-archive-NNN.md (local-only, like the baton).

## Phase map

| Phase | Content | Status |
|---|---|---|
| 1–3 | Core RAG · measured quality + eval harness · document store + library UI | done |
| 4 | Citation graph + doc-similarity edges | done |
| 5 | Embedding & eval foundation (config-driven embedder, golden set, provenance) | done |
| 6 | Figures & tables, dual-layer interpretation, reviewer + self-improvement loop | in progress |
| 7 | Gap detection — wiki/synthesis layer + cross-document concept graph (incl. 7d engine) | in progress |
| 8 | UI polish (settings page exposing the RAG sandbox knobs) + ongoing chat/citation UX | **open — iterative UI track** (not closed): U2/U3/U1/U1b/U1c built & committed (`09afd0c`); end-to-end verification still owed + more UI elements planned. Living status: `docs/ui-checklist.md` |
| 9 | Literature-review generation (PRISMA-trAIce export) | planned |
| — | Extract eval harness to a standalone repo (Feature 5) | planned |

## Open questions

- **`CANDIDATE_K=20` verdict is unvalidated.** The 10-paper public corpus (one-paper-per-topic) is
  too small to rank `CANDIDATE_K`; retest on the larger private set (`cases.yaml`, multi-paper-per-topic)
  with `--repeat` before locking it as a measured win. `CANDIDATE_K=10` reproduces pre-split behaviour.
- **Per-project embedder routing (Feature 1b) deferred** — no model beats `bge-base` on an identifiable
  sub-corpus yet (SPECTER2 lost on every retrieval signal). Re-run SPECTER2 `--repeat 5` first.
- **Graph vocabulary is now SCOPED (2026-07-18, ADR-018, staged).** Tag families (ADR-015) and concept-graph
  nodes are the same `Concept` rows; the graph filters on an additive **opt-in `graph_include`** flag while
  `library.list_keyword_families()` deliberately does not. Root cause of the flood: **all 344**
  `source="keyword"` concepts were created by **one `seed_concepts.py --promote-all` on 2026-07-05**.
  Real corpus (CPU box, $0 Ollama): graph **357 → 13 nodes / 19 edges / 15 gaps**, families still 357.
  **Curation is CLI-only** (`add_concept`, `set_graph_include`, `scripts/backfill_graph_include.py`) — no UI yet.
- **⚠ TWO BOXES, TWO CORPORA.** This box's `data/library.db` carries **47 docs**; ADR-017 / PR-G1 / PR-G2a /
  the L5 row were measured on a **76-doc** corpus elsewhere, and none of their cited concept labels
  (`Res2Net`/`PHATE`/`SBERT`/`Embeddings`) exist here. **Never assume a spec's "live-verified" number
  reproduces on the box in front of you — re-measure.**
- **Concept graph redesign (2026-06-18) — RESOLVED, fully built.** Node A (deterministic skeleton,
  2026-06-30) + Node B (confined LLM stance, PR #6) both shipped; RG-001/008/009 validated the edges
  (R5 PASS, ADR-008, `CONCEPT_SKELETON_MIN_COOCCURRENCE=2` + `boundary` presence); the superseded
  open-vocabulary `concept_graph.py` is deleted (2026-07-07, KI-7 resolved). **Resolved
  2026-07-08 (G3, code-built + host-applied):** the year-aware pass is deterministic, not Node-B/LLM —
  `node_weights_for_epistemics` produces `superseded_trend`, gated (G6) on >= 2 dated docs per side.
  **G7 (2026-07-08) fixed the id/label mismatch (KI-15)** that kept this from reaching a chunk at
  answer time — `epistemics.concepts_in_text` now matches on label; real corpus went from 0 to
  4008 marked-eligible chunks. KI-15 RESOLVED.
- **Gap-detection layer (2026-06-26) — Tier 1 + Tier-2a floor BUILT (2026-07-07, G2); the Tier-2a
  stochastic ceiling BUILT (2026-07-08, G5).** Deterministic detectors
  (`isolated`/`single_source`/`thin_bridge`/`under_connected`) + the `unsourced_claim` floor are live
  (`gaps.py` + `GapRow` + `scripts/build_gaps.py`); `citation_missing` (the other floor kind) is not
  yet built. `gap_suggest.py` (Ollama-default, quarantined, apply-gated via
  `llm.assert_provider_intent`) adds one LLM suggestion per `under_connected` concept — real-validated
  on this RTX/Ollama box (12/12 suggested, $0, `tests/eval/baselines/gap_suggest_ollama_2026-07-08.md`);
  llama3.1:8b's `rating` output is flat (~0.8, not discriminating) and it only ever chose
  `suggested_concept` in that run — a local-model calibration ceiling, not a code defect. Still
  deferred: Tier 2b (external "anti-blind-spot" reach — the idea-generator is rejected for it, ADR-004
  option 3).
- **BM25/vector `0.4/0.6` weights — MEASURED 2026-07-03 (resolved, kept).** `--bm25-weight` flag
  (config `BM25_WEIGHT` → `pipeline`/`run_eval`) + `scripts/sweep_bm25_weight.py` added; swept
  `{0.0..1.0}` retrieval-only over `cases.yaml`. Post-rerank recall is **flat across the whole
  range** — LangChain's `EnsembleRetriever` hands the cross-encoder the *full* candidate union, so
  the weight is structurally inert on the shipped top-K (only pre-rerank ordering moves; the
  instrument discriminates via `pre@5`). Kept `0.4/0.6` (negative result). The weight would only
  become live if the candidate pool were truncated pre-rerank or the reranker ablated. Baseline:
  `tests/eval/baselines/bm25_weight_sweep_2026-07-03.md`.
