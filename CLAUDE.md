# CLAUDE.md — doc_assistant

Read this at the start of every session. Update when decisions are made or status changes.

---

## What This Is

A local-first personal research assistant. Ingests PDF/EPUB/HTML/DOCX/MD documents, builds a hybrid RAG pipeline (BM25 + vector + cross-encoder reranker), and answers questions with inline citations. Designed for academic/research document libraries.

Not a general-purpose chatbot. The goal is: reliable answers grounded in *your* documents, with measurable quality.

---

## Key documents — read these

- **[`docs/architecture.md`](docs/architecture.md)** — pipeline flow, module responsibilities, public contracts, engineering standards (CI, security, logging, errors, testing layout)
- **[`docs/decisions.md`](docs/decisions.md)** — all design decisions with rationale, full phase roadmap, deferred improvements, open questions
- **[`docs/doc-assistant-roadmap.md`](docs/doc-assistant-roadmap.md)** — source of intent for Phase 5+ additions (embedding layer, eval harness, figures/tables, research integrity layer) and the Claude Code PR execution order
- **[`docs/figures-and-tables.md`](docs/figures-and-tables.md)** — Feature 4 detection layer: the page content classifier (`regions.py`), caption + curve-density + image-area signals, table extraction/splice, tooling, and the pdfplumber/Marker decision

Always check `decisions.md` before suggesting architectural changes. Most non-obvious decisions are already there with the reasoning.

---

## Current Status

**Active phase:** Phase 6 in progress — reviewer agent (2026-05-28) and LLM provider protocol (2026-06-02) shipped. Feature 4 detection foundation in progress: `regions.py` classifies each page (table/chart/photo/figure/text) from caption + curve-density + image-area signals, and `tables.py` extracts only on classified table pages (fixes figure-as-table at the root; gives 4b its figure signal). **Engine decision MADE (2026-06-02, RTX eval):** **Marker wins; default pdfplumber dropped as primary.** Measured on the arXiv corpus — pdfplumber recall on borderless/booktabs tables is unreliable (DPR 0/6, SPECTER2 0/6; SBERT extracts but collapses rows), while Marker reproduced all 7 DPR tables faithfully (columns, multi-row cells, bold). **Constraint:** marker-pdf can't co-resolve with our langchain/transformers/torch stack, so it must run **out-of-process in an isolated env** (`uvx marker_single`), gated to `regions.table_candidate_pages` via `--page_range`, as a sidecar enrichment step — never an in-process import. **Open (the 4a build):** wire that isolated-Marker ingest path + parse/splice its markdown tables + inline de-dup of pymupdf4llm's lossy tables + a table-retrieval eval-hook. (✅ `eval_marker_tables.py` fixed 2026-06-04 — now shells out to isolated `marker_single` with 1-based→0-based page-range mapping; this was 4a step 1.) Region-level (multi-region-per-page) splitting is the proper 4b build. Chunk 2a (dual interpretation) is the other main remaining node.

**Shipped this session (per-PR detail in `docs/DEVLOG.md`):**

| PR | What |
|---|---|
| 1 | Phase 4 close-out: doc-level mean-pool vectors → top-K cosine similarity edges, `/similar` |
| 1.5 | `--path` scoped ingest, duplicate detector, BibTeX exporter (`/bibtex`) |
| 2 | Config-driven embedding layer (`EMBEDDING_MODEL` env var, factory, per-model Chroma collections) |
| 3 | Eval harness v0 (generic core + adapters; 5 scorers; DuckDB store) |
| 3.1 | `scoreable` column + `scorer_stats` to distinguish "scored zero" from "didn't run" |
| 4 | 35-case eval set; hardened LLM judge (temp=0, reference-only); BGE > SPECTER2 measured |
| 4.1 | `--repeat N` for variance; methodology rigor notes on legacy measurements; trajectory section |
| 4.2 | `trial_mean_std` vs `score_std` in aggregate; flaky-case detection |
| 5 | Provenance card (Integrity Chunk 1): `AnswerRecord` table; `/records`, `/export-record` |
| 5.1 | Heuristic confidence signals — quiet UI on clean answers, ⚠ on flagged |
| 6 | Reviewer agent (Integrity Chunk 2b): `/review`, runs only on flagged answers |

**Snapshot:** 318 tests green (unit+integration; +2 regression tests this session) · ruff format/check + mypy --strict + bandit clean · **CUDA/GPU enabled on the RTX box** (Windows; embeddings + reranker on `cuda:0`, retrieve+rerank ~68 ms).

**LLM provider protocol (2026-06-02, committed `37dcbdc` — folded into Feature 1, generation side):**
- `src/doc_assistant/llm.py` — `LLMClient.complete()` protocol + `AnthropicClient`/`OllamaClient` adapters + `make_client`/`get_reviewer_client`/`get_judge_client`/`reviewer_available`. The one-shot path (reviewer + eval judge) is now provider-agnostic; the streaming analysis path stays LangChain but reads `LLM_PROVIDER`/`LLM_MODEL`.
- `/review` and auto-review gate on `reviewer_available()` (Ollama needs no key), not a bare `ANTHROPIC_API_KEY` check — so review works fully local.
- Spec: `docs/specs/llm-provider-isolation.md`. ✅ **Generator path verified live on a real Ollama server (RTX box, 2026-06-02)** — fully-local query end-to-end on `llama3.1:8b`, `ANTHROPIC_API_KEY` empty, grounded + correctly-cited answers. ⚠ Reviewer (`/review`) + eval-judge on Ollama still unverified live — see operational TODOs.

**Since the PR table (2026-05-31 → 06-01):**
- **Chunking sweep infra** (reopened Phase 2.4): config-driven `*_CHUNK_SIZE`, `scripts/sweep_chunking.py`, guard tests. Measurement run still pending.
- **Public reproducibility corpus**: 10 arXiv papers behind the project's own methods (RAG, dense retrieval, SBERT, BGE, SPECTER2/SciRepEval, BERT re-ranking, ColBERT, HyDE, LLM-as-judge, AI Usage Cards). Download-only via `scripts/download_corpus.py` + `tests/eval/corpus_manifest.yaml` (nothing re-hosted; arXiv license safe). Standalone `tests/eval/cases.public.yaml` (10 cases). Separate from the private, mostly-copyrighted neuroscience benchmark (`cases.yaml`).
- **Public eval baseline** (bge-base, n=5): `citation_overlap` 1.000 ± 0.000, `contains_all` 0.927 ± 0.034, `llm_judge` 3.894 ± 0.075. Recorded in `tests/eval/baselines/public_eval_baseline_2026-06-01.md`.
- **`data/eval.duckdb` now gitignored** (live run log, regenerated on run; committed reference results live in `tests/eval/baselines/`).

**Benchmarks locked (2026-06-04, primary CPU box):**
- **Blocker found + fixed:** the locked `cu130` torch build **segfaults on model inference on the CPU box** (no GPU; correct only on the RTX box). Unblocked locally with a forced CPU torch install + **`uv run --no-sync`** — committed lock untouched, RTX box stays on GPU. See Known issues + memory `cpu-box-torch-cu130-segfault`. Proper fix (GPU as an opt-in uv extra) deferred.
- **Public baseline reproduced** (bge, n=5): `citation_overlap` 1.000 ± 0.000, `contains_all` 0.927 ± 0.027, `llm_judge` 3.738 ± 0.093 — matches the 2026-06-01 reference (judge within noise; same `sbert_motivation` flaky case). Recorded in `public_eval_baseline_2026-06-01.md`. **Locked.**
- **BGE > SPECTER2, symmetric on the public corpus** (n=5): specter2's collection was stale (0 of the 10 public papers); an incremental `EMBEDDING_MODEL=specter2 ingest --skip-cleanup` added just those 10 (both now 61 docs / 27168 chunks). BGE wins all three scorers (citation 1.000 vs 0.900, contains_all 0.927 vs 0.800, judge 3.738 vs 3.447). `bge-base` stays default. Recorded in `bge_vs_specter2_public_2026-06-04.md`. **Locked.**
- **Chunking sweep: deferred to the RTX box.** Each config's re-embed is ~45 min on CPU (×6 ≈ 5h); minutes on GPU. Config 1 (control = current defaults) run on CPU as a preview; configs 2–6 → RTX. One-command resume + restore steps in [`docs/chunking-sweep-rtx-resume.md`](docs/chunking-sweep-rtx-resume.md). Sweep `--cases` passthrough added so it runs on the public verified-10 set.

**Operational TODOs (don't need code changes):**
- **Fully-local Ollama path — generator + reviewer verified live on the RTX (GPU) box (2026-06-02).** ✅ **Generator**: ingested the 10-paper public corpus and ran real queries end-to-end on `llama3.1:8b`, `ANTHROPIC_API_KEY` empty — grounded, correctly-cited answers. ✅ **Reviewer**: `/review` returns a real verdict on Ollama (live run: faithfulness 5 · citation 3 · hedging 4 · unsupported 0). Two transport bugs found and fixed getting there — `OllamaClient` passed `temperature` as an `invoke` kwarg (`TypeError` from `Client.chat()`), and the reviewer choked on llama's non-JSON output (now `format="json"` + tolerant `_extract_json`); both have regression tests. ⚠ Remaining for the DoD bullet: the **eval-judge** on Ollama (`tests/eval/TESTING.md` calibration gate). Pick the local model(s) via `LLM_MODEL`/`REVIEWER_MODEL`.
- ✅ **CUDA/GPU enabled on the RTX box (2026-06-02).** torch now resolves to `2.12.0+cu130` on Windows via a `[[tool.uv.index]]` + win32-scoped `[tool.uv.sources]` in `pyproject.toml`; embeddings + reranker run on the RTX 4070 (`cuda:0`), retrieve+rerank ~68 ms (was CPU-seconds). No app-code change — sentence-transformers auto-detects CUDA. The `sys_platform == 'win32'` marker keeps Linux/CI on the default CPU wheel. (Watch the next CI run to confirm the lock resolves cleanly on Linux.)
- ✅ **SPECTER2 re-run at `--repeat 5` done (2026-06-04).** Symmetric BGE-vs-SPECTER2 on the public corpus — BGE wins all scorers (see Benchmarks-locked block above). bge-base stays default.
- **Chunking sweep configs 2–6: run on the RTX box.** See [`docs/chunking-sweep-rtx-resume.md`](docs/chunking-sweep-rtx-resume.md) (one command + the destructive-rebuild restore step). Then record the winner and update the Locked-settings chunk-sizes row.
- Most `expected_answer` fields in `tests/eval/cases.yaml` are best-effort (`author_verified: false`) — refine over time.
- LNCS colon-separator format + multi-column PDF extraction: known tier-1 citation-extractor weaknesses; cosmetic, deferred.

**Next priority (RESUME HERE — 2026-06-04):** benchmarks locked except the chunking sweep (deferred to RTX). Two open threads:

**A. Chunking sweep on the RTX box** — configs 2–6, one command + restore step in [`docs/chunking-sweep-rtx-resume.md`](docs/chunking-sweep-rtx-resume.md). Then record the winner / confirm defaults and update the Locked-settings chunk-sizes row.

**B. Feature 4a build** — step 1 (`eval_marker_tables.py` → isolated `marker_single`) is ✅ done (2026-06-04). Remaining, in order:
1. Ingest path: gate on `regions.table_candidate_pages` → shell out to isolated `uvx --from marker-pdf marker_single --page_range <those pages>` → parse Marker's markdown tables → splice into the `.md` cache (Enrichment-Layer pattern, mirrors `extract_tables.py`).
2. Inline de-dup of pymupdf4llm's lossy inline tables.
3. Table-retrieval eval-hook.

**Lighter alternatives** if not starting the build: a **demo dry-run** of the 7-question script (in Notion: "Demo Script — doc_assistant", under the doc-assistant page) — verify Q7 actually trips the ⚠ low-confidence path; or the **measurement runs** that are cheap now with GPU (chunking sweep, SPECTER2 `--repeat 5`). A working-app **demo is upcoming** (runs on Anthropic API; `.env` already configured — key present locally, gitignored). **Chunk 2a** (Dual Interpretation; `SYNTHESIS_MODE`, evidence vs interpretation) remains the other main Phase 6 node — needs a code-level spec first. Feature 4b (figure detection / region-level splitting) is unblocked once the 4a build lands.

---

## For Claude Code

Execution is happening in Claude Code. Source of truth for what to build, in what order:

1. `docs/doc-assistant-roadmap.md` — Implementation order (PR-by-PR) table at the bottom. Each PR is scoped to one chunk with file lists and decision-doc references.
2. `docs/decisions.md` — locked architectural choices. Every Phase 5+ feature has a subsection. Read the relevant section before starting a PR.
3. `docs/DEVLOG.md` — append-only log. Add one entry per logical change per project rule.
4. `docs/specs/` — ready, code-level build specs (file lists, contracts, guard tests) for individual nodes. Currently: `llm-provider-isolation.md`; `feature-4a-marker-table-ingest.md` (Marker table ingest, grilled 2026-06-04); `chunk-2a-dual-interpretation.md` (dual evidence/interpretation layer, grilled 2026-06-04).

PR-by-PR cadence. Do not bundle multiple PRs. Do not start a PR without reading its `decisions.md` dependency.

---

## Setup

```bash
# Prerequisites: Python 3.12, uv
git clone <repo>
cd doc_assistant
uv sync
cp .env.example .env   # fill in API keys
uv run python -m doc_assistant.ingest
uv run chainlit run apps/chainlit_app.py   # web UI
# or
uv run python apps/cli.py                  # terminal
```

---

## Project structure

```
src/doc_assistant/
├── config.py             # paths, env vars, feature flags
├── extractors.py         # PDF/EPUB/HTML/DOCX/MD → markdown
├── ingest.py             # streaming ingest: extract, chunk, embed, store
├── pipeline.py           # RAG runtime: retrieve, rerank, generate
├── health.py             # document health scoring (healthy/marginal/broken)
├── library.py            # SQLite document store queries (UI-agnostic)
├── query_router.py       # library-vs-content query detection + metadata responses
├── commands.py           # slash-command parsing, execution, markdown formatters
├── prompts.py            # prompt templates
├── tracking.py           # token usage + cost estimation
├── citations.py          # Phase 4: tier-1 regex citation extractor
├── metadata_extractor.py # Phase 4: doc-level metadata (title/authors/year/DOI)
├── doc_vectors.py        # Phase 4: mean-pool doc vectors + similarity edges
├── bibtex.py             # PR 1.5: BibTeX export from Document rows
├── embeddings.py         # Phase 5 / Feature 1: model registry + factory
├── llm.py                # Phase 6 / Feature 1 (gen side): LLMClient protocol + Anthropic/Ollama adapters (one-shot complete())
├── eval/                 # Phase 5 / Feature 2: generic eval harness
│   ├── cases.py          # YAML loader
│   ├── results.py        # dataclasses
│   ├── scorers.py        # 5 scorer implementations
│   ├── runner.py         # exception-tolerant loop
│   ├── store.py          # DuckDB persistence + aggregation
│   ├── report.py         # summary + diff + aggregate
│   └── adapters.py       # the only doc_assistant-aware file
├── regions.py            # Phase 6 / Feature 4: page content classifier (table/chart/photo/figure/text) — shared detection layer
├── tables.py             # Phase 6 / Feature 4a: pdfplumber table extraction on classified table pages, spliced into the markdown cache
├── provenance.py         # Phase 5 / Integrity Chunk 1: per-answer audit record + confidence signals
├── reviewer.py           # Phase 6 / Integrity Chunk 2b: LLM reviewer agent (faithfulness/citation/hedging rubric)
└── db/
    ├── models.py         # SQLAlchemy ORM (Document, Tag, Folder, Citation, ...)
    ├── session.py        # engine + session_scope() context manager
    └── migrations.py     # schema versioning

apps/
├── chainlit_app.py       # web UI (thin shell — lifecycle hooks + streaming only)
└── cli.py                # terminal (thin entrypoint, no business logic)

docs/
├── architecture.md           # ← read this
├── decisions.md              # ← read this
├── doc-assistant-roadmap.md  # ← read this (Phase 5+ source of intent + PR order)
└── DEVLOG.md                 # append-only development log

scripts/
├── extract_citations.py      # Phase 4 CLI: extract refs, persist Citation rows
├── extract_doc_metadata.py   # Phase 4 CLI: backfill title/authors/year/doi
├── compute_doc_vectors.py    # Phase 4 CLI: mean-pool vectors, write similarity edges
├── find_duplicates.py        # PR 1.5: report byte-identical and content-identical source files
├── export_bibtex.py          # PR 1.5: write docs/library.bib from Document rows
├── run_eval.py               # PR 3: drive the eval harness over the RAG pipeline
├── extract_tables.py         # PR 7 / Feature 4a: caption-gated tables → spliced into markdown cache
├── debug_tables.py           # PR 7 / Feature 4a: visual table-detection debug (overlay PNGs; dev tool)
├── eval_marker_tables.py     # PR 7 / Feature 4a: Marker-vs-pdfplumber engine eval (RTX machine)
└── migrate_to_content_hash.py # one-shot migration (Phase 3, completed)

tests/
├── unit/                 # fast, no I/O
├── integration/          # cross-module, mocked LLM
└── eval/                 # RAG evaluation harness (manual runs at phase checkpoints; cases.yaml lands in Phase 5)
```

---

## Architecture in one paragraph

Documents are extracted to markdown (two-tier cache: extraction + embeddings), chunked with parent-child retrieval (child ~400 chars embedded, parent ~2000 chars sent to LLM), stored in Chroma + SQLite. At query time: hybrid retrieval (BM25 0.4 + vector 0.6) returns TOP_K=10 candidates → cross-encoder reranker → top 5 parent passages → LLM with citations. Parent-child and TOP_K=10 are empirically validated (see `decisions.md`).

Post-ingest enrichment layers (citations, metadata, figures/tables coming in Phase 6) run as separate modules + CLI runners and write to sidecar tables or splice into the markdown cache. They never mutate the primary chunk store. See **Enrichment-Layer Pattern** in `decisions.md`.

---

## Locked settings (do not change without an experiment)

| Setting | Value | Rationale |
|---|---|---|
| TOP_K | 10 | Correctness 4.55, Faithfulness 4.74 — peak on both |
| BM25/vector weights | 0.4 / 0.6 | Best hit rate on technical docs |
| Parent-child retrieval | enabled | +0.62 correctness vs single-chunk baseline |
| Multi-query expansion | disabled | Dilutes reranker; tested twice, regressed both times |
| Embedding model | BGE-base-en-v1.5 | Stable; Phase 5 makes it swappable; SPECTER2 comparison gates Phase 6 routing |
| Chunk sizes | parent 2000/200, child 400/50, baseline 1000/200 | Config-driven since 2026-05-31 (`*_CHUNK_SIZE` env vars). **Defaults still not fully measured** — sweep (`scripts/sweep_chunking.py`) deferred to the RTX box (configs 2–6); control config previewed on CPU 2026-06-04. See [`docs/chunking-sweep-rtx-resume.md`](docs/chunking-sweep-rtx-resume.md) |

---

## LLM provider configuration (user's choice — not locked)

The model is the user's call; these are just defaults that reproduce historical behaviour when unset. All selectable in `.env` (see `.env.example`):

| Call shape | Provider var | Model var | Default (provider / model) |
|---|---|---|---|
| Analysis / chat (streaming) | `LLM_PROVIDER` | `LLM_MODEL` | from `LLM_MODE` / `haiku` (anthropic) or `llama3` (ollama) |
| Reviewer agent (`/review`, auto-review) | `REVIEWER_PROVIDER` | `REVIEWER_MODEL` | = `LLM_PROVIDER` / `haiku` |
| Eval LLM-judge | `JUDGE_PROVIDER` | `JUDGE_MODEL` | = `LLM_PROVIDER` / `haiku` |

- Provider is `anthropic` or `ollama`. Adding a backend = one adapter in `src/doc_assistant/llm.py`.
- **Fully local:** set all three providers to `ollama`, pick local models, leave `ANTHROPIC_API_KEY` empty. `.env.example` has a ready-to-uncomment block. Reviewer/judge defaults stay on the pinned reference model for comparable eval numbers, but you can point them at a local model too (gate: `tests/eval/TESTING.md`).
- **Not yet verified on a live Ollama server** — see the RTX re-test in operational TODOs.

---

## Engineering standards (non-negotiable)

- No secrets in code. `.env` gitignored. `.env.example` committed.
- `bandit` HIGH findings block merge.
- CI must be green before merge.
- Coverage floor: 45% (CI-enforced). Raise incrementally as integration tests are added.
- Structured JSON logging in staging/production (`structlog`). No `print()` in `src/`.
- All exceptions chain with `raise X from e`. User-facing messages translated at UI boundary.
- `apps/` contains no business logic. All logic in `src/doc_assistant/`.
- Maintain `docs/DEVLOG.md` — append one entry per logical change (what / why / rejected / opens). See dev-log skill for format.
- **Enrichment-Layer Pattern** — any new derived data (figures, tables, vectors, future enrichments) ships as a separate module + CLI runner. Idempotent. Sidecar by default. Never mutate the primary chunk store from an enrichment module.

Full standards in `docs/architecture.md` → Engineering standards section.

---

## Phase roadmap summary

| Phase | Status | Goal |
|---|---|---|
| 1 — Core RAG | ✅ Complete | End-to-end working RAG |
| 2 — Quality Foundation | ✅ Complete | Measurable quality, experiments |
| 3 — Document Store + Library UI | ✅ Complete | Library as first-class object |
| 4 — Citation Graph | ✅ Complete | Explicit + implicit document relationships |
| 5 — Embedding & Eval Foundation | ✅ Complete | Embedding factory, eval harness, golden set (BGE > SPECTER2), provenance card all shipped |
| 6 — Per-project routing + Figures & Tables + Dual interpretation | 🔄 In progress | Reviewer agent ✅ + LLM provider protocol ✅; Feature 4 detection layer (`regions.py`) in progress (engine eval pending); Chunk 2a + figure extraction (4b) next |
| 7 — Gap Detection | ⬜ | What the library knows vs the field |
| 8 — UI Polish | ⬜ | Design pass |
| 9 — Literature Review Generation | ⬜ | End-game synthesis + PRISMA-trAIce export |

Full roadmap with sub-tasks and deferred improvements in `docs/decisions.md`. PR-by-PR execution order in `docs/doc-assistant-roadmap.md`.

---

## Open questions (unresolved)

- UI framework for Phases 6–9: Chainlit will hit limits at graph visualization and at the adjudication UI for Chunk 2a. Candidates: Reflex, FastAPI + custom frontend, Streamlit. Decision point: Phase 8.
- Embedding model upgrade: BGE-base-en-v1.5 vs SPECTER2 measurement is the gate for Phase 6 per-project routing.
- Multi-user support: single-user architecture; multi-user needs a DB redesign.
- Tier-2 LLM citation fallback: deferred until corpus grows or no-section docs become problematic.

---

## Known issues

- **Locked `cu130` torch segfaults on the CPU dev box.** The lock pins `torch==2.12.0+cu130` for *all* win32 machines (`sys_platform=='win32'` source) — right for the RTX box, but on the GPU-less CPU box `torch.cuda.is_available()` is False and the **transformer forward pass segfaults** (exit 139) on any eval/sweep. Workaround (committed lock untouched): `uv pip install "torch==2.12.0+cpu" --index-url https://download.pytorch.org/whl/cpu --reinstall-package torch`, then run everything with **`uv run --no-sync`** (a bare `uv run` re-syncs and reverts to `+cu130`). Proper cross-machine fix (GPU as an opt-in uv extra, CPU default everywhere) deferred. Memory: `cpu-box-torch-cu130-segfault`.
- **Python 3.14 + Chainlit incompatible.** anyio 4.13.0 + starlette breaks Chainlit's static file serving on 3.14 (`NoEventLoopError`). Workaround: `uv run --python 3.12 chainlit run apps/chainlit_app.py`. Dev tools (pytest, ruff, mypy) work fine on 3.14.
- Section detection regex may misclassify some PDF formats as marginal health — known, non-blocking.
- **`reference_flagged_ratio` health signal not wired.** Defined in `health.py`/`models.py` but `ingest.py` hardcodes `0.0`. Phase 4 citation extractor populates the data needed to wire it; pending integration into the health score.
- **Sandbox file-sync issue (recurring).** Edit-tool writes to Windows side sometimes fail to fully sync to the bash sandbox view, causing partial files and stale `.pyc` bytecode. Workarounds: `touch` to force re-read; full rewrites via bash heredocs.
- **pip-audit:** 28 CVEs in transitive deps (torch, transformers, ollama, joblib, pyjwt, langchain-community). All upstream, none in our code. CI set to `continue-on-error`.
- **Flaky LLM-judge call on `sbert_motivation`** (public eval). Skipped ~3/5 trials (API timeout / JSON parse failure on that prompt); `llm_judge` mean is over scored trials, not all 50. Non-blocking. **Confirmed still flaky on the 2026-06-04 bge re-run** (scored 2/5) — if it keeps recurring, inspect the judge JSON-parse/timeout path for that specific prompt.
- **Token/cost tracking is provider-dependent.** `TokenCounter` hooks Anthropic-style callbacks, so real token/cost numbers appear only on the Anthropic path. The Ollama path reports no usage; rather than the old misleading `0 in / 0 out / $0.0000`, the UI now shows an honest "local model — no metered cost (~N output tokens, estimated)" note (chainlit_app, 2026-06-02). True local token counts would need an Ollama-native usage hook.

### Resolved
- ~~App hard-crashed on a fresh install with an empty library~~ → `pipeline.py` falls back to a vector-only ensemble when there are 0 retrievable chunks (2026-06-02).
- ~~First ingest on a fresh clone died with `no such table: documents`~~ → `ingest.main()` now calls `init_db()` idempotently at startup; the documented `uv sync → ingest → run` flow works as-written (2026-06-02).
- ~~Path+content hash drift~~ → content-only hashing implemented and migrated (27 docs).
- ~~`dedupe_documents.py` failed for `cajal-lecture.pdf`~~ → root cause eliminated by content-only hashing.
- ~~CLAUDE.md said "Next priority: build citations.py"~~ → built; status now reflects actual Phase 4 close-out work.
