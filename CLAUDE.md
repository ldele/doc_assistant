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

Always check `decisions.md` before suggesting architectural changes. Most non-obvious decisions are already there with the reasoning.

---

## Current Status

**Active phase:** Phase 6 in progress — Integrity Chunk 2b (reviewer agent) shipped 2026-05-28. Chunk 2a (dual interpretation) and Feature 4a (pdfplumber tables) remain.

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

**Snapshot:** 264 tests · 63.9% coverage · ruff format/check + mypy + bandit clean.

**Since the PR table (2026-05-31 → 06-01):**
- **Chunking sweep infra** (reopened Phase 2.4): config-driven `*_CHUNK_SIZE`, `scripts/sweep_chunking.py`, guard tests. Measurement run still pending.
- **Public reproducibility corpus**: 10 arXiv papers behind the project's own methods (RAG, dense retrieval, SBERT, BGE, SPECTER2/SciRepEval, BERT re-ranking, ColBERT, HyDE, LLM-as-judge, AI Usage Cards). Download-only via `scripts/download_corpus.py` + `tests/eval/corpus_manifest.yaml` (nothing re-hosted; arXiv license safe). Standalone `tests/eval/cases.public.yaml` (10 cases). Separate from the private, mostly-copyrighted neuroscience benchmark (`cases.yaml`).
- **Public eval baseline** (bge-base, n=5): `citation_overlap` 1.000 ± 0.000, `contains_all` 0.927 ± 0.034, `llm_judge` 3.894 ± 0.075. Recorded in `tests/eval/baselines/public_eval_baseline_2026-06-01.md`.
- **`data/eval.duckdb` now gitignored** (live run log, regenerated on run; committed reference results live in `tests/eval/baselines/`).

**Operational TODOs (don't need code changes):**
- Re-run SPECTER2 at `--repeat 5` for the symmetric BGE-vs-SPECTER2 confidence-interval comparison.
- Most `expected_answer` fields in `tests/eval/cases.yaml` are best-effort (`author_verified: false`) — refine over time.
- LNCS colon-separator format + multi-column PDF extraction: known tier-1 citation-extractor weaknesses; cosmetic, deferred.

**Next priority:** three independent, ready nodes — **LLM provider protocol** (normalized `complete()` + Anthropic/Ollama adapters; makes the app fully local; spec'd in `docs/specs/llm-provider-isolation.md`, no deps), **Chunk 2a** (Dual Interpretation, biggest UX shift; `SYNTHESIS_MODE` flag, evidence vs interpretation layers), or **Feature 4a** (pdfplumber tables, smallest). File sets are disjoint except `pipeline.py`/`config.py`.

---

## For Claude Code

Execution is happening in Claude Code. Source of truth for what to build, in what order:

1. `docs/doc-assistant-roadmap.md` — Implementation order (PR-by-PR) table at the bottom. Each PR is scoped to one chunk with file lists and decision-doc references.
2. `docs/decisions.md` — locked architectural choices. Every Phase 5+ feature has a subsection. Read the relevant section before starting a PR.
3. `docs/DEVLOG.md` — append-only log. Add one entry per logical change per project rule.
4. `docs/specs/` — ready, code-level build specs (file lists, contracts, guard tests) for individual nodes. Currently: `llm-provider-isolation.md`.

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
├── eval/                 # Phase 5 / Feature 2: generic eval harness
│   ├── cases.py          # YAML loader
│   ├── results.py        # dataclasses
│   ├── scorers.py        # 5 scorer implementations
│   ├── runner.py         # exception-tolerant loop
│   ├── store.py          # DuckDB persistence + aggregation
│   ├── report.py         # summary + diff + aggregate
│   └── adapters.py       # the only doc_assistant-aware file
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
| Chunk sizes | parent 2000/200, child 400/50, baseline 1000/200 | Config-driven since 2026-05-31 (`*_CHUNK_SIZE` env vars). **Defaults are historical, never measured** — sweep with `scripts/sweep_chunking.py` before trusting as optimal |

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
| 5 — Embedding & Eval Foundation | 🔄 In progress | Feature 1 ✅; eval harness + provenance card next |
| 6 — Per-project routing + Figures & Tables + Dual interpretation | 🔄 In progress | Chunk 2b (reviewer agent) ✅; Chunk 2a + Feature 4a next |
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

- **Python 3.14 + Chainlit incompatible.** anyio 4.13.0 + starlette breaks Chainlit's static file serving on 3.14 (`NoEventLoopError`). Workaround: `uv run --python 3.12 chainlit run apps/chainlit_app.py`. Dev tools (pytest, ruff, mypy) work fine on 3.14.
- Section detection regex may misclassify some PDF formats as marginal health — known, non-blocking.
- **`reference_flagged_ratio` health signal not wired.** Defined in `health.py`/`models.py` but `ingest.py` hardcodes `0.0`. Phase 4 citation extractor populates the data needed to wire it; pending integration into the health score.
- **Sandbox file-sync issue (recurring).** Edit-tool writes to Windows side sometimes fail to fully sync to the bash sandbox view, causing partial files and stale `.pyc` bytecode. Workarounds: `touch` to force re-read; full rewrites via bash heredocs.
- **pip-audit:** 28 CVEs in transitive deps (torch, transformers, ollama, joblib, pyjwt, langchain-community). All upstream, none in our code. CI set to `continue-on-error`.
- **Flaky LLM-judge call on `sbert_motivation`** (public eval). Skipped ~3/5 trials (API timeout / JSON parse failure on that prompt); `llm_judge` mean is over scored trials, not all 50. Non-blocking; inspect the judge JSON-parse path if it persists.

### Resolved
- ~~Path+content hash drift~~ → content-only hashing implemented and migrated (27 docs).
- ~~`dedupe_documents.py` failed for `cajal-lecture.pdf`~~ → root cause eliminated by content-only hashing.
- ~~CLAUDE.md said "Next priority: build citations.py"~~ → built; status now reflects actual Phase 4 close-out work.
