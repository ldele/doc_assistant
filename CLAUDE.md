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

**Active phase:** Phase 6 in progress — reviewer agent (2026-05-28), LLM provider protocol (2026-06-02), and **Chunk 2a dual-interpretation (2026-06-04, merged PR #1)** shipped. Feature 4 detection foundation in progress: `regions.py` classifies each page (table/chart/photo/figure/text) from caption + curve-density + image-area signals, and `tables.py` extracts only on classified table pages (fixes figure-as-table at the root; gives 4b its figure signal). **Engine decision MADE (2026-06-02, RTX eval):** **Marker wins; default pdfplumber dropped as primary.** Measured on the arXiv corpus — pdfplumber recall on borderless/booktabs tables is unreliable (DPR 0/6, SPECTER2 0/6; SBERT extracts but collapses rows), while Marker reproduced all 7 DPR tables faithfully (columns, multi-row cells, bold). **Constraint:** marker-pdf can't co-resolve with our langchain/transformers/torch stack, so it must run **out-of-process in an isolated env** (`uvx marker_single`), gated to `regions.table_candidate_pages` via `--page_range`, as a sidecar enrichment step — never an in-process import. **4a build — code shipped 2026-06-04 (commit `2933881`):** `src/doc_assistant/tables_marker.py` (parse Marker's paginated markdown → per-page GFM blocks with fidelity preserved; **page-anchored inline splice** = de-dup of pymupdf4llm's lossy twin + placement, anchored on the cache's `<!-- page:N -->` markers) + `scripts/extract_tables_marker.py` (parallel `MARKER_MAX_WORKERS` pool, `regions.table_candidate_pages`-gated, shells out to isolated Marker, supersedes pdfplumber) + a deterministic CI gate (`tests/integration/test_marker_table_retrieval.py`) + opt-in `tests/eval/cases.tables.yaml`. **Marker now runs + validated on the CPU box (2026-06-04):** ran Marker end-to-end on the CPU dev box via `uvx --python 3.12` and **cleared the two deferred flag/delimiter items live** — `--page_range`/`--output_format`/`--output_dir`/`--paginate_output` all valid in the pinned marker-pdf, and the paginated separator (`\n\n{N}`+`'-'*48`, 0-based `{N}` before each page) is matched by `_PAGE_DELIM_RE` with correct section→page attribution (no parser change needed). Two fixes shipped getting there (uncommitted, for review): a `hasattr(sys.stdout,"reconfigure")` guard across 13 `scripts/*.py` (fixes Jupyter-import `AttributeError`), and `config.MARKER_PYTHON` (default `"3.12"`) wired into `_marker_command()` (bare `uvx` picked the box default Python 3.14, whose missing `pillow==10.4.0` cp314 wheel forced a from-source build → `exit 1`). **Net: the Marker path is no longer RTX-dependent for correctness — only wall-clock.** Genuinely deferred now (CPU-runnable, just slow): the full-corpus `extract_tables_marker --apply` and verifying `cases.tables.yaml` numbers against the real DPR table. Region-level (multi-region-per-page) splitting is the proper 4b build. **Chunk 2a (dual interpretation) is shipped** (backend + Chainlit UI, merged PR #1) — `SYNTHESIS_MODE=human|ai`, deterministic evidence layer, citation-anchored claims with retrieval-derived markers, eager `answer_claims` adjudication log, in-context accept/reject/edit buttons (quiet-on-clean). With 4a code shipped, **Feature 4b** (figure detection + the ordered objects-manifest, building on 4a's page-locatability) is the next main Phase 6 build.

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

**Snapshot:** ~365 tests green (unit+integration; +23 Chunk 2a, +12 Feature 4a: Marker table parse/splice + CI gate, +9 retrieval cleanups 2026-06-07: CANDIDATE_K guard ×3, router topical-routing ×5, reranker sigmoid guard ×1 — confirm exact total via a full `pytest` run) · ruff format/check + mypy (strict) + bandit clean · **CUDA/GPU enabled on the RTX box** (Windows; embeddings + reranker on `cuda:0`, retrieve+rerank ~68 ms).

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
- ~~**Chunking sweep: deferred to the RTX box.**~~ ✅ **Done 2026-06-06** (RTX, public corpus, n=3, judge): all 6 configs measured, **defaults `2000/200·400/50` confirmed** (no config beats them). Baseline: [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](tests/eval/baselines/chunking_sweep_public_2026-06-06.md).

**Operational TODOs (don't need code changes):**
- **Fully-local Ollama path — generator + reviewer verified live on the RTX (GPU) box (2026-06-02).** ✅ **Generator**: ingested the 10-paper public corpus and ran real queries end-to-end on `llama3.1:8b`, `ANTHROPIC_API_KEY` empty — grounded, correctly-cited answers. ✅ **Reviewer**: `/review` returns a real verdict on Ollama (live run: faithfulness 5 · citation 3 · hedging 4 · unsupported 0). Two transport bugs found and fixed getting there — `OllamaClient` passed `temperature` as an `invoke` kwarg (`TypeError` from `Client.chat()`), and the reviewer choked on llama's non-JSON output (now `format="json"` + tolerant `_extract_json`); both have regression tests. ⚠ Remaining for the DoD bullet: the **eval-judge** on Ollama (`tests/eval/TESTING.md` calibration gate). Pick the local model(s) via `LLM_MODEL`/`REVIEWER_MODEL`.
- ✅ **CUDA/GPU enabled on the RTX box (2026-06-02).** torch now resolves to `2.12.0+cu130` on Windows via a `[[tool.uv.index]]` + win32-scoped `[tool.uv.sources]` in `pyproject.toml`; embeddings + reranker run on the RTX 4070 (`cuda:0`), retrieve+rerank ~68 ms (was CPU-seconds). No app-code change — sentence-transformers auto-detects CUDA. The `sys_platform == 'win32'` marker keeps Linux/CI on the default CPU wheel. (Watch the next CI run to confirm the lock resolves cleanly on Linux.)
- ✅ **SPECTER2 re-run at `--repeat 5` done (2026-06-04).** Symmetric BGE-vs-SPECTER2 on the public corpus — BGE wins all scorers (see Benchmarks-locked block above). bge-base stays default.
- ~~**Chunking sweep configs 2–6: run on the RTX box.**~~ ✅ **Done 2026-06-06** — full 6-config grid on the RTX box; defaults confirmed, Locked-settings chunk-sizes row updated. See [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](tests/eval/baselines/chunking_sweep_public_2026-06-06.md).
- Most `expected_answer` fields in `tests/eval/cases.yaml` are best-effort (`author_verified: false`) — refine over time.
- LNCS colon-separator format + multi-column PDF extraction: known tier-1 citation-extractor weaknesses; cosmetic, deferred.
- ⚠ **RE-RUN ON THE FULL CORPUS (needs the neuroscience library — only on the other machines, not this box).** Three coupled retrieval items from the 2026-06-07 session were measured **only on the 10-paper public corpus**, which is one-paper-per-topic and so over-states "crowding" (single-paper questions legitimately pull all K from one source). Re-validate on `cases.yaml` (multi-paper-per-topic) before locking anything:
  1. **`CANDIDATE_K=20` (shipped, uncommitted) changes retrieval** — pool widened from 10→20 per retriever before rerank. Must run the public + private eval vs the bge baseline; if `contains_all`/`llm_judge` regress, set `CANDIDATE_K=10` (exact pre-split behaviour). Until measured, treat as **provisional, not locked**.
  2. **Per-source diversity cap (#1)** — `scripts/diagnose_crowding.py` on the public corpus said crowd 10/10, but that's inflated (see above). Re-run `--cases tests/eval/cases.yaml` to see crowding where multi-paper synthesis actually happens. Only then decide; build as **focused-vs-broad**, never a flat cap (a flat cap evicts relevant content from single-paper-deep questions).
  3. **Endgame = context/query-dependent max-K.** `K = min(quality_ceiling, context_budget)`. **A hard cap stays mandatory even fully-local**: Ollama defaults to a small `num_ctx` (~4096) and **silently truncates** prompt overflow (drops the front — lost evidence/citations, no error), so local needs the cap *more* than cloud. Budget against the actual `num_ctx` sent (read via Ollama `POST /api/show`), default conservatively until wired. Belongs with the query-expansion work in the review phase.

**Next priority (RESUME HERE — 2026-06-06):** This RTX-box session: **(a)** ran the full 6-config chunking sweep → **defaults `2000/200·400/50` confirmed** (no config wins; tied-best contains_all 0.933, best judge 3.951; baseline [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](tests/eval/baselines/chunking_sweep_public_2026-06-06.md)); **(b)** got **Marker running on the GPU** — `UV_TORCH_BACKEND=auto` gives the isolated `uvx` marker env a `cu130` torch (`cuda_available True`), ~30× faster than the Windows-default CPU wheel (RTX idle before, 83% util / 7.9 GB after); **(c)** ran live `extract_tables_marker --apply --doc <dpr>` on GPU (**7 faithful tables** spliced, `<br>`/bold intact, Top-20 78.4 / Top-100 85.4 correct) + `cases.tables.yaml` → found a wide-table retrieval gap, **now FIXED (2026-06-06, uncommitted):** the table's **caption** ("Table 2: Top-20 & Top-100 retrieval accuracy …", the query magnet) sat in a *different* parent from the numeric grid, so retrieval surfaced the caption parent and `85.4` never reached the LLM (`contains_all 0.750`, judge 2.333). Fix = caption-anchored splice (`tables_marker._place_block_in_span`) + table-aware chunking that keeps the block whole and co-located with its caption (`ingest._table_aware_parents`). After re-splice + `ingest --rebuild`: **`contains_all` 1.000, `llm_judge` 5.000**, public eval unchanged, 356 tests green (+3). See DEVLOG 2026-06-06. **Splice was always faithful — the gap was chunking/retrieval, not Marker.** README gained GPU-tier + Ollama system-requirements sections. **Open next:** (1) optional full-corpus Marker `--apply` (now GPU-fast, ~10–15 min); (2) wire `UV_TORCH_BACKEND=auto` into the Marker CLI as a documented GPU opt-in; (3) **Feature 4b** (figure detection / region-level splitting — can refine per-region splice placement on multi-table pages). Nothing committed this session — awaiting review.

**Prior (RESUME context — 2026-06-04):** Chunk 2a dual-interpretation **shipped + merged (PR #1)**; Marker (4a) now **runs + validated on the CPU box** — flag/delimiter items cleared live, two fixes uncommitted (see below). **First action on resume: commit this session's two fixes** (stdout `hasattr` guard ×13 scripts; `MARKER_PYTHON` pin) — real bug fixes that unblock the marker notebook, standalone. Benchmarks locked except the chunking sweep. Open threads:

**A. Chunking sweep on the RTX box** — configs 2–6, one command + restore step in [`docs/chunking-sweep-rtx-resume.md`](docs/chunking-sweep-rtx-resume.md). Then record the winner / confirm defaults and update the Locked-settings chunk-sizes row.

**B. Feature 4a — code ✅ shipped (commit `2933881`); Marker now runs + validated on CPU (2026-06-04).** `tables_marker.py` + `extract_tables_marker.py` + CI gate + opt-in `cases.tables.yaml`; 353 tests green. **The flag/delimiter validation is DONE on the CPU box** (`--paginate_output` valid; `\n\n{N}`+`'-'*48` separator matched by `_PAGE_DELIM_RE`; attribution correct — no parser change). Fixes to land first: stdout `hasattr` guard (13 scripts) + `config.MARKER_PYTHON="3.12"` pin on the `uvx` path. **Only-remaining 4a items (CPU-runnable, just slow — not RTX-blocked):** `extract_tables_marker --apply` on the corpus, then `cases.tables.yaml` (`--with-llm-judge`) with its expected values tweaked to the real DPR table. After that, **Feature 4b** (figure detection + ordered objects-manifest) is the next code build.

**Lighter alternatives** if not starting the build: a **demo dry-run** of the 7-question script (in Notion: "Demo Script — doc_assistant", under the doc-assistant page) — verify Q7 actually trips the ⚠ low-confidence path; or the **measurement runs** that are cheap now with GPU (chunking sweep, SPECTER2 `--repeat 5`). A working-app **demo is upcoming** (runs on Anthropic API; `.env` already configured — key present locally, gitignored). **Chunk 2a** (Dual Interpretation; `SYNTHESIS_MODE`, evidence vs interpretation) **shipped 2026-06-04** (spec `docs/specs/chunk-2a-dual-interpretation.md`; see DEVLOG). Known v1 follow-ups: coarse sentence-segmentation can orphan a stat-sentence from its citation (flagged `unsupported`; `edit` is the escape hatch) → refine by inheriting an adjacent citation or structured-generation v2; and the **Phase 8** rich per-claim inline-edit GUI (current UI is minimal in-context buttons). Feature 4b (figure detection / region-level splitting) is unblocked once the 4a build lands.

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
├── tables.py             # Phase 6 / Feature 4a: pdfplumber table extraction — LEGACY/fallback; superseded as primary by Marker (see tables_marker.py + 2026-06-02 engine decision)
├── tables_marker.py      # Phase 6 / Feature 4a: parse Marker paginated markdown → per-page GFM blocks; caption-anchored inline splice into the markdown cache (primary table path)
├── synthesis.py          # Phase 6 / Integrity Chunk 2a: dual interpretation — evidence layer + citation-anchored, retrieval-marked AI claims (pure; no LLM/DB/IO)
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
| Chunk sizes | parent 2000/200, child 400/50, baseline 1000/200 | Config-driven (`*_CHUNK_SIZE` env vars). **Measured + confirmed 2026-06-06** — full 6-config sweep on the RTX box (public corpus, n=3, judge); no config beats the defaults (tied-best `contains_all` 0.933, best `llm_judge` 3.951; larger parent regressed). Closest alt = `256/32` child (matches contains_all, lower variance) but doesn't exceed control. See [`tests/eval/baselines/chunking_sweep_public_2026-06-06.md`](tests/eval/baselines/chunking_sweep_public_2026-06-06.md) |

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
| 6 — Per-project routing + Figures & Tables + Dual interpretation | 🔄 In progress | Reviewer agent ✅ + LLM provider protocol ✅ + Chunk 2a dual-interpretation ✅ (merged PR #1); Feature 4 detection layer (`regions.py`) done, engine = Marker; 4a Marker ingest **code ✅** (commit `2933881`; live RTX `--apply` run + `cases.tables.yaml` verification pending); **4b figure extraction** next |
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
- **MCP server interface (open / unscheduled):** expose the RAG pipeline as an MCP server so external MCP hosts — Claude Desktop, claude.ai connectors — can call the local library as a tool (e.g. `search_library` / `ask`), letting a paid Claude subscription query your documents. Fits the thin-entrypoint pattern (a new `apps/mcp_server.py` over `pipeline.py`, no core changes). Nuance: Claude Desktop can talk to a local stdio server directly, but claude.ai connectors need a reachable HTTP endpoint + auth (a real consideration for a local-first app). Not yet scoped to a phase.

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
