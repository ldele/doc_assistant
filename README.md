# Document Assistant

A local-first RAG assistant over your own research library (PDF, EPUB, HTML, DOCX, Markdown) that answers questions with inline, page-level citations — and measures whether those answers are any good. Built for researchers and students who need to search the *content* of their documents, with retrieval quality treated as an engineering problem, not a vibe.

Not a chatbot wrapper: the goal is reliable, grounded answers with the measurement and provenance to back them up.

## Why this is interesting (engineering)

- **Decisions are measured, not guessed.** TOP_K, parent-child retrieval, and BM25/vector weights were each chosen from experiments; an in-repo eval harness scores the pipeline and tracks it over time. Every non-obvious choice — and what lost — is in [`docs/decisions.md`](docs/decisions.md).
- **Honest, reproducible benchmarks.** A public eval set runs over a corpus anyone can reconstruct from arXiv, 5 trials, reported as mean ± trial-mean std with caveats stated up front (see [Benchmarks](#benchmarks)). The strategy — what each scorer measures and why — is in [`tests/eval/TESTING.md`](tests/eval/TESTING.md).
- **Research-integrity layer.** Every answer carries a provenance record (retrieved chunks, model, cost); a separate-context reviewer agent re-grades flagged answers; confidence signals keep the UI quiet on clean answers. Built so AI-assisted output is auditable.
- **Architecture that scales by addition.** Enrichment layers (citations, doc-vectors, tables) are sidecar modules + idempotent CLI runners that never mutate the chunk store — new capability is a new module, not a rewrite.

## What it does

- **Grounded answers with inline citations** — page numbers and sections, every passage inspectable.
- **Hybrid retrieval + reranking** — BM25 + vector ensemble, cross-encoder reranker, parent-child chunks.
- **Citation graph** — extracts references, resolves them against your library, exposes in/out edges.
- **Measurable quality** — eval harness with 5 scorers and a DuckDB result store.
- **Local-first and pluggable** — Chroma + SQLite on disk; Claude API or local Ollama.
- **Cost transparency** — per-turn and per-session token tracking.

## Stack

| Component | Choice |
|---|---|
| Embeddings | `bge-base-en-v1.5` (default, swappable via `EMBEDDING_MODEL`; `specter2` also registered) |
| Reranker | `bge-reranker-base` (local) |
| Vector store | Chroma (local, persistent) |
| Keyword search | BM25 |
| LLM | Claude (API) or Llama 3 / Mistral (local via Ollama) |
| Orchestration | LangChain |
| Document store | SQLite (via SQLAlchemy) |
| UI | Chainlit (web) + CLI |
| PDF extraction | PyMuPDF4LLM (default) or Marker (high-quality fallback) |

## Setup

```bash
# Prerequisites: Python 3.12, uv
# Note: Python 3.14 works for development/testing but Chainlit
# requires 3.12 at runtime (anyio event loop incompatibility).
git clone <your-repo-url> doc-assistant
cd doc-assistant
uv sync

# Configure
cp .env.example .env   # then fill in your API key
```

## Usage

```bash
# Drop your documents in data/sources/
mkdir -p data/sources
cp ~/your-papers/*.pdf data/sources/

# Build the index (one-time, then incremental)
uv run python -m doc_assistant.ingest

# Launch the chat UI
uv run chainlit run apps/chainlit_app.py

# Or use the CLI
uv run python apps/cli.py
```

To rebuild from scratch (after changing chunking strategy, for example):

```bash
uv run python -m doc_assistant.ingest --rebuild
```

### Citation graph + similarity edges (Phase 4)

After ingestion, three post-passes populate the data layer:

```bash
# Pull title / authors / year / DOI off each document header
uv run python -m scripts.extract_doc_metadata --apply

# Parse References sections, match to library docs, persist edges
uv run python -m scripts.extract_citations --apply

# Mean-pool chunk embeddings -> doc vectors -> top-K cosine similarity edges
uv run python -m scripts.compute_doc_vectors --apply
```

All three are idempotent. `extract_*` accept `--doc <hash-prefix>` to scope;
`compute_doc_vectors` accepts `--top-k`, `--threshold`, and `--force`.

From the chat UI or CLI:

```
/library                  show all documents (use first 8 chars of ID below)
/document <doc-id>        full details for one document
/cites <doc-id>           papers this document cites (internal + external)
/cited-by <doc-id>        library documents that cite this one
/graph <doc-id>           Mermaid subgraph of internal citation edges
/similar <doc-id>         top-N semantically-similar documents
/bibtex                   render the whole library as BibTeX
```

Also available as CLI utilities:

```bash
uv run python -m scripts.find_duplicates    # byte + content dedup report; never deletes
uv run python -m scripts.export_bibtex      # write docs/library.bib
```

## Project layout

```
src/doc_assistant/    # core library (pipeline, ingestion, extractors, health, library)
apps/                 # UIs — thin shells, no business logic
scripts/              # maintenance utilities (hash migration, sync verification)
tests/                # unit, integration, eval harness
docs/                 # architecture and design decisions
data/                 # runtime data (sources, caches, vector stores, SQLite) — not committed
```

See [`docs/architecture.md`](docs/architecture.md) for the data flow and module contracts, and [`docs/decisions.md`](docs/decisions.md) for the reasoning behind key design choices.

## Running tests

```bash
# Unit + integration (free, fast)
uv run pytest tests/unit/ tests/integration/

# With coverage
uv run pytest tests/unit/ tests/integration/ --cov=src --cov-report=term-missing

# Evaluation harness — free deterministic scorers
uv run python -m scripts.run_eval

# With LLM judge (Claude Haiku, ~$0.10 for 35 cases)
uv run python -m scripts.run_eval --with-llm-judge

# Public eval — 10 cases on the RAG-literature demo corpus (see Reproducing)
uv run python -m scripts.download_corpus           # fetches 10 papers from arXiv
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --with-llm-judge
```

## Docker

```bash
cp .env.example .env             # fill in your API key
mkdir -p data/sources && cp ~/your-papers/*.pdf data/sources/

docker compose build
docker compose run --rm doc-assistant python -m doc_assistant.ingest
docker compose up
```

Open `http://localhost:8000`.

For local LLM via Ollama, set `LLM_MODE=local` in `.env`. On Linux, ensure Ollama listens on all interfaces: `OLLAMA_HOST=0.0.0.0 ollama serve`.

```bash
docker compose down            # stop, keep data
docker compose down -v         # stop, delete model cache volume
```

## Benchmarks

Quality is measured, not asserted. The eval harness ([`src/doc_assistant/eval/`](src/doc_assistant/eval/)) runs the full RAG pipeline — retrieve, rerank, generate — over a fixed question set and scores each answer on retrieval and answer-quality signals. The full strategy (test tiers, what each scorer measures and why, the reproducibility stance) is documented in **[`tests/eval/TESTING.md`](tests/eval/TESTING.md)**.

The headline benchmark is **reproducible by anyone**: a public demo corpus of the 10 arXiv papers behind this project's own methods (RAG, dense retrieval, sentence embeddings, the BGE and SPECTER2 embedders, BERT re-ranking, ColBERT, HyDE, LLM-as-a-judge, AI Usage Cards). Nothing is re-hosted — [`corpus_manifest.yaml`](tests/eval/corpus_manifest.yaml) pins each paper's arXiv ID + SHA-256 and a script fetches the PDFs.

5 trials on `bge-base` (`--repeat 5`), reported as mean ± trial-mean std:

| Scorer | Mean (n=5) | Trial-mean std | What it measures |
|---|---:|---:|---|
| `citation_overlap` (0-1) | **1.000** | 0.000 | retrieval cited the correct source |
| `contains_all` (0-1) | **0.927** | 0.034 | answer surfaces the required facts |
| `llm_judge` (1-5) | **3.894** | 0.075 | reference-graded answer quality |

`citation_overlap` is **1.000 with zero variance** — retrieval depends only on the deterministic index, so it cites the right paper in all 10 cases, every trial. `contains_all` scores the stochastic generated answer, so single runs wobble (0.88–0.98) around a stable 0.927 mean. `llm_judge` **3.894/5** confirms the answers are genuinely good — the `contains_all` shortfall is phrasing, not correctness. Cases are deliberately strict, not tuned to score 1.0. Committed reference results live in [`tests/eval/baselines/`](tests/eval/baselines/).

One honest caveat: the judge call on `sbert_motivation` is flaky — skipped in 3 of 5 trials (API timeout / JSON parse), so the `llm_judge` mean is over 47 of 50 scores.

> The embedder choice (`bge-base` vs `specter2`) was settled on a separate, non-redistributable corpus and is recorded in [`docs/decisions.md`](docs/decisions.md) → Phase 5 / Feature 3. That comparison is queued to be re-run on this public corpus so the embedder evidence is reproducible too.

### Reproducing

[`tests/eval/corpus_manifest.yaml`](tests/eval/corpus_manifest.yaml) lists the 10 papers (pinned arXiv versions + SHA-256); `download_corpus.py` fetches them from arXiv (download-only, so arXiv's license is not an issue). [`tests/eval/cases.public.yaml`](tests/eval/cases.public.yaml) is a standalone 10-case set written against them.

```bash
uv run python -m scripts.download_corpus            # 10 PDFs from arXiv -> data/sources/
uv run python -m scripts.download_corpus --verify-only  # checksum against the manifest
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --with-llm-judge
```

`--with-llm-judge` adds the reference-graded answer-quality score (Claude Haiku); it needs an `ANTHROPIC_API_KEY` in `.env` and costs a few cents for 10 cases. Drop the flag for a free, deterministic-only run (retrieval + keyword scorers).

**Where runs are stored.** Every `run_eval` invocation appends to `data/eval.duckdb` — a binary working log, **gitignored** and regenerated on first run, not a source artifact. Committed, human-readable reference results live in [`tests/eval/baselines/`](tests/eval/baselines/) — diff a new run against those, not the binary DB. The harness is structured for extraction: every file except `adapters.py` is project-agnostic and can be lifted into a standalone repo.

**Chunking sweep** (re-embeds the corpus per config — slow; run against a representative subset):

```bash
uv run python -m scripts.sweep_chunking --dry-run            # print the grid + commands
uv run python -m scripts.sweep_chunking --with-embedding --repeat 3
```

## Status

**Phase 6 in progress.** Shipped: core RAG (Phase 1); measured quality + eval harness (Phases 2 & 5); document store + library UI (Phase 3); citation graph + doc-similarity edges (Phase 4); and the research-integrity layer — provenance card, heuristic confidence signals, and a separate-context LLM reviewer agent. **264 tests · ~64% coverage · ruff / mypy / bandit clean.**

`bge-base` is locked as the default embedder; the comparison that settled it is in [`docs/decisions.md`](docs/decisions.md) → Phase 5 / Feature 3.

**Next (independent, spec'd):** a normalized LLM-provider protocol for fully-local runs (analysis *and* reviewer on Ollama, no API key — [`docs/specs/llm-provider-isolation.md`](docs/specs/llm-provider-isolation.md)), dual-layer evidence/interpretation answers, and pdfplumber table extraction. Full rationale and roadmap: [`docs/decisions.md`](docs/decisions.md), [`docs/doc-assistant-roadmap.md`](docs/doc-assistant-roadmap.md).

A 60-second walkthrough for first-time readers: [`docs/DEMO.md`](docs/DEMO.md).

---

## License

MIT
