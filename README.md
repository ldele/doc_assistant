# Document Assistant

A local RAG (retrieval-augmented generation) assistant over personal document libraries — PDFs, EPUBs, HTML, DOCX, Markdown.
Ask questions in natural language, get answers grounded in citations from your own files.

Built for researchers and students who want to actually search the *content* of their documents, not just titles.

## What it does

- **Multi-format ingestion** — PDF, EPUB, HTML, DOCX, Markdown, RTF, ODT, plain text
- **Hybrid retrieval** — BM25 keyword search + vector similarity ensemble
- **Cross-encoder reranking** — precision scoring on top of recall
- **Parent-child retrieval** — small chunks for retrieval accuracy, large passages for LLM context
- **Document health scoring** — automatic extraction quality assessment per document
- **Library management** — SQLite-backed document store with folders, tags, and ingestion history
- **Citation graph** — extracts references from each document, resolves them against the library, exposes incoming/outgoing edges and small subgraphs
- **Two-tier caching** — markdown extraction cache + embedding cache, both auto-invalidated
- **Streaming ingest** — constant memory regardless of library size
- **Conversation memory** — follow-up questions understand context
- **Inline citations** — every answer backed by inspectable passages with page numbers and sections
- **Token tracking** — real-time cost transparency per turn and per session
- **Pluggable LLM** — Anthropic API by default, Ollama for fully offline use

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

## Benchmark results

A 35-question eval set ([`tests/eval/cases.yaml`](tests/eval/cases.yaml)) over a 51-document neuroscience corpus, run via the harness in [`src/doc_assistant/eval/`](src/doc_assistant/eval/) against two embedding models. **5 trials per model** (`--repeat 5`); reported as **mean ± trial-mean std** so the rerun-reliability is visible.

**Why those two embedders ?** Specter2 is expected to perform well on academic papers (on which my testings are done). BGE-base is more flexible and works well on a vast array of documents (generalist).

| Scorer | bge-base (n=5) | specter2 (n=5) | Δ (bge − specter2) | Verdict |
|---|---:|---:|---:|---|
| `citation_overlap` (0-1) | **0.907 ± 0.000** | 0.887 ± 0.002 | +0.020 | Real (deterministic gap) |
| `contains_all` (0-1) | **0.804 ± 0.013** | 0.767 ± 0.014 | +0.037 | Real (~4σ) |
| `llm_judge` (1-5) | 2.209 ± 0.053 | 2.224 ± 0.092 | −0.015 | **Tied** (~0.3σ — within noise) |

**bge-base wins on retrieval-side signals; bge-base and specter2 are tied on the LLM judge.** The cross-encoder reranker and the answer LLM together level out the embedder differential at the chunk level — what matters is which documents got retrieved (where bge-base wins by 0.02) and whether the answer surfaces the expected keywords (where bge-base wins by 0.04). The judge's faithfulness/relevance/completeness rubric over the resulting answers doesn't see a meaningful difference.

### Why specter2 doesn't beat bge-base here (training-task mismatch)

specter2 is the "academic paper" embedder, but it was trained for a different task than chunk-level QA:

- **specter2 training:** predict citations between paper *abstracts* — paper-level similarity.
- **our task:** retrieve the right *chunk* (400-2000 chars of methods/results) for a natural-language question.

Two domain mismatches: paper-level vs chunk-level, abstract vs full-text. specter2 has never seen "a paragraph from a methods section" during training. bge-base's training corpora (MS MARCO, NQ, SQuAD, HotpotQA) are much closer to QA retrieval.

specter2 may still help for paper-level similarity tasks (e.g., powering `/similar`) — gated on an explicit eval against that task. Per-folder embedder routing stays deferred until there's a use case where specter2 demonstrably wins.

### Caveats

1. **Sample size: 35 cases × 5 trials.** With trial-mean-std bounds in single-digit-percent territory, the deterministic and contains_all gaps are statistically real; the llm_judge gap is within noise.
2. **LLM-judge mean ~2.2/5 across both models.** The judge enforces strict reference-only grading — even true answers score low when the reference doesn't mention them. Cross-model *gap* is the signal, not absolute scores.
3. **Reference answers partly author-verified.** 4 of 35 cases have hand-verified expected_answers; the rest are best-effort. Affects absolute scores symmetrically; doesn't bias the gap.
4. **`embedding_similarity` excluded.** The scorer uses the active model's own embedder, so the comparison is confounded across models. Fix is queued.
5. **The project has still features in development**. The benchmark gives a taste on how the pipeline interacts with those two embedders. For now (Phase 5), this is enough. Testings will need to be redone and more cases built to reflect the end-usage of the application.

### Reproducing

```powershell
$env:EMBEDDING_MODEL="bge-base"
uv run python -m scripts.run_eval --with-llm-judge --repeat 5 --note "bge n=5"
$env:EMBEDDING_MODEL="specter2"
uv run python -m scripts.run_eval --with-llm-judge --repeat 5 --note "specter2 n=5"
```

Cost: ~$1 per model run. Results land in `data/eval.duckdb`. The harness is structured for extraction — see [`src/doc_assistant/eval/`](src/doc_assistant/eval/); every file except `adapters.py` is project-agnostic and can be lifted into a standalone repo.

## Status

Phase 5 in progress — embedding factory + eval harness shipped. bge-base locked as default based on the benchmark above. Next: provenance card (Integrity Chunk 1).

Roadmap: figures/tables + dual interpretation (Phase 6) → gap detection (Phase 7) → UI polish (Phase 8) → literature review generation (Phase 9). See [`docs/decisions.md`](docs/decisions.md) and [`docs/doc-assistant-roadmap.md`](docs/doc-assistant-roadmap.md) for full details.

---

## License

MIT
