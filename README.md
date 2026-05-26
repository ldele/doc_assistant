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
| Embeddings | `bge-base-en-v1.5` (local, CPU-friendly) |
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

### Citation graph (Phase 4)

After ingestion, two post-passes populate the citation graph:

```bash
# Pull title / authors / year / DOI off each document header
uv run python -m scripts.extract_doc_metadata --apply

# Parse References sections, match to library docs, persist edges
uv run python -m scripts.extract_citations --apply
```

Both scripts are idempotent and accept `--doc <hash-prefix>` to scope to a
single document, and `--force` to overwrite.

From the chat UI or CLI, ask:

```
/library                  show all documents (use first 8 chars of ID below)
/cites <doc-id>           papers this document cites (internal + external)
/cited-by <doc-id>        library documents that cite this one
/graph <doc-id>           Mermaid subgraph of internal citation edges
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

# Evaluation harness (costs API tokens — manual only)
uv run python tests/eval/run_eval.py
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

## Status

Phase 4 in progress — citation graph data layer + slash commands shipped; doc-level similarity edges deferred to Phase 5 prep.

Roadmap: gap detection (Phase 5) → UI polish (Phase 6) → literature review generation (Phase 7). See [`docs/decisions.md`](docs/decisions.md) for the full roadmap.

---

## License

MIT
