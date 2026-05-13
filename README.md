# Document Assistant

A local RAG (retrieval-augmented generation) assistant over personal document libraries — PDFs, EPUBs, HTML, DOCX, Markdown. 
Ask questions in natural language, get answers grounded in citations from your own files.

Built for researchers or students who wants to actually search the *content* of their books, not just the titles.

## What's inside

A personal-scale RAG system that integrates established techniques and 
measures them empirically.

- **Multi-format ingestion** — PDF, EPUB, HTML, DOCX, Markdown, plain text
- **Hybrid retrieval** — BM25 keyword search ensembled with vector similarity
- **Cross-encoder reranking** — BGE reranker for precision on top of recall
- **Two-tier caching** — markdown cache + embedding cache, both invalidated automatically
- **Streaming ingest** — memory stays flat regardless of library size
- **Conversation memory** — follow-up questions understand context
- **Source previews** — every answer is backed by inspectable passages with page numbers and section headings
- **Token tracking** — real-time cost transparency per turn and per session
- **Pluggable LLM** — Anthropic API by default, Ollama for fully offline use

## Stack

- **Embeddings**: `bge-base-en-v1.5` (local, CPU-friendly)
- **Reranker**: `bge-reranker-base` (local)
- **Vector store**: Chroma (local, persistent)
- **Keyword search**: BM25
- **LLM**: Claude (API) or Llama 3 / Mistral (local via Ollama)
- **Orchestration**: LangChain
- **UI**: Chainlit
- **PDF extraction**: PyMuPDF4LLM (default) or Marker (high-quality fallback)

## Setup

```bash
# Clone and enter
git clone <your-repo-url> doc-assistant
cd doc-assistant

# Create venv (use `py` instead of `python` on Windows)
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.\.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install
pip install -e ".[dev]"

# Configure
cp .env.example .env               # then edit .env with your API key
```

## Usage

```bash
# Drop your documents in data/sources/
mkdir -p data/sources
cp ~/your-papers/*.pdf data/sources/

# Build the index (one-time, then incremental)
python -m doc_assistant.ingest

# Launch the chat UI
chainlit run apps/chainlit_app.py

# Or use the CLI
python -m apps.cli
```

To rebuild from scratch (after changing chunking strategy, for example):

```bash
python -m doc_assistant.ingest --rebuild
```

## Project layout

src/doc_assistant/    # core library
apps/                 # UIs (CLI, Chainlit)
scripts/              # maintenance utilities
tests/                # pytest tests
docs/                 # architecture and design notes
data/                 # runtime data (sources, caches, vector store) — not committed

See [`docs/architecture.md`](docs/architecture.md) for the data flow and module responsibilities, and [`docs/decisions.md`](docs/decisions.md) for the reasoning behind key design choices.

## Running tests

```bash
pytest
```

## Status

Working but evolving. Things on the roadmap:

- Evaluation set with retrieval metrics
- Citation graph extraction for academic papers
- Semantic chunking
- Persisted conversations across sessions

---

## Running with Docker

If you have Docker Desktop installed, you can run the app without installing Python or any dependencies on your machine.

```bash
# 1. Configure
cp .env.example .env             # then edit .env with your API key

# 2. Add documents
mkdir -p data/sources
cp ~/your-papers/*.pdf data/sources/

# 3. Build the image (one-time, ~5 minutes first run)
docker compose build

# 4. Ingest your documents (one-time per new batch)
docker compose run --rm doc-assistant python -m doc_assistant.ingest

# 5. Launch
docker compose up
```

Open `http://localhost:8000`.

### Local LLM via Ollama

If you set `LLM_MODE=local` in your `.env` and have Ollama running on your **host machine** (not in the container), the container reaches it via `host.docker.internal`. No extra config needed on Docker Desktop for Windows or macOS.

On Linux, make sure Ollama listens on all interfaces:
```bash
OLLAMA_HOST=0.0.0.0 ollama serve
```

### Stopping and cleaning up

```bash
docker compose down            # stop, keep data and model cache
docker compose down -v         # stop, also delete the model cache volume
```

Documents in `data/` should always be preserved on the host.

---

## Ingest, Verify, and Toggle Workflow

The system maintains two Chroma stores in parallel: `data/chroma` (baseline 
single-chunk) and `data/chroma_pc` (parent-child). Most operations 
automatically update both.

### Adding new documents

Drop files into `data/sources/`, then run:

```bash
python -m doc_assistant.ingest
```

This processes new files, skips already-indexed ones, removes orphans whose 
source file was deleted, and updates both Chroma stores plus SQLite.

### Verifying integrity

After any ingest, you can verify that the stores are in sync:

```bash
python -m scripts.verify_chroma_sync
```

This checks:
- Both Chroma stores contain the same set of documents
- SQLite has a row for every document in Chroma
- Chunk counts agree between SQLite and Chroma

### Switching retrieval mode

The retrieval mode is controlled by an environment variable, not a config 
file (so you can switch without modifying code):

```bash
# Use parent-child retrieval (default, higher quality, slower)
$env:USE_PARENT_CHILD = "true"

# Use single-chunk baseline (faster, lower quality)
$env:USE_PARENT_CHILD = "false"
```

The active mode is logged at the top of every eval run and Chainlit session.

### Cleanup utilities

If verification surfaces issues:

```bash
# Remove chunks for documents no longer tracked in SQLite
python -m scripts.cleanup_stale_chunks --apply

# Merge duplicate SQLite rows (rarely needed)
python -m scripts.dedupe_documents --apply
```

---

## License

MIT