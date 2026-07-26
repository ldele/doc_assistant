<!-- status: active · updated: 2026-07-26 (split out of README) · class: living -->

# Usage

Day-to-day commands. Install first: see [`setup.md`](setup.md).

## Ingest and launch

```bash
# Drop your documents in data/sources/
mkdir -p data/sources
cp ~/your-papers/*.pdf data/sources/

# Build the index (one-time, then incremental)
uv run python -m doc_assistant.ingest

# Launch the desktop app (Tauri + Svelte over the FastAPI backend), one command:
just app          # starts backend (8001) + dev UI (1420) in their own windows, opens the browser
                  # (no `just`? scripts/launch_app.cmd double-clicks to the same thing)

# ... or manually, in two shells:
uv run --no-sync uvicorn apps.api.main:app --host 127.0.0.1 --port 8001   # backend
cd apps/desktop && npm install && npm run dev    # dev UI (or: npx tauri dev for a native window)

# Or use the CLI
uv run python apps/cli.py
```

To rebuild from scratch (after changing chunking strategy, for example):

```bash
uv run python -m doc_assistant.ingest --rebuild
```

## No corpus of your own yet?

```bash
uv run python -m scripts.download_corpus --demo
```

Fetches 28 classic AI papers from arXiv into `data/sources/`: the public eval corpus plus the arXiv
subset of the rumoured [Sutskever→Carmack reading list](https://30papers.com/). Then ingest as
above. Done exploring? `--remove-demo --apply` cleanly removes them again (matched by content hash,
so renames don't fool it; files go to the Recycle Bin and library entries are safe-deleted).

Benchmark numbers always come from the 10-paper eval corpus alone; see [`../evals/`](../evals/README.md).

## Enrichment passes

Derived data ships as idempotent sidecar runners that never mutate the chunk store. Every one is
**dry-run by default**; `--apply` writes.

```bash
# Citation graph + similarity edges
uv run python -m scripts.extract_doc_metadata --apply   # title / authors / year / DOI per document
uv run python -m scripts.extract_citations --apply      # parse References, match to library docs
uv run python -m scripts.compute_doc_vectors --apply    # doc vectors -> top-K cosine similarity edges

# Knowledge layer
uv run python -m scripts.extract_keywords --apply
uv run python -m scripts.build_concept_skeleton --apply --enrich --provider ollama
uv run python -m scripts.build_gaps --apply
uv run python -m scripts.propose_taxonomy --apply
```

`extract_*` accept `--doc <hash-prefix>` to scope; `compute_doc_vectors` accepts `--top-k`,
`--threshold`, and `--force`.

> **Cost guard.** `.env` defaults are Anthropic-wide, and every enrichment runner inherits the
> provider unless told otherwise, so pass `--provider ollama` on enrichment runs to keep them free.
> Paid runs must clear an explicit intent check.

## Library commands

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

## Move your library between machines

`data/sources/` is gitignored (your library is yours), so cloning the repo elsewhere doesn't carry
your documents. Keep a small **sources manifest**, the private analog of the public-corpus
downloader, to reconstitute them:

```bash
uv run python -m scripts.sync_sources               # record data/sources/ -> data/sources_manifest.yaml
# fill in the `url:` for any file not auto-matched, then copy the manifest across out-of-band
uv run python -m scripts.sync_sources --download    # on the other machine: re-fetch into data/sources/
uv run python -m scripts.sync_sources --verify-only # checksum what's on disk against the manifest
```

The manifest pins each file by SHA-256 + size plus the URL it came from; files matching the public
corpus get their URL filled in automatically. It's **gitignored**, so share it out-of-band and never
commit it (the repo is public).

## Tests and evaluation

```bash
# Unit + integration (free, fast)
uv run pytest tests/unit/ tests/integration/

# With coverage
uv run pytest tests/unit/ tests/integration/ --cov=src --cov-report=term-missing

# Evaluation harness, free deterministic scorers
uv run python -m scripts.run_eval

# With LLM judge (Claude Haiku, ~$0.10 for 35 cases)
uv run python -m scripts.run_eval --with-llm-judge

# Public eval: 10 cases on the RAG-literature demo corpus
uv run python -m scripts.download_corpus           # fetches 10 papers from arXiv
uv run python -m doc_assistant.ingest
uv run python -m scripts.run_eval --cases tests/eval/cases.public.yaml --with-llm-judge
```

What each scorer measures and why: [`../tests/eval/TESTING.md`](../tests/eval/TESTING.md).
Committed reference baselines: [`../tests/eval/baselines/`](../tests/eval/baselines/).
