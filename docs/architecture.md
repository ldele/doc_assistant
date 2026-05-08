# Architecture

## High-level flow
Documents (PDF/EPUB/HTML/DOCX/MD)
↓
Extractors → Markdown cache (data/cache/)
↓
Chunker (markdown-aware)
↓
Embeddings (BGE-base) → Chroma vector store (data/chroma/)
↓
Hybrid retrieval (BM25 + vector) → select top 10 candidates
↓
Cross-encoder reranker → select top 5
↓
LLM (Claude or local Ollama) → streamed answer with citations

## Module responsibilities

| Module | Role |
|---|---|
| `doc_assistant.config` | Paths and env vars |
| `doc_assistant.extractors` | Convert any supported format to markdown |
| `doc_assistant.ingest` | Ingest sources, extract, chunk, embed, store |
| `doc_assistant.pipeline` | RAG runtime: retrieve, rerank, generate |
| `doc_assistant.prompts` | Prompt templates |
| `doc_assistant.tracking` | Token usage tracking |
| `apps/cli.py` | Plain terminal interface |
| `apps/chainlit_app.py` | Web chat UI |
| `scripts/debug_metadata.py` | checks vector store and link to metadata|

## Two-tier caching

1. **Extraction cache** (`data/cache/*.md`): mirrors `data/sources/` structure. 
   Skips re-extracting PDFs when files are unchanged.
2. **Embedding cache** (Chroma `doc_hash` metadata): skips re-embedding documents 

Both caches are invalidated automatically by file modification time and content hash respectively. You can rebuild with `python -m doc_assistant.ingest --rebuild`.