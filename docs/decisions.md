# Design decisions

## Local-first by default

Designed to run locally on a personal machine. Cloud LLMs are available as a practical option. Embeddings, vector store, reranker are all local.

## Markdown as the universal intermediate format

Instead of having each extractor produce its own data structure, every format converts to markdown first. This means:

- One chunking strategy works for all formats.
- The cache is human-inspectable (you can open the `.md` files and see what got extracted).
- Adding a new format is one extractor function, no downstream changes.

The trade-off: some structural information is flattened. I considered it worth it for the simplification.

## Why not LlamaIndex or a managed RAG service

Wanted to practice using LangChain. Using LlamaIndex would have hidden some of the parts I wanted to learn.

## Streaming ingest over batch

A batch ingest holds the entire corpus in memory before storing. For most cases, that's fine ; for someone having 500+ research PDF papers, it isn't. 
Memory stays flat regardless of corpus size, and a crash on document 200 doesn't lose documents 1-199.

## Two-tier cache

Extraction is far slower than embedding. Caching only embeddings means re-extracting PDFs every time chunking strategy changes. Caching markdown separately decouples extraction from chunking — change the splitter, re-embed, but don't re-extract.

## Hybrid search instead of pure vector

Vector search loses on exact terms (author surnames, equation labels, library function names). BM25 loses on paraphrased questions. Ensemble retrieval at weights `[0.4, 0.6]` (keyword:vector) gave better hits during testing on technical documents.

## What I'd do next

- Evaluation set + measurement (right now improvements are subjective)
- Citation graph extraction for academic papers
- Section-aware chunking (semantic split rather than character count)
- Persisted conversation history across sessions