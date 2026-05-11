# Design decisions

This document contains the *why* behind my choices, and the planned evolution of the project.

---

## Core Decisions

### Local-first by default

Designed to run locally on a personal machine. Cloud LLMs are available as a practical option. Embeddings, vector store, reranker are all local.

### Markdown as the universal intermediate format

Instead of having each extractor produce its own data structure, every format converts to markdown first. This means:

- One chunking strategy works for all formats.
- The cache is human-inspectable (you can open the `.md` files and see what got extracted).
- Adding a new format is one extractor function, no downstream changes.

The trade-off: some structural information is flattened. I considered it worth it for the simplification.

### Why not LlamaIndex or a managed RAG service

Wanted to practice using LangChain. Using LlamaIndex would have hidden some of the parts I wanted to learn.

### Streaming ingest over batch

A batch ingest holds the entire corpus in memory before storing. For most cases, that's fine ; for someone having 500+ research PDF papers, it isn't. 
Memory stays flat regardless of corpus size, and a crash on document 200 doesn't lose documents 1-199.

### Two-tier cache

Extraction is far slower than embedding. Caching only embeddings means re-extracting PDFs every time chunking strategy changes. Caching markdown separately decouples extraction from chunking — change the splitter, re-embed, but don't re-extract.

### Hybrid search instead of pure vector

Vector search loses on exact terms (author surnames, equation labels, library function names). BM25 loses on paraphrased questions. Ensemble retrieval at weights `[0.4, 0.6]` (keyword:vector) gave better hits during testing on technical documents.

### Cross-encoder reranking on top of retrieval

Embedding-based retrieval is fast but not perfect. The reranker is slower per item 
but only sees ~10 candidates, so total cost is small.

### Document hierarchy with orthogonal topic tags
Folder (user-defined) → Document → Part (section/chapter) → Chunk
↑
(topic tags)

A chunk has one structural parent (its part within a document) but can carry 
many topic tags. This separates *structural* hierarchy (the document's 
inherent organization) from *semantic* membership (what concepts a chunk is 
about). Both should be queryable independently.

Folders are user-defined groupings that don't have to align with topics or 
documents — useful for projects, reading lists, course materials.

### Metadata as soft filter, not hard gate

The RAG remains the core of the project while using hybrid search + 
reranking. The addition of a metadata layer would act as an optional scope ("only search in folder X" or 
"only methodology sections"). 
This should preserve the baseline behavior while adding precision when explicitly requested.

### Two-step metadata cleanup

Current:
    Cleanup scripts use a dry-run + apply pattern:
    1. Dry run reports what would change, makes no modifications
    2. User reviews, then re-runs with `--apply`

Reasoning: the goal is to improve retrieval quality while not always re-running ingest which is costly. 
It only needs to be done once for each document, so it is extra worth it to run a cleanup on the files.

---

## Roadmap


### ✅ Phase 1: Core RAG (complete)

Goal: working end-to-end RAG over personal documents.

- Multi-format ingestion (PDF, EPUB, HTML, DOCX, Markdown, plain text)
- Hybrid retrieval (BM25 + vector ensemble)
- Cross-encoder reranking
- Two-tier caching (markdown + embeddings)
- Streaming ingest
- Hash-based deduplication
- Conversation memory with question rewriting
- Inline citations with page numbers and section headings
- Source previews in the UI
- Token tracking with cost estimation
- Pluggable LLM (Anthropic API or local Ollama)
- Chainlit web UI + CLI fallback
- Docker support

### 🔄 Phase 2: Quality Foundation (in progress)

Goal: make the existing Q&A measurably smarter. No quality improvement is 
worth claiming without a metric.

- 2.0 Section metadata cleanup (references, garbage, title-as-section)
- 2.1 Evaluation set: 20+ question/answer pairs spanning factual recall, 
  methodology, synthesis, state-of-the-art, and negative-answer cases
- 2.2 Metrics: retrieval recall@K, answer faithfulness, answer correctness,
  latency, token cost
- 2.3 Baseline measurement
- 2.4 Experiment: semantic chunking
- 2.5 Experiment: parent-child retrieval
- 2.6 Experiment: section-aware retrieval
- 2.7 Experiment: query rewriting / HyDE
- 2.8 Lock in best combination
- 2.9 Tech debt: content-only document hashing

Deliverable: a measurable improvement in answer quality, with numbers.

### Phase 3: Document Store + Library UI

Goal: treat the library as a first-class object. Browse, upload, organize, 
inspect, and control how documents are processed.

Core components:
- SQLite document store implementing the Folder → Document → Part → Chunk 
  hierarchy
- Library browser in the Chainlit UI
- Drag-and-drop upload with debounced ingest
- Per-document, per-folder, and per-part metadata editing
- Optional metadata-scoped retrieval ("answer using only documents tagged X")
- Tag and folder management

**Document Health & Ingestion Control** (sub-feature):

Silent extraction failures destroy trust and pollute downstream metrics. 
The current global-env-var approach to extractor selection doesn't scale — 
different documents need different treatments.

- Pre-ingestion profiling: detect scanned vs born-digital, presence of text 
  layer, page count, estimated work
- Classification: "born-digital", "scanned", "hybrid"
- Recommended extractor per file with rationale shown to the user
- Per-document extractor override (PyMuPDF4LLM vs Marker, eventually others)
- Post-ingestion health check: chunk count, section detection, page coverage, 
  "looks broken" flag
- Per-document health badges in library view: ✅ good / ⚠️ marginal / ❌ broken
- One-click re-extraction with a different extractor (no manual cache deletion)
- Persistent ingestion log

Rationale: this experience was discovered the hard way — historical PDFs 
(Hodgkin & Huxley, Hubel & Wiesel, Rizzolatti) extracted to 1-10 chunks each 
because they lack proper text layers. We had to manually diagnose and fix this. 
A non-technical user wouldn't have noticed until eval results were inexplicable.

### Phase 4: Citation Graph

Goal: model relationships between documents. Both explicit (citations) and 
implicit (semantic similarity). Surface the structure of the literature.

- Citation extraction (GROBID for academic papers, regex/LLM for others)
- Internal citation matching: when paper A cites paper B *and B is in the 
  library*, create an explicit edge
- External citation tracking: when A cites B but B isn't in the library, that's 
  a known unknown — surface as a recommendation candidate
- Semantic similarity edges between documents (embedded with a doc-level vector)
- Interactive graph visualization (Obsidian-style): nodes are papers, edges 
  are citations or strong similarity
- Cluster detection — automatic topical groupings

### Phase 5: Gap Detection and Cartography

Goal: quantitative view of what the library knows vs what the field knows.
This is the distinguishing capability — most existing tools stop short of this.

- External literature API integration (Semantic Scholar, OpenAlex, arXiv)
- DOI lookup at ingest time → fetch authoritative metadata, references, 
  classifications
- Topic clustering across the library
- Field coverage estimation: "you have 47 of the top 100 papers in this cluster 
  by citation count"
- Frontier detection: recent papers heavily cited by your library's papers but 
  missing from your library
- "What to read next" recommendations, ranked
- Knowledge map dashboard: bubble plot or similar, clusters sized by coverage, 
  recency color-coded, gaps visible at a glance

### Phase 6: UI Polish

Goal: design pass on everything built so far.

- Visual design refinement
- Animations and micro-interactions
- Better empty states, loading states, error states
- Demo recording for portfolio / sharing
- Onboarding flow for new users

### Phase 7: Literature Review Generation

Goal: synthesize a structured review across documents in the library on a 
chosen topic. The endgame feature.

Requires everything from Phases 2-5 to work well:
- Reliable multi-document retrieval (Phase 2)
- Document and section structure as data (Phase 3)
- Citation graph for grounding (Phase 4)
- Coverage awareness so the system knows what it's missing (Phase 5)

Capabilities:
- Per-paper synthesis: what does each paper contribute on the topic?
- Cross-paper synthesis: agreements, disagreements, evolution of ideas
- Citation-grounded claims: every assertion linked to specific passages
- Structural output: introduction, main themes, methodological comparison, 
  open questions, references
- Configurable scope: by folder, by topic, by date range, by document set
- Export: markdown, PDF, or BibTeX-aware Word document

---

## What I'd Position This As

> A personal research workspace that combines local-first RAG with explicit 
> modeling of the literature graph, designed to help a researcher identify 
> what they know, what they don't, and what to read next. Unlike general-
> purpose RAG tools, it treats the document collection as a first-class 
> object to curate and analyze, not just a corpus to search.

Keywords: *personal*, *local-first*, *literature graph*, *knowledge gaps*, *what to read next*.

---

## Open Questions

Decisions I haven't made yet:

- **UI framework for Phases 4-7.** Chainlit handles chat well but is awkward 
  for graph visualization and library browsers. Possible migrations: Reflex, 
  FastAPI + custom frontend, Streamlit. Decision point: when graph 
  visualization is being built and Chainlit's element API hits its limits.

- **Embedding model upgrade path.** current choice: `bge-base-en-v1.5`
    I should test new models.

- **Multi-user support.** Currently single-user. Multi-user suppport in the future ?
    Redesign of db architecture needed. 

- **Citation extraction quality vs. external API quality.** TBD