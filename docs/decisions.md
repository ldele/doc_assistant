# Design decisions

This document contains the *why* behind my choices, and the planned evolution of the project.

---

## About This Project

This project implements and evaluates established RAG techniques on a personal 
document library. It does not introduce new algorithms or methods.

- Hybrid retrieval (BM25 + dense vectors): standard practice since 2023.
- Cross-encoder reranking: standard practice, originally popularized by Microsoft 
  research and Sentence-Transformers.
- Parent-child retrieval: a LangChain pattern, predates this project.
- Section-aware chunking: a common refinement, not unique here.
- Query rewriting / HyDE: documented techniques from the literature.

What this project does contribute is:
- An end-to-end working system over real research documents.
- A measurement harness that lets us compare these techniques empirically.
- Documented design decisions for a specific domain (personal academic libraries).
- Practical engineering: caching, streaming ingest, metadata cleanup, regression 
  tracking.

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

### Parent-child retrieval is the default

According to literature and testing, this choice should improve performance.

Measurement with strict judge (temperature=0, tight rubric):

|---|---|---|
| Retrieval recall | 0.90 | 0.95 |
| Correctness | 3.48 / 5 | 4.10 / 5 |
| Faithfulness | 3.90 / 5 | 4.35 / 5 |
| Methodology correctness | 3.00 | 4.40 |
| Latency | 5.24s | 4.52s |

A +0.62 correctness improvement with a strict judge is a substantial real
effect. Methodology questions in particular benefit (+1.40), because they
require procedural context that small chunks can't convey.

The mechanism: child chunks (~400 chars) are embedded and used for precise 
retrieval. When a child is retrieved, its containing parent passage 
(~2000 chars) is passed to the LLM instead. This gives the LLM richer 
context per retrieved item without sacrificing retrieval precision.

The baseline (single-chunk) retrieval remains available via 
`USE_PARENT_CHILD=false` in the environment.

Note: This setting is for what I currently have in my library.
What matters is that I can quickly check the performance of the settings.

### TOP_K tuning — adopted at TOP_K=10

Hypothesis: increasing the number of retrieved chunks would help when the 
correct content was being found but not all of it made the top-5 cutoff.

Measurement swept TOP_K over [3, 5, 8, 10, 12, 15] to find optimal value
(strict judge, parent-child enabled):

| K | Correctness | Faithfulness | Latency | Notes |
|---|---|---|---|---|
| 3 | 3.57 | 3.95 | 4.3s | Too little context |
| 5 | 3.81 | 4.05 | 4.8s | Previous default, undertuned |
| 8 | 4.42 | 4.72 | 5.9s | Previously considered optimum |
| 10 | **4.55** | **4.74** | 10.9s | Peak on both metrics |
| 12 | 4.43 | 4.55 | 11.7s | Past peak, faithfulness drops |
| 15 | 4.33 | 4.45 | 13.2s | Clearly past optimum |

Decision: K=10 adopted. Both correctness and faithfulness peak there, with
clear declines on both sides. The +0.13 correctness over K=8 comes at +5s
latency, accepted as worthwhile for a research tool.

Outcome: substantial improvement across all measures except latency.
Five of six known regressions resolved, including all three historical PDF
failures that previous experiments couldn't fix (scanned documents).

Mechanism: the reranker's 6th, 7th, and 8th choices are still highly 
relevant (cutting at 5 was removing relevant context).

Trade-offs:
- Going below K=10 loses retrieval breadth on distributed-information questions
- Going above K=10 dilutes the LLM's context with marginally-relevant chunks
- The faithfulness drop past K=10 confirms this is the inflection point, not noise

Note: This setting is for what I currently have in my library.
What matters is that I can quickly check the performance of the settings.

### Multi-Query expansion — tested twice, not adopted

Hypothesis: generating 3 variations of each query and combining retrieval 
results would improve recall on ambiguous or distributed-information questions.

Two prompt variants tested:
- Variant A: paraphrase + narrower + broader
- Variant B: paraphrases only (no scope shifts)

Both regressed against the no-MQ baseline:

| Metric | TOP_K=8 alone | TOP_K=8 + MQ-A | TOP_K=8 + MQ-B |
|---|---|---|---|
| Correctness | 4.29 | 3.62 | 4.00 |
| Faithfulness | 4.55 | 4.25 | 4.30 |
| Regressions resolved | 5/6 | 2/6 | 2/6 |

Mechanism of failure: query variations multiply the number of candidates 
the reranker has to discriminate among. Even when variations stay tightly 
scoped (paraphrase-only), they dilute the reranker's selection of the 
top-K positions. The highest-relevance chunks survive, but the marginal 
positions (6th-8th) get displaced by slightly-less-relevant candidates 
from variations.

Note: This setting is for what I currently have in my library.
What matters is that I can quickly check the performance of the settings.
As of phase 2, the cost outweighs the benefits. 

The MQ code remains available but disabled by default.

### Document health tracking

Health classification implemented as a pure-Python scoring model in 
`src/doc_assistant/health.py`. Each document is classified as healthy, 
marginal, or broken based on observable signals at ingest time:

- Chunk count and chunks-per-page ratio
- Average chunk length
- Section detection rate
- PDF page marker presence
- Reference-flagged chunk ratio

Score ≥75 = healthy, ≥40 = marginal, <40 = broken.

The classification is informational, not blocking: broken documents are 
still retrievable, but visually flagged for the user's attention.
Broken document content might still be valuable. 

Known limitations:
- Some documents might be classified as marginal. (Issues with some pdf formats)
  Not blocking; accepted as a known issue.
- Section detection regex now strips markdown formatting from heading 
  text to prevent future occurrences. 
  (Might have to build something more robust after further testing).

To resolve some edge cases, added a dedup script (`scripts/dedupe_documents.py`) 
to resolve all duplicate Document rows from the path+content hash drift.
Might be an artifact from chroma to sqllite migration.

### Library browser via chat commands

Implementation pattern: `/library` command renders all documents as a 
structured Chainlit message. Chosen for implementation simplicity and 
reliability within Chainlit's native message API.

Known scaling limit: this pattern works well for libraries up to ~100 
documents. Beyond that, rendering hundreds of cards in a single message 
becomes unwieldy and slow.

Future migration path when needed:
1. Add pagination (Phase 6): `/library page:2` for 20-doc pages
2. Or migrate to a real sidebar/separate UI (Phase 6+) if Chainlit gets 
   better layout support, or migrate the whole app to Streamlit/Reflex 
   with proper navigation
3. The data layer in `library.py` is independent of the UI — switching 
   UI frameworks doesn't require redesigning queries

The library.py abstraction is specifically designed so that any UI 
change is a UI change, not a data layer rewrite.

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

### ✅ Phase 2: Quality Foundation (in progress)

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

### 🔄 Phase 3: Document Store + Library UI

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
(Hodgkin & Huxley, Hubel & Wiesel) extracted to 1-10 chunks each 
because they lack proper text layers.
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

---

## Deferred Improvements

### Better document pre-processing

Current extraction (PyMuPDF4LLM + Marker for scanned) produces usable but 
imperfect markdown. Specific issues observed:

- Mid-word breaks from PDF columns ("M isregistration", "W ith")
- Figure captions captured as headings
- Equations rendered inconsistently
- Tables sometimes lose structure
- Reference list parsing varies by paper format

A dedicated pre-processing pass — between extraction and chunking — could 
clean these systematically. Possible techniques: regex-based artifact repair, 
LLM-based "did this extract correctly?" check, format-specific cleaners.

Cost: might not be worth the cost but worth trying. LLM-based could be costly.

### Domain-aware tokenization / keyword embedding

Standard embedding models treat all tokens equivalently. Scientific text has 
*disproportionately important keywords* — drug names, protein names, technique 
names, mathematical operators — that carry most of the meaning of a sentence.

A potential improvement: identify a vocabulary of high-signal scientific terms 
(via TF-IDF on the corpus, or pre-built domain dictionaries like UMLS for 
biomedical), then either:

1. Boost their weight in BM25 retrieval
2. Generate auxiliary embeddings that emphasize these terms
3. Use them as explicit metadata for hybrid retrieval

Cost: medium. Value: probably real for technical scientific corpora.
