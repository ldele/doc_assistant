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

**Methodology rigor: ⚠ vibes-locked, no numbers behind the weights.** "Gave better hits during testing" was a qualitative impression from before the eval harness existed. The *architectural* choice (hybrid retrieval) is well-justified by the asymmetric failure modes of pure-vector vs pure-BM25; the *specific weights* (0.4/0.6) are essentially arbitrary. Re-measure via `scripts/run_eval.py` with a weight sweep when the `--bm25-weight` flag exists (small follow-up). Citation_overlap alone (deterministic, free) is enough for this; no answer-LLM needed.

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

**Methodology rigor: ⚠ moderate. Point estimates, no error bars.** The original measurement predates the eval harness; sample size (N), variance, and the case set are all unrecorded. The +1.40 lift on methodology questions is too large to be pure noise even at small N, so direction is trustworthy; magnitude is not. To re-validate, toggle `USE_PARENT_CHILD` and run `uv run python -m scripts.run_eval --with-llm-judge --repeat 3` against the current 35-case set — would give a defensible mean ± std comparison.

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

**Methodology rigor: ⚠ moderate. Strong shape, weak error bars.** The 3 → 5 → 8 → 10 → 12 → 15 sweep with a clear peak on *both* correctness and faithfulness is hard to attribute to noise (an accidental peak would not be coherent across two independent metrics). Sample size, variance, and the case set are unrecorded. To re-validate, the easiest path is a script-driven sweep over `TOP_K` values, persisting per-K runs in DuckDB and aggregating via `Store.aggregate_runs`. Existing `scripts/run_topk_sweep.py` is a pre-harness artefact and should be rewritten on top of the new harness.

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

**Methodology rigor: ✓ acceptable.** Two independent prompt designs both regressed against the same baseline — convergent evidence that the regression is mechanistic, not noise. Sample size is still unrecorded, but the qualitative consistency carries the conclusion. Re-validation low priority; if revisited, the bar is "show that MQ wins on *some* sub-population of questions" (e.g., genuinely ambiguous queries), which would imply a different design — query-type-conditional MQ rather than always-on.

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
1. Add pagination (Phase 8): `/library page:2` for 20-doc pages
2. Or migrate to a real sidebar/separate UI (Phase 8+) if Chainlit gets 
   better layout support, or migrate the whole app to Streamlit/Reflex 
   with proper navigation
3. The data layer in `library.py` is independent of the UI — switching 
   UI frameworks doesn't require redesigning queries

The library.py abstraction is specifically designed so that any UI 
change is a UI change, not a data layer rewrite.

### Enrichment-Layer Pattern (post-ingest, idempotent, sidecar by default)

Established by Phase 4 (`citations.py`, `metadata_extractor.py`) and adopted
as the standing pattern for all further data enrichment.

Primary ingest (Phase 1–3) is locked: extract → markdown → chunk → embed →
store. Any new derived data — citations, figures, tables, doc-level
vectors, future enrichments — ships as a **separate module + CLI runner**
that reads from the markdown cache (and the PDF source where needed) and
writes to its own sidecar table or manifest.

Rules:

- One module per enrichment kind (`citations.py`, `metadata_extractor.py`, 
  `figures.py`, `tables.py`, `doc_vectors.py`, ...).
- One CLI runner per module under `scripts/extract_*.py` with 
  `--dry-run` / `--apply` / `--force` / `--doc <hash>` flags.
- Idempotent: re-running on the same input produces the same output. Skip 
  docs that already have results unless `--force`.
- Sidecar by default. Splicing back into the markdown is only allowed when 
  the enrichment is *text-shaped* (e.g., tables). Binary artifacts (figures, 
  embeddings) live outside the markdown.
- No mutation of the primary chunk store from an enrichment module.

Trade-offs:

- Re-runnability is the goal. As extractors improve, every enrichment can 
  be re-run without touching primary ingest.
- Cost: small duplication of scaffolding (one CLI per module). Accepted.
- Risk: enrichment results can drift from the underlying markdown if the 
  cache is regenerated. Mitigated by content-only hashing — re-extraction 
  produces the same hash if the content is unchanged.

### Research Integrity Layer

A cross-cutting layer that records how each answer was produced and 
separates *evidence* from *AI interpretation* in synthesis-mode answers. 
Ships in three chunks across Phases 5, 6, and 9.

**Sources (influences, not specs):**

- AI Usage Cards (arXiv 2303.03886) — provenance card schema.
- PRISMA-trAIce (PMC12694947) — Phase 9 export target for AI-assisted 
  literature reviews.
- BE WISE framework (Frontiers, April 2026) — influence on the `human` 
  synthesis mode. Treated as a vendor framework, not adopted as a spec.
- Nature Methods — disclosure norm; satisfied as a byproduct of trAIce 
  export.

**Chunk 1 — Provenance card (Phase 5).** New `answer_records` table stores
query, retrieved chunk IDs + scores, reranker scores, model name, prompt 
version, token cost, latency, timestamp. Rendered as a collapsible card 
under each answer. CLI export `/export-record <answer_id>` → JSON. The 
eval harness consumes the same record schema.

**Chunk 2a — Dual interpretation layer (Phase 6).** Synthesis-mode answers
return two layers: *evidence* (retrieved passages, faithfulness-scored, no
synthesis) and *interpretation* (LLM synthesis, clearly labeled, with 
retrieval-derived uncertainty markers). Per-claim accept/reject/edit, 
persisted on the answer record. Pre-interpretation checkpoint (coverage + 
citation density + faithfulness) warns the user before generating.

**Chunk 2b — Reviewer agent (Phase 6).** ✅ Done (2026-05-28).
`src/doc_assistant/reviewer.py` + `answer_reviews` sidecar table.
Reviewer defaults to Haiku, runs ONLY when PR 5.1's heuristic 
confidence flags fire (cost discipline). 4-dim rubric: 
faithfulness, citation_density, hedging_adequacy (1-5 each) + 
unsupported_claims_count. **Reference-FREE**: judges answer vs 
retrieved evidence, not vs a ground-truth reference — distinct from 
the eval harness's `LLMJudgeScorer` which compares against expected 
answers. `temperature=0`, single-turn, no system prompt — same 
isolation contract as the eval judge. JSON-in-prompt (no tool-use; 
keeps consistency with eval judge). No auto-retry. `/review <id>` 
slash command for manual re-review. 14 unit tests. Schema is 
reviewer-kind-agnostic so a future human-review path slots into the 
same table.
*Deferred (idea, 2026-06-02):* the trigger policy is currently fixed —
auto-run on heuristically-flagged answers + manual `/review`. A later
option could expose the trigger as a user setting (e.g. `REVIEW_MODE =
flagged | always | manual`). Current behaviour is the wanted default;
this is an opt-in knob, not a change to it.

**Chunk 2c — Reviewer aggregation & self-improvement loop (Phase 6).** Counts
a categorical `failure_tag` enum across reviewer verdicts to find *systematic*
failure modes, anchored against the verified eval set to separate reviewer bias
from real system fault. **Minimum-N gate (locked).** The reviewer is a biased
sampler (runs only on flagged answers) over a small corpus, so a raw count is
noise with a label. A `failure_tag` is **not reported as actionable** until it
clears `MIN_FAILURE_TAG_COUNT` occurrences across `MIN_FAILURE_TAG_DOCS` distinct
documents (config-driven and user-tunable, with conservative defaults ~10/5
tuned on the first real distribution). Below the gate the aggregation reports
"insufficient evidence," never a suggested fix. Counts are always shown with
their denominator ("4 / 7 flagged"), never bare. v1 is instrumentation, not
auto-action; the action layer waits until accumulated answer records clear the
gate. Read-only aggregation, Enrichment-Layer Pattern.

**Judge/reviewer are pinned instruments.** The eval `LLMJudgeScorer` and the
Chunk 2b reviewer are measurement instruments, not generators: their model +
version are pinned and recorded per run, and swapping is an explicit logged
event. Chunk 2c's bias-vs-fault anchor is only falsifiable if the instrument
does not drift. The full evaluation strategy, scorer limitations, verified-10
headline rule, and the local-judge calibration gate live in
[`tests/eval/TESTING.md`](../tests/eval/TESTING.md).

**Chunk 3 — PRISMA-trAIce export (Phase 9).** When the literature review 
generator produces a review, it also produces a structured trAIce-aligned 
disclosure (JSON + markdown). Pulls from `answer_records` and the 
adjudication log; no new instrumentation needed.

**Mode flag.** `SYNTHESIS_MODE = human | ai` (default `ai`).

- `human` — AI returns evidence only; interpretation is the user's. 
  BE WISE-influenced path.
- `ai` — dual-layer with reviewer scoring.

Both modes share the same pipeline; the flag selects the output template.

**Why retrieval-derived uncertainty markers, not self-reported confidence.**
LLMs are systematically overconfident and the mapping between 
self-reported scores and actual correctness is non-linear and 
model-dependent. Building a calibration model (Platt scaling or similar) 
requires a labeled dataset and ongoing maintenance — out of scope for a 
single-user tool. Retrieval-derived markers (citation density, source 
diversity, reranker score spread) are observable, not self-reported, and 
map to instrumentation that already exists.

---

## Roadmap

Phases renumbered to absorb the doc-assistant-roadmap.md additions. See 
that document for the source of intent; this section records the locked 
phase plan.

---

## Production Engineering Standards

These apply across all phases. They are not deferred to Phase 6.

### CI/CD (GitHub Actions)

Every push and PR runs: ruff → mypy → pytest (coverage ≥70%) → bandit → pip-audit.
Merging on a red pipeline is never allowed. Branch protection enforced on `main`.

Rationale: mechanical checks before human review. Catching lint/type/security issues in CI is 10x cheaper than finding them in code review or production.

### Cross-machine toolchain — torch backend auto-detect, interpreter left to the box (added 2026-06-10)

The project must run unchanged on a GPU box (RTX, CUDA) and a CPU-only box. Two
toolchain hazards surfaced; one is fixed in-repo, the other is deliberately left
to the machine.

**Torch backend — `torch-backend = "auto"`, not a platform pin.** Earlier the uv
lock pinned `torch==2.12.0+cu130` for *all* `sys_platform == 'win32'`. That forced
the CUDA wheel onto CPU-only Windows boxes, where the transformer forward pass
segfaults (exit 139). Replaced with `[tool.uv] torch-backend = "auto"` (drop the
explicit `pytorch-cu130` index + the `[tool.uv.sources] torch` marker): uv probes
each machine's accelerator at sync time and resolves the CUDA wheel on the GPU box,
the CPU wheel everywhere else (incl. CI). One `uv sync` works on every machine; no
per-box torch swap. Rejected: (a) keeping the win32 pin + a documented manual CPU
swap (the prior workaround — recurring friction, easy to forget); (b) a GPU-opt-in
extra (`uv sync --extra gpu`) — explicit and deterministic, but adds a flag the GPU
box must remember; auto-detect needs nothing. Validated CPU-side (lock resolved the
`+cpu` wheel, full suite green); GPU-side wants one confirming `uv sync` on the RTX box.

**Interpreter OpenSSL — not pinned in-repo (deliberate).** Separately, the
uv-managed **Astral python-build-standalone** interpreter ships an OpenSSL that
hard-crashes (`no OPENSSL_Applink`) on *some* Windows boxes the moment any real SSL
client is built (`ssl.create_default_context`) — taking down every HTTPS call
(Anthropic, Ollama) and the SSL-touching tests. An official python.org CPython is
unaffected; the fix is to rebuild the venv on one. We did **not** encode this as a
repo-level `python-preference = "system"` + `.python-version`, because the GPU box
runs fine on the managed build and forcing system-Python resolution could break it.
Decision: keep it a documented remedy (`.claude/KNOWN_ISSUES.md` + README setup
note), not an enforced toolchain change — revisit only if more boxes hit it.

### Security tooling

- `bandit`: SAST on `src/`. HIGH findings block merge. MEDIUM acknowledged in PR.
- `pip-audit`: dependency CVE scan on every push. CVEs require fix or documented exception.
- `detect-secrets`: baseline committed. Pre-commit hook diffs against it. No secrets ever enter git history.

### Pre-commit (mandatory)

Hooks: ruff (lint + format), mypy, bandit, detect-secrets, file hygiene. Not optional.
`no-commit-to-branch` on `main` forces all changes through PRs.

### Structured logging

`structlog` with JSON output in staging/production. No `print()` in `src/`.
Every log entry carries: level, timestamp, module, event, and operation context.
Secrets and PII are never logged. Context binding per-request for the web UI.

### Exception hierarchy

`DocAssistantError` as base. Typed subclasses: `ExtractionError`, `IngestError`,
`PipelineError`, `StorageError`, `ExternalServiceError`. Exceptions chain with
`raise X from e` to preserve tracebacks. User-facing messages translated at UI boundary.

### Content-only hashing — implemented

Previous path+content hashing caused duplicate Document rows whenever a file
was moved, renamed, or re-extracted with a different extractor. This was a
data integrity issue blocking Phase 4 (citation graph depends on stable
document identity).

Change: `doc_hash(text, source)` → `doc_hash(text)`. SHA-256 of the extracted
markdown content only, truncated to 16 hex chars. Path is no longer part of
the identity.

Migration script: `scripts/migrate_to_content_hash.py` (dry-run + --apply).
Recomputes hashes in SQLite and both Chroma stores. Handles dedup collisions
when two paths had identical content by merging into the row with the highest
chunk count.

This also obsoletes `scripts/dedupe_documents.py` — the root cause (path
changes creating new hashes) no longer exists.

---

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
- 2.4 Experiment: semantic chunking — **reopened 2026-05-31.** Chunk *sizes*
  were never measured (only parent-child structure was). Sizes are now
  config-driven (`PARENT/CHILD/BASELINE_CHUNK_SIZE`); sweep via
  `scripts/sweep_chunking.py` through the Phase 5 eval harness. Update the
  locked-settings table with the winning config once measured.
- 2.5 Experiment: parent-child retrieval
- 2.6 Experiment: section-aware retrieval
- 2.7 Experiment: query rewriting / HyDE
- 2.8 Lock in best combination
- 2.9 Tech debt: content-only document hashing

Deliverable: a measurable improvement in answer quality, with numbers.

### ✅ Phase 3: Document Store + Library UI (complete)

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

**Phase 3 completion gate — all done (2026-05-21):**
- ✅ Content-only hashing implemented and migrated (27 docs, all 16-char content-only hashes)
- ✅ CI pipeline green (ruff, mypy strict, pytest ≥45% — was 70%, lowered to 45% pending integration tests, bandit, pip-audit advisory)
- ✅ Pre-commit hooks installed and baseline clean
- ✅ `.env.example` committed with all 8 env vars documented
- ✅ Library metadata routing extracted to `query_router.py`
- ✅ `chainlit_app.py` refactored to thin shell; business logic in `query_router.py` + `commands.py`

### ✅ Phase 4: Citation Graph

Goal: model relationships between documents. Both explicit (citations) and 
implicit (semantic similarity). Surface the structure of the literature.

**Done (2026-05-26 session — explicit edges):**
- Tier-1 regex citation extractor (`src/doc_assistant/citations.py`). 
  Two-tier design with LLM fallback was the locked decision; tier-1 
  measured at 22/27 docs (81%) on the existing corpus, 1,234 citations 
  parsed. Tier-2 LLM fallback deferred — measure first, escalate when 
  data warrants.
- Doc-level metadata extractor (`src/doc_assistant/metadata_extractor.py`) 
  for title / authors / year / DOI. Coverage on the 27-doc corpus: 27/27 
  title, 26/27 authors, 23/27 year, 7/27 DOI.
- Internal citation matcher (DOI → first-author-surname + year → fuzzy 
  title via stdlib `SequenceMatcher`). Cross-citation rate is 
  structurally low on the current corpus (mostly recent papers citing 
  classics not in the library).
- CLI runners: `scripts/extract_citations.py`, `scripts/extract_doc_metadata.py`. 
  Both idempotent (skip already-processed docs unless `--force`).
- Slash commands: `/cites`, `/cited-by`, `/graph` (Mermaid for ≤25-node 
  subgraphs).
- 45 unit tests across citations + metadata.
- GROBID and refextract evaluated and deferred — Docker + Java service 
  cost not justified by the recall data so far.

**Done (2026-05-28 session — implicit edges):**
- Doc-level vector enrichment (`src/doc_assistant/doc_vectors.py`). 
  Mean-pools chunk embeddings from the baseline Chroma store per 
  document, L2-normalises, and computes pairwise cosine similarity. 
  Returns directed top-K=10 nearest-neighbour edges per source above a 
  0.5 cosine threshold.
- Sidecar table `doc_similarities (source_document_id, target_document_id, 
  embedding_model, score, computed_at)` — composite PK so future Phase 5 
  embedder swaps (Feature 1) don't collide with existing rows.
- CLI runner `scripts/compute_doc_vectors.py` — `--dry-run` / `--apply` / 
  `--force` / `--doc <prefix>` / `--top-k` / `--threshold` flags. 
  Idempotent: refuses to overwrite without `--force`.
- `library.similar_docs(doc_id, limit=10, embedding_model=...)` query 
  joined to Document for filenames/titles.
- `/similar <id>` slash command in `commands.py`.
- 15 unit tests over the pure-numpy core (mean-pool, edge computation, 
  threshold + top-K behaviour).

**Locked design choices for doc vectors:**
- Persist edges only, not the mean-pooled vectors themselves. Vectors 
  are recomputed from Chroma on demand. Rationale: at personal-library 
  scale (10s–100s of docs) the recompute is seconds; a separate vector 
  table is bloat with no measurable benefit. Rejected: persist vectors 
  for incremental updates (premature at this scale).
- Source store: baseline Chroma (`CHROMA_PATH`), not PC. Fewer, larger 
  chunks per doc, simpler iteration, equivalent mean-pool semantics. 
  Rejected: PC child store (finer granularity but more chunks per doc; 
  no signal it improves the doc-level vector).
- Directed top-K, asymmetric. The cosine relation is symmetric but the 
  top-K trim makes the persisted edge set directional. Gives a stable 
  "most similar to A" UX. Consumers wanting symmetric edges can union 
  both directions. Rejected: undirected edges (loses "top-K of A" 
  intuition; would need an arbitrary disambiguation rule).
- O(N²) similarity is fine for the current scale. Rejected: ANN index 
  (premature; Phase 7 if and when the library grows past ~1000 docs).

**Remaining (operational, not architectural):**
- Apply the citation extraction, metadata backfill, and doc-vector 
  computation on the local DB. All three CLI runners exist; this is the 
  15-minute shell run flagged in CLAUDE.md.
- LNCS colon-separator format and multi-column extraction artifacts are 
  known tier-1 weaknesses; cosmetic, deferred.

### Phase 5: Embedding & Eval Foundation

Goal: domain-aware retrieval backed by reproducible measurement, plus 
the first deliverable of the Research Integrity Layer.

See `docs/doc-assistant-roadmap.md` for the source of intent.

- **Feature 1** — ✅ Done (2026-05-28). `EMBEDDING_MODEL` env var; 
  `src/doc_assistant/embeddings.py` registry + factory. Initial options: 
  `bge-base` (default → legacy `"langchain"` collection name for zero-migration 
  upgrade), `specter2` (academic, uses its own collection). 15 unit tests. 
  Per-folder routing (Feature 1b) gated on Feature 3 measurement.
- **Feature 2** — ✅ Done (2026-05-28). `src/doc_assistant/eval/` 
  package: `cases.py` (YAML loader), `results.py` (dataclasses), 
  `scorers.py` (5 scorers: ExactMatch, ContainsAll, CitationOverlap, 
  EmbeddingSimilarity, LLMJudge), `runner.py` (Runner with progress 
  callback + exception capture), `store.py` (DuckDB, 3-table schema, 
  context-manager), `report.py` (summary + diff), `adapters.py` 
  (RAGPipeline wrapper — only file with doc_assistant deps; 
  extractable per Feature 5). CLI `scripts/run_eval.py` with paid 
  scorers gated behind explicit flags. 43 unit tests; coverage 61%.
- **Feature 3** — ✅ Done (2026-05-28). 35-case eval set, run via the 
  harness against two embedders (BGE vs SPECTER2) with a hardened 
  reference-only LLM judge (`temperature=0`). 5 trials per model 
  (`--repeat 5`); reported as mean ± trial-mean std. **Result: bge-base 
  wins on the retrieval-side signals; the two are tied on the LLM judge.**

  | Scorer | bge-base (n=5) | specter2 (n=5) | Δ (bge − specter2) | Verdict |
  |---|---:|---:|---:|---|
  | `citation_overlap` (0-1) | **0.907 ± 0.000** | 0.887 ± 0.002 | +0.020 | Real (deterministic gap) |
  | `contains_all` (0-1) | **0.804 ± 0.013** | 0.767 ± 0.014 | +0.037 | Real (~4σ) |
  | `llm_judge` (1-5) | 2.209 ± 0.053 | 2.224 ± 0.092 | −0.015 | Tied (within noise) |

  The cross-encoder reranker and the answer LLM together level out the 
  embedder differential at the chunk level: what matters is which 
  documents got retrieved (bge wins) and whether the answer surfaces the 
  expected keywords (bge wins); the judge's rubric over the resulting 
  answers sees no meaningful difference.

  **Key finding — SPECTER2 lost because of training-task mismatch.** 
  SPECTER2 was trained for paper-level citation prediction over 
  abstracts; our task is chunk-level QA retrieval over full text. 
  bge-base's training corpora (MS MARCO, NQ, SQuAD, HotpotQA) are 
  much closer to QA retrieval. **Implication:** lock bge-base as 
  default; defer Feature 1b (per-folder embedder routing) until a 
  workflow emerges where SPECTER2's strengths apply (e.g., the 
  `/similar` paper-level task). Locked in `embeddings.py` registry; 
  re-confirmed empirically.

  **Provenance + reproducibility.** This comparison was measured on a 
  private, non-redistributable corpus, so the numbers above are kept 
  here as the record but cannot be reproduced by a third party. It is 
  **queued to be re-run on the public demo corpus** 
  (`tests/eval/cases.public.yaml` + `corpus_manifest.yaml`) so the 
  embedder evidence becomes reproducible alongside the rest of the 
  benchmark. Until then, treat the embedder verdict as provisional.

  **Methodology caveats:** 35 cases is small — effect sizes are 
  suggestive, not significant. LLM-judge means are low (~2.2/5) because 
  the rubric is strict reference-only grading; absolute scores are not 
  directly interpretable, the cross-model gap is the signal. 
  `embedding_similarity` is excluded — it uses the active embedder, so 
  the comparison is confounded across models (fix pending).
- **Integrity Chunk 1** — Provenance card. `answer_records` table; 
  collapsible card under each answer; `/export-record <id>` → JSON. 
  Hooks into existing `tracking.py`.
- **Provider layer (generation side of Feature 1)** — the config-driven-provider pattern Feature 1 applied to embeddings extends to generation. A normalized `LLMClient.complete()` protocol (`src/doc_assistant/llm.py`) with Anthropic + Ollama adapters backs the reviewer and the eval judge; the streaming chat path stays LangChain but reads `LLM_PROVIDER`/`LLM_MODEL`. **Local-first is the end goal, hybrid today** — the *generator* should run fully local (analysis + chat on Ollama, no `ANTHROPIC_API_KEY`). The *judge and reviewer are pinned instruments*, not generators: they default to a fixed, version-recorded reference model and only move to local once the local-judge calibration gate passes (see `tests/eval/TESTING.md`). Note: Feature 4c's VLM figure description is API-only and not part of the local path. Reviewer context isolation (evidence-only prompt, separate instance) is pinned by a guard test. Full design: `docs/specs/llm-provider-isolation.md`.

### Phase 6: Per-project routing + Figures & Tables + Dual-layer interpretation

Goal: turn the static embedding config into per-corpus routing; promote 
figures and tables to first-class content; ship the dual-layer 
interpretation + reviewer agent.

- **Feature 1b** — Per-project embedder routing. **Gated on Feature 3 
  results.** `Folder.embedding_model` column. Collection naming 
  `{folder_id}__{model_name}`. Cross-folder queries hit each collection 
  independently and the cross-encoder reranker resolves the mixed-space 
  merge.
- **Feature 4a** — table extraction pass. 🔄 In progress (2026-06-02);
  splice mechanics + extraction done, **extraction engine not yet final.**
  `src/doc_assistant/tables.py` + `scripts/extract_tables.py`. Tables go in
  one demarcated block appended to the cached `.md` (`<!-- tables:pdfplumber:begin
  … :end -->`), each tagged `<!-- table-extracted-by: pdfplumber page=N table=M -->`.
  Idempotent (re-splice replaces the block). Enters retrieval on the next
  re-ingest; never touches the chunk store.
  - **Detection is a shared page classifier (`regions.py`), not a table
    detector (locked).** A visual check (`scripts/debug_tables.py`) showed
    geometric detectors (pdfplumber, pymupdf `find_tables`) mis-read
    scientific *figures* as tables (a bar-chart grid → 13 "tables") and
    both mis-fire on shaded *prose*. Root-cause fix: classify each page
    once — table / chart / photo / figure / text — from three measured
    signals and route. (1) **"Table N" caption** (figures say "Figure N",
    prose has none); (2) **vector curve-density** — charts are 1k-78k
    curves vs 8-187 on table/text pages; (3) **raster image-area fraction**
    — figures cover 0.09-0.60 of the page vs 0 on table/text. A page is a
    table-candidate only if it has a table caption AND is neither
    chart- nor image-dominated, so a chart with a stray "Table" mention is
    not mis-routed. `tables.py` extracts only on classified table pages;
    content guards (`MAX_CELL_CHARS`, ≥`MIN_COLS` non-empty cols) are the
    second line of defence. Validated: no-table paper → 0; Table-1 paper →
    Table 1. This same classifier is the **Feature 4b foundation** (its
    chart/photo/figure verdict + image-area signal feed figure handling);
    v1 is page-level, per-region bbox splitting is the deeper 4b step.
    Thresholds (`CHART_CURVE_MIN`, `IMAGE_AREA_MIN`) measured on 2 eLife
    docs — tunable, validate on a wider corpus.
  - **Engine eval pending (RTX machine).** pdfplumber fragments multi-part
    tables and misses some the geometric detector doesn't see. Marker
    (ML-based, GPU-friendly, not a default dep) is the candidate better
    engine; `scripts/eval_marker_tables.py` compares them on the
    caption-gated pages. Run on the RTX box; then pick the engine.
  - **De-dup pending.** pymupdf4llm already emits a *lossy* inline copy of
    each table; once the engine is chosen, strip the inline copy (regex over
    the `<!-- page:N -->`-marked region) so the cache holds one clean table,
    not two. (User: "we don't need duplicate tables.")
  - **Verification (planned).** A table-retrieval eval case (ask a table
    question → assert the table chunk is retrieved post-ingest) is the
    chosen check; a hand-verified gold-set fidelity scorer is a roadmap
    future, not built now.
- **Feature 4b** — Figure region detection + caption pairing (OpenCV, 
  no LLM cost). Sidecar `figures` table; images on disk under 
  `data/figures/{doc_hash}/`. Caption text stays in the markdown.
- **Feature 4c** — VLM figure description (gated). Anthropic tool-use 
  with Pydantic-validated schema. `caption + VLM description` embedded as 
  a chunk with `chunk_type='figure'`. `MAX_VLM_CALLS_PER_DOC` budget.
- **Integrity Chunk 2a** — Dual interpretation layer. `SYNTHESIS_MODE 
  = human | ai` (default `ai`). Evidence + interpretation as separate 
  output sections. Per-claim adjudication.
- **Integrity Chunk 2b** — Reviewer agent. Cheaper LLM scores the 
  interpretation against a structured rubric (faithfulness, citation 
  density, hedging adequacy, claims-without-sources). No auto-retry.
- **Integrity Chunk 2c** — Reviewer aggregation & self-improvement loop.
  Adds a categorical `failure_tag` enum to the reviewer (countable
  patterns alongside free-text notes), aggregates `AnswerReview` over
  time, and resolves the central ambiguity — *reviewer bias vs system
  fault* — by anchoring against the golden eval set (flagged-but-verified-
  good ⇒ reviewer bias; flagged-and-low-correctness ⇒ system fault).
  Read-only aggregation, Enrichment-Layer Pattern, no auto-remediation.
  An unanchored "pattern in suggestions" must not ship — it can't tell
  reviewer fault from system fault. **Minimum-N gate:** a tag is actionable
  only past `MIN_FAILURE_TAG_COUNT` / `MIN_FAILURE_TAG_DOCS` (tunable;
  ~10/5 default); below that it reads "insufficient evidence." Counts always
  carry their denominator. v1 is instrumentation, not auto-action. See the
  Research Integrity Layer subsection above and roadmap Chunk 2c.
- **Engineering: chunking sweep** (reopens Phase 2.4). Chunk sizes are now
  config-driven (`PARENT/CHILD/BASELINE_CHUNK_SIZE`, shipped 2026-05-31);
  `scripts/sweep_chunking.py` measures size grids through the eval harness.
  Defaults are historical, not measured — lock from the sweep result.

### Phase 7: Gap Detection and Cartography

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
- **Feature 6 — Self-organizing wiki / synthesis layer.** A derived markdown
  layer over the corpus: per-topic notes (summary + tags + `[[links]]` +
  citations) distilled from retrieval, regenerable, sidecar `data/wiki/`.
  The Karpathy LLM-wiki pattern (proven in the cross-project atlas) applied
  *on top of* RAG, not as a replacement. Its real payoff here is making gap
  detection computable — a note with thin citations and no links is a
  structural gap signal (reuses `provenance.py` confidence heuristics), not
  just an LLM opinion. Living precursor to the Phase 9 review generator. See
  roadmap Feature 6.

### Phase 8: UI Polish

Goal: design pass on everything built so far.

- Visual design refinement
- Animations and micro-interactions
- Better empty states, loading states, error states
- Demo recording
- Onboarding flow for new users
- Polish on the adjudication UI from Chunk 2a if it shipped rough.

Note: content-only hashing completed in Phase 3 (was moved from Phase 8 to Phase 3 completion gate).

### Phase 9: Literature Review Generation

Goal: synthesize a structured review across documents in the library on a 
chosen topic. The endgame feature.

Requires everything from Phases 2–7 to work well:
- Reliable multi-document retrieval (Phase 2)
- Document and section structure as data (Phase 3)
- Citation graph for grounding (Phase 4)
- Domain-aware retrieval + eval-backed quality (Phase 5)
- Structured figures + dual-layer interpretation (Phase 6)
- Coverage awareness so the system knows what it's missing (Phase 7)

Capabilities:
- Per-paper synthesis: what does each paper contribute on the topic?
- Cross-paper synthesis: agreements, disagreements, evolution of ideas
- Citation-grounded claims: every assertion linked to specific passages
- Structural output: introduction, main themes, methodological comparison, 
  open questions, references
- Configurable scope: by folder, by topic, by date range, by document set
- Export: markdown, PDF, or BibTeX-aware Word document
- **Integrity Chunk 3** — PRISMA-trAIce export. Structured methodology 
  disclosure alongside the generated review, pulled from `answer_records` 
  and the adjudication log.

---

## What I'd Position This As

> A personal research workspace that combines local-first RAG with explicit 
> modeling of the literature graph, designed to help a researcher identify 
> what they know, what they don't, and what to read next. Unlike general-
> purpose RAG tools, it treats the document collection as a first-class 
> object to curate and analyze, not just a corpus to search.

Keywords: *personal*, *local-first*, *literature graph*, *knowledge gaps*, *what to read next*.

---

## Forward-looking considerations (added 2026-05-28)

The architecture is well-positioned for the current ~50-doc personal-library 
scale and handles ~500 docs without changes. Several trajectory questions are 
worth keeping in view as the project grows; none requires action now, but 
none should be silently undermined by short-term decisions either.

### Eval set as a living artefact

`tests/eval/cases.yaml` ships with 35 cases. The expected pattern: every real 
question the user asks the system in earnest is a candidate eval case. The 
file should grow with use — 100 cases by Phase 6, 200+ by Phase 9. Each new 
case is a regression test against future PRs. The harness (`scripts/run_eval.py` 
with `--repeat`) is built to scale linearly with the case count.

**Risk:** authored expected_answers go stale as the corpus changes or as 
the user's understanding sharpens. Periodic review needed — the 
`metadata.author_verified: false` flag is the marker.

### Cost discipline

Once `/chat` is in daily use, Haiku token spend matters. The provenance 
card (PR 5, Integrity Chunk 1) is the data primitive that makes cost 
analysis evidence-based — every answer carries its token cost. Options 
that become available once the records exist:

1. **Answer cache** keyed by `(query_hash, retrieved_chunk_ids, model, prompt_version)`. 
   Same question with the same context returns the same answer for free.
2. **Tiered LLM routing**: Haiku for simple lookups, Sonnet for hard 
   synthesis, Ollama for routine queries. The reviewer agent in 
   Chunk 2b will need this anyway (cheap model judges expensive model's 
   output).
3. **Per-session cost budgets**: warn at $X, hard-stop at $Y.

None of these is built yet; the schema for PR 5 should make all three 
feasible later without re-instrumenting.

### Database scale story

| Library size | What works | What needs work |
|---|---|---|
| **~50 docs** (now) | Everything | — |
| **~500 docs** | Chunk retrieval, BM25 in-memory, O(N²) doc similarity | None |
| **~2-3k docs** | Chunk retrieval still fine (Chroma) | BM25 rebuild slow; O(N²) similarity edges (~5M pairs) becoming uncomfortable |
| **~5k+ docs** | Chroma + reranker still scale | BM25 needs persistence; similarity needs ANN index (FAISS, hnswlib); SQLite joins start to feel slow |
| **~50k+ docs** | None of the current architecture as-is | Need Postgres or sharded SQLite; persistent BM25 (Tantivy / Lucene); ANN; possibly distributed |

**Architectural invariants to preserve so the scale-up isn't a rewrite:**

- `Document.id` is a UUID, not an auto-increment — works across multiple DBs / shards.
- Sidecar tables (`citations`, `doc_similarities`, soon `answer_records`) decouple from the chunk store — can be migrated independently.
- All retrieval goes through `pipeline.retrieve()` — single chokepoint to swap from in-memory BM25 to a persistent index.
- Embedding factory routes model choice through one function — adding ANN-backed collections is a Chroma-config change, not a pipeline rewrite.

**What would actually trigger a database redesign:** multi-user (next section).

### UI ceiling

Chainlit covers chat well. It will hit limits at:

- **Graph visualisation** (Phase 6+) — citation subgraphs beyond ~25 nodes are already at the inline-Mermaid limit.
- **Adjudication UI** for Chunk 2a (Phase 6) — per-claim accept/reject/edit needs richer interaction than Chainlit's element API supports cleanly.
- **Library browser** with filters, sorting, bulk actions.

Decision deferred to Phase 8. Three plausible directions:

1. **Reflex** — Python full-stack, lowest migration cost.
2. **FastAPI + custom React frontend** — most flexible, most work.
3. **Streamlit** — quickest to prototype, weakest at custom interactions.

The data layer (`library.py`, `commands.py`, `pipeline.py`) is UI-agnostic precisely so this swap doesn't ripple. Keep it that way.

### Multi-user collaboration

Currently single-user assumptions are baked in only minimally:
- One SQLite DB
- No auth
- One `data/` directory
- No notion of "who created this tag / who adjudicated this answer"

Going multi-user requires:
1. **Auth** (out of scope until needed — could be a Reflex/FastAPI add-on, not a from-scratch design).
2. **Per-user scoping** on Documents, Folders, Tags, AnswerRecords (the things that *belong* to a user) — vs *shared* Documents (a lab corpus).
3. **DB redesign**: SQLite is single-writer; multi-user concurrent edits need either WAL-mode tuning, server-mode (`sqld`), or a switch to Postgres.
4. **`created_by` and `modified_by` columns** on every mutable table.

**Don't bake in single-user assumptions you can't reverse.** The current 
schema mostly avoids this — UUIDs everywhere, sidecar tables. The 
`/folders` and `/tags` features (when added) need to think about per-user 
vs shared from day one.

**When to revisit:** the moment a workflow emerges that involves >1 person 
on the same corpus. Until then, the cost of designing for it now exceeds 
the cost of deferring.

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

- **Citation extraction quality vs. external API quality.** TBD — revisit 
  during Phase 7 (Gap Detection) when Semantic Scholar / OpenAlex / arXiv 
  integrations land and can provide a ground-truth comparison.

- **Tier-2 LLM citation fallback.** Deferred until corpus grows or 
  no-section docs become problematic.

- **MCP server interface (external tool access).** Open / unscheduled.
  Expose the RAG pipeline as an MCP server so external MCP hosts — Claude
  Desktop, claude.ai connectors — can call the local library as a tool (e.g.
  a `search_library` / `ask` tool), letting a paid Claude subscription query
  the user's own documents. Architecturally low-friction: another **thin
  entrypoint** (`apps/mcp_server.py`) over `pipeline.py`, no core changes,
  consistent with the `apps/` boundary rule. Open nuance: Claude Desktop can
  talk to a local **stdio** MCP server directly, but claude.ai **connectors**
  need a reachable HTTP endpoint + auth — a real consideration for a
  local-first app. Not yet scoped to a phase.

---

## Deferred Improvements

### CI coverage floor — raise from 40% over time

Current `--cov-fail-under=40` in `.github/workflows/ci.yml`. Was 70% pre-Phase-3,
lowered to 45% then 40% as `pipeline.py` and `ingest.py` are I/O-heavy and
hard to test without a real Chroma/SQLite/embeddings stack. Phase 4 added
~52 tests against pure-logic citations + metadata modules.

Path to raise:
- Each new phase: add at least one integration test that hits a real (temp)
  DB or Chroma store. Phase 4 added one (`tests/integration/test_citation_pipeline.py`).
- Phase 5 work on retrieval-side gap detection — add tests that exercise
  `pipeline.retrieve` with a small frozen Chroma fixture.
- Raise floor in 5%-point increments after each phase ships, never lower it.

Target: 60% by end of Phase 6. Not a strict deadline — coverage is a lagging
indicator of "did we write the tests we should have", not a goal in itself.

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

### ~~pdfplumber for table extraction~~ — promoted to Phase 6 (Feature 4a)

Was deferred; now scheduled as Feature 4a under the Phase 6 enrichment layer.
See `docs/doc-assistant-roadmap.md` and the Phase 6 entry above for the
implementation plan.

### Expose remaining retrieval knobs — toward a config-complete RAG sandbox

Most of the pipeline is already config-swappable (embedder, chunk sizes, `TOP_K`,
parent-child, multi-query, PDF extractor, LLM/reviewer/judge backends) and the eval
harness can measure the effect — so the project already *functions as* a local RAG
sandbox. Three knobs are still hardcoded, which is exactly what keeps that
description qualified rather than absolute:

1. **BM25 / vector weights** — literal `[0.4, 0.6]` in `pipeline.py` (`EnsembleRetriever`).
   Expose as `BM25_WEIGHT` / `VECTOR_WEIGHT` (or a single `HYBRID_WEIGHTS`) env knob.
2. **Reranker** — `BAAI/bge-reranker-base` hardcoded in `pipeline.py`, always-on. Give
   it a registry + factory (mirror `embeddings.py`) and a `USE_RERANKER` toggle.
3. **General config sweep** — only `sweep_chunking.py` exists. Generalize it into a
   grid runner over any of the above (TOP_K, weights, reranker, toggles), reusing the
   eval harness, so "vary X and measure" isn't manual for everything but chunking.

Each is a small extension of an existing pattern (the embedder registry and the chunk
sweep already prove it). Sequence them behind the higher-priority Phase 6 nodes;
landing all three makes the "local RAG sandbox" claim in the README unqualified.
Once exposed, these knobs feed the **Phase 8 settings page** (user-facing sandbox, with
the benchmarked default pre-selected — see `doc-assistant-roadmap.md` → Phase 8).
Noted 2026-06-04.

### SPECTER2 for paper-level similarity (gated on use case)

Phase 5 / Feature 3 measurement showed SPECTER2 loses to bge-base on
chunk-level QA retrieval — a training-task mismatch (SPECTER2 was
trained for paper-level citation prediction over abstracts; we feed it
chunks of methods/results text). SPECTER2 stays registered but unused
by default.

**However**, paper-level similarity (the `/similar` task) is a useful
research feature — finding related papers to one you're reading is
exactly what SPECTER2 was built for. Implementation sketch when the
workflow becomes worth it:

1. Add an abstract extractor to `metadata_extractor.py` (regex/heuristic
   over the markdown cache; LLM fallback for failures).
2. Extend `doc_vectors.py` with a per-model strategy: BGE mean-pools
   chunks (current); SPECTER2 embeds `title + abstract` directly.
3. Run `compute_doc_vectors --apply --force` with
   `EMBEDDING_MODEL=specter2`. SPECTER2-similarity edges live alongside
   the BGE ones thanks to the existing `embedding_model` PK column.
4. `/similar` chooses which set to surface (or shows both side-by-side
   for comparison).

**Gate:** ship only when paper-level `/similar` is a regular workflow
worth optimising. Effort: 1 evening + measurement to verify
SPECTER2-on-abstracts beats BGE-mean-pooled-chunks for this task.
Currently no evidence either way.

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

### ~~Known data integrity edge cases~~ — resolved

Resolved by content-only hashing. Documents no longer produce new hashes
when moved or re-extracted. The `dedupe_documents.py` script is obsolete;
the migration script `scripts/migrate_to_content_hash.py` handles any
remaining inconsistencies from the old scheme.