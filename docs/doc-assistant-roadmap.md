# doc_assistant — additions roadmap

Features to add on top of the current doc_assistant repo (github.com/ldele/doc_assistant). Purpose: strengthen the portfolio story for AI/ML/medtech roles by demonstrating domain-aware engineering, reproducible evaluation, and measured improvements.

---

## Goals

1. Show domain-aware embedding selection (academic / biomedical text).
2. Demonstrate reproducible evaluation methodology — not vibes.
3. Quantify the impact of a design change with real numbers.
4. Keep the eval harness extraction-friendly so it can later become a standalone repo.

---

## Feature 1 — Config-driven embedding layer

**What:** Make the embedding model swappable via env config instead of hardcoded.

**Why:** Different content domains benefit from different embeddings. Hardcoding `bge-base` is a missed opportunity for academic papers, which are doc_assistant's main use case.

**Implementation**

- Add `EMBEDDING_MODEL` env var. Options:
  - `bge-base` (current default, general-purpose).
  - `specter2` (academic papers — trained on citation graphs).
  - `pubmedbert` or `medcpt` (biomedical literature).
- Embedding factory function that returns the right model loader.
- Vector store handling: collections are dimension-specific. Either:
  - **Option A:** one Chroma collection per embedding model — fast switching, more disk. *Recommended.*
  - **Option B:** single collection, force re-ingest on model change (`--reset-index` flag).
- Document the trade-off in `docs/decisions.md`.

**Effort:** 1–2 evenings.

---

## Feature 1b — Per-project embedder routing

**What:** Let each user-defined project (Folder) declare its own embedding model. Ingest and retrieval automatically use the right embedder for the project a document belongs to.

**Why:** A personal library is rarely single-domain. Academic + biomedical + CS papers each have a better-fitting embedder. Routing by project converts Feature 1's static config switch into a dynamic, per-corpus quality lever — and it makes the portfolio story concrete: *"I measured X vs Y, found Y wins on biomedical content, so I added per-project routing."*

**Gate:** Ship only after Feature 3 results exist. Otherwise this is a switch with no evidence behind it, which is a weaker story than "switch backed by numbers."

**Implementation**

- Add `embedding_model` column on the `Folder` table. Default = `EMBEDDING_MODEL` env value (preserves current behavior for any folder created before the migration).
- Collection naming: `{folder_id}__{model_name}`. Already dimension-segregated by model name, so collisions are impossible.
- Ingest path: reads `folder.embedding_model` → resolves via the Feature 1 factory → writes to the matching collection.
- Query path (scoped to folder): queries the folder's collection. Trivial.
- **Query path (unscoped / cross-folder)**: query each affected collection independently → merge candidates → cross-encoder rerank resolves the mixed-space problem (the reranker is embedding-agnostic). Adds latency proportional to number of collections hit. *This is the only non-trivial piece.*
- BM25 is dimension-agnostic and stays global.
- UI: dropdown on folder create/edit to pick the embedder. Changing it triggers re-ingest of that folder only.

**Trade-offs**

- Disk: linear in `(folders × models actually used)`. Bounded in practice.
- Cross-folder query latency: rises with number of collections hit. Acceptable for a personal library (10s of folders, not 1000s).
- Eval becomes per-project, not global. Adds rows to the eval matrix but doesn't change the harness.

**What NOT to do**

- Don't keep a duplicate "global" BGE collection. Storage doubles, sync gets fragile.
- Don't allow changing a folder's embedder without re-ingest. Mixed embeddings in one collection are meaningless.

**Effort:** 2 evenings on top of Feature 1.

---

## Feature 2 — Eval harness module

**What:** A module inside doc_assistant that runs the RAG pipeline against a versioned eval set, scores outputs, and tracks results over time.

**Why:** Most LLM engineers iterate on prompts via vibes. An eval harness puts you in the top 5%. Risklick's job posting literally asks for "evaluation of model outputs against scientific and operational benchmarks".

**Build it inside doc_assistant first, with extraction in mind.**

**Layout**

```
src/doc_assistant/eval/
  __init__.py
  runner.py        # generic: runs eval set against any callable
  scorers.py       # generic: deterministic, LLM-judge, embedding similarity
  store.py         # generic: DuckDB persistence
  report.py        # generic: diff between runs
  adapters.py      # doc_assistant-specific: wraps the RAG pipeline
```

**Rule:** everything except `adapters.py` imports nothing from `doc_assistant.*`. This lets Feature 4 happen cleanly.

**Scorer types to support**

- Deterministic: exact match, regex, JSON schema validation, citation overlap, latency, token cost.
- LLM-as-judge: Claude scores faithfulness, relevance, completeness on a 1–5 rubric.
- Embedding similarity: cosine similarity between output and reference answer.

**Effort:** 1 weekend for v0 (minimal — 1 scorer, basic runner, DuckDB store).

---

## Feature 3 — Golden eval set + measured comparison

**What:** A versioned eval set of 30–50 questions over a fixed paper corpus, plus a comparison run across embedding models.

**Why:** Without measurement, Features 1 and 2 are just claims. With numbers in the README, they're credentials.

**Implementation**

- `tests/eval/cases.yaml` — 30–50 questions with expected answers or expected source documents.
- Run the eval harness across:
  - `bge-base` (baseline).
  - `specter2` (academic embedding).
  - Optionally a biomedical model.
- Persist results in DuckDB. Generate a comparison table.
- Add a "Benchmark results" section to README with retrieval@k, faithfulness scores, and a short interpretation.

**Effort:** 1 day (writing the eval set is the bottleneck).

---

## Feature 4 — Figure & table understanding (vision + structured extraction)

**What:** Make figures, charts, and tables in scientific papers first-class, searchable content — not lossy text artifacts.

**Why:** A large share of the actual signal in a scientific paper lives in figures, charts, and tables. The current markdown pipeline flattens these into captions at best, garbage at worst. Borrowed from the financial-anomalies project, which uses pdfplumber + OpenCV + gated VLM to extract structured signal from PDFs. The pattern transfers directly; only the goal changes (extract for retrieval vs flag for anomalies).

**Implementation**

Three independent sub-features, ordered cheap → expensive:

- **4a. pdfplumber pass for tables** (promote from `decisions.md` → Deferred Improvements).
  - Run after primary extraction. Detect table regions, extract with pdfplumber, splice structured markdown tables back into the document.
  - Each table becomes a chunk with explicit row/column structure preserved in markdown.
  - Effort: 1 evening.

- **4b. Figure region detection + caption pairing (OpenCV, no LLM cost).**
  - PyMuPDF already exposes image blocks (`block_type=1`); OpenCV refines region boundaries and detects chart-like regions (line-detection, pixel variance, contour heuristics).
  - Pair each figure region with its caption via nearest-text-block heuristic ("Figure N: …").
  - Persist a **figure manifest sidecar** alongside the markdown cache: `{doc_id, page, bbox, caption, image_path}`. Markdown stays the primary intermediate (preserves the `decisions.md` core call).
  - Effort: 1 weekend.

- **4c. VLM figure description (gated).**
  - For each figure in the manifest, call Claude vision with a schema-first prompt: `{figure_type, summary, key_quantities, axes, trend}` (Pydantic-validated).
  - Embed `caption + VLM description` as a chunk linked back to the figure's bbox.
  - **Gating:** skip figures already well-described by their caption (length + caption-only embedding similarity check). Enforce `MAX_VLM_CALLS_PER_DOC` budget.
  - Effort: 1 weekend.

**Schema-first LLM call (port from financial-anomalies).** Every VLM call uses Anthropic tool-use with a JSON Schema derived from a Pydantic model. No string parsing.

**Eval hook.** Add a "figure retrieval" scorer to the eval harness (Feature 2): given a held-out caption, does the system retrieve the right figure?

**Effort total:** 2–3 weekends across the three sub-features. Each ships independently.

**Trade-offs**

- Markdown-as-universal-intermediate is preserved; the figure manifest is a sidecar, not a replacement.
- VLM cost is the main risk — gating + per-doc budget keeps it bounded.
- Caption-only chunks are a useful baseline if VLM is skipped (4b alone is shippable).

---

## Feature 5 — Extract eval harness as standalone repo (later)

**What:** Pull the generic eval-harness module out of doc_assistant into its own repo: `llm-eval-harness`.

**Why:** Standalone repo is a stronger portfolio artifact. Two pieces (doc_assistant + eval harness) tell a better story than one.

**Trigger:** After Features 1–3 are shipped and you've talked through them in one interview. By then the design is battle-tested.

**Process**

- Copy `runner.py`, `scorers.py`, `store.py`, `report.py` into a new repo.
- Write a clean README positioning the tool as general-purpose.
- Add a second adapter (e.g. for multi-agent-llm-system) to prove reusability.
- Add minimal docs + a usage example.

**Effort:** 1 weekend, post-Risklick process.

---

## Implementation order

| Order | Feature | Effort | Status |
|-------|---------|--------|--------|
| 1 | Config-driven embedding layer + SPECTER2 | 1–2 evenings | Not started |
| 2 | Eval harness v0 (inside doc_assistant) | 1 weekend | Not started |
| 3 | Golden set + BGE vs SPECTER2 comparison + README writeup | 1 day | Not started |
| 1b | Per-project embedder routing | 2 evenings | Gated on Feature 3 results |
| 4a | pdfplumber table-extraction pass | 1 evening | Not started |
| 4b | Figure region detection + caption pairing (OpenCV) | 1 weekend | Not started |
| 4c | VLM figure description (schema-first, gated) | 1 weekend | Not started |
| 5 | Extract eval harness to standalone repo | 1 weekend | Deferred — post-Risklick |

---

## What NOT to do

- Don't refactor doc_assistant's overall architecture. Low ROI for the recruiter call.
- Don't add SPECTER2 *and* PubMedBERT *and* MedCPT at once. Pick one (SPECTER2).
- Don't over-engineer the eval harness v0. Pydantic + pytest + DuckDB + Anthropic judge. No frameworks.
- Don't extract the standalone repo before you've used the integrated version in at least one real conversation.
- Don't port the financial-anomalies `DocumentModel` (bbox-preserving through the whole pipeline). Conflicts with the markdown-as-universal-intermediate decision. Use a figure-manifest sidecar instead.
- Don't try to synthesize a scientific-paper corpus for eval. Curate a held-out caption set from your own library instead.
- Don't ship 4c (VLM description) before 4b (figure detection + manifest). Caption-only chunks are a working baseline; 4c is a quality lever on top.

---

## Talking points for interviews (once shipped)

- "I swapped in SPECTER2 embeddings for academic content and measured a [X%] improvement in retrieval@5 over BGE on a 40-question eval set."
- "I built a small eval harness inside the project — generic enough that I'm pulling it into its own repo — that catches regressions between prompt or model changes."
- "The takeaway from that work was that domain-adapted retrieval matters more than the LLM swap. Most of the quality lives in the wrap-around, not the model."
- "Figures and tables carry a lot of the signal in a scientific paper, and standard markdown extraction drops most of it. I added a pdfplumber pass for tables and an OpenCV + gated VLM pass for figures, with a per-doc VLM budget so cost stays bounded. Figure captions and VLM-generated descriptions get embedded as their own chunks linked back to a page bbox."
- "The cheap-first, expensive-second gating pattern came from a different project — a financial-report anomaly detector — and ported cleanly because the underlying problem is the same: PDFs need multiple extractors layered, and you want LLM calls only where deterministic methods can't reach."
- "Once the eval showed embedder choice mattered per domain, I added per-project routing — each user-defined project picks its own embedder, stored as a separate Chroma collection. Cross-project queries hit each collection independently and the cross-encoder reranker resolves the mixed-space merge."
