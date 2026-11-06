<!-- status: active · updated: 2026-11-05 (the screens: components move to screen folders, ADR-064; 2026-11-01: Getting started and Provider & model; the letter-spacing domain, ADR-056; 2026-10-17: the concept panel's term signals route; 2026-10-01: CI's secret scan blocks — security S-13) · class: living -->

# Architecture

## High-level flow

```
Documents (PDF/EPUB/HTML/DOCX/MD)
↓
Extractors → Markdown cache (data/cache/)
↓
Chunker (markdown-aware, parent-child)
↓
Embeddings (BGE-base) → Chroma vector store (data/chroma/)
             ↕
         SQLite document store (Folder → Document → Part → Chunk)
↓
Hybrid retrieval (keyword arm = on-disk SQLite/FTS5 index `sparse_index.py`, ADR-036/038 · vector arm = Chroma; `BM25_WEIGHT` 0.4/0.6 is inert post-rerank, measured 2026-07-03) → CANDIDATE_K (default 20) candidates per retriever
↓
Cross-encoder reranker → TOP_K (default 10) parents (parent context returned)
↓
LLM (Claude or local Ollama) → streamed answer with citations
```

### Full pipeline (Mermaid)

The ingest path, the post-ingest enrichment layers, and the query→answer path,
with the stores they read/write:

```mermaid
flowchart TD
    subgraph ING["Ingest (incremental)"]
        SRC["Sources<br/>PDF · EPUB · HTML · DOCX · MD"] --> EXT["extractors.py"]
        EXT --> CACHE[("Markdown cache<br/>data/cache")]
        CACHE --> CHUNK["ingest/chunking.py<br/>table-aware parent–child chunker"]
        CHUNK --> EMB["embeddings.py<br/>BGE-base"]
    end

    subgraph ENR["Enrichment layers — post-ingest · idempotent · sidecar"]
        REG["ingest/regions.py<br/>page classifier"] --> TBL["ingest/tables_marker.py / ingest/tables.py<br/>Marker · pdfplumber"]
        CIT["ingest/citations.py"]
        MD["metadata_extractor.py"]
        DV["doc_vectors.py"]
        KNW["knowledge/ (ADR-023)<br/>keywords · concept skeleton (Node A/B) · wiki<br/>gaps · epistemics · graph view"]
    end

    subgraph ST["Stores"]
        CHROMA[("Chroma<br/>vectors")]
        SQL[("SQLite<br/>Document · DocumentMeta · SourceFile · Citation · DocSimilarity<br/>AnswerRecord · AnswerClaim · AnswerReview<br/>(+ Concept* · chunk_epistemics · Figure · Keyword sidecars)")]
    end

    subgraph QRY["Query → Answer"]
        Q["User query"] --> ROUTE["query_router.py<br/>library vs content"]
        ROUTE -->|content| RET["pipeline.retrieve<br/>FTS5 sparse arm (sparse_index.py) + vector → CANDIDATE_K=20 candidates/retriever"]
        RET --> RR["cross-encoder rerank<br/>→ TOP_K=10 parents (+ scores)"]
        RR --> GEN["LLM generate<br/>Claude / Ollama (cited)"]
        GEN --> SYN["synthesis.py — Chunk 2a<br/>evidence (deterministic) + AI interpretation<br/>per-claim markers from rerank scores"]
        SYN --> PROV["provenance.py<br/>answer record + confidence signals"]
        PROV -->|flagged only| REV["reviewer.py<br/>LLM reviewer"]
    end

    EMB --> CHROMA
    CHUNK --> SQL
    CIT --> SQL
    MD --> SQL
    DV --> SQL
    KNW --> SQL
    TBL -. splices tables .-> CACHE
    CHROMA --> RET
    SQL --> ROUTE
    RR --> PROV
    SYN --> SQL
    REV --> SQL
```

## Module responsibilities

| Module | Role | Public contract |
|---|---|---|
| `doc_assistant.config` | Paths, env vars, feature flags | Read-only after init; no side effects |
| `doc_assistant.extractors` | Convert any supported format → markdown | Returns `str`; raises `ExtractionError` on failure |
| `doc_assistant.ingest` (package — pipeline: `cache` · `chunking` · `store` · `cleanup` · `registry` (S1 `SourceFile` source registry + selection-scoped ingest) + `__init__` orchestration / `__main__` CLI; document-feature extraction: `citations` · `tables` · `tables_marker` · `figures` · `regions`) | Extract, chunk, embed, store; orphan cleanup + partial-write self-heal; source scan/select (`--files`/`--dry-run`, exclude flags); table/figure/citation extraction (sidecar) | Idempotent per content hash; per-document failures isolated; selection never bypasses the locked six-stage ingest |
| `doc_assistant.pipeline` | RAG runtime: retrieve, rerank, generate | Returns `Answer` with citations; raises `PipelineError` |
| `doc_assistant.chat_controller` (package — `session` · `views` · `events` · `helpers` · `controller`; direction is strictly session/views → events/helpers → controller) | UI-agnostic turn orchestration | Yields `TurnEvent`s → `TurnResult`; no UI-framework import (PR-M0) |
| `doc_assistant.health` | Document health scoring and classification | Pure function; no I/O; returns `HealthResult` |
| `doc_assistant.library` (package — `models` · `documents` · `pins` · `folders` · `keywords` · `chunks` · `citations` · `similarity` · `source_view` · `spacing`; sub-domain names match `apps/api/routers/library/`) | Document store queries (browse, filter, tag) + the Library's write paths: `DocumentMeta` overrides (ADR-013) and `delete_document` (ADR-014 — trash-first source-file recycle, then row/meta/chunks/figures/cache) | Queries + two explicit, ADR-recorded write paths; UI-framework-agnostic |
| `doc_assistant.knowledge` (package — `keywords` · `keyword_families` · `concept_skeleton` (Node A) · `concept_skeleton_enrich` (Node B) · `concept_curation` · `concept_semantics` · `concept_graph_view` · `wiki` · `gaps` · `gap_suggest` · `epistemics`) | The Phase-7 knowledge layer: mined vocabulary, curated concept skeleton, wiki notes, gap detection, chunk epistemics — all derived *from* the corpus (ADR-023) | Enrichment-Layer Pattern throughout: additive sidecars, idempotent `scripts/` runners, never writes the chunk store; the answer path reads it but never depends on it |
| `doc_assistant.prompts` | Prompt templates | Pure string interpolation; no I/O |
| `doc_assistant.tracking` | Token usage tracking and cost estimation | Append-only; never raises |
| `apps/cli.py` | Terminal renderer | Renders `TurnResult`; no business logic |
| `apps/api/` | Desktop HTTP renderer (PR-M2) | FastAPI over `127.0.0.1`; `TurnEvent` → SSE, requests → controller calls; no business logic |
| `apps/desktop/` | Tauri desktop frontend (PR-M3) | Svelte 5 + Vite UI in a Tauri 2 shell; renders the API's `TurnResult`; no business logic |
| `scripts/` | One-off maintenance scripts | Not part of the importable package |

This table is non-exhaustive — it covers the core ingest/runtime modules. The research-integrity layer (`query_router`, `synthesis`, `provenance`, `reviewer`, `metadata_extractor`, `metadata_enrich` (deterministic backfill runner, L5), `doc_vectors`, `embeddings`, `bibtex`, `commands`, `llm`) is shown in the Mermaid diagram above. The document-feature extractors (`citations`, `tables`, `tables_marker`, `figures`, `regions`) live inside the `doc_assistant.ingest` package. The cross-document knowledge layer lives inside `doc_assistant.knowledge` (ADR-023, imported as `doc_assistant.knowledge.<name>`): `concept_skeleton` is a deterministic, zero-LLM enrichment sidecar over the curated `Concept`/`ConceptAlias` vocabulary + `Citation`/`DocSimilarity` (producers `scripts/seed_concepts.py` + `scripts/build_concept_skeleton.py`), `concept_skeleton_enrich` is the confined Node-B LLM pass, and `epistemics`/`wiki`/`gaps` read the skeleton directly. The superseded open-vocabulary `concept_graph.py` was **deleted 2026-07-07** (G1, KI-7 resolved).

**Boundary rule:** `apps/` contains no business logic. All logic lives in `src/doc_assistant/`. The UI layer calls the library layer; never the reverse.

## Repository layout

```
src/doc_assistant/    # core library — the RAG answer path lives at the top level
  db/                 #   SQLAlchemy models + additive migrations
  ingest/             #   extract → markdown → chunk → embed → store (+ tables/figures/citations)
  knowledge/          #   corpus-derived layer: keywords, concept skeleton, wiki, gaps, epistemics
  eval/               #   the eval harness (runner, scorers, result store)
apps/                 # UIs — thin shells, no business logic (FastAPI/SSE · Tauri/Svelte · CLI)
scripts/              # idempotent enrichment/eval runners + build tooling
tests/                # unit, integration, eval harness cases + committed baselines
evals/                # benchmark results — the write-ups + how to reproduce each number
docs/                 # architecture, ADRs (docs/decisions/), specs, roadmap, the demo GIF
data/                 # runtime data (sources, caches, vector stores, SQLite) — not committed
```

### `apps/` — the domain spine

Both shells are organised on **one axis: the domain**, and the domain words are the same on
both sides of the wire. To review a feature end to end, read one row. On the desktop the domain
axis holds the logic: each domain's rune modules and tested `.ts` are `lib/<domain>/`, and its
wire client and types are `lib/core/api|types/<domain>.ts`. The components are on the other axis,
the screens (next section, ADR-064), and the last column says where a domain is shown.

| Domain | Wire model | Route | Shown in (`apps/desktop/src/`) |
|---|---|---|---|
| chat | `models/chat.py` | `routers/chat.py` | `screens/chat/` (Turn, SourcePanel, ClaimReview…) |
| compare | `models/compare.py` | `routers/chat.py` (`/api/compare`) | `screens/chat/CompareCard.svelte` |
| conversations | `models/conversations.py` | `routers/conversations.py` | `screens/chat/ChatHistory.svelte` |
| library | `models/library.py` | `routers/library/documents.py` | `screens/library/`, `screens/document-page/` |
| summary | `models/summary.py` | `routers/library/summary.py` | `screens/library/Summary.svelte` |
| connections | `models/connections.py` | `routers/library/documents.py` | `screens/document-page/Connections.svelte` |
| folders | `models/folders.py` | `routers/library/folders.py` | `screens/library/ManageFolders.svelte` |
| keywords | `models/keywords.py` | `routers/library/keywords.py` | `screens/manage-keywords/ManageKeywords.svelte` |
| spacing | `models/spacing.py` | `routers/library/spacing.py` | `screens/document-page/LetterSpacing.svelte`, `screens/settings/SpacingIgnoreList.svelte` |
| concepts / graph | `models/concepts.py` | `routers/concepts.py` | `screens/graph/` (Graph, ConceptIndex, GapList) |
| taxonomy | `models/taxonomy.py` | `routers/taxonomy.py` | `screens/taxonomy/Taxonomy.svelte` |
| definitions | `models/definitions.py` | `routers/definitions.py` | `screens/graph/Definition.svelte` (in the concept panel) |
| settings | `models/settings.py` | `routers/settings.py` | `screens/settings/Settings.svelte` |
| setup (first run) | `models/setup.py` | `routers/setup.py` | `screens/settings/GettingStarted.svelte` · `ProviderSetup.svelte` |
| sources (ingestion) | `models/sources.py` | `routers/sources.py` | `screens/settings/ManageFiles.svelte` (*Manage files*) |
| health | — | `routers/health.py` | `shell/StatusBar.svelte` |

Two frontend folders have no API counterpart, by design:

- `src/shell/` — the frame around every screen (top bar, sidebar, status bar), the windows it opens
  (search, keyboard shortcuts, about, indexing), and the two components every screen uses,
  `Icon.svelte` and `ConfirmDialog.svelte`. Its state and tested logic are `lib/shell/`.
- `lib/core/` — the wire boundary itself, split by the same domain names: `api/<domain>.ts`
  (thin fetch clients, shared base + error unwrapper in `_base.ts`) and `types/<domain>.ts`
  (mirrors `apps/api/models/<domain>.py`), plus `theme.ts` and `fonts.css`. Both carry an
  `index.ts` barrel, so `from '../core/api'` still resolves; prefer the domain module.

One naming trap this table exists to prevent:

- **`library` is three sub-domains.** Documents, folders and keyword families all live under
  `/api/library/*` and are split one module per sub-domain on both sides.

### `apps/desktop/src/` — the screens

A person who sees a screen or a dialog finds its code by its name (ADR-064). A **screen** is a
place you go: a tab, the document page, or a view that covers what is under it and has parts of
its own. Each has one folder under `screens/`, named as the app names it (GLOSSARY U-004), with one
entry file named for the folder, the screen's parts, and the dialogs it opens. A **dialog** asks
one thing and closes, is named for its heading, and lives with the screen whose thing it acts on,
whichever screen has the door. `App.svelte` is the shell: navigation, the readiness gate, mode
switching, global keys, and every overlay's mount.

Imports point one way, and `src/layout.test.ts` holds it: `screens/` and `shell/` hold only
components; `lib/` holds none and imports none; `shell/` and `lib/` never import `screens/`; a
screen takes only `Icon` and `ConfirmDialog` from `shell/`.

| What the app shows | Its doors | File (`apps/desktop/src/`) |
|---|---|---|
| **Frame** | | `shell/` |
| Top bar · Sidebar (collapse, resize, the tab's rail) · Status bar, its indexing chip | always | `Topbar.svelte` · `Sidebar.svelte` · `StatusBar.svelte`, `IngestChip.svelte` |
| *Search chats and documents* | top bar, Ctrl/⌘ K | `GlobalSearch.svelte` |
| *Keyboard shortcuts* · *About Provenote* | app menu | `KeyboardShortcuts.svelte` · `About.svelte` |
| Indexing (a run's progress) | the status bar's chip, *View indexing* in Settings | `Indexing.svelte` |
| The library switch failed | start-up | `LibrarySwitchFailed.svelte` |
| A confirm (delete chats, stop indexing, switch library) · icons | several · every screen | `ConfirmDialog.svelte` · `Icon.svelte` |
| **Chat tab** | top bar | `screens/chat/` |
| the tab | *Chat* | `Chat.svelte` |
| Chat history (the sidebar rail: filter, select mode, sort, a row's ⋯ menu) | sidebar, Chat | `ChatHistory.svelte` (App hands it to the sidebar frame) |
| an answer and its parts | | `Turn`, `ReadonlyTurn`, `Markdown`, `Provenance`, `ClaimReview`, `SourceEvaluation` |
| the comparison | *Test override* | `CompareCard.svelte` |
| a citation's source | an answer's [n] | `SourcePanel`, `SourceCard`, `ChunkContextView` |
| **Library tab** | top bar | `screens/library/` |
| the tab, the empty Library · the header bar and the select bar · the grid or list · the keyword filter strip · the library summary | *Library* · *Select* · · · the layout toggle's *Summary* | `Library.svelte` · `HeaderBar.svelte` · `DocumentGrid.svelte` · `FilterStrip.svelte` · `Summary.svelte` |
| Library navigation (the sidebar rail) | sidebar, Library | inside `shell/Sidebar.svelte`: its styles are the frame's tree rows and filter (ADR-064 choice 6) |
| *Filter by keyword* | the filter strip | `FilterByKeyword.svelte` |
| *Manage folders* | the rail's *Manage…*, a tile's *Add to folder*, the select bar's *Add to folder…* | `ManageFolders.svelte` |
| *Edit metadata* · *Delete this document?* | a tile's ⋯ menu | `EditMetadata.svelte` · `DeleteDocument.svelte` |
| *Add documents*, first step · its review sheet | app menu, the empty Library, a drop · the staged-files bar's *Review* | `AddDocumentsChooser.svelte` · `AddDocumentsReview.svelte` |
| **Document page** | a document in the Library, Search, a graph concept | `screens/document-page/` |
| the page · its index (*Metadata* … *References*, *Source*, *Re-run…*) · its header · the Chunks block | | `DocumentPage.svelte` · `PageIndex.svelte` · `DocumentHeader.svelte` · `Chunks.svelte` |
| Connections · Figures · References | the page's index | `Connections.svelte` · `Figures.svelte` · `References.svelte` |
| the figure viewer · the source viewer | a figure · *Source* | `FigureViewer.svelte` · `SourceViewer.svelte` |
| *Letter-spacing on this document* · *Edit spacing* | the page · a run's *Edit spacing* | `LetterSpacing.svelte` · `EditSpacing.svelte` |
| *Re-run ingestion* | *Re-run…*, *Re-run text now…* (page and Settings), the Library's select bar | `RerunIngestion.svelte` |
| **Graph tab** | top bar | `screens/graph/` |
| the tab, its empty states, the graph bar · the concept panel | *Graph* · a concept | `Graph.svelte` · `ConceptPanel.svelte` |
| *Definition* · the concept index (the sidebar rail) · the gap list | the concept panel · sidebar, Graph · the rail's *Gaps* | `Definition.svelte` · `ConceptIndex.svelte` · `GapList.svelte` |
| ***Manage keywords*** | *Manage keywords…* in *Filter by keyword*; on the Graph tab, *Manage keywords* (the graph bar and the empty state) and a concept's *Edit* | `screens/manage-keywords/ManageKeywords.svelte` |
| *Detect proposals* · *New family* · *Concepts* · *Terms* · one family's row | | `DetectProposals.svelte` · `NewFamily.svelte` · `Concepts.svelte` · `Terms.svelte` · `FamilyRow.svelte` |
| ***Taxonomy*** | the concept index's footer, *Place* on a concept | `screens/taxonomy/Taxonomy.svelte` |
| the field forest · a field's detail and *Proposed placements* | | `FieldForest.svelte` · `FieldDetail.svelte` |
| ***Settings*** | top bar, app menu, *Finish setup…* on the Chat tab, the letter-spacing block's link | `screens/settings/` |
| the window and its rail · its sections *Getting started*, *Documents*, *Provider & model*, *Retrieval*, *General* | the rail | `Settings.svelte` · `GettingStarted.svelte`, `Documents.svelte`, `ProviderAndModel.svelte`, `Retrieval.svelte`, `General.svelte` |
| the providers: *Answering with* · the list · the open provider's form | *Provider & model* | `ProviderSetup.svelte` · `AnsweringWith.svelte` · `ProviderList.svelte` · `ProviderForm.svelte` |
| *Manage files* · *Letter-spacing ignore list* | *Documents* | `ManageFiles.svelte` · `SpacingIgnoreList.svelte` |
| *Re-index this folder?* · *Open this library?* | *Re-index* · *Open another library…*, in *Documents* | `ReindexFolder.svelte` · `OpenLibrary.svelte` |

## Two-tier caching

1. **Extraction cache** (`data/cache/*.md`): mirrors `data/sources/` structure.
   Invalidated by file modification time. Skips re-extraction on unchanged files.
2. **Embedding cache** (Chroma `doc_hash` metadata): invalidated by content hash.
   Skips re-embedding when content is unchanged.

Both tiers are independent: changing the chunking strategy invalidates embeddings but not extraction. Rebuild with `python -m doc_assistant.ingest --rebuild`.

Hashing is content-only (SHA-256 of extracted markdown, truncated to 16 hex chars). Documents survive path changes and re-extractions without creating orphan rows. Migration from the old path+content scheme: `scripts/archive/migrate_to_content_hash.py`.

## Chunking & retrieval units

The retriever's unit of work is not "the document" — it's a **chunk**. How documents
are split into chunks, and what gets embedded vs. what reaches the LLM, is the
parent–child scheme below. This is the default mode (`USE_PARENT_CHILD=true`); a
flat single-store mode (`baseline`) exists as a fallback.

**Two grain sizes, one link.** A document is split twice:

- **Parents** — 2000 chars, 200 overlap, split on `## `/`### `/paragraph
  boundaries (`PARENT_CHUNK_SIZE`/`_OVERLAP`). A parent is a coherent passage —
  the unit of *context* sent to the LLM.
- **Children** — 400 chars, 50 overlap (`CHILD_CHUNK_SIZE`/`_OVERLAP`). A child is
  a narrow span — the unit of *retrieval* that gets embedded and matched.

The link between them is **not a relational foreign key**. There is no `chunks`
table in SQLite; chunks live only as Chroma vectors, and the parent–child
relationship is carried in each child's Chroma **metadata**: `parent_text`,
`parent_index`, `child_index`. A child retrieves; the pipeline then unpacks that
child's `parent_text` and sends the *parent* to the model. Small unit for
precision, large unit for context.

```mermaid
flowchart TB
    D["Source document (.md)<br/>cached, page/section markers"]
    D -->|"split · 2000/200"| P0["parent_index 0<br/>'## Method  We use a hybrid…'"]
    D --> P1["parent_index 1<br/>'The ensemble fuses results…'"]
    D --> PN["parent_index N<br/>table / figure chunk (atomic)"]
    P0 -->|"split · 400/50"| C0["child_index 0"]
    P0 --> C1["child_index 1<br/>'The dense retriever embeds…'"]
    C1 -->|"only children are embedded"| CH[("Chroma vector<br/>+ metadata")]
    CH -->|"child wins → unpack parent_text"| LLM["LLM context = parent_text"]
```

**What a stored record looks like** (Chroma, not SQL — metadata *is* the schema):

```json
{
  "page_content": "The dense retriever embeds queries and documents…",
  "metadata": {
    "document_id": "550e8400-e29b-41d4-a716-446655440000",
    "doc_hash": "abc123", "filename": "dpr.pdf", "format": "pdf", "health": "healthy",
    "parent_text": "## Method  We use a hybrid retriever…",   // full parent → LLM context
    "parent_index": 0, "child_index": 1,
    "page": 3, "section": "Method",
    "chunk_type": null, "figure_id": null                     // "figure" for VLM-described figures
  }
}
```

**Tables & figures are atomic chunks.** A table is spliced into the cached markdown
and merged with its caption into a single parent==child block (the caption is the
retrieval "magnet"). A figure becomes a `chunk_type="figure"` chunk only *after* the
VLM description pass — `(caption + vlm_description)`; the PNG image itself is never
embedded. See `figures-and-tables.md`.

**Document structure is scaffolded but unused.** The `DocumentPart` table
(`db/models.py`) can hold a chapter/section tree (`kind`, `title`,
`parent_part_id`), but it is not currently populated and chunks do not link to it.
This is the seam the book-oriented redesign builds on (ADR-009): the current scheme
is tuned for short, section-headed papers and degrades on long, chaptered books.

All chunk sizes and retrieval weights are **locked settings** — changed only via an
eval-harness experiment, never edited ad hoc. See `.claude/CONTEXT.md`.

## Concept & knowledge system

The Phase-7 knowledge layer has grown over many increments (ADR-006/008/015/017/018/019/023/028).
This is the canonical map of how its pieces relate; the module list is in *Module responsibilities*
above, and each design choice is in the ADR named beside it.

**One table, four hats.** Everything hangs off a single `Concept` table (`db/models.py::Concept`) —
one id-space by deliberate choice (PR-G1 fixed the KI-15 id/label confusion by making it so). The same
row is read four ways:

1. **Keyword candidate → Concept.** `Keyword` rows are per-document mined terms (`knowledge/keywords.py`,
   ADR-006 contrastive termhood) — *candidates only*, never auto-promoted. The user promotes one into a
   curated `Concept` (`promote_keyword`). The 2026-07-05 `--promote-all` flood (ADR-018) is why that
   boundary is load-bearing.
2. **keyword family** — a `Concept` whose `ConceptAlias` rows hold member keyword names, used to collapse
   near-duplicates (`llm`/`llms`) in the Library filter (ADR-015). Families **ignore `graph_include`** by
   design; the graph respects it. Same rows, two consumers — the boundary ADR-015 named and ADR-018 paid for.
3. **concept-skeleton node** — a `Concept` with `graph_include=true` becomes a node in the derived graph.
4. **taxonomy node** *(decided, unbuilt)* — under ADR-028 a `Concept` also carries a `kind`
   (`concept` | `domain`), letting abstract field nodes share the id-space.

> **Orientation:** [`knowledge-layer.md`](knowledge-layer.md) is the one-page map of what this whole
> layer is *for* (corroboration · coverage · navigation), how the pieces connect, and — since
> 2026-08-03 — a **trust table** marking which of its signals are sound. `contested` /
> `superseded_trend` are **not** corpus measurements (KI-33, ADR-040/ADR-041); read that table before
> citing any marker. This section stays the mechanism reference.

**Two graph layers over the same nodes.** These are distinct and must not be conflated:

| Layer | What | Store | Lifecycle | ADR |
|---|---|---|---|---|
| **Derived** (association) | co-occurrence edges + Louvain communities (Node A), optionally LLM relation/stance (Node B) | `concept_edges` + `data/skeleton/skeleton.json` | **dropped & rebuilt** every `build_concept_skeleton` run (~7 s, deterministic) | ADR-008 |
| **Curated** (classification) | the user's `is_a`/`in_field` hierarchy — a polyhierarchical SKOS DAG | `concept_hierarchy` table *(spec'd, unbuilt)* | **survives a rebuild** — lives beside `Concept`, never in `concept_edges` | ADR-019→ADR-028 |

The load-bearing rule: **curated structure must never live in `concept_edges`**, because a routine rebuild
would wipe it (the KI-17/KI-20 class of bug). Node A is zero-LLM; **Node B is a confined LLM pass that
only annotates existing edges** — it never creates a node or edge, and `build_concept_skeleton --apply`
*without* `--enrich` silently wipes its annotations (rebuild with `--apply --enrich` together).

**What reads the graph.** `epistemics.py` projects skeleton node weights onto chunks (`chunk_epistemics`)
→ the answer-path markers + E2 source strip; `gaps.py` runs deterministic detectors over the skeleton
(`isolated`/`single_source`/`thin_bridge`/`under_connected`/`unsourced_claim`) → the gap list, with a
`gap_triage` override table that survives rebuilds; `wiki.py` clusters over communities. All are
**read-only over the vocabulary** — the graph UI never edits concepts, it deep-links to Manage-keywords
(ADR-017 A1). The single write surface for the curated hierarchy will be a dedicated taxonomy view (ADR-028).
**One exception, by the user's choice (ADR-053, 2026-09-21):** a concept's *definition* is read and chosen
in the graph's concept panel — `knowledge/definitions.py` keeps every candidate (a verbatim passage with
its page, the user's own words, later a model's text) and choosing is the only write to
`Concept.definition`, recorded and undoable. Beside it, read-only, *how your library uses it*
(`GET /api/concepts/{id}/usage`, 2026-09-22): a plain sentence from each of the few documents that
use the word most, found through the keyword index and never stored. And what the text says about the
name itself (`GET /api/concepts/{id}/signals`, 2026-10-17; `knowledge/term_signals.py`): a short or long
form given beside one of its forms, a name mostly cited as an author, and where a label no sentence uses
occurs — offered, never applied; a form is added in Manage keywords.

**Current build state (2026-09-10).** Node A skeleton, keyword families, gap layer, epistemics projection,
and the read-only graph/gap UI are **built and shipped**. The taxonomy layer is **built** (TX1–TX3, 2026-07):
`kind` column, curated `concept_hierarchy` + `document_field` tables, `knowledge/taxonomy.py` (write seam,
acyclic), `taxonomy_view.py` (read model), `taxonomy_propose.py` (quarantined LLM proposals, ADR-028 D8),
`apps/api/routers/taxonomy.py`, `screens/taxonomy/Taxonomy.svelte`, `scripts/seed_taxonomy.py` (ANZSRC 2020 FoR) and
`scripts/propose_taxonomy.py` — but the taxonomy is **empty of concept placements** (13 of 357 placed, 0 `is_a`;
ADR-045). Node-B stance is **withheld from the UI** (KI-33, ADR-040/041); regeneration is a local-LLM cost
decision (KI-4). The superseded open-vocabulary `concept_graph.py` was deleted 2026-07-07
(KI-7); `data/graph/graph.json` is a stale empty decoy from that era — the live artifact is
`data/skeleton/skeleton.json`.

## Document health model

Each ingested document is scored on five signals: chunk count, chunks-per-page ratio, average chunk length, section detection rate, reference-flagged chunk ratio.

- Score ≥ 75 → **healthy**
- Score ≥ 40 → **marginal** (retrievable, flagged)
- Score < 40 → **broken** (retrievable, prominently flagged)

Classification is informational, never blocking. Broken documents remain queryable.

## Engineering standards

### Security
- No secrets in code. `.env` is gitignored. `.env.example` committed with placeholders.
- `bandit` SAST runs in CI and pre-commit. HIGH findings block merge.
- `pip-audit` runs in CI on every push and blocks on any advisory not reviewed in
  `pip-audit-ignore.toml` (`scripts/pip_audit_gate.py`, since 2026-09-30).
- `detect-secrets` baseline committed; the hook runs in pre-commit on the staged files, and CI
  scans every tracked file with `scripts/secret_scan_gate.py`, which fails on a secret the
  baseline does not record (since 2026-10-01).

### CI/CD
- GitHub Actions on every push and PR: ruff lint + format-check → mypy → pytest with coverage (fail-under 40) → bandit → pip-audit (blocking, reviewed ignores) → secret scan (blocking, recorded findings in the baseline).
- Merging on red pipeline is never allowed.
- Coverage floor: 40% (CI-enforced; `--cov-fail-under=40` in ci.yml). Raise toward 45%+ as integration tests land. Target: 85% for core pipeline and ingest logic.

### Pre-commit (mandatory)
Hooks: ruff (lint + format), mypy, bandit, detect-secrets, standard file hygiene.

### Logging
Structured JSON logging in staging/production via `structlog`. Development uses pretty console output. No `print()` in `src/`. Log entries include: level, timestamp, module, event, and operation-specific context fields. Secrets and PII are never logged.

### Development log
Maintain `docs/DEVLOG.md` — append one entry per logical change (what / why / rejected / opens). See dev-log skill for format. Append only, never edit past entries.

### Error handling
Exception hierarchy rooted at `DocAssistantError`. Domain errors (ExtractionError, IngestError, PipelineError) are typed and documented. Infrastructure errors (StorageError, ExternalServiceError) propagate with context via `raise X from e`. User-facing errors are translated at the UI boundary; internal traces go to logs only.

### Testing
```
tests/
├── fixtures/             # shared fixtures (synthetic_corpus.py)
├── unit/                 # fast, no I/O, no LLM
│   └── test_<module>.py
├── integration/          # cross-module, may use temp files, mocked LLM
│   └── test_<flow>.py
└── eval/                 # RAG evaluation harness (not part of standard CI run)
    ├── cases.yaml / cases.public.yaml   # consumed by scripts/run_eval.py
    ├── TESTING.md        # what each tier and scorer measures
    └── baselines/        # recorded eval baselines
```

Unit tests run on every commit (pre-commit). Full suite (unit + integration) runs in CI — free, no API calls. Eval harness runs manually at phase checkpoints and costs money (Anthropic API for the LLM judge).

The testing strategy — what each tier and each eval scorer measures, why, and the reproducible public-corpus benchmark — is documented in [`tests/eval/TESTING.md`](../tests/eval/TESTING.md). The benchmark *results* write-ups (headline numbers, sweeps, reproduction) live in the top-level [`evals/`](../evals/README.md) folder (ADR-024); the underlying run data stays in `tests/eval/baselines/`.

Run commands:
- `uv run pytest tests/unit/ tests/integration/` — free, fast, CI default
- `uv run python -m scripts.run_eval` — manual, costs API tokens (the canonical harness; reads `tests/eval/cases.yaml`, persists to `data/eval.duckdb`)
- `uv run pytest -m api` — any future tests marked with `@pytest.mark.api`
