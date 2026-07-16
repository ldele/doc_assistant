# Spec — Library space, L1: read-only chunk browser

**Status:** ✅ **SHIPPED 2026-07-13 (L1, commit `aa288d9`; SPRINT-014 archived).** Was DESIGN-LOCKED
(grilled 2026-07-13, `grill-me`); retained as the design record. Roadmap PR **L1**. Owner: Claude Code.
**One PR.** Ledger + routing at the foot of this file. Supersedes the `docs/ui-checklist.md` §3 backlog
row "In-app ingestion + Calibre-style chunk browser" for its **browser** half only — the ingestion half
(L2) routes to `docs/specs/feature-selective-ingestion.md` (S1/S2); chunk annotation (L3) is parked.

**Pattern reference:** the read-layer shape is a copy of the just-shipped `conversations.py`
(`docs/specs/feature-conversation-history.md`) — a pure-ish read module over an existing store, thin
pydantic payloads, a GET endpoint pair, no writes. The shell is the SPRINT-013 `sidebar│main│drawer`.

**Requirement.** The desktop app can answer *from* the corpus but can't *show* it: there is no way to
see which documents are ingested, or to read the chunks the retriever actually holds. The reserved (and
today `disabled`) **Library** sidebar tab lights up as a **read-only browser** — list ingested documents
→ open one → read its chunks the way the two-tier retriever stores them (parent blocks, each expandable
to its embedded child chunks). Read-only, **no model, no writes** — reads only the existing SQLite
`Document` rows + the live Chroma handle, so the whole feature is verifiable **$0/offline** on the real
corpus (76 docs / 30,882 child chunks on this box).

---

## Grounding (read from the live corpus 2026-07-13, not assumed)

- **Documents** live in SQLite (`db/models.py::Document`): reliably populated are `id`, `filename`,
  `format`, `chunk_count`, `extraction_health`, `added_at`; `title`/`authors`/`year` are **often NULL**
  on this corpus (show only when present — honest, no empty labels).
- **Chunk text** lives in Chroma. Live store = `PC_CHROMA_PATH` (`chroma_pc`, collection `langchain`)
  because `USE_PARENT_CHILD` is **default-on** (`.claude/CONTEXT.md`). Each **child** chunk's metadata
  carries: `document_id` (== `Document.id`), `doc_hash`, `filename`, `format`, `health`,
  `parent_index`, `parent_text` (present on every chunk), `child_index`, `source_original`,
  `source_cache`. **NULL on this corpus:** `chunk_index`, `page`, `section`, `keep_for_retrieval`.
- The **doc→chunks join is `where={"document_id": <Document.id>}`** — cleaner than `doc_hash`, no
  ambiguity. Group children by `parent_index`; order parents by `parent_index`, children by `child_index`.
- `keep_for_retrieval` is `None` (kept) on this corpus; when it is `False` (figure/table artifacts on
  other corpora) the browser still shows the chunk, flagged "not retrievable" — it is real ingested
  content, and this is a "what's in your corpus" view.

## Decisions

| # | Decision | Deciding reason |
|---|---|---|
| 1 | **Read-only, derived, no persistence.** The browser writes nothing; it reads `Document` (SQLite) + the live Chroma handle. No new table, no sidecar (annotation = L3, deferred). | Enrichment-Layer discipline; the data already exists |
| 2 | **v1 = text + metadata only.** Chunk/parent text + doc-level metadata (present fields only). **Marker chips + figure thumbnails are deferred to L1b** — `chunk_epistemics` = 0 rows and `figures` = 0 rows on this box, so they'd render empty and can't be proven $0/offline; figures additionally need the paid VLM pass. | Grounded in the corpus reality; ships provable code |
| 3 | **Granularity = parent blocks, inline-expandable to children.** The detail view lists `parent_text` blocks (the unit the LLM reads + a citation shows); each expands in place to its `child_index`-ordered child chunks (the embedded retrieval units). | The full two-tier mental model; matches how retrieval dedups by parent |
| 4 | **Navigation: Library mode swaps both panes.** The sidebar Chat/Library tabs switch a client `mode`; in `library` mode the sidebar lists **documents** (same row idiom it uses for conversations) and the main pane renders the selected document's chunks. | Maximal reuse of the SPRINT-013 shell; mirrors Chat mode |
| 5 | **Reuse the live Chroma handle** (`ChatController.rag.db`), never a second client/embedder. The read layer takes the handle as a parameter (impure boundary); the pure grouping is unit-tested with a fake `.get()` return. | Handle is already open at construction; no double-load, no embeddings needed for `.get()` |
| 6 | **Honest empty-states.** A document with 0 Chroma chunks (drift, or nothing embedded) → "no chunks stored"; a NULL metadata field is omitted, never shown blank; a `keep_for_retrieval=False` chunk is shown with a "not retrievable" flag. | inform-don't-block; no fake richness |
| 7 | **Ordering.** Documents by `filename` asc (stable; `title` is often NULL). Parents by `parent_index`; children by `child_index`. | Deterministic, filename is always present |

## Contract — `src/doc_assistant/library.py` (new)

Pure-ish read layer, `conversations.py` shape. Frozen dataclasses + functions; `session_scope()` used
internally for the SQLite read, the Chroma handle passed in for the chunk read.

```
@dataclass(frozen=True) LibraryDocument:
    doc_id: str; filename: str; format: str; chunk_count: int | None
    extraction_health: str | None; added_at: datetime
    title: str | None; authors: str | None; year: int | None   # present-only; may be None

@dataclass(frozen=True) LibraryChildChunk:
    child_index: int; text: str; retrievable: bool               # keep_for_retrieval is not False

@dataclass(frozen=True) LibraryParentBlock:
    parent_index: int; parent_text: str; children: list[LibraryChildChunk]

@dataclass(frozen=True) LibraryDocumentDetail:
    document: LibraryDocument
    parents: list[LibraryParentBlock]
    child_count: int
```

- `list_documents() -> list[LibraryDocument]` — **impure** (own `session_scope`); `Document` rows
  ordered by `filename`. SQLite only, no Chroma. Excludes nothing (archived handling = out of scope).
- `group_children(rows: list[dict]) -> list[LibraryParentBlock]` — **pure**; the core. Input = the
  `[{metadata..., "document": text}, …]` shape unzipped from a Chroma `.get(...)`; groups by
  `parent_index`, orders parents/children, carries `parent_text` from the first child of each parent,
  drops a child whose `parent_index`/`child_index` is missing (logged count). Exhaustively unit-tested.
- `get_document_detail(doc_id: str, chroma) -> LibraryDocumentDetail | None` — **impure** orchestration:
  `Document` by id (SQLite; `None` → return `None`) → `chroma.get(where={"document_id": doc_id},
  include=["documents", "metadatas"])` → `group_children` → assemble. `chroma` is the live
  `Chroma` handle (typed by the `.get` protocol, not imported heavy).

**NOT responsible for:** any write; markers/figures (L1b); Chroma construction (handle is injected);
ingestion (L2); annotation (L3).

## Contract — `apps/api/models.py` (additive)

`LibraryDocumentPayload` (+ `from_doc`), `LibraryChildPayload`, `LibraryParentPayload`,
`LibraryDocumentDetailPayload` (+ `from_detail`) — mirror the dataclasses; `added_at` UTC-tagged the same
way `ConversationSummaryPayload` tags naive DB datetimes (`_as_utc`).

## Contract — `apps/api/main.py` (additive)

- `GET /api/library/documents` → `list[LibraryDocumentPayload]` (`library.list_documents()`).
- `GET /api/library/documents/{doc_id}` → `LibraryDocumentDetailPayload`; passes the live handle
  `request.app.state.controller.rag.db`; **404** on unknown `doc_id`. Lazy-import `library` inside the
  route (the `conversations` route precedent).

## Contract — `apps/desktop/src` (thin renderer, zero business logic)

- `App.svelte` — a `mode: 'chat' | 'library'` `$state`; on entering `library`, fetch
  `/api/library/documents` once (cache); hold `libraryDocId: string | null`; render `LibraryBrowser`
  in the main pane when `mode==='library'`. Chat state (`turns`, `viewing`, `sessionId`) is untouched by
  the mode switch (same isolation as history's `viewing`).
- `Sidebar.svelte` — enable the Library tab; gains `mode` + `documents` + `onSelectMode` +
  `onSelectDocument` + `selectedDocId` props; in `library` mode the list renders documents
  (filename + `chunk_count` chunks + relative `added_at`) using the existing `.row` idiom; the Chat
  history list is preserved (re-shown when mode flips back). The `↻ New chat` button hides in library mode.
- `lib/LibraryBrowser.svelte` (new) — given `docId`, fetch `/api/library/documents/{id}`; render a doc
  header (filename, format, chunk_count, health, + title/authors/year when present) then the parent
  blocks as an accordion (`parent_text` visible; a disclosure expands the `child_index`-ordered children).
  Read-only; both themes; no horizontal overflow; wide text wraps/scrolls in its own container.
- `lib/api.ts` — `getLibraryDocuments()`, `getLibraryDocument(id)`.
- `lib/types.ts` — `LibraryDocument`, `LibraryChildChunk`, `LibraryParentBlock`, `LibraryDocumentDetail`
  mirroring `apps/api/models.py`.

## Tests

**Unit (`tests/unit/test_library.py`, new):** `group_children` — multi-parent grouping; parent/child
ordering by `parent_index`/`child_index`; `parent_text` taken from each parent's first child; a row
missing `parent_index`/`child_index` dropped (logged); empty input → `[]`; a `keep_for_retrieval=False`
row → `retrievable=False`.

**Integration (`tests/integration/test_api_library.py`, new; tmp SQLite + a fake/tiny Chroma or a
monkeypatched handle — no corpus, no network, no model):** `GET /api/library/documents` lists seeded
docs ordered by filename; `GET /api/library/documents/{id}` returns grouped parents/children; unknown id
→ 404; a doc with 0 chunks → detail with `parents=[]`, `child_count=0` (not a 404). **Guard:** the read
path issues no write (SQLite row counts + a Chroma write-trap monkeypatch unchanged).

Full gate green (`ruff` / `ruff format` / `mypy --strict src` / `bandit` 0 HIGH·MED / `pytest`);
`svelte-check` 0 errors.

## Definition of done

- Library tab enabled; browsing the **real corpus** ($0/offline, no model): the doc list shows 76 docs;
  opening one shows its parent blocks; a parent expands to its children; NULL metadata omitted; a large
  doc (e.g. `hebb_1949.pdf`, 1280 chunks) renders without body overflow.
- Both themes; mobile off-canvas sidebar still works in library mode.
- Preview-harness screenshot/snapshot proof (per `.claude/KNOWN_ISSUES.md`: snapshots + synchronous
  evals on this box).
- One `docs/DEVLOG.md` entry; ROADMAP L1 row + `docs/ui-checklist.md` updated.

## Out of scope (deferred, with owners)

- **Marker chips + figure thumbnails (L1b)** — the rendering + the data-population runbook
  (`compute_epistemics` for markers, the VLM pass for figures); reopens when the sidecars are populated.
- **In-app ingestion management (L2)** — `docs/specs/feature-selective-ingestion.md` (S1 backend / S2 panel).
- **Chunk annotation / comment (L3)** — a new Enrichment-Layer sidecar table; **needs its own ADR**
  (first write path in the Library space) before it's buildable.
- **Search / filter / sort within the library**, **in-app PDF source viewer**, **doc metadata editing or
  backfill** (titles/authors/year), **pagination/virtualization** of very large docs — each a later item.

## Ledger (grill-me, 2026-07-13)

F0 scope → L1 browser is v1 (L2/L3 parked) · F1 → SQLite `Document` · F2 → `library.py` + GET pair,
reuse live handle · F4a → parents expandable to children · F4b → text+metadata; markers/figures → L1b
(markers=0/figures=0 here) · F5 → sidebar=doc list, main=chunks · F6 → inline accordion · F7 →
preview-harness $0/offline. Reopens: F4b when sidecars populate.
