<!-- status: archived · updated: 2026-07-13 · class: disposable -->
<!-- BUILT + COMMITTED 2026-07-13 (`aa288d9` "UI: In-App Ingestion (v1)"). Flipped active→archived
     when SPRINT-015 (A/B-compare) became the active contract. Design lock:
     docs/specs/feature-library-browser.md (grilled 2026-07-13). Feature: the Library space L1 —
     a read-only chunk browser. Lights up the reserved (disabled) Library sidebar tab: list ingested
     documents -> open one -> read its chunks as parent blocks, each expandable to its child chunks.
     Read-only, no model, no writes: reads SQLite Document rows + the live Chroma handle. Markers +
     figure thumbnails deferred to L1b (chunk_epistemics=0 / figures=0 on this corpus). Flip to
     status:archived after commit. Pre-existing [lifecycle] warn applies (tolerated across SPRINT-*.md). -->

# SPRINT-014 — library-browser

- **base:** main
- **DoD:** The Library sidebar tab is enabled. `GET /api/library/documents` lists ingested documents
  (from SQLite `Document`, ordered by filename; NULL metadata omitted). `GET /api/library/documents/{id}`
  returns the document's chunks grouped into parent blocks (by `parent_index`), each with its
  `child_index`-ordered child chunks; unknown id → 404; a 0-chunk document → empty parents (not a 404).
  The read layer writes nothing (SQLite + Chroma untouched — guard-tested). Desktop: a client `mode`
  switches Chat/Library; in Library mode the sidebar lists documents and the main pane renders the
  selected document (header + accordion of parent blocks expandable to children); read-only; both themes;
  no horizontal overflow. `svelte-check` 0; full gate green (ruff / ruff format / mypy src / bandit /
  pytest); preview-harness-verified live on the real corpus ($0/offline — no model).

<!-- One path (or glob) per bullet. uses/affects/contracts/docs are machine-read by sprint_check.py. -->

## uses
- docs/specs/feature-library-browser.md
- src/doc_assistant/conversations.py
- src/doc_assistant/db/models.py
- src/doc_assistant/db/session.py
- src/doc_assistant/pipeline.py
- apps/api/main.py
- apps/api/models.py
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Sidebar.svelte
- apps/desktop/src/lib/api.ts
- apps/desktop/src/lib/types.ts

## affects
- src/doc_assistant/library.py
- apps/api/main.py
- apps/api/models.py
- apps/desktop/src/App.svelte
- apps/desktop/src/lib/Sidebar.svelte
- apps/desktop/src/lib/LibraryBrowser.svelte
- apps/desktop/src/lib/api.ts
- apps/desktop/src/lib/types.ts
- tests/unit/test_library.py
- tests/integration/test_api_library.py

## contracts
- test: tests/unit/test_library.py::test_group_children_orders_and_groups
- test: tests/integration/test_api_library.py::test_library_endpoints
- map: apps/desktop/src/lib/types.ts | when: apps/api/models.py

## docs
- docs/DEVLOG.md
- docs/ROADMAP.md
- docs/ui-checklist.md
