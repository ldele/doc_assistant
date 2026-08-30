<!-- status: active · updated: 2026-08-28 (EPUB + HTML round trip; ADR-046 delete half; AD4; CS1/CS2 — all work items built) · class: living -->

# Spec — Add documents: the accept surface (AD1–AD4) + the Settings changes it forces (CS1–CS2)

Implements Track A and Track C of `docs/PLAN_2026-08-20_user-friendly-ingestion.md` under
**ADR-046**. Track B (ingest honesty) is specced separately and needs none of this.

## Goal

Provenote can index a folder and **cannot accept a document**. The only way in today is a text box
reading *"Paste the full path to the folder holding your documents."* This spec adds the one missing
step: drag-and-drop or pick files, see per-file verdicts before anything happens, then copy them in
**or** reference them where they live, and index.

Everything downstream already exists and must be reused, not rebuilt: `POST /api/ingest` with an
explicit `paths` list, `GET /api/ingest/status`, `registry.scan_sources`, the Sources panel.

## Hard constraints

1. **ADR-046 governs placement, delete and identity.** Both copy-in and reference-in-place ship.
   `SourceFile` becomes `(root, rel_path)`. Delete asks and defaults to library-only. Add-time
   identity is `sha256(source bytes)`.
2. **Nothing is copied, registered or indexed before the review sheet is shown and confirmed.**
3. **Inform, don't block.** An unsupported or damaged file is shown and skipped, never a silent drop
   and never a blocking error.
4. **No new ingest endpoint.** Indexing goes through the existing `POST /api/ingest` `paths` form.
5. **The frontend is a deliberate 1-dep artifact** (`marked`). Any new npm dependency is a decision,
   not a detail — see W0.
6. **Robustness contract:** works at 0 documents and at 10k. No corpus-tuned constants; the batch is
   uncapped (ledger branch 7).
7. **`structlog` only, no `print()` in `src/`**; exceptions chain; `encoding="utf-8"` on every file
   read/write (CONTEXT.md §9).

## W0 — RESOLVED 2026-08-21: the accept surface costs **zero npm dependencies**

**Route chosen: `withGlobalTauri: true` + the injected `window.__TAURI__` API.** Drag-and-drop uses
the Tauri drag-drop event; the picker uses `tauri-plugin-dialog`, which injects itself into the same
global. Constraint 5 (the 1-dep frontend) is **preserved** — this is a better answer than the spike
was written to expect.

**Evidence, all read from the pinned crate sources, not from memory** (`tauri` is locked at 2.11.3;
2.11.2 is the vendored source, same minor API):

| Claim | Evidence |
|---|---|
| The drag-drop event carries **real filesystem paths** | `tauri/src/manager/window.rs` emits `DRAG_DROP_EVENT` (`tauri://drag-drop`) with a payload built as `paths: Some(paths), position` |
| The **injected** JS can listen for it with no npm package | `tauri/scripts/bundle.global.js` — what `withGlobalTauri` puts on `window.__TAURI__` — contains all four `tauri://drag-*` event names, an `onDragDropEvent` helper, and exports the `core`, `event` and `webview` namespaces |
| `withGlobalTauri` is the injection switch, default off | `tauri-utils/src/config.rs:3075` — *"Whether we should inject the Tauri API on `window.__TAURI__` or not"*, `with_global_tauri: false` by default |
| **HTML5 drag-and-drop is ruled out, not merely worse** | `tauri-utils/src/config.rs:1943` — `drag_drop_enabled` defaults **true**, and its doc says *"Disabling it is **required** to use HTML5 drag and drop on the frontend on Windows."* Disabling it would also surrender the paths, which every contract here needs |
| The picker needs **no npm package either** | `tauri-plugin-dialog-2.7.1/api-iife.js` does `Object.defineProperty(window.__TAURI__, "dialog", {value: __TAURI_PLUGIN_DIALOG__})`, so with `withGlobalTauri` on it lands at `window.__TAURI__.dialog`. The crate is already in the local cargo registry |

**Total cost of the accept surface:**
- `tauri.conf.json`: `"withGlobalTauri": true` (one line; leave `dragDropEnabled` at its default).
- `src-tauri/Cargo.toml`: `tauri-plugin-dialog = "2"` + one `.plugin(...)` registration.
- `capabilities/default.json`: add `dialog:allow-open` (narrower than the plugin's `dialog:default`,
  which also grants `message` and `save` — neither is needed).
- **npm: nothing.**

**RUNTIME-CONFIRMED 2026-08-24 in a real Tauri window.** The two assertions were run inside
`npx tauri dev` (a devtools console is not reachable from an agent session, so a temporary probe
reported the result by pinging the API, whose request log carried it back; the probe has been
removed). Result, verbatim:

```json
{"tauriKeys": ["app","core","dpi","event","image","menu","mocks","path","tray",
               "webview","webviewWindow","window"],
 "isTauri": true, "canReceiveDrops": true, "canPickFiles": true,
 "listenOk": true, "listenError": null, "href": "http://localhost:1420/"}
```

- **The API is injected.** Twelve namespaces on `window.__TAURI__`.
- **`listen('tauri://drag-drop')` registers.** `listenOk: true` — the transport works.
- **The picker is reachable.** `canPickFiles: true`.
- `href` confirms the window loaded **this** app's dev server, not the other project that
  periodically owns 1420.

⚠ **CORRECTION — the assertion below expected `dialog` in `Object.keys()`, and it is NOT there.**
`tauri-plugin-dialog` registers itself with `Object.defineProperty(window.__TAURI__, "dialog", …)`,
and `defineProperty` defaults to **`enumerable: false`** — so the plugin is present and fully
functional (`canPickFiles: true`) while being invisible to `Object.keys()`. Do not read its absence
from that list as a failed registration. Probe `typeof window.__TAURI__.dialog?.open` instead:

```js
// 1 · the API is injected (dialog is deliberately absent — see above)
console.log(Object.keys(window.__TAURI__ ?? {}))
console.log(typeof window.__TAURI__?.dialog?.open)   // expect: "function"

// 2 · the event carries paths
window.__TAURI__.event.listen('tauri://drag-drop', e => console.log(e.payload.paths))
```

**Still not proven, and only a human can:** that a real OS drag delivers a non-empty `paths` array.
The *subscription* is confirmed above and the *payload shape* is confirmed from
`tauri/src/manager/window.rs` (`paths: Some(paths), position`); what remains is one drag.

**One thing the run revealed that the code comments deny.** `lib.rs` says *"in dev there is no
frozen binary (run the backend separately with `just api`)"* — but a `binaries/doc-assistant-api`
sidecar **does** exist on this machine, so `tauri dev` spawned it and it failed to bind 8001
against an already-running API. Harmless (the spawn failure is non-fatal by design), but the
comment is out of date and the port clash is confusing if you are not expecting it.

## Work items

### AD1 — the accept surface
- Drag-and-drop onto the Library pane, plus a whole-window drop target.
- **`[+ Add documents]` in the Library header row**, beside Sort / View / Select
  (`LibraryPane.svelte`, `.libnav`) — moved there 2026-08-24 on the user's call, from the topbar's
  right cluster, which is identity + config by design. An **app-menu entry** keeps the action
  reachable from Chat; window-wide drag-drop was never mode-specific.
- A dropped folder recurses fully, matching `registry.scan_sources` (`root.rglob("*")`, no depth
  limit). The count is stated before anything is copied.

### AD2 — the review sheet
Per-file verdicts, computed server-side, shown before any mutation.

| Verdict | Source | Row behaviour |
|---|---|---|
| **Add** | `extractors.is_supported` | ticked |
| **Unsupported** | `extractors.get_format_status` | greyed, unticked, advisory **verbatim** |
| **Already in your library** | `sha256(source bytes)` vs registry | unticked, links the existing document |
| **No text layer** | page-1 probe — **cost unmeasured, RG-030 gates this row** | ticked, warned |

Above a threshold the sheet paginates; the batch is never capped. **Every non-Add verdict sorts to
the top**, so the visible rows always contain all warnings, duplicates and unsupported files and
"and N more" is only ever clean ones.

### AD3 — apply
Copy **or** register per the user's choice, write `SourceFile`, refresh the panel. Indexing is
**ticked by default** with a time estimate, and untickable for staging. A batch that fails part-way
**stops and asks**: *Keep the N* / *Undo all*.

### AD4 — first-run empty state — **done 2026-08-28**
The Library empty state becomes the drop target. ~~alongside the existing demo-corpus offer~~ —
**there is no demo-corpus offer in the UI to sit alongside.** The demo corpus is a CLI script
(`scripts/download_corpus.py`); nothing in the app offers it, and `readiness.setup_state`'s
`documents` step only says "Point the app at a folder". Recorded rather than invented: building a
download offer to satisfy the wording would have been a feature nobody asked for.

Built: a dashed drop zone carrying the format list and an **Add documents** button that opens the
AD1 chooser, highlighting from `accept.dragging` — the same window-level Tauri signal the chooser
and the full-window veil use, since a DOM drop handler would never fire. Where the accept surface
does not exist (a plain browser) the button is **not rendered at all** and the reason is shown
instead — verified in both branches.

It also retired the old copy, "Point doc_assistant at a folder of your documents", which named the
**code identity** rather than the product (ADR-012 is wordmark-only) and described the pre-AD3b
model where the library was a folder you aimed at rather than somewhere you add documents. The
same stale sentence still lives in `readiness.setup_state` — that one is CS1/CS2's to fix.

### CS1/CS2 — Settings — **done 2026-08-28**
Folder picker replacing the paste-a-path input (text field kept as an override; the
"folder doesn't exist yet" warning stays). Relabel: the folder is *where Provenote keeps the
documents you add*, not *the folder holding your documents* — the same sentence currently describes
both models. **Nothing else in Settings** — the reorganisation is P7.

Built: **Library folder** (was "Source folder") with a **Browse…** button beside the field, opening
a single-select folder picker titled *"Choose the folder Provenote keeps your documents in"*. The
picker *fills the field*; the user still confirms with the same button they would have used after
typing, so the text input remains a real override — which it has to be, because a picker cannot
express a folder that does not exist yet, and that is exactly what the retained warning is about.
`pickPaths` gained `multiple`/`title`, defaulting to the old behaviour so no other caller moved.
No Browse button is rendered where there is no picker.

The relabel also retired the *other* copy of the pre-AD3b sentence, in
`readiness.setup_state` — it told first-run users to "Point the app at a folder", which is the
model AD3b replaced. **Nothing else in Settings was touched.**

## Contract — `src/doc_assistant/db/models.py` (modified, additive)

`SourceFile` gains:
- `root_id: str` — FK to the new `SourceRoot`. Existing rows backfill to the copy-in root.
- `origin: Literal["copied", "referenced"]` — what the delete dialog branches on.
- `source_sha256: str | None` — the ADR-046 add-time identity. Nullable: rows predating this spec
  have never been hashed, and backfilling is a scan, not a migration.

New `SourceRoot`: `id`, `path`, `kind` (`library` | `referenced`), `added_at`. Exactly one `library`
root, resolved from `app_settings.get_source_dir()`.

Rides the additive `init_db()` `create_all`, like `Figure`. **`uq` on `(root_id, rel_path)`**, which
is the pair that replaces today's `rel_path` key.

## Contract — `src/doc_assistant/ingest/registry.py` (modified)

- `scan_sources(session, root)` becomes per-root; the whole-library scan iterates roots.
- `derive_status` unchanged in meaning; `missing` is now per-root.
- `resolve_selection` accepts paths under **any** registered root; a path under none is still
  `InvalidSelection` (the 400-before-anything-starts guarantee holds).

## Contract — `src/doc_assistant/library/add.py` (new)

```
inspect(paths: list[Path]) -> list[FileVerdict]     # pure-ish: stat + hash + probe, no writes
add(verdicts, *, mode, index: bool) -> AddResult    # copy/register + SourceFile writes
```
`FileVerdict` carries `path`, `verdict`, `advisory`, `size`, `sha256`, `existing_document_id`.
`AddResult` carries per-file outcome — never a batch total (the Track B lesson, applied here).

**`inspect` performs no mutation and `add` performs no extraction.** Indexing stays the ingest
path's job.

## Contract — `src/doc_assistant/library/documents.py` (modified)

`delete_document(document_id, chroma_db, *, delete_file: bool)`. **ADR-046 amendment:**
- `delete_file=False` (default) — row + chunks + figures + cache; source untouched.
- `delete_file=True` — today's ADR-014 behaviour, bin-first-then-remove, ordering preserved.
- A `referenced` document with `delete_file=True` is allowed but the **caller must have shown the
  real path** (UI contract, asserted in the API test, not enforceable in the library).

## Contract — `apps/api/routers/sources.py` (modified)

- `POST /api/documents/inspect` → `list[FileVerdict]`. No mutation. 400 on a path outside every
  registered root **and** outside the user's chosen drop set.
- `POST /api/documents/add` → `AddResult`. 409 while an ingest is running (mirrors `/api/ingest`).
- `DELETE /api/library/documents/{id}` gains `delete_file: bool = False`.

## Contract — `apps/desktop` (modified)

`lib/library/AddDocuments.svelte` (the sheet) · `lib/library/dropzone.svelte.ts` (pure verdict
sorting + pagination — **testable under `node --test`, and it must be**) ·
`lib/core/api/documents.ts` · `Topbar.svelte` (the button) · `Settings.svelte` (CS1/CS2) ·
`LibraryBrowser.svelte` (empty state + drop target).

## Test cases (write these first)

**Unit (`tests/unit/`)**
1. `inspect` returns `unsupported` with `get_format_status`'s advisory verbatim for `.doc`/`.tex`/`.mobi`.
2. `inspect` flags a byte-identical file already registered, under a *different name*, as a duplicate.
3. `inspect` mutates nothing (registry row count unchanged).
4. Verdict sort puts every non-Add row before every Add row, stable within groups.
5. Pagination reports the true total, never the page length.
6. `delete_document(delete_file=False)` leaves the file on disk and removes the row.
7. `delete_document(delete_file=True)` preserves ADR-014's bin-then-remove ordering, and a locked
   file leaves the row intact.
8. 0 files in, 0 files out — no crash, no partial state.

**Integration (`tests/integration/`)**
9. Add 3 files copy-mode → registered under the library root, files present at the destination.
10. Add 3 files reference-mode → registered under a second root, **no copies made**.
11. The same file added under both modes is one document, not two.
12. A batch that fails at file N halts and reports N-1 kept; nothing is half-registered.
13. `POST /api/documents/add` returns 409 while an ingest runs.
14. Adding with `index=true` reaches `POST /api/ingest` with an explicit `paths` list.

**Frontend (`node --test`)**
15. Verdict sorting and pagination in `dropzone.svelte.ts` (pure — no harness needed).

## Definition of done

- W0 recorded with evidence and the route named (**done 2026-08-21**); the two runtime
  assertions **confirmed in a real Tauri window 2026-08-24**. Outstanding: one human drag,
  to prove a real drop delivers a non-empty `paths` array.
- **AD3b landed 2026-08-25** — `SourceRoot`, `SourceFile.root_id`, the `(root_id, rel_path)` key,
  multi-root scan/selection/exclusion, and `apply_add(mode="reference")`. Two consequences worth
  reading before touching this area: the wire identifier for a registered file is now
  `registry.source_key` (`"<root_id>:<rel_path>"`, bare rel_path still means the library root),
  and duplicate detection joins each row to its root — resolving a `rel_path` against the library
  folder is what silently missed duplicates under a referenced root.
- **The delete half landed 2026-08-28** — `delete_document(*, delete_file=False)` (test cases 6 and
  7), `DELETE /api/library/documents/{id}?delete_file=`, `source_path` on every library row, and a
  dialog that asks, defaults to library-only and **names the path**. Closed KI-52 with it (the
  registry row now goes with the file, and only with the file).
- **AD4 landed 2026-08-28** — the empty state is the drop target.
- **CS1/CS2 landed 2026-08-28** — folder picker + the relabel. **Every work item is now built**;
  what remains of the DoD is the one human drag (below).
- All 15 test cases pass; `just typecheck` clean; ruff clean; `svelte-check` 0/0.
- A document can be added by drag-and-drop **and** by the button, in both modes, and indexed.
- ~~Delete asks, defaults to library-only, and names the path for referenced files.~~ — **done
  2026-08-28.** The path is named for *every* document, not only referenced ones: a copied
  document's path is equally worth seeing before a destructive click, and it means the UI needs no
  join to `SourceFile.origin`. Where no path is recorded, the destructive option is not offered at
  all — refusing to name the target is refusing the guarantee ADR-046 asked for.
- `docs_check --strict` 0/0; DEVLOG entry per logical change; ADR-046 status → `accepted (built)`.
- ~~**Not done until** an EPUB and an HTML file have been added and indexed through the UI~~ —
  **done 2026-08-28.** `tests/fixtures/documents/article.html` and `treatise.epub` were added
  together through the chooser (which also exercised the multi-file path), indexed with 0 errors,
  and rendered with correct EPUB/HTML type badges. Extraction was clean: accents survived, the
  table round-tripped, all three chrome markers were stripped, and the content reached both vector
  stores and the keyword index. **It found two defects in what the row says about them —
  `extraction_health='broken'` for any short document and a hardcoded `extractor_used='pymupdf'`
  — filed as KI-53.** DEVLOG 2026-08-28 (7).

## Open questions

| # | Question | Status |
|---|---|---|
| 1 | Which accept route, and does it add an npm dep? | **resolved 2026-08-21** — `withGlobalTauri` + injected API; **no npm dep**; see W0 |
| 2 | Cost of the page-1 text-layer probe across a 500-file batch | **parked — RG-030** |
| 3 | Does a referenced root need a "re-scan on app start" pass, or is missing-detection on demand enough? | **resolved 2026-08-25** — **on demand is enough; no startup pass.** `scan_sources` already runs whenever the Library loads, so a vanished file derives `missing` the moment anyone looks, and one behaviour covers both root kinds. A startup pass would `rglob` every referenced root at launch — on a network share or an unplugged drive that is slow or blocking, at the worst possible moment (RG-012 tracks cold start). **Consequence, decided with it:** an unreachable *root* is not the same fact as deleted *files*, so `RootView.available` is derived per scan and `SourceView.root_available` carries it, letting the UI say "the drive is not connected" instead of showing 400 identical `missing` badges. Availability is never persisted — a drive that is unplugged now may be back in a second |

*(cpc `docs_check` rule 15 fails a started sprint with an `open` row — resolve 1 and 3 at
sprint-start, not mid-build.)*

## Decision ledger — scoped grill, 2026-08-21

Full ledger with deciding reasons and "reopens if" in
`docs/PLAN_2026-08-20_user-friendly-ingestion.md` §8. Task-level resolutions owned here:

| Branch | Resolution |
|---|---|
| Folder recursion | Fully recursive — matches `scan_sources`; count shown first |
| Default library folder | `app_settings.get_source_dir()` precedence, unchanged |
| Partial copy failure | **Stop and ask** — *Keep the N* / *Undo all* |
| Large batch | **Paginate**, no cap; non-Add verdicts sort to the top |
| Add ⇒ Index | **Ticked by default**, untickable |

ADR-class resolutions (placement, delete, identity) live in **ADR-046**, not here.

## Out of scope, with owners

- **LLM-assisted ingestion** — P2, its own spec + ADR, cost-gated by KI-4.
- **Settings reorganisation** — P7. Only CS1/CS2 here.
- **Zotero / Calibre adapters** — ROADMAP row 17; this spec builds the substrate they need.
- **OCR recovery** — ADR-039. Track B *surfaces* the scanned-PDF problem; nothing here solves it.
- **Collapsing the two identities** — RG-027 / ADR-042. This spec deliberately adds the source-hash
  side and states the coexistence.
