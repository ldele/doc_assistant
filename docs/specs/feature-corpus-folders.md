# Spec — Corpus folders, F1: folders end-to-end (CRUD + Library UI)

**Status:** contract for **F1** of the ADR-025 carve (F1 folders → F2 retrieval scoping → F3 demo
auto-assign). Written at build time per the house pattern. **F1 only** — this spec does not touch
`pipeline.py`, chat, or the demo manifest.

**Owner:** Claude Code. **ADR:** `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md`
(design-locked 2026-07-20). **Supersedes** the Phase-B contract in
`docs/specs/feature-library-redesign.md` (L4) — see "Reconciliation with L4" below, which is the
part of this spec that carries new judgment.

---

## Reconciliation with L4 (the `⚠ RECONCILE AT F1 SPEC TIME` note in the baton)

The baton asked F1 to "compose both — subfolder mirroring for the user's own files, manifest
sha-match for the demo set". **That instruction is stale and is not followed.** L4's own
§"Folder population — REOPENED then SHELVED (user, 2026-07-15)" already killed mirroring: Decision 6
carried an explicit *reopens-if the `source_dir` is intentionally flat*, and the user **confirmed
that condition true** on 2026-07-15. L4's recorded consequence is verbatim: path-derived mirroring
is shelved, and if folder grouping is ever wanted on a flat corpus the path is **manual assignment
(option B) — its own ADR, the first browse-time write path**.

**ADR-025 is that ADR.** So the composition is:

| L4 Phase-B item | Fate in F1 |
|---|---|
| Ingest-time subfolder mirroring | **Stays shelved** (source dir is flat by design). Not built. |
| Backfill runner for mirrored paths | **Stays shelved** — F3's sha-match backfill is a different runner for a different rule. |
| Manual assignment (option B) | **THIS IS F1.** Unblocked by ADR-025. |
| `GET /api/library/folders` + filter params | **Built here**, extended to full CRUD. |
| Rail Collections tree filters the grid | **Built here** (rail renders from the API, not from doc payloads — see D3). |
| L4 Decision 7 "read-only still / browsing writes nothing" | **Narrowed, not dropped** — see D7. |

Inherited L4 locks that still hold: the nav-tree rail shape, drill-down + Back, the honest-empty
rule (D8), `docsFor` client-side collection filtering, grid⇄list persistence. Nothing in F1 changes
them.

## Decisions (F1-local; the ADR-025 forks are upstream and not re-litigated)

| # | Decision | Reason / reopens-if |
|---|---|---|
| **D1** | **Flat folders in v1** — every folder is created with `parent_folder_id = NULL`; no nesting control in the UI. The hierarchical schema column is left untouched and unused. | ADR-025 fork 1 names "deep nesting vs flat scope" as the *reopener for the whole folders-are-groups identity*. Nesting also poses a question ADR-025 never answered — **does scoping a parent include its children's documents?** — which F2 would have to invent an answer to. Flat keeps F2 unambiguous and costs nothing to reverse (the column is already there). **Reopens if** the user asks for nesting; the answer must then be written into ADR-025 as an amendment *before* F2. |
| **D2** | **Membership is by folder id, everywhere.** `list_documents(folder=<name>)` is replaced by `list_documents(folder_id=<id>)`; the wire payload gains `folder_ids`; the rail's collection value is a folder id. | `uq_folder_name_parent UNIQUE (name, parent_folder_id)` **does not bite at root level** — SQLite treats NULLs as distinct, so two root folders may legally share a name, and a name-keyed filter is then ambiguous. Zero callers used `folder=`, so this is a free correction. F2 also needs ids to resolve doc-hash sets. |
| **D3** | **The rail renders folders from `GET /api/library/folders`, not from `folderGroups(documents)`.** | A folder derived from document payloads **cannot exist while empty** — and an empty folder you cannot see is a folder you cannot add documents to. The derived-groups helper is retired for folders (it stays for types/keywords). |
| **D4** | **Application-level uniqueness: folder names are unique case-insensitively at root.** `create_folder` is an idempotent get-or-create on the folded name (mirrors `create_keyword_family`); `rename_folder` rejects a collision with `ValueError`. | The DB constraint can't enforce it (D2). Idempotent create mirrors the shipped families surface, so the API behaves the same way twice. |
| **D5** | **Counts and membership exclude archived documents**, matching `list_documents`. Archived members keep their rows (un-archiving restores them). | The grid never shows archived docs; a count that disagrees with the grid is a lie. Restated because F2 inherits it: a scoped turn will not retrieve archived members. |
| **D6** | **Deleting a folder never touches documents.** It deletes the `folders` row; `document_folders` rows fall away by FK `ON DELETE CASCADE` (verified: `PRAGMA foreign_keys=ON` is set in `db/session.py:35`). No ADR-014 delete path is involved. | Folder ≠ container-of-files. A guard test asserts document count is unchanged after a folder delete. |
| **D7** | **The write-trap guard is narrowed, not deleted.** L4's "browsing writes nothing" becomes: *rendering* the library (`GET /api/library/documents`, `/folders`, opening a doc) writes nothing; writes occur only on the explicit folder mutation routes. | F1 is by definition the browse-time write path. The guard still has value against accidental writes on read routes. |
| **D8** | **F1 must not imply that chat is scoped.** No "chat with this folder" affordance, no scope selector, no folder chip on answers — and the Manage-folders view carries one honest line: folders organise the Library; chat still searches every document (until F2). | This is the `is_archived` lesson applied to F1 itself. A user who filters the Library to "Demo corpus" and then asks a question will otherwise reasonably assume the answer is scoped. Shipping the filter silently ahead of the scoping is exactly the class of lie ADR-025 exists to prevent. **Removed by F2**, which is what makes the statement true. |
| **D9** | **UI vocabulary is "Folders"** (ADR-025 fork 1), so the rail section header "Collections" is renamed. `LibraryCollection` stays the *code* name for the rail's selection union (all/type/date/folder/keyword) — it is not user-visible. | One word per concept; "collection" already means "the active rail selection" in `lib/library.ts`. |

## Contract

### Backend — `src/doc_assistant/library.py` (all logic; mirrors the keyword-families surface)

```
@dataclass FolderSummary: id, name, description|None, parent_id|None, doc_count:int

list_folders()                                    -> list[FolderSummary]      # name-sorted, casefold
get_folder(folder_id)                             -> FolderSummary | None
create_folder(name, description=None)             -> FolderSummary            # idempotent on folded name; blank -> ValueError
rename_folder(folder_id, new_name)                -> FolderSummary | None     # None = unknown; collision/blank -> ValueError
delete_folder(folder_id)                          -> bool                     # True if it existed
add_documents_to_folder(folder_id, document_ids)  -> FolderSummary | None     # idempotent; unknown doc ids skipped
remove_documents_from_folder(folder_id, doc_ids)  -> FolderSummary | None     # idempotent
folder_document_ids(folder_id)                    -> list[str]                # non-archived members
```

`list_documents(..., folder_id=None)` replaces the `folder` (name) parameter — join
`document_folders` on the id. `DocumentSummary` gains `folder_ids: list[str]` beside the existing
display-only `folders: list[str]`.

### API — `apps/api/main.py` + `models.py` (thin; mirrors `/api/library/keyword-families`)

```
GET    /api/library/folders                                  -> list[LibraryFolderPayload]
POST   /api/library/folders                     FolderCreate -> LibraryFolderPayload
PATCH  /api/library/folders/{folder_id}         FolderRename -> LibraryFolderPayload   404 unknown, 400 blank/collision
DELETE /api/library/folders/{folder_id}                      -> {"ok": bool}
POST   /api/library/folders/{folder_id}/documents  FolderMembers -> LibraryFolderPayload
DELETE /api/library/folders/{folder_id}/documents/{doc_id}      -> LibraryFolderPayload
```

`LibraryDocumentPayload` gains `folder_ids: list[str]`. `types.ts` mirrors in the same change.

### Frontend — `apps/desktop/`

- **`lib/LibraryManageFolders.svelte` (new)** — modal overlay reusing the
  `LibraryManageKeywords.svelte` shell (scrim + centred dialog + Esc-to-close, dumb-by-design:
  `App.svelte` owns the list and calls the API). Create a folder; rename inline; delete with
  confirm; select a folder → a searchable, checkbox document picker (all non-archived docs, member
  state pre-checked) that adds/removes in bulk. Carries the D8 honest line.
- **`lib/LibraryGrid.svelte`** — one item, **"Add to folder…"**, in the existing per-tile `⋯` menu
  beside Edit metadata / Reveal / Delete. ⚠ the menu's flip math hardcodes its own height
  (`r.bottom + 96 > window.innerHeight`) — bump it for the extra row.
- **`lib/Sidebar.svelte`** — library mode: section header "Collections" → **"Folders"**; render
  `folders` (from the API, with `doc_count`) instead of `folderGroups(documents)`; collection value
  is the folder **id**; empty-state copy replaces the stale "arrives with source-dir mirroring
  (Phase B)" with a "No folders yet — create one in Manage folders" + the entry point.
- **`lib/library.ts`** — `docsFor` matches `{kind:'folder', value}` against `d.folder_ids`;
  `collectionLabel` resolves a folder id → name via a passed map; `folderGroups` deleted.
- **`App.svelte`** — `folders` state + load on library mount, `manageFoldersOpen`, the five
  mutation handlers (each refreshes folders + documents).
- **`lib/api.ts` / `lib/types.ts`** — the six calls + `LibraryFolder`, `folder_ids`.

## Tests

- **Unit** (`tests/unit/test_library_folders.py`, new): create idempotency (case-insensitive);
  blank name → `ValueError`; rename to an existing name → `ValueError`; rename/add/remove on an
  unknown id → `None`; add/remove idempotent; `doc_count` excludes archived (D5); `list_documents`
  filters by `folder_id`; a document in two folders (m2m overlap, ADR-025 fork 1).
- **Integration** (`tests/integration/test_api_folders.py`, new): the six routes end-to-end on the
  fake-controller app; 404 on unknown folder, 400 on blank/collision; **D6 guard** — document count
  and `document_folders` rows for *other* folders unchanged after a delete; **D7 guard** — a
  full read cycle (`GET /documents`, `GET /folders`, `GET /documents/{id}`) writes no rows.
- **Gates:** `ruff` · `ruff format` · `mypy --strict src` · `bandit src` · full `pytest` ·
  `docs_check --strict` · `npm run check` (svelte-check 0/0).
- **Live:** preview harness on the real 76-doc corpus ($0/offline) — create a folder, add docs,
  rail shows it with a count, selecting it filters the grid, remove + delete, both themes, 375px.

## Definition of done

1. Folder CRUD + membership in `library.py`, returning dataclasses, `mypy --strict` clean.
2. Six routes + wire models + `types.ts` mirror, integration-tested.
3. Manage-folders overlay; rail renders real folders (including empty ones) and filters the grid.
4. D8 honest line present; no chat-scoping affordance anywhere.
5. Unit + integration tests above green; full gate battery green.
6. One `docs/DEVLOG.md` entry; `docs/ui-checklist.md` §3 row updated (F1 done, F2/F3 open);
   ROADMAP row; L4 spec cross-linked to this reconciliation.

## Out of scope (F1)

Retrieval scoping and the per-turn selector + provenance/answer chip (**F2**) · demo sha-match
auto-assign and its backfill (**F3**) · nesting (D1) · drag-and-drop assignment · per-folder
enrichment (ADR-025 fork 5, parked) · `Tag` CRUD (the same shape, deliberately not bundled) ·
RG-020 / RG-021 (both F2/eval-harness rigor items, neither gates F1).
