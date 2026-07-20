<!-- status: active · updated: 2026-07-20 · class: contract -->

# Spec — Corpus folders, F3: demo auto-assign (ingest sha-match + one-time backfill)

**Status:** contract for **F3**, the last step of the ADR-025 carve (F1 folders → F2 retrieval
scoping → **F3 demo auto-assign**). Written at build time per the house pattern.

**Owner:** Claude Code. **ADR:** `docs/decisions/ADR-025-corpus-folders-retrieval-scope.md`
(fork 2). **Depends on:** F1 (`docs/specs/feature-corpus-folders.md` — the folder API this writes
through) · the demo collection + `--remove-demo` cleanup shipped in `0c777d8` / `ed3a7c8`.

The ADR line this implements, verbatim:

> **Demo membership is automatic at ingest + one-time backfill; user edits always win** — a newly
> ingested file whose SHA-256 matches a demo manifest pin joins the folder; assignment happens only
> at ingest of a new document, so a user removal is never re-fought (ADR-013 user-wins pattern).
> Reverses if the manifest must become the ongoing authority.

---

## Decisions (F3-local; the ADR forks are upstream and not re-litigated)

| # | Decision | Reason / reopens-if |
|---|----------|---------------------|
| **M1** | **The trigger is "a `Document` row that did not exist before this ingest run", computed in `ingest.main()` as a set-difference of `get_document_row_hashes()` around the processing loop.** Not `process_one_document`'s `"added"` return. | `"added"` is also returned for **re-**ingests — the inverse-orphan repair path (`ingest/__init__.py:471`) reprocesses a document that already has a row, and `--rebuild` reprocesses everything. Keying on "the row is new" is literally the ADR's "ingest of a **new** document", and it keeps `process_one_document` (the locked ingest hot path) **completely untouched** — no new parameter, no new return type. |
| **M2** | **User removals are never re-fought, because a document is only ever considered once — on the run that first created its row.** No periodic reconciliation, no "the manifest is the truth" sweep. | ADR-013 user-wins. The manifest is an *origin* signal, not an ongoing authority. **Reverses if** the ADR's own "manifest must become the ongoing authority" clause is ever triggered — that needs an ADR amendment, not a code change. |
| **M3** | **`--rebuild` is the one honest exception, and it is logged.** A rebuild deletes every `Document` row, so every document looks new and demo membership is re-applied. | This is *desirable* here (see M9: rebuild destroys all folder membership; re-applying is a partial self-heal) but it **is** a re-fight of a demo removal. It is stated in the runner log and in this spec rather than hidden. |
| **M4** | **Matching is by file **bytes** (size fast-path, then SHA-256), never by filename.** Same rule as `match_pinned_sources`. The per-run cost on a non-demo corpus is one `stat()` per newly-ingested file; a SHA-256 read happens only on a size collision with a pin. | Rename-proof, and the manifest already pins `sha256` + `bytes`. Reusing the existing rule means the *same* files that `--remove-demo` finds are the ones auto-assigned — one definition of "is a demo file" (ADR-015). |
| **M5** | **The folder is resolved by a persisted id (`demo_folder_id` in `settings.json`), not by name** — so **renaming it is respected**, which is what ADR-025's "ordinary folder — renamable, deletable" promises. Creation falls back to `create_folder("Demo corpus")`, which is an idempotent get-or-create, so a pre-existing same-named folder is adopted rather than duplicated. | Name-keyed resolution would create a *second* "Demo corpus" the first time a user renamed theirs. `app_settings` (JSON in the data home) is chosen over a `folders.origin` column because it is a per-install **pointer**, not document data, and needs no schema change. **Reopens if** a second auto-managed folder ever appears — then it earns an additive `folders.origin` column. |
| **M6** | **Deleting the folder is respected until a *new* demo document is ingested** — at which point the folder is re-created to hold it. No tombstone. | A tombstone would mean coupling the generic `delete_folder` to demo semantics. The thing ADR-013 protects is the **per-document** removal, and that is never re-fought (M2). Ingesting a *new* demo paper is a fresh user action that reasonably wants a home. Stated in the runner log. |
| **M7** | **The folder is created only when there is at least one document to put in it.** No empty "Demo corpus" ever appears. | The 0-document robustness contract, and the honest-empty rule F1 already follows. |
| **M8** | **The backfill runner refuses to re-run once it has succeeded** (`demo_backfill_done` in `settings.json`); `--force` overrides with a loud warning. Dry-run is the default; `--apply` writes. | A second backfill would re-add exactly the documents the user removed — the ADR-013 violation this whole feature is shaped to avoid. The flag turns a silent re-fight into an explicit, opt-in one. **Reopens if** a per-document "was auto-assigned" marker is ever wanted (that would make re-runs safe without a flag). |
| **M9** | **`--rebuild` silently destroys ALL folder membership** (`delete(DBDocument)` → `document_folders` `ON DELETE CASCADE`), leaving every folder as an empty shell. F3 does **not** fix this — it adds a **warning that names the count** before the delete, and logs it as a KNOWN ISSUE. | Found while specifying M1/M3. Out of F3's scope to fix properly (the fix is a membership snapshot + restore, its own change), but a data loss the user cannot currently see is exactly the class of thing "inform, don't block" exists for. |
| **M10** | **A missing manifest is a silent no-op, by design.** `tests/eval/corpus_manifest.yaml` is **not** bundled into the PyInstaller sidecar and `PROJECT_ROOT` climbs into `%TEMP%` when frozen, so the packaged app never auto-assigns. | Coherent, not a gap: the demo corpus is a **repo-clone flow** end to end (`python -m scripts.download_corpus --demo`), so a packaged-app user has no demo files to assign either. Logged at `debug`, never at `warning` — a frozen build hitting this is normal, not broken. |
| **M11** | **No UI change, no API route, no "demo" badge.** The folder is ordinary: it shows in the rail, filters the grid, and scopes retrieval (F2) exactly like a hand-made folder. | ADR-025 fork 1: one organizing concept, one write surface. A demo-specific affordance would be a second one. |

## Contract

### New — `src/doc_assistant/demo_corpus.py`

```
MANIFEST_PATH        = config.PROJECT_ROOT / "tests" / "eval" / "corpus_manifest.yaml"
DEMO_COLLECTION      = "demo"
DEFAULT_FOLDER_NAME  = "Demo corpus"

@dataclass(frozen=True) AssignResult: folder_id|None, folder_name|None, added:list[str], already_member:int

load_demo_pins(manifest_path=None)      -> list[SourcePin]        # [] if absent/unparseable (M10)
pins_by_size(pins)                      -> dict[int, list[SourcePin]]
file_matches_demo(path, by_size)        -> bool                   # stat fast-path, then sha256 (M4)
resolve_demo_folder(*, create)          -> FolderSummary | None   # id-keyed (M5)
apply_assignments(document_ids)         -> AssignResult           # idempotent; creates folder iff non-empty (M7)
assign_new_documents(doc_hashes)        -> AssignResult           # the ingest hook (M1)
backfill_matches(sources_dir=None)      -> list[SourceMatch]      # reuses match_pinned_sources (M4)
```

### Changed — `src/doc_assistant/ingest/__init__.py`

- `main()`: snapshot `get_document_row_hashes()` immediately before the processing loop (i.e. **after**
  the `--rebuild` wipe — M3), diff after, hand the new hashes to `assign_new_documents`. Wrapped so a
  failure logs and never fails an otherwise-successful ingest.
- `--rebuild` branch: count `document_folders` rows and `log.warning("rebuild_clears_folder_membership", …)`
  before the delete (M9).
- `process_one_document` is **not touched.**

### Changed — `src/doc_assistant/app_settings.py`

`get_demo_folder_id()` / `set_demo_folder_id(id)` · `demo_backfill_done()` / `mark_demo_backfill_done()`.
Same fail-safe read pattern as the existing getters.

### New — `scripts/backfill_demo_folder.py`

```
python -m scripts.backfill_demo_folder                  # plan only (default)
python -m scripts.backfill_demo_folder --apply          # assign; refuses if already run (M8)
python -m scripts.backfill_demo_folder --apply --force  # re-run anyway (loud warning)
python -m scripts.backfill_demo_folder --dest <dir>     # scan a different sources dir
```

Prints per-file state (ingested / file-only / ambiguous / already a member) and a summary. Never
touches the chunk store, never deletes anything, no LLM, no network — so no provider guard is needed.

## Tests

- **Unit** (`tests/unit/test_demo_corpus.py`): the manifest loader selects **only** `collection: demo`
  (eval entries excluded) and returns `[]` for a missing/blank/malformed file (M10) · `pins_by_size`
  groups correctly · `file_matches_demo` is True on exact bytes, False on a size match with different
  bytes, False on an unknown size, and **True on a renamed file** (M4) · `apply_assignments([])`
  creates **no** folder (M7) · `apply_assignments` is idempotent and reports `already_member` ·
  `resolve_demo_folder` follows the stored id **through a rename** and re-creates after a delete (M5/M6).
- **Integration** (`tests/integration/test_demo_auto_assign.py`): `assign_new_documents` assigns a
  demo-matching row and skips a non-matching one · **the user-wins guard** — assign, remove by hand,
  assign the *same* hash again → the document stays out (M2) · a hash with no row is skipped · an
  empty hash set is a no-op that creates no folder · the backfill run-once flag blocks a second
  `--apply` and `--force` overrides it (M8).
- **Gates:** `ruff` · `ruff format` · `mypy --strict src` · `bandit src` · full `pytest` ·
  `docs_check --strict` · `integrity_check`. No frontend change → `svelte-check` unaffected (run anyway).
- **Live ($0, offline):** on the real corpus — the runner's dry run against `data/sources/`, then
  `--apply`, then the folder visible in the rail with the right count and usable as a chat scope
  (F2). Every probe reverted; the DB left as found.

## Definition of done

1. `demo_corpus.py` with the surface above, `mypy --strict` clean, 0-document / no-manifest safe.
2. Ingest assigns newly-ingested demo files and **never** re-assigns an existing document (M1/M2).
3. Backfill runner with dry-run default + run-once guard (M8).
4. M9's rebuild warning in place; the issue logged in `.claude/KNOWN_ISSUES.md`.
5. Unit + integration tests above green; full gate battery green.
6. One `docs/DEVLOG.md` entry · ROADMAP L10 row · `docs/ui-checklist.md` folders row closed
   (F1+F2+F3 shipped) · baton entry.

## Out of scope (F3)

An "Eval corpus" folder for the other collection (not in ADR-025) · any UI affordance for demo
membership (M11) · fixing M9 properly (membership snapshot/restore across `--rebuild`) · making
`--remove-demo` also drop the now-empty folder (leaving it is the reversible end state) ·
per-document auto-assign provenance (the M8 reopener) · nesting · per-folder enrichment (ADR-025
fork 5, parked) · RG-020's synthetic 10k measurement.

## Known end states (stated, not fixed)

- `download_corpus --remove-demo --apply` removes the demo documents but **leaves the folder**, now
  empty. Reversible and visible; deleting it is one click.
- After `--rebuild`, all folder membership is gone (M9) and only the **demo** folder repopulates
  (M3). A hand-made folder stays an empty shell until refilled by hand.
