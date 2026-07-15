# Spec — Selective ingestion: source registry + selection-scoped ingest (S1) + Tauri sources panel (S2)

**Status:** **LOCKED (grilled 2026-07-15, `grill-me`; ledger at foot).** Designed 2026-07-02; grilled
after the L4 library redesign confirmed a **flat, all-PDF source** (user, 2026-07-15). Roadmap PRs S1/S2.

> **Grill lock amendment (2026-07-15) — governs where it conflicts with the original body below.**
> The corpus is a **single flat folder of all-PDF papers** (confirmed, not a metadata gap). That
> changes the selection model:
> - **`doc_type` behavior is DEFERRED** (was Decision 4 / part of ADR-1). On an all-PDF corpus it is
>   manual classification with **no consumer** (explicitly not a chunking/embedding lever — per-type
>   routing needs a measured eval win first). S1 ships **no** `doc_type` seeding, **no** `doc_type`
>   in `PATCH /api/sources`, **no** `doc_type` UI, and reads it nowhere. Selection in v1 is
>   **status (`new`/`changed`/`ingested`/`missing`) + `excluded` + explicit `paths`** only.
> - **BUT the `doc_type` column ships now, dormant** — a nullable `SourceFile.doc_type` with nothing
>   wired to it. Reason: `init_db()`/`create_all` creates new tables but cannot ALTER a column onto an
>   existing one (the build-time note below), and no ALTER-helper exists — so a dormant column now
>   makes doc_type's future return a pure behavior add, not a migration. **Reopens (doc_type behavior)
>   if:** a second format enters the corpus, or a per-type routing eval wins.
> - **The registry (`SourceFile`) is KEPT** (not reduced to a stateless listing): persistent
>   `excluded` has nowhere else to live (an un-ingested file has no `Document` row), and the table is
>   still the status-listing source and the PR-17 adapter seam. It is just **minimal** now
>   (identity + `excluded` + dormant `doc_type`).
> - **`SourcePatch` / `PATCH /api/sources` accept `excluded` only** in v1 (drop the `doc_type` field);
>   `SourceView` still *reports* `doc_type` (always null) so the wire shape is forward-stable.
> - **S2 UI shape** (dedicated `Sources.svelte` sidebar mode vs folding the file list into the existing
>   Settings ingest section) is **PARKED to S2 kickoff** — it does not block S1 (the API is UI-agnostic).
> - **Unchanged and kept as drafted:** `dry-run` (Decision 6), explicit-selection-overrides-`excluded`
>   (Decision 5), the `new/changed/ingested/missing` status truth table (Decision 3), the
>   `SourceFile`-rides-`create_all` migration, vendor-neutral registry (ADR-3), rename→per-path-metadata
>   loss as a documented v1 limitation (ADR-1 consequence).

**Owner of execution:** Claude Code (code + tests). One PR per increment: **S1** (backend + CLI + API), **S2** (Tauri UI).
**Pattern reference:** the **locked primary ingest path** (extract → markdown → chunk → embed → store,
`docs/decisions.md`) is untouched — this feature only changes *which files enter it*. The registry is
pre-ingest bookkeeping in the library SQLite, not an enrichment sidecar (it derives nothing from content).
**Foundation (shipped):** CLI `--path` scoping (`ingest/__main__.py`), the content-hash dedup gate +
inverse-orphan reconciliation (`ingest/__init__.py:381-408`), the M4 data-home flow (`/api/settings` +
`/api/ingest` + `apps/desktop/src/lib/Settings.svelte`), and `ingest.main(scope=…)` already accepting a
walk scope — the API just hardcodes it to the whole source dir (`apps/api/main.py:320`).

**Requirement.** Ingest today is all-or-nothing over one configured source dir (modulo the CLI-only
`--path`), and the Tauri app's only affordance is "ingest everything". A mixed corpus (scientific papers,
books, web captures, notes) needs user-controlled **selective ingestion**: (a) *batch* — pick a subset now;
(b) *on-need* — drive the pick from corpus metadata (status, document type). Both a script path and a UI
path, sharing one backend. **Hard constraint (user, 2026-07-02): no dependency on Zotero or any external
catalog — our SQLite is the system of record.** Zotero/Calibre (PR 17) become optional *producers* for
this registry later, never a requirement.

---

## ADR-1 — Registry: native `SourceFile` table; persist identity + user intent, derive status

**Context.** Selection-by-metadata needs somewhere to hang pre-ingest metadata (document type, "never
ingest this") on files that have no `Document` row yet. Today a file is either invisible (just bytes in
`sources/`) or fully embedded.

**Decision.** A new **`SourceFile`** table in `library.db`, one row per discovered file, keyed by
`rel_path` (relative to the current source dir). It persists **identity** (`rel_path`, `format`, `size`,
`mtime`, `first_seen`/`last_seen`) and **user intent** (`doc_type`, `excluded`) — and *nothing derivable*:
ingestion **status is computed at read time** from what already exists (`new` / `changed` / `ingested` /
`missing`; see Decisions 3). The scan that populates it is stat-only — no extraction, no hashing, no
content reads — so listing a large corpus is instant.

**Options considered.**
1. *Stateless computed listing, no table (rejected).* Cheapest, but there is nowhere to persist `excluded`
   or a pre-ingest `doc_type` — the "on-need with metadata" half of the requirement dies.
2. *Registry persisting identity + intent, status derived (chosen).* User data survives; nothing drifts.
3. *Full mirror with persisted status (rejected).* Status duplicates what the cache mtime + `Document`
   rows already encode; persisted copies drift (same reasoning as "wiki notes are derived and regenerable").

**Consequences.** Renaming a file orphans its row (`missing`) and creates a `new` one — per-path user
metadata is lost on rename (v1 limitation, documented; the *content*-hash dedup gate still prevents
re-embedding an unchanged renamed file, so the cost is metadata re-entry, not compute). Changing the
source dir in settings orphans all rows the same way; the next scan repopulates.

**Not to be confused with** `sources_manifest.py` (2026-06-27): that is a gitignored URL+sha256 manifest
for *reconstituting* the library on another machine. Complementary, different axis — the manifest answers
"where did these bytes come from", the registry answers "what should be ingested, as what". Neither
imports the other; the registry does no hashing precisely because the manifest already owns content pins.

## ADR-2 — Selection reaches the ingest core as an explicit file list

**Decision.** `ingest.main()` gains `files: list[Path] | None`. All predicates (checkbox picks, "all new",
"all papers", a folder) are resolved **above** the core — in `registry.resolve_selection` for the API/UI,
in flag parsing for the CLI — into explicit paths. The core's six locked stages see a list of files instead
of a walk; nothing else changes. Orphan cleanup stays global-only: it is skipped whenever `files` or
`scope` is set (the existing `--path` rule at `ingest/__init__.py:372-375`, extended) — a partial run must
never interpret "not selected" as "deleted".

**Options considered.** A predicate DSL pushed into the core (rejected — the core is locked and the
predicates belong to the UX layer); reusing `scope` with globs (rejected — `_resolve_walk_root` is a single
path by design, and glob semantics differ per shell/OS).

## ADR-3 — Vendor neutrality: adapters are optional second producers

**Decision.** The registry is populated by our own scanner. PR 17's Zotero (later Calibre) adapter becomes
*another writer* to the same registry through the same public functions (`scan`/`set_source_meta` shapes) —
mapping collections/item-types onto `doc_type` and future fields — and lives entirely behind the adapter
module. Nothing in `registry.py`, the API, or the UI imports or assumes vendor code (restates the roadmap
guard: *"Don't let Zotero/Calibre adapters leak vendor specifics past the extractor boundary"*). The
feature is **fully functional with no external tool installed**.

---

## Decisions

| # | Decision |
|---|---|
| 1 | **Register ≠ ingest.** `scan_sources()` walks the source dir with `stat` only and upserts `SourceFile` rows (`last_seen` refreshed; vanished files kept, they derive `missing`). Runs inside `GET /api/sources` and CLI `--dry-run`; no background watcher. |
| 2 | **Key = unique `rel_path`** under the current `app_settings` source dir. POSIX-style separators stored (portable across the KI-11 ASCII-relocation machinery — registry lives in SQLite, which is non-ASCII-safe). |
| 3 | **Derived status (pure fn):** `missing` (row, no file) · `new` (no cache entry and no `Document` row) · `changed` (source mtime newer than cache — the exact `is_cache_fresh` test from `ingest/cache.py:28`) · `ingested` (otherwise). `Document` linkage is derived by joining `Document.source_original` to the absolute path at read time — **no persisted FK** (nothing to drift). |
| 4 | **`doc_type ∈ {paper, book, web, note}`, nullable, on `SourceFile` only** (v1). Scan seeds a suffix default (`epub→book`, `html/htm→web`, `md/txt→note`, `pdf/docx/odt/rtf→None` — user decides); PATCH/CLI can override. It is a *selection* facet and the substrate for future routing — **not** a chunking/embedding lever (those are locked settings; per-type routing only after a measured eval win). |
| 5 | **`excluded` flag** (default false): an excluded file is skipped by *every implicit* ingest walk — full-dir runs from CLI or API — with an `excluded_skipped` count logged (inform-don't-block). An **explicit** selection (`--files` / `paths` naming the file) **overrides** exclusion, with a log line: a direct user action outranks a standing preference. |
| 6 | **`--dry-run` (CLI):** scan + print the per-file plan — `would_add` / `would_reembed` (changed) / `skip_unchanged` / `excluded` — and exit **without loading the embedding model or opening Chroma**. This is the "know before you burn an hour" affordance (the 2026-07-02 R1 re-ingest re-embedded 57/62 docs; a dry run would have said so up front). |
| 7 | **API:** `GET /api/sources` (scan + list, incl. derived status) · `PATCH /api/sources` (`{rel_path, excluded?, doc_type?}`) · `POST /api/ingest` gains an **optional** JSON body `{paths?: [rel_path, …]}` — absent body = all non-excluded supported files (today's behavior minus exclusions). The 409-if-running lock and status polling are unchanged. |
| 8 | **CLI:** `--files P [P …]` added next to `--path`; `--files` / `--path` / `--rebuild` mutually exclusive. Plain `python -m doc_assistant.ingest` keeps its exact current behavior except honoring `excluded` (Decision 5). |
| 9 | **UI (S2):** new `Sources.svelte` panel — file table with status chips + doc-type select + exclude toggle, checkbox selection, "select all new/changed", **Ingest selected (N)** posting `{paths}`; reuses the existing ingest-status polling; the empty-corpus banner links here. Enrichment runners (figures/tables/citations/keywords) are **not** on this panel — the VLM pass is paid API (KI-4) and must never ride along a click. |

**Edge cases (spec explicitly):**
- *Selected file vanishes mid-run* → per-file `error`, run continues (existing per-doc isolation).
- *Selected but unchanged* → the dedup gate skips it (no re-embed); reported `skipped`, not an error.
- *`paths` entry outside the source dir, unsupported suffix, or unknown* → API 400 / CLI error naming each offender; nothing partial starts.
- *Registry writes never touch Chroma or the markdown cache* — SQLite-only (guard-tested).
- *Concurrent ingest* → unchanged: API 409 lock; CLI unguarded as today.
- *Source-dir change via settings* → previous rows derive `missing`; next scan repopulates (ADR-1 consequence).

**Build-time confirmations:**
- The exact freshness predicate to reuse for `changed` (`ingest/cache.py:is_cache_fresh` — cached mtime ≥ original) and where cache paths resolve (`get_cache_path`).
- `app_settings.get_source_dir()` shape (str vs Path) for `rel_path` computation.
- **`create_all` creates new tables but does NOT add columns to existing tables** — `SourceFile` (new table) rides the additive `init_db()` migration like `Figure` did; that is exactly why v1 puts `doc_type` on `SourceFile` and adds **no column to `Document`** (a `Document.doc_type` copy is deferred until an ALTER-style migration helper exists).

---

## Contract — `src/doc_assistant/ingest/registry.py` (new)

Pure core + thin impure boundary (house split):

- `derive_status(file_exists: bool, cache_fresh: bool, has_document: bool) -> str` — **pure**, the
  truth table above; exhaustively unit-tested.
- `default_doc_type(suffix: str) -> str | None` — **pure** (Decision 4 mapping).
- `validate_selection(requested: list[str], known: set[str]) -> list[str]` — **pure**; rejects
  out-of-dir (`..`/absolute), unsupported suffixes, unknowns; returns normalized rel_paths or raises
  with all offenders listed.
- `scan_sources(session, source_dir: Path) -> list[SourceView]` — **impure**; walk + `stat`, upsert
  rows, refresh `last_seen`, compute derived status per row. No content reads.
- `set_source_meta(session, rel_path: str, *, excluded: bool | None = None, doc_type: str | None = None) -> SourceView` — **impure**; the PATCH/adapter seam.
- `resolve_selection(session, source_dir: Path, requested: list[str] | None) -> list[Path]` — **impure**
  orchestration: `None` → all supported non-excluded files; a list → `validate_selection` then absolute
  paths (explicit picks override `excluded`, logged).

**NOT responsible for:** extraction, hashing, chunking, any Chroma/cache access, vendor catalogs (PR 17),
URL/manifest concerns (`sources_manifest.py`).

## Contract — `src/doc_assistant/ingest/__init__.py` (modified)

- `main(force_rebuild=False, skip_cleanup=False, scope=None, files: list[Path] | None = None, dry_run: bool = False) -> dict[str, int]`.
- Mutual exclusion: `files` vs `scope` vs `force_rebuild` (ValueError, mirroring the existing
  rebuild/path rule at `ingest/__init__.py:350-352`).
- Cleanup runs only when neither `files` nor `scope` is set (extend the existing condition, line 375).
- File resolution: `files` given → use as-is (already validated); else walk as today, then subtract
  `registry.excluded` rel_paths (Decision 5) with a logged count.
- `dry_run=True` → resolve files, compute per-file plan from the dedup gate + cache freshness, log the
  table, return `{"would_add": …, "would_reembed": …, "skip_unchanged": …, "excluded": …}` — **without**
  constructing embeddings or Chroma handles (guard-tested via a monkeypatched `get_embeddings` that fails
  if called).

## Contract — `src/doc_assistant/ingest/__main__.py` (modified)

`--files` (nargs `+`) and `--dry-run` added; help text states the exclusivity rules and that explicit
`--files` overrides `excluded`.

## Contract — `src/doc_assistant/db/models.py` (SourceFile, additive)

`class SourceFile(Base)`: `id` (uuid PK) · `rel_path` (unique, indexed) · `format: str` ·
`size: int` · `mtime: float` · `doc_type: str | None` · `excluded: bool = False` ·
`first_seen` / `last_seen` (`_utcnow`). No FK to `Document` (Decision 3). Migration = the additive
`init_db()` `create_all`, same note as the `Figure` table (4b spec).

## Contract — `apps/api` (modified)

- `apps/api/models.py`: `IngestRequest {paths: list[str] | None = None}` ·
  `SourcePatch {rel_path: str, excluded: bool | None, doc_type: str | None}` ·
  `SourceView {rel_path, format, size, mtime, status, doc_type, excluded}`.
- `apps/api/main.py`: `GET /api/sources` → scan + list; `PATCH /api/sources` → `set_source_meta`
  (404 unknown rel_path, 422 bad doc_type); `POST /api/ingest` accepts the optional `IngestRequest`
  body → `registry.resolve_selection` → `ingest_fn(files=…)` (no body → `files=None` current path,
  which now honors exclusions). Errors from validation → 400 listing offenders.

## Contract — `apps/desktop` (S2 only)

`apps/desktop/src/lib/Sources.svelte` (new panel, Decision 9) · `api.ts` (`getSources`,
`patchSource`, `startIngest(paths?)`) · `types.ts` (SourceView). Thin renderer: zero logic beyond
selection state; all rules live server-side.

---

## Build node

**S1 depends on:** nothing unshipped (M4 flow + `scope=` seam are live). **S2 depends on:** S1.
**Feeds:** PR 17 (adapters write this registry — sequence S1 *before* 17). No GPU, no LLM, no paid calls.

**Files owned (S1):** `src/doc_assistant/ingest/registry.py` (new) · `ingest/__init__.py` ·
`ingest/__main__.py` · `db/models.py` (`SourceFile`) · `apps/api/models.py` · `apps/api/main.py` ·
`tests/unit/ingest/test_registry.py` (new) · `tests/integration/ingest/test_selective_ingest.py` (new) ·
`tests/integration/api/…` (extend) · docs (`ROADMAP.md` rows, `architecture.md` ingest section,
`DEVLOG.md` per change).
**Files owned (S2):** `apps/desktop/src/lib/Sources.svelte` (new) · `api.ts` · `types.ts` · `App.svelte`
(panel routing) · frontend tests as per M3 conventions.

### Unit tests (S1)
- `derive_status`: full truth table (8 combos → 4 statuses).
- `default_doc_type`: every supported suffix + unknown.
- `validate_selection`: absolute path rejected · `..` traversal rejected · unsupported suffix rejected ·
  unknown rel_path rejected (all offenders listed in one error) · happy path normalizes separators.

### Integration tests (S1, CI gate — tmp dirs, no corpus, no network)
- Scan lifecycle: new file → row `new`; ingest → `ingested`; touch source → `changed`; delete → `missing`;
  `last_seen`/`first_seen` semantics.
- `main(files=[a])` ingests exactly `a`; **guard:** document `b`'s chunks + `Document` row untouched
  (the "selected ≠ everything else deleted" invariant); no cleanup ran.
- Second `main(files=[a])` with unchanged `a` → `skipped` via the dedup gate.
- `dry_run` writes nothing (SQLite row counts + Chroma dirs byte-identical) and never constructs
  embeddings (monkeypatch trap).
- Full run skips an `excluded` file with the logged count; explicit `files=[excluded]` ingests it.
- API flow: `GET /api/sources` → `PATCH` excluded/doc_type persists → `POST /api/ingest {paths}` →
  status polling reaches `done` with the right counts; bad paths → 400.

## Definition of done

- **S1:** batch (`--files`), scoped (`--path`, unchanged), dry-run, and exclusion all work from the CLI on
  the real corpus; `POST /api/ingest {paths}` ingests exactly the selection; with an empty registry and no
  body, behavior is byte-identical to today (no regression for existing users); gates green
  (ruff / `mypy --strict` / bandit / coverage floor); one DEVLOG entry per logical change.
- **S2:** the Sources panel lists status-chipped files, selection → ingest-selected round-trips through the
  API with live progress; exclude + doc-type edits persist; no business logic in the frontend.

## Decision ledger (grill-me 2026-07-15)

| Branch | Resolution | Deciding reason / reopens-if |
|---|---|---|
| `doc_type` behavior (v1) | **Deferred** — no seeding, no PATCH field, no UI, read nowhere | All-PDF corpus → manual busywork with no consumer (not a chunk/embed lever). **Reopens if** a 2nd format enters or a per-type routing eval wins |
| `doc_type` schema | **Ship the nullable column now, dormant** | `create_all` can't ALTER a column onto an existing table + no ALTER-helper exists → dormant column makes the future return a behavior-only add. Cost: one dead nullable column |
| Registry `SourceFile` vs stateless listing | **Keep the table (minimal: identity + `excluded` + dormant `doc_type`)** | Persistent `excluded` has nowhere else to live (no `Document` row pre-ingest); also the status-listing source + PR-17 adapter seam. Stateless can't persist exclude |
| Selection primitive (v1) | **status + `excluded` + explicit `paths`** | doc_type deferral removes the metadata-selection axis; incremental (`new`/`changed`) + exclude is the real need for a flat personal corpus |
| `PATCH /api/sources` shape | **`excluded` only**; `SourceView` still reports `doc_type` (always null) | Drop the unused field but keep the wire shape forward-stable |
| S1 PR boundary | **One PR: registry + CLI + API**; S2 (UI) separate | Project one-increment-per-PR norm; S1 shrank with doc_type gone but still coheres as backend |
| S2 UI shape | **Parked** — dedicated `Sources` sidebar mode vs fold into Settings | Doesn't block S1 (API is UI-agnostic); decide at S2 kickoff with the flat-corpus file-count in view |
| `dry-run`, exclude-override, status truth table, vendor-neutral registry, rename-loss | **Kept as drafted** | No trade-off surfaced; the flat-source fact doesn't touch them |

**Routing:** resolutions live in this spec (the design-lock, which already embeds ADR-1/2/3) + the amendment
block up top; the `SESSION.md` handoff carries the ledger. **No new standalone ADR** — the first
browse-time *write* path (manual folder/tag editing, deferred) remains the ADR trigger, unchanged from L4.

## Out of scope (deferred, with owners)

- **Zotero/Calibre adapters** — PR 17 (they populate this registry; ADR-3 is the seam).
- **`Document.doc_type` column** — needs an ALTER-migration helper first (build-time confirmation above).
- **Rename detection / content-hash identity in the registry** — the manifest owns content pins; revisit
  only if rename-metadata-loss hurts in practice.
- **File watcher / auto-scan** — scan-on-read is enough for a personal corpus.
- **Per-doc-type chunking or embedder routing** — locked settings / deferred Feature 1b; requires an
  eval-harness win, not a spec line.
- **Enrichment chaining from the UI** (figures/tables/citations/keywords after ingest) — separate,
  opt-in surface later; the VLM pass is paid (KI-4).
- **Parallel ingest** — orthogonal performance work; nothing here blocks it.
- **Pre-ingest Tag/Folder assignment** — post-ingest tags/folders already exist; `doc_type` covers the
  selection need in v1.
