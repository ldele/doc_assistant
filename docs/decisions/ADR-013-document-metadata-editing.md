<!-- status: active · updated: 2026-07-16 · class: append-only -->

# ADR-013 — Document metadata editing (first browse-time write path)

- **Status:** accepted
- **Date:** 2026-07-16
- **Deciders:** user + Claude Code

## Context

The library shows auto-extracted `title`/`authors`/`year` (the metadata-enrichment pass). The
extraction is ~97% good but wrong on a few docs (a Springer licence line as a title) and blank on
~19 authors, and the user had no way to correct it. The user asked for an **edit-metadata**
affordance per document, mirroring the chat management UX (hover → `⋯` → Edit / Reveal), where the
auto-extracted values are the *default* and user edits *override* them reversibly, plus a
**Reveal in file explorer** action.

This is the **first browse-time write path** — every prior feature was read-only over the ingested
store, or a derived-sidecar enrichment run from the CLI. The library-redesign spec and several DEVLOG
entries flagged "manual metadata/tag editing" as the change that needs an ADR, because it decides how
user writes coexist with the extraction-populated registry and re-runnable enrichment. Two forces:
(1) a user edit must survive a re-run of `enrich_metadata` (which repopulates `Document.title` etc.);
(2) "reset to default" must be possible, so the auto value can't simply be overwritten in place.

## Options

**A. Where the override lives**
1. **Override columns on `documents`** (`title_override`, …) — no join, but four additive-migration
   entries and mixes user writes into the extraction-populated registry row (a re-run of enrichment
   writes the same row the user's data sits on).
2. **A `DocumentMeta` sidecar table** keyed by `document_id`, holding only the overrides — mirrors the
   `ConversationMeta` sidecar (the chat feature's pattern), a new table so `create_all` makes it with
   no migration, and cleanly separates user writes from the registry. Requires one join/batch-load.

**B. Reveal-in-explorer mechanism**
1. **Tauri command** + a `shell:allow-open` capability — idiomatic for the packaged app, but would be
   the app's *first* Tauri command (the frontend is otherwise 100% API-driven), not exercisable in the
   browser-preview harness, and diverges from the established pattern.
2. **A backend endpoint** that shells out on the local host (`explorer /select` / `open -R` /
   `xdg-open`, list-form args, no shell) — consistent with the API-only frontend, testable in dev, and
   the API always runs on the user's machine (local-first), so it can reach the desktop.

## Decision

**A2 — the `DocumentMeta` sidecar.** Auto-extracted values stay in `Document.title/authors/year` (the
default); `DocumentMeta.{title,authors,year}_override` hold user edits. The **effective** value shown
in the library is `override ?? default`; a document is `customized` when any override is set. "Reset
to default" deletes the row. `set_document_meta` dedups each field against its auto default, so
re-saving an untouched field creates no override, and a re-run of `enrich_metadata` (which touches
only `Document`) can never clobber a user edit. `PATCH` replaces the whole small override set (the
editor sends the full form).

**B2 — the backend reveal endpoint** `POST /api/library/documents/{id}/reveal`, resolving the path
server-side from `Document.source_original` (with the `DOCS_PATH/filename` fallback the extract-*
scripts use) and 404-ing when the file has moved.

## Consequences

- **Easy:** correcting bad titles/authors/years; the override model directly supports the user's future
  "choose which metadata columns the library shows" idea; enrichment stays idempotent and safe to re-run.
- **Committed to:** a write surface in `library.py` (`set_document_meta`/`clear_document_meta`) and the
  three routes; the reveal action opens a real OS window on whatever host runs the API (always the
  user's machine). The reveal is not exercisable in the browser preview (verified via the endpoint +
  a mocked reveal in tests; the live OS window is a manual check).
- **Hard / deferred:** editing DOI/notes/tags/folders, per-field (vs whole-doc) reset, and bulk edit are
  out of scope; a Tauri-native reveal remains an option only if the API-shell approach proves
  inadequate in the packaged app.
