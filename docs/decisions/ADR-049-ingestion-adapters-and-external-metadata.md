<!-- status: active · updated: 2026-08-31 · class: append-only -->

# ADR-049 — An ingestion adapter is a route to paths, and a catalogue's metadata is a third slot

**Status:** accepted (2026-08-31) · **Supersedes:** nothing · **Amends:** `feature-selective-ingestion.md`
ADR-3 (the adapter seam, now built) · **Relates to:** ADR-013 (user overrides), ADR-046 (copy vs
reference), ADR-014 as amended (delete), KI-54 (the title picker) · **Roadmap:** 17

## Context

Row 17 has said the same thing since the selective-ingestion spec: *"Zotero (Calibre TBD) — optional
producers for the S1 source registry, never a dependency."* The hard constraint behind it is the
user's, from 2026-07-02: **our SQLite is the system of record; nothing may depend on an external
catalogue.** The spec's own ADR-3 named the seam ("adapters are optional second producers") and
stopped there, and the roadmap added one guard: *"Don't let Zotero/Calibre adapters leak vendor
specifics past the extractor boundary."*

Two things had to be decided before any of that could be built, and neither was in the spec.

**1. What an import actually *is*.** Adding documents already has a considered flow — a review sheet
that says what would happen to each file, a duplicate rule, a copy-or-reference choice (ADR-046),
an add, an index. An import could reuse it or replace it.

**2. Where a catalogue's metadata goes.** This is the half worth having. A reference manager's
title, authors and year are *curated by a person*; `metadata_extractor` infers them from a PDF's
first page and sometimes picks the journal name instead (KI-54, open). Discarding a curated answer
in favour of a guess would be perverse. But there was nowhere to put it: `Document.title` is the
extractor's slot and every metadata pass overwrites it, and `DocumentMeta.*_override` is the user's
own in-app edit (ADR-013), which an import must never silently replace.

## Decision

### D1 — An adapter returns paths. It does not add anything.

`POST /api/catalogue/zotero/scan` reads the library and hands back absolute paths. The client stages
them, and the **existing** review sheet takes over: same verdicts, same duplicate rule, same
copy-or-reference choice, same progress. Zotero gets no second add path.

The consequence that makes this obviously right: importing a library you have already imported is
simply an add where everything reads as a duplicate — and it was, first try, with no code written
for it.

### D2 — The vendor boundary is a package split, not a promise.

- `adapters/catalogue.py` — **neutral.** `ExternalDocument` (a path plus the fields any reference
  manager has), and the reading and writing of `ExternalMetadata`. Everything downstream talks only
  to this half.
- `adapters/zotero.py` — **vendor.** The only module that knows what a `linkMode` is.

Adding Calibre means one module beside `zotero` and one route. It means changing nothing in
`catalogue`, `registry`, or the client.

### D3 — A catalogue's answer is a third metadata slot, between the extractor and the user.

A new `ExternalMetadata` table, keyed `(source, path_key)` — **by path, not by document**, because
the metadata arrives at import, before the file has been extracted, and may never lead to a
`Document` at all. The ordering it establishes:

| Slot | Written by | Beats |
|---|---|---|
| `Document.title/authors/year/doi` | the extractor | nothing |
| `ExternalMetadata.*` | an import | the extractor |
| `DocumentMeta.*_override` | the user, here | everything (ADR-013, unchanged) |

`ingest.main` applies it as a post-loop pass beside `_assign_demo_folder`, so a document gets its
curated metadata the moment it exists. And `reingest._rerun_metadata` **re-applies the catalogue's
answer** instead of extracting: without that, the cheapest, safest-looking box in the re-run dialog
would replace a curated title with a guess at it, and would reintroduce KI-54 on exactly the
documents that were immune to it.

### D4 — An import names its own root; a root that already contains a file wins.

`apply_add` takes an optional `reference_root`. Reference-adding registers a root for a file's
*parent directory* — right for a dropped folder, and catastrophic for Zotero, which gives **every
attachment its own `storage/<key>/` directory**: one `SourceRoot` per document, five hundred rows
for one library, each stat-ed on every scan. The pre-existing docstring even cited "a twenty-paper
Zotero folder" as the case per-parent solved; Zotero's actual layout defeats it.

Two changes, both narrow: an adapter reports the catalogue's storage folder and the client passes it
through as the batch's root; and `_reference_target` now prefers an **already-registered root above
the file**, deepest first. The second is not the guess the per-parent rule refuses to make — an
ancestor root exists only because the user, or an import acting for them, established it — and it
improves the ordinary case too.

### D5 — The adapter reads a **copy** of the catalogue, and declines out loud.

Zotero holds `zotero.sqlite` open. The file is copied to a temp path (with its `-wal`/`-shm`
companions, or recent edits would be invisible) and the copy opened read-only, so the user's own
library is never at risk from a feature they can live without.

Everything the adapter declines is **counted under a reason a person can read** — `412 a web-page
snapshot · 88 not downloaded to this computer` — never summed. A scan reporting "37 found" out of a
500-item library looks like a broken import; the reasons look like a working filter and tell the
user what to change. Web snapshots are excluded by default and can be asked for.

## Consequences

- **A scan writes one thing: the metadata record.** It registers no root, stages no file, adds no
  document (guard-tested). The records are keyed by path, so a cancelled import leaves rows
  describing files the library does not have — inert, and corrected by the next import.
- **`adapters/` is a new top-level package** in the ADR-023 layout. Justified by D2: the boundary is
  the feature.
- **The Zotero schema is not verified against a real library.** There is no Zotero on the machine
  this was built on. The tests construct a database to the documented Zotero 5/6/7 schema, so they
  prove the *mapping*, not the schema. Every query is defensive: a missing table produces a sentence
  rather than a stack trace, and an optional part that will not read (collections, creators) costs
  its enrichment rather than the import. **First contact with a real library is the thing to check**,
  and the fixture is what to correct if it disagrees.
- **Linked attachments under Zotero's "Linked Attachment Base Directory" are skipped**, counted
  under their own reason. That base is a *preference*, not a database value, so it cannot be
  discovered; `read_library` accepts it from the caller, and no UI offers it yet.
- Collections and item types are recorded and not yet acted on. They are the substrate for the
  dormant `SourceFile.doc_type` and for folders — stored now so activating either needs no second
  import.

## Rejected

- **Writing the catalogue's metadata into `DocumentMeta`.** It is the *user's* slot (ADR-013). An
  import would silently overwrite corrections they had typed here, which is the one direction that
  loses work rather than improving it.
- **Writing it into `Document.*` with no record of where it came from.** The next metadata re-run
  destroys it, and the library cannot tell a curated title from a guessed one.
- **Depending on Zotero's SQLite as a live source.** Explicitly forbidden (user, 2026-07-02), and
  it would make every document's metadata hostage to a program being installed.
- **A separate Zotero add/index path.** It would need its own duplicate rule and its own placement
  question, and the two would drift. Reusing the review sheet is why this ADR is short.
- **Registering the catalogue's root during the scan.** It would make merely *looking* at a library
  create state the user never confirmed.
