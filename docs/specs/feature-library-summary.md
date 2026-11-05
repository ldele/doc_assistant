<!-- status: built · updated: 2026-11-05 (built the same day, then revised to the user's seven recommendations; § As built lists where it departs from the text below) · class: living -->

# Spec — Library summary: what went wrong at indexing, and what is in the documents (ROADMAP 112)

> The executor brief. Target executor: **Claude Code**. **Buildable:** ROADMAP 112's gate was the
> user's markup of [`docs/wireframes/library-summary.md`](../wireframes/library-summary.md), and
> they approved it as drawn on 2026-11-05 with no markup, so its four choices stand as recommended
> and the wire type below is the one to build. No ADR owns the *why*: the request is the user's
> (2026-11-05). This spec reads, and must not
> re-litigate, ADR-013 (effective metadata), ADR-049 (catalogue metadata), ADR-056 (the
> letter-spacing record) and ADR-063 (button roles).

## As built (2026-11-05)

Built in the session that wrote this spec, after the user approved the wireframe. Where the build
departs from the text below, this list wins:

- **Wire files follow the router, not the domain:** `apps/api/models/summary.py`,
  `apps/desktop/src/lib/core/types/summary.ts` and `core/api/summary.ts` (the `apps/api` rule: one
  model module per router module, as `spacing` does), not `models/library.py` / `types/library.ts`.
  The dataclasses live in `library/summary.py` itself, as `library/spacing.py`'s do.
- **`folders` is one shape** with three nullable counts (`in_a_folder`, `in_none`,
  `also_elsewhere`) rather than a union, so the TypeScript side needs no narrowing.
- **The issue predicates run in Python** over one fact record per document (`_Facts`), not as SQL;
  `list_documents(issue=…)` filters its query result by the ids they select. One pass over the
  collection plus one grouped query per related table; measured below.
- **The screen:** *Summary* is a third button in the pane's layout toggle, shown for *All documents*
  and a folder, transient (Grid and List persist, Summary does not). *Show the N* opens a new
  collection kind, `issue`, carrying the ids the filter returned, so the grid shows exactly N.
- **Measured at 10,000 documents** (an invented library, `docs/performance.md` § 3): the route
  0.68–0.69 s warm and 1,416 bytes; *Show the N* 0.76–0.98 s. Letter-spacing costs 0.29 s of the
  0.66 s in process and stays in the one route.
- **Revision 2, the user's seven recommendations (2026-11-05, "Send all"),** six applied, the
  wireframe redrawn to match (`docs/wireframes/library-summary.md`, rev 1001):
  1. A count is navigation: every count row is the click target (row, hover, chevron), a zero row
     is plain text, and the screen's only buttons are *Count again* (quiet) and *Add documents*.
     *Show the N* below means "the row that opens the N".
  2. *Need attention* is the first tile and scrolls to *From indexing*; passages and pages are a
     context line under the title.
  3. *Metadata* and *Found in each document* rows show "N of M" with a thin bar; nothing of the kind
     at 0 documents.
  4. "Not checked" apart from "none found": the wire gains `metadata.with_any_field` (documents with
     at least one of title, authors, year, DOI). At 0 with documents present, the card says *Not
     read for M documents* once; references, figures and keywords say *not looked for* while none
     is on record anywhere.
  5. *From indexing* rows name where to fix them: files that failed in this session's run open the
     indexing screen (`openIndexing`), other files Settings › Documents, document issues the grid.
     It opens the fix, never runs it.
  6. Bars per the dataviz guide: horizontal, one hue, value at the tip, *Types* ranked; *Years* keep
     decade order (a time scale; "sorted by count" was not applied there), *No year* a bar apart.
  7. Keeping each count run for a trend: not built, later work.
- **A known limit:** "present at the folder's last look" is a registry row whose `last_seen` equals
  its root's newest. A scan that finds the folder empty stamps nothing, so after every file has
  left, the previous look's rows still count until a file returns. A per-root scan time would fix it;
  not built.

## Goal

One screen per collection, the whole library or one folder, that says what went wrong when the
documents were indexed and what is known about them: documents that gave no text, extractions that
were flagged, files that never made it in, missing metadata, references, figures, letter-spacing and
keywords on record, types and years. Every number says what it is a count *of*, including when the
honest answer is "on record", because the app never looked (§ *Signals*). It is the screen KI-87's
library needed: an index of a folder that left every document with no title, year or author looked
fine everywhere else.

## Hard constraints

- **Read-only.** The route writes nothing: no registry scan (`GET /api/sources` writes registry rows,
  so the summary does not call it), no cache write, no database write. Checked by a test.
- **Logic in `src/doc_assistant/library/summary.py`** (new). `apps/api` maps it to HTTP and adds the
  one thing only it holds, this session's run (`app.state.ingest_run`); `apps/desktop` renders.
  Non-negotiable #3.
- **One definition per count.** Each *issue* (`no_text`, `missing_year`, …) is one predicate in one
  table in `summary.py`, read both by the counts and by the document list's `issue` filter, so a
  count and the documents behind *Show the N* cannot disagree.
- **Robustness contract** (`.claude/CONTEXT.md`): correct at **0 documents**, cheap at **~10,000**.
  The wire carries counts, never ratios, and its size does not grow with the library (types and
  decades are bounded). No threshold in this feature; health keeps its own (`health.py`).
- **Archived documents count nowhere**, as in `library.pins.library_summary`.
- **ADR-063:** a count row that opens documents is navigation (the row is the target, revision 2); *Count again* is quiet; the screen has no primary
  and no danger action. GLOSSARY U-001: the view is called **Summary** on every door to it.
- **Windows encoding:** the letter-spacing record is read through `ingest.cache.read_spacing_record`
  (UTF-8); any new read passes `encoding="utf-8"`.
- **No real library in the repo:** tests and the wireframe use invented documents and counts.

## Signals — what exists, where, and how far to trust it

The inventory the brief asked for, read in the code on 2026-11-05. *Indexing writes it* means the
app's own index run or *Add documents* (`ingest.main`, called by `apps/api/services.py:274`). The
trust column cites `docs/knowledge-layer.md` where it covers the signal; that page covers the
knowledge layer only, so the ingest signals cite the code's own record and the known issues.

| Signal | Stored / computed at | Indexing writes it? | Trust | In the summary as |
|---|---|---|---|---|
| Documents, passages, pages | `Document.chunk_count`, `page_count` (`db/models.py:167-168`), by `upsert_document_in_sqlite` (`ingest/store.py:205`) | yes | ✅ counts. `page_count` is null for a format without pages | the three tiles; pages say *of N paged documents* when some have none |
| Type | `Document.format`, the file suffix (`db/models.py:155`; `ingest/__init__.py:399`) | yes | ✅. It is the file type, not the publication type: `ExternalMetadata.item_type` (`db/models.py:1100`) is a catalogue's, and `SourceFile.doc_type` is dormant, never written (`db/models.py:1253`) | *Types* |
| No text | `chunk_count = 0`, recorded `broken` (`ingest/__init__.py:385-389`) | yes | ✅ known for every document. KI-47 (a scan needs Tesseract) is the usual cause; KI-46 (the orphan sweep cannot see a zero-chunk row) can leave a duplicate that this counts | `no_text` |
| Extraction health | `Document.extraction_health` (`db/models.py:166`), from `classify_document_health` (`health.py:52`; bands at `health.py:197-201`), also `IngestionEvent.health_status` (`db/models.py:1037`) | yes | ⚠️ "a useful rough sort, not a precise judgment" (`health.py:9`). Since KI-53 `broken` means text missing relative to its container, not a small document. **The reasons are logged, never stored** (`ingest/__init__.py:407-410`) | `flagged` = marginal or broken with text; `null` health counted apart as *not classified* |
| Metadata: title, authors, year | `Document.title/authors/year` (`db/models.py:158-160`), effective = the user's override ?? the default (`library/documents.py:30`, `DocumentMeta` at `db/models.py:1067`) | **no** — only a catalogue import fills them at index time (`ingest/__init__.py:784`, ADR-049). Otherwise *Re-run… › Metadata* (`reingest.py:236`) or `scripts/enrich_metadata.py` (`metadata_enrich.py`) | ✅ as presence; ⚠️ as correctness: KI-54, the title picker can choose a journal name (1 document in 16 when measured). *Missing* is exact, *present* is not *right* | `missing_title`, `missing_authors`, `missing_year`; a year outside `plausible_year` (`library/citations.py:144`) counted as *implausible*, not as a decade |
| Metadata: DOI | `Document.doi` (`db/models.py:161`); no override exists | **no**, as above | ✅ as presence | `missing_doi` |
| References | `Citation` rows by `source_document_id` (`db/models.py:250-256`, **no index**) | **no** — *Re-run… › References* (`reingest.py`, `PARTS`) or `scripts/extract_citations.py` | ⚠️ **on record only**: a document whose references were parsed to nothing has no rows, exactly like one never parsed. Links into the library are KI-45's (frozen at extraction, part 2 open) and are not counted | `no_references`, with the total rows |
| Figures | `Figure` rows (`db/models.py:320`, `document_id` indexed at `:343-345`); described = `vlm_description` (`:363`) | **no** — *Re-run… › Figures* (`reingest.py:317`) or `scripts/extract_figures.py`; PDF only | ⚠️ on record only, as references | `no_figures` among PDFs, with totals and described |
| Letter-spacing | `<cache>.spacing.json` beside each cached text (`ingest/cache.py:444`, read at `:451`, written at `:551-556`; an empty record deletes the file); shown per document by `library/spacing.py:100` | yes, since ADR-056, for PDFs | ✅ where a record exists. **No file means none found, or text read before the record existed** (`ingest/cache.py:452`) | `letter_spaced` = a record with a run that reached the text |
| Keywords | `document_keywords` (`db/models.py:62-70`), by `extract_keywords` (`knowledge/keywords.py:1042`) | **no** — the CLI pass | ✅ as presence; what a keyword is worth is `docs/knowledge-layer.md` § 2 | `no_keywords` |
| Files not in the library | registry `SourceFile` rows (`db/models.py:1199`), status `new` = file present, no document (`ingest/registry.py:45`); rows refreshed, `last_seen` stamped, by `scan_root` (`registry.py:331`, `:387`) whenever Settings › Documents lists files or an index runs; `excluded` is the user's | refreshed by it | ✅ **as of the last look at the folder**; *never indexed* and *failed* are one state here | `files_not_in_library`, `files_excluded`, `files_seen_at`; library scope only |
| A failure's reason | `IngestRun` (`ingest/run.py:124`; the reason at `run.py:95-115`), held on `app.state.ingest_run` (`apps/api/routers/sources.py:76-81`) | yes, in memory | ✅ for this session's last run; **gone when the app closes**. `IngestionEvent.error_message` (`db/models.py:1038`) is never written | `last_run_failed`, `null` when no run this session |
| Folders | `document_folders` (`db/models.py:44-48`); folders are flat (`library/folders.py:25`), a document can be in several | n/a | ✅ | *in a folder / in none*; in a folder's scope, *also in another folder* |
| Added | `Document.added_at` (`db/models.py:172`) | yes | ✅ | first and last added |
| Concept-graph reach | skeleton coverage, `docs/knowledge-layer.md` § 6 (✅ "an honest count") | no | ✅ | **not in v1** (wireframe decision 4) |

**What the inventory changes about the request.** Three of the user's examples (missing metadata,
no references, no figures) are signals the app's own indexing never produces. A library indexed only
in the app reads 100% missing on all of them, which is a true statement about what is on record and a
misleading one about the documents. The wireframe states it in its footnotes; the fix is two
follow-ups outside this feature (§ *Opens*).

**Already in the tree:** `library.pins.library_summary()` (`library/pins.py:181`, `LibrarySummary`
at `library/models.py:79`) counts documents, chunks, health and format for the CLI and
`query_router`. It stays as is; `summarise_collection` computes a superset, and a later cleanup may
make the old one read from the new one.

## The route

`GET /api/library/summary` · `GET /api/library/summary?folder_id=<id>`

- New router module `apps/api/routers/library/summary.py`, composed in
  `apps/api/routers/library/__init__.py` (its path is disjoint from the other four).
- Calls `doc_assistant.library.summarise_collection(folder_id)`, then sets `last_run_failed` from
  `app.state.ingest_run` (`IngestRun.counts()["failed"]`, or `null` with no run). Nothing else.
- `404` for an unknown `folder_id`. Otherwise `200`, including at 0 documents.
- **One change to an existing route:** `GET /api/library/documents` gains optional `issue` and
  `folder_id` query parameters, passed to `list_documents(issue=…, folder_id=…)`. No parameter: the
  response is unchanged. An unknown `issue`: `400` naming the known ones. This is what *Show the N*
  opens.

### Wire type

`src/doc_assistant/library/models.py` gains the dataclasses; `apps/api/models/library.py` the
payloads (`CollectionSummaryPayload` and its parts, `from_summary` as `LibraryDocumentPayload`
does); `apps/desktop/src/lib/core/types/library.ts` the same names. Every count is an `int ≥ 0`.

```text
CollectionSummary
  scope:            "library" | "folder"
  folder_id:        str | null
  folder_name:      str | null
  counted_at:       datetime, UTC
  documents:        int
  passages:         int                 sum of chunk_count
  pages:            int                 sum of page_count
  paged_documents:  int                 documents with a page_count
  need_attention:   int                 no_text + flagged + (files_not_in_library or 0)
  indexing:
    no_text:              int
    flagged_marginal:     int           with text, health marginal
    flagged_broken:       int           with text, health broken
    not_classified:       int           health null (indexed before health existed)
    files_not_in_library: int | null    null in a folder's scope
    files_excluded:       int | null    null in a folder's scope
    files_seen_at:        datetime | null   the folder's last look; null: never looked
    last_run_failed:      int | null    set by the route; null: no run this session
  metadata:
    missing_title, missing_authors, missing_year, missing_doi: int
    implausible_year:     int           a year outside plausible_year, not in any decade
    set_by_you:           int           documents with a metadata override
    with_any_field:       int           documents with any of title, authors, year, DOI (revision 2)
    from_a_catalogue:     int           documents with an ExternalMetadata row
  found:
    references:   { on_record: int, none_on_record: int, rows: int }
    figures:      { on_record: int, none_on_record: int, rows: int, described: int, pdfs: int }
    letter_spacing: { recorded: int, pdfs: int }
    keywords:     { on_record: int, none_on_record: int }
  types:            [ { format: str, documents: int } ]   most first, then by name
  years:            { by_decade: [ { decade: int, documents: int } ], no_year: int,
                      oldest: int | null, newest: int | null }
  folders:          { in_a_folder: int, in_none: int } | { also_elsewhere: int }
  added:            { first: datetime | null, last: datetime | null }
```

Issue ids, the `issue` filter's values, defined once in `summary.py`: `no_text`, `flagged`,
`missing_title`, `missing_authors`, `missing_year`, `missing_doi`, `no_references`, `no_figures`,
`no_keywords`, `letter_spaced`.

### How every number behaves at 0 documents and at ~10,000

| Field | At 0 documents | At ~10,000 documents |
|---|---|---|
| `documents`, `passages`, `pages`, `paged_documents` | 0 each; the screen shows 0, never a percentage | one aggregate query |
| `need_attention` | equals `files_not_in_library` (0 or more) | arithmetic on the counts |
| `indexing.*` document counts | 0 | one `GROUP BY extraction_health` pass with a `chunk_count` condition |
| `files_not_in_library`, `files_excluded` | counted as usual: the one section that can be non-zero (`[s3]`) | one read of the registry rows and one set of document path keys (`registry.pathkey`), O(files + documents) in memory, ~10,000 short strings; **no directory walk**. A row counts when its `last_seen` equals its root's newest `last_seen` (present at the last look) |
| `files_seen_at` | `null` when the folder was never looked at | one `MAX` |
| `last_run_failed` | `null`, or the run's failures if a run this session failed every file | read from memory |
| `metadata.missing_*` | 0 | one `LEFT JOIN document_meta` pass; a blank string counts as missing |
| `found.references` | 0, 0, 0 | **one `GROUP BY source_document_id` pass over `citations`**, never a per-document `EXISTS`: the column has no index, and an `EXISTS` per document would read the table once per document |
| `found.figures` | 0s | one `GROUP BY document_id` (indexed) |
| `found.keywords` | 0s | one `GROUP BY document_id` on the composite key |
| `found.letter_spacing` | 0, 0 | **the one per-document cost:** a `stat` per PDF, and a parse only where a record exists. Budget below; if it is missed, this count moves behind its own request and the screen shows it when it arrives |
| `types` | `[]` | bounded by the number of formats (6 today) |
| `years.by_decade` | `[]`, `oldest`/`newest` `null` | bounded by the decades in the plausible range (`MIN_PLAUSIBLE_YEAR`, 1800, `library/citations.py:141`, to next year): 23 entries at most today |
| `folders` | `{0, 0}`; in an empty folder `{also_elsewhere: 0}` | one `GROUP BY` on `document_folders` |
| `added` | `null`, `null` | one `MIN`/`MAX` |
| **response size** | ~1 KB | the same order: no field grows with the library |
| **time** | — | **≤ 1 s warm** for the whole route at 10,000 documents on the development machine, measured (work item 6) |

The `issue` filter returns the documents themselves, so its response grows with the bucket, exactly
as the unfiltered list does today; at 10,000 documents a bucket can be large. The grid already
renders the whole library from one list call, so this adds no new scale problem; paging that list is
its own row if it is ever needed.

## Execution checklist

1. ~~Settle the wireframe's four choices with the user.~~ Done 2026-11-05: approved as drawn.
2. Write the test cases below.
3. Work items 1–6 in order; the tests of the code touched after each (`uv run --no-sync pytest
   tests/unit/test_library_summary.py`, then the API test, then `npm test`).
4. Fast checks before the full gate: ruff, `uv run --no-sync mypy src`, `npm run check`, the docs
   gate. Then the live preview: light, dark, 375 px, 0 console errors, against a scratch copy of a
   library (`DOC_DATA_DIR`), never the library the installed app has open.

## Work items

| # | Item | Files | Acceptance |
|---|------|-------|------------|
| 1 | `summarise_collection(folder_id)` and the issue table | `src/doc_assistant/library/summary.py` (new), `library/models.py`, `library/__init__.py` | the unit tests pass; no write to the database or the disk |
| 2 | `list_documents(issue=…)` reads the same predicates | `src/doc_assistant/library/documents.py` | each count equals the length of its filtered list on the fixture library |
| 3 | The route and the `issue`/`folder_id` parameters | `apps/api/routers/library/summary.py` (new), `routers/library/__init__.py`, `routers/library/documents.py`, `apps/api/models/library.py` | the API tests pass; no SQL in `apps/` |
| 4 | Types | `apps/desktop/src/lib/core/types/library.ts`, `core/api` | `npm run check` clean |
| 5 | The *Summary* layout, the issue collection, *Show the N* | `apps/desktop/src/lib/library/LibrarySummary.svelte` (new), `library/summary.ts` (new, pure, tested), `LibraryPane.svelte` (a third value of the layout preference beside `grid` and `list`), the collection label | the wireframe's three screens, driven in the preview; reading order as the wireframe doc states it, not as the canvas reading numbers it (T-006) |
| 6 | Measure at 10,000 | a synthetic library: the real schema, 10,000 generated documents, references, figures, spacing records, at an ASCII scratch path through `DOC_DATA_DIR` | the route's warm time and response size recorded in `docs/performance.md`; ≤ 1 s, or the letter-spacing count moves behind its own request |

## Test cases (write these first)

| Name | Assertion | Path |
|------|-----------|------|
| `test_empty_library_reads_zero` | 0 documents: every count 0, `types` and `by_decade` empty, `oldest`/`newest`/`added` null, `need_attention == files_not_in_library`, no exception | `tests/unit/test_library_summary.py` |
| `test_each_count_matches_its_filter` | for every issue id, the count equals `len(list_documents(issue=id))`, on a fixture with at least one document per issue and one with none | same |
| `test_override_counts_as_present` | a null title with a `DocumentMeta` title override is not missing; a blank string is | same |
| `test_no_text_is_not_also_flagged` | a zero-chunk `broken` document counts in `no_text` only | same |
| `test_archived_counts_nowhere` | an archived document moves no count | same |
| `test_folder_scope` | only the folder's documents count; `files_not_in_library` and `files_excluded` are null; `also_elsewhere` counts a document in two folders | same |
| `test_files_not_in_library_from_last_look` | a registry row with no document counts; an excluded one counts apart; a row not seen at the root's last look does not count | same |
| `test_letter_spacing_reads_the_record` | a PDF with a record holding a run in the text counts; a record with no run in the text and a missing record do not; a non-PDF is never read | same |
| `test_implausible_year_is_apart` | a year of 0 or of 3000 lands in `implausible_year`, in no decade | same |
| `test_wire_size_does_not_grow` | 10 and 1,000 generated documents give responses within a few hundred bytes of each other | same |
| `test_summary_writes_nothing` | the database file's bytes and the registry rows are unchanged after a summary | same |
| `test_summary_route` | `200` at 0 documents; `404` for an unknown folder; `last_run_failed` null without a run and equal to a fake run's failures with one | `tests/integration/test_api_library_summary.py` |
| `test_documents_issue_filter` | `?issue=missing_year` returns exactly those documents; no parameter returns today's list; an unknown issue is `400` | same |
| `summary.test.ts` | a zero count reads as a sentence; no percentage is formed when `documents` is 0; decade labels; *Need attention* is the server's number, not re-summed | `apps/desktop/src/lib/library/summary.test.ts` |

## Estimate

- **estimate:** M / one session for the backend and the route, one for the screen. *(Planner
  judgment, not a measurement.)*

## Out of scope

- **Indexing that reads metadata**, and a per-part *ran* record for *Re-run…* (so *none found* and
  *never looked* split). Both are follow-ups this screen exposes; see *Opens*.
- Storing why an extraction was flagged (`HealthReport.reasons` is logged only).
- Documents whose file is gone from disk (a `stat` per document), and the concept graph's reach
  (wireframe decision 4).
- Fixing anything from this screen: it counts and opens; the fixes stay where they are (*Re-run…*,
  Settings › Documents, the document page).
- Paging the document list.

## Opens

- **Indexing reads metadata.** `extract_metadata` is milliseconds per document and is never run by
  the app's index (KI-87's library). Its own row, with KI-54 in mind.
- **A record that a part ran.** `IngestionEvent` (`db/models.py:1026`) already has `event_type` and
  `notes`; a row per *Re-run…* part and per CLI runner, with no schema change, would let the summary
  say *looked, found none* apart from *never looked*. Its own row; the summary reads it once it exists.
- **A failure's reason outlives the session** only if it is stored; today it lives on
  `app.state.ingest_run`.

## Docs to touch when this lands

- `docs/DEVLOG.md`, `docs/ROADMAP.md` (row 112), `docs/performance.md` (work item 6),
  `GLOSSARY.md` (if *Summary* or *on record* needs pinning), `docs/release-ux-checklist.md` (drive
  the three screens), `.claude/SESSION.md`.
