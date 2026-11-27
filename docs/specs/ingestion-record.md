<!-- status: built · updated: 2026-11-27 (built in 0.9.0 session 3: the backend, then the three screens after the wireframe was approved as drawn; the installed-build cost is still owed) · class: living -->

# Spec: indexing reads the whole document, and says what it found

> The executor brief for 0.9.0 session 3 (`docs/plans/PLAN_2026-11-27_release-0.9.0-ingestion.md`,
> decision 3). Target executor: **Claude Code**. Reads, and must not re-litigate: ADR-043 (received
> content is never rewritten; a table splice is the cache's one allowed exception, `ingest/tables.py`),
> ADR-047 (identity is the source file), ADR-048 (per-part re-runs), the Enrichment-Layer Pattern.
> The faults this record is built to show are measured in
> `docs/reviews/REVIEW_2026-11-27_extraction-faults.md` (local-only).

## As built (2026-11-27, the backend half)

- **Built:** the splice in `cache.load_or_extract` with `_table_fingerprint`; `ingest/record.py`
  (states, warnings, write, prune, reads); the index run's `_read_whole_documents`; `reingest.rerun`
  writes a `rerun` record; `GET …/extraction-record` with its TypeScript type and client. Tests:
  `tests/unit/test_extraction_record.py` (16), `tests/integration/test_api_extraction_record.py` (3).
- **The screens, built the same day** after the owner approved the wireframe as drawn:
  `FileOutcome.warning` and the run's `found` (the indexing window's row: the record's line after
  the passage count, the first warning tinted with its icon); `library.summary.RecordCounts` and
  six issues (`references_heading_only`, `_none`, `_few`, `_not_looked_for`, `figures_uncaptioned`,
  `tables_missed`), rendered as count rows under each pass; `WhatIndexingFound.svelte` under the
  document header, in the page index, its words shaped in `lib/library/extractionRecord.ts`.
  **Departs from the drawing:** the metadata fix reads *Edit metadata*, the control's name
  everywhere else (GLOSSARY), not *Edit details*. Driven in the Vite preview over a scratch data
  home (light and dark), with a fifth paper indexed through the engine for the queue row.
- **Measured on three public papers in a new data home** (2026-11-27, this machine, CPU): figures
  0.35–0.53 s, the table splice 0.75–1.60 s, references 0.02–0.06 s per document, so about 1.5–2 s
  per paper next to tens of seconds of text extraction. The table rows reach both vector stores.
  The installed build's measurement is still owed (Cost, below).
- **A table block sits after the last page** (`splice_tables` appends), so its passages carry the
  last page's number. Placing each table on its page is a later fix.

## Goal

An indexed document has its figures, its reference list and its tables, the same as *Re-run…*
gives it. Each document also carries a **record** of what indexing found and what it didn't, with
the parser's own reasons. A warning reads the record, so the app can say "a references heading, but
no entries" without knowing what kind of document it is.

## Hard constraints

1. **One copy of each pass.** Indexing calls the functions *Re-run… › Figures* and *› References*
   call (`reingest._rerun_figures`, `_rerun_references`), and the table splice `scripts/extract_tables.py`
   performs (`tables.extract_tables` + `splice_tables`). No second implementation.
2. **A pass never fails an ingest.** As with `_fill_missing_metadata`: a pass that raises is
   recorded as `error` with its reason, and the document stays indexed.
3. **Tables enter the passages in the same run.** The splice happens in `cache.load_or_extract`
   between extraction and the cache write, so the chunks are cut from the spliced text and the
   document's `doc_hash` covers its tables. Splicing after chunking would leave the chunks a run
   behind the cache.
4. **The fingerprint covers the splice.** A PDF's `document_fingerprint` hashes the bytecode of
   `ingest/tables.py` and `ingest/regions.py` and pdfplumber's version. A change to table detection
   then re-reads PDFs, as a change to the extractor does (KI-40). **Consequence:** on the upgrade to
   0.9.0 every PDF reads as *changed* and is re-read. The CHANGELOG's *Install* note says so, with
   the cost.
5. **Sidecar, additive, idempotent.** The record is a new table that `create_all` makes. It never
   touches the chunk store or the document's text. Re-running a pass writes a new record; it never
   edits an old one.
6. **Robustness contract.** At 0 documents there are no records and the summary says *nothing indexed
   yet*. At 10,000 the summary reads the latest record per document in one grouped query, never one
   query per document. The record's warnings are document-derived (a share of the document's own
   pages), never a constant tuned to the owner's library.

## The record

Table `extraction_records`, one row **per document per run**: an index run, or a *Re-run…* of any
part. The newest row per document is the current one. Older rows are kept so a fix's before and
after can be read (sessions 4 and 5). At most `RECORDS_KEPT` (5) per document; a write prunes beyond
that.

| Column | Type | What it is |
|---|---|---|
| `id` | str (uuid) | |
| `document_id` | FK `documents.id`, cascade | |
| `doc_hash` | str | the text the record describes. A record whose hash is not the document's current one is **stale**, and the reads say so |
| `run_kind` | `index` \| `rerun` | what wrote it |
| `record_version` | int | the shape of `detail_json` (1) |
| `document_type` | str, nullable | **the slot for a later session.** Always null here, and nothing reads it |
| `references_state` | state | see below |
| `figures_state` | state | |
| `tables_state` | state | |
| `detail_json` | text (JSON) | the counts and reasons below |
| `created_at` | datetime | |

**A state** is one of: `found` (the pass ran and found at least one) · `none_found` (it ran and
found nothing, with a reason) · `partial` (it ran and found something, but the record names a gap:
a heading with fewer entries than the parser expects, pages naming a table none was extracted
from) · `skipped` (not applicable: not a PDF, the file not reachable, with the reason) · `error`
(it raised; the reason is the exception's type and message) · `not_run` (no recorded run has looked
for it). **Each row is complete:** a run that did not run a pass (a *Re-run… › References* alone)
copies that pass's state and section from the newest row, marked `carried_from` with that row's
date, so the newest row always says everything and the summary counts its columns directly.

**`detail_json`** (version 1), every key always present:

```json
{
  "text": {"pages": 14, "chars": 61234, "empty_pages": [], "source": "extracted"},
  "metadata": {"title": true, "authors": false, "year": true, "doi": true},
  "references": {"heading": true, "entries": 38, "without_year": 1, "matched": 4,
                 "notes": ["…the parser's own notes, verbatim…"], "seconds": 0.1, "reason": null},
  "figures": {"found": 6, "cropped": 5, "with_caption": 4, "caption_only": 1,
              "seconds": 0.7, "reason": null},
  "tables": {"extracted": 2, "pages_naming_a_table": [5, 9], "pages_without_one": [9],
             "seconds": 1.3, "reason": null}
}
```

`metadata` is what the document has **after** the catalogue import and the blank-filling (ADR-049),
so the record says what the user will see. `tables.seconds` is the splice's time, measured where
it runs (in the extraction worker), and travels with the cache as a small sidecar file beside the
`.md` (`<name>.tables.json`), so the record can read it after the loop.

## The warnings (what each reads)

A warning is a predicate over the current record, defined once in `ingest/record.py` and used by the
document page, the queue row and the summary, so their counts cannot disagree.

| Warning | Reads | Says |
|---|---|---|
| `references_none` | `references_state == none_found` | *No reference list found* + the parser's note |
| `references_heading_only` | `references.heading` and `entries == 0` | *A references heading, but no entries* |
| `references_few` | `partial`: `0 < entries < REFERENCES_FEW` (5) | *Only N references found* |
| `figures_uncaptioned` | `figures.found > 0` and `with_caption < found` | *N figures without a caption* |
| `tables_missed` | `pages_without_one` not empty | *Pages N, M name a table that wasn't extracted* |
| `metadata_missing` | any of `metadata.*` false | *No authors* (each field named) |
| `text_empty_pages` | `empty_pages` covers ≥ 20 % of `pages` | *N pages have no text* |
| `stale` | `doc_hash` ≠ the document's | *Recorded for an earlier reading of this file* |

`REFERENCES_FEW` and the 20 % share are **structural** (named in `record.py` with why), not tuned:
five is below any real paper's reference list, and the share is of the document's own pages.

## Where it is shown (the wireframe decides the layout)

Drawn in [`docs/wireframes/extraction-record.md`](../wireframes/extraction-record.md) (four
screens); the screens are built after the owner's markup of it.

1. **The queue's finished row** (walk item 17's first form): the row of a document indexed this run
   says what was found in one line (*38 references · 6 figures · 2 tables*) and names its first
   warning.
2. **The document page:** a section *What indexing found* listing the record, its warnings and when
   it was written, with *Re-run…* beside a warning that a re-run can address.
3. **The Library summary:** *not looked for* becomes a count. A document with no record is *not looked
   for* (indexed before 0.9.0, until it is re-read); one with a record is counted by its state. Each
   warning is an issue (`ISSUES`), so *Show the N* opens exactly those documents.

## The route

`GET /api/library/documents/{id}/extraction-record` → the current record (the newest), its
warnings, and `stale`; `404` for an unknown document; `{"record": null}` for a document never
recorded. The summary's existing route gains the per-state counts. The queue row reads the record
through the run's finished-file outcome (`ingest.run.FileOutcome`), which gains the one-line
summary, so the queue needs no second request.

## Test cases (the guard tests)

1. Indexing a PDF with a reference list, a figure and a table writes one record with all three
   `found`, and the chunks contain the table's rows.
2. A pass that raises leaves the document indexed and the record's state `error` with the reason.
3. A non-PDF records figures and tables `skipped` ("PDF-only"), references from its text.
4. `references_heading_only` fires on a heading with no entries; `references_none` on no heading.
5. A change to a function in `tables.py` changes the PDF fingerprint and not the EPUB one.
6. *Re-run… › References* writes a new record with `references` fresh and figures carried from
   the previous record (`carried_from` set); tables are read from the cache as it stands.
7. More than `RECORDS_KEPT` runs leave exactly that many rows.
8. The summary at 0 documents has no record counts and says so; at N documents with no records,
   every one is *not looked for*.
9. A record whose `doc_hash` is not the document's reads as `stale`.

## Cost

Measured in session 2 on this machine, in memory (the review's *Cost per pass*): figures 0.74 s per
document (median), the table pages and tables about 1.35 s on average, references 3 ms. Added to an
average 47 s of text extraction, that is about 5 s per document, or roughly 14 hours on top of 130 at
10,000 documents. Session 3 re-measures it on the installed build and writes the result to
`docs/performance.md`.

## Out of scope

The document type (the slot only). Fixing any extraction fault (sessions 4 and 5). The in-depth
track (session 6). The VLM figure descriptions, which cost money per figure and never run at
indexing (ADR-048, KI-4).
