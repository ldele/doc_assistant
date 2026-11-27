<!-- status: active · updated: 2026-11-27 (approved by the owner as drawn, no markup; drawn for 0.9.0 session 3) · class: living -->

# Wireframe: what indexing found (the extraction record)

The scene is `extraction-record.excalidraw` beside this file; open it on the canvas with
`excalidraw-canvas import docs/wireframes/extraction-record.excalidraw`, edit, and export it back over
the same file. The record, its states and what each warning reads are in
[`docs/specs/ingestion-record.md`](../specs/ingestion-record.md). Every number and title on the
drawing is invented or from the public papers; none comes from the owner's library.

1. **S1, the indexing window after a run** (`[s1]`): each finished row's *What happened* column says
   what the record found (`[s1-row0]`), and a row with a warning is tinted and names it after the
   counts (`[s1-row1]`, `[s1-row3]`). A non-PDF says why figures and tables are absent (`[s1-row2]`).
   A failed row keeps today's wording (`[s1-row4]`).
2. **S2, the document page** (`[s2]`): a block *What indexing found* (`[s2-card]`) under the header,
   before *Connections*, and in the page index. Warnings first, each with the fix beside it
   (`[s2-warn0-fix]` *Re-run…* opens the existing dialog with that part ticked; `[s2-warn2-fix]`
   *Edit details*), then one line per pass and one for metadata (`[s2-fact0]` to `[s2-fact3]`).
3. **S3, the same block when there is nothing current** (`[s3]`): never recorded, the document indexed
   before 0.9.0 (`[s3-card]`); recorded for an earlier reading of the file (`[s3-card2]`). Both offer
   *Re-run…*.
4. **S4, the Library summary's *Found in each document* card** (`[s4-card]`): references counted by
   state, *Not looked for yet* last (`[s4-row1]` to `[s4-row4]`), figures with the uncaptioned
   count (`[s4-row5]`), tables with the documents whose pages name a table none was extracted
   from (`[s4-row6]`). Every row opens its documents in the grid, as the summary's rows do today.

**Approved as drawn** by the owner on 2026-11-27, with no markup, so each open choice below stands as
drawn: one warning per queue row, the block under the header, the summary split by state.

**What the markup should decide.**

- Whether the queue row carries one warning (drawn) or all of them.
- Whether the document page's block comes first under the header (drawn) or at the foot, where
  *Letter spacing* sits.
- Whether the summary splits each pass into its states (drawn, references only) or keeps one row
  per pass with the split inside the documents it opens.

## Wireframe

Read 2026-11-27 from the canvas as exported: `wireframe --score` gave
`screens 4 · components 52 · fallbacks 0 · inferred 0 · unnamedScreens 0 · orphans 0`, and
`wireframe --lint` found nothing.

```
4 screens, 52 components, 0 connections, 0 annotations.
Approved: 0 of 4 screens.

Roles are inferred from geometry, style and wording — a trailing `?` marks a low-confidence guess. Raw type and size follow each entry, so you can disagree with the reading. Numbering is reading order (top-to-bottom, left-to-right within a row).

Left out: the markup legend [legend] (18 elements), the key to the markup colours rather than part of the design.

### Screen 1 "S1 — Indexing: a finished run"  [s1] frame 1160x600 · not approved yet
  1. heading "Indexed 4 of 4"  [s1-h] text 700x30 · 20px
  2. text "Each finished row says what indexing found, fro…"  [s1-sub] text 900x20 · 13px
  3. header "File Status What happened"  [s1-cols] rectangle 1096x36 · fill #fbfaf8
  4. list-item "resnet_he_2015.pdf Indexed 41 references · 9 fi…"  [s1-row0] rectangle 1096x48 · fill #ffffff
  5. list-item "deep_speech2_amodei_2015.pdf Indexed 65 referen…"  [s1-row1] rectangle 1096x48 · fill #ffec99
  6. list-item "note.md Indexed no references (figures and tabl…"  [s1-row2] rectangle 1096x48 · fill #ffffff
  7. list-item "scan_1987.pdf Indexed no references · 0 figures…"  [s1-row3] rectangle 1096x48 · fill #ffec99
  8. list-item "broken.pdf Failed The file could not be read: n…"  [s1-row4] rectangle 1096x48 · fill #ffffff
  9. text "A warning row is tinted and its text names the …"  [s1-note] text 1096x40 · 12px

### Screen 2 "S2 — Document page: What indexing found"  [s2] frame 1160x880 · not approved yet
  1. heading "Deep Speech 2: End-to-End Speech Recognition in…"  [s2-title] text 900x40 · 26px
  2. text "No authors · 2015 · PDF · 28 pages"  [s2-meta] text 900x20 · 13px
  3. card  [s2-card] rectangle 1096x560 · fill #ffffff
    3.1. heading "What indexing found"  [s2-card-h] text 600x30 · 20px
    3.2. text "Recorded when this document was indexed, 27 Nov…"  [s2-card-when] text 900x18 · 12px
    3.3. list-item "Pages 6, 8, 9 and 9 more name a table that wasn…"  [s2-warn0] rectangle 880x44 · fill #ffec99
    3.4. button "Re-run…"  [s2-warn0-fix] rectangle 160x40 · fill #ffffff
    3.5. list-item "3 figures without a caption"  [s2-warn1] rectangle 880x44 · fill #ffec99
    3.6. button "Re-run…"  [s2-warn1-fix] rectangle 160x40 · fill #ffffff
    3.7. list-item "No authors"  [s2-warn2] rectangle 880x44 · fill #ffec99
    3.8. button "Edit details"  [s2-warn2-fix] rectangle 160x40 · fill #ffffff
    3.9. divider  [s2-div] rectangle 1060x2 · fill #d7d5cc
    3.10. list-item "References 65 found · 1 without a year · 4 of t…"  [s2-fact0] rectangle 1060x44 · fill #ffffff
    3.11. list-item "Figures 8 found · 5 with a caption · 2 found by…"  [s2-fact1] rectangle 1060x44 · fill #ffffff
    3.12. list-item "Tables 1 extracted · 12 pages name a table"  [s2-fact2] rectangle 1060x44 · fill #ffffff
    3.13. list-item "Metadata title, year and DOI found · no authors"  [s2-fact3] rectangle 1060x44 · fill #ffffff
    3.14. text "Re-run… opens the existing dialog with that par…"  [s2-foot] text 1060x36 · 12px

### Screen 3 "S3 — Document page: not looked for, and stale"  [s3] frame 1160x560 · not approved yet
  1. heading "The Organization of Behavior"  [s3-title] text 900x40 · 26px
  2. card  [s3-card] rectangle 1096x140 · fill #ffffff
    2.1. heading "What indexing found"  [s3-card-h] text 600x30 · 20px
    2.2. text "Not looked for yet. This document was indexed b…"  [s3-card-t] text 860x40 · 13px
    2.3. button "Re-run…"  [s3-rerun] rectangle 160x40 · fill #ffffff
  3. card  [s3-card2] rectangle 1096x140 · fill #ffffff
    3.1. heading "What indexing found"  [s3-card2-h] text 600x30 · 20px
    3.2. text "Recorded for an earlier reading of this file (1…"  [s3-card2-t] text 860x40 · 13px
    3.3. button "Re-run…"  [s3-rerun2] rectangle 160x40 · fill #ffffff

### Screen 4 "S4 — Library summary: Found in each document"  [s4] frame 1160x820 · not approved yet
  1. heading "Summary of your library"  [s4-title] text 700x40 · 26px
  2. text "140 documents · 41,200 passages · 6,980 pages"  [s4-ctx] text 900x20 · 13px
  3. card  [s4-card] rectangle 1096x520 · fill #ffffff
    3.1. heading "Found in each document"  [s4-card-h] text 600x30 · 20px
    3.2. text "Counted from each document's newest record. Not…"  [s4-card-d] text 1000x18 · 12px
    3.3. list-item "References: a list in 96 of 140 ›"  [s4-row0] rectangle 1060x44 · fill #ffffff
    3.4. list-item "A references heading, but no entries · 3 ›"  [s4-row1] rectangle 1060x44 · fill #ffffff
    3.5. list-item "No reference list found · 9 ›"  [s4-row2] rectangle 1060x44 · fill #ffffff
    3.6. list-item "Only a few references found · 4 ›"  [s4-row3] rectangle 1060x44 · fill #ffffff
    3.7. list-item "Not looked for yet · 28 ›"  [s4-row4] rectangle 1060x44 · fill #ffffff
    3.8. list-item "Figures: found in 102 of 140 · 210 without a ca…"  [s4-row5] rectangle 1060x44 · fill #ffffff
    3.9. list-item "Tables: 61 extracted in 18 documents · 40 docum…"  [s4-row6] rectangle 1060x44 · fill #ffffff
    3.10. text "Each row opens its N documents in the grid (ADR…"  [s4-foot] text 1060x36 · 12px
```
