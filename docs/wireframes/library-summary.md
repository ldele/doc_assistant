<!-- status: active · updated: 2026-11-05 (revision 2: the user's seven recommendations, drawn at rev 1001; revision 1 was approved with no markup at rev 1060) · class: living -->

# Wireframe: the library summary

The scene is `library-summary.excalidraw` beside this file; open it on the canvas with
`excalidraw-canvas import docs/wireframes/library-summary.excalidraw`, edit, and export it back over
the same file. The route, its wire type and what each number is read from (with how far it can be
trusted) are in [`docs/specs/feature-library-summary.md`](../specs/feature-library-summary.md).

**Two revisions, both 2026-11-05.** Revision 1 (three screens) was approved by the user with no
markup. The user then sent seven recommendations, all accepted (*"Send all"*), with the condition
*"as long as you follow the design guidelines in the implementation"* (ADR-063, GLOSSARY U-003).
Revision 2, drawn here, is revision 1 with six of them applied; the seventh (keep each count run, to
show a trend) is later work. It was drawn on a fresh canvas and not marked up again.

1. **Summary of your library** (`[s1]`): *All documents* is open, *Summary* chosen in the pane's
   layout toggle.
2. **Summary of Thesis chapter 2** (`[s2]`): a folder opened from the rail's *Folders*.
3. **The same view at 0 documents** (`[s3]`): an index that stopped before any file finished.
4. **A library indexed only in the app** (`[s4]`): nothing has read metadata, references or
   figures, which is what KI-87's library looks like.

**What the drawing decides.**

- **Where it opens.** A third layout of the Library pane, beside *Grid* and *List* (`[s1-grid]`,
  `[s1-list]`, `[s1-summary]`), for the whole library or the open folder, so a folder's summary is
  one click from opening the folder in the rail (`[s1-rail-f0]` → `[s2]`). Sort and *Select* are
  hidden in this layout; *Add documents* stays.
- **A count is navigation, not an action** (recommendation 1, ADR-063). Every row with something
  behind it is itself the click target: the whole row, a hover background and a chevron, the same
  pattern as the Graph's concept view lists its documents. A row whose count is 0 is plain text.
  The screen's only buttons are *Count again* (quiet) and *Add documents* (primary, unchanged).
- **What needs action leads** (recommendation 2). *Need attention* is the first tile, tinted when it
  is not 0, and it scrolls to *From indexing*; *Documents* is beside it. Passages and pages are
  context: one line under the title (`[s1-context]`).
- **From indexing: what happened, how many, where to fix it** (recommendation 5). Each row says the
  reason and the count, and names the fix at its trailing end: files that failed in this session's
  run open the indexing screen (which has *Retry* and *Index the remaining*); files never indexed
  open Settings › Documents; documents with no text or a flagged extraction open in the grid, where
  *Re-run… › Text & chunks* is. The summary opens the fix and never runs it. A status tint always
  comes with an icon, never colour alone.
- **N of M with a thin bar** (recommendation 3) on every *Metadata* and *Found in each document* row:
  how many have it, of how many could, and what is left to open (*12 missing*, *42 none*). No bar
  and no percentage at 0 documents; there, the cards give way to one sentence (`[s3]`).
- **"Not checked" is not "none found"** (recommendation 4). When no document has any of the four
  metadata fields, the card says so once (`[s4-card0-r0]`: *Not read for 140 documents*) instead of
  four rows of defects. References and figures do the same while none is on record anywhere
  (`[s4-card1-r0]`, `[s4-card1-r1]`), because nothing yet records that a re-run looked.
- **Charts** (recommendation 6, the dataviz guide): horizontal bars, one hue, the value at each
  bar's tip. *Types* are ranked, largest first. *Years* keep decade order, which departs from the
  recommendation's "sorted by count": decades are an ordered scale, and a time axis sorted by size
  stops reading as time. *No year* is a bar apart, after a gap. With no year anywhere (`[s4]`), a
  sentence replaces the chart.
- **A folder** cannot count files that are not documents, because they belong to no folder: its row
  says so and opens the library's summary (`[s2-to-library]` → `[s1]`).

**Reading order.** The reading below numbers the pane content of `[s1]`, `[s2]` and `[s4]` column
by column, so the two tiles and *Count again* come out of order. That is a defect in the canvas
toolkit, its ticket T-006 (a corner-to-corner sidebar merges its siblings into one row), not the
intended order, and the drawing is left as drawn. The intended order: the rail; the top bar; the
title, the context line, *Count again*; *Need attention*, then *Documents*; *From indexing* and its
rows; *Metadata*, then *Found in each document*; *What is in it*, then *Types*, then *Years*.

The file names and counts are examples, not a real library.

## Wireframe

Score at rev 1001: `fallbacks 0 · inferred 0 · unnamedScreens 0 · orphans 0`, 4 screens, 319
components.

```text
## Wireframe reading
4 screens, 319 components, 2 connections, 0 annotations.

Roles are inferred from geometry, style and wording — a trailing `?` marks a low-confidence guess. Raw type and size follow each entry, so you can disagree with the reading. Numbering is reading order (top-to-bottom, left-to-right within a row).

### Screen 1 "Summary of your library"  [s1] rectangle 1160x1312
  1. sidebar  [s1-rail] rectangle 240x1310 · fill #f8f7f3
    1.1. button "All documents 312"  [s1-rail-all] rectangle 208x36 · fill #e4e1f5
    1.2. text "Folders"  [s1-rail-fh] text 80x17 · 12px
    1.3. button "Manage…"  [s1-rail-manage] rectangle 74x28
    1.4. button "Thesis chapter 2 24"  [s1-rail-f0] rectangle 208x36 · fill #ffffff
    1.5. button "Reading group 9"  [s1-rail-f1] rectangle 208x36 · fill #ffffff
    1.6. button "Methods 41"  [s1-rail-f2] rectangle 208x36 · fill #ffffff
  2. header  [s1-bar] rectangle 918x72 · fill #f8f7f3
    2.1. text "Library"  [s1-crumbs] text 360x21 · 15px
    2.2. button "Grid"  [s1-grid] rectangle 72x40 · fill #ffffff
    2.3. button "List"  [s1-list] rectangle 72x40 · fill #ffffff
    2.4. button "Summary"  [s1-summary] rectangle 104x40 · fill #e4e1f5
    2.5. button "Add documents"  [s1-add] rectangle 180x40 · fill #4a3fa6
  3. heading "Summary of your library"  [s1-title] text 600x35 · 26px
  4. text "41,180 passages · 6,904 pages in 300 paged docu…"  [s1-context] text 700x18 · 13px
  5. text "Counted at 14:02. Nothing here changes your lib…"  [s1-when] text 600x17 · 12px
  6. card  [s1-kpi0] rectangle 420x96 · fill #fbeecb
    6.1. heading "11"  [s1-kpi0-v] text 120x35 · 26px
    6.2. text "Need attention: files or documents. Opens From …"  [s1-kpi0-k] text 388x17 · 12px
  7. heading "From indexing"  [s1-idx-h] text 300x28 · 20px
  8. text "Each row: what happened, how many, and where to…"  [s1-idx-d] text 700x17 · 12px
  9. list-item  [s1-files] rectangle 856x68 · fill #fbeecb
    9.1. icon  [s1-files-i] rectangle 18x18
    9.2. text "6 files in your documents folder are not in the…"  [s1-files-t] text 560x21 · 15px
    9.3. text "2 failed in this session's run; the others were…"  [s1-files-d] text 560x17 · 12px
    9.4. text "See why on the indexing screen"  [s1-files-f] text 210x17 · 12px
    9.5. heading "›"  [s1-files-c] text 14x28 · 20px
  10. list-item  [s1-row1] rectangle 856x68 · fill #fbdcd8
    10.1. icon  [s1-row1-i] rectangle 18x18
    10.2. text "2 documents gave no text"  [s1-row1-t] text 560x21 · 15px
    10.3. text "Probably scans with no text layer: nothing in t…"  [s1-row1-d] text 560x17 · 12px
    10.4. text "Fix: Re-run… > Text & chunks"  [s1-row1-f] text 210x17 · 12px
    10.5. heading "›"  [s1-row1-c] text 14x28 · 20px
  11. list-item  [s1-row2] rectangle 856x68 · fill #fbeecb
    11.1. icon  [s1-row2-i] rectangle 18x18
    11.2. text "3 extractions were flagged"  [s1-row2-t] text 560x21 · 15px
    11.3. text "Marked marginal when indexed: little text per p…"  [s1-row2-d] text 560x17 · 12px
    11.4. text "Fix: Re-run… > Text & chunks"  [s1-row2-f] text 210x17 · 12px
    11.5. heading "›"  [s1-row2-c] text 14x28 · 20px
  12. card  [s1-card0] rectangle 420x312 · fill #ffffff
    12.1. heading "Metadata"  [s1-card0-h] text 380x28 · 20px
    12.2. text "Documents that have it, of 312. A value you typ…"  [s1-card0-d] text 388x17 · 12px
    12.3. list-item  [s1-card0-r0] rectangle 388x44 · fill #ffffff
      12.3.1. text "Title"  [s1-card0-r0-n] text 96x20 · 14px
      12.3.2. text "300 of 312"  [s1-card0-r0-v] text 110x18 · 13px
      12.3.3. chart  [s1-card0-r0-track] rectangle 90x8 · fill #efece5
        12.3.3.1. chart  [s1-card0-r0-bar] rectangle 86x8 · fill #4a3fa6
      12.3.4. text "12 missing"  [s1-card0-r0-tail] text 64x17 · 12px
      12.3.5. heading "›"  [s1-card0-r0-c] text 12x28 · 20px
    12.4. list-item  [s1-card0-r1] rectangle 388x44 · fill #ffffff
      12.4.1. text "Authors"  [s1-card0-r1-n] text 96x20 · 14px
      12.4.2. text "281 of 312"  [s1-card0-r1-v] text 110x18 · 13px
      12.4.3. chart  [s1-card0-r1-track] rectangle 90x8 · fill #efece5
        12.4.3.1. chart  [s1-card0-r1-bar] rectangle 81x8 · fill #4a3fa6
      12.4.4. text "31 missing"  [s1-card0-r1-tail] text 64x17 · 12px
      12.4.5. heading "›"  [s1-card0-r1-c] text 12x28 · 20px
    12.5. list-item  [s1-card0-r2] rectangle 388x44 · fill #ffffff
      12.5.1. text "Year"  [s1-card0-r2-n] text 96x20 · 14px
      12.5.2. text "290 of 312"  [s1-card0-r2-v] text 110x18 · 13px
      12.5.3. chart  [s1-card0-r2-track] rectangle 90x8 · fill #efece5
        12.5.3.1. chart  [s1-card0-r2-bar] rectangle 83x8 · fill #4a3fa6
      12.5.4. text "22 missing"  [s1-card0-r2-tail] text 64x17 · 12px
      12.5.5. heading "›"  [s1-card0-r2-c] text 12x28 · 20px
    12.6. list-item  [s1-card0-r3] rectangle 388x44 · fill #ffffff
      12.6.1. text "DOI"  [s1-card0-r3-n] text 96x20 · 14px
      12.6.2. text "122 of 312"  [s1-card0-r3-v] text 110x18 · 13px
      12.6.3. chart  [s1-card0-r3-track] rectangle 90x8 · fill #efece5
        12.6.3.1. chart  [s1-card0-r3-bar] rectangle 35x8 · fill #4a3fa6
      12.6.4. text "190 missing"  [s1-card0-r3-tail] text 64x17 · 12px
      12.6.5. heading "›"  [s1-card0-r3-c] text 12x28 · 20px
    12.7. text "Indexing does not read these. Re-run… > Metadat…"  [s1-card0-f] text 388x16 · 11px
  13. heading "What is in it"  [s1-what-h] text 300x28 · 20px
  14. text "180 are in a folder and 132 in none. Added betw…"  [s1-what-d] text 856x17 · 12px
  15. card  [s1-chart0] rectangle 420x240 · fill #ffffff
    15.1. heading "Types"  [s1-chart0-h] text 300x28 · 20px
    15.2. chart "Ranked horizontal bars, largest first, one hue,…"  [s1-chart0-p] rectangle 388x172
  16. card  [s1-kpi1] rectangle 420x96 · fill #ffffff
    16.1. heading "312"  [s1-kpi1-v] text 160x35 · 26px
    16.2. text "Documents"  [s1-kpi1-k] text 200x17 · 12px
  17. card  [s1-card1] rectangle 420x312 · fill #ffffff
    17.1. heading "Found in each document"  [s1-card1-h] text 380x28 · 20px
    17.2. text "On record, of 312 (figures: of 300 PDFs)."  [s1-card1-d] text 388x17 · 12px
    17.3. list-item  [s1-card1-r0] rectangle 388x44 · fill #ffffff
      17.3.1. text "References"  [s1-card1-r0-n] text 96x20 · 14px
      17.3.2. text "270 of 312"  [s1-card1-r0-v] text 110x18 · 13px
      17.3.3. chart  [s1-card1-r0-track] rectangle 90x8 · fill #efece5
        17.3.3.1. chart  [s1-card1-r0-bar] rectangle 77x8 · fill #4a3fa6
      17.3.4. text "42 none"  [s1-card1-r0-tail] text 64x17 · 12px
      17.3.5. heading "›"  [s1-card1-r0-c] text 12x28 · 20px
    17.4. list-item  [s1-card1-r1] rectangle 388x44 · fill #ffffff
      17.4.1. text "Figures"  [s1-card1-r1-n] text 96x20 · 14px
      17.4.2. text "250 of 300"  [s1-card1-r1-v] text 110x18 · 13px
      17.4.3. chart  [s1-card1-r1-track] rectangle 90x8 · fill #efece5
        17.4.3.1. chart  [s1-card1-r1-bar] rectangle 75x8 · fill #4a3fa6
      17.4.4. text "50 none"  [s1-card1-r1-tail] text 64x17 · 12px
      17.4.5. heading "›"  [s1-card1-r1-c] text 12x28 · 20px
    17.5. list-item  [s1-card1-r2] rectangle 388x44 · fill #ffffff
      17.5.1. text "Spacing"  [s1-card1-r2-n] text 96x20 · 14px
      17.5.2. text "18 of 300"  [s1-card1-r2-v] text 110x18 · 13px
      17.5.3. chart  [s1-card1-r2-track] rectangle 90x8 · fill #efece5
        17.5.3.1. chart  [s1-card1-r2-bar] rectangle 5x8 · fill #4a3fa6
      17.5.4. text "18 open"  [s1-card1-r2-tail] text 64x17 · 12px
      17.5.5. heading "›"  [s1-card1-r2-c] text 12x28 · 20px
    17.6. list-item  [s1-card1-r3] rectangle 388x44 · fill #ffffff
      17.6.1. text "Keywords"  [s1-card1-r3-n] text 96x20 · 14px
      17.6.2. text "300 of 312"  [s1-card1-r3-v] text 110x18 · 13px
      17.6.3. chart  [s1-card1-r3-track] rectangle 90x8 · fill #efece5
        17.6.3.1. chart  [s1-card1-r3-bar] rectangle 86x8 · fill #4a3fa6
      17.6.4. text "12 none"  [s1-card1-r3-tail] text 64x17 · 12px
      17.6.5. heading "›"  [s1-card1-r3-c] text 12x28 · 20px
    17.7. text "None on record: none found, or never looked for…"  [s1-card1-f] text 388x16 · 11px
  18. card  [s1-chart1] rectangle 420x240 · fill #ffffff
    18.1. heading "Years"  [s1-chart1-h] text 300x28 · 20px
    18.2. chart "Horizontal bars in decade order, one hue, value…"  [s1-chart1-p] rectangle 388x172
  19. button "Count again"  [s1-recount] rectangle 140x40

### Screen 2 "Summary of Thesis chapter 2"  [s2] rectangle 1160x1312
  1. sidebar  [s2-rail] rectangle 240x1310 · fill #f8f7f3
    1.1. button "All documents 312"  [s2-rail-all] rectangle 208x36 · fill #ffffff
    1.2. text "Folders"  [s2-rail-fh] text 80x17 · 12px
    1.3. button "Manage…"  [s2-rail-manage] rectangle 74x28
    1.4. button "Thesis chapter 2 24"  [s2-rail-f0] rectangle 208x36 · fill #e4e1f5
    1.5. button "Reading group 9"  [s2-rail-f1] rectangle 208x36 · fill #ffffff
    1.6. button "Methods 41"  [s2-rail-f2] rectangle 208x36 · fill #ffffff
  2. header  [s2-bar] rectangle 918x72 · fill #f8f7f3
    2.1. text "Library > Thesis chapter 2"  [s2-crumbs] text 360x21 · 15px
    2.2. button "Grid"  [s2-grid] rectangle 72x40 · fill #ffffff
    2.3. button "List"  [s2-list] rectangle 72x40 · fill #ffffff
    2.4. button "Summary"  [s2-summary] rectangle 104x40 · fill #e4e1f5
    2.5. button "Add documents"  [s2-add] rectangle 180x40 · fill #4a3fa6
  3. heading "Summary of Thesis chapter 2"  [s2-title] text 600x35 · 26px
  4. text "3,020 passages · 512 pages in 22 paged documents"  [s2-context] text 700x18 · 13px
  5. text "Counted at 14:02. Nothing here changes your lib…"  [s2-when] text 600x17 · 12px
  6. card  [s2-kpi0] rectangle 420x96 · fill #fbeecb
    6.1. heading "1"  [s2-kpi0-v] text 120x35 · 26px
    6.2. text "Need attention: documents. Opens From indexing."  [s2-kpi0-k] text 388x17 · 12px
  7. heading "From indexing"  [s2-idx-h] text 300x28 · 20px
  8. text "Each row: what happened, how many, and where to…"  [s2-idx-d] text 700x17 · 12px
  9. list-item  [s2-row0] rectangle 856x68 · fill #fbeecb
    9.1. icon  [s2-row0-i] rectangle 18x18
    9.2. text "1 extraction was flagged"  [s2-row0-t] text 560x21 · 15px
    9.3. text "Marked marginal when indexed: little text per p…"  [s2-row0-d] text 560x17 · 12px
    9.4. text "Fix: Re-run… > Text & chunks"  [s2-row0-f] text 210x17 · 12px
    9.5. heading "›"  [s2-row0-c] text 14x28 · 20px
  10. list-item  [s2-row1] rectangle 856x68 · fill #f8f7f3
    10.1. text "0 documents gave no text"  [s2-row1-t] text 560x21 · 15px
    10.2. text "Every document in this folder gave text when it…"  [s2-row1-d] text 560x17 · 12px
  11. list-item  [s2-to-library] rectangle 856x68 · fill #ffffff
    11.1. text "Files that are not in the library belong to no …"  [s2-to-library-t] text 560x21 · 15px
    11.2. text "The library's summary counts them: 6 today."  [s2-to-library-d] text 560x17 · 12px
    11.3. text "Open the library's summary"  [s2-to-library-f] text 210x17 · 12px
    11.4. heading "›"  [s2-to-library-c] text 14x28 · 20px
  12. card  [s2-card0] rectangle 420x312 · fill #ffffff
    12.1. heading "Metadata"  [s2-card0-h] text 380x28 · 20px
    12.2. text "Documents that have it, of 24. A value you type…"  [s2-card0-d] text 388x17 · 12px
    12.3. card  [s2-card0-r0] rectangle 388x44 · fill #f8f7f3
      12.3.1. text "Title"  [s2-card0-r0-n] text 96x20 · 14px
      12.3.2. text "24 of 24"  [s2-card0-r0-v] text 110x18 · 13px
      12.3.3. chart  [s2-card0-r0-track] rectangle 90x8 · fill #efece5
      12.3.4. chart  [s2-card0-r0-bar] rectangle 90x8 · fill #4a3fa6
    12.4. list-item  [s2-card0-r1] rectangle 388x44 · fill #ffffff
      12.4.1. text "Authors"  [s2-card0-r1-n] text 96x20 · 14px
      12.4.2. text "21 of 24"  [s2-card0-r1-v] text 110x18 · 13px
      12.4.3. chart  [s2-card0-r1-track] rectangle 90x8 · fill #efece5
        12.4.3.1. chart  [s2-card0-r1-bar] rectangle 78x8 · fill #4a3fa6
      12.4.4. text "3 missing"  [s2-card0-r1-tail] text 64x17 · 12px
      12.4.5. heading "›"  [s2-card0-r1-c] text 12x28 · 20px
    12.5. list-item  [s2-card0-r2] rectangle 388x44 · fill #ffffff
      12.5.1. text "Year"  [s2-card0-r2-n] text 96x20 · 14px
      12.5.2. text "23 of 24"  [s2-card0-r2-v] text 110x18 · 13px
      12.5.3. chart  [s2-card0-r2-track] rectangle 90x8 · fill #efece5
        12.5.3.1. chart  [s2-card0-r2-bar] rectangle 86x8 · fill #4a3fa6
      12.5.4. text "1 missing"  [s2-card0-r2-tail] text 64x17 · 12px
      12.5.5. heading "›"  [s2-card0-r2-c] text 12x28 · 20px
    12.6. list-item  [s2-card0-r3] rectangle 388x44 · fill #ffffff
      12.6.1. text "DOI"  [s2-card0-r3-n] text 96x20 · 14px
      12.6.2. text "10 of 24"  [s2-card0-r3-v] text 110x18 · 13px
      12.6.3. chart  [s2-card0-r3-track] rectangle 90x8 · fill #efece5
        12.6.3.1. chart  [s2-card0-r3-bar] rectangle 37x8 · fill #4a3fa6
      12.6.4. text "14 missing"  [s2-card0-r3-tail] text 64x17 · 12px
      12.6.5. heading "›"  [s2-card0-r3-c] text 12x28 · 20px
    12.7. text "Indexing does not read these. Re-run… > Metadat…"  [s2-card0-f] text 388x16 · 11px
  13. heading "What is in it"  [s2-what-h] text 300x28 · 20px
  14. text "5 of these are also in another folder. Added be…"  [s2-what-d] text 856x17 · 12px
  15. card  [s2-chart0] rectangle 420x240 · fill #ffffff
    15.1. heading "Types"  [s2-chart0-h] text 300x28 · 20px
    15.2. chart "Ranked horizontal bars, one hue, value at the t…"  [s2-chart0-p] rectangle 388x172
  16. card  [s2-kpi1] rectangle 420x96 · fill #ffffff
    16.1. heading "24"  [s2-kpi1-v] text 160x35 · 26px
    16.2. text "Documents"  [s2-kpi1-k] text 200x17 · 12px
  17. card  [s2-card1] rectangle 420x312 · fill #ffffff
    17.1. heading "Found in each document"  [s2-card1-h] text 380x28 · 20px
    17.2. text "On record, of 24 (figures: of 22 PDFs)."  [s2-card1-d] text 388x17 · 12px
    17.3. list-item  [s2-card1-r0] rectangle 388x44 · fill #ffffff
      17.3.1. text "References"  [s2-card1-r0-n] text 96x20 · 14px
      17.3.2. text "22 of 24"  [s2-card1-r0-v] text 110x18 · 13px
      17.3.3. chart  [s2-card1-r0-track] rectangle 90x8 · fill #efece5
        17.3.3.1. chart  [s2-card1-r0-bar] rectangle 82x8 · fill #4a3fa6
      17.3.4. text "2 none"  [s2-card1-r0-tail] text 64x17 · 12px
      17.3.5. heading "›"  [s2-card1-r0-c] text 12x28 · 20px
    17.4. list-item  [s2-card1-r1] rectangle 388x44 · fill #ffffff
      17.4.1. text "Figures"  [s2-card1-r1-n] text 96x20 · 14px
      17.4.2. text "18 of 22"  [s2-card1-r1-v] text 110x18 · 13px
      17.4.3. chart  [s2-card1-r1-track] rectangle 90x8 · fill #efece5
        17.4.3.1. chart  [s2-card1-r1-bar] rectangle 73x8 · fill #4a3fa6
      17.4.4. text "4 none"  [s2-card1-r1-tail] text 64x17 · 12px
      17.4.5. heading "›"  [s2-card1-r1-c] text 12x28 · 20px
    17.5. list-item  [s2-card1-r2] rectangle 388x44 · fill #ffffff
      17.5.1. text "Spacing"  [s2-card1-r2-n] text 96x20 · 14px
      17.5.2. text "2 of 22"  [s2-card1-r2-v] text 110x18 · 13px
      17.5.3. chart  [s2-card1-r2-track] rectangle 90x8 · fill #efece5
        17.5.3.1. chart  [s2-card1-r2-bar] rectangle 8x8 · fill #4a3fa6
      17.5.4. text "2 open"  [s2-card1-r2-tail] text 64x17 · 12px
      17.5.5. heading "›"  [s2-card1-r2-c] text 12x28 · 20px
    17.6. card  [s2-card1-r3] rectangle 388x44 · fill #f8f7f3
      17.6.1. text "Keywords"  [s2-card1-r3-n] text 96x20 · 14px
      17.6.2. text "24 of 24"  [s2-card1-r3-v] text 110x18 · 13px
      17.6.3. chart  [s2-card1-r3-track] rectangle 90x8 · fill #efece5
      17.6.4. chart  [s2-card1-r3-bar] rectangle 90x8 · fill #4a3fa6
    17.7. text "None on record: none found, or never looked for…"  [s2-card1-f] text 388x16 · 11px
  18. card  [s2-chart1] rectangle 420x240 · fill #ffffff
    18.1. heading "Years"  [s2-chart1-h] text 300x28 · 20px
    18.2. chart "Horizontal bars in decade order, one hue, value…"  [s2-chart1-p] rectangle 388x172
  19. button "Count again"  [s2-recount] rectangle 140x40

### Screen 3 "Summary of your library"  [s3] rectangle 1160x600
  1. sidebar  [s3-rail] rectangle 240x598 · fill #f8f7f3
    1.1. text "No documents indexed yet."  [s3-rail-empty] text 190x18 · 13px
  2. header  [s3-bar] rectangle 918x72 · fill #f8f7f3
    2.1. text "Library"  [s3-crumbs] text 360x21 · 15px
    2.2. button "Grid"  [s3-grid] rectangle 72x40 · fill #ffffff
    2.3. button "List"  [s3-list] rectangle 72x40 · fill #ffffff
    2.4. button "Summary"  [s3-summary] rectangle 104x40 · fill #e4e1f5
    2.5. button "Add documents"  [s3-add] rectangle 180x40 · fill #4a3fa6
  3. heading "Summary of your library"  [s3-title] text 600x35 · 26px
  4. text "0 passages · 0 pages"  [s3-context] text 700x18 · 13px
  5. text "Counted at 14:02. Nothing here changes your lib…"  [s3-when] text 600x17 · 12px
  6. card  [s3-kpi0] rectangle 420x96 · fill #fbeecb
    6.1. heading "6"  [s3-kpi0-v] text 120x35 · 26px
    6.2. text "Need attention: files. Opens From indexing."  [s3-kpi0-k] text 388x17 · 12px
  7. heading "From indexing"  [s3-idx-h] text 300x28 · 20px
  8. text "Each row: what happened, how many, and where to…"  [s3-idx-d] text 700x17 · 12px
  9. list-item  [s3-row0] rectangle 856x68 · fill #fbeecb
    9.1. icon  [s3-row0-i] rectangle 18x18
    9.2. text "6 files in your documents folder are not in the…"  [s3-row0-t] text 560x21 · 15px
    9.3. text "Indexing stopped before them, or they failed; n…"  [s3-row0-d] text 560x17 · 12px
    9.4. text "Index them in Settings > Documents"  [s3-row0-f] text 210x17 · 12px
    9.5. heading "›"  [s3-row0-c] text 14x28 · 20px
  10. text "Nothing else to count: the library holds no doc…"  [s3-nothing] text 856x18 · 13px
  11. card  [s3-kpi1] rectangle 420x96 · fill #ffffff
    11.1. heading "0"  [s3-kpi1-v] text 160x35 · 26px
    11.2. text "Documents"  [s3-kpi1-k] text 200x17 · 12px
  12. button "Count again"  [s3-recount] rectangle 140x40

### Screen 4 "Summary of a library indexed only in the app"  [s4] rectangle 1160x1312
  1. sidebar  [s4-rail] rectangle 240x1310 · fill #f8f7f3
    1.1. button "All documents 140"  [s4-rail-all] rectangle 208x36 · fill #e4e1f5
    1.2. text "Folders"  [s4-rail-fh] text 80x17 · 12px
    1.3. button "Manage…"  [s4-rail-manage] rectangle 74x28
  2. header  [s4-bar] rectangle 918x72 · fill #f8f7f3
    2.1. text "Library"  [s4-crumbs] text 360x21 · 15px
    2.2. button "Grid"  [s4-grid] rectangle 72x40 · fill #ffffff
    2.3. button "List"  [s4-list] rectangle 72x40 · fill #ffffff
    2.4. button "Summary"  [s4-summary] rectangle 104x40 · fill #e4e1f5
    2.5. button "Add documents"  [s4-add] rectangle 180x40 · fill #4a3fa6
  3. heading "Summary of a library indexed only in the app"  [s4-title] text 600x35 · 26px
  4. text "51,200 passages · 3,480 pages in 140 paged docu…"  [s4-context] text 700x18 · 13px
  5. text "Counted at 14:02. Nothing here changes your lib…"  [s4-when] text 600x17 · 12px
  6. card  [s4-kpi0] rectangle 420x96 · fill #fbeecb
    6.1. heading "12"  [s4-kpi0-v] text 120x35 · 26px
    6.2. text "Need attention: files. Opens From indexing."  [s4-kpi0-k] text 388x17 · 12px
  7. heading "From indexing"  [s4-idx-h] text 300x28 · 20px
  8. text "Each row: what happened, how many, and where to…"  [s4-idx-d] text 700x17 · 12px
  9. list-item  [s4-row0] rectangle 856x68 · fill #fbeecb
    9.1. icon  [s4-row0-i] rectangle 18x18
    9.2. text "12 files in your documents folder are not in th…"  [s4-row0-t] text 560x21 · 15px
    9.3. text "Never indexed, or indexing failed; no run this …"  [s4-row0-d] text 560x17 · 12px
    9.4. text "Index them in Settings > Documents"  [s4-row0-f] text 210x17 · 12px
    9.5. heading "›"  [s4-row0-c] text 14x28 · 20px
  10. list-item  [s4-row1] rectangle 856x68 · fill #f8f7f3
    10.1. text "0 documents gave no text"  [s4-row1-t] text 560x21 · 15px
    10.2. text "Every document gave text when it was indexed."  [s4-row1-d] text 560x17 · 12px
  11. list-item  [s4-row2] rectangle 856x68 · fill #f8f7f3
    11.1. text "0 extractions were flagged"  [s4-row2-t] text 560x21 · 15px
    11.2. text "None was marked marginal or broken when indexed."  [s4-row2-d] text 560x17 · 12px
  12. card  [s4-card0] rectangle 420x312 · fill #ffffff
    12.1. heading "Metadata"  [s4-card0-h] text 380x28 · 20px
    12.2. text "Said once, because no document has any of the f…"  [s4-card0-d] text 388x17 · 12px
    12.3. card  [s4-card0-r0] rectangle 388x64 · fill #f8f7f3
      12.3.1. text "Not read for 140 documents"  [s4-card0-r0-t] text 364x18 · 13px
      12.3.2. text "Indexing does not read metadata yet. Re-run… > …"  [s4-card0-r0-d] text 364x17 · 12px
    12.4. text "Not 560 missing fields: nothing has looked yet."  [s4-card0-f] text 388x16 · 11px
  13. heading "What is in it"  [s4-what-h] text 300x28 · 20px
  14. text "140 in no folder. Added on 1 Nov 2026."  [s4-what-d] text 856x17 · 12px
  15. card  [s4-chart0] rectangle 420x240 · fill #ffffff
    15.1. heading "Types"  [s4-chart0-h] text 300x28 · 20px
    15.2. chart "Ranked horizontal bars, one hue, value at the t…"  [s4-chart0-p] rectangle 388x172
  16. card  [s4-kpi1] rectangle 420x96 · fill #ffffff
    16.1. heading "140"  [s4-kpi1-v] text 160x35 · 26px
    16.2. text "Documents"  [s4-kpi1-k] text 200x17 · 12px
  17. card  [s4-card1] rectangle 420x312 · fill #ffffff
    17.1. heading "Found in each document"  [s4-card1-h] text 380x28 · 20px
    17.2. text "On record, of 140 (figures: of 140 PDFs)."  [s4-card1-d] text 388x17 · 12px
    17.3. card  [s4-card1-r0] rectangle 388x64 · fill #f8f7f3
      17.3.1. text "References: not looked for in 140"  [s4-card1-r0-t] text 364x18 · 13px
      17.3.2. text "Indexing does not look. Re-run… > References do…"  [s4-card1-r0-d] text 364x17 · 12px
    17.4. card  [s4-card1-r1] rectangle 388x64 · fill #f8f7f3
      17.4.1. text "Figures: not looked for in 140 PDFs"  [s4-card1-r1-t] text 364x18 · 13px
      17.4.2. text "Indexing does not look. Re-run… > Figures does."  [s4-card1-r1-d] text 364x17 · 12px
    17.5. list-item  [s4-card1-r2] rectangle 388x44 · fill #ffffff
      17.5.1. text "Spacing"  [s4-card1-r2-n] text 96x20 · 14px
      17.5.2. text "6 of 140"  [s4-card1-r2-v] text 110x18 · 13px
      17.5.3. chart  [s4-card1-r2-track] rectangle 90x8 · fill #efece5
        17.5.3.1. chart  [s4-card1-r2-bar] rectangle 4x8 · fill #4a3fa6
      17.5.4. text "6 open"  [s4-card1-r2-tail] text 64x17 · 12px
      17.5.5. heading "›"  [s4-card1-r2-c] text 12x28 · 20px
    17.6. text "A row of bars returns once some document has so…"  [s4-card1-f] text 388x16 · 11px
  18. card  [s4-chart1] rectangle 420x240 · fill #ffffff
    18.1. heading "Years"  [s4-chart1-h] text 300x28 · 20px
    18.2. chart "No document has a year: one sentence, no chart"  [s4-chart1-p] rectangle 388x172
  19. button "Count again"  [s4-recount] rectangle 140x40

### Navigation (2)
  button "Thesis chapter 2 24" [s1-rail-f0] → screen "Summary of Thesis chapter 2" [s2]   (arrow [f-folder])
  list-item [s2-to-library] → screen "Summary of your library" [s1]   (arrow [f-library])
```
