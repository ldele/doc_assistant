<!-- status: active · updated: 2026-11-04 · class: living -->

# Wireframe: the indexing screen

The scene is `indexing.excalidraw` beside this file; open it on the canvas with
`excalidraw-canvas import docs/wireframes/indexing.excalidraw`, edit, and export it back over the
same file. Four screens, approved by the user on 2026-11-01 with no markup, at rev 858: Settings >
Documents during a run, the run in progress, paused, and stopped. Built 2026-11-04 (DEVLOG
2026-11-04 (2)), with three changes the user asked for while it was driven: *Stop* is red and asks
first, *Pause* is amber, and the status bar reads *Indexing… 14/30 done*. The drawing is left as
approved.

**What the drawing decides.**

- Settings closes during a run, by Close, Escape or a click outside it. *View indexing* and the
  status bar chip both open the screen, and closing the screen leaves the run going.
- Every file has a row: its name, a status and what happened. A failure shows its reason. Rows in
  progress and failed rows come first.
- *Pause* and *Stop* sit in the header and wait for the document in progress to finish, because
  documents run one at a time.
- *Stop* has no confirmation. It loses nothing: finished documents stay and the next run skips them.
  **Superseded by the user, 2026-11-04:** Stop is red and asks first.
- A stopped run offers *Index the remaining 15* and *Retry the 2 that failed*. A run that reaches
  the end looks the same, without the first.
- The title counts failed files as done, as the app's label does: 12 finished and 2 failed read
  *14 of 30*.

**What the backend needs before this can be built.** Status reports position only, so it gains each
file's outcome and a failure's reason. Pause and stop are new, and take effect between documents.
Stop leaves the loop rather than raising, so the steps after a run still happen and finished
documents go live. At 10,000 documents the status poll cannot carry the list: it keeps the counts,
and the files come from a paged route.

The file names and counts are examples, not a real library.

## Wireframe

Score at rev 858: `fallbacks 0 · inferred 0 · unnamedScreens 0 · orphans 0`, 4 screens, 164 components.

```text
## Wireframe reading
4 screens, 164 components, 6 connections, 0 annotations.
Roles are inferred from geometry, style and wording — a trailing `?` marks a low-confidence guess. Raw type and size follow each entry, so you can disagree with the reading. Numbering is reading order (top-to-bottom, left-to-right within a row).
### Screen 1 "Settings"  [s1] rectangle 1160x820
  1. header  [s1-hdr] rectangle 1160x72 · fill #f8f7f3
    1.1. heading "Settings"  [s1-title] text 400x34 · 26px
    1.2. button "Close"  [s1-close] rectangle 88x44 · fill #ffffff
  2. sidebar  [s1-rail] rectangle 220x704 · fill #f8f7f3
    2.1. button "Getting started"  [s1-rail0] rectangle 188x44 · fill #ffffff
    2.2. button "Documents"  [s1-rail1] rectangle 188x44 · fill #4a3fa6
    2.3. button "Provider & model"  [s1-rail2] rectangle 188x44 · fill #ffffff
    2.4. button "Retrieval"  [s1-rail3] rectangle 188x44 · fill #ffffff
    2.5. button "General"  [s1-rail4] rectangle 188x44 · fill #ffffff
  3. heading "Your documents"  [s1-docs] text 400x26 · 20px
  4. text "Library folder"  [s1-folder-l] text 300x18 · 12px
  5. input  [s1-folder] rectangle 700x44 · fill #ffffff
    5.1. text "C:\Users\you\Documents\Papers"  [s1-folder-v] text 660x18 · 13px
  6. button "Indexing…"  [s1-index] rectangle 160x44 · fill #efece5
  7. card  [s1-run] rectangle 876x116 · fill #ffffff
    7.1. text "Indexing 14 of 30"  [s1-run-t] text 500x22 · 16px
    7.2. text "You can close Settings. Indexing keeps running;…"  [s1-run-d] text 640x20 · 13px
    7.3. button "View indexing"  [view-indexing] rectangle 166x44 · fill #4a3fa6
    7.4. text "Close, Escape and a click outside this panel al…"  [s1-run-d2] text 640x20 · 13px
  8. heading "Manage files"  [s1-manage] text 500x26 · 20px
  9. table "The existing file list, unchanged. Ingest selec…"  [s1-files] rectangle 876x306
  10. button "Browse…"  [s1-browse] rectangle 160x44 · fill #ffffff
  11. footer  [s1-ftr] rectangle 1160x44 · fill #f8f7f3
    11.1. text "Status bar, on every screen of the app"  [s1-ftr-t] text 500x20 · 12px
    11.2. button "Indexing 14 of 30"  [chip] rectangle 248x32 · fill #ffffff
### Screen 2 "Indexing 14 of 30"  [s2] rectangle 1160x820
  1. header  [s2-hdr] rectangle 1160x112 · fill #f8f7f3
    1.1. heading "Indexing 14 of 30"  [s2-title] text 500x34 · 26px
    1.2. text "From C:\Users\you\Documents\Papers"  [s2-sub] text 500x20 · 13px
    1.3. button "Pause"  [pause] rectangle 112x44 · fill #ffffff
    1.4. button "Stop"  [stop] rectangle 112x44 · fill #ffffff
    1.5. button "Close"  [s2-close] rectangle 88x44 · fill #ffffff
  2. chart "Progress bar: 14 of 30 documents done, filled f…"  [s2-bar] rectangle 1096x28
  3. button "All 30"  [s2-f0] rectangle 96x40 · fill #4a3fa6
  4. button "In progress 1"  [s2-f1] rectangle 136x40 · fill #ffffff
  5. button "Waiting 15"  [s2-f2] rectangle 120x40 · fill #ffffff
  6. button "Finished 12"  [s2-f3] rectangle 124x40 · fill #ffffff
  7. button "Failed 2"  [s2-f4] rectangle 108x40 · fill #ffffff
  8. text "Pause and Stop take effect when the document in…"  [s2-when] text 468x18 · 12px
  9. text "File"  [s2-col-file] text 400x20 · 12px
  10. text "Status"  [s2-col-status] text 120x20 · 12px
  11. text "What happened"  [s2-col-detail] text 460x20 · 12px
  12. divider  [s2-col-rule] rectangle 1096x2 · fill #e5e1d8
  13. list-item  [s2-row0] rectangle 1096x52 · fill #ffffff
    13.1. text "attention-is-all-you-need.pdf"  [s2-row0-name] text 436x20 · 14px
    13.2. text "Reading"  [s2-row0-st] rectangle 120x30 · fill #e4e1f5
    13.3. text "Reading the text. Its passages are stored next."  [s2-row0-d] text 476x20 · 13px
  14. list-item  [s2-row1] rectangle 1096x52 · fill #ffffff
    14.1. text "scanned-report-1998.pdf"  [s2-row1-name] text 436x20 · 14px
    14.2. text "Failed"  [s2-row1-st] rectangle 120x30 · fill #fbdcd8
    14.3. text "No text found: this scan needs OCR."  [s2-row1-d] text 476x20 · 13px
  15. list-item  [s2-row2] rectangle 1096x52 · fill #ffffff
    15.1. text "meeting-notes.docx"  [s2-row2-name] text 436x20 · 14px
    15.2. text "Failed"  [s2-row2-st] rectangle 120x30 · fill #fbdcd8
    15.3. text "The file is password-protected."  [s2-row2-d] text 476x20 · 13px
  16. list-item  [s2-row3] rectangle 1096x52 · fill #ffffff
    16.1. text "chapter-03.epub"  [s2-row3-name] text 436x20 · 14px
    16.2. text "Waiting"  [s2-row3-st] rectangle 120x30 · fill #f8f7f3
  17. list-item  [s2-row4] rectangle 1096x52 · fill #ffffff
    17.1. text "notes-on-retrieval.md"  [s2-row4-name] text 436x20 · 14px
    17.2. text "Waiting"  [s2-row4-st] rectangle 120x30 · fill #f8f7f3
  18. list-item  [s2-row5] rectangle 1096x52 · fill #ffffff
    18.1. text "field-guide.pdf"  [s2-row5-name] text 436x20 · 14px
    18.2. text "Added"  [s2-row5-st] rectangle 120x30 · fill #dcefe1
    18.3. text "412 passages"  [s2-row5-d] text 476x20 · 13px
  19. list-item  [s2-row6] rectangle 1096x52 · fill #ffffff
    19.1. text "survey-2024.pdf"  [s2-row6-name] text 436x20 · 14px
    19.2. text "Unchanged"  [s2-row6-st] rectangle 120x30 · fill #efece5
    19.3. text "Already in your library"  [s2-row6-d] text 476x20 · 13px
  20. footer  [s2-ftr] rectangle 1160x92 · fill #f8f7f3
    20.1. text "You can close this screen. Indexing keeps runni…"  [s2-ftr-t] text 1096x20 · 13px
### Screen 3 "Paused at 15 of 30"  [s3] rectangle 1160x820
  1. header  [s3-hdr] rectangle 1160x112 · fill #f8f7f3
    1.1. heading "Paused at 15 of 30"  [s3-title] text 500x34 · 26px
    1.2. text "From C:\Users\you\Documents\Papers"  [s3-sub] text 500x20 · 13px
    1.3. button "Resume"  [resume] rectangle 112x44 · fill #4a3fa6
    1.4. button "Stop"  [stop3] rectangle 112x44 · fill #ffffff
    1.5. button "Close"  [s3-close] rectangle 88x44 · fill #ffffff
  2. card  [s3-banner] rectangle 1096x40 · fill #fbeecb
    2.1. text "Paused. Searching and chatting work as usual; n…"  [s3-banner-t] text 1060x20 · 13px
  3. button "All 30"  [s3-f0] rectangle 96x40 · fill #4a3fa6
  4. button "In progress 0"  [s3-f1] rectangle 136x40 · fill #ffffff
  5. button "Waiting 15"  [s3-f2] rectangle 120x40 · fill #ffffff
  6. button "Finished 13"  [s3-f3] rectangle 124x40 · fill #ffffff
  7. button "Failed 2"  [s3-f4] rectangle 108x40 · fill #ffffff
  8. text "Quitting the app ends the run. What is finished…"  [s3-when] text 468x18 · 12px
  9. text "File"  [s3-col-file] text 400x20 · 12px
  10. text "Status"  [s3-col-status] text 120x20 · 12px
  11. text "What happened"  [s3-col-detail] text 460x20 · 12px
  12. divider  [s3-col-rule] rectangle 1096x2 · fill #e5e1d8
  13. list-item  [s3-row0] rectangle 1096x52 · fill #ffffff
    13.1. text "attention-is-all-you-need.pdf"  [s3-row0-name] text 436x20 · 14px
    13.2. text "Added"  [s3-row0-st] rectangle 120x30 · fill #dcefe1
    13.3. text "380 passages"  [s3-row0-d] text 476x20 · 13px
  14. list-item  [s3-row1] rectangle 1096x52 · fill #ffffff
    14.1. text "scanned-report-1998.pdf"  [s3-row1-name] text 436x20 · 14px
    14.2. text "Failed"  [s3-row1-st] rectangle 120x30 · fill #fbdcd8
    14.3. text "No text found: this scan needs OCR."  [s3-row1-d] text 476x20 · 13px
  15. list-item  [s3-row2] rectangle 1096x52 · fill #ffffff
    15.1. text "meeting-notes.docx"  [s3-row2-name] text 436x20 · 14px
    15.2. text "Failed"  [s3-row2-st] rectangle 120x30 · fill #fbdcd8
    15.3. text "The file is password-protected."  [s3-row2-d] text 476x20 · 13px
  16. list-item  [s3-row3] rectangle 1096x52 · fill #ffffff
    16.1. text "chapter-03.epub"  [s3-row3-name] text 436x20 · 14px
    16.2. text "Waiting"  [s3-row3-st] rectangle 120x30 · fill #f8f7f3
  17. list-item  [s3-row4] rectangle 1096x52 · fill #ffffff
    17.1. text "notes-on-retrieval.md"  [s3-row4-name] text 436x20 · 14px
    17.2. text "Waiting"  [s3-row4-st] rectangle 120x30 · fill #f8f7f3
  18. list-item  [s3-row5] rectangle 1096x52 · fill #ffffff
    18.1. text "field-guide.pdf"  [s3-row5-name] text 436x20 · 14px
    18.2. text "Added"  [s3-row5-st] rectangle 120x30 · fill #dcefe1
    18.3. text "412 passages"  [s3-row5-d] text 476x20 · 13px
  19. list-item  [s3-row6] rectangle 1096x52 · fill #ffffff
    19.1. text "survey-2024.pdf"  [s3-row6-name] text 436x20 · 14px
    19.2. text "Unchanged"  [s3-row6-st] rectangle 120x30 · fill #efece5
    19.3. text "Already in your library"  [s3-row6-d] text 476x20 · 13px
  20. footer  [s3-ftr] rectangle 1160x92 · fill #f8f7f3
    20.1. text "You can close this screen. The run stays paused…"  [s3-ftr-t] text 1096x20 · 13px
### Screen 4 "Stopped at 15 of 30"  [s4] rectangle 1160x820
  1. header  [s4-hdr] rectangle 1160x112 · fill #f8f7f3
    1.1. heading "Stopped at 15 of 30"  [s4-title] text 500x34 · 26px
    1.2. text "From C:\Users\you\Documents\Papers"  [s4-sub] text 500x20 · 13px
    1.3. button "Index the remaining 15"  [index-rest] rectangle 224x44 · fill #4a3fa6
    1.4. button "Retry the 2 that failed"  [retry] rectangle 208x44 · fill #ffffff
    1.5. button "Close"  [s4-close] rectangle 88x44 · fill #ffffff
  2. card  [s4-banner] rectangle 1096x40 · fill #f8f7f3
    2.1. text "13 documents are in your library. 15 were not s…"  [s4-banner-t] text 1060x20 · 13px
  3. button "All 30"  [s4-f0] rectangle 96x40 · fill #ffffff
  4. button "Not started 15"  [s4-f1] rectangle 136x40 · fill #ffffff
  5. button "Finished 13"  [s4-f2] rectangle 124x40 · fill #ffffff
  6. button "Failed 2"  [s4-f3] rectangle 108x40 · fill #4a3fa6
  7. text "File"  [s4-col-file] text 400x20 · 12px
  8. text "Status"  [s4-col-status] text 120x20 · 12px
  9. text "What happened"  [s4-col-detail] text 460x20 · 12px
  10. divider  [s4-col-rule] rectangle 1096x2 · fill #e5e1d8
  11. list-item  [s4-row0] rectangle 1096x52 · fill #ffffff
    11.1. text "scanned-report-1998.pdf"  [s4-row0-name] text 436x20 · 14px
    11.2. text "Failed"  [s4-row0-st] rectangle 120x30 · fill #fbdcd8
    11.3. text "No text found: this scan needs OCR."  [s4-row0-d] text 476x20 · 13px
  12. list-item  [s4-row1] rectangle 1096x52 · fill #ffffff
    12.1. text "meeting-notes.docx"  [s4-row1-name] text 436x20 · 14px
    12.2. text "Failed"  [s4-row1-st] rectangle 120x30 · fill #fbdcd8
    12.3. text "The file is password-protected."  [s4-row1-d] text 476x20 · 13px
  13. list-item  [s4-row2] rectangle 1096x52 · fill #ffffff
    13.1. text "chapter-03.epub"  [s4-row2-name] text 436x20 · 14px
    13.2. text "Not started"  [s4-row2-st] rectangle 120x30 · fill #f8f7f3
  14. list-item  [s4-row3] rectangle 1096x52 · fill #ffffff
    14.1. text "notes-on-retrieval.md"  [s4-row3-name] text 436x20 · 14px
    14.2. text "Not started"  [s4-row3-st] rectangle 120x30 · fill #f8f7f3
  15. list-item  [s4-row4] rectangle 1096x52 · fill #ffffff
    15.1. text "attention-is-all-you-need.pdf"  [s4-row4-name] text 436x20 · 14px
    15.2. text "Added"  [s4-row4-st] rectangle 120x30 · fill #dcefe1
    15.3. text "380 passages"  [s4-row4-d] text 476x20 · 13px
  16. list-item  [s4-row5] rectangle 1096x52 · fill #ffffff
    16.1. text "field-guide.pdf"  [s4-row5-name] text 436x20 · 14px
    16.2. text "Added"  [s4-row5-st] rectangle 120x30 · fill #dcefe1
    16.3. text "412 passages"  [s4-row5-d] text 476x20 · 13px
  17. list-item  [s4-row6] rectangle 1096x52 · fill #ffffff
    17.1. text "survey-2024.pdf"  [s4-row6-name] text 436x20 · 14px
    17.2. text "Unchanged"  [s4-row6-st] rectangle 120x30 · fill #efece5
    17.3. text "Already in your library"  [s4-row6-d] text 476x20 · 13px
  18. footer  [s4-ftr] rectangle 1160x92 · fill #f8f7f3
    18.1. text "A run that reaches the end looks the same, with…"  [s4-ftr-t] text 1096x20 · 13px
### Navigation (6)
  button "View indexing" [view-indexing] → screen "Indexing 14 of 30" [s2]   (arrow [f-view])
  button "Indexing 14 of 30" [chip] → screen "Indexing 14 of 30" [s2]   (arrow [f-chip])
  button "Pause" [pause] → screen "Paused at 15 of 30" [s3]   (arrow [f-pause])
  button "Resume" [resume] → screen "Indexing 14 of 30" [s2]   (arrow [f-resume])
  button "Stop" [stop] → screen "Stopped at 15 of 30" [s4]   (arrow [f-stop])
  button "Stop" [stop3] → screen "Stopped at 15 of 30" [s4]   (arrow [f-stop3])
```
