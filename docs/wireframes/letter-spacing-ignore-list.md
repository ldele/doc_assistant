<!-- status: active · updated: 2026-11-01 · class: living -->

# Wireframe: the letter-spacing ignore list (ADR-056 point 5)

The scene is `letter-spacing-ignore-list.excalidraw` beside this file; open it on the canvas with
`excalidraw-canvas import docs/wireframes/letter-spacing-ignore-list.excalidraw`, edit, and export
it back over the same file. Version 3 (2026-10-24): three screens. The document page lists the
runs extraction closed and the ones kept, each row a crop of the page beside the closed text, with
Show on page, Edit spacing, and Keep spaced or Remove. Settings > Documents lists every entry, with
Show runs, Remove, and Remove all with Undo. The Edit spacing dialog sets a run's word breaks;
only spaces change. The example data are Cajal's real runs (page 1's title line, page 2's running
head); the Settings rows show the shape of a list, not this library's list, which starts empty.

**Built 2026-10-24, and where the app differs from the sketch.** A row's buttons sit under its
caption, not beside it (beside, they left the line a few characters wide in the document pane),
and rows stack when the block is narrow. The second group is titled *Kept or spaced your way*: it
holds both. The Edit spacing dialog shows the page crop above the letters. The example rows' counts
were wrong in the first reading: the title loses 15 spaces and the running head 26, not 14 and 29;
the scene and this reading are corrected.

**Reading quality at canvas revision 527:** 3 screens, 81 components, fallbacks 0, inferred 0,
unnamed screens 0, orphans 0 (`excalidraw-canvas wireframe --score`). One markup round: the user
accepted version 1 as drawn, then asked for more options than Remove ("Yes and maybe edit ?") and
for a better way to show spaced against closed; versions 2 and 3 answer those.

## Reading

```text
3 screens, 81 components, 2 connections, 0 annotations.

Roles are inferred from geometry, style and wording — a trailing `?` marks a low-confidence guess. Raw type and size follow each entry, so you can disagree with the reading. Numbering is reading order (top-to-bottom, left-to-right within a row).

### Screen 1 "The structure and connexions of neurons"  [s-doc] rectangle 1160x820
  1. header  [doc-nav] rectangle 1160x64 · fill #fbfaf8
    1.1. button "Metadata"  [doc-nav-0] rectangle 120x40 · fill #ffffff
    1.2. button "Connections"  [doc-nav-1] rectangle 120x40 · fill #ffffff
    1.3. button "Chunks"  [doc-nav-2] rectangle 120x40 · fill #ffffff
    1.4. button "Figures"  [doc-nav-3] rectangle 120x40 · fill #ffffff
    1.5. button "References"  [doc-nav-4] rectangle 120x40 · fill #ffffff
    1.6. button "Source"  [doc-source] rectangle 120x40 · fill #ffffff
    1.7. button "Re-run…"  [doc-rerun] rectangle 140x40 · fill #ffffff
  2. heading "The structure and connexions of neurons"  [doc-title] text 620x36 · 26px
  3. text "cajal-lecture.pdf · pdf · 34 pages · healthy"  [doc-meta] text 460x18 · 13px
  4. text "Authors · Year · Keywords lines, as today"  [doc-meta2] text 420x18 · 13px
  5. card  [doc-ls] rectangle 1096x434 · fill #ffffff
    5.1. heading "Letter-spacing on this document"  [doc-ls-h] text 420x28 · 20px
    5.2. text "The letter-spaced runs extraction closed here, …"  [doc-ls-d] text 1040x18 · 13px
    5.3. text "Closed · 17 runs"  [doc-closed-h] text 200x22 · 16px
    5.4. list-item  [doc-c1] rectangle 1060x64 · fill #ffffff
      5.4.1. image "crop: page 1, the title line"  [doc-c1-crop] rectangle 200x48
      5.4.2. text "SANTIAGO RAMÓN Y CAJAL"  [doc-c1-t] text 360x21 · 15px
      5.4.3. text "15 spaces removed, a tick at each · page 1"  [doc-c1-c] text 360x16 · 12px
      5.4.4. button "Show on page"  [doc-c1-show] rectangle 130x40 · fill #ffffff
      5.4.5. button "Edit spacing"  [doc-c1-edit] rectangle 130x40 · fill #ffffff
      5.4.6. button "Keep spaced"  [doc-c1-keep] rectangle 130x40 · fill #ffffff
    5.5. list-item  [doc-c2] rectangle 1060x64 · fill #ffffff
      5.5.1. image "crop: page 2, the running head"  [doc-c2-crop] rectangle 200x48
      5.5.2. text "NEURONS: STRUCTURE AND CONNEXIONS"  [doc-c2-t] text 360x21 · 15px
      5.5.3. text "26 spaces removed, a tick at each · 16 pages"  [doc-c2-c] text 360x16 · 12px
      5.5.4. button "Show on page"  [doc-c2-show] rectangle 130x40 · fill #ffffff
      5.5.5. button "Edit spacing"  [doc-c2-edit] rectangle 130x40 · fill #ffffff
      5.5.6. button "Keep spaced"  [doc-c2-keep] rectangle 130x40 · fill #ffffff
    5.6. text "Kept as extracted · none"  [doc-kept-h] text 280x22 · 16px
    5.7. text "A run you keep spaced appears here, with Show o…"  [doc-kept-d] text 640x18 · 13px
    5.8. button "Keep every run spaced"  [doc-ls-keepall] rectangle 230x40 · fill #ffffff
    5.9. button "Re-run text now"  [doc-ls-rerun] rectangle 180x40 · fill #ffffff
    5.10. button "All entries in Settings"  [doc-ls-settings] rectangle 240x40 · fill #ffffff
  6. card "Connections · Chunks · Figures · References blo…"  [doc-blocks] rectangle 1096x92 · fill #ffffff

### Screen 2 "Settings"  [s-set] rectangle 1160x860
  1. sidebar  [set-rail] rectangle 240x860 · fill #fbfaf8
    1.1. heading "Settings"  [set-title] text 160x36 · 26px
    1.2. button "Setup"  [set-rail-0] rectangle 208x40 · fill #ffffff
    1.3. button "Documents"  [set-rail-1] rectangle 208x40 · fill #74c0fc
    1.4. button "Provider & model"  [set-rail-2] rectangle 208x40 · fill #ffffff
    1.5. button "Answers"  [set-rail-3] rectangle 208x40 · fill #ffffff
    1.6. button "Display"  [set-rail-4] rectangle 208x40 · fill #ffffff
    1.7. button "Updates"  [set-rail-5] rectangle 208x40 · fill #ffffff
  2. text "Where your documents live, what is indexed, and…"  [set-blurb] text 600x18 · 13px
  3. card "Your documents (as today)"  [set-sec-0] rectangle 856x56 · fill #ffffff
  4. card "Manage files (selective indexing) (as today)"  [set-sec-1] rectangle 856x56 · fill #ffffff
  5. card "Corpus (as today)"  [set-sec-2] rectangle 856x56 · fill #ffffff
  6. card  [set-ls] rectangle 856x556 · fill #ffffff
    6.1. heading "Letter-spacing ignore list"  [set-ls-h] text 400x28 · 20px
    6.2. text "Runs you keep as extracted or spaced your own w…"  [set-ls-d] text 820x18 · 13px
    6.3. list-item  [set-e1] rectangle 820x52 · fill #ffffff
      6.3.1. text "Computational_neuroanatomy.pdf · 10 runs kept"  [set-e1-t] text 440x19 · 14px
      6.3.2. button "Show runs"  [set-e1-show] rectangle 120x40 · fill #ffffff
      6.3.3. button "Remove"  [set-e1-rm] rectangle 120x40 · fill #ffffff
    6.4. list-item  [set-e2] rectangle 820x52 · fill #ffffff
      6.4.1. text "elife-78635-v2.pdf · 1 run kept"  [set-e2-t] text 440x19 · 14px
      6.4.2. button "Show runs"  [set-e2-show] rectangle 120x40 · fill #ffffff
      6.4.3. button "Remove"  [set-e2-rm] rectangle 120x40 · fill #ffffff
    6.5. list-item  [set-e3] rectangle 820x52 · fill #ffffff
      6.5.1. text "cajal-lecture.pdf · 1 run spaced your way"  [set-e3-t] text 440x19 · 14px
      6.5.2. button "Show runs"  [set-e3-show] rectangle 120x40 · fill #ffffff
      6.5.3. button "Remove"  [set-e3-rm] rectangle 120x40 · fill #ffffff
    6.6. button "Remove all"  [set-ls-all] rectangle 160x40 · fill #ffffff
    6.7. card  [set-undo] rectangle 820x112 · fill #fbfaf8
      6.7.1. text "Removed 12 entries. They take effect when the 3…"  [set-undo-t] text 640x19 · 14px
      6.7.2. button "Undo"  [set-undo-btn] rectangle 120x40 · fill #74c0fc
      6.7.3. button "Re-run text now"  [set-undo-rerun] rectangle 180x40 · fill #ffffff
    6.8. text "Empty list: one line, "No runs are kept as extr…"  [set-empty] text 520x18 · 13px

### Screen 3 "Edit spacing"  [s-edit] rectangle 1160x460
  1. heading "Edit spacing"  [edit-title] text 260x36 · 26px
  2. text "Choose where the words break. The letters stay …"  [edit-d] text 760x18 · 13px
  3. text "cajal-lecture.pdf · page 1"  [edit-src] text 300x18 · 13px
  4. input "SANTIAGO RAMÓN Y CAJAL (a tick at each of the 1…"  [edit-strip] rectangle 1096x64 · fill #ffffff
  5. text "Reads as"  [edit-prev-k] text 100x18 · 13px
  6. text "SANTIAGO RAMÓN Y CAJAL"  [edit-prev] text 300x22 · 16px
  7. text "Saved to the ignore list as this run, spaced yo…"  [edit-note] text 760x16 · 12px
  8. button "Save"  [edit-save] rectangle 140x40 · fill #74c0fc
  9. button "Back to the rule's reading"  [edit-reset] rectangle 240x40 · fill #ffffff
  10. button "Cancel"  [edit-cancel] rectangle 120x40 · fill #ffffff

### Navigation (2)
  button "All entries in Settings" [doc-ls-settings] → screen "Settings" [s-set]   (arrow [flow-1])
  button "Edit spacing" [doc-c1-edit] → screen "Edit spacing" [s-edit]   (arrow [flow-2])
```
