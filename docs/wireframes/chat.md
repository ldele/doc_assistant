<!-- status: active · updated: 2026-11-20 (two labels replaced by the A8 spec, G3 and G4; revision 3 approved by the user in chat; revisions 1 and 2 marked up and replaced the same day) · class: living -->

# Wireframe: the chat page (A8, ROADMAP row 103)

The scene is `chat.excalidraw` beside this file. Open it on the canvas with
`excalidraw-canvas import docs/wireframes/chat.excalidraw`, edit it, and export it back over the
same file. The code-level contract is the A8 spec (`docs/specs/chat-interface-pass.md`, to be
written from this drawing).

**Three revisions, all 2026-11-20.**
- **Revision 1** put a mode picker and a folder picker in the composer. The user circled both
  and said they get in the way: most chats never change either.
- **Revision 2** moved both into a setup card shown when a chat starts. The mode is fixed for the
  chat, and *Continue in a new chat* replaces changing it. The rail gained a *Group by* control.
- **Revision 3**, drawn here, keeps revision 2's model and follows the user's three corrections:
  - the rail stays as it is today;
  - the options go on the start screen *New chat* already opens;
  - the defaults sit in a small window beside *New chat*, separate from Settings.

  The user approved it in chat ("I approve we can go with this") rather than by ticking the
  canvas boxes, so the reading below says `not approved yet` on every screen.

## The screens

1. **An answered turn, its source open** (`[s1]`)
2. **New chat: the options on the start screen** (`[s2]`)
3. **New chat: choosing where it searches** (`[s3]`)
4. **The defaults New chat starts from** (`[s4]`)
5. **Continuing in a new chat** (`[s5]`)
6. **The new chat, continued from another** (`[s6]`)
7. **A question answered from the records** (`[s7]`)
8. **What the answer carried** (`[s8]`)
9. **The chat list: grouped by mode, filtered by what it cited** (`[s9]`)
10. **A past chat, reopened and continued** (`[s10]`)

## What the drawing decides

- **A chat's mode and folders are set when it starts, and shown for its whole life.**
  - **The start screen.** It gains mode chips (`[s2-mode0..2]`) and *Search in*
    (`[s2-scope]`), which opens the folder list (`[s3-pop]`).
  - **The folder list.** *All documents* is set apart, then one tickbox per folder, and the
    count says how many documents are searched, each once (T-015).
  - **The sample questions** follow the chosen mode (`[s2-sample*]`, `[s3-sample*]`).
  - **After the first question**, the choice shows in a fixed line at the top of the chat
    (`[s1-chatline]`).
- **The composer holds only the text box and *Send*.** Neither picker stays in it.
- **Changing the mode or the folders means a new chat.** *Continue in a new chat…*
  (`[s1-continue-new]`) opens `[s5]`.
  - **The dialog** has the same choices, prefilled, and *Carry the last 4 exchanges, as they
    were*. They are fitted to the window like any earlier message (A5), never summarised.
  - **The new chat** says where it came from, with the carried exchanges folded at its top
    (`[s6-from]`, `[s6-carried]`).
  - **The old chat** stays as it was.
- **Defaults.** The ⚙ beside *New chat* (`[s4-defaults]`) opens *New chat defaults*: a default
  mode and default folders, kept apart from Settings. *Make this my default*
  (`[s2-make-default]`) writes to the same place.
- **The rail is unchanged.** Sorting and grouping live in the ⇅ menu (`[s9-menu]`): *Newest
  first · Oldest first*, then *No groups · By mode · By folder*.
  - **Filter by citation.** *Only chats that cited* takes a folder or a document (A7b's route).
  - **Labels.** Each row's date line gains what its answers cited, read from the answer
    records each time.
- **Claims in the answer** (row 42, `[s1-c1..c3]`).
  - **Lilac:** a claim matched to the passage it cites; clicking it opens that passage in the
    source panel.
  - **Dark lilac:** the claim whose passage is open now.
  - **Amber:** a claim the reviewer did not find in its source.
  - **No colour:** no claim was matched, and nothing is said about it.
- **A reopened chat** (row 44, `[s10]`) keeps its claims and the reviewer's verdicts as
  recorded. New turns run under the same mode and folders, with today's library.
- **A records answer** (A6, `[s7]`) says that no model and no passages were used, and lists
  what it looked up, each row opening the document.
- **What the answer carried** (A5, `[s8]`) sits inside *How this answer was made*: the earlier
  messages and passages kept, shortened and dropped, the window and where its size came from,
  and the mode.

## Changed after approval, in the spec's grill

Two labels in the drawing were replaced by the A8 spec (`docs/specs/chat-interface-pass.md`, G3
and G4), and the spec wins where they differ:
- `[s5]` carries **the whole chat**, fitted to the window, not "the last 4 exchanges".
- **Amber** (`[s1-c3]`, `[s10-c2]`) means a claim that cites a passage which ranked low for the
  question. It does not mean "the reviewer did not find it in its source", a check the app does
  not run.

## What this changes in earlier decisions (for the A8 spec to record)

- **A7, decision 3** (`docs/specs/chat-modes.md`): `/mode` switched an open chat from its next
  turn. Here a chat's mode is fixed, and switching means continuing in a new chat.
- **ADR-025 S9** (`docs/specs/feature-corpus-folders-scope.md`): the folder choice was never
  stored, so a forgotten scope could not narrow later answers. Here it is stored on the chat, and
  the safeguard is the line at the top of the chat that always shows it. It needs an amendment.
  A one-off question elsewhere stays `/folder <name>: <question>` (A7b).

## Not drawn, parked

- An editor for a person's own modes. A7 built the routes, but there is no screen for them.
- The per-chat "no memory" switch (the user, 2026-11-12). It is in no spec yet.
- Night mode on the folder choice (T-015) is a styling fix, made in code and checked in both
  themes. There is nothing to draw for it.

## Wireframe

Reading at canvas rev 5283: `screens 10 · components 446 · fallbacks 0 · inferred 0 ·
unnamedScreens 0 · orphans 0`. `wireframe --lint`: no errors, and three advisories, all
`sibling-overlap` where a popover or dialog covers the page on purpose (`[s3-pop]`, `[s5-dlg]`).

## Wireframe reading

```text
## Wireframe reading
10 screens, 446 components, 0 connections, 0 annotations.
Approved: 0 of 10 screens.

Roles are inferred from geometry, style and wording — a trailing `?` marks a low-confidence guess. Raw type and size follow each entry, so you can disagree with the reading. Numbering is reading order (top-to-bottom, left-to-right within a row).

Left out: the markup legend [legend] (12 elements), the key to the markup colours rather than part of the design.

### Screen 1 "S1 — An answered turn, its source open"  [s1] frame 1160x920 · not approved yet
  1. header  [s1-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s1-i-menu] rectangle 16x16
    1.2. icon  [s1-i-cog] rectangle 16x16
    1.3. icon  [s1-i-side] rectangle 16x16
    1.4. icon  [s1-i-search] rectangle 16x16
    1.5. icon  [s1-i-back] rectangle 16x16
    1.6. icon  [s1-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s1-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s1-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s1-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s1-brand] text 83x21 · 15px
  2. sidebar  [s1-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s1-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s1-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s1-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s1-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s1-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s1-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s1-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s1-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s1-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s1-row3] rectangle 262x44 · fill #f8f7f3
  3. text "Critical reading · whole library"  [s1-chatline] rectangle 252x36 · fill #f8f7f3
  4. button "Continue in a new chat…"  [s1-continue-new] rectangle 224x36 · fill #ffffff
  5. text "You"  [s1-you] text 28x17 · 12px
  6. text "How does dopamine signalling change with age?"  [s1-q] text 456x21 · 15px
  7. button "Dopamine release in the striatum falls with age…"  [s1-c1] rectangle 456x40 · fill #e9e6fa
  8. text "Most of the sources measure this in rodents; tw…"  [s1-plain1] text 456x20 · 14px
  9. button "Receptor density declines by about 6–10% per de…"  [s1-c2] rectangle 456x40 · fill #d9d4f7
  10. button "This decline causes the motor slowing seen in a…"  [s1-c3] rectangle 456x40 · fill #fbeecb
  11. text "Lilac: a claim matched to the passage it cites;…"  [s1-hl-key] text 456x48 · 12px
  12. button "Sources (3) ▾"  [s1-sources-toggle] rectangle 456x36 · fill #ffffff
  13. list-item "[1] Smith 2021 · Striatal release across the li…"  [s1-src0] rectangle 456x36 · fill #ffffff
  14. list-item "[2] Lee 2019 · Ageing and the dopamine system ·…"  [s1-src1] rectangle 456x36 · fill #ffffff
  15. list-item "[3] Ortiz 2020 · Motor ageing in mice · p. 7 ⚠ …"  [s1-src2] rectangle 456x36 · fill #ffffff
  16. button "How this answer was made ▸"  [s1-prov-toggle] rectangle 456x36 · fill #ffffff
  17. card  [s1-composer] rectangle 488x64 · fill #ffffff
    17.1. input "Ask your documents…"  [s1-ask] rectangle 352x44 · fill #ffffff
    17.2. button "Send"  [s1-send] rectangle 92x36 · fill #4a3fa6
  18. card  [s1-panel] rectangle 360x872 · fill #ffffff
    18.1. heading "Source 2"  [s1-panel-h] text 97x28 · 20px
    18.2. icon  [s1-panel-close] rectangle 20x20
    18.3. text "Lee 2019 · Ageing and the dopamine system p. 12…"  [s1-panel-meta] text 320x35 · 13px
    18.4. card  [s1-passage] rectangle 320x300 · fill #f8f7f3
      18.4.1. text "…striatal D2 binding was measured in 40 adults …"  [s1-pass-a] text 288x35 · 13px
      18.4.2. text "Receptor density declined by 6 to 10 percent pe…"  [s1-pass-hl] rectangle 288x64 · fill #d9d4f7
      18.4.3. text "The decline was steepest in the caudate, and di…"  [s1-pass-b] text 288x35 · 13px
    18.5. button "Open in the viewer"  [s1-open-viewer] rectangle 156x36 · fill #ffffff
    18.6. button "Show in Library"  [s1-show-lib] rectangle 156x36 · fill #ffffff

### Screen 2 "S2 — New chat: the options on the start screen"  [s2] frame 1160x920 · not approved yet
  1. header  [s2-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s2-i-menu] rectangle 16x16
    1.2. icon  [s2-i-cog] rectangle 16x16
    1.3. icon  [s2-i-side] rectangle 16x16
    1.4. icon  [s2-i-search] rectangle 16x16
    1.5. icon  [s2-i-back] rectangle 16x16
    1.6. icon  [s2-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s2-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s2-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s2-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s2-brand] text 83x21 · 15px
  2. sidebar  [s2-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s2-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s2-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s2-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s2-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s2-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s2-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s2-row0] rectangle 262x44 · fill #f8f7f3
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s2-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s2-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s2-row3] rectangle 262x44 · fill #f8f7f3
  3. icon  [s2-st-icon] rectangle 28x28
  4. heading "Ask your library a question"  [s2-st-h] text 310x28 · 20px
  5. text "Every answer is grounded in your own documents,…"  [s2-st-sub] text 380x33 · 12px
  6. button "Research assistant"  [s2-mode0] rectangle 150x32 · fill #e9e6fa
  7. button "Critical reading"  [s2-mode1] rectangle 130x32 · fill #ffffff
  8. button "Plain summary"  [s2-mode2] rectangle 124x32 · fill #ffffff
  9. text "Plain summary is yours: its answers are marked …"  [s2-mode-note] text 420x16 · 11px
  10. text "Search in"  [s2-scope-l] text 68x17 · 12px
  11. button "All documents ▾"  [s2-scope] rectangle 260x30 · fill #ffffff
  12. button "What are the main themes across my documents?"  [s2-sample0] rectangle 330x32 · fill #ffffff
  13. button "What are the key findings, with citations?"  [s2-sample1] rectangle 330x32 · fill #ffffff
  14. button "Which documents discuss receptor density?"  [s2-sample2] rectangle 330x32 · fill #ffffff
  15. button "Make this my default"  [s2-make-default] rectangle 220x26 · fill #ffffff
  16. text "Your first question starts the chat with these;…"  [s2-st-foot] text 420x30 · 11px
  17. card  [s2-composer] rectangle 848x64 · fill #ffffff
    17.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s2-ask] rectangle 712x44 · fill #ffffff
    17.2. button "Send"  [s2-send] rectangle 92x36 · fill #4a3fa6

### Screen 3 "S3 — New chat: choosing where it searches"  [s3] frame 1160x920 · not approved yet
  1. header  [s3-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s3-i-menu] rectangle 16x16
    1.2. icon  [s3-i-cog] rectangle 16x16
    1.3. icon  [s3-i-side] rectangle 16x16
    1.4. icon  [s3-i-search] rectangle 16x16
    1.5. icon  [s3-i-back] rectangle 16x16
    1.6. icon  [s3-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s3-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s3-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s3-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s3-brand] text 83x21 · 15px
  2. sidebar  [s3-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s3-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s3-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s3-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s3-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s3-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s3-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s3-row0] rectangle 262x44 · fill #f8f7f3
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s3-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s3-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s3-row3] rectangle 262x44 · fill #f8f7f3
  3. icon  [s3-st-icon] rectangle 28x28
  4. heading "Ask your library a question"  [s3-st-h] text 310x28 · 20px
  5. text "Every answer is grounded in your own documents,…"  [s3-st-sub] text 380x33 · 12px
  6. button "Research assistant"  [s3-mode0] rectangle 150x32 · fill #ffffff
  7. button "Critical reading"  [s3-mode1] rectangle 130x32 · fill #e9e6fa
  8. button "Plain summary"  [s3-mode2] rectangle 124x32 · fill #ffffff
  9. text "Plain summary is yours: its answers are marked …"  [s3-mode-note] text 420x16 · 11px
  10. text "Search in"  [s3-scope-l] text 68x17 · 12px
  11. button "Thesis ch.2 + Methods · 21 ▾"  [s3-scope] rectangle 260x30 · fill #e9e6fa
  12. button "Where do my sources agree and disagree?"  [s3-sample0] rectangle 330x32 · fill #ffffff
  13. button "What are the limits of the evidence on ageing?"  [s3-sample1] rectangle 330x32 · fill #ffffff
  14. button "Which claims rest on a single study?"  [s3-sample2] rectangle 330x32 · fill #ffffff
  15. button "Make this my default"  [s3-make-default] rectangle 220x26 · fill #ffffff
  16. text "Your first question starts the chat with these;…"  [s3-st-foot] text 420x30 · 11px
  17. card  [s3-pop] rectangle 278x300 · fill #ffffff
    17.1. checkbox "All documents · 85"  [s3-all] rectangle 20x20 · fill #ffffff
    17.2. text "Ticking it unticks every folder"  [s3-all-n] text 198x16 · 11px
    17.3. divider  [s3-div] rectangle 250x2 · fill #e5e1d8
    17.4. text "Folders"  [s3-fold-h] text 55x17 · 12px
    17.5. checkbox "Thesis ch.2 · 14"  [s3-cb0] rectangle 20x20 · fill #4a3fa6
    17.6. checkbox "Methods · 9"  [s3-cb1] rectangle 20x20 · fill #4a3fa6
    17.7. checkbox "Reading group · 22"  [s3-cb2] rectangle 20x20 · fill #ffffff
    17.8. checkbox "Archive · 40"  [s3-cb3] rectangle 20x20 · fill #ffffff
    17.9. text "2 folders · 21 documents 2 are in both and are …"  [s3-count] text 243x33 · 12px
  18. card  [s3-composer] rectangle 848x64 · fill #ffffff
    18.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s3-ask] rectangle 712x44 · fill #ffffff
    18.2. button "Send"  [s3-send] rectangle 92x36 · fill #4a3fa6

### Screen 4 "S4 — The defaults New chat starts from"  [s4] frame 1160x920 · not approved yet
  1. header  [s4-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s4-i-menu] rectangle 16x16
    1.2. icon  [s4-i-cog] rectangle 16x16
    1.3. icon  [s4-i-side] rectangle 16x16
    1.4. icon  [s4-i-search] rectangle 16x16
    1.5. icon  [s4-i-back] rectangle 16x16
    1.6. icon  [s4-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s4-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s4-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s4-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s4-brand] text 83x21 · 15px
  2. sidebar  [s4-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s4-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s4-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s4-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s4-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s4-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s4-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s4-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s4-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s4-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s4-row3] rectangle 262x44 · fill #f8f7f3
  3. card  [s4-dlg] rectangle 440x560 · fill #ffffff
    3.1. heading "New chat defaults"  [s4-dlg-h] text 198x28 · 20px
    3.2. text "Every New chat starts with these. Separate from…"  [s4-dlg-sub] text 400x17 · 12px
    3.3. text "Answer as"  [s4-d-m-h] text 68x17 · 12px
    3.4. radio "Research assistant"  [s4-d-r0] ellipse 20x20 · fill #4a3fa6
    3.5. text "Built in · the default"  [s4-d-rd0] text 360x17 · 12px
    3.6. radio "Critical reading"  [s4-d-r1] ellipse 20x20 · fill #ffffff
    3.7. text "Built in · where sources agree, disagree, and t…"  [s4-d-rd1] text 360x17 · 12px
    3.8. radio "Plain summary"  [s4-d-r2] ellipse 20x20 · fill #ffffff
    3.9. text "Yours · its answers are marked not evaluated"  [s4-d-rd2] text 360x17 · 12px
    3.10. divider  [s4-d-div] rectangle 400x2 · fill #e5e1d8
    3.11. text "Search in"  [s4-d-f-h] text 68x17 · 12px
    3.12. checkbox "All documents · 85"  [s4-d-all] rectangle 20x20 · fill #4a3fa6
    3.13. text "unticks every folder"  [s4-d-all-n] text 142x17 · 12px
    3.14. divider  [s4-d-div2] rectangle 400x2 · fill #e5e1d8
    3.15. checkbox "Thesis ch.2 · 14"  [s4-d-cb0] rectangle 20x20 · fill #ffffff
    3.16. checkbox "Methods · 9"  [s4-d-cb1] rectangle 20x20 · fill #ffffff
    3.17. checkbox "Reading group · 22"  [s4-d-cb2] rectangle 20x20 · fill #ffffff
    3.18. checkbox "Archive · 40"  [s4-d-cb3] rectangle 20x20 · fill #ffffff
    3.19. text "The whole library · 85 documents"  [s4-d-count] text 400x17 · 12px
    3.20. button "Cancel"  [s4-d-cancel] rectangle 110x34 · fill #ffffff
    3.21. button "Save"  [s4-d-save] rectangle 120x34 · fill #4a3fa6
  4. text "The open chat shows here."  [s4-main] text 204x20 · 14px
  5. card  [s4-composer] rectangle 848x64 · fill #ffffff
    5.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s4-ask] rectangle 712x44 · fill #ffffff
    5.2. button "Send"  [s4-send] rectangle 92x36 · fill #4a3fa6

### Screen 5 "S5 — Continuing in a new chat"  [s5] frame 1160x920 · not approved yet
  1. header  [s5-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s5-i-menu] rectangle 16x16
    1.2. icon  [s5-i-cog] rectangle 16x16
    1.3. icon  [s5-i-side] rectangle 16x16
    1.4. icon  [s5-i-search] rectangle 16x16
    1.5. icon  [s5-i-back] rectangle 16x16
    1.6. icon  [s5-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s5-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s5-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s5-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s5-brand] text 83x21 · 15px
  2. sidebar  [s5-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s5-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s5-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s5-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s5-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s5-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s5-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s5-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s5-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s5-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s5-row3] rectangle 262x44 · fill #f8f7f3
  3. text "Critical reading · whole library"  [s5-chatline] rectangle 612x36 · fill #f8f7f3
  4. button "Continue in a new chat…"  [s5-continue-new] rectangle 224x36 · fill #ffffff
  5. text "You"  [s5-you] text 28x17 · 12px
  6. text "How does dopamine signalling change with age?"  [s5-q] text 816x21 · 15px
  7. text "The answer sits here."  [s5-a] text 172x20 · 14px
  8. card  [s5-dlg] rectangle 560x680 · fill #ffffff
    8.1. heading "Continue in a new chat"  [s5-dlg-h] text 254x28 · 20px
    8.2. text "This chat stays as it is. The new one starts wi…"  [s5-dlg-sub] text 512x17 · 12px
    8.3. text "Answer as"  [s5-n-m-h] text 68x17 · 12px
    8.4. radio "Research assistant"  [s5-n-r0] ellipse 20x20 · fill #4a3fa6
    8.5. text "Built in · the default"  [s5-n-rd0] text 360x17 · 12px
    8.6. radio "Critical reading"  [s5-n-r1] ellipse 20x20 · fill #ffffff
    8.7. text "Built in · where sources agree, disagree, and t…"  [s5-n-rd1] text 360x17 · 12px
    8.8. radio "Plain summary"  [s5-n-r2] ellipse 20x20 · fill #ffffff
    8.9. text "Yours · its answers are marked not evaluated"  [s5-n-rd2] text 360x17 · 12px
    8.10. divider  [s5-n-div] rectangle 400x2 · fill #e5e1d8
    8.11. text "Search in"  [s5-n-f-h] text 68x17 · 12px
    8.12. checkbox "All documents · 85"  [s5-n-all] rectangle 20x20 · fill #ffffff
    8.13. text "unticks every folder"  [s5-n-all-n] text 142x17 · 12px
    8.14. divider  [s5-n-div2] rectangle 400x2 · fill #e5e1d8
    8.15. checkbox "Thesis ch.2 · 14"  [s5-n-cb0] rectangle 20x20 · fill #ffffff
    8.16. checkbox "Methods · 9"  [s5-n-cb1] rectangle 20x20 · fill #4a3fa6
    8.17. checkbox "Reading group · 22"  [s5-n-cb2] rectangle 20x20 · fill #ffffff
    8.18. checkbox "Archive · 40"  [s5-n-cb3] rectangle 20x20 · fill #ffffff
    8.19. text "2 folders · 21 documents: 2 are in both, search…"  [s5-n-count] text 400x17 · 12px
    8.20. checkbox "Carry the last 4 exchanges, as they were"  [s5-carry] rectangle 20x20 · fill #4a3fa6
    8.21. text "Fitted to the window like any earlier message; …"  [s5-carry-n] text 480x17 · 12px
    8.22. button "Cancel"  [s5-dlg-cancel] rectangle 120x36 · fill #ffffff
    8.23. button "Continue"  [s5-dlg-go] rectangle 164x36 · fill #4a3fa6
  9. card  [s5-composer] rectangle 848x64 · fill #ffffff
    9.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s5-ask] rectangle 712x44 · fill #ffffff
    9.2. button "Send"  [s5-send] rectangle 92x36 · fill #4a3fa6

### Screen 6 "S6 — The new chat, continued from another"  [s6] frame 1160x920 · not approved yet
  1. header  [s6-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s6-i-menu] rectangle 16x16
    1.2. icon  [s6-i-cog] rectangle 16x16
    1.3. icon  [s6-i-side] rectangle 16x16
    1.4. icon  [s6-i-search] rectangle 16x16
    1.5. icon  [s6-i-back] rectangle 16x16
    1.6. icon  [s6-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s6-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s6-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s6-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s6-brand] text 83x21 · 15px
  2. sidebar  [s6-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s6-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s6-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s6-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s6-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s6-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Today"  [s6-g0] rectangle 41x20 · fill #e9e6fa
    2.7. list-item "Receptor methods"  [s6-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "Dopamine and ageing"  [s6-row1] rectangle 262x44 · fill #f8f7f3
    2.9. text "Earlier"  [s6-g1] rectangle 54x20 · fill #e9e6fa
    2.10. list-item "Methods: rodent models"  [s6-row2] rectangle 262x44 · fill #f8f7f3
    2.11. list-item "Reading group, week 3"  [s6-row3] rectangle 262x44 · fill #f8f7f3
  3. text "Research assistant · Methods (9 documents)"  [s6-chatline] rectangle 612x36 · fill #f8f7f3
  4. button "Continue in a new chat…"  [s6-continue-new] rectangle 224x36 · fill #ffffff
  5. text "Continued from “Dopamine and ageing” (Critical …"  [s6-from] text 848x17 · 12px
  6. button "4 earlier exchanges, carried as context ▸"  [s6-carried] rectangle 600x36 · fill #ffffff
  7. text "Shown folded; each one opens as it was answered…"  [s6-carried-n] text 600x17 · 12px
  8. text "You"  [s6-you] text 28x17 · 12px
  9. text "Which of these methods measure receptor density…"  [s6-q] text 816x21 · 15px
  10. text "The new answer sits here, searched in Methods o…"  [s6-a] text 407x20 · 14px
  11. card  [s6-composer] rectangle 848x64 · fill #ffffff
    11.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s6-ask] rectangle 712x44 · fill #ffffff
    11.2. button "Send"  [s6-send] rectangle 92x36 · fill #4a3fa6

### Screen 7 "S7 — A question answered from the records"  [s7] frame 1160x920 · not approved yet
  1. header  [s7-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s7-i-menu] rectangle 16x16
    1.2. icon  [s7-i-cog] rectangle 16x16
    1.3. icon  [s7-i-side] rectangle 16x16
    1.4. icon  [s7-i-search] rectangle 16x16
    1.5. icon  [s7-i-back] rectangle 16x16
    1.6. icon  [s7-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s7-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s7-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s7-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s7-brand] text 83x21 · 15px
  2. sidebar  [s7-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s7-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s7-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s7-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s7-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s7-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s7-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s7-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s7-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s7-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s7-row3] rectangle 262x44 · fill #f8f7f3
  3. text "Research assistant · whole library"  [s7-chatline] rectangle 612x36 · fill #f8f7f3
  4. button "Continue in a new chat…"  [s7-continue-new] rectangle 224x36 · fill #ffffff
  5. text "You"  [s7-you] text 28x17 · 12px
  6. text "How many documents are in Thesis ch.2?"  [s7-q] text 816x21 · 15px
  7. text "Answered from the library's records · no model,…"  [s7-rule-tag] rectangle 420x28 · fill #f8f7f3
  8. text "Thesis ch.2 holds 14 documents."  [s7-ans] text 268x21 · 15px
  9. card  [s7-looked] rectangle 600x252 · fill #ffffff
    9.1. text "What it looked up"  [s7-looked-h] text 160x22 · 16px
    9.2. text "list_documents · folder Thesis ch.2 · 14 rows"  [s7-looked-call] text 560x18 · 13px
    9.3. list-item "Smith 2021 · Striatal release across the lifesp…"  [s7-rec0] rectangle 568x36 · fill #ffffff
    9.4. list-item "Lee 2019 · Ageing and the dopamine system"  [s7-rec1] rectangle 568x36 · fill #ffffff
    9.5. list-item "Ortiz 2020 · Motor ageing in mice"  [s7-rec2] rectangle 568x36 · fill #ffffff
    9.6. button "11 more"  [s7-more] rectangle 160x32 · fill #ffffff
    9.7. text "Each row opens the document in the Library"  [s7-hint] text 290x17 · 12px
  10. text "Same answer from /folder Thesis ch.2. Ask withi…"  [s7-cmd] text 760x17 · 12px
  11. card  [s7-composer] rectangle 848x64 · fill #ffffff
    11.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s7-ask] rectangle 712x44 · fill #ffffff
    11.2. button "Send"  [s7-send] rectangle 92x36 · fill #4a3fa6

### Screen 8 "S8 — What the answer carried"  [s8] frame 1160x920 · not approved yet
  1. header  [s8-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s8-i-menu] rectangle 16x16
    1.2. icon  [s8-i-cog] rectangle 16x16
    1.3. icon  [s8-i-side] rectangle 16x16
    1.4. icon  [s8-i-search] rectangle 16x16
    1.5. icon  [s8-i-back] rectangle 16x16
    1.6. icon  [s8-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s8-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s8-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s8-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s8-brand] text 83x21 · 15px
  2. sidebar  [s8-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s8-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s8-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s8-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s8-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s8-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s8-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s8-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s8-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s8-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s8-row3] rectangle 262x44 · fill #f8f7f3
  3. text "Research assistant · whole library"  [s8-chatline] rectangle 612x36 · fill #f8f7f3
  4. button "Continue in a new chat…"  [s8-continue-new] rectangle 224x36 · fill #ffffff
  5. text "You"  [s8-you] text 28x17 · 12px
  6. text "And how does that compare with the human studie…"  [s8-q] text 816x21 · 15px
  7. text "The human studies agree on the direction [1] bu…"  [s8-ans] text 800x20 · 14px
  8. button "How this answer was made ▾"  [s8-prov-toggle] rectangle 600x36 · fill #ffffff
  9. card  [s8-carried] rectangle 600x236 · fill #ffffff
    9.1. text "What the answer carried"  [s8-carried-h] text 214x22 · 16px
    9.2. text "Fitted to the window: 3 earlier messages and 2 …"  [s8-fit-warn] rectangle 568x28 · fill #fbeecb
    9.3. text "Earlier messages"  [s8-k0] text 124x18 · 13px
    9.4. text "6 of 9 · the 3 oldest left out"  [s8-v0] text 396x18 · 13px
    9.5. text "Passages"  [s8-k1] text 66x18 · 13px
    9.6. text "8 of 10 · 2 left out, 1 shortened"  [s8-v1] text 396x18 · 13px
    9.7. text "Window"  [s8-k2] text 51x18 · 13px
    9.8. text "8,192 tokens · reported by the model's server"  [s8-v2] text 396x18 · 13px
    9.9. text "Mode"  [s8-k3] text 37x18 · 13px
    9.10. text "Research assistant · template 3f2a…"  [s8-v3] text 396x18 · 13px
  10. text "When the server does not say its window: "Windo…"  [s8-floor] text 760x17 · 12px
  11. card  [s8-composer] rectangle 848x64 · fill #ffffff
    11.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s8-ask] rectangle 712x44 · fill #ffffff
    11.2. button "Send"  [s8-send] rectangle 92x36 · fill #4a3fa6

### Screen 9 "S9 — The chat list: grouped by mode, filtered b…"  [s9] frame 1160x920 · not approved yet
  1. header  [s9-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s9-i-menu] rectangle 16x16
    1.2. icon  [s9-i-cog] rectangle 16x16
    1.3. icon  [s9-i-side] rectangle 16x16
    1.4. icon  [s9-i-search] rectangle 16x16
    1.5. icon  [s9-i-back] rectangle 16x16
    1.6. icon  [s9-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s9-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s9-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s9-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s9-brand] text 83x21 · 15px
  2. sidebar  [s9-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s9-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s9-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s9-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s9-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s9-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Critical reading"  [s9-g0] rectangle 113x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s9-row0] rectangle 262x44 · fill #efece5
    2.8. list-item "Motor slowing in older mice 13.11.2026 · 2 turn…"  [s9-row1] rectangle 262x44 · fill #f8f7f3
    2.9. text "Research assistant"  [s9-g1] rectangle 126x20 · fill #e9e6fa
    2.10. list-item "Which methods measure receptor density? 12.11.2…"  [s9-row2] rectangle 262x44 · fill #f8f7f3
    2.11. list-item "Dopamine review plan 10.11.2026 · 4 turns · The…"  [s9-row3] rectangle 262x44 · fill #f8f7f3
  3. card  [s9-menu] rectangle 300x420 · fill #ffffff
    3.1. text "Sort"  [s9-m-h] text 34x17 · 12px
    3.2. radio "Newest first"  [s9-s0] ellipse 20x20 · fill #4a3fa6
    3.3. radio "Oldest first"  [s9-s1] ellipse 20x20 · fill #ffffff
    3.4. divider  [s9-div1] rectangle 268x2 · fill #e5e1d8
    3.5. text "Group"  [s9-g-h] text 41x17 · 12px
    3.6. radio "No groups"  [s9-grp0] ellipse 20x20 · fill #ffffff
    3.7. radio "By mode"  [s9-grp1] ellipse 20x20 · fill #4a3fa6
    3.8. radio "By folder"  [s9-grp2] ellipse 20x20 · fill #ffffff
    3.9. divider  [s9-div2] rectangle 268x2 · fill #e5e1d8
    3.10. text "Only chats that cited"  [s9-c-h] text 149x17 · 12px
    3.11. input "A folder or a document"  [s9-c-q] rectangle 268x32 · fill #ffffff
    3.12. button "Thesis ch.2 ✕"  [s9-c-on] rectangle 268x30 · fill #e9e6fa
    3.13. text "A row's labels: what its answers cited, read fr…"  [s9-c-n] text 268x30 · 11px
  4. text "The open chat shows here."  [s9-main] text 204x20 · 14px
  5. card  [s9-composer] rectangle 848x64 · fill #ffffff
    5.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s9-ask] rectangle 712x44 · fill #ffffff
    5.2. button "Send"  [s9-send] rectangle 92x36 · fill #4a3fa6

### Screen 10 "S10 — A past chat, reopened and continued"  [s10] frame 1160x920 · not approved yet
  1. header  [s10-hdr] rectangle 1160x40 · fill #f8f7f3
    1.1. icon  [s10-i-menu] rectangle 16x16
    1.2. icon  [s10-i-cog] rectangle 16x16
    1.3. icon  [s10-i-side] rectangle 16x16
    1.4. icon  [s10-i-search] rectangle 16x16
    1.5. icon  [s10-i-back] rectangle 16x16
    1.6. icon  [s10-i-fwd] rectangle 16x16
    1.7. button "Chat"  [s10-tab-chat] rectangle 70x28 · fill #4a3fa6
    1.8. button "Library"  [s10-tab-lib] rectangle 70x28 · fill #ffffff
    1.9. button "Graph"  [s10-tab-graph] rectangle 70x28 · fill #ffffff
    1.10. text "provenote"  [s10-brand] text 83x21 · 15px
  2. sidebar  [s10-rail] rectangle 280x880 · fill #f8f7f3
    2.1. input "Filter chats…"  [s10-filter] rectangle 200x30 · fill #ffffff
    2.2. button "✓"  [s10-select] rectangle 26x26 · fill #ffffff
    2.3. button "⇅"  [s10-sort] rectangle 26x26 · fill #ffffff
    2.4. button "+ New chat"  [s10-new-chat] rectangle 222x30 · fill #f8f7f3
    2.5. button "⚙"  [s10-defaults] rectangle 34x26 · fill #ffffff
    2.6. text "Recent"  [s10-g0] rectangle 47x20 · fill #e9e6fa
    2.7. list-item "How does dopamine signalling change with age? 1…"  [s10-row0] rectangle 262x44 · fill #f8f7f3
    2.8. list-item "What is DeepLabCut and what does it solve? 07.0…"  [s10-row1] rectangle 262x44 · fill #f8f7f3
    2.9. list-item "What did Liu et al. find about axonal projectio…"  [s10-row2] rectangle 262x44 · fill #f8f7f3
    2.10. list-item "What is the rich-club organisation of the brain…"  [s10-row3] rectangle 262x44 · fill #f8f7f3
  3. text "Critical reading · Reading group (22 documents)…"  [s10-chatline] rectangle 612x36 · fill #f8f7f3
  4. button "Continue in a new chat…"  [s10-continue-new] rectangle 224x36 · fill #ffffff
  5. text "You"  [s10-you] text 28x17 · 12px
  6. text "What do the sources say about the caudate?"  [s10-q] text 816x21 · 15px
  7. button "The decline is steepest in the caudate [1]"  [s10-c1] rectangle 560x40 · fill #e9e6fa
  8. button "It begins before the age of thirty [2]"  [s10-c2] rectangle 560x40 · fill #fbeecb
  9. card  [s10-review] rectangle 560x64 · fill #ffffff
    9.1. text "Reviewer, as recorded: 1 claim supported, 1 not…"  [s10-review-t] text 528x33 · 12px
  10. button "How this answer was made ▸"  [s10-prov-toggle] rectangle 560x36 · fill #ffffff
  11. divider  [s10-divider] rectangle 816x2 · fill #e5e1d8
  12. text "continuing below"  [s10-div-t] text 115x17 · 12px
  13. text "You"  [s10-you2] text 28x17 · 12px
  14. text "Is that true in women too?"  [s10-q2] text 816x21 · 15px
  15. text "The new answer sits here; same mode and folders…"  [s10-new-a] text 800x20 · 14px
  16. card  [s10-composer] rectangle 848x64 · fill #ffffff
    16.1. input "Ask your documents… (Enter to send, Shift+Enter…"  [s10-ask] rectangle 712x44 · fill #ffffff
    16.2. button "Send"  [s10-send] rectangle 92x36 · fill #4a3fa6
```
