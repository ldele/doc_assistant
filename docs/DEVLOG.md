<!-- status: active · updated: 2026-09-02 · class: append-only -->

# DEVLOG — doc_assistant

Real-time development log. One entry per logical change.
Append only — never edit past entries.

Format: What changed | Why | Rejected alternatives | What it opens

> **This file keeps 2026-08-12 onward.** Older entries live in the archives, moved verbatim:
> **2026-08-08 (1) → 2026-08-11 (4)** in [`docs/archive/DEVLOG-archive-005.md`](archive/DEVLOG-archive-005.md)
> (rotated 2026-08-30) ·
> **2026-08-05 → 2026-08-07** in [`docs/archive/DEVLOG-archive-004.md`](archive/DEVLOG-archive-004.md)
> (rotated 2026-08-28) · **2026-08-02 → 2026-08-03** in
> [`docs/archive/DEVLOG-archive-003.md`](archive/DEVLOG-archive-003.md) (rotated 2026-08-26 — this
> row was missing until 2026-08-28) · **2026-07-15 → 2026-08-01** in
> [`docs/archive/DEVLOG-archive-002.md`](archive/DEVLOG-archive-002.md)
> (rotated 2026-08-15) · **2026-05-21 → 2026-07-14** in
> [`docs/archive/DEVLOG-archive-001.md`](archive/DEVLOG-archive-001.md) (rotated 2026-07-21).
>
> **This file is capped at 4,000 lines** by `tests/unit/test_doc_sizes.py`. When it trips, rotate
> the oldest entries into a new `DEVLOG-archive-NNN.md` and update the list above — **do not raise
> the cap.** The cap exists because this log reached 8,244 lines before anyone noticed: every entry
> is individually small and correct, so unbounded growth is invisible per commit.

---
## 2026-09-02 — The version check now reads the two Cargo files, and its file list is a test

**What changed.** `scripts/release_preflight.py`'s `versions` check went from five sources to
**seven**: `apps/desktop/src-tauri/Cargo.toml` (`[package] version`) and `Cargo.lock` (the
`doc-assistant-desktop` entry, found by name in the package list) now join the five it already
read. `collect_versions()` is split out of `check_versions()` so the *list of files* is importable
and therefore testable, and `docs/RELEASE.md` §1 grew from six rows to eight.

Three tests, in `tests/unit/test_release_preflight.py`:

- **the source list, pinned by equality** — adding a version-carrying file means adding it here;
- **no source may read as a sentinel** — `(not found)` and `(missing)` compare equal to each
  other, so seven simultaneously-broken readers would have "agreed";
- **a drift in any single file must FAIL**, parametrised over the source list rather than
  spot-checked, so a file added to the list gets its negative case for free.

**Why.** The `versions` check reported green while `Cargo.toml` and `Cargo.lock` held `0.4.1`
through **v0.4.2, v0.5.0 and v0.5.1** — three tagged releases (verified by reading each tag:
`git show vX.Y.Z:apps/desktop/src-tauri/Cargo.toml`). It never opened them, and neither did the
runbook table. This is the *inverse* of the `uv.lock` incident that created the check: not a file
someone forgot to edit, but a file nothing was looking at. An agreement check is worth exactly as
much as its file list, and until now that list existed only inside a function body.

Surfaced at 0.6.0 the hard way: the release build regenerated `Cargo.lock` from 0.4.1 to 0.6.0
*after* the release commit, and `tree_clean` — not `versions` — was what caught it.

**Verified by reverting the fix.** With the two Cargo sources removed from `collect_versions()`,
exactly three tests fail (the list test and both Cargo drift cases) and the other 14 pass. The
guard reproduces the historical bug rather than merely describing it.

**The re-lock command is verified too.** `docs/RELEASE.md` §1 now carries
`cargo update --manifest-path apps/desktop/src-tauri/Cargo.toml -p doc-assistant-desktop --offline`
— run against a deliberately desynced tree, exit 0, one line changed, no network. `cargo metadata`
was tried first and rejected on evidence: it wants metadata for every locked package including
Android-only ones this box has never downloaded, so it exits **101** under `--offline` (after
writing the lock) and needs the network without it; `--no-deps` exits 0 and updates nothing.

**Rejected.** *Deriving the file list from the runbook table* — a docs parser is a second thing to
break, and the table is prose. *Hand-editing `Cargo.lock`* — it is a lock; cargo overwrites it at
build time anyway, which is precisely the failure being fixed. *Extending the existing
"do they agree?" test* — it structurally cannot catch a missing source, which was the bug.

**What it opens.** `artifact_fresh` has the same shape of weakness one layer over: it compares
**mtimes**, so `git checkout main` re-materialising a byte-identical file (blob `a789456…` at both
`ef4a6d8` and `663c290`) fails it. Comparing `git diff <built-commit> HEAD` instead would say what
the check means. Not done here — it needs the built commit recorded next to the artifact, which is
a change to the build, not to the check.

---
## 2026-09-01 (6) — Two UI corrections from using the app: controls too small to find, and a dropdown painted by the OS

**What changed.** Both reported after driving the merged build in the native Tauri window.

1. **The source pane's header controls were too small to find.** 0.15rem of padding under a 0.7rem
   label, drawn in the muted `--fg-2`, inside a border that reads as part of the pane frame. Three
   changes, no redesign: a real hit target (**26px** row, steppers squared to 26x26 from 23px wide,
   close 29x26), the resting colour moved off `--fg-2` onto **`--fg`**, and hover that fills the
   button rather than only tinting the glyph. The active fit preset now takes the accent as a
   *fill* (`--accent` / `--accent-fg`) instead of tinting its text, so which mode is on is legible
   at a glance. Focus rings added on all three groups.
2. **The chat folder-scope dropdown did not match the app.** Its options and popup were painted by
   the user agent in the **OS** scheme — a light menu over a dark app.

**The second one was not a colour bug in that component.** `.scopepick select` sets
`background: none`, so the closed control was already correct; what was wrong is that **the app
never declared `color-scheme`**. Without it the UA paints every native widget in the system scheme
regardless of the page's palette. So the fix is one declaration per theme state in `app.css`
(`:root`, `[data-theme='dark']`, `[data-theme='light']`, and the `prefers-color-scheme` block) —
next to the palettes, not on the one control, because the same mismatch was in **all five**
`<select>`s and every native scrollbar fallback.

**Verified live** in the running app, both themes: `color-scheme` resolves `dark` / `dark` / `light`
across system-default, forced-dark and forced-light, and the scope `<select>` inherits it in all
three. Control contrast checked in both — active preset indigo-on-white in light, and the close
button now `#ece5d6` on dark where it was the muted `#a79e8b`.

**Rejected.** *Styling `option` backgrounds directly* — works in Chromium, does nothing for the
popup chrome or the scrollbars, and would need repeating in five places. *Hardcoding
`color-scheme: dark` on the control* — correct in one theme and wrong in the other.

**Gates.** node:test 257/257 · svelte-check 219/0. CSS-only plus one icon size; no logic touched.
**$0 — no model call.**

## 2026-09-01 (5) — Row 18 closed out: a citation now opens its page, and the two branches no test could reach were driven for real

**What changed.** Two gaps, both named in the 2026-09-01 (1) baton as unfinished.

1. **A chat citation can open its page.** The source card gains **Show the page**, which navigates
   to the Library and opens the pane where the passage is. `GET /api/library/chunk-page` now
   returns `{document_id, page}` rather than a bare page: a chat citation carries a `chunk_key` and
   **no document id**, and turning one into the other means reading the chunk store — so it happens
   on the server rather than by parsing the key's shape in the client, where the second copy of
   that contract would rot. New `library.locate_chunk` + `ChunkLocation`; `page_for_chunk` is now a
   thin wrapper on it.
   **It is offered for a figure too**, which the card's own comment had anticipated: a figure has
   no position in the text, so the page image is the only place it can be shown.
2. **The unavailable and text-only arms were driven**, against the live library, with a backup taken
   first — a file moved out from under a document, then a document's `format` flipped to `epub`.
   Both restored; the library matches `data/library.db.bak-20260901-183252-prearms` row for row.

**And that is where the two real defects were, neither of which a test could have caught** — the
corpus is 98/98 PDF with every file present, so no test fixture stands in for driving it:

- **The size and zoom controls rendered over a document that has no page.** "Fit page | Width |
  − 100% +" sat above the sentence *"The file is not where the library expects it"*. A dead
  control, and this project's own rule is that a dead control is worse than none. Now gated on a
  renderable page, with the close button taking the right-hand margin when they are absent.
- **Two pieces of copy that were wrong.** The backend said *"a epub document has no pages"* — an
  article cannot agree with a value read from the database, so it is now *"a document in EPUB
  format"*. And the pane said the extracted text was **below** while its own hint said **beside**:
  Chunks is beside the pane in the split layout and above it when stacked, so any direction is
  wrong half the time. Both directions dropped.

**A false defect, avoided by the project's own rule.** After the jump, the citation panel appeared
to stay open over the Library — the DOM still held it 2 s later and a screenshot showed it. It had
in fact closed; the node was mid-transition, exactly the stranding the baton warns about
(2026-08-31 (3)). **When the DOM and the state disagree, believe the state**: querying again showed
the card gone. Nothing was "fixed".

**A test that had to be rewritten, for a reason worth keeping.** The first version of the route
test monkeypatched `source_view.locate_chunk` and failed — the route resolves the name through the
**package** re-export (`from doc_assistant.library import locate_chunk`), which is a separate
binding. That is the trap in `src/doc_assistant/CLAUDE.md`, met from the other side. Rather than
patch the re-export, the fake chunk store now holds real rows, so the tests exercise the real
derivation — cache markers on disk, offset in the metadata, page derived.

**Verified live.** A mocked `/api/chat` turn (fabricated sources, a **real** `chunk_key`; no model
call, no cost — the discipline from the 2026-08 chat-UI note) → click the citation → **Show the
page** → the Library opens `03-Zuo2014_NBR_fconn.pdf` at **"Page 10 of 19 · cited here"**, the page
computed independently beforehand. Unavailable and text-only arms both render a sentence, no image,
no broken image, no page nav, no size controls.

**Gates.** pytest source-viewer suites **44/44** (+7) · node:test 257/257 · svelte-check 219/0 ·
mypy 98/0 · ruff + format clean · bandit 0 · `detect-secrets` clean against the baseline.
**$0 — no model call.**

## 2026-09-01 (4) — The page fits because the reader decides how: a real zoom, a draggable split, and renders that get sharper instead of bigger

**What changed.** The source pane stops being a fixed picture in a fixed box.

1. **Zoom** — a `− 76% +` stepper beside the fit presets, **Ctrl/Cmd + wheel** (a bare wheel still
   scrolls), and the reading itself is a button that returns to the chosen fit.
2. **A draggable split** — a `separator` between the document and the pane, dragged with pointer
   events (so trackpad, pen and touch all work), moved with ← → (Shift for a coarse step, Home to
   centre), double-clicked to reset. Persisted, clamped to 25-75% so neither side can be dragged away.
3. **Sharper renders, not magnification** — `GET …/page/{n}` takes a `dpi`, and the pane climbs a
   ladder (110 → 150 → 200 → 260 → 330 → 400) as the page is drawn larger.

**Why (3) is the part that matters.** Zoom on a fixed image is just blur. Asking the server to
draw the page again at the resolution it is being displayed at is what makes zoom mean anything —
verified live: at 354% the pane requested **260 dpi** and got a **1831px** render in place of the
775px one, and stepping back out returned to 200 then 110.

**Three decisions worth keeping.**

- **Zoom is a multiple of the pane's width**, not of "actual size" — a page that was never on
  paper here has no actual size, and a percentage of one would shift under the reader every time
  they dragged the split. `Width` is therefore always 100%, and `Fit page` is whatever fits.
- **The dpi ladder is quantised.** Requesting exactly what each zoom level needs would issue a
  render per frame of a drag. Snapping **up** to a rung keeps it to a handful of fetches and never
  asks for an image blurrier than the one it replaces. The ceiling is enforced server-side
  (`clamp_dpi`, 72-400): render cost grows with the square of dpi, so an unbounded query parameter
  is a work generator. Out-of-range is **clamped, not refused** — a zoom level is not a validation
  error.
- **`dpi` is clamped in the library, not the route.** One expression of the bound, called by the
  route, so no caller can reach the renderer around it.

**Two things the work corrected in itself.**

- **A test disproved a claim in my own comment.** `renderDpi` said a 2x display makes the default
  soft at rest. It does not: at the pane's real width (433 CSS px, 612pt page) the render needs 51
  dpi at 1x and 102 at 2x, both under the 110 served. Device pixel ratio starts to bite once
  *zoomed* (153 dpi at 1.5x on a 2x display) or on a pane dragged wide (212 dpi at 900px). Comment
  corrected; the number is now pinned by a test, because nothing else would have caught it.
- **The fit was inferred and raced.** The first version decided "has the reader zoomed?" by
  comparing `zoom` against the computed fit — which is 1 before the box is measured, and 1 is also
  a legitimate zoom, so the pane opened at 100% instead of fitted. Replaced by an explicit
  `userZoomed` flag with the fit *derived*; a flag cannot race.

**Also fixed while verifying.** The fit was measured against the body's **border** box, so a
"fitted" page still needed 16px of scroll — `ResizeObserver`'s `contentRect` and the image's own
2px border now give a true fit (measured: `scrollY: 0`). And `setPointerCapture` is wrapped: it
throws for a pointer the browser no longer considers active, and the exception would abort
`onSplitDown` *before* its listeners attach — a handle that looks grabbed and does nothing.

**Verified live** on `cajal-lecture.pdf`: fitted at 72% with **0px scroll on both axes**; +/- and
Ctrl+wheel step through 38% → 188% → 354% with the dpi ladder following; a plain wheel scrolls
without zooming; a real mouse drag moved the pane 667 → 433px, persisted **0.438**, and the zoom
re-fitted 49% → 76% on its own. Both themes.

**Rejected.** *A zoom slider* — a stepper plus Ctrl+wheel covers coarse and fine without a control
that is hard to hit at the pane's size. *Re-rendering at exactly the needed dpi* — see the ladder.
*Refusing an out-of-range dpi with a 4xx* — the honest answer to "sharper than we draw" is the
sharpest we draw.

**Gates.** pytest source-viewer suites 37/37 (3 new dpi tests) · node:test **257/257** (+20) ·
svelte-check **219/0** · mypy 98/0 · ruff + format clean · bandit 0. **$0 — no model call.**

## 2026-09-01 (3) — No page in the corpus actually fit the source pane, including the ordinary ones

**What changed.** The source pane gains a **Fit page / Width** toggle in its header, defaulting to
**Fit page**, persisted client-side (`libPrefs.sourceFit`, localStorage, the same class as the
theme toggle and the grid/list switch — never a backend setting).

**Why.** The pane sized a page to the pane's *width* and let height scroll. Measured on the running
app at 1280x720, that means **no page fits**, at any shape in the corpus:

| Page aspect (h/w) | Example | Visible at fit-width | Width if fitted |
|---|---|---:|---:|
| 1.29 (US Letter, **57 of 98 docs**) | most of the corpus | **94%** | 405px of 433 |
| 1.41 (A4, 19 docs) | European journals | 86% | 371px |
| 1.57 | `cajal-lecture.pdf` | 77% | 333px |
| 1.79 | `middleton-2001.pdf` | **67%** | 292px |

Confirmed live rather than computed: the Cajal page rendered 433x679 into a 519px body — 178px of
scroll to see the bottom of a page. Even the most common size in the library needed a scroll to
show its last inch.

**The default is the argument, not the toggle.** ADR-050 D1 already settled what this pane is for:
the image carries *fidelity and provenance* — "this is the page it came from" — while row 19's
extracted text is the reading and searching surface. A view whose job is **where** should show the
whole page; a view that cannot show the whole page cannot answer that question. Fitting costs
little on the common case — 405px against 433px, a 6% loss of width for the last 6% of the page —
and the reader who does want to read has one click to Width, remembered thereafter.

It also removes a prerequisite from ROADMAP 24: a highlight band low on a page is worth nothing if
the pane opens showing the top two-thirds. Fit page means the band is on screen the moment it exists.

**Verified live** on `cajal-lecture.pdf` (the 1.57 case): default **100% visible, 0px scroll**;
toggled to Width **76%, 178px**; toggled back, 100%; the choice persisted. Both themes, and the
stacked (<900px) layout, where the pane's own `max-height: 60vh` becomes the binding constraint and
the page still fits inside it.

**Rejected.** *Reclaiming chrome* — the pane spends 51px of 574 on its header and footer; even
deleting both would not fit the 1.79 page. *Widening the pane* — a wider page is a **taller** one,
so it makes fitting strictly worse. *Fit page with no escape* — at 41% scale the body text is not
readable, and pretending otherwise would push people back to the OS viewer.

**What it opens.** In the stacked layout the height cap binds while horizontal room goes unused, so
the fitted page sits small between wide margins; raising the cap trades that against how far the
reader must scroll past the pane. Left alone deliberately — it fits, which was the requirement.

**Gates.** node:test 237/237 · svelte-check 219/0 · ruff clean · mypy 98/0. Frontend-only; the
Python suite is unmoved from 2026-09-01 (1). **$0 — no model call.**

## 2026-09-01 (2) — ADR-050 D5 measured: the on-image highlight is viable, and twice the measurement lied before it told the truth

**What changed.** No code. ADR-050 gains a dated **Addendum** answering the question D5 left open —
*can a cited passage be located as rectangles on its page image, and how accurately?* — and
ROADMAP row 24 files the follow-on with what it actually costs. Read-only, $0, on the live corpus.

**Why now.** D5 scoped the highlight out and called its accuracy "unmeasured", naming that the
follow-on's first question. Answering it before anyone commits to building is cheaper than
answering it afterwards, and the answer changes what the follow-on is.

**What it found.**

*Recall inverts with anchor length.* A **3-word** anchor places **91%** of single-page prose
sentences; a **12-word** one places **69%** (730 sentences). Longer anchors cross line breaks,
where hyphenation and the extractor's reflow stop matching. At 4 words: 90% placed, and only **5%**
genuinely ambiguous.

*The design the numbers point to is an envelope, not per-sentence rects.* Highlighting sentences
individually leaves ~10% unlit and scattered through the passage, and a reader cannot read a gap as
anything but "this part was not the evidence". A parent chunk is contiguous text, so highlighting
the band between the first and last unambiguous anchor gives **97% median purity** (highlighted
words that really are the passage; >= 90% on 88% of passages). Coverage measured 45% median, but
that is a **floor, not a verdict** — the probe grouped anchors into columns by `int(x0 // 60)`,
which splits an indented paragraph across two bands.

**Two measurement traps, both of which produced a confident wrong answer first, and both worth
keeping.** (1) The first run scored **68%** — because its needles still carried the cache's list
markers and table pipes. It was measuring the probe. Cleaned, the same method scores **94%**.
(2) "More than one rect" was then read as ambiguity, which made *longer* anchors look *less*
precise — an inversion that should have been the tell. `search_for` returns one rect **per line a
match spans**, so a wrapped phrase is indistinguishable from a repeated one until you separate them
geometrically — and the rects of a wrapped phrase are horizontally **disjoint** (tail of one line,
head of the next), so the natural test, "do they overlap in x?", misclassifies every one of them.
Only a vertical test works. Ambiguity fell from a fictional 22-32% to a real **5-7%**.

**Why it was not built this session.** The row implies a detail; the measurement says increment.
Three things have to be solved that nothing had named: real column detection (the probe's proxy is
not shippable), **43% of parent chunks straddle a page break** so the opening page can only ever
show part of the passage and the pane must say so, and a stated policy for the 5% ambiguous anchors
(decline, never guess). Filed as ROADMAP 24 with those three named, rather than started and left
half-done.

**Rejected.** *Building it on the 94% figure* — that number is single-page prose with tables
excluded, and quoting it for the feature as a whole would be the same error the first probe made,
one level up. *Per-sentence highlighting* — higher coverage, but its gaps make a false claim about
what the evidence was. *Treating the 45% coverage as the answer* — it is an artifact of the probe's
column proxy, and shipping a "known 45%" would bake in a limit that was never measured.

**What it opens.** ROADMAP 24. Also a question worth asking before that is built: with 43% of
parents crossing a page break, the highlight's honest unit may be *the passage across two pages*
rather than one page's band — which is a pane-layout decision, not a locating one.

**Gates.** Docs-only: `docs_check --strict` 0/0 · doc guards 9/9. No code changed, so the code
gates are unmoved from 2026-09-01 (1). **$0 — no model call.**

## 2026-09-01 (1) — ROADMAP 18: the document beside its library entry — and the row's stated reason for it being free was wrong

**What changed.** A source pane on the Library document view (`SourceViewer.svelte`, opened from a
new **Source** button beside Re-run), rendering the file itself one page at a time. Backend:
`library/source_view.py` + three routes — `GET /api/library/documents/{id}/source` (can this be
shown, and why not), `.../page/{n}` (PNG, rendered on demand), and `GET /api/library/chunk-page`
(which page a cited chunk sits on). Behind **ADR-050**, which row 18 did not have.

Each open parent block in **Chunks** now carries *"Show this page in the document"*, which resolves
that block's chunk key — the same `{document_id}:p{parent_index}` a chat citation carries — and
opens the pane there. ROADMAP 19 shows a passage in the extracted *text*; this shows the page of the
original it came off.

**Why.** Row 18 asked for it in 2026-08-25, and row 19 shipped the text half already noting the page
image was 18's job.

**The measurement that changed the design.** The row asserted *"page-level jump costs no ingest
change — chunks already carry `page`"*. It does not hold for the path the app retrieves on:
`USE_PARENT_CHILD` defaults true, and the parent-child store carries `page` on **615 of 39,705
chunks (1.5%)** — all of them figure chunks, whose page comes from figure detection. The flat
baseline store is 100%, and it is not the retrieval path. Building on the row as written would have
produced a feature that worked on figures and nothing else.

The conclusion survives for a different reason: the **cache** is page-annotated (`<!-- page:N -->`,
`extractors.py:99`) on **98/98** documents, with marker count equal to `Document.page_count`
exactly and sequential from 1, and chunks carry `parent_char_start` at 100% after row 19's re-chunk.
So the page is a read-time scan of markers against an offset — the rule `chunking.extract_chunk_metadata`
already applies at ingest for the flat store. `ChunkContext.page` therefore goes from **2.0% to
98.0%** populated on the live path (measured over 300 sampled parents), which also fills in a field
row 19's payload documented as permanently sparse. The remaining 2% are figure chunks, which have no
text span to place — and `page_for_chunk` still gives them a page from the stored value, so a figure
citation opens correctly where the *text* view honestly cannot show anything.

**Cost, measured before choosing.** A page render is 19-31 ms and 140-261 KB (median over 18 pages of
the 6 longest documents; 110 dpi ships). Nothing is pre-rendered or cached: the whole corpus is 2,973
pages, or ~760 MB and ~90 s to render up front, to save 19 ms.

**What driving it found — KI-57, and it is not this feature's bug.** Block 400 of `hebb_1949`
resolves to page 202, but its text is visibly on page 201. The cause is upstream: markers 201 and 202
delimit **byte-identical** segments — the cache holds page 201 twice and page 202 not at all.
Measured: **13 of 355 pages (3.7%) in `hebb_1949`, all 13 exact duplicates**, against **1 of 657
(0.2%)** across a 25-document sample. The marker *rule* is sound (342/355 and 656/657 segments match
their own page); what is occasionally wrong is the text placed under a marker. Filed rather than
fixed — the fix is an extraction change that re-invalidates every cache, and this is 0.2% of pages.
The suspicion that `_recover_lost_page` causes it is **wrong**: the other two recovery documents are
clean, 0 of 61.

**Rejected.** *PDF.js in the frontend* — better on selectable text and in-page find, but puts
document parsing in the thin shell, adds a worker and a Tauri CSP fight, and ships whole files to
show one page; the searchable surface already exists as the extracted text. *Tauri asset protocol* —
bypasses the ADR-002 boundary and dies in browser dev mode. *Backfilling `page` onto the
parent-child store* — a 39,705-chunk re-chunk to persist something derivable for free and
invalidated by the next extraction change. *Converting non-PDFs to PDF to give them pages* — invents
pages a document never had; they degrade to their extracted text instead, which is what they are.

**What it opens.** The passage highlight *on the page image* (ADR-050 D5, scoped out): offsets are
not coordinates, so it needs `page.search_for`, whose accuracy against normalised extraction is
**unmeasured** — that measurement is the follow-on's first question. Also: the pane is most of the
substrate an annotation layer would need, and nothing about it is speculative yet. And KI-57 has a
cheap exact detector if anyone picks it up — a page segment byte-identical to its predecessor found
13 of 13 with no false positives.

**Gates.** pytest **2349/0** (2315 + 34) · mypy 98/0 · ruff + format clean · bandit 0 ·
svelte-check **219/0** · node:test **237/237** (216 + 21) · doc guards 9/9 · `docs_check --strict`
0/0 · `test_api_check` 0/0 (240 files). Driven live on the real 98-document library in both themes
and at 820px. **$0 — no model call.**

## 2026-08-31 (4) — The graph now says how much of the library it covers, and why the obvious version of that number would have lied

**What changed.** `GraphStaleness` gains `n_documents_in_library`, and the Graph workspace states
**"Covers 30 of your 98 documents — a document appears once it mentions one of the 13 concepts on
your graph."** One field, one pure helper (`graph.graphCoverage`), 5 node:tests, 1 pytest case. No
extra query: the live document set was already being read for `missing_document_ids`.

**Entry (3) closed with the wrong open item, and checking it is what corrected the design.** It
said *"nothing watches the inverse — documents the corpus has that the graph has never seen … a
count of it would tell a user whether a rebuild is worth 10 seconds."* Measured before building it:
the library holds **98** documents, the graph cites **30**, and the other **68** are not waiting for
anything — they mention none of the **13** concepts in the graph vocabulary (of **593** curated). A
rebuild would return the same 30. So "68 documents not yet in the graph" would have been a number
that reads as a backlog, dressed a no-op button as the fix, and sent the user away from the lever
that actually moves it: **curating vocabulary** (ADR-018, ROADMAP 23).

**So the number is coverage, and it ships with the rule that produces it.** A fraction plus the
sentence explaining the fraction, in plain text rather than a warning — partial coverage is how the
feature works, not a fault. The test that matters asserts the *absence* of the misleading framing:
the string must not contain "missing", "not yet", "rebuild" or "pending".

**Rejected: a `built_at` timestamp in the skeleton.** The honest form of "documents added since the
build" needs one, and `_graph_version` is documented as a **timestamp-free** fingerprint precisely
so identical inputs produce a byte-identical `skeleton.json` (Decision 3). Stamping the artifact
would trade a verified determinism property for a number that coverage already answers well enough.

**Rejected: folding coverage into the staleness banner.** `stale` means *the graph is wrong* —
vocabulary drift or a reference it cannot resolve. Coverage is neither, and putting it behind a
warning icon would teach the user to dismiss the icon.

**What it opens.** The 68 uncited documents are a **vocabulary** signal, not a graph one: 13 of 593
curated concepts are on the graph, and that ratio — not a rebuild — is what decides coverage. The
Manage-keywords view is where that would be worth surfacing.

---
## 2026-08-31 (3) — Driving the app found four defects; three were real, and the fourth was the harness

**What changed.** A sweep of Chat, Library, Graph and Settings against the live corpus, and the
fixes for what it found. Three code changes (graph staleness gains a corpus dimension, two empty
states stop claiming emptiness before they know, the usage line stops reporting an unmeasured
zero), one data rebuild, 8 new tests. **KI-56** filed and fixed the same hour.

**1. The Graph cited documents that no longer exist, and printed their ids as titles.** Selecting a
concept listed entries like `c495b879-9b57-427c-b61e-1767a35808a2` where a title belongs — 8 of the
30 documents `skeleton.json` cited were gone, which is the pre-ADR-047 story: a re-extraction minted
a new id and the build artifact kept the old one. Two faults, fixed separately:

* **The view had no way to know.** `GraphStaleness` watched the *vocabulary* — concepts added or
  deleted since the build — and nothing watched the **corpus the graph was built over**. It now
  carries `missing_document_ids`, computed the same way (one id-set comparison at read time,
  nothing persisted). Deliberately asymmetric with the vocabulary rule: a document *added* since
  the build is not staleness (that is true of every build the moment it finishes), while a document
  the graph *cites* and cannot resolve is a broken reference.
* **The UI printed the key.** `docTitle` returned `docId` when the lookup missed — an identifier in
  a label's place, which is the exact thing `FileVerdict.duplicate_of` warns about two folders away.
  The list now renders only what resolves, the count follows it, and the shortfall is stated rather
  than silently dropped.

Then the data: `build_concept_skeleton --apply` is Node A, **zero LLM calls**, and took **10.5 s** —
`8 of 30` dead references became **0 of 30**. Live afterwards: 5 documents, 5 real titles, no UUIDs.

**2 and 3. Two surfaces asserted emptiness before they had an answer.** The Library said *"Your
library is empty"* and the sidebar *"No conversations yet"* while their fetches were still in
flight — a wrong claim standing where a loading state belongs, and the same distinction ADR-044
draws for update checks (*a failed check is `unknown`, never "up to date"*). Both lists start empty
whether or not anything has been asked, so **the fix is a latch, not a spinner**: render the empty
state only once a fetch has completed, success or failure. `svelte-check` earned its place here —
`documentsLoaded` was a plain `let`, fine as an internal fetch-once latch and silently non-reactive
the moment it became a prop, which would have pinned the loading line up forever.

**4. `0 tokens · local` reported a measurement of nothing where nothing was measured.** Ollama
returns no usage, so the counters sat at their initial `0`. The line now reads **`local · tokens not
reported`**. The zero is what is checked, not `is_local`: a local provider that *does* report counts
should have them shown, and a *metered* zero is a real measurement that must not be relabelled —
both pinned in `chat/usage.ts` (6 node:tests).

**The fourth "defect" was mine, and it is worth more than the three fixes.** The report said the
chat Source panel stayed open across Chat → Library → Graph, measured at 420x720 on all three. It
does not. `selectMode` nulls `activeCitation` correctly — confirmed by reading the live rune module
from the page, which showed `activeCitation === null` while the node was still in the DOM. The panel
uses `transition:fly`, and its eleven animations all reported `playState: "finished"` at
`currentTime: 0`: started while the automation pane was hidden at `innerWidth: 0`, so Svelte's
transition-end callback never fired and the node was never removed. Its final transform put it at
`left: 1280` in a 1280px viewport — **fully off-screen, scrim at opacity 0**, invisible to any user.
That is the hidden-pane trap `apps/desktop/CLAUDE.md` documents in as many words, and it was walked
into *after* dodging it once the same hour on a geometry question. **The lesson that generalises:
when a DOM observation and the state disagree, the state is the app and the DOM is the harness.**

**Four other things that looked like defects and were checked rather than reported.** 88
conversations with repeated titles (genuinely 88 distinct sessions, three runs of one battery
minutes apart on 2026-08-07); Enter-not-sending (the readiness gate during warm-up — it works);
the source panel appearing clipped at the window edge (screenshot cropping; `scrollWidth ===
clientWidth`); and two 500s at start-up (Vite proxying to uvicorn before it was listening).

**And one mess, cleaned up.** Probing the Settings rail, a `querySelector('nav')` matched the
sidebar instead and the loop clicked every row's action buttons, **pinning 75 conversations**.
Restored by diffing against the morning's backup — 8 rows unpinned in place, 80 stray rows removed,
`conversation_meta` back to **111 rows, 0 pinned, 0 differing, 0 lost**. Nothing archived, nothing
deleted. The three test conversations are soft-deleted; the library is as found: 98 documents, 881
figures, 615 descriptions, 1 root.

**What the sweep confirmed working.** Retrieval put the right five papers behind a RAG question, and
the reviewer caught the local model inventing `[24][26][27]` out of reference lists —
*"0 valid citation(s); 25/28 sentences uncited; out-of-range citations"* — which is KI-36 exactly as
documented, and which Settings had already predicted by quoting 36% for `llama3.1:8b` against 81%
for Haiku. Row 19's *In context* on a real citation: *"1% of the way in · in the extracted text of
rag_lewis_2020.pdf"*, highlight at 75 px inside a 223 px window. KI-50's crops render, and *Figure
images* reports *"0 re-run · 1 skipped — all 3 figure image(s) are already on disk"*.

**Rejected: auto-rebuilding the skeleton when it detects missing documents.** The module's own
docstring already refuses this for the vocabulary case — *"never to auto-rebuild (that would spend
the user's time unasked and destroy the seeded-layout determinism the view is verified with)"* — and
the corpus case has no better claim on the user's time.

**What it opens.** Nothing watches the *inverse*: documents the corpus has that the graph has never
seen. That is ordinary lag rather than a broken reference, but a count of it would tell a user
whether a rebuild is worth 10 seconds.

---
## 2026-08-31 (2) — Row 17: importing from Zotero is a route to the review sheet, and the catalogue's metadata is a slot the extractor cannot overwrite

**What changed.** ROADMAP 17, behind **ADR-049**. A new `src/doc_assistant/adapters/` package —
neutral `catalogue.py`, vendor `zotero.py` — plus `POST /api/catalogue/zotero/scan`, an
`ExternalMetadata` table, and a third route in the Add-documents dialog. 25 new pytest cases for the
reader, 10 for the metadata layer, 7 for the route, 3 for root scoping, 5 node:tests.

**The shape is the decision: an adapter returns paths and stops.** The scan hands back absolute
paths; the client stages them; the *existing* review sheet takes over. The proof that this was the
right cut came free — importing a library that overlaps your corpus produced *"3 files · 1 would be
added"*, with the two known files flagged as duplicates naming what they matched, and no code was
written for that. Same duplicate rule, same copy-or-reference choice, same progress bar.

**The half worth having is the metadata, and it needed a third slot.** A reference manager's title
is curated by a person; `metadata_extractor` guesses from a PDF's first page and sometimes picks the
journal name (KI-54, still open). But there was nowhere to put a curated answer: `Document.title` is
the extractor's slot and every metadata pass overwrites it, and `DocumentMeta.*_override` is the
user's own edit, which an import must never silently replace. So `ExternalMetadata` sits between
them, keyed **by path rather than by document** — the metadata arrives before the file is extracted,
and may never lead to a document at all. `ingest.main` applies it post-loop beside
`_assign_demo_folder`, and **`_rerun_metadata` re-applies it rather than extracting**: without that,
the safest-looking box in the re-run dialog would replace a curated title with a guess at it.

**Driving it end to end found a scaling defect no test would have.** Reference-adding registers a
root for a file's *parent directory*, and `_reference_target`'s docstring cites "a twenty-paper
Zotero folder" as the case that solves. Zotero's real layout defeats it: **every attachment lives in
its own `storage/<key>/` directory**, so one library would mint one `SourceRoot` per document — five
hundred rows, each stat-ed on every scan, against a robustness contract that says 10,000 documents.
Observed as three roots for three files, then fixed twice over: an adapter reports the catalogue's
storage folder and it is passed through as the batch's root, and `_reference_target` now prefers an
**already-registered root above the file**. Re-run: **one root, three files, rel_paths
`ZOTATT001/…`.** The second half improves the ordinary case too, and is not the guess the per-parent
rule refuses to make — an ancestor root exists only because someone established it.

**The catalogue is read from a copy.** Zotero holds `zotero.sqlite` open; the file and its
`-wal`/`-shm` companions are copied to a temp path and the copy opened read-only. A guard test
asserts the user's database is byte-identical afterwards. Their library is not ours to risk for a
feature they can live without.

**Everything declined is counted under a reason, never summed.** *"412 a web-page snapshot · 88 not
downloaded to this computer"* reads as a working filter; *"37 found"* out of a 500-item library reads
as a broken import. Snapshots are off by default — a library of any age holds hundreds.

**What these tests do not prove.** There is no Zotero on this machine, so the fixture *constructs* a
database to the documented Zotero 5/6/7 schema. That makes the suite a proof of the **mapping**, not
of the schema. Every query is written to fail with a sentence rather than a stack trace for exactly
that reason, and optional parts (collections, creators) degrade to "no authors" rather than losing
the import. **First contact with a real library is the open item**, and it is recorded in ADR-049
rather than in a comment.

**Verified live, on the real library, and left as found.** `~/Zotero` does not exist here, so the
button produced the intended 404 sentence and its *Choose the folder…* fallback; pointed at a
synthetic library built from two corpus PDFs plus one new one, it staged 3, flagged 2 duplicates,
reference-added and indexed the third — and the document came back titled **"Notes On A Synthetic
Paper · Ada Lovelace · 2026"**, which is what the catalogue said and not what the extractor would
have derived. Then deleted, and the registry rows and roots removed: 98 documents, 98 source files,
1 root, 0 external rows, 881 figures with 615 descriptions.

**Rejected.** Writing the catalogue's answer into `DocumentMeta` (that is the user's slot — an import
would overwrite what they typed); a separate Zotero add/index path (two duplicate rules that would
drift); registering the catalogue's root during the scan (merely *looking* would create state nobody
confirmed). All in ADR-049.

**What it opens.** Calibre is now one module and one route. Collections and item types are recorded
and unused — the substrate for the dormant `SourceFile.doc_type` and for folders. And the
linked-attachment base directory has no UI, so those attachments are skipped with a reason.

---
## 2026-08-31 (1) — KI-50: the 723 missing figure crops are back, and the button that would have destroyed the descriptions no longer does

**What changed.** Two opposite failures around the same rows. **KI-50** (open since 2026-08-27): 723
of 811 cropped PNGs were gone from disk while every row and every paid VLM description survived.
**KI-55** (found while fixing it, filed and fixed the same hour): `reingest._rerun_figures` rebuilt a
document's rows from scratch and wrote `vlm_description=None` into every one of them. A new `crops`
re-run part, `ingest.figures.restore_crops`, a `--repair-crops` mode on `scripts/extract_figures`,
and the carry-over. 9 new pytest cases.

**The repair re-renders; it does not re-detect.** Every row already carries the page and the bbox, so
the crop can be reproduced exactly. Re-detecting to recover a *file* would risk moving the rectangle
a description was written for — and a description attached to a different picture is worse than a
missing picture, which is the rule the chunk locator already lives by. Measured on the live library
before touching it: 811 rows with an `image_path`, **every one** with a complete bbox, a canonical
path, and a page matching its filename. Nothing had to be guessed.

**Result: 723 restored, 0 still missing, 0 errors, 57 seconds.** Verified against the database rather
than the script's own report — 811/811 resolve, no zero-byte files, and every crop's pixel size
matches its recorded bbox at 150 DPI. Rows unchanged at 881, descriptions unchanged at 615. ResNet's
page-1 crop is the 56-layer-vs-20-layer training-error chart its caption describes.

**KI-55 is the one that would have cost money.** `figures` looked like the cheapest useful box in the
re-run dialog, and the banner on the figures panel said in as many words *"re-run the figure
extraction pass"*. It deleted the rows and re-inserted them, so 552 paid descriptions on this library
would have gone — and because retrieval admits a figure on its **description**, not its image, those
figures would have dropped out of search as well. Descriptions are now carried across the rebuild,
and the guard fails without the fix (checked by patching it back out: *"2 description(s) kept"* while
every row came back `None`).

**Carried only when the region is recognisably the same.** The identity key is the page plus the bbox
rounded to whole points — the bbox *is* what a description describes. A region that moved gets no
description and the run says so: *"…, 3 dropped (their regions changed)"*. Both directions are
pinned, because "descriptions are kept" on its own would be satisfied by carrying them onto the wrong
pictures.

**A registry-ordering contract was about to break silently.** The client quotes the *last selected*
part as the dearest one, so `PARTS` must stay cheapest-first — an assumption living only in a comment
on the client. Inserting `crops` after `figures` would have made "instant" the quoted cost of a run
including a "few seconds" part. It sits after `metadata` instead, and a test now pins the literal
order with the reason.

**Cause: still not established, and now bounded.** The four retained backups (2026-08-24 onward) all
hold the identical 881/811 counts, and the ten stale directories on disk match no `doc_hash` current
in any of them — so the loss predates every backup we have. The standing hypothesis remains an older
`--rebuild` sweep. What *is* established is that the current code cannot repeat it:
`cleanup_orphan_figures` takes `gone` hashes only since ADR-047, and `repoint_figures` moves a
directory across a re-extraction rather than deleting it.

**Rejected: `extract_figures --force`.** It is the existing way to re-make crops and it deletes the
rows first — the exact loss KI-55 is about. Rejected too: a corpus-wide restore button in the app.
This was a one-time repair; the per-document and per-selection controls cover stragglers, and ADR-048
already puts corpus-wide passes in a runner rather than in the dialog.

**The banner now names the cheap part.** It said "re-run the figure extraction pass", which pointed at
the destructive one. It now says *"re-run **Figure images** to put them back. Descriptions and search
are unaffected."*

**Verified in the app:** ResNet's figures panel renders its three restored crops with no
missing-image banner, the "no image" cards are the caption-only rows that never had one, and
re-running *Figure images* reports **"0 re-run · 1 skipped — all 3 figure image(s) are already on
disk"**.

**What it opens.** The three CLI runners' duplicated per-document orchestration (ADR-048's first
consequence) now has a fourth reason to move into `src/`. And KI-50's cause stays open — if crops
vanish again, that is the signal to trace it rather than repair it.

---
## 2026-08-30 (9) — A citation can now show where it came from: the passage in place, with what surrounds it

**What changed.** ROADMAP 19's UI half. Clicking a citation opens the source panel; **In context**
expands to show the cited passage inside a window of the extracted text, highlighted, with the
lines before and after it and a one-line "8% of the way in". New
`library.get_chunk_context` + `GET /api/library/chunk-context`, a pure `chat/chunkcontext.ts`
(+9 node:tests) and `ChunkContextView.svelte`, wired into `SourceCard`. 15 new pytest cases.

**It reads the offsets rather than re-deriving them.** `char_start`/`char_end` were recorded at
ingest and are now 100% populated (entries 7 and 8), so placing a citation is a slice, not a
search. `chunk_key` — which every citation already carries — is the key, so no new identifier
crosses the wire. A parent-child citation resolves against the **parent** span, because the parent
is the passage the LLM actually read.

**Every failure is `None`, and the panel says so.** An unresolved span, a cache that is gone, an
unknown or malformed key: the endpoint 404s and the panel reads *"This passage can't be placed in
the source text."* On this corpus that is 3 chunks in 39,090 — the OCR'd chart labels the locator
refuses to guess at. **A window centred on the wrong paragraph is worse than no window**, which is
the same trade the ingest side makes.

**It expands in place rather than opening a modal.** The reader is mid-answer, checking a claim
against its source; a dialog would take the answer off screen, which is the one thing someone
verifying a claim must not lose.

**Honest about what it is showing: the extracted text, not the page.** The panel says "in the
extracted text of <file>" rather than implying a page image, and it leads with the character
position because that is what is always known — the parent-child path records no page on a parent,
so most chat citations have none. `whereLabel` shows a page only when one exists; "Page ?" would be
worse than saying nothing about pages. The document viewer is ROADMAP 18.

**Two bugs found by driving it, both mine, both in the verification rather than the feature.**

1. **The window opened at the top of the *before* text**, so the passage the reader clicked for was
   below the fold. Fixed by scrolling to it — but the first fix used `scrollIntoView`, which walks
   every scrollable ancestor: it dragged the transcript and the panel with it and left the box at
   `scrollTop` 21,792 on a 225px window with the highlight 11,000px out of view. It now sets
   `scrollTop` from two rects, touching this element and nothing else, and lands the passage a
   third of the way down so the lines above it stay visible.
2. **A "43,661px of content in a 225px box" panic was the hidden Browser pane.** `innerWidth` was
   `0`, which collapses every width and makes `scrollHeight` meaningless — the trap
   `apps/desktop/CLAUDE.md` documents in as many words. With the pane up: window 357x225,
   `scrollHeight` 1,093, highlight at 74px. **Geometry measured through a hidden pane is not
   geometry.**

**Verified live** with `/api/chat` mocked and a **real** `chunk_key`, so the LLM was never called
and the "In context" fetch hit the real endpoint: the passage renders highlighted at 8% of the way
into `rag_lewis_2020.pdf`, with its surroundings greyed; it collapses and reopens without
refetching; **0 console errors on a clean page life** (an earlier `ClaimReview` error was traced to
my own first mock omitting `flagged_claims`, not to this change).

**Gates:** pytest 2257/0 · mypy 94/0 · ruff clean · svelte-check 215/0 · node:test 200/200.

---
## 2026-08-30 (8) — The corpus is re-chunked: offsets go 63.3% to 100%, in 6m34s and without re-extracting a single file

**What changed.** `ingest --rebuild` over all 98 documents, to claim the offsets the cursor fix
(entry 7) made correct. **39,087 of 39,090 text chunks now carry a char span, and every one of them
resolves to text containing its own chunk.** Nothing was re-extracted.

**The cost estimate I gave was wrong by an order of magnitude, and checking the premise is what
found it.** I had told the user "~40 minutes, moves every `doc_hash`" — conflating *re-extract* with
*re-chunk*. The offsets are computed during chunking; the extraction fingerprints were byte-identical
to this morning's (`chunking.py` is not in the extraction closure), so every cache was still fresh.
The warmup confirmed it: 98 documents "extracted" in 16 seconds because it read them. So the real
operation is re-chunk + re-embed:

* **6m34s**, not 40 minutes — and no OCR, so none of the KI-47/KI-48 exposure a real re-extraction
  carries;
* **`doc_hash` did not move for a single document** (98/98 identical, ids identical), because the
  extracted text never changed. ADR-047's identity fallback was not even needed.

**Nothing was lost, and one thing was corrected.** Documents 98, `chunk_count` 15,173, figures 881,
citations 4,428, folder memberships 18, keyword links 1,455, the one `DocumentMeta` override — all
byte-identical across the rebuild. The store went from 624 figure chunks to **615**, which is
exactly one per described figure (881 figures, 615 with a VLM description): the rebuild swept 9
orphans that had accumulated. A correction, not a loss. The sparse index noticed the change by
itself and rebuilt (39,705 chunks, 54.9 MB).

**Three spans out of 39,090 are still absent, and they should be.** All three are picture-text
blocks — OCR'd chart axis labels like `20 20 20` and `7 7` — where a head/tail probe cannot
distinguish occurrences. The locator returns `None` rather than guessing, which is the trade the
feature is built on.

**I measured this wrong twice before measuring it right, and both errors are worth keeping.**

1. A first pass reported **580 spans (1.5%) pointing at the wrong text** — alarming, since a wrong
   highlight is the one failure this design refuses. It was my comparison: `page_content` has been
   through `clean_chunk_text` (page markers stripped) while the cache slice is raw, so any chunk
   straddling a `<!-- page:N -->` marker looked like a mismatch. Comparing like with like:
   **39,087 of 39,087, zero mismatches.**
2. Earlier, a 400-chunk sample of the store reported 0% coverage. `store.get(limit=400)` returns
   insertion order, so it sampled only the oldest documents. **A sample that is not random is not a
   sample.**

**A guard was added on the false premise and is kept on an honest one.** Composing a parent offset
with a child offset has a gap neither `locate_span` can see — if the parent matched a duplicate,
both halves are exact and the sum is still wrong — so the composed span is now verified against the
full text before it is recorded. Its first comment cited the 580 as evidence; that number was not
real, and a fabricated measurement in a code comment is exactly the failure entry 7 was about. The
comment now says plainly that the case **is not currently observed** (all 39,087 hold) and that the
guard is kept for its cheapness, and the test says plainly that it **does not discriminate** — it
is an invariant test, not a regression test.

**No second rebuild is needed for that guard:** zero spans on this corpus fail it, so re-running
would produce identical output. Verified by measurement rather than assumed.

**Gates:** pytest 2242/0 · mypy 94/0 · ruff clean · `docs_check --strict` 0/0.

---
## 2026-08-30 (7) — A 70-86% resolve rate was written down as a property of the corpus. It was a cursor parked one chunk too far along

**What changed.** `build_parent_child_chunks` advances its parent and child cursors to each
located span's **start + 1** instead of its **end**. Two lines. Measured on 12 documents of the
live corpus: **3,652 of 3,652 spans located, against 2,761 before — and 122 parents that were
previously never located at all.**

**How it was found — by checking a queued item's premise instead of trusting it.** ROADMAP 19
("locate a chunk in its source text") frames two routes as a genuinely open choice: (a) read-time
matching, (b) ingest-time offsets, where (b) is *"exact, needs a schema field + a re-ingest to
backfill"*. The schema field turned out to already exist, so the real question was how much of the
library already carries offsets. Counting them is where the trouble showed:

* a first 400-chunk sample said **0%** — wrong, because `store.get(limit=400)` returns insertion
  order and hit only the oldest documents. **A sample that is not random is not a sample;**
* counted properly, **25,125 of 39,705 chunks (63.3%)** carry a span, no document is *fully*
  spanned, and the 19 documents with none are exactly those ingested before the field landed;
* figures legitimately have no span, so they were separated out: **76.1% of text chunks**, and
  figures account for only 1.6% of the gap. They were not the explanation.

**The reconciling fact.** Replaying `locate_span` over the same documents by hand resolved
**100%** — which contradicted the store outright, and that contradiction is what located the bug.
The replay advanced its cursor to each span's *start*; the shipped code advances to the *end*. Both
splitters emit **overlapping** chunks (`PARENT_CHUNK_OVERLAP` 200, `CHILD_CHUNK_OVERLAP` 50), so the
next chunk begins *before* the previous one ended and a search starting at the previous end begins
past its own answer. Deterministic on a synthetic wall of prose: pre-fix it loses **every other
parent** — 1, 3, 5, 7, 9, 11, 13, 15, 17, 19 — and 60 of 124 children with them.

**The second failure mode is worse than the missing offsets.** A cursor past the true position does
not only fail to find; it can find a *later duplicate* and record a confidently wrong offset. The
locator's own docstring conceded that "a pathological duplicate can still land on the wrong one" —
with the cursor mis-set, it needed no pathology.

**This file's docstring had recorded the symptom for months.** `tests/unit/test_chunk_locations.py`
opens with *"Measured on the real corpus while building it: 70-86% of chunks resolve"* — a
measurement taken honestly, written down, and never explained. **A rate nobody can account for is a
question, not a finding**, and this one described a defect rather than the text. The docstring now
says so, with the after number beside it.

**Two false starts, both worth keeping visible.** The first regression fixture used
paragraph-separated prose and **passed against the broken code** — clean `\n\n` splits barely
overlap, so the bug never fired; that is the "a revert that does not reproduce the bug proves
nothing about the test" lesson arriving from the other direction. And the first ordering assertion
demanded globally ascending starts, which **failed on correct output**: parents overlap by design,
so the first child of parent N+1 legitimately begins before the last child of parent N. Ordering is
asserted per parent now, and the docstring says why.

**Rejected: advancing to the span start exactly.** `start + 1` costs nothing and keeps the property
the cursor exists for — a passage that repeats verbatim maps to its own occurrence rather than
matching the previous one twice.

**What it does NOT do: fix the library.** Offsets are written at ingest, so the 25,125 already
stored are the lossy ones and the 19 oldest documents still have none. Getting the full set means
re-extracting, which as of today is a per-document button (ROADMAP 20/21) rather than a whole-corpus
run — but it is ~15-35 s per document and it moves `doc_hash`, so it is the user's call, not a
migration to run unasked.

**What it opens.** ROADMAP 19's choice is now decided by measurement rather than left open: with the
cursor fixed, route (b) resolves everything the locator can see, so (a) is a fallback for
re-extraction-resistant documents rather than a competing design. The row is updated.

**Gates:** pytest 2241/0 · mypy 94/0 · ruff clean · `docs_check --strict` 0/0.

---
## 2026-08-30 (6) — The re-run earns its keep on the first real use: 15 titles a library had been missing, and a second extractor bug found by watching it work

**What changed.** ROADMAP 21's selection flow was used in anger over the **16 documents in the live
library with no title**, and it exposed one more extractor defect, now fixed: a title candidate
ending in `:` is a lead-in to something else, never the thing itself, so `_is_skippable_heading`
rejects it.

**Measuring first changed the plan, which is the point of measuring.** KI-54 (the title picker
preferring a bare journal name) looked like it might be systemic. The right population is not the 82
documents that already carry a title — an older picker wrote those — but what a re-run would write
*now*, so `extract_metadata` was dry-run over the 16 untitled documents' cached markdown, writing
nothing: **14 correct titles, 1 journal name, 1 lead-in fragment, 1 empty.** One document in
sixteen. That is what downgraded KI-54 from "design a structural fix" to "correct one row in the
metadata editor".

**A screen that did not work, recorded so nobody rebuilds it.** A first pass over the 82 existing
titles flagged 25 as "possible bare periodical names" using short + title-case + no verb + no
digits. Eyeballing them showed ~23 were genuine paper titles — `Neural Turing Machines`,
`Pointer Networks`, `Deep Residual Learning for Image Recognition`. That heuristic does not
describe a journal name; it describes an ML paper title. It is named in KI-54 as a detector not to
reuse.

**Then the run itself, through the grid's Select mode.** 16 tiles selected, Metadata only, *"16
documents · instant each."* — 16 outcomes, every one `filled`, and the library's NULL titles went
**16 → 1**. Most rows also gained a year, several an author list or a DOI. `98` documents and
`15,173` chunks unchanged throughout, because metadata touches neither identity nor the stores.

**The second bug, and why the fix is better than a suppression.** `ai_usage_cards_2023.pdf` stored
**`Preprint of the paper:`** — the line that precedes a title, stored as one. The guard rejects a
candidate whose cleaned text ends in a colon; a real title carries its colon in the *middle*, which
is left alone. Rejecting the lead-in let the picker fall through to the real answer:
**`AI Usage Cards: Responsibly Reporting AI-generated Content`** — a title that contains a
mid-string colon, so the same document proves the guard does not over-reach. Verified against the
pre-fix code (fails), then driven through the app end to end.

**Rejected: fixing the journal-name case at the same time.** It is one row in ninety-six. A
periodical detector would be new heuristic risk across all of them to correct that one, and the
first attempt at such a detector had already misfired on 23 real titles an hour earlier. It stays
KI-54, with the metadata editor as the workaround — an ADR-013 override survives any number of
re-runs.

**A backup was taken before writing to the live library** (`data/library.db.bak-20260830-163740-premetadata`),
because this was the first time the new feature wrote to 16 real rows at once.

**What it opens.** The `~~` fix, this colon guard and KI-54 are all the same shape — the title
picker accepting something that is adjacent to a title. If a third appears, that is the signal to
rebuild the candidate ranking rather than keep adding guards.

**Gates:** pytest 1798/0 (unit) · mypy 94/0 · ruff clean · `docs_check --strict` 0/0.

---
## 2026-08-30 (5) — Per-part re-ingest: the app can finally re-run one pass on one document, and it says what that costs first

**What changed.** ROADMAP rows **20 and 21**, behind **ADR-048**. A new
`src/doc_assistant/reingest.py` re-runs chosen parts of ingestion for chosen documents; a
`202 + poll` route pair drives it; one dialog serves both entry points — the document panel's block
nav (row 20) and the Library grid's Select mode (row 21). 20 new pytest cases, 11 new node:tests.

**The roadmap filed one open question and the answer changed the shape.** It asked *which parts can
re-run without moving `doc_hash`*. Answering it from `docs/performance.md` and the code turned up
three more, and all four are in ADR-048:

1. **The parts differ in cost by four orders of magnitude** — metadata is milliseconds (0.58 s for
   the *whole corpus*), text is 14.7 s mean and ~35 s when OCR fires. Four equivalent-looking
   checkboxes would be lying by omission, so every part carries a `cost` string, served from the
   registry rather than copied into the client, and the dialog states it before anything runs.
2. **Some passes have no per-document form at all.** `extract_keywords` and `compute_doc_vectors`
   are corpus-global by construction — scoping keywords to one document was *measured* to save
   **4%** — and epistemics and gaps have no scope flag. They are **named and declined** in the
   dialog, because a user who cannot find a button deserves to know there is no button.
3. **The selective ingest path skips cleanup, by construction.** `ingest.main` runs
   `cleanup_orphans_*` only when `files is None`, and a per-document re-extract *is* the
   `files is not None` branch. Re-extraction moves `doc_hash` (ADR-042) and ADR-047 keeps the row
   attached — but the previous hash's chunks would stay in **both** stores and stay retrievable.
   **Reproduced before it was fixed:** with the purge disabled, the guard test reports
   `superseded chunks survived: {'baseline': 1, 'pc': 1}` — the document indexed twice. `_rerun_text`
   now records the hash before it starts and purges it after, if it moved.
4. **Metadata has a safe overwrite, and it is not obvious.** ADR-013 keeps a user's edits in the
   separate `DocumentMeta` override table, so `Document.title` holds only the extractor's previous
   answer. A re-run may replace it without touching anything a human typed — the opposite of the
   usual `--force` hazard, and worth checking rather than assuming.

**Parts run in registry order, so `text` runs last.** It rewrites the cached markdown that metadata
and references read; running them first would derive from the text the user asked to replace. The
dialog reports the same order it executes.

**Row 21 cost nothing extra, which was the design.** `rerun` takes `document_ids: Sequence[str]`
from the start, so the grid's select bar opens the *same* dialog with a list — no second component,
no second cost statement to drift. One failing document is recorded as an error and the batch
continues: a selection of forty must not be lost to one unreadable PDF.

**Rejected: exposing the CLI runners over the API.** `scripts/extract_doc_metadata`,
`extract_citations` and `extract_figures` already do per-document work behind `--doc`, and shelling
out would have been the fastest path. But `apps/` are thin shells over `src/doc_assistant/`, never
over `scripts/` (non-negotiable #3), and those runners' orchestration is written for a console
report, not a caller. The orchestration moved into `src/` instead — where it should have been.
**The duplication that leaves is real and is recorded rather than discovered later** (ADR-048's
first consequence): those three runners still carry their own copies, so per-document metadata
logic currently has two homes. Rewiring three working runners is its own increment.

**Driving it on a real document found a metadata bug, and then a second one underneath.** Re-running
metadata on `03-Zuo2014_NBR_fconn.pdf` — which had **no** title, authors, year or references — filled
authors (`Xi-Nian Zuo, Xiu-Xia Xing`), year (2014) and 18 references correctly, and stored the title
as **`~~Neuroscience and Biobehavioral Reviews~~`**, markers included. `_clean_markdown` stripped
`*` and `_` and never `~`; that is now a one-line fix with a guard test that fails without it.
Stripping the markers then made the real problem legible: the title picker had chosen the **journal
name**, not the paper's title. That is **KI-54**, filed not fixed — the machinery to prevent it
already exists (`_JOURNAL_HEADER`) and did not fire because it matches a *dated* journal header, and
the fix must not be a list of journal names (that is corpus-tuning, which the robustness contract
forbids).

**Verified live on the real library, and it did real work.** Row 20: progress bar reading
*"Re-running 03-Zuo2014_NBR_fconn.pdf · metadata"*, `0 of 2`, then the report — *"metadata · filled
authors, title, year"* and *"references · 18 reference(s), 0 matched in your library"*. Row 21: three
tiles selected → the same dialog titled *"3 documents"*, the summary multiplying to *"3 documents ·
… each"*, the `text` confirmation appearing and gating the button until ticked. 0 new console
errors; no horizontal overflow at 375 px (modal 345 px, every part row fitting). **The library is
left as found plus the wins:** the title I wrote is reverted to NULL (it was worse than the filename
it replaced), authors, year and the 18 references are kept, 98 documents and every `doc_hash`
unchanged.

**One thing driving the UI caught that no test would have.** The first cost summary rendered as
*"1 document · about instant, per document."* The cost strings are already whole phrases, so the
sentence had to be built around them rather than prefixed onto them. Now pinned by a test.

**What it opens.** Re-running a part across the *whole* corpus from the app — still Settings' index
button plus the CLI. And the three runners' duplicated orchestration, which now has a `src/` home to
move into.

**Gates:** pytest 2237/0 · mypy 94/0 · ruff clean · svelte-check 213/0 · node:test 192/192.

---
## 2026-08-30 (4) — Graph vocabulary gets the in-app toggle ADR-018 left as a follow-up, and the Graph tab stops being a dead end

**What changed.** A concept can now be put on the concept graph, or taken off it, from **Manage
keywords** — a per-row toggle, an "On the graph (N)" lens, and a line saying how many of your
concepts are on it and what that means. Full stack: `library.set_family_graph_include` +
`KeywordFamily.graph_include`, `graph_include` on the wire, the family `PATCH` extended, the
`invalidateGraph()` latch drop, and two pure helpers in `library.ts` (+4 node:tests, +8 pytest).

**This is not a new decision — it is the one ADR-018 already made and deferred.** That ADR's
Consequences say it outright: *"Curation has no UI yet. Opting a concept in is CLI-only
(`add_concept`, the backfill runner) until a follow-up PR adds the toggle. Its natural home is the
Manage-keywords view, which keeps ADR-017 A1 intact (the graph still never writes the vocabulary —
the keywords view does)."* So no ADR was written for this; the location, the polarity and the
consequence were all settled in July. I had filed ROADMAP row 23 saying it "wants a decision before
code" — that was wrong, and reading ADR-018 before starting is what corrected it.

**Why it mattered enough to do today.** Re-enabling the Graph tab this morning surfaced that
`graph_include=true` is set by **exactly one thing**: `scripts/curate_concepts.py`. Every in-app
path (`create_keyword_family`, `promote_keyword`) creates concepts with it off, by ADR-018's own
design. Measured here: 593 concepts, **13** on the graph, all `source='manual'`. A fresh install's
graph was therefore empty with no in-app action that could ever change it — a shipped tab whose
only escape hatch was a Python script the packaged app does not carry.

**The feedback loop already existed and needed nothing new.** `concept_graph_view._staleness`
re-reads `load_concepts()` on every render, so the moment the vocabulary moves, the graph says
*"Graph is N concepts behind your vocabulary"* with the rebuild beside it. That is why
`set_family_graph_include` deliberately does **not** trigger a rebuild: the skeleton is a derived
sidecar (Enrichment-Layer Pattern), and a write path that kicks off a multi-second derived rebuild
is exactly the coupling that pattern exists to prevent. The one thing that did need wiring was
`invalidateGraph()` — the graph's lazy-load latch is one-shot, so a graph already loaded in this
session would have kept reporting the pre-toggle count. Dropping the latch beats refetching: it
pays for the fetch only if the user actually goes to look.

**Two decisions inside the PATCH that are load-bearing.**

- **Rename runs before the flag.** One PATCH carries two independent fields, and the rename is the
  half that can 409 on a taken label. Running it first means a refused rename cannot leave a family
  renamed-but-not-flagged, or flagged-but-not-renamed. `test_route_409_leaves_the_flag_alone`
  checks that, not the docstring.
- **An empty body is a 400, not a 200.** Both fields are optional so each can be sent alone (the
  rename path predates the flag and still sends only `canonical`), but a body with neither is
  always a caller bug, and returning a correct-looking payload for it hides the bug.

**The lens is over ALL families, not the visible ones — and that is a bug I wrote and then found.**
Most opted-in concepts have no member keywords and no documents, so `splitInheritedFamilies` files
them as *glossary-only*, which the Manage view hides by default. Applying the lens to the visible
subset showed an **empty list** while the count beside it said 13. Fixed by lensing the full set,
and pinned by a node:test that asserts the fixture row really is glossary-only before asserting the
lens reaches it — otherwise the test would pass on a fixture that never exercised the case.

**Adding a write route exposed a disagreement the read side had been getting away with.**
`list_keyword_families` excludes `kind="domain"` taxonomy field nodes (ADR-028 D4 — the 236 seeded
ANZSRC fields are not keyword families), but `get_keyword_family` looked up by id with no kind
filter and cheerfully returned one. Harmless while every family write arrived through a list the
user had clicked; not harmless the moment an id-addressed route could set `graph_include`, because
`load_concepts` filters on **that flag alone** — so a flagged field node would enter the graph
vocabulary, and `db/models.py` states outright that presence-assuming code must read only
`kind="concept"`. Both the lookup and the setter now refuse a non-concept row (the setter checks
independently: reaching the read guard would mean the write had already landed). Pinned by a test
that asserts at the **column**, not the status code — a 404 with the flag set would be exactly the
failure it exists to prevent, and it fails without the guard.

**Driving the new path found a bug that predates it, in the load latch.** `selectMode('library')`
guarded *all three* library sidecar loads behind `!documentsLoaded` — but the graph path loads
documents on its own (the ego panel resolves doc_ids → titles) **without** the family or folder
lists. So once Graph had been visited, entering Library skipped both fetches for the rest of the
session, and Manage keywords opened reading **"Families (0)"** on a library with 357. The graph's
per-concept **Edit** button has had this since it shipped; it only became visible now because the
empty state made that view a primary action reached directly from Graph. Fixed by giving families
their own `familiesLoaded` latch and checking folders on their own terms — `documentsLoaded` was
standing in for "the library's sidecars are loaded", and one caller loaded documents without them.
Verified live: the same click now opens on *Families (253) · 13 of 357*. **Not unit-tested** —
this is `App.svelte` orchestration, which `node:test` cannot run (apps/desktop/CLAUDE.md); the live
drive is the gate, and it is the only thing that would have caught it.

**Rejected: making the graph write its own vocabulary** (a "+ add to graph" button on a graph
node). It is the more obvious place and it is what ADR-017 A1 forbids — the graph is a read surface
over a vocabulary curated elsewhere, and letting it write would put two writers on one table with
no story for which wins. The graph's existing "Edit" action already deep-links into Manage keywords;
that is the seam.

**Verified live** on the real library, and the library is restored exactly. Opened Manage keywords
(13 of 357 · lens shows 13) → toggled **BM25** off (12 of 357, row leaves the lens) → the Graph
showed *"⚠ Graph is 1 concept behind your vocabulary. [Rebuild]"* → toggled it back on (13 of 357) →
the banner cleared. Both PATCHes 200, each followed by a family refetch and a `/api/concepts/graph`
refetch (the latch drop, working). DB after: 236 anzsrc/0 · 344 keyword/0 · 13 manual/1 — identical
to before, `BM25` back at `manual/1`. 0 new console errors; no horizontal overflow at 375px
(`famhead` 301/301 with the extra button).

**What it opens.** A user can now build graph vocabulary, so the caveats the ROADMAP files against
the graph become testable rather than theoretical: whether 13 concepts is too thin is a question
somebody can now answer by adding more. It also re-raises ADR-018's own consequence — *"the gap
distribution must be re-measured, not assumed"* — because the vocabulary is no longer fixed at
whatever the CLI left.

---
## 2026-08-30 (3) — Settings becomes a rail and five categories, because a flat list stopped scaling before it stopped growing

**What changed.** The Settings drawer is now a **category rail + one pane** instead of one flat
scroll of ten sections. Five categories — *Getting started · Documents · Provider & model ·
Retrieval · General* — in a new pure `settings/sections.ts` (+5 `node:test`s). The panel widened
from 420px to `min(760px, 96vw)` to pay for the rail; below 700px the rail becomes a horizontally
scrolling strip above the pane. Section markup and logic are unchanged — they were regrouped, not
rewritten.

**Why a rail and not accordions or a search box.** Accordions keep the flat list and add a click;
search answers "where is X" only for a user who already knows X is there. A rail answers the
question a first-time reader actually has — *what kinds of thing can I change here* — and it is the
seam that makes the next setting cheap: it joins a category, and the category says where it
belongs. That was the ask ("so we can add more settings later"), and it is why the category list is
data in a plain `.ts` rather than markup: it is testable, and adding one is an entry, not a
refactor.

**The landing category is read from state the shell already had.** `initialSection` returns `setup`
while any setup step is outstanding and `documents` otherwise, computed from `shell.setup` —
already loaded by App for the chat pane's banner. So a fresh install opens **on** the checklist
(ADR-034) rather than opening elsewhere and jumping a moment later when a fetch of our own lands,
and it costs no second request. The rail badges the outstanding count for the same reason, and only
while it is non-zero: a badge that is always there is decoration.

**Deliberately NOT `role="tablist"`.** Tabs carry a keyboard contract — roving tabindex, arrow-key
movement — and a rail that claims the role without honouring it is worse for a screen-reader user
than one that never claimed it. It is a `<nav>` of buttons with `aria-current="page"`; Tab reaches
every category, which is what actually happens. `svelte-check` caught the first attempt
(`<nav role="tablist">` is a non-interactive element with an interactive role) — the warning was
right about more than the element name.

**Two behaviours wired before they were needed, not after.** Switching category scrolls the new
rail item into view and resets the pane to the top. Both only bite once the rail outgrows its box —
which is precisely the "more settings later" case this restructure exists for. Without the second,
a pane scrolled deep in Documents opens General halfway down a page the user has never seen.

**Rejected: extracting each category into its own component.** It is the better long-term shape and
it is what a bigger panel will want, but every section leans on the same ~80 lines of local CSS
(`.hint`, `.err`, `dl`, `.switch-row`, the button styles), and Svelte scopes styles per component —
so five components means five copies, or a global-CSS refactor whose specificity ties with the
child components' own scoped rules and resolves on source order. The rail is the part that had to
land; `sections.ts` is where the extraction starts when the panel earns it.

**Verified live** at 1440x900 and 375x812, light and dark, 0 new console errors: all five
categories render their own sections; `.body` is `195px 510px` wide and `1fr` narrow; the rail
strip scrolls to keep the active item visible (`scrollLeft` 260, active fully in view) and the pane
resets (297 → 0); no horizontal overflow at either width (`scrollWidth === innerWidth`).

**What it opens.** Component extraction per category, once the shared CSS is worth moving. And the
categories are now the obvious home for settings that do not exist yet — the selective re-ingest
controls (ROADMAP 20/21) have a `Documents` to land in.

---
## 2026-08-30 (2) — The Graph tab is back, and its empty state now says the thing that is actually true

**What changed.** `GRAPH_TAB_ENABLED` → `true` (ROADMAP row 22). With it, `ConceptGraph` and
`GraphIndex` gained a **built-and-empty** state, distinct from the *never built* one they already
had.

**Why the flip needed more than the flip.** The tab was hidden on 2026-08-12 because an empty page
reads as a failure, and the review that hid it
(`docs/REVIEW_2026-08-12_release-readiness.md` §2b R4) asked for *either* a real empty state *or*
the hide. Only the hide was done. Flipping the flag alone would have shipped the original defect to
anyone whose graph is empty — and that is not hypothetical: a built graph with zero nodes fell into
the `{:else}` arm, so the rail rendered its filter box over nothing and said **"No concepts
match"**, blaming a query the user never typed, while the main pane invited them to *"Select a
concept"* from a list with none.

**And the empty state must not offer a build, because a build does not fix it.** Graph vocabulary
is curated, not automatic (ADR-018): a concept enters at `graph_include=true`, and **only**
`scripts/curate_concepts.py` sets it — `library.create_keyword_family` and
`concept_skeleton.promote_keyword` both create concepts with it **off**. Measured on this library:
593 concepts, `graph_include=1` on **13**, all `source='manual'`. So a fresh install's graph is
empty and no in-app action can change that. The new state says so — *"Choosing that vocabulary
isn't in the app yet, so rebuilding won't add anything on its own"* — and demotes the button to
*Rebuild anyway*. A spinner-and-hope would have been the version that reads as broken.

**Rejected: deleting the flag now that it is true.** The ADR's *second* reason for hiding is still
open — placement is entangled with the Project ADR — so this tab may yet move. The flag is what
keeps reversing or relocating the call at one line, which is the reason the file gives for having
flags at all.

**Verified live**: the tab renders 13 concepts with their gap badges, and selecting *contrastive
learning* draws the ego graph (5 links) plus "Appears in 10 documents" with per-document mention
counts. The empty state was driven by stubbing `/api/concepts/graph` to `nodes: []` in the page —
rail reads *"Nothing in the graph yet."*, pane reads *"No concepts in the graph yet"*.

**What it opens.** **In-app graph-vocabulary curation** — the missing route, and the thing that
makes this tab mean anything on a library other than this one. It touches ADR-018's rule about what
opts in, so it wants a decision before code. Filed as ROADMAP row 23.

---
## 2026-08-30 (1) — KI-53: the ingest record described every document as a PDF, and called a complete extraction broken

**What changed.** Two honesty fixes in what ingestion *records*, both filed on 2026-08-28 from the
first EPUB/HTML round trip and neither touching extraction itself, which was clean.

1. **`health.classify_document_health` no longer equates short with broken.** It opened with
   `chunk_count <= 1 → score -= 100`, so a 2 KB web article that extracted into one full chunk was
   filed `broken` and the document view rendered **"html · broken"**. A user's rational response to
   that is to delete a document that worked.
2. **`extractor_used` names the extractor that ran.** It was `config.PDF_EXTRACTOR`
   unconditionally, so an EPUB and an HTML file both recorded `pymupdf` — a PDF extractor that
   never touched them. New `extractors.extractor_name(path, pdf_extractor)` follows the same branch
   the dispatch does: `bs4`, `ebooklib+bs4`, `python-docx`, `odfpy`, `striprtf`, `verbatim`.

**The rule the classifier now follows.** *Broken* means the extractor failed, so every rule that
can return it is a statement about text missing **relative to its container**: nothing extracted at
all; a paged document whose pages produced a single chunk (two pages of prose cannot fit in one
1,000-char baseline chunk); a lone fragment under 200 chars. Where there is no container to compare
against — an HTML page, an EPUB chapter — the honest answer is that we cannot tell, so it withholds
the penalty instead of inventing one. `.txt`/`.md` are exempt from the fragment rule outright:
reading a short file back is not a failure, it is the file.

**Two count-based rules collapsed into one graded measurement, and that mattered.** The
`chunks_per_page < 2` floor fired at ~2,000 characters per page — an ordinary sparse paper — and a
flat `chunk_count <= 3` cost 50 points. Both are now **characters per page**, in two tiers: under
200 (a scrap per page) the text layer failed and the document is `broken`; under 500 (half a
baseline chunk) it is thin and merely penalised. Reviewing my own first version caught why this is
not cosmetic: I had kept a `pages > chunk_count` penalty beside the yield rule, and it filed a
3-page note with two *full* chunks as `broken` — with a 1,000-char baseline chunk, "fewer chunks
than pages" simply *is* "under ~1,000 characters per page", stated in the unit that misleads.
Zero blast radius on the live library: all 98 documents are `healthy` today, and a rule that only
ever fires later can move a verdict toward healthy, never away.

**The one hazard checked before touching anything.** `_EXTRACTOR_NAMES` is a *parallel* dict rather
than a second field on `_EXTRACTORS`, because `ingest.cache.extraction_fingerprint` **walks**
`_EXTRACTORS.values()` and a walk that trips falls back to the whole-module fingerprint — i.e. a
re-extraction of every cached document (KI-48, which cost 97 PDF caches for an EPUB-only change).
All ten fingerprints were captured before and after and are **byte-identical**. The drift a
parallel dict allows is bought back by `test_every_supported_format_names_its_extractor`.

**Rejected: passing the source file's size in as a yield signal.** It would catch the one failure
this leaves uncovered — a 500 KB JS-heavy HTML page yielding 900 characters — but a bytes→chars
floor has to hold across raw HTML, zip-compressed EPUB and DOCX, where the honest ratio differs by
more than an order of magnitude. That is a calibration exercise, not a bug fix. The blind spot is
named in the module docstring instead.

**Every new test was run against the pre-fix code and seen to fail** — 7 of 8 new unit tests (the
eighth, "a lone scrap is still broken", passed pre-fix for the wrong reason and is a loosening
guard, not a regression test), and the integration test on both halves in turn: `article.html
extracted into 1 chunk(s) and read broken`, then `assert 'pymupdf' == 'bs4'`. Health is asserted
first precisely so the extractor assertion cannot mask it.

**Gates:** pytest 2216/0 · mypy 93/0 · ruff clean · svelte-check 211/0 · node:test 181/181.

---
## 2026-08-28 (10) — CS1/CS2: the last spec item, and the last sentence describing the product Provenote used to be

**What changed.** Settings' **Source folder** is now **Library folder**, with a **Browse…** button
opening a single-select folder picker beside the text field. `pickPaths` gained `multiple` and
`title` (defaulting to today's behaviour, so no other caller moved). `readiness.setup_state`'s
first-run sentence was retired. Spec items CS1 and CS2 closed — **every AD/CS work item is now
built**.

**Why the relabel is not cosmetic.** One sentence — "Paste the full path to the folder holding your
documents" — described *both* models at once, and only one of them is still true. Since AD3b the
library is somewhere you **add** documents (copied in, or referenced where they live); it is not a
folder you aim the app at. The folder is now named for what it is: *where Provenote keeps the
documents you add — anything already in it can be indexed here too*, which keeps the
index-a-folder path honestly on offer without pretending it is the folder's definition.

**Why the text field stays.** The spec asked for it and the reason holds: a picker cannot express a
folder that **does not exist yet**, and the "folder doesn't exist yet" warning directly below is
about exactly that case. The picker *fills* the field rather than saving, so the confirm step is
the same one a typist would use.

**`multiple: false` matters more than it looks.** `pickPaths` defaulted to multi-select, which is
right for documents and wrong here: a dialog that lets you select three folders while the caller
silently keeps the first has lied about what it asked. The option defaults to the old value so
every existing caller is untouched.

**And the same stale sentence had a second home.** `readiness.setup_state` told first-run users to
"Point the app at a folder of PDFs, EPUBs, HTML, DOCX or Markdown", with the action "Choose a
folder below, then index it" — the pre-AD3b model again, on the *first* screen a new user reads.
Now: "Add documents from the Library, or index a folder…". Two surfaces, one wrong idea; fixing one
and leaving the other would have been worse than fixing neither, because it would look done.

**Verified live** in the Tauri window: the section reads **Library folder** with the path and a
**Browse…** button; clicking it opened a real folder picker titled "Choose the folder Provenote
keeps your documents in" while the button showed *Choosing…*; cancelling restored the button and
left the path — and the backend `source_dir` — unchanged.

**What it opens.** The spec's DoD is now down to the **one human drag** and its `docs_check
--strict` line. Nothing in Settings was reorganised (that is P7), deliberately.

---
## 2026-08-28 (9) — AD4: the empty state stops describing the old product and becomes the drop target

**What changed.** `LibraryPane.svelte`'s zero-document state is now a dashed drop zone carrying the
format list and an **Add documents** button into the AD1 chooser, highlighting from
`accept.dragging`. Where the accept surface does not exist it renders the reason and **no button**.
Spec item AD4 closed.

**Why.** The one screen whose entire job is to say "there is nothing here yet" said nothing about
how to change that. The window has always accepted a drop; a first-run user had no way to learn it.

**What the old copy was actually saying.** "Point doc_assistant at a folder of your documents" —
which named the **code identity** rather than the product (ADR-012 is wordmark-only, and this was
the one string in the app breaking it), and described the *pre-AD3b* model where the library was a
folder you aimed at rather than somewhere you add documents. Two stale things in eleven words, on
the first screen a new user sees.

**The spec said "alongside the existing demo-corpus offer", and there is no such offer.** The demo
corpus is a CLI script (`scripts/download_corpus.py`); nothing in the app offers it, and
`readiness.setup_state`'s `documents` step only says "Point the app at a folder". Recorded in the
spec rather than invented — building a download offer to satisfy the wording would have been a
feature nobody asked for, shipped on the strength of a phrase.

**Rejected: a DOM `dragover` handler on the zone.** Tauri intercepts the OS drag before the DOM
sees it, so it would never fire; the zone reads the same window-level signal the chooser and the
full-window veil already use. It is an aiming point and a label — dropping anywhere still works.

**Verified in both branches**, against a throwaway API pointed at an empty `DOC_DATA_DIR` so the
real 98-document library was never touched: the Tauri window rendered the zone, the format list and
the button; the browser rendered "Your library is empty / Adding documents works in the desktop
app." with `hasAddButton: false` — no dead control.

**What it opens.** CS1/CS2 is the last spec item, and it inherits the other half of the stale
sentence: `readiness.setup_state` still tells users to "Point the app at a folder". The drag
*highlight* is the one state not visually confirmed — the desktop-control grant here gives File
Explorer click-tier, which blocks drag-and-drop, so `accept.dragging` was exercised through the
chooser and not through a real drop.

---
## 2026-08-28 (8) — Delete stops meaning "delete the file": ADR-046's other half, and KI-52 with it

**What changed.** `delete_document(document_id, chroma_db, *, delete_file: bool = False)` — the
default is now **library-only**, and ADR-014's bin-first-then-remove is the opt-in branch.
`DELETE /api/library/documents/{id}` takes `delete_file`; `source_path` ships on every library row
(dataclass → pydantic → TS, one change); `LibraryDeleteConfirm.svelte` is now a two-option dialog
with the pure wording in a tested `deletetarget.ts`. Closes **KI-52**. ADR-046 → `accepted (built)`.

**Why.** ADR-046 §2 amended ADR-014 back in August and the half that *asks* was never built, so
"delete" still meant "delete the file" — right for a copy the app made, wrong for a file the user
keeps in their own folder and merely pointed the library at. The spec has carried it as outstanding
since AD3b.

**Naming the destination is part of the decision, not copy-writing** — the ADR says so. The
accepted risk of a per-delete choice is a mis-click, and the path is what makes the click informed.
Two consequences worth stating: the path is shown for **every** document, not only referenced ones
(a copy's path is equally worth seeing, and it means the UI needs no join to `SourceFile.origin`),
and where **no** path is recorded the destructive option is *not offered at all* — refusing to name
the target is refusing the guarantee the ADR asked for, and a test pins that.

**KI-52 fell out of it, which is why it was deferred to here.** The registry row now goes with the
file — and only with the file. With `delete_file=False` the file is still on disk and no longer
indexed, which is exactly what `derive_status` calls `new`: the row is still true, and keeping it
is what lets the next scan offer the file again. `_forget_source_row` resolves each row through
**its own root**, so a same-named file under another root survives; that is the mistake that once
deleted an unrelated document out of the library, and it has a test.

**Flipping a default breaks callers silently, and the suite is what said so.** Six tests failed:
five asserted ADR-014's unconditional binning and were updated to opt in (they are now testing the
opt-in branch, which is what they always meant), but the sixth was **production code** —
`library/pins.py::remove_pinned_sources`, the demo-corpus removal, whose docstring says "Recycle
Bin first". Left alone it would have quietly stopped binning, stranding every downloaded demo file
with nothing pointing at it. It now passes `delete_file=True` explicitly, with a comment saying why,
so the next default change cannot move it either.

**Verified live** in the Tauri window on a throwaway document: the dialog opened with *Remove from
library* preselected, read "The file stays where it is. Its 1 chunk leaves the search index."
(singular verb — the first draft said "1 chunk leave", which is how a confirm dialog looks
machine-generated at the moment it asks for trust), and named the destination as
"Moves C:\Projects\doc_assistant\d…ources\delete-dialog-demo.md to your Recycle Bin". Choosing
the second option changed the button from *Remove* to *Delete both*; confirming binned the file,
dropped the row, and left **0** rows in `/api/sources` where the old code left one reading
`missing` forever.

**What it opens.** CS1/CS2 and AD4 are the last spec items, plus the one human drag. The
`shortenPath` middle-ellipsis is deliberately dumb about path separators — it keeps both ends by
character count, which is right for the cases seen so far and would need thought for a path whose
filename alone exceeds the budget.

---
## 2026-08-28 (7) — EPUB and HTML finally went through the UI, and the extraction was fine; what the row *says* about them is not

**What changed.** No production code — a verification, plus `.claude/KNOWN_ISSUES.md` KI-53 and the
spec's Definition of Done. `docs/specs/feature-add-documents.md` had a standing "**not done until**
an EPUB and an HTML file have been added and indexed through the UI", because the corpus is 98/99
PDF and this branch shipped the EPUB/HTML extractor work with those paths never exercised in the
wild. They have been now.

**What was done.** `tests/fixtures/documents/article.html` and `treatise.epub` — the project's own
frozen round-trip fixtures — were added *together* through the new chooser, which also closed the
untested multi-file path. Result: 2 added, 0 errors; correct **EPUB** and **HTML** type badges and
new type rows in the sidebar; the grid auto-refreshed 98 → 100 on completion.

**Extraction is genuinely good, which is worth stating because the row says otherwise.** Accents
survived intact (`Renée`, `Björn Åkesson`, `Méthodes`, `Résultats`), the HTML table round-tripped
with its caption and rows, and all three chrome markers — `NAVIGATION_CHROME_MARKER`,
`SCRIPT_BODY_MARKER`, `FOOTER_CHROME_MARKER` — were stripped. Both documents reached both vector
stores (1 baseline chunk each; 3 and 2 child chunks) and the keyword index.

**What it found (KI-53).** `classify_document_health` returns `broken` for any document with
`chunk_count <= 1` before considering another signal, so both fixtures were filed as broken and the
document view renders it as **"html · broken"** — a failure verdict on a clean extraction, and the
first thing a user reads about their new document. And `extractor_used` is passed as
`config.PDF_EXTRACTOR` unconditionally, so an EPUB records `pymupdf`: provenance that is simply
false, in an app whose selling point is provenance.

**A retrieval scare that was not one.** Two chat turns failed to surface the fixtures, which reads
as a retrieval bug until you count what they were competing with: 98 real neuroscience papers, many
about layer-five pyramidal neurons, against one 2 KB synthetic article. The keyword index settles
it — `Delacroix` matches exactly 1 chunk of 39,710 — so the content is indexed and this was
ranking. Recorded as *not* a defect, because filing it as one would have been the easy and wrong
call.

**One thing I got wrong mid-check, recorded so it is not repeated.** I read
`/api/library/documents/{id}/chunks` returning `{"detail":"Not Found"}` as a broken endpoint before
checking a PDF, which 404s identically — the route does not exist and I had invented the path. No
defect; my error.

**Rejected: fixing KI-53 here.** The health thresholds are a tuning decision about what "broken"
should mean for a non-paper, not a patch — and changing them re-classifies every short document in
the corpus at once. The `extractor_used` half is genuinely trivial, but shipping half of a KI makes
the other half harder to find later.

**What it opens.** KI-53. Also one under-evidenced thread noted there rather than claimed:
`Akesson` returns 0 keyword hits against text holding `Åkesson`, so the tokeniser may not fold
accents — one observation, not a measurement.

---
## 2026-08-28 (6) — Undo now finishes the job: the document, its chunks and the reference go with the row (KI-51 closed)

**What changed.** `library/documents.py`: the record-removal half of `delete_document` is extracted
as `purge_document_record` (row + cascades + chunks + figure dir + cached markdown — everything
*except* the file). `library/add.py::undo_add` takes an optional `chroma_db` and gains
`_row_path`, `_purge_document_for` and `_drop_root_if_emptied`; the undo route passes
`controller.rag.db`. Six tests in `tests/unit/test_library_add.py`.

**Why.** KI-51 parts 1 and 2, filed this morning from the walkthrough. Undo removed the registry
row and stopped, so an add that had already been indexed was only half undone: the `Document` row
and its chunks survived, and the library went on listing — and could still cite — a document whose
file undo had just deleted. In reference mode the `source_roots` row survived too, so the next scan
re-found the file as `new` and the next "index all" re-ingested exactly what the user had undone.

**Why the extraction rather than a second implementation.** The two callers differ *entirely* on
the file and *not at all* on the record: delete bins the source, undo either removes the copy the
app made or must not touch the file at all. Sharing the record half is what stops undo growing a
drifting second answer to "what does removing a document mean".

**The guard is the feature, not the removal.** A path can already carry a document the user has had
for months — under ADR-047 a replacement even inherits the previous document's id — so
`_purge_document_for` removes one only if it resolves to the same path *and* was added inside
`UNDO_DELETE_WINDOW_SECONDS`, the same window that lets undo delete a file. A decline is logged,
never silent. Two of the six tests exist to prove it *declines*.

**Rejected: purging without a `chroma_db`.** Dropping the row while its chunks stayed in the index
would leave chunks retrievable with no document behind them — the same orphan, mirrored. A caller
that cannot reach the index cannot finish the job, so it does not start; the row is left alone and
a test pins that.

**Verified live against the real 98-document library**, in both directions: an indexed add went
98 → 99 documents / 39,705 → 39,706 chunks and returned to **98 / 39,705** on undo with no
`documents` or `source_files` row left behind; a reference-mode add went 1 → 2 roots and back to
**1**, with the user's own file byte-identical (sha256 before and after) and the scan no longer
offering it as `new`.

**The route change broke a test, and that is worth recording.** Reaching for `controller.rag.db`
in the undo route made `test_undo_removes_what_add_created` fail with
`'FakeController' object has no attribute 'rag'` — the full suite caught it, the targeted runs did
not, because the six new unit tests exercise `undo_add` directly and never go through the route.
The fake now carries a `.rag.db` the way the real controller does (the idiom already existed in
`test_document_delete.py`), and a new API-level test pins the *wiring* rather than the removal,
since the wiring is the half that was actually wrong.

**What it opens.** **KI-52**, found while checking the cleanup: `delete_document` never touches
`SourceFile`, so deleting a document from the Library leaves a registry row that `/api/sources`
reports as `missing` — the app misreporting its own action. It is the same seam mirrored, and it is
deliberately *not* fixed here: it lands in the one path that bins a user's file, where the
2026-08-27 review already found two file-deleting bugs, so it deserves its own change and its own
tests rather than a tidy-up at the end of an unrelated branch.

---
## 2026-08-28 (5) — The progress channel pays for itself twice: the grid refreshes when a run ends, and undo stops racing it

**What changed.** `ingestRun.completedRuns`, a counter bumped once per *witnessed* run completion,
plus `isCompletion(previous, next)` in the pure module. `App.svelte` watches the counter and
re-reads the document list and `health` when it moves. `AddDocuments.svelte` records that it
started a run (`startedIngest`), derives `indexing` from the shared watcher, says "Still indexing —
progress is in the status bar", and disables **Undo all** until the run ends. Closes **KI-51 part
3**; parts 1 and 2 stay open.

**Why.** Both are the same defect as entry (3), one step further along. Everything that shows the
corpus refreshed on the 202 — i.e. *before* the run had indexed anything — so a just-added document
was missing from the grid and the chunk count was stale until something else happened to refresh
them. And "Added N" with a live **Undo all** appeared while the worker was still reading the file.
The progress channel already knew when a run ended; nothing was listening.

**Correction to KI-51's original filing, made while writing this.** That second defect is *not* the
data-loss bug it was logged as. `routers/sources.py::_running` predates this branch and already
409s undo while an ingest is in flight, with `status.state` set synchronously before the 202
returns — so there is no gap and no deleted file. Clicking Undo mid-run produced an unexplained
error. The fix is still right, for a smaller reason: the UI offered a control that could only fail.
KI-51 has been corrected.

**Why a counter and not a callback list or a boolean.** An `$effect` that reads a counter re-runs
on every completion, which is the subscription shape Svelte 5 already gives us — no registry to
leak, no flag to reset. `lastSeenRun` in `App` is a plain `let`, not `$state`, so writing it inside
the effect cannot retrigger it.

**Why `isCompletion` is a function and not two inlined conditions.** It carries a real failure:
without the `previous?.state === 'running'` half, the `done` left behind by a run that ended hours
ago is the state the *first* poll after launch sees, so every app start would light the progress
bar and fire a spurious refresh. That is a rule worth a name and three tests, not a `&&`.

**Rejected: making `indexPaths` await the run.** It would have turned a 202 into a long block held
by the sheet, and given the sheet a second, private view of a run the status bar already tracks.
The gate reads the shared watcher instead. **Rejected: disabling Done as well** — the sheet stays
closable throughout; only the destructive control waits.

**Verified live** in the Tauri window: on apply, "Still indexing" appeared and Undo greyed out; on
completion the line cleared, Undo re-enabled, and the grid went 98 → 99 with a new **MD** type row
and **Today 2**, with no manual reload. The test document was deleted afterwards; the library is
back to 98 / 39,705.

**What it opens.** KI-51 parts 1 and 2 — undoing a *finished* indexed add still leaves the
`documents` row, its chunks, and (in reference mode) the registered root. The race is gone; the
orphan is not.

---
## 2026-08-28 (4) — Two things the add flow never said out loud: where the drop is, and what "Earlier" means

**What changed.** `AddDocumentsChooser.svelte` (new) — a dialog between the "Add documents" button
and the OS picker, carrying a drop zone and two browse routes. `accept.svelte.ts` gains `chooser`
state plus `openChooser`/`closeChooser`; `LibraryPane` and the app menu open the dialog instead of
calling `pickDocuments()` directly. `library.ts`: the `earlier` bucket is now labelled **"Over a
month ago"**. Two new `Icon` glyphs (`chevron-up`/`chevron-down`, Lucide).

**Why, from the user directly.** The button went straight to Explorer, which meant the *other* half
of AD1 was invisible: the window has always accepted a dropped file, but nothing on screen ever
said so unless you were already mid-drag. The dialog names both routes in one place, and adds the
sentence a first-time user actually needs — nothing is copied or indexed yet, and the placement
choice is still ahead of you.

The relabel is the same problem in one word. `earlier` means *added more than 29 days ago*, but it
sat directly under "This month" where it read as a calendar term — this year? last year? A relative
bucket has to name its own boundary or the reader cannot tell which of two neighbours a document
will land in.

**Rejected: adding an "Added today" bucket.** The user asked for one, and it already exists —
`today` is the first of four, and `dateGroups` hides empty buckets on purpose (L1's honest-empty
rule). The reason it had never appeared is that nothing had been added *that day* and, when
something finally was, the run reported nothing. Building a second one would have papered over
entry (3)'s bug with a permanently-visible "Today 0". Verified after the fix: **Today 1**.

**Rejected: a DOM drop handler on the zone.** Tauri intercepts the OS drag before the DOM sees it,
so it would never fire. The zone highlights from the same window-level `accept.dragging` signal
that drives the full-window veil — it is an aiming point and a label, not a second target.

**What it opens.** The spec's "one human drag" is still open: the desktop-control grant here gives
File Explorer click-tier only, which blocks drag-drop, so the picker route was driven end to end
and the drop route was not.

---
## 2026-08-28 (3) — The indexer now says where it is, because "I was not even sure it was working" was literally true

**What changed.** A position channel from the ingest loop to the status bar.
`ingest.main` takes an optional `on_progress(done, total, current)` and calls it before each
document and once after the loop (`_report` guards it — a progress sink must not be able to kill a
30-minute run). `_IngestStatus` gains `total`/`done`/`current`, `GET /api/ingest/status` reports
them, and the route installs a sink that writes *position only*. Frontend: `core/ingest.ts` (pure,
tested), `core/ingest.svelte.ts` (the shared poller), `shell/IngestChip.svelte` (a determinate bar
on the right of the status bar plus a click-through detail panel).

**Why.** The user added a document and could not tell whether anything had happened. That was not a
misreading: `POST /api/ingest` is a 202, and `added`/`skipped`/`errors` were only written *after*
`ingest_fn` returned, so for the entire duration of a run the endpoint answered
`{running, 0, 0, 0, message: null}`. There was nothing to render. Their document had in fact
indexed — 216 chunks, confirmed in the row — which is exactly the failure mode: it worked, silently.

**Why determinate rather than a spinner.** `to_process` is materialised before the first document,
so the total is genuinely known and a real "4 of 12" costs one callback. A spinner would have been
cheaper and would have taught the user nothing about how long to wait.

**The distinction the whole design turns on: position is not outcome.** `added`/`skipped`/`errors`
stay end-of-run totals and are 0 throughout, so nothing may render them as progress — a document
counted at position 3 can still fail at position 4. That separation is asserted directly
(`test_status_carries_position_while_a_run_is_in_flight` checks the outcome triple is still zero
mid-run), and `fraction()` returns **null**, not 0, for an uncounted batch so the bar shows
indeterminate rather than claiming 0%.

**Rejected: passing `on_progress` only to callables that accept it.** Five injected test fakes had
explicit keyword-only signatures, and sniffing them with `inspect.signature` would have made the
seam's contract unstateable. The fakes were updated instead — that seam is a contract, and the
tests exist to notice when it changes.

**Verified live**, in the Tauri window against the real library: the bar went
"Preparing to index…" (indeterminate) → "Indexing 1 of 1" (determinate, filled), the panel opened
with the source dir and the file in flight, and the 8-second linger retired it. Three throwaway
documents were added and deleted; the corpus is back to the user's 98 / 39,705.

**What it opens.** KI-51 part 3 is *not* fixed — the sheet still says "Added N" and offers "Undo
all" while the run it started is in flight. The bar makes that visible rather than resolving it,
and undoing mid-run still deletes a file out from under the indexer. Also unfixed: the library list
refreshes on the 202, so a just-added document does not appear until the next refresh.

---
## 2026-08-28 (2) — The add-documents feature was driven for the first time, and undo turns out to stop one table short

**What changed.** No production code — a walkthrough, and `.claude/KNOWN_ISSUES.md` KI-51. AD1-AD3b
is the branch's headline feature and had never been exercised; it now has been, in the native Tauri
window and against the live 97-document library, which was restored to baseline afterwards and
verified row-for-row against `data/library.db.bak-20260828-prewalkthrough`.

**What works, confirmed by driving it.** The picker stages paths and the review sheet opens with
them. Verdicts are right and come from the server: an unsupported file carries its advisory, and a
*renamed copy* of an already-ingested PDF was caught as a duplicate by content hash — exceptions
sorted above clean files, as the docstring claims. Both placement modes work: `copy` puts the file
in the library with `origin='copied'`, `reference` registers a new `referenced` root and copies
nothing. Index-now indexes. Undo, clicked in the UI for the first time, deleted the copy and left
the user's own file **byte-identical** (sha256 before and after) — the ADR-014 amendment holding.
In a plain browser the button is disabled and says why, which is the honest degradation.

**Why it found what the gates could not.** `svelte-check` (205/0) and `node:test` (150) cannot
reach a `.svelte` component at all, and the Python suite tests `undo_add` against `source_files`
only. Nothing anywhere indexes a document and *then* undoes it — which is the one sequence that
breaks.

**What it found (KI-51, three parts).** Undo drops the registry row and stops: after index-now the
`documents` row and its 4 chunks **survive**, so the library still lists — and can still cite — a
document whose file the app just deleted (97->98 documents and 39,131->39,135 chunks persisted
across the undo; the proper `DELETE` afterwards reported `chunks_removed: 4`). In reference mode the
`source_roots` row survives too, so the next scan re-discovers the file as `new` and the next
"index all" would re-ingest exactly what was undone. And because `POST /api/ingest` is a 202 that
`indexPaths` does not await, "Undo all" is offered while the indexing it would undo is still
running — though **not** as the data-loss path this entry first claimed: entry (5) records the
correction, the API already refuses undo mid-run with a 409.

**Rejected.** Fixing it in this session. It is scope, not a regression — every guard the 2026-08-27
review added to `undo_add` held — and the branch is being merged for the CI fix, not for new
behaviour. Filed with the reproduction rather than patched in a hurry beside a merge.

**What it opens.** KI-51. A fix belongs in `undo_add` (which already resolves each row through its
own root, so it can also drop a `documents` row it can prove the same add created, and drop a
`referenced` root once its last file goes) plus an awaited `indexPaths`. Any fix must keep
reference mode incapable of touching the user's file.

---
## 2026-08-28 (1) — CI had been red for four commits over two path literals that only mean what they say on Windows

**What changed.** `tests/unit/test_document_identity.py` — the hardcoded `C:\library\...` constants
became platform-selected, and the "same file, different spelling" probes became a
platform-appropriate tuple. No production code.

**Why.** `test_the_path_comparison_is_normalised` and
`test_the_path_index_answers_exactly_as_the_table_scan_did` asserted that `c:\LIBRARY\paper.pdf`
and `C:/library/paper.pdf` resolve to the same document as `C:\library\paper.pdf`. They do — on
Windows. `_pathkey` is `os.path.normcase(os.path.abspath(...))`, deliberately the **host** OS's
semantics, because `source_original` is always written by the machine the library runs on. On the
Linux runner `normcase` is identity and a drive letter is not a drive, so both tests failed while
passing on every developer's machine. Green locally, red in CI since `a289d3f` (2026-08-25) —
four consecutive failed runs, all the same two tests.

**Rejected: skipping the tests off Windows.** It would have gone green by never running the
ADR-047 fallback in CI again — the opposite of what the failure was telling us. Instead each
platform is probed with spellings that are genuinely equivalent *on it*: case and separators on
Windows, redundant `.`/`..` segments on POSIX. Both branches were verified before pushing — the
Windows one by running the suite, the POSIX one by rebinding `os.path` to `posixpath` and checking
every chosen spelling collapses onto `SRC` and nothing else does. That check earned its keep:
`//library/paper.pdf` does **not** collapse (POSIX reserves a leading double slash and Python
preserves it), so it was excluded rather than shipped as a third probe that would have failed in CI.

**What it opens.** Nothing structural, but it names a gap: CI runs Linux and every developer runs
Windows, so a path-literal assumption is invisible until push. The two files that carry `C:\`
literals are now `tests/unit/ingest/test_registry.py` (one, harmless — it asserts a rejection) and
this one.

---
## 2026-08-27 — The AD3b migration, run against a real pre-AD3b database and then given the test it never had

**What changed.** `tests/integration/test_source_roots_migration.py` — 8 tests over the project's
second rebuild migration, which shipped with **none**.

**Why it was worth checking by hand first.** `_migrate_source_roots` does
`CREATE temp → copy → DROP TABLE → RENAME` on a table holding data users cannot regenerate (their
`excluded` flags), and it runs on every `init_db` — every app start. `data/library.db.bak-20260824-preroots`
is a genuine 97-document pre-AD3b database, so it was run against a **copy** of it (the original's
mtime is unchanged; the live `data/library.db` was never opened). Result: 27 tables → 28, 97 rows →
97 with every `rel_path`, `size`, `excluded` and `source_sha256` byte-identical, all backfilled to
the library root, 0 rows pointing at a missing root, `PRAGMA foreign_key_check` clean, second run a
no-op.

**And then the same database was driven, not just inspected** — KI-49's lesson is that a component
proved is not a system proved. A scan with the sources folder absent left all 97 rows and reported
them `missing` with `root_available=False` (an unplugged drive is not a deletion); restoring two
files produced 2 present / 95 missing and **no duplicate rows**; `library.add.inspect` found a
byte-identical file and returned `duplicate_of='library:…'`; and the new UNIQUE and the FK both
refused a bad insert.

**⚠ The `CreateIndex` loop in `_rebuild_table` is load-bearing for CORRECTNESS, not tidiness, and
the code comment undersells it.** Removing it and re-running these tests leaves the migrated table
with `sqlite_autoindex_source_files_1` — the primary-key autoindex — **and nothing else**. The
composite key is declared as `Index(..., unique=True)` in `__table_args__` rather than as a
`UniqueConstraint`, so it *lives in an index*: without the loop, `uq_source_files_root_rel` never
exists and the key the whole migration is for is not enforced at all. AD2's
`ix_source_files_source_sha256` goes with it. Three of the eight tests catch this.

**Rejected:** keeping the one-off script as the record (it proves the migration works once, on one
machine, against one database — the pre-AD3b DDL is now pinned verbatim in the test instead) ·
asserting on the backup file itself from the test suite (it is gitignored local data; a test that
only runs on this machine is not a gate).

**What it opens.** `.claude/REVIEWS.md` still records backend and frontend as **never** reviewed as
a whole — this pass reviewed a diff, not the modules. The add-documents UI has still never been
launched: `svelte-check` and `node:test` cannot see a mount failure (apps/desktop/CLAUDE.md).

---
## 2026-08-26 (5) — The loose thread from entry (4): a rename could silently re-extract the whole corpus

**What changed.** `_extraction_closure` now returns `{attribute name: function}` and the caller
hashes the **object it was handed** instead of re-resolving the name with
`getattr(extractors, name)`. The walk keys on the name a function is *bound to on the module*
rather than on its `__name__`.

**The defect, found while writing the finding-12 test rather than by reading the code.** The
closure returned names; the caller looked each one up again. A function whose `__name__` differs
from its attribute — an alias, or any decorator that does not carry `functools.wraps` — made that
lookup raise `AttributeError`, the blanket `except` caught it, and the per-format fingerprint
silently became the **whole-module** one. Direction-safe (it over-invalidates) and expensive in
fact: a corpus-wide re-extraction, **61 min for 97 documents and ~55 h projected at 10,000**,
triggered by a rename that could not change one byte of output. Entry (4) left it alone as
out-of-scope; it is fixed here.

**A second copy of the same mistake, in the same function.** `blocked` was built from
`fn.__name__` too, but it is tested against the attribute names the walk reads out of `co_names` —
so the same mismatch would have put an unmatchable string in `blocked` and quietly left the
*other* formats' entry points inside this format's scope. That one has no symptom at all: it
under-invalidates, which is the KI-40 failure the whole layer exists to prevent. Both now derive
from one `id(obj) -> attribute` map over `vars(extractors)`.

**Two tests, and the tell is precise.** Under the defect the format's fingerprint *equals*
`extraction_fingerprint(None)`, so `test_a_renamed_extractor_does_not_silently_re_extract_the_corpus`
asserts it does not — and that no `extraction_fingerprint_fallback` was logged, via
`structlog.testing.capture_logs`. The complement pins that the other six formats are untouched.
Both verified against the reinstated name round-trip: red before, green after.

**Rejected:** returning the constants as objects too (their names *are* attribute names by
construction — each came from a successful `getattr` on the same module in the same call — so
that lookup cannot raise; adding churn there would be motion, not safety) · keeping a name-keyed
API and hardening the lookup with a `getattr(..., None)` guard (it would silently drop the
function from the hash instead of raising, turning a loud over-invalidation into a quiet
under-invalidation — strictly worse).

---
## 2026-08-26 (4) — The rest of the branch review: nine findings, and one of them was hiding behind a test that could not fail

**What changed.** Findings 6-14 from the `feat/eval-comparability` review, fixed. Two are scaling
work against the 10k-document contract, four are correctness, three are honesty-of-the-record.

**⚠ Two guards in this branch asserted things that were true by construction.** Both were written
to catch a real defect and neither could ever fail:

1. `extraction_fingerprint` raised if the format's extractor was "unreachable from
   `extract_to_markdown`" — but that extractor is passed in as a **seed**, and every seed is
   recorded unconditionally. That is the entire reason seeds exist (the `_EXTRACTORS` dispatch is
   a runtime lookup no static walk can follow). The guard is gone; the invariant it was named for
   is now `test_a_change_to_a_formats_entry_point_moves_only_that_format`, which patches each
   format's entry point and asserts that format's fingerprint moves and no other's does.
   **Verified by deleting the seeds: the new test fails, the old one passed.**
2. `test_a_format_entry_point_is_inside_its_own_scope` did the same thing in the test suite —
   handed `_extraction_closure` the entry point as a seed, then asserted the seed came back.
   Replaced by the behavioural version above.

That is the same shape as the `test_budgets_are_ordered` finding earlier today. Three in one
branch is a pattern worth naming: **a guard written from the inside of the mechanism tends to
assert the mechanism's own construction.** The behavioural form — change the input, watch the
output move — cannot be satisfied that way.

**6 · The identity fallback was O(documents²).** `_existing_document_id` re-read every Document
row to resolve one normalised path, and during a corpus-wide re-extraction *every* hash moves, so
that fallback is the path taken for every document rather than the exception. `build_path_index`
reads it once per run and `main` threads it through. Deliberately not module-level state: a stale
map in the long-lived API process, or shared between tests, resolves a document to an id the
database no longer holds. Safe to build once because each source path is processed exactly once.

**7 · `repoint_figures` moved rows without moving the directory.** When the destination hash's
directory already existed the rename was skipped — and the row updates ran anyway, rewriting every
`image_path` into a directory that does not hold those crops and leaving the real ones under a
hash `hashes_with_no_figure_rows` would then read as dead and delete. It now bails, the same trade
the `OSError` arm already made: a stored path that resolves beats a tidy hash.

**8 · The review sheet never said *which* file a duplicate matched.** The server always sets
`advisory` on a duplicate, so the branch naming the match was unreachable. Reaching it exposed the
other half: `duplicate_of` is a `source_key`, so naming it raw would have shown the user a root
uuid. `sourceKeyName` renders the filename, and it lives in `accept.ts` rather than the component
because a `.svelte` file cannot be reached by `node:test` (apps/desktop/CLAUDE.md).

**9 · `resolve_run_id` let LIKE wildcards through.** `_` matches any single character, so `a_c`
matched `abc12345` — and because exactly one run matched it resolved *silently*, handing back a
run the caller never named, which is the one outcome that resolver exists to prevent. Escaped
rather than character-validated: a whitelist would also have refused the synthetic ids the tests
and older stores use, and refusing a legitimate id is a worse failure than the one being fixed.

**10 · `--workers --help` named the wrong default** (`balanced`; it is `light`).

**11 · An unplugged reference drive failed the whole selection.** An unreachable root contributes
nothing to `on_disk`, so validating against it reported every one of its files as `unknown` and
raised — a 400 for the entire request, blocking the library that *is* present, which is exactly
what `resolve_selection`'s own docstring said it avoided. Its files are dropped with a count now,
as the implicit branch already did. A guard test pins that an unknown file under a *reachable*
root still raises, so this did not soften validation generally.

**13 · `library/citations.py`'s `__all__` declared two symbols it does not own** and hid the six it
does. Latent (nothing star-imports it) but wrong for anything honouring `__all__`.

**14 · `reresolve_stored_citations` opened a nested session per citation row**, each scanning the
document table twice — O(citations x documents). `load_library_candidates` reads the library once
and `match_to_library` takes it. **That rewrite fixed a bug nobody had filed:** the DOI rule was
`Document.doi.ilike(parsed.doi)`, and `ilike` reads `_` and `%` in the *parsed* DOI as wildcards.
DOIs legitimately contain underscores, so `10.1234/abc_def` could match a different paper. Same
defect as finding 9, in a different table.

**Verification.** Every new regression test was run against the real pre-fix code and seen to fail
— including one where the first attempt did **not** fail, because the revert was not faithful
(replacing the `dest_dir.exists()` guard let the rename raise `FileExistsError`, which the OSError
arm caught and turned into the same return value). Restoring the actual original made it fail.
A revert that does not reproduce the bug proves nothing about the test.

**Rejected:** a `source_key` column on `Document` with a migration for finding 6 (the run-scoped
map costs one query and no schema risk; a column is the right answer if this ever needs to be
correct across a run rather than within one) · validating the run-id character set instead of
escaping (see 9) · fixing the fingerprint's silent whole-module fallback when a function's
`__name__` does not match its module attribute — **found while writing the finding-12 test, not
filed before, and left alone**: it degrades safely (over-invalidates) but its cost is a full
re-extraction, so it is worth its own look.

**What it opens.** The `__name__`-mismatch fallback above. The nine findings are closed; the five
from entry (3) were the severe half.

---
## 2026-08-26 (3) — Review of `feat/eval-comparability` before merge: five findings fixed, and two of them could delete a file

**What changed.** A full `/code-review` of the branch against `main` (15 commits, 100 files,
+19,398/-826 — none of it read by anyone but its author) produced 14 findings. The five most
severe are fixed here; the rest are recorded in the review and left for a follow-up.

**⚠ The two that mattered were both about deleting the user's files, and neither raised anything.**

1. **`scan_root` claimed ownership of folders it had no business owning.** It never set `origin`,
   so every row it created took the column DEFAULT — `'copied'`, the value that tells delete it may
   bin the file. Reference *one* paper out of a Zotero folder and `register_root` registers that
   folder; the next scan then walks it and marks **every other document in it** as app-owned.
2. **`undo_add` built its delete path as `library / rel_path` whatever root the row belonged to.**
   Combined with (1): a key naming `notes.pdf` in the user's Zotero folder deleted an unrelated,
   months-old `notes.pdf` out of the library folder. Reproduced end to end before the fix; the
   Zotero file survived and the library file did not.

   `origin` is now stated by the scan from its root's kind, and the impossible combination
   (`copied` under a `referenced` root) is repaired in place on the next scan — self-healing rather
   than a migration, since the scan is what owns these rows. The reverse is legal and untouched: a
   file already inside the library folder registers under the *library* root with
   `origin='referenced'`. The delete now resolves through the row's own root, requires the library
   root, and requires the row to be younger than `UNDO_DELETE_WINDOW_SECONDS` — undo is an undo,
   not a delete-by-key for anything the app ever copied in. A declined delete is logged; the row
   goes either way, because a file left for the next scan to re-register is recoverable and a
   deleted one is not (KI-49).

**3 · A failure that took the failure report with it.** `apply_add` caught `(OSError, ValueError)`,
so the registry's `(root_id, rel_path)` uniqueness — reachable by re-referencing a path, or by a
library file whose size changed since the last scan so `inspect`'s size index missed it — raised
`IntegrityError` straight out of the function. The client got a 500 and lost `added` entirely,
which is the one thing undo needs. Now reported as a per-file failure like any other, with a
sentence instead of a driver's SQL string.

**4 · A failed copy outlived its own rollback.** `shutil.copy2` runs inside the `session_scope`,
and the filesystem does not roll back. An orphaned copy is invisible to `undo_add` (no row, no key)
and perfectly visible to the next `scan_root`, which would adopt a document the user was just told
had failed to add. The destination is now claimed *before* the copy, so a copy that dies partway is
cleaned up too.

**5 · Exclusions silently stopped applying.** `_drop_excluded` swapped `f.resolve()` for
`registry.pathkey`, which normalises case and separators but deliberately does no filesystem read —
so it cannot reconcile an 8.3 short name, junction or symlink against the *resolved* form every
writer of `SourceRoot.path` stores. Reproduced: `skipped=0`, no error, no warning, Decision 5's
standing exclusions quietly ignored. `_resolve_walk_root` now resolves in both branches (it already
promised to in its docstring), `_seed_library_root` stores resolved like the other two writers, and
the precondition is stated where it can bite.

**6 · The worker budgets were not a ladder.** `balanced` returned 1 worker at 4 cores and `full`
returned 1 at 2, while `light` returned 2 — so a user on an ordinary laptop moving *up* from the
default to speed up ingest silently got serial extraction. Each rung is now floored at the one
below it.

**⚠ The ordering test existed and could not fail.** `test_budgets_are_ordered` asserted exactly
this invariant, pinned to `cpu_count=32` — the one region where the fractions were still monotonic.
The next test looked straight at `cpu_count=2` and asserted only `1 <= n <= 2`, which passes for
both the right answer and the wrong one. A ladder is a claim about a whole range; testing it at one
point tested it where it could not fail. It now sweeps 1..64, and a second test compares the rungs
to each other rather than to a range.

**Every regression test was verified to fail without its fix** — all 8, by reverting the five fixes
and re-running. A guard that has never been seen red is not a guard (KI-41).

**Rejected:** making `registry.pathkey` resolve (its no-I/O property is deliberate, and it also
keys paths whose file is gone, where `resolve()` does nothing useful) · a migration for the bad
`origin` rows (the scan already rewrites these rows every run, so repairing there needs no new
machinery and heals databases that never run a migration) · scoping `undo_add` to un-indexed rows
(the sheet indexes before offering Undo when *Index now* is ticked, so it would break the ordinary
path).

**What it opens.** Nine findings remain unfixed and are listed in the review: the `_existing_document_id`
O(n²) path scan against the 10k-document contract, `repoint_figures` rewriting rows without moving
the directory, the dead `duplicate_of` branch in the review sheet (the user is never told *which*
file a duplicate matches), `resolve_run_id`'s unescaped LIKE wildcards, and five smaller ones.

---
## 2026-08-26 (2) — KI-45 part 1: a citation link now has to agree on the title, and one word is enough to disagree

**What changed.** `resolution_is_credible` moves into `ingest/citations.py` (its dependencies all
live there; the library layer re-exports it), `match_to_library`'s rules 2 **and** 3 now call it,
a word-coverage test joins the ratio, and `--reresolve` re-points stored rows without
re-extracting. **Applied to the live library: 16 links → 41.**

**The defect.** Rule 2 matched **first-author surname + publication year with no title comparison
at all**. On a 97-document corpus holding many same-year papers by common surnames it fired
constantly: 13 of 16 stored resolutions were false, one document had 9 of its 11 links pointing at
two unrelated papers. A read-side guard (2026-08-10) hid this from the References block by
re-checking at render time, but every other consumer — `cited_by`, the citation-network view, the
concept graph's provenance annotators, the CLI — still trusted the rows.

**One predicate, used everywhere.** Rule 2 now only *narrows the candidates*; the title decides.
Rule 3 calls the same function rather than comparing ratios itself, so the write rules and the read
guard cannot drift apart again — which is exactly how the surname+year rule came to disagree with
the reference list's own idea of a credible link.

**⚠ AND THE FIRST VERSION OF THE FIX CREATED THREE NEW FALSE LINKS.** Verifying the 40 proposed
links by hand — rather than trusting the count — found *"Bidirectional recurrent neural networks"*
resolving to *"Relational recurrent neural networks"*, and *"…grammars"* to *"…regularization"*.
`SequenceMatcher` scores those **0.91 and 0.80**, over `FUZZY_TITLE_THRESHOLD`, because two long
titles differing in one decisive word are mostly the same characters. **The ratio cannot see a
substitution.** Comparing the *sets* of words can: measured over all 40, every false link scored
word-coverage **0.75** and every true one **0.88 or above**, so `MIN_TITLE_WORD_COVERAGE = 0.80`
sits in an empty measured band rather than being tuned to a target. Coverage is deliberately
asymmetric — a reference is often a prefix, or carries an author-list tail, so extra words in the
longer title are expected; a *missing* word is the signal.

**Result on the live library** (backup `data/library.db.bak-20260826-precitations`): 4,410 rows
unchanged, nothing deleted, **12 false links dropped** (exactly the set the read-side guard was
already hiding) and **37 gained**, every one inspected by hand. Self-resolutions are refused —
a document's own reference list resolving to itself is noise in the citation graph.

**Rejected:** raising `FUZZY_TITLE_THRESHOLD` to compensate (KI-45 says not to, and it would have
cost true matches at 0.91 while still admitting the 0.91 false one — the ratio is the wrong
instrument, not the wrong number).

**What it opens.** KI-45 part 2 — **freshness** — is still unbuilt: resolution re-runs only when
asked, so ingesting a paper does not yet re-point references that name it. `--reresolve` is the
manual form of that, and is what a future ingest hook would call.

---
## 2026-08-26 — the ingest politeness budget, and two performance claims of mine that were wrong

**What changed.** `ingest/workers.py` (budget resolver + parallel extraction warm-up), an
`ingest_budget` setting, `--workers`, `DOC_INGEST_WORKERS`, `multiprocessing.freeze_support()` at
the sidecar entry point, and the measured cost model written into `docs/performance.md` §1b and the
README.

**A politeness control, not a performance knob — and that distinction is what makes it legal.**
ADR-037 keeps cost knobs out of the UI, but its test was **output-neutral**: a knob that trades only
time is safe to expose. Worker count cannot change what an answer says, so it passes. Its other
objection — *"no restart semantics; every knob is decided at pipeline construction"* — does not
apply either, because ingest is a discrete invoked run, not construction. The budget is resolved
**per run** so the eventual multi-user build passes its own number instead of inheriting a desktop
preference.

**⚠ I CLAIMED ~8x FROM PARALLELISM. IT IS 1.74x.** The reasoning — "extraction is 89% of cost and
serial on 28 cores" — assumed the work was single-threaded. Measured (16 documents, cold cache each
round): serial 367.7 s · 2 workers 250.1 s (**1.47x**) · 4 → 1.68x · 7 → 1.73x · 14 → 1.74x. The
10k projection moves from the ~13 h I quoted to **~55 h**, against ~96 h serial.

**Two explanations tested, both false.** The long-tail document: the slowest here is 50 s, far under
the 211 s floor observed at 14 workers. OpenMP contention: `OMP_THREAD_LIMIT=1` measured marginally
**worse** (225 s vs 217 s). Recorded as measured-but-unexplained rather than guessed at a third
time; the docs tell anyone projecting a large ingest to use ~1.5x, not their core count.

**The measurement chose the default, and it vindicates the user's instinct.** Two workers buy
**85% of the entire achievable gain for 2 cores instead of 14**, so `light` is the default: the
polite setting is very nearly the fast one, and nobody has to trade one for the other. Even `full`
deliberately leaves half the cores.

**⛔ AND IT CAUGHT A BUG THAT WOULD HAVE SHIPPED: there was no `freeze_support()` anywhere.**
Windows spawns rather than forks, so every child re-imports `__main__` — and in a PyInstaller
bundle that is the *whole application*, so each worker would have started **another API server**
instead of extracting a PDF. Found because my own benchmark runner lacked the guard and recursed.
`freeze_support()` is now called at the sidecar entry point, **and** `warm_extraction_cache`
refuses to parallelise when `sys.frozen` is set until a packaged build has been observed doing it
correctly — a fork bomb of sidecars is bad enough to warrant declining by default.
`DOC_INGEST_PARALLEL_FROZEN` lifts the guard for that verification without a code change.

**The cost model is now documented for two audiences**, because the shape is what transfers and the
seconds are not. Per-stage shares for a reference 30-page digital PDF: **extraction ~90%**, embedding
~5%, everything else (figures, citations, keywords, epistemics, metadata) ~5% combined. Estimate by
**pages, not megabytes** — measured, a 15 MB/20-page paper indexed faster than a 5 MB/22-page one.
**Metadata is the cheapest stage by three orders of magnitude**, which retires the idea that a
Calibre-style cheap-metadata path would save meaningful time; the useful version of that idea is a
*shallow ingest* that defers extraction entirely, which is a different feature.

**Rejected:** a boolean on/off (it offers 1 or 28, and 28 is the monopolising the control exists to
prevent); parallelising the whole document loop (embedding is GPU-bound and already batched, and
concurrent writers on one Chroma collection is a corruption risk).

**What it opens.** Why the curve plateaus is unexplained. The frozen guard needs one packaged-build
verification to lift. **OCR remains the larger unclaimed win** — measured at 1.4-2.8x of extraction
for +0-8% text, on a corpus whose text layers the project already documented as good — but it
changes stored text, so it is an eval-harness question, not a judgement call.

---
## 2026-08-25 (5) — the corpus is re-ingested, and a full re-extraction is a four-day job at 10k documents

**What changed.** No code. The 97-document library was re-extracted and re-embedded end to end,
after the identity work below made that non-destructive. **6m31s**, because the caches were warm
from the two earlier passes; the cold run took 61 minutes.

**Nothing was lost, and that is the measured claim rather than the hopeful one.** Against the
pre-run backup: documents 97→97, figures 881→881, keywords 1455→1455, epistemics 445→445,
concept_presence 31→31, document_field 82→82, folders 18→18, metadata overrides 1→1, and
**97/97 document ids preserved — 0 lost, 0 new**. Every figure's `doc_hash` agrees with its
document. Chunks moved 13,925 → 14,957, which is the point. KI-48's symptom is gone: all 97
documents now derive `ingested` rather than `changed`.

**Scaling, since it was asked and is worth keeping (n=96, one run, this box, cu130).** Time tracks
**pages, not bytes** — r=+0.73 against page count, r=+0.37 against file size. About **1.06 s/page
plus 2.5 s fixed** per document; 1.19 s/page on the mean. The bytes correlation is real but weak
and comes from OCR: `elife-04250-v1.pdf` took 414 s for **21 pages**, while `hebb_1949.pdf` took
507 s for 365. Megabytes are close to useless as an estimate; pages are not.

**⚠ THE PROJECTION MUST USE THE MEAN, NOT THE MEDIAN, and I got that wrong first time.** The
distribution has a heavy right tail — mean 34.6 s/doc against a median of 17.5 — so a
median-based total halves the real answer. Total time is `n × mean`:

| corpus | full re-extraction |
|---|---|
| 100 | ~1 h |
| 1,000 | ~9.6 h |
| **10,000** | **~96 h** |

The per-page figure agrees independently: 10,000 documents at this corpus's 30 pages/doc is ~98 h.
**Against the ~10,000-document robustness contract, that makes "never invalidate a cache you did
not have to" a scaling requirement rather than a nicety** — which is exactly what the per-format
fingerprint below buys.

**⚠ AND THE HEADLINE NUMBER HIDES WHERE THE TIME GOES — 89% of it is extraction, done SERIALLY on
a 28-core box.** Three full runs over the same 97 documents and the same work split (78 processed,
19 skipped) separate the phases cleanly:

| run | extraction | wall clock |
|---|---|---|
| cold caches | 97 documents extracted | **61 min** |
| warm caches | 0 extracted (verified 97/97 fresh) | **6.5 min** |

So extraction is ~54.5 min of the 61 — **~34 s/document** — and chunk + embed + store is ~5 s/doc
across two Chroma collections. The dominant phase is single-threaded while 28 logical cores sit
idle: roughly **4% CPU utilisation** during the part that costs 89% of the time.

**That reframes the 10,000-document projection.** ~96 h is a *serial* number, not a floor. Extraction
is per-document and independent, so a process pool over it is the obvious lever — at 8 workers the
cold pass here would be ~7 min rather than 54, and 10k documents lands nearer ~13 h. Embedding
should stay as it is (GPU, already batched). Not attempted, and it needs care — PyMuPDF wants
processes rather than threads, and OCR pages are memory-hungry — but the measurement says the
headroom is real and large. **Filed as the honest reading of the scaling answer: the cost is not
that extraction is slow per page, it is that we do one page at a time.**

**Chunk locations are live at 81% (baseline) / 63% (parent-child) of chunks, across 78 of 97
documents.** The 19 missing documents extracted identically, so they were skipped and never
re-chunked; they carry no spans and fall back to read-time matching. **Careful reading the
coverage:** `Chroma.get(limit=N)` returns earliest-inserted chunks, which are exactly those 19 —
sampling that way reports 0% and looks like a total failure. Ask for a document whose hash moved.

**What it opens.** Those 19 documents gain spans only on a forced re-chunk. The figure PNGs remain
11% resolvable (88/811) — **pre-existing**, unrelated to this work, and worth its own look.

---
## 2026-08-25 (4) — a re-extraction destroyed 767 figure rows, because the sweep runs before identity

**What changed.** `cleanup_orphans_sqlite` returns `OrphanHashes(gone, stale)` instead of one list
and deletes rows for `gone` only; `cleanup_orphan_figures` receives `gone` plus stale hashes no
`Figure` row claims; `store.repoint_figures` carries a document's figures onto its new hash. Filed
as the fix for a real data loss, not a hypothetical.

**⛔ WHAT HAPPENED.** ADR-047 gave `_existing_document_id` a source-path fallback so a document
survives its own re-extraction. It was verified in isolation — 97/97 documents kept their identity
— and then the real re-ingest **destroyed 767 of 881 figure rows, 1,170 keywords and 381
epistemics rows, reporting `exit=0` throughout.**

The fallback never ran. `main()` executes the orphan sweep **first**, and that pass was hash-keyed
too: it re-extracts every file, finds 78 hashes no Document row still matches, classifies them
`stale`, and deletes those rows — FK-cascading the sidecars — before `_existing_document_id` is
ever consulted. Restored from backup, verified count-for-count. **Filed as KI-49**, because the
shape recurs: a component was proved and the system was not.

**The distinction that fixes it.** A source file that is **gone** ends its document — row, chunks,
figure PNGs. A source file that is **still there** and merely extracts differently does not: only
the chunks are dead. Those were one list, which is precisely how a maintenance operation came to
delete data.

**Figures were the only sidecar needing real work.** Everything else — keywords, epistemics,
concept_presence, folders, metadata overrides, document_field — is keyed on `document_id` alone
and survives automatically once the row is not deleted. `figures` is keyed **both** ways, plus
on-disk PNGs under `FIGURE_DIR/{doc_hash}/` whose **absolute** paths are stored in `image_path`,
so `repoint_figures` does three things together: rename the directory, rewrite the stored path,
set the new hash. Worth carrying rather than regenerating — a figure is a crop of *the PDF's own
page*, so a **text** extractor change cannot invalidate it, and regenerating means paying for 881
VLM descriptions again.

**And narrowing the sweep opened a leak, which the test suite caught.** A stale directory with no
`Figure` row behind it has nothing to move it and nothing to read it. `hashes_with_no_figure_rows`
splits those out so they are still collected. The failing test was not relaxed; it became two.

**Verified the way the first attempt was not: a full ingest against a faithful 1.1 GB copy of
`data/` before touching anything.** That trial caught two further problems that would have made it
meaningless — copying changes the paths the identity fallback matches on (so the DB's
`source_original` had to be rewritten), and **`_chroma_base()` silently relocates the vector store
to `%PROGRAMDATA%` when `DATA_PATH` is non-ASCII**, which the scratchpad path is once resolved
(`C:\Users\Lucas Délez\…`). The first trial therefore ran against an empty Chroma, classified
nothing, and exercised none of the fix.

**What it opens.** KI-49. The `%PROGRAMDATA%` relocation is deliberate (a documented chromadb
persistence bug) but invisible at the point of use — any tooling pointing `DOC_DATA_DIR` under a
non-ASCII home gets a fresh store rather than the real one.

---
## 2026-08-25 (3) — extraction identity, scoped per format and no longer hostage to the extractor

**What changed.** `extraction_fingerprint(suffix)` is scoped to one format's call graph;
`get_cache_path`/`is_cache_fresh`/`write_cache` carry the suffix; chunks record where they came
from; and ADR-047 makes a document's identity its source file rather than its extracted text.

**KI-48, fixed at the cause.** The fingerprint hashed the bytecode of *every* function in
`extractors`, so an EPUB/HTML-only refactor invalidated all 97 **PDF** caches — a whole-corpus
re-extraction for a change that provably could not alter a single PDF. The scope is now the
transitive closure of what one format actually executes, walked from `extract_to_markdown` with
the *other* formats' entry points blocked, so a shared post-processing step is picked up
automatically and a per-format one is not. Measured: `_soup_to_markdown` moves `.epub`/`.html` and
nothing else; `_TEXT_LAYER_KEPT_MIN` moves `.pdf` and nothing else; `strip_image_placeholders`
moves all seven. Any failure in the walk falls back to the whole-module scope — over-invalidating
costs CPU, under-invalidating serves text no current version would produce (KI-40).

**⚠ TWO BUGS THE MEASUREMENT CAUGHT THAT REVIEW DID NOT.** Every non-PDF format is dispatched
through the `_EXTRACTORS` **dict**, a runtime lookup a static walk cannot follow — so `.epub`'s
closure held two functions and `extract_epub` was not one of them. **An EPUB fix would not have
invalidated a single EPUB cache: KI-40 reintroduced, silently.** And `_EXTRACTORS` reprs its
functions *with their memory addresses*, so hashing it gave a different answer every process and
no cache would ever have read fresh again. Both are pinned by tests; determinism is asserted
across three separate processes.

**Chunk locations came almost free** (ROADMAP 19): the baseline path already computed the offset
for `extract_chunk_metadata` and threw it away. The parent/child path needed the cursor walk, and
there the splitter does **not** emit verbatim substrings — it strips and rejoins on its separator,
so 2 of 8 children were unfindable by `str.find`. `locate_span` probes head and tail, then
**verifies the span contains its own text** and returns `None` when it cannot. That check is not
decoration: without it three chunks in one paper resolved onto a neighbouring figure caption. A
missing location costs a highlight; a wrong one points confidently at the wrong paragraph.

**Rejected:** a hand-maintained format→dependency map (it rots, and the failure is silent);
sentinel `-1` offsets (a caller reads them as a position).

**What it opens.** ADR-047's remainder is RG-027's: identity is now *path*-shaped, so a file moved
between folders still reads as new.

---
## 2026-08-25 (2) — a referenced file could be registered but never ingested, and the corpus turned out to be uncached

**What changed.** `get_cache_path` resolves for a file anywhere on disk, the AD3b migration is
applied to the live library, and KI-48 records what checking that turned up.

**AD3b shipped a document you could add and then not use.** `get_cache_path` did
`original.relative_to(config.DOCS_PATH)`, which **raises** for anything outside the library folder.
It is called **unguarded** at four points on the ingest path, so ingesting a referenced document
was a `ValueError`, not a degraded result. `registry._cache_is_fresh` swallowed the same error and
returned False, which is why nothing screamed: every referenced file simply derived `new` forever
and looked like it was merely waiting to be ingested. Registration worked, so the failure sat one
step past the part AD3b tested.

**The fix keeps the library's layout byte-identical.** A file under the library folder still mirrors —
its path under `data/sources/` reappears under `data/cache/` with an `.md` suffix — verified across
all 97 documents, because moving
that path would silently re-extract the corpus. A referenced file has no relative path to mirror, so
its entry is keyed by a digest of its case-normalised absolute path: same file → same entry, same
filename in two folders → no collision, any source path → a legal filename. Digest over the *path*,
not the bytes, because it must resolve before the file is read and `scan_root` asks once per file
per listing. Proven end to end: extract → cache hit on re-extract → touch the source → stale.

**The migration ran on the real library.** Backup `data/library.db.bak-20260824-preroots`. 97 rows,
0 NULLs, 0 FK violations, `integrity_check ok`, the library root pointing at `data/sources`, and
all four indexes present. `origin` is `copied` on all 97, which is a fact rather than a default —
every one of them was found by scanning the source dir.

**⚠ AND THE CHECK THAT WAS SUPPOSED TO BE A FORMALITY FOUND KI-48.** Confirming no cache entry had
moved also revealed that **97 of 97 are stale** — recorded fingerprint `2ce7639c…`, current
`686aa2df…`. Cause: `extraction_fingerprint()` hashes the bytecode of every function in
`extractors`, so yesterday's `f285212` (EPUB/HTML only) invalidated the **PDF** caches too;
bytecode hashing cannot tell which format a change touched. That commit's DEVLOG says *"blast
radius zero … so nothing was re-ingested"* — true of the output then, **wrong about the cache**.
Worse, re-extracting today is not a no-op: measured on 3 of 97, the fresh text differs, which is
**KI-47** (Tesseract is now on PATH so scans OCR) and walks straight into **KI-43** (identity is
the extracted content, so changed text orphans every id-keyed sidecar). Filed with three options
and no decision taken — it is a call about the corpus, not a bug to patch.

**What it opens.** ROADMAP 18 and 19, from the user: a document viewer in a Library right-split
(gated on the file being reachable — `root_available` already answers that, and `page` is already
on every chunk, so page-level costs no ingest change), and locating a cited chunk in its source
text. The second has a real fork: read-time matching works on today's corpus, exact offsets need a
schema field and a re-ingest — and **KI-48's pending re-extraction is the cheap moment** for that
if it is ever wanted.

---
## 2026-08-25 — AD3b: the registry grows a root, and the key that was never unique alone

**What changed.** `SourceRoot` + `SourceFile.root_id` + the `(root_id, rel_path)` unique index, an
ADR-026 rebuild migration, a multi-root `registry`, `apply_add(mode="reference")`, and the sheet's
radio becoming a real control. **A document can now be added without being copied anywhere.**

**The rebuild was forced by a SQLite rule, not chosen.** With `PRAGMA foreign_keys=ON` — which
`db/session.py` sets — SQLite **refuses** `ADD COLUMN` carrying a `REFERENCES` clause unless the
default is NULL, and a nullable rootless row is precisely what this change exists to prevent. The
alternative (add the column without the FK) leaves a migrated database structurally different from
a fresh one, which is how `document_meta` earned ADR-026 in the first place. So: rebuild, with the
literal `DEFAULT 'library'` doing the backfill *during the copy* — `_rebuild_table` only carries
columns present in both shapes, so the server default is load-bearing, not decoration.
**Measured on a copy of the real library: 97 rows, 0 NULLs, FK clean, idempotent on re-run.**

**`_rebuild_table` silently dropped every index, and nobody had noticed.** `CreateTable` renders
the table only; the old indexes die with the dropped table. It was latent because `document_meta`,
its only previous caller, declares none — `source_files` declares four. Fixed there rather than
worked around here. The same pass found `source_sha256` indexed on live databases by the AD2
migration but **not declared on the model**, so a fresh database never had that index; declaring it
fixes the divergence and makes the rebuild preserve it.

**⚠ THE ONE THAT WOULD HAVE BEEN A SECURITY BUG.** Canonicalising each selection request to
`"<root_id>:<rel_path>"` *before* validating it defeats the traversal guard outright:
`PurePosixPath("library:../evil.pdf").parts` is `("library:..", "evil.pdf")`, so `".." in parts` is
**False** and `../evil.pdf` walks straight through a check written to stop exactly that. Caught by
an existing test (`test_resolve_selection_rejects_bad_paths`) that I had assumed was reporting a
cosmetic message change. Requests are now split by root **first** and each root's rel_paths
validated against that root's own on-disk set — which also restores offender fidelity. Pinned by a
new guard that asserts both spellings still read as traversal.

**Duplicate detection was quietly single-root.** `_size_index` resolved every registered `rel_path`
against the *library* folder, so a row under a referenced root resolved to a path that does not
exist, the read raised, and the candidate came back a clean `add` — the library would hold the same
bytes twice, which is the one thing that gate exists to prevent. The index now joins each row to
its root and carries a real absolute path. This is the second time in this feature that a bare
`rel_path` turned out to be a lie; `duplicate_of` and the hash cache are both keyed by
`source_key` now.

**Undo had to change meaning, not just plumbing.** It previously *refused* any row whose origin was
not `copied` — correct while reference mode did not exist, and stranding after AD3b: the user
rejects a reference add and the row stays forever. Undo now always drops the row and deletes the
file **only** when the app made the copy. That is the ADR-014 amendment enforced where it cannot be
forgotten.

**Open question 3 is resolved, and it bought a state.** Missing-detection stays **on demand** — no
startup pass, because `rglob`-ing every referenced root at launch is slow or blocking on exactly
the paths most likely to be network or removable, at the worst moment (RG-012). The consequence,
decided with it: an unreachable **root** is not the same fact as deleted **files**, so availability
is derived per scan (never stored — a drive unplugged now may be back in a second) and an
unavailable root keeps its rows *and their `last_seen`* untouched. A disconnected drive must not
be indistinguishable from the user losing 400 documents.

**A limitation I chose rather than hid: referenced roots are per-*parent-directory*.** `apply_add`
receives files, not the gesture that produced them — a dropped folder is already expanded by the
time it arrives — so a file's root is its own parent. Drop a Zotero folder with twenty
subdirectories and you get twenty referenced roots, one per subdirectory. Nothing breaks (each
root scans correctly, and referencing twenty papers from *one* folder still yields one root), but
the roots list is noisier than it should be. Fixing it properly means passing the original drop
set through so a dropped directory can become the root; that is a signature change through the API
and was out of AD3b's scope. Recorded here rather than discovered later.

**What it opens.** The delete half of ADR-046 (`delete_document(delete_file=…)`, spec cases 6 and
7) is still unbuilt — the schema it branches on now exists, so it is no longer blocked. AD4,
CS1/CS2 and RG-030 are untouched. **Not proven:** no referenced document has been ingested end to
end through the UI, and the spec's own DoD still wants an EPUB and an HTML file added that way.

---
## 2026-08-24 (3) — the W0 assertions run in a real Tauri window, and the spec's own expectation was wrong

**What changed.** `docs/specs/feature-add-documents.md` §W0 gains the runtime result. No code
changed — a temporary probe was added, run, and removed (`svelte-check` back to 205/0, no residue).

**How a devtools-only check was run without devtools.** A Tauri window's console is not reachable
from an agent session, so the two §W0 one-liners could not simply be pasted anywhere. The probe
executed them inside the webview and **reported by pinging the API with the payload in a query
string** — uvicorn logs every request path, so the answer came back through the server log. The
app's CSP already allows `connect-src http://127.0.0.1:8001`, so this needed no config change.

**Result, and both assertions pass:**
```json
{"tauriKeys": ["app","core","dpi","event","image","menu","mocks","path","tray",
               "webview","webviewWindow","window"],
 "isTauri": true, "canReceiveDrops": true, "canPickFiles": true,
 "listenOk": true, "listenError": null, "href": "http://localhost:1420/"}
```
`withGlobalTauri` injects the API · `listen('tauri://drag-drop')` registers · the picker is
reachable · and `href` proves the window loaded **this** app rather than the other project that
periodically owns 1420 — the exact confusion §W0 warned about.

**⚠ The run falsified an expectation I had written into the spec.** §W0 said to expect `dialog`
among `Object.keys(window.__TAURI__)`. **It is absent** — `tauri-plugin-dialog` registers itself
with `Object.defineProperty`, which defaults to **`enumerable: false`**, so the plugin is present
and fully working (`canPickFiles: true`) while invisible to `Object.keys`. A future reader
following the old instruction would have concluded the plugin failed to register and gone looking
for a bug that does not exist. The corrected probe is
`typeof window.__TAURI__.dialog?.open === "function"`.

**Still unproven, and stated as such:** that a real OS drag delivers a **non-empty `paths` array**.
The *subscription* is now confirmed at runtime and the *payload shape* from
`tauri/src/manager/window.rs`; the gap is one human drag, which no probe can perform.

**A stale comment found in passing.** `lib.rs` says *"in dev there is no frozen binary (run the
backend separately with `just api`)"* — but a `binaries/doc-assistant-api` sidecar exists on this
machine, so `tauri dev` spawned it and it failed to bind 8001 against the already-running dev API.
Non-fatal by design; the comment is simply out of date, and the port clash is confusing if
unexpected.

**Rejected alternative.** *Declaring W0 done on the source evidence alone* — the sources were
right about the mechanism and wrong about what the check would look like, which is precisely the
difference between reading a contract and exercising it.

**What it opens.** The Tauri question no longer blocks a merge. What remains is a judgment call:
the feature is deliberately partial (AD3b, AD4 unbuilt) and nothing is committed.

---
## 2026-08-24 (2) — the Add-documents button moves to the Library header row

**What changed.** `[+ Add documents]` leaves the app toolbar and joins the Library header row
(`LibraryPane.svelte`, `.libnav`) beside Sort / View / Select. An **"Add documents…" entry is added
to the app menu** so the action stays reachable from Chat.

**Why (user's call, and the reasoning is worth keeping).** Three things pointed the same way:
- The toolbar's right cluster is **identity + config by design** — `Topbar.svelte` says it in a
  comment: *"Brand = identity anchor only (small mark + wordmark), parked on the right beside
  Settings."* Everything else in that bar is navigation. An action there was the odd one out, and
  had to be styled like chrome to fit — exactly wrong for a pane's primary action.
- It acts on the Library's content, and that row already holds the document actions.
- **Not** the keyword row: that is the facet bar, and "Filter by keyword" is a filter, not an
  action. Mixing a create-action into a filter row mixes two grammars.

**The cost was named before the move and is mitigated, not ignored.** In the toolbar the button was
reachable from Chat; in the pane it is not. The app-menu entry costs no new chrome and restores
that reach — and window-wide drag-drop was never mode-specific anyway. Without it the move would
have been a straight loss of capability from one mode.

**Now a filled accent button** rather than toolbar chrome — which is the point of the move. It
still degrades the same way outside the Tauri window: **visible, disabled, and saying why**
(*"Adding documents works in the desktop app."*), because hiding it would make a browser look like
a missing feature.

**Verified live, and one trap re-encountered.** First geometry read came back with every
coordinate near zero and `Select` apparently absent — because the Browser pane was hidden and
`window.innerWidth` was **0**, exactly as `apps/desktop/CLAUDE.md` warns ("a hidden pane collapses
it to 0"). Forcing a 1280 viewport gave the real numbers: **Sort 941 · View 980 · Select 1053 ·
Add 1132**, all on one row, Add last and right-aligned; disabled styling computed correctly
(`--surface` background, opacity 0.7, `--text-sm`). The structural check (`.libnav` children:
`crumbs · libsort · viewtoggle · selecttoggle · addbtn`) was valid either way and is what confirmed
the placement while the geometry was garbage.

**Gates:** svelte-check 205/0 · 145 frontend · 39 add-documents tests. Wireframe screen 1 still
shows the button in the toolbar and is now **out of date** — it needs the same move.

---
## 2026-08-24 — AD3a: documents can be added for real, and the live test caught what the type gate could not

**What changed.** `apply_add` / `undo_add` in `library/add.py`, the `origin` column + migration,
`POST /api/documents/add` and `/undo-add`, and the review sheet's primary action. **A document can
now be dropped into Provenote and land in the library.**

**Scoped as AD3a, and the split is a finding rather than a shortcut.** ADR-046 chose *both*
placement modes for v1. The schema half of reference-in-place is cheap — `source_roots` is a new
table (`create_all` covers it), `root_id` is additive, and `rel_path`'s uniqueness turns out to be
a **separate unique index** (`ix_source_files_rel_path`), so it can be swapped without a table
rebuild. **The cost is the call sites:** `scan_sources`, `derive_status`, `resolve_selection` and
`list_sources` are all keyed on a bare root-relative path and each needs the root dimension first.
That is AD3b. ADR-046 is unchanged.

**`reference` is accepted and refused, not silently downgraded.** `apply_add` raises
`NotImplementedError`, the API answers **501**, and the sheet shows the radio **disabled with the
reason**. Hiding the option would misrepresent the decision; enabling it would copy files to a
place the user did not choose. Pinned by a test asserting the refusal is *total* — no copy lands
under any name.

**The failure story is the honest half (grill branch 6).** `apply_add` stops at the first failure
and returns `added` / `failed` / **`not_attempted`** — the files after the failure were never
touched, and reporting them as "skipped" would be a lie. The sheet then offers *Keep the N* or
*Undo all*. `undo_add` deletes outright rather than binning (ADR-014): these are copies the app
made seconds ago and the user is rejecting, so nothing is at risk — and it **refuses any row whose
`origin` is not `copied`**, which is the ADR-014 amendment enforced where it cannot be forgotten.

**A name collision does not overwrite.** Two different papers can share a filename, and ADR-043
keeps received content verbatim, so a second `paper.pdf` becomes `paper-2.pdf`. Byte-identical
files never reach here — those are AD2 `duplicate` verdicts.

**Indexing is not done inside `apply_add`.** It is a separate call to the existing
`POST /api/ingest` with an explicit `paths` list (spec constraint 4), so the system keeps one
ingest path rather than two. The sheet indexes **only what actually landed**, never what a stopped
run failed to reach.

**⚠ THE LIVE TEST EARNED ITS KEEP: the sheet unmounted before the user could see the outcome.**
`onAdded` cleared the staged paths, and the staged paths are also what keep the sheet mounted — so
on success the sheet vanished mid-flow and **the Undo button was unreachable**. svelte-check was
green, 39 backend tests were green, and the bug was invisible to both. Clearing now belongs to
`onClose`. This is `apps/desktop/CLAUDE.md`'s "the type gate is not the run gate" rule paying for
itself twice in three days.

**Verified against the real library, and it was left exactly as found.** Baseline 97 documents /
97 source_files / 97 files on disk. Staged one throwaway file **with indexing switched off** (so
nothing reached Chroma), added it — copy on disk, row with `origin='copied'` and a hash, documents
still 97 — then undid it through the UI. **Final state: 97 / 97 / 97, no probe row, no probe
file.**

**What it opens.** AD3b (reference-in-place) with the call-site work named above. AD4 (the empty
state). RG-030 still gates AD2's fourth verdict. And the `[+ Add documents]` button placement is
still open — recommendation on record: the Library header row with Sort/Grid/List/Select, plus an
app-menu entry for the Chat case.

---
## 2026-08-21 (5) — AD2: the review sheet, and a duplicate check that does not read the library

**What changed.** `src/doc_assistant/library/add.py` (new), `POST /api/documents/inspect`, the
`source_sha256` column + its migration, `AddDocuments.svelte`, and the wire types/client for both.
**2,055 python tests** (was 2,029 — the 26 added here) - **145 frontend** - svelte-check 205/0 -
mypy `src` 92/0 - ruff clean.

**The design point worth keeping: size is the discriminator, sha256 only ever confirms.** ADR-046
made add-time identity the sha256 of the source bytes, and the obvious implementation hashes every
registered file on every inspect — hundreds of megabytes at 97 documents, for a question that is
almost always "no". Instead `inspect` builds a `{size: [rows]}` index from the registry's
**already-scanned** `size` column and hashes only rows whose byte length a candidate actually
matches. In the common case the library is never read at all. **Asserted by counting reads**, not
by trusting the comment: `test_no_registered_file_is_hashed_when_no_size_matches` monkeypatches
`sha256_file` and requires an empty call list.

**`source_sha256` needed a real migration, not `create_all`.** `source_files` already exists in
every live database, and `create_all` never ALTERs an existing table (`migrations.py:33` says so).
Registered in `_ADDITIVE_COLUMNS` with an index, and **applied to the live library** — the column
lands NULL everywhere and is deliberately **not backfilled**: filling it means reading every source
file, and the cache fills itself on exactly the paths that need it. NULL reads as *"not computed
yet"*, never as *"not a duplicate"* — the size index decides what gets compared, so the two cannot
be confused.

**`inspect` and `apply` are two endpoints so constraint 2 stays structural.** Nothing is copied,
registered or indexed before the sheet is confirmed; `inspect`'s only write is the hash cache on
rows it already had to read. Asserted at both levels — unit
(`test_inspect_adds_no_registry_rows`) and over the wire (`test_inspecting_registers_nothing`).

**The no-text-layer verdict is deliberately absent.** RG-030 gates it and its cost across a
500-file batch is unmeasured; shipping it on a guess is the thing that entry exists to prevent.
The other three verdicts do not need it.

**AD2 stops at reviewing, and the sheet says so.** No disabled "Add" to nowhere — copy-vs-reference
and indexing are AD3. The `advisory` string is rendered **verbatim** from `get_format_status`; the
frontend never re-derives which formats are supported, because that list in a second language is
the ADR-013 shape.

**Verified live against the real API** (5 real paths, one nonexistent):
`5 files - 3 would be added`, rows ordered **unsupported, unreadable, then the three adds** —
grill branch 7 holding in the running UI — with *"Format .toml is not supported."* passed through
untouched. Sheet closed and staged state cleared, so the app was left as found.

**Two incidentals worth writing down.** The API runs without `--reload`, so a newly added route
**404s until the server is restarted** — the sheet's error branch surfaced it correctly rather than
spinning, which is how it was caught. And `Icon.svelte`'s `IconName` is declared in the instance
script, so it **cannot be imported**; a narrower literal union at the call site is the fix, not a
change to Icon.

**What it opens.** AD3 (copy/register + index) is next and needs the multi-root schema from
ADR-046. RG-030 still gates the fourth verdict. The `[+ Add documents]` button's placement is under
discussion — it does not affect this sheet, which is why AD2 proceeded.

---
## 2026-08-21 (4) — AD1: the app can accept a document, and the CSS trap from 2026-08-19 fired again

**What changed.** The accept surface. `withGlobalTauri: true`, the dialog plugin, a `[+ Add
documents]` button beside Settings, a window-wide drop target, and a staged-files summary. **145
frontend tests** (was 127), svelte-check 202/0, `cargo check` exit 0.

**Config (W0's route, applied).** `tauri.conf.json` gains `withGlobalTauri: true`; `Cargo.toml`
gains `tauri-plugin-dialog = "2"`; `lib.rs` registers it; `capabilities/default.json` grants
**`dialog:allow-open`** and nothing more — the plugin's own `dialog:default` would also grant
`message` and `save`, which nothing uses. **No npm dependency was added**; the frontend is still a
1-dep artifact.

**A config comment nearly broke the build.** The first attempt documented the flag with a
`"_comment_withGlobalTauri"` key. `tauri-utils`' `AppConfig` is `#[serde(deny_unknown_fields)]`
(`config.rs:3006`), so that would have failed to parse at build time — caught by checking the
struct rather than by waiting for the compiler. **`tauri.conf.json` takes no comments;** the
rationale lives in the spec and here.

**Code.** `lib/core/tauri.ts` is the **only** module allowed to touch `window.__TAURI__`, and every
export degrades instead of throwing: outside the Tauri window `isTauri()` is false, subscribing
returns a no-op teardown, and the picker resolves to `null`. That matters because the entire
dev/test loop for this app runs in a plain browser, where the global does not exist.
`lib/library/accept.ts` is pure and fully tested (18 cases); `accept.svelte.ts` holds the staged
paths and **cannot write anything** — no copy, no register, no index — which is how spec constraint
2 stays true by construction rather than by discipline.

**`accept.ts` deliberately does not filter by extension.** The format list lives in
`extractors.is_supported` / `get_format_status`, and AD2 computes verdicts server-side from it.
Filtering client-side would put that list in a second language and let the two drift — the ADR-013
shape (one rule, two copies, one stale).

**AD1 ends at *staged*, and says so.** Dropping files shows a summary bar naming the count and the
first three files, with a Clear. There is no disabled "Continue" to nowhere: the review sheet is
AD2 and the apply step is AD3. A build node that reports exactly what it can do beats one that
mimes the finished feature.

**The 2026-08-19 CSS accident repeated, and was caught by measuring rather than looking.**
`.tb-add` and `.tb-btn` tie on specificity, so `.tb-btn:disabled` — 27 lines later — won: computed
`opacity: 0.32` instead of 0.55 and `font-size: 15px` instead of `--text-sm`. Fixed as
`.tb-btn.tb-add`, the same two-class remedy the tick-row bug got, with a comment saying not to
simplify it back. **Re-measured after the fix: 0.55 and 12.3px.** A screenshot would not have shown
either value.

**Verified in the running app** (browser, where `window.__TAURI__` is `undefined`): the button
renders between the brand and Settings — x 985 (brand) / **1095 (add)** / 1241 (settings) — is
**disabled**, and its tooltip reads *"Adding documents works in the desktop app."* rather than
being hidden. Staging six paths (one a duplicate) yielded **five** pending and the bar
*"5 files ready · hubel-wiesel-1959.pdf · cajal-1899.pdf · treatise.epub · and 2 more"* — Windows
and POSIX basenames both correct; Clear emptied it. The console's 15 x 500 are `/api/health`
cold-start polls that recovered on the 16th, unrelated to this change.

**What it opens.** The two W0 runtime assertions still have to run in a real Tauri window — nothing
here proves the injection works, only that the code degrades correctly without it. AD2 (the review
sheet) is next and replaces the staged bar. RG-030 still gates AD2's no-text-layer row.

---
## 2026-08-21 (3) — W0: the accept surface costs zero npm dependencies, and HTML5 drag-drop was never on the table

**What changed.** `docs/specs/feature-add-documents.md` §W0 resolved; open question 1 closed;
AD1's contracts are no longer provisional.

**The answer is better than the spike was written to expect.** The spec assumed a file picker meant
a second npm dependency, breaking the *"deliberate 1-dep artifact"* property the frontend claims.
It does not. **`withGlobalTauri: true` injects the Tauri JS API on `window.__TAURI__`, and plugins
inject themselves into the same global — so the whole accept surface adds no npm package at all.**

**Every claim read from the pinned crate sources rather than from memory** (`tauri` locked 2.11.3,
2.11.2 vendored):
- `tauri/src/manager/window.rs` emits `tauri://drag-drop` with a payload built as
  **`paths: Some(paths), position`** — the event carries real filesystem paths.
- `tauri/scripts/bundle.global.js` (what `withGlobalTauri` injects) contains all four `tauri://drag-*`
  event names, an `onDragDropEvent` helper, and exports `core` / `event` / `webview`.
- `tauri-plugin-dialog-2.7.1/api-iife.js` does
  `Object.defineProperty(window.__TAURI__, "dialog", {…})` — the picker rides the same global.
- `tauri-utils/src/config.rs:3075` documents `with_global_tauri` as the injection switch, default
  **false**.

**The HTML5 route was not merely worse — it was never possible.** `drag_drop_enabled` defaults
**true**, and its doc string says *"Disabling it is **required** to use HTML5 drag and drop on the
frontend on Windows."* Turning it off to get HTML5 events would also surrender the paths that every
contract in the spec is built on. Recorded so nobody re-proposes it.

**Total cost:** one config line, one Rust dependency (`tauri-plugin-dialog`, already in the local
cargo registry), one plugin registration, one capability permission — `dialog:allow-open`, narrower
than the plugin's `dialog:default`, which also grants `message` and `save` that nothing needs.

**What was NOT done, stated rather than glossed.** W0's original DoD asked for a live round-trip
printing a real path. **It was not run.** That needs an OS-level drag into a native Tauri window;
this session can drive a browser but cannot perform a desktop drag, and a Tauri window's console is
not reachable from the Browser pane. The **decision** W0 exists to make is settled by the sources;
what remains is confirmation. Two one-line devtools assertions are written into §W0 to run at AD1's
first build, and the DoD now names them explicitly instead of quietly dropping the requirement.

**A design constraint the spike surfaced, which AD1 must build around:** `window.__TAURI__` exists
**only inside the Tauri window**. The entire dev/test loop for this app runs in a plain browser
(port 5731 all session), where it is `undefined`. The accept surface has to degrade there — drop
target inert, button hidden or self-explaining — rather than throw. Also noted: `devUrl` is
hardcoded to 1420, a port another project on this machine periodically owns (baton 2026-08-19), and
`tauri dev` will silently load whatever answers it.

**What it opens.** AD1 is unblocked. RG-030 (the page-1 text-layer probe's cost) still gates AD2's
warning row and is untouched by this.

---
## 2026-08-21 (2) — the add-documents SPEC, and the accept surface turns out to need a spike first

**What changed.** New `docs/specs/feature-add-documents.md` — the code-level contract for Track A
(AD1-AD4) and Track C (CS1-CS2) of the ingestion plan, under ADR-046. Contracts per file, 15 test
cases written before the code, a DoD, the task-level decision ledger from the grill, and an open-
questions table.

**The finding that reshaped the spec: there is no way to get a filesystem path yet.** Measured
2026-08-21 — `src-tauri/Cargo.toml` carries **only** `tauri-plugin-shell`; `capabilities/
default.json` grants `core:default` plus one sidecar-execute permission; and `package.json` has
exactly **one** runtime dependency (`marked`), which `forceLayout.ts` calls *"a deliberate 1-dep
artifact"*. So the accept surface is not a UI task sitting on ready infrastructure:
- a **dialog plugin** means a Rust dependency, a registration, a capability *and* a second npm
  dependency — which breaks a stated project property, so it is a decision, not a detail;
- the **Tauri drag-drop event** is believed to carry real paths and may need neither, but that is
  belief, not evidence;
- an **HTML `<input type="file">`** is believed insufficient (a webview yields `File` objects with
  no path, and every contract in the spec is path-based).

Rather than design around a guess, the spec opens with **W0**, a gating spike whose definition of
done is a working round-trip that prints a real path, plus a note recording the chosen route and
whether it adds a dependency. **AD1's contracts are marked provisional until W0 lands.** Writing the
UI first and discovering the path problem during the build is the failure this avoids.

**Contracts recorded** (all additive, all reusing what ships): `SourceFile` gains `root_id` /
`origin` / `source_sha256` with a new `SourceRoot` table and a `uq(root_id, rel_path)` replacing
today's `rel_path` key · `registry` goes per-root · a new `library/add.py` splits **`inspect`
(no mutation)** from **`add` (no extraction)** · `delete_document` gains `delete_file: bool = False`
per ADR-046 · two API routes that mutate nothing until confirmed · a pure
`dropzone.svelte.ts` for verdict sorting and pagination, testable under the `node --test` runner the
repo already has, because the alternative is another untestable component.

**Definition of done includes a corpus fact, deliberately:** not done until an **EPUB and an HTML
file** have been added and indexed through the UI. The corpus is 97/97 PDF, so those paths are
otherwise untested in the wild — the same gap that let three extraction defects sit undetected until
2026-08-20.

**Also recorded, unresolved:** `docs_check --strict` flipped from **0/0 to 15 errors** during this
session, all of them rule-11a ordering violations in `docs/archive/SESSION-archive-001.md` and
`-002.md`. Both are gitignored, local-only, and were last written **before** any of today's work;
neither was touched. The flip is **unexplained** — the gate ran 0/0 six times earlier today with
those files in their current state. What is *not* in doubt is that the content genuinely is
oldest-first: `.claude/SESSION.md` records that archive 001 holds *"the pre-convention
bottom-appended tail"*, so those entries predate the newest-on-top rule. **They should not be
reordered** — rewriting an append-only historical record to satisfy a linter is the wrong repair.
Archive 002's header claims "Newest first" while its body is oldest-first, which is a real (small)
inconsistency worth fixing in the header rather than the body.

---
## 2026-08-21 — ADR-046: an added document is copied in or referenced in place, and three contracts move

**What changed.** New `docs/decisions/ADR-046-added-documents-copy-or-reference.md`, indexed in
`docs/decisions.md`, with **ADR-014's row annotated as partially amended**. It records the scoped
grill's three ADR-class resolutions (`docs/PLAN_2026-08-20_user-friendly-ingestion.md` §8,
branches 1, 2 and 5) — not one decision, three.

**The decision, and that it overruled the recommendation.** Both modes — copy-in **and**
reference-in-place — ship in v1. The recommendation was copy-in-first with the reference model
merely *designed*; the user chose the larger option, and the deciding reason is the part worth
keeping: **it is the only option that does not make the Zotero adapter a second migration.**
Copy-in-first would build the copy path, then rebuild `SourceFile`'s key and re-decide delete when
the adapter arrives. The ADR records the recommendation, the overrule and the reason, because an
ADR is a record of a decision rather than of who was right.

**Why it is three contracts and not a UI task.** Each was verified in code before being written down:
- **`SourceFile` becomes `(root, rel_path)`.** It is keyed by `rel_path`, documented *"relative to
  the source dir"*, with **no root column** — so referencing a file that lives elsewhere is a
  **schema** change. This is the fact that makes the whole feature bigger than it looks.
- **ADR-014 is amended.** `delete_document` sends the source to the Recycle Bin **first**. Safe only
  while the source folder is Provenote's own; with reference-in-place it would take files out of a
  user's Zotero folder. Delete now asks, defaults to **library-only**, and **must name the real
  path** when the file is referenced — the accepted risk of a per-delete choice is a mis-click, and
  showing the destination is what makes the click informed, so it is part of the decision rather
  than copy-writing. **This absorbs the queued "library-only delete" checklist row**, which was
  never a separate feature.
- **Add-time identity becomes `sha256(source bytes)`.** There is no cheap pre-extraction identity
  today: the duplicate gate is `doc_hash(text)` on the *extracted* text and fires only after
  `load_or_extract`. Source-hash identity is the direction ADR-042 already chose and RG-027 has yet
  to build, so the add path **deliberately leads** it. The ADR states the consequence plainly so it
  is not later read as drift: **two identities coexist until RG-027** — source-hash answers
  *"already added?"*, `doc_hash(extracted)` answers *"already indexed?"*, and they can disagree.

**Rejected alternatives** (all four options are in the ADR): copy-in only — cheapest, ignores the
organisation the user already has; reference only — no copy path for users who want the app to own
its files; copy-in with reference merely designed — the recommendation, kept in the ADR as the
documented fallback with a note that nothing built for the chosen path is wasted if it is taken.

**What it opens.** The SPEC (`docs/specs/feature-add-documents.md`) is now unblocked and is the next
artifact; the task-level resolutions (indexing ticked by default, pagination, partial-copy failure)
are deliberately **not** in the ADR because they change no existing contract, and belong in the
SPEC's own ledger. RG-030 gates the review sheet's *no text layer* row, not this decision.

---
## 2026-08-20 (9) — a plan for the one thing the app cannot do: accept a document

**What changed.** New `docs/PLAN_2026-08-20_user-friendly-ingestion.md` — three tracks, wireframes,
a grill list, and the process route from here to code.

**The finding it is built on.** Provenote indexes a folder well and **cannot accept a document**.
Grep across `apps/desktop/src` returns no file picker, no drag-and-drop, no upload endpoint
anywhere; the entire front door is a text box reading *"Paste the full path to the folder holding
your documents."* Everything downstream of that step already exists and is tested — registry scan,
selection resolution with up-front validation, background job, status polling, per-file exclude.
The plan therefore adds one step and reuses `POST /api/ingest` unchanged.

**Two structural facts turned up while measuring, and both change the shape of the feature:**
1. **`SourceFile` is single-root by construction** — keyed by `rel_path`, *"relative to the source
   dir"*, with no root column. "Add documents from anywhere" is a **schema** change, not a UI one.
2. **Delete bins the user's file.** `delete_document` moves the source to the Recycle Bin *first*
   (ADR-014). Safe today because the folder is Provenote's own; the moment a document can be
   registered in place, deleting it takes a file out of the user's Zotero folder.

So the innocuous-looking question *"where does an added document live?"* is an ADR with a schema
consequence and a delete-semantics consequence — **queued as ADR-046**, and it also absorbs the
separately-queued *"library-only delete"* checklist row, which turns out not to be a separate item
at all. Recommendation recorded: copy-in for v1, with the reference model *designed* in the ADR so
the planned Zotero adapter is not later found to be blocked on a decision nobody wrote down.

**A second track that needs no decision.** Making ingestion easy makes KI-47 urgent: the shipped
installer has no OCR, so a scanned PDF becomes a 0-chunk `broken` document with no explanation.
Today that is hidden behind a CLI most users never run. The plan detects it **at add time** and
says so in the review sheet, and notes that Track B beats most of Track A on value if the ADR
stalls.

**Wireframes are in the plan** (empty state · review sheet · per-file indexing progress · the two
Settings changes), ASCII so they diff and survive in git. The review sheet is the load-bearing one:
nothing is copied or indexed before the user sees per-file verdicts, and the unsupported rows carry
`get_format_status`'s advisory **verbatim** — that string already exists and already names the fix.

**Rejected alternative.** *Writing the spec now.* `docs/ui-checklist.md` says this item "needs a
grill to scope" and it is right — nine open branches are listed in the plan (duplicate rule, folder
recursion depth, mid-batch copy failure, batch caps…). Writing a contract over unresolved branches
is how a spec becomes fiction. Route recorded: **grill-me → ADR-046 → `docs/specs/
feature-add-documents.md` → ROADMAP/ui-checklist rows → build**.

**Explicitly out of scope**, with owners: P2's LLM-assisted ingestion (quality, not friendliness),
P7's settings reorganisation (only the folder picker and its relabel are taken here), the Zotero
adapter, and ADR-039's OCR recovery.

---
## 2026-08-20 (8) — the three extraction defects are fixed, and the tripwires fired on cue

**What changed.** `extractors.py`: a new `_soup_to_markdown` helper, shared by `extract_epub` and
`extract_html`. All three defects found by the committed fixtures are fixed; the four
`xfail(strict=True)` markers are gone and their tests are plain regression guards.

**Fixed at zero cost, because the blast radius was measured first.** The corpus is **97 documents,
97 PDF — zero EPUB, zero HTML**. Only those two formats reach the BeautifulSoup path (PDF goes
through PyMuPDF4LLM, DOCX/ODT/RTF through their own libraries), so **no document needed
re-ingesting**. Entries (1) and (5) deferred these as "decisions, not patches" on the ADR-042
re-ingest cost — correct in general, but that cost is *currently zero* and only ever grows. Doing it
before ingestion gets easy is the cheapest this will ever be.

**The fixes.**
- **EPUB nav** — skip `EpubNav` items in the `ITEM_DOCUMENT` loop. The generated navigation
  document is a manifest item like any other, which is why it was being emitted as body prose.
- **`<head>`** — added to the decompose list, so a page or chapter `<title>` can no longer land
  above the real `<h1>` as a bare line.
- **Inline fragmentation** — `unwrap()` every inline tag, then `soup.smooth()`, *then* `get_text`.
  The separator stays `
`: it is what keeps `<p>`, `<li>` and headings apart, and changing it
  would have run blocks together. **`smooth()` is the load-bearing call** — without it the newline
  simply moves from the tag boundary to the boundary between the adjacent text nodes an unwrap
  leaves behind, and nothing improves.

**One helper, not two.** The two extractors held the same rule twice and **had already drifted** —
HTML dropped page chrome and EPUB did not. That is the exact shape of the ADR-013 display bug fixed
on 2026-08-19, so unifying them was the point rather than a tidy-up. EPUB consequently gains chrome
removal and `<head>` removal it never had; three new tests pin that, using chapter markup the
committed fixture deliberately does not carry.

**The strict xfails did their job.** All four flipped to failing XPASS the moment the helper landed
— the suite refused to let a fixed defect sit behind a stale marker. Converting them to plain
assertions was the mechanical last step, which is precisely the workflow `strict=True` exists to
force.

**Also pinned:** that block elements *still* separate after inline unwrapping. That is the one way
this fix could have gone wrong, so it is asserted rather than assumed.

**What it opens.** `extract_epub` still parses XHTML with the `lxml` **HTML** parser
(`XMLParsedAsHTMLWarning`, two per run) — correct output today, but a parser choice nobody has
decided. Both extractors still emit long runs of blank lines that nothing collapses. And the
fixtures now guard formats the live corpus does not contain, which is the point: the guard is in
place *before* the first EPUB arrives.

---
## 2026-08-20 (7) — a ledger of when each aspect of the project was last actually reviewed

**What changed.** New `.claude/REVIEWS.md`, and a fifth row in `AGENTS.md`'s coordination list
pointing at it. Six aspects — documentation/presentation, backend, frontend, unit testing, the cpc
system (lint, gates, CI/CD, vault), UX/UI + MCP — each with a cadence, a last-reviewed date, and a
log entry recording what that pass did **not** cover.

**Why a new file rather than a section of the baton.** `SESSION.md` is append-only with a 10-entry
cap; a cadence table needs to be *updated in place* and to survive rotation, and it would have been
archived out of existence inside two weeks. The DEVLOG is the wrong home for the opposite reason —
it records changes, and the useful signal here is the **absence** of one.

**The design decision that makes it worth having: `never` is a legal value, rendered as loudly as a
date.** Seeded honestly from the record rather than from optimism, and two rows came back `never`:
- **Backend code** — no dedicated pass is recorded, ever. Feature work and incidental fixing, yes;
  a read end to end, no. `KNOWN_ISSUES.md` shows where that lands — nine tracked defects in
  ingest/extraction alone.
- **Frontend code** — never *as a code pass*. 2026-08-19 was a live behavioural review of one
  feature, which is a different thing, and recording it as "frontend reviewed" is exactly the
  overstatement this file exists to prevent.

Two more rows are marked **partial** for the same reason: the 2026-08-19 cpc pass audited gate
wiring only (CI, the lint rule set and the vault were untouched), and **MCP has never been reviewed
because it does not exist** — the Global CLI + MCP server is parked (`docs/ROADMAP.md`, user call
2026-07-13). An empty cell there would read as an oversight rather than as a fact.

**Rejected alternatives.** *A section inside `SESSION.md`* — see above; the cap kills it. *Tracking
it in the repo* — it is working state (ADR-029), so it lives in `.claude/` with the rest; the file
says how to move it under `docs/` if it should ever be public, and nothing depends on the location.
*Recording only completed reviews* — that reproduces the blind spot the file exists to remove.

**Flagged for the cpc default template**, as the user asked, with a section at the foot naming the
two properties that must survive generalisation: `never` as a first-class value, and every log
entry stating its own gaps. A partial review recorded as a bare date reads as full coverage six
months later, which is worse than no entry at all.

**One thing it caught immediately.** `AGENTS.md` said "All four are local-only working state" about
a list whose third item is `docs/DEVLOG.md` — which is tracked. Corrected while adding the fifth
row.

---
## 2026-08-20 (6) — a fake document id read as a secret; fixed at the source rather than in the baseline

**What changed.** One id in `tests/unit/test_commands_formatters.py`:
`"abcdef0123456789abcdef0123456789"` became `"document-under-test-000000000000"`. <!-- pragma: allowlist secret -->

**Why.** The `detect-secrets` pre-commit hook failed on the change set. The finding was a **Hex
High Entropy String** — a 32-character hex document id invented as a test fixture. Not a secret,
but indistinguishable from one by entropy.

**The fix was to stop generating the false positive, not to record it.** `git add .secrets.baseline`
was the offered path and would have worked, but a baseline entry carries a `line_number`: it goes
stale the moment anything above it moves, and the next contributor inherits a suppression whose
subject they cannot locate. The other builders in that file already use patterned ids
(`f"{name}aaaa…"`, `"f"*32`, `f"{i:032d}"`) and none of them trips the detector; this one was the
odd case, so it was made to match. `.secrets.baseline` is unchanged — no new suppression exists.

Commented at the site, because the reason a test id avoids hex is not self-evident and the obvious
"tidy-up" is to put it back.

**Verified:** the full hook set passes over the change set (ruff · ruff format · bandit ·
detect-secrets · hygiene), and `.secrets.baseline` shows no diff.

---
## 2026-08-20 (5) — extraction checked against committed documents, and the third defect that found

**What changed.** New `tests/fixtures/documents/` holding a real `treatise.epub` and a
hand-authored `article.html`, a `make_fixtures.py` regenerator, a README, and
`tests/unit/test_extraction_fixtures.py` (28 passing, 4 xfailed). `.gitattributes` gains
`*.epub binary`.

**Why, and it is the gap the morning's work admitted to.** `test_extractors_formats.py` builds each
fixture with the same library that reads it back, and says so in its own docstring: a round-trip
cannot prove anything about files from *other* producers. These files are **frozen artifacts** —
once committed they stop tracking what `ebooklib` emits today, so the assertions keep describing
what a real file on disk does.

**They immediately found a third defect, and it is the worst of the three.** `get_text(separator=
"
")` breaks a line at **every inline tag boundary**, so `Emphasis <em>inside</em> a sentence`
arrives as three lines. It affects EPUB and HTML alike. Scientific prose italicises constantly —
gene names, species, emphasis — so this shatters sentences across the corpus, degrading both the
embedding and the BM25 token stream, and **no health check would ever flag it**: every character is
still present. The two already recorded in entry (1) are confirmed here on real files rather than
inferred: the EPUB nav document lands as trailing prose under a heading repeating the book title,
and the HTML `<title>` lands as the first line, above the real `<h1>`.

**All three are pinned as `xfail(strict=True)` — the first use of xfail in this suite.** Asserting
the broken output would cement the wart as the contract; a plain comment rots. `strict=True` makes
the test **fail when the bug is fixed**, so whoever fixes it is told to come back and update the
expectation. None is fixed here because each changes extraction *content* and therefore `doc_hash`
for every affected document (ADR-042) — decisions, not patches.

**`strict=True` earned its keep within the hour: it caught a bad test of mine.** The first draft of
the sentence-fragmentation test normalised newlines away before asserting, which rejoins the
fragments — so it passed against the broken output and XPASSed loudly instead of sitting there
green and worthless. It now asserts per line.

**Two smaller things the fixtures forced.** `*.epub binary` in `.gitattributes`: `* text=auto`
decides by heuristic, and a zip that gets line-ending normalisation is an unreadable file — the
suite also asserts the committed container still starts with `PK`. And the `&minus;` sign in the
HTML fixture is referred to in the test as `chr(0x2212)`, because a literal U+2212 is exactly what
ruff's RUF001 confusables rule rejects — the same rule that stopped `test_docs_encoding.py` quoting
its own evidence.

**Rejected alternatives.** *Generating these at test time too* — that is the existing file, and it
is the thing this one exists to complement. *Asserting current (broken) behaviour* — it reads as
approval and gives a future fixer nothing. *Fixing the three defects here* — they change every
affected document's identity; that belongs with a re-ingest plan, not a test commit.

**What it opens.** A decision on all three defects, which is now a single conversation with three
tripwires attached. Also noted while reading real output: bs4 raises `XMLParsedAsHTMLWarning` on
EPUB chapters (`extractors.py:152` parses XHTML with the `lxml` HTML parser), and both extractors
emit long runs of blank lines that nothing collapses.

---
## 2026-08-20 (4) — the CLI's formatters and the eval harness's one adapter

**What changed.** New `tests/unit/test_commands_formatters.py` (42 tests) and
`tests/unit/test_eval_adapters.py` (21 tests). `commands.py` was **16%** — the largest single
uncovered block in `src/` at 215 lines — and `eval/adapters.py` **26%**.

**`commands.py` is the CLI's entire user-facing surface and is almost all pure.** `test_commands.py`
covered `parse_command` and a few `execute_command` branches; every formatter had nothing. The
interesting behaviour in them turns out to be **truncation** — six independent caps (authors 50,
titles 80, keywords 10, citation snippets 120, external references 30, graph nodes 25), each
deciding how much of the truth the user sees. Where the code reports what it hid, that report is
now asserted; where a cap changes the output shape entirely, both sides are. Also pinned: health
grouping is worst-first (a broken document must not sort below a healthy one), a null health lands
in `unknown` rather than raising, absent optional fields are omitted rather than rendered as empty
labels, and the mermaid graph escapes double quotes in a filename — an unescaped one ends the label
early and corrupts the whole diagram.

**`eval/adapters.py` is small but structurally special.** Its own docstring calls it *the only
module in `doc_assistant.eval` that depends on the rest of `doc_assistant`*, so it is the seam
Feature 5 cuts along. The two things it owns are both silent-corruption risks rather than crashes:
a **fresh `TokenCounter` per query** (a leaked one would make every row after the first overstate
cost while still looking plausible) and the **citation list** that `citation_overlap` — the
harness's *zero-variance* scorer — reads. Deduplication, first-appearance order, skipping chunks
with no filename, and `0 tokens -> None` (unknown cost, not free) are now all asserted.

The pipeline is a hand-rolled ~20-line stub, not a `Mock`: it documents the shape an adapter must
present, and unlike a `Mock` it fails when a refactor changes that shape instead of accepting any
call at all.

**Rejected alternative.** *Testing `execute_command`'s remaining branches* — they reach SQLite,
DuckDB and config, so they are integration tests wearing a unit test's clothes. The formatters they
delegate to are where the logic and the line count both are.

**A tooling trap worth writing down.** `pytest --cov=doc_assistant.eval.adapters` (dotted target)
**fails at collection** with `ModuleNotFoundError: No module named '_duckdb._sqltypes'`. Coverage
imports the named module very early, which pulls the `doc_assistant.eval` package and `store.py`'s
top-level `import duckdb` into a partially-initialised C extension. duckdb 1.5.3 imports fine
normally and the full suite passes; only that invocation trips. Use `--cov=src/doc_assistant`
(path form) — which is what the full run uses — and do not go hunting for a duckdb bug.

---
## 2026-08-20 (3) — the frontend's largest untested module, tested with the runner already in the repo

**What changed.** New `apps/desktop/src/lib/graph/forceLayout.test.ts` (19 tests). The frontend
suite goes **108 to 127**, still `node --test`, still zero new dependencies.

**Why this module.** At 189 lines `forceLayout.ts` was the largest untested file in the frontend,
and its own header states the case for testing it: *"the layout is the risk; determinism is the
safety net"*. The safety net had nothing checking it. It is also pure — node ids in, a
`Map<string, Point>` out, no DOM — so it needs none of the harness the repo lacks.

**Asserted as properties, not as a golden snapshot.** Every claim the module makes about itself in
prose is now a test: identical input gives identical positions; a different `seed` gives a
different layout (without that second half the first would also pass if the seed were ignored and
the layout were merely constant); no coordinate is ever non-finite, including the dense all-pairs
case the EPS floor exists for and a 40-node graph past the spec's 21-node hub; every node lands
inside the padded box, and a larger padding demonstrably shrinks the occupied area; unknown edge
endpoints are ignored, so an unfiltered edge list lays out identically to a pre-filtered one, which
is what the docstring invites callers to rely on. A snapshot of 300 cooled iterations would break
on any tuning change while proving none of that.

**The degenerate cases are the robustness contract in the graph**: zero nodes returns an empty map
instead of throwing, one node is centred, one node with a self-edge stays finite, and two nodes
never end up coincident (they would render as a single dot). Exactly one behavioural claim is made
— a bonded pair settles closer than an unbonded one — and it is kept deliberately weak, because
anything tighter is a snapshot of the cooling schedule wearing a property's clothes.

**Rejected alternatives.** *Adding vitest + jsdom + @testing-library/svelte first* — it is the
right next step and the standing gap is real (**39 `.svelte` components cannot be tested at all**
under `node --test`, which is why yesterday's CSS source-order bug and the four-read-path merge bug
were both caught by eye rather than by a test), but it is a tooling decision with a lockfile
change, and it should not ride along inside a test-writing commit. `forceLayout` needed none of it.
*Testing `conversations.svelte.ts` in the same pass* — it is the other module worth covering (it
owns the new `archiveConversations`), but it is `$state`-backed and fetch-driven, so it wants the
harness decision made first.

**What it opens.** The component harness question, unchanged and now the largest single gap in the
project's testing. Also worth noting for whoever takes it: the frontend runner is
`node --test src/**/*.test.ts`, so a test file placed outside `src/` runs nowhere and reports
nothing.

---
## 2026-08-20 (2) — the two other untested user-facing read paths: the chat router and the citation graph

**What changed.** New `tests/unit/test_query_router.py` (45 tests) and a `graph_subgraph` section
appended to `tests/unit/test_library.py` (9 tests). `query_router.py` goes **29% to 100%**;
`library/citations.py` **73% to 99%**. Same ranked coverage pass that produced the extractor work
above; these were the next two entries on it where the uncovered code is read by a user.

**`query_router` was the lowest-covered module that runs on a user request.** `ChatController`
consults `is_library_query` on *every* message (`chat_controller/controller.py:305`); a match
short-circuits the entire RAG pipeline and `answer_library_query`'s string is shown verbatim.
Neither half had a test.

The load-bearing part is the negative lookahead `_NOT_TOPICAL`, which is what keeps *"show my
papers about RAG"* out of the metadata branch. If it regresses, a content question silently
receives a document count — no exception, no log line, no way to notice except reading the answer.
**The test asserts both halves of that difference**: the bare phrase must match *and* the
qualified one must not. Asserting only the rejection would keep passing if the pattern stopped
matching altogether, leaving a green test guarding nothing — which is the failure mode that made
KI-41's chunking sweep compare one configuration with itself six times.

Also now pinned: the empty-library branches (`answer_library_query` on 0 documents names
`data/sources/` rather than returning a bare zero), the "documents exist but have no add dates"
case that refuses to nominate a latest, the `and N more` truncation on long broken lists — a
silent cap would read as *these are all of them* — and, over every phrasing the router claims,
that an answer actually comes back. That last one is the contract *between* the two functions,
which is where a blank reply to the user would come from.

**`graph_subgraph` had 0% coverage** and is one of the four surfaces KI-45 names as still trusting
`Citation.target_document_id` unguarded — the References block re-checks resolutions at read time,
the citation graph does not. It is also the only traversal in the module that walks both
directions and dedupes. Pinned: exactly one centre, both directions at depth 1, unresolved
citations excluded (a bibliography is mostly external — 36 references and 0 library matches on the
recovered scan), `depth` actually bounding the walk, repeated citations collapsing to one edge
(nine of one document's eleven resolutions pointed at two papers), termination on a cycle, and an
unknown id returning an empty graph instead of raising into the panel.

**One line is deliberately left uncovered.** `citations.py:343` — the `if nid in visited: continue`
guard — is unreachable as the traversal is written: a node only enters `next_frontier` when it is
not already in `nodes`, and every visited node is in `nodes`, so no node can be queued twice. It is
defensive, and deleting a defensive guard to buy a coverage point is the wrong trade. Recorded here
so the next coverage pass does not re-litigate it.

**Rejected alternatives.** *A new `test_citations_graph.py`* — `temp_database` is defined locally
in `test_library.py`, and copying a 40-line engine-patching fixture to a second file is how the two
copies start drifting (the same argument that made `effective_metadata` one function yesterday).
*Patching `doc_assistant.library.list_documents` for the router tests* — `query_router` binds both
names at import, so patching the source module leaves this module's references untouched; the
patch targets `doc_assistant.query_router`, the trap `chat_controller/__init__.py` already
documents.

**What it opens.** The router's regexes are still English-only and pattern-based; the tests pin the
behaviour, not its adequacy. And `graph_subgraph` is now tested but still *unguarded* — KI-45's
false resolutions reach it exactly as before. A test that pins current behaviour is not a fix, and
the write-side precision fix is still the thing that matters there.

---
## 2026-08-20 — five of the seven supported formats had no extraction test at all; now they do

**What changed.** New `tests/unit/test_extractors_formats.py` (37 tests). `extractors.py` goes
from **44% to 91%** line coverage. The only lines still uncovered are `extract_pdf_pymupdf`'s body
(92-103, 262), which needs a committed PDF.

**Why, measured before writing anything.** A coverage pass over the whole suite (1,839 tests, 89%
overall) put `extractors.py` at the bottom of the ranked list with **lines 141-220 never executed
once** — that range is the entirety of `extract_epub`, `extract_html`, `extract_docx`,
`extract_rtf` and `extract_odt`. `test_extractors.py` covered format *detection*, `.txt`/`.md`, and
the PDF placeholder strip; five of the seven extensions in `SUPPORTED_EXTENSIONS` had no test
touching their extractor. That is the app's front door, in the subsystem with the most tracked
defects (KI-14/26/34/40/42/43/44/46/47 — nine issues, three of the five currently open).

**What the tests assert**, beyond "output is non-empty": the structure heuristic per format (DOCX
style names to `#`/`##`/`###`; the EPUB Dublin Core title leading as an H1; HTML heading
conversion), the one content decision `extract_html` makes (script/style/nav/footer are
decomposed — stated per tag, so a future edit to that tuple fails on the tag it dropped), RTF
control-word stripping and ANSI escape decoding, dispatch through `extract_to_markdown` for all
five, `.htm` and `.html` reaching the same extractor, and `get_format_status`'s advisories.

**Every format asserts a non-ASCII round-trip**, because that is the failure this project has
already shipped: nothing on Windows defaults to UTF-8 (CONTEXT.md section 9), four tracked docs
were committed double-encoded, and an extractor that mangles an accent produces garbage that still
*reads* as prose — so it passes ingest, reaches the chunk store, and nothing downstream notices.
`test_html_reads_as_utf8_regardless_of_the_ansi_codepage` pins the explicit `encoding="utf-8"` on
that read specifically.

**Two defects surfaced while writing the fixtures. Neither is fixed here** — they are extraction
*content* decisions, not test bugs, and fixing them changes `doc_hash` for every affected document
(ADR-042), which is not a thing to do inside a test commit:
- **EPUB pulls the navigation document into the body.** `get_items_of_type(ITEM_DOCUMENT)` includes
  the generated nav/TOC item, so a book's markdown ends with its own table of contents as prose.
  The tests assert title, headings and body are present; they deliberately do **not** assert the
  nav text is absent, so the wart is not cemented as expected behaviour.
- **`extract_html` leaks `<head><title>` into the text.** Only script/style/nav/footer are
  decomposed, so `soup.get_text()` emits the title as a bare leading line, indistinguishable from
  body prose. Same shape as the page-furniture problem behind the keyword layer's singleton rate.

**Rejected alternatives.** *Committed binary fixtures* — better at catching producer variance, but
an opaque blob in a public repo, and every writer library here (`python-docx`, `odfpy`,
`ebooklib`) is a **base** runtime dependency, so an in-test fixture can never skip for a missing
library and shows the reader the input beside the required output. HTML and RTF are hand-authored
markup, so those two are producer-independent regardless. *Conditional skips per format* — the
suite has 8 skip markers and all 8 are environment-conditional; a skip here would hide exactly the
breakage the file exists to catch. *A literally empty EPUB body for the zero-content case* —
`ebooklib` cannot serialise one (lxml raises "Document is empty" building the nav), so it would
have tested the fixture builder; a whitespace-only paragraph tests the extractor.

**What it opens.** These are round-trips through the same library that wrote the file, so they
cannot prove the extractors survive files from *other* producers — the variance that broke tier-1
citation parsing (KI-45). Said explicitly in the module docstring rather than left implied. Closing
the PDF gap (92-103) needs a committed fixture and is the same binary-in-a-public-repo decision,
deferred. The two leaks above want an issue each before anyone re-ingests.

---
## 2026-08-19 (3) — a metadata override applied in the Library grid and not on the document's own page

**What changed.** ADR-013's `override ?? auto` merge now lives in one function,
`effective_metadata`, and **all four read paths that display a document use it**:
`list_documents` (which already merged, inline), `get_document_details`, `get_document_chunks`
(the document page's own header) and `list_document_figures`. Three of the four did not.

**How it surfaced.** Correcting the OCR-derived title on the recovered scan
(*"A Revised Neuroanatom of Frontal—Subcortical Cireuits"* → *"…Neuroanatomy of
Frontal-Subcortical Circuits"*) wrote the override correctly and changed nothing on the page it
was edited on. The list endpoint returned the corrected title; the detail endpoint returned the
extracted one. `list_documents` merged inline, `get_document_details` read `doc.title` straight —
the same rule written twice, and the second copy was simply missing.

**Worth noting about the write path: it was already right.** The PATCH cleared `year_override`
because the submitted 2001 equalled the extracted 2001 — ADR-013's "an effective value equal to the
default clears that field" — so only title and authors persisted. The bug was entirely on the read
side, which is why editing appeared to half-work rather than fail.

**One path deliberately does NOT merge.** `document_years` feeds the year-aware epistemics rule
(G3), which is an analysis of what the corpus *says*, not a display of what the user prefers to
see. Letting a metadata edit move a `superseded_trend` verdict would change a knowledge-layer
result for a reason no baseline records — that is a decision with an eval behind it, not a
consistency fix. Said in a comment at the call site and pinned by a test.

**Guarded by eight tests**, the load-bearing one being *every display surface agrees on the title*
— stated once over all four paths, so a fifth surface added without the merge fails in the suite
rather than in a screenshot. Also pinned: no override means both report what
extraction found (the override stays additive), and a title-only override leaves authors and year
extracted.

**Rejected alternative.** *Merging in the detail function too* — that is the fix that created the
bug: two copies of one rule, drifting the moment someone touches one. The helper is the point, not
the call site.

**What it opens.** Nothing else reads `Document.title` directly for display — checked — but the
same shape exists wherever an additive sidecar has a "merge for display" step. Also unresolved: the
extracted values under the override are still the OCR ones, which is correct (ADR-043 keeps
received content verbatim; the override is a separate, inspectable layer) but means a re-extraction
never silently corrects them.

---
## 2026-08-19 (2) — chat select mode reviewed: the tick was stacked ON the row by a CSS ordering accident, and the reported "connection issues" were another project's dev server

**What changed** (`Sidebar.svelte`, `App.svelte`, `conversations.svelte.ts`), from a live review:

- **The tick now sits left of the title, email-client style.** It had been rendering *above* the
  row and centred, clipping names. The markup was always right — the tick precedes the body in the
  DOM — and the cause was CSS: the element carries both `.pickrow` (`display:flex`) and `.rowmain`
  (`flex-direction: column`, for the normal row). Equal specificity, so **source order decided it**,
  and `.rowmain` is defined 60 lines later. Fixed as `.rowmain.pickrow` — a two-class selector, so
  the row layout no longer depends on where either rule sits in the file. Verified in the running
  app: `flex-direction: row`, tick at x=15 and body at x=39, vertically centred, and **all 100 rows
  share one left edge**, with titles truncating on `text-overflow: ellipsis` instead of being cut.
- **The action bar is sized like a toolbar** (0.5rem padding, 0.8rem type, 41px tall against ~24px)
  — it is the only way out of a destructive mode and it read as a caption.
- **"Done" is gone.** The header toggle already reads *Leave select mode* and does the same thing;
  two controls for one action, and the redundant one sat where a confirm button would.
- **Bulk archive added**, mirroring bulk delete. It is a *toggle*: when every ticked chat is already
  archived the button reads **Unarchive**, which is what the "Show archived" view needs. Applied
  immediately rather than through a confirm — archiving is reversible and the toggle right below it
  brings them back (delete keeps its confirm). `archiveConversations` PATCHes concurrently and
  refreshes the list **once**: N single-archive calls would each refresh, rerendering the sidebar
  under the user mid-action. A failure is logged per conversation and does not abandon the rest.

**Verified end to end in the running app, and the data left exactly as found** — ticked one chat,
Archive, confirmed archived server-side; re-selected it under "Show archived", confirmed the button
now read *Unarchive*, clicked it, confirmed 100 conversations / 0 archived again.

**The "connection issue inside library" was not a defect in this app.** Reported as
`TypeError: Failed to fetch` on connections, chunks, figures and references. The API was healthy
throughout (200s in ~14 ms directly on 8001). **Port 1420 was being served by a different
project's dev server** — the page at `localhost:1420` is titled "Scribe", and
`localhost:1420/api/health` returns **404 with that project's HTML**. Provenote's Vite owns `/api`
as a proxy to 8001; with the port taken (and `strictPort: true`), an already-open Provenote window
keeps rendering while every later `/api` call goes to a server that has never heard of it. Same
cause for the graph panel. Re-verified on the alt port from `.claude/launch.json`: connections,
figures, references and the document payload (66 parent blocks) all 200.

**Two diagnostic notes worth keeping.** `curl http://127.0.0.1:1420` reported nothing while
`localhost:1420` answered — Vite binds `::1` only, so an IPv4 probe says "dead" about a server that
is running; check both before concluding a process is down (I concluded it once and was wrong).
And the *shape* of the error was the tell: `Failed to fetch` is connection-level, so it points at
who is answering the port, not at the handler.

**Rejected alternatives.** *Setting `flex-direction: row` on `.pickrow` alone* — it would work today
and break again the next time a rule is appended below it. *A checkbox `<input>`* — the whole row is
the target on purpose (a 15px hit area in a list is poor), and the tick is presentational. *A
confirm for bulk archive* — reversible actions that ask twice train people to click through
confirms. *Leaving "Done"* — the user's point stands: the escape hatch already exists above it.

---
## 2026-08-19 — the corpus's one broken document is recovered (96 → 97 retrievable), and the reason it was broken is not what it looked like

**What changed.** No code. `middleton-2001.pdf` — 15 pages, `extraction_health='broken'`,
`chunk_count=0`, the only unhealthy document in the library — is now **healthy, 52 chunks, and
retrieved rank 1** on three of the RG-025 baseline's queries against the full 97-document library.
`data/library.db.bak-20260819-preocr` is the pre-change backup.

**The finding is the trigger, not the text.** RG-025 measured this document's OCR a fortnight ago
and concluded the text was worth retrieving; that was never in doubt. What is new is that **the
shipped extractor produced it on its own**: same code, same pinned dependencies, same PDF, 0
characters on 2026-08-08 and **34,600 today**. PyMuPDF4LLM finds a `tesseract` binary on `PATH` and
OCRs pages with no text layer. Nothing in the repo changed — the box's `PATH` did. Filed as
**KI-47**, because extraction output is supposed to be a function of things the cache fingerprint
can hash, and an external binary is not one of them: the stale 349-byte extraction stayed "fresh"
forever with a fingerprint that still matched the live one to the digit. **The installer ships no
OCR** (`tesseract` is a system binary; the spec, README and setup docs never mention it), so the
app's real behaviour on a scan is still the 0-character one and this box is the outlier.

**A second defect fell out of the fix.** `doc_hash` hashes the extracted *text*, so recovery mints a
**new document identity** — and the old row survived a full cleanup pass, leaving the file in the
library twice. `cleanup_orphans_sqlite` builds its candidate set from **Chroma metadata**, so a
document with **zero chunks is invisible to the orphan sweep** and its row can never be classified
stale. That is the shape of every recovery, not a one-off: a broken document has no chunks by
definition. Filed as **KI-46**; the stale row was removed with the same `session.delete(row)` call
cleanup uses (FK cascades drop the outbound enrichment), guarded on `chunk_count == 0` plus a
healthy sibling — **not** through the app's delete path, which would have sent the source PDF to the
Recycle Bin (ADR-014).

**One measured improvement over the 2026-08-08 run, unexpectedly.** That baseline's single
highest-value carried item was *de-hyphenate line-broken words* — raw Tesseract left **88** of them
(`cortico-` + `spinal` never matching a query for `corticospinal`). This path leaves **0**: mean
line length 250 characters, 3 lines ending in a hyphen, and `corticospinal` present intact. The
reflow defect that motivated the work does not occur here.

**Rejected alternatives.** *`--rebuild`* — it wipes the vector store and re-embeds all 97 documents
to fix one. *Hand-running Ghostscript + Tesseract and pasting the text in* (what the baseline did)
— unreproducible, and it would put unmarked OCR prose in the library by a path no runner owns.
*`delete_document`* — bins the source file. *Fixing the cleanup blind spot inline* — a real change
to a core ingest path, which deserves its own tests rather than riding on a data repair.

**What it opens.** The corpus is now **97 retrievable documents, digest changed** — so by the
comparability layer's own rules every earlier run (96 documents) is *not comparable* to anything run
from here, and `compare_runs` will now say so instead of leaving it to memory. ADR-039 is still
**proposed**: what happened today is recovery that is neither opt-in nor marked as OCR-derived,
which is exactly what that ADR wanted to avoid — recovered text now enters as ordinary prose that
retrieval and citation cannot distinguish from a real text layer.

---
## 2026-08-18 (2) — a committed baseline now carries its own evidence, so it can be checked without the run store

**What changed.** New `eval/baseline_doc.py` + `scripts/emit_baseline.py` (+ `just emit-baseline`)
write a baseline document **from the run record**, and `compare_runs --against <file>` checks a
later run against that document.

**The gap.** `tests/eval/baselines/` is the committed reference record; `data/eval.duckdb` is
gitignored. So the numbers travel and the evidence does not — a fresh clone holds ~30 documents
whose setup sections were typed by hand, and nothing can contradict them. That is exactly where the
last error hid: the Haiku-vs-llama split across the 2026-08-08 arms lived only in prose, and prose
is recoverable by a human reading the right file and by nothing else.

**What the emitter writes, and what it refuses to write.** Settings, corpus composition, generator,
judge, and the aggregate table — copied from `config_json`, never re-derived — plus a visible fenced
JSON provenance block (visible, not an HTML comment: the reader should see exactly what the checker
reads). It **refuses to emit from runs that are not one experiment**, because a baseline averages
its trials and mixing two would present them as one number; the refusal prints the comparability
report that explains why, and exits 4. It writes `TODO` for the judgement, because the caveats are
what make a baseline worth keeping and no emitter can derive them. A key the runs disagreed on is
**dropped**, not taken from the first trial, and then renders as "not recorded" — the document
cannot vouch for it.

**Older baselines parse to nothing, on purpose.** `parse_provenance` returns `{}` for a document
with no block, a malformed block, or a JSON block that is not provenance — so `--against` an
existing hand-written baseline reports *unknown across the board* and says the document never
recorded those facts. Verified against the real folder: every committed baseline parses without
raising, and the check against `chunking_sweep_private_2026-08-08.md` names 20 unrecorded settings
rather than inventing agreement.

**Verified end to end, $0.** Emitted a baseline from the two identical-settings validation runs,
then checked runs against the committed file: the matching run exits **0**, the qwen2.5:3b run exits
**4** with `contains_all` not comparable and `citation_overlap` still fine, and the hand-written
baseline path exits **3** with the "carries no provenance block" note. 27 new unit tests.

**Rejected alternatives.** *An HTML-comment block* — hidden from the reader, which invites the
document and its record to drift; this layer exists because a claim and its evidence drifted apart.
*Emitting the caveats too* — an emitter can produce a table, not a judgement, and a baseline that is
only numbers is what the project already has too many of. *Taking the first trial's value when
trials disagree* — it would state something no reader could act on. *Raising on a baseline with no
provenance* — a checker that rejects the entire existing corpus of baselines is a checker nobody
runs; `{}` and an honest "unknown" is the useful behaviour. *Back-filling provenance into the ~30
existing documents* — the same rule as everywhere else here: an inference must not become
indistinguishable from a recording.

**A document that argues with itself is caught too.** `table_drift` compares the visible results
table against the document's own provenance block and reports any row where they disagree — the
small nasty failure where someone tidies a number in the pretty table, or copies a document and
edits it, while the machine block still carries the original. Verified by tampering with an emitted
baseline: the check names the scorer and both values before the verdict, because the reader needs
to know which half of the document they have been quoting.

**What it opens.** The existing baselines stay unpinned unless someone re-runs and re-emits them,
which is a deliberate non-goal — their numbers are still valid readings, they simply cannot be
machine-checked. No gate *requires* a new baseline to carry a block, so the discipline is still a
habit rather than a rule; and drift is only checked on the mean, not on the std or the counts.

---
## 2026-08-18 — the harness now answers "may these two runs be compared?", per scorer, and says UNKNOWN when nobody wrote it down

**What changed.** Yesterday's entry made a run record what it measured. This makes something *read*
that record. New `eval/comparability.py` (generic, no app import — it travels with the harness),
plus `report.compare_runs` / `format_comparability`, a `scripts/compare_runs.py` CLI, a
`--baseline RUN_ID` flag on `run_eval`, and `just compare`.

**The idea it encodes, which this project had already written by hand twice.** A score depends on a
*prefix* of the pipeline: cases → index → retrieval → generation. `citation_overlap` and
`figure_retrieval` read the **retrieved documents** (`output.citations`, `output.raw["retrieved"]`,
both filled from `pipeline.retrieve` before a token is generated), so they survive a generator swap.
Everything else reads `output.answer` and does not. That asymmetry is exactly what localised RG-029
— `citation_overlap` reproduced to the digit while `contains_all` moved — and it is the reasoning in
the generator caveat at the top of `chunking_sweep_private_2026-08-08.md`. A hand-written caveat
protects only the reader who opens the right file, so now one differing setting yields **per-scorer**
verdicts rather than a wholesale pass/fail.

**Three states, and UNKNOWN is the common case, not the corner one.** Of the 75 runs in the live
store, **not one** records its generator or its corpus. So the honest verdict for almost every
historical pair is *unknown*, and the layer says so: an unrecorded setting is never assumed to have
matched, and it is never inferred — not from the run's `note` prose, not from a sibling run. The
five annotated Haiku trials carry their generator in `note` precisely so that an inference cannot be
mistaken for a recording, and `--list` shows them as "not recorded" for the same reason.

**Verified against real runs, $0.** Three one-case runs on the live 96-document index over local
Ollama, two of them differing *only* in generator:

| | `contains_all` | `citation_overlap` | verdict |
|---|---:|---:|---|
| llama3.1:8b vs llama3.1:8b | 0.500 → 0.500 | 1.000 → 1.000 | **comparable**, exit 0 |
| llama3.1:8b vs qwen2.5:3b | **0.500 → 1.000** | 1.000 → 1.000 | **not comparable** on `contains_all`, comparable on `citation_overlap`, exit 4 |

The second row is RG-029 reproduced deliberately: a 3B model "beats" an 8B one by +0.500 on the
answer score while the retrieval score does not move a digit. Also verified on the real 75-run
store, where every pair comes back UNKNOWN with the missing keys named.

**`--varying` is the half that makes it useful to a sweep, and it catches KI-41 from the record.**
A sweep *intends* to differ in one setting, so a bare comparison would object to the experiment
itself. Declaring the independent variable (`--varying child_chunk_size`) stops that difference
blocking while **everything else still blocks** — which is the useful direction, because a sweep's
real risk is that something besides the grid moved. The opposite failure gets its own field and its
own exit code: a declared variable that came back **identical** means the arms are one configuration
compared with itself, which is KI-41 exactly (the 2026-06-06 sweep drove its grid through
environment variables `.env` silently overwrote). Verified on two real runs: verdict `ok`, and a
banner above the table saying the declared variable did not change, exit `5`. Note the deliberate
split — those runs *are* comparable; it is the experiment between them that is void, so it must not
be folded into the comparability status. `sweep_chunking`'s preflight catches this before a run;
this catches it afterwards, from the record, where a preflight-less runner cannot hide it.

**Two smaller things fell out of building it.** `judge_provider` / `judge_model` are now recorded —
`llm_judge` grades with a model that is *not* the generator, so two runs can share a generator and
still have been graded differently; it is the same membership rule as yesterday's keys, and its
absence would have been an invisible hole in exactly the scorer that costs money. And
`Store.resolve_run_id` now owns prefix resolution, so `run_eval --baseline` and `compare_runs` cannot
resolve an id two different ways.

**Rejected alternatives.** *A plain config diff* — it would flag `trial_index` (which differs by
design between trials of one experiment) and would report a generator swap as invalidating
everything, including the one number that survives it; the per-scorer prefix is the whole value.
*Treating an unrecorded setting as unchanged* — that is precisely the assumption RG-029 was, and it
would return the store to printing two means side by side. *Parsing the `note` prose* to recover the
five attributed Haiku trials — the annotation exists because a back-filled inference must stay
distinguishable from a recording. *Suppressing the score tables on a NOT COMPARABLE verdict* —
informing beats blocking; a suppressed table sends the reader to the raw store, where there is no
verdict at all. *Comparing every trial against the baseline under `--repeat N`* — the trials share
their settings by construction, so it would print the same verdict N times. *Folding an
ineffective variation into the comparability status* — "these runs cannot be compared" and "these
runs are identical when they should not be" are opposite diagnoses with opposite fixes.

**What it opens.** RG-021 is now closed end to end (record + warn). Not built: nothing *emits* a
baseline markdown from a run record, so `tests/eval/baselines/` stays hand-written; nothing compares
a run against a *committed baseline file* (only against another stored run); and the 75 historical
runs remain permanently unknown, which is a fact about them, not a gap to fill. `sweep_bm25_weight`
still persists nothing, so it participates in none of this.

---
## 2026-08-17 — an eval run now records the corpus it measured and the generator it used, and a paid one says so before it spends

**What changed.** Three keys and a flag, closing the "make the run record and the run agree" class
that RG-029 opened and RG-021 kept open.

- `sparse_index.doc_set_digest(doc_hashes)` — SHA-256 over the sorted, de-duplicated `doc_hash`
  set. A *corpus* identity, deliberately not `fingerprint`, which identifies a *build*: chunk ids
  move on every re-ingest, so it cannot answer "same documents?" — the question a chunking sweep
  needs answered while the geometry is changing under it.
- `RAGPipeline.indexed_doc_hashes` — every document retrieval can currently reach, read from the
  keyword arm because that arm **is** the BM25 corpus (same `keep_for_retrieval` filter as the
  vector arm). Three states: the hashes, `set()` for an empty corpus, `None` only when a build
  failed over a non-empty corpus and this process genuinely never learned what the store held.
- `scripts/run_eval.py` records `index_doc_count` + `index_doc_digest` (RG-021) and
  `llm_provider` / `llm_model` **as the run actually ran them**, and grows `--provider` /
  `--model`. A paid generator now trips `llm.assert_provider_intent` — the same banner + 3 s abort
  window every enrichment runner has had since Feature 7.

**Why the composition comes from the pipeline and not from `run_defining_settings`.** Config does
not know the corpus; it is state on disk. Asking the live pipeline is also the only source that
cannot drift from `pipeline.py`'s own "which collection is active" rule — a second copy of that
resolution in the harness edge is exactly the restated-gate failure `sweep_chunking`'s preflight
docstring warns about. The cost is the usual injection cost: a new runner must record it, so
`scripts/CLAUDE.md` now carries that rule beside the `settings_provider` one.

**Why `--provider` refuses to change provider without `--model`.** `.env` pairs `anthropic` with
`claude-haiku-4-5-…`; inheriting that model under `--provider ollama` hands Ollama a name it has
never heard of, and it fails once per case, minutes in. The error names the model it refused to
inherit and the flag that fixes it.

**`synthesis_mode` was asked for and is deliberately NOT recorded — the finding is bigger than the
key.** `ai` vs `human` does change the answer, but not this harness's answer: `eval/adapters.py`
calls `pipeline.retrieve` + `pipeline.stream_answer` directly, and `stream_answer` is a bare
`ANSWER_PROMPT | llm` chain. `SYNTHESIS_MODE` is read only in `chat_controller.helpers`, which the
eval path never enters. Recording it would pin a setting the run did not honour — RG-029 with the
sign flipped. The consequence worth carrying: **the eval harness measures the raw answer path, not
the shipped one** (no synthesis split, no provenance, no reviewer). Written into
`eval/run_settings.py`'s docstring, where the membership rule lives.

**Verified, $0.** One case over the live corpus on `llama3.1:8b` into a scratch DuckDB:
`config_json` came back with `index_doc_count 96`, a 64-char digest, and `llm_provider ollama` —
which is proof the override reached the record, since `config.LLM_PROVIDER` still reads
`anthropic`. The paid banner was confirmed on the default path by killing the process inside its
own abort window, before the pipeline loaded. Unit tests: corpus digest (order/duplicate
independence, one-extra-document sensitivity, and that it survives a re-chunk where the build
fingerprint does not), the pipeline's three states, and both CLI helpers.

**The first thing the new key found.** The index holds **96** documents where `library.db` holds
**97**: `middleton-2001.pdf` has `chunk_count = 0` and `extraction_health = 'broken'`, so it has
never been retrievable. Every eval run on this box has been over 96 documents, and now says so.
Not a regression and not caused by this change — but it is the shape RG-021 predicted, found on
the first read, and it is very likely the same document as the carried "one document still
extracting zero keywords" item (no text, no keywords).

**Rejected alternatives.** *Composition from `library.db`* — it is the registry, not the index:
its non-archived filter does not match what retrieval can reach (archiving hides a document from
the UI, not from BM25), and it would have reported 97. *Composition inside
`run_defining_settings`* — it would reach the sweeps for free, but that function is a pure
config snapshot called on every `persist_run`, including in unit tests and in the sweep's probe
subprocess; giving it disk I/O makes the suite depend on machine state and CI has no index at
all. *Recording `synthesis_mode` anyway* — see above. *A bare `nosec`-style default flip to
`ollama`* — moving the default is a policy change with an eval consequence; the flag plus the
banner makes the current default impossible to trip over accidentally, which was the actual
defect.

**What it opens.** RG-021's **build** half is done; its **warn** half is not — nothing yet
compares a run's composition against the baseline it is being read beside, and the 70 historical
runs carry no composition to compare against (same unattributable-history problem RG-029 has).
`sweep_bm25_weight.py` still writes its own baseline markdown rather than persisting runs, so it
records none of this. And `middleton-2001.pdf` needs a decision: re-extract, or accept 96.

---
## 2026-08-16 — `just clean` exists because `cargo clean` would delete the one installer with no copy anywhere

**What changed.** New `scripts/clean_build.ps1` + a `clean *ARGS` recipe in the `justfile`. It drops
`target/debug` in full and everything under `target/release` **except `bundle/`**. Flags: `-DryRun`
(report only), `-Registry` (also drop `~/.cargo/registry/src`), `-Force` (override the guard). Invoked
through `powershell -File` like the existing `app:` recipe, because `set windows-shell := cmd.exe` means
recipe bodies are single commands. ASCII-only per the `.ps1` rule (§9).

**Why `cargo clean` is the wrong tool here specifically.** It wipes all of `target/`, and this repo's
`target/` holds two things that are not intermediates: the current installers in `target/release/bundle`
(3.06 GB at 0.5.1) and a 1.53 GB build-time copy of the frozen sidecar. Of those installers **only the
NSIS `.exe` is attached to the GitHub release — the MSI exists nowhere else on earth**, and `cargo clean`
offers no way to keep it while dropping the rest. Measured on this box today: `target/` was 5.97 GB, of
which 2.89 GB was reclaimable with zero loss. The frozen sidecar in `src-tauri/binaries/` is outside
`target/` and so was never at risk — worth stating, since it is the artifact that costs a PyInstaller
re-freeze.

**Why the active-build guard is scoped rather than machine-wide.** First cut refused to run whenever any
`cargo`/`rustc` lived. On this machine that is close to always — a `cargo build --workspace` and a
`cargo test -p harper-core` from an unrelated project blocked it twice while it was being written, and
during the final verification **24 foreign `rustc` processes** were alive. A guard that always cries wolf
is one you learn to `-Force` past, so it now blocks on what each scope can actually break: a build **in
this repo** blocks the target clean (detected via `rustc`/`link` absolute `--out-dir`, with a fresh
`target/` mtime as the cwd-independent backstop), while **any** build blocks `-Registry`, since that cache
is shared with every Rust project on the box. Both branches were verified live: the target clean proceeded
with 24 foreign processes running, and `-Registry` aborted against the same 24.

**Rejected alternatives.** *Plain `cargo clean` in the recipe* — the whole point is that it takes the
installers and gives nothing back cheaply. *Dry-run by default*, matching `scripts/`' enrichment-runner
convention — a `clean` that cleans nothing is a worse surprise than one that does; `-DryRun` is opt-in and
exempt from the guard instead, so looking is always free. *Registry cleanup in the default path* — it is a
cache shared across projects, so it stays opt-in. *Teaching the recipe to prune installers* — that is
`docs/RELEASE.md` §8's job and it is deliberately manual and deliberately about the **previous** release.

**What it opens.** Nothing deletes an installer without a human: §8 remains the only path. The wider
finding this came from is that **this repo was never the disk problem** — the reclaimable GB on this box
sit in *other* projects' `target/` dirs (~18 GB across two of them), which no recipe here should touch.

---
## 2026-08-15 (3) — the DEVLOG had grown to 8,244 lines because nothing made rotation happen; now a test does

**What changed.** 112 entries (**2026-07-15 → 2026-08-01**) moved verbatim to new
`docs/archive/DEVLOG-archive-002.md`; the live log goes **8,244 → 2,550 lines** and keeps
2026-08-02 onward. New `tests/unit/test_doc_sizes.py` caps the append-only coordination docs —
`docs/DEVLOG.md` 4,000 · `.claude/KNOWN_ISSUES.md` 1,800 · `.claude/RIGOR_TODO.md` 1,500 — skipping
the local-only ones when absent. The DEVLOG header now lists both archives and states the cap.

**Why — the belief that a cap already existed is the interesting part.** `scripts/conventions.toml`
caps the entry file (`entry_max_lines = 600`), each module `CLAUDE.md` (40), and the baton
(`session_max_entries = 10`, cpc ADR-018 rule 11b — the rule that rotated `2026-08-10` out of the
baton earlier today). **None of them covers the DEVLOG**, and `docs_check` implements no DEVLOG rule
at all: it reads the file only as a *route source*. So the log had exactly one rotation ever
(2026-07-21, by hand) and nothing to make the second one happen. **623 KB and 6.4x the next largest
doc in the repo** is the result — reached without any single bad commit, because every entry is
individually small and correct and the growth is only visible in aggregate.

**Why a pytest guard rather than a cpc gate.** The cpc tooling under `tools/conventions/` is
vendored, gitignored and owned by another project (ADR-001/ADR-007) — a cap added there is lost on
the next refresh, and it cannot run in CI because cpc is private while this repo is public. This
repo's own suite is the durable home, which is already why `test_docs_encoding.py` lives there.

**Rejected alternatives.** *Cap by bytes* — lines match `conventions.toml`'s existing
`*_max_lines` vocabulary and are what a reader actually feels. *Cap `.claude/SESSION.md` too* — it
is already capped at 10 *entries*; a second cap in a different unit could fail an entry-compliant
baton, so the two rules would contradict each other (the test says so where someone would think to
add it). *Rotate KNOWN_ISSUES and RIGOR_TODO now as well* — at 1,299 and 1,034 lines they are
under cap with real headroom; rotating them today would be churn, and the guard will say when.
*Keep a whole month live* — that is what produced the problem; two weeks is the working window.

**What it opens.** The caps are calibrated to ~40-60% above post-rotation size, which at this
project's busiest rate (~250 DEVLOG lines/day) is roughly a fortnight per rotation — if that proves
too frequent the answer is a smaller live window, **not a bigger cap**, and the test says so in its
failure message. `KNOWN_ISSUES.md` already has its rotation target
(`docs/archive/KNOWN_ISSUES-resolved-001.md`); **`RIGOR_TODO.md` has none yet**, so the first time
its cap trips someone must create `RIGOR_TODO-archive-001.md` and decide what "closed enough to
archive" means for an RG entry — closed ones like RG-026/RG-028 are the obvious candidates.

---
## 2026-08-15 (2) — an eval run now records which LLM wrote its answers (RG-029)

**What changed.** `eval/run_settings.py` records `llm_provider` + `llm_model` alongside
`embedding_model`; its docstring carries this incident next to the KI-41 one it was written for.
Two guards in `tests/unit/test_eval_run_settings.py`: the generator is recorded, and two runs on
different models are distinguishable. New `RG-029` in `.claude/RIGOR_TODO.md`. Committed `83b730b`.

**Why — a model swap was reading as a pipeline win.** Auditing a 5-trial control run on the private
35 against the committed baselines: `contains_all` **0.822** where the 2026-08-08 control recorded
**0.777**. That is not an improvement — the 08-08 run generated with `llama3.1:8b` and this one
inherited `.env`'s Anthropic Haiku. Nothing in the run record could say so: across all 75 runs in
`data/eval.duckdb`, the union of `config_json` keys contained exactly one model name,
`embedding_model`.

**The tell, and why the diagnosis is trustworthy rather than plausible:** `citation_overlap`
reproduced *to the digit* (0.9363 vs 0.936). Retrieval is deterministic and generator-independent,
so an unchanged retrieval score sitting beside a moved answer score localises the change to
generation. Any reading in which the pipeline improved has to explain why the deterministic half did
not move.

**This is KI-41 one layer up.** KI-41 was a sweep whose arms were not what its note claimed,
invisible because `config_json` recorded no chunk sizes. Here the Haiku-vs-`llama3.1:8b` split
across the 2026-08-08 public and private arms is real and documented — but only in
`evals/README.md` **prose**, recoverable by a human reading a document and by nothing auditing the
data. The module's own membership rule already decided this case: *a value belongs here if changing
it changes what the run measures.*

**Rejected alternatives.** *Retro-fill the 70 historical runs by inference* — their generator was
never written down; "unknown" is the honest state and a guessed provenance is worse than none.
*Also record `synthesis_mode` in the same commit* — it has the identical defect (`ai` vs `human`
changes the answer, so it changes `contains_all`), but it was outside what was asked; flagged for
the next pass instead of quietly widening the change. *Add a `--provider` flag to `run_eval`* —
a real gap, but a bigger interface decision than this fix needed.

**What it opens.** **RG-021 should be fixed in the same pass** — it asks `run_eval` to record
**index composition** so a run over a polluted index is not silently incomparable. Same defect
class (the run record does not pin what the run measured), same file, same one-line-per-key remedy;
with `synthesis_mode` that is three keys and the class is closed. **The five 2026-08-15 trials were
annotated the same day** — their `note` (not `config_json`, which would disguise an inference as a
recording) carries the attributed generator and an explicit "not comparable to the 2026-08-08
private control" warning. The 70 pre-2026-08-15 runs stay unattributable and should remain so; they
were annotatable only via a `.env` read on the same day, and that chain does not exist for them.
**The policy call was taken the same day (user): the private arm stays local-only on
`llama3.1:8b` and the two arms are never compared** — so nothing is re-baselined, the 2026-08-08
local numbers remain the private control, and the Haiku trials are off-policy evidence rather than
a baseline. Written into `evals/README.md`, where the comparison would actually be attempted.
⚠ It is a convention, not a control: `run_eval` has no `--provider` flag and `.env` defaults to
`anthropic`, so the next bare invocation on the private set inherits Haiku again and bills. The
override (`LLM_PROVIDER=ollama LLM_MODEL=llama3.1:8b`) is verified to take; a `--provider` flag is
the durable fix and belongs with the RG-021 pass.
**⚠ Cost:** `--with-llm-judge` was not passed, but `run_eval` *always* generates an answer per case
and has no `--provider` flag, so generation inherits `.env`; those 5 trials billed ~839 K input +
~46 K output tokens for 175 generations.

---
## 2026-08-15 — the shipped v0.5.1 re-tested on a clean box: RG-012 PASS, and RG-028 closes as contention

**What changed.** No code. `.claude/RIGOR_TODO.md`: **RG-028 closed**, RG-012's gate-defect table
gains run 4 and its status line the 2026-08-15 re-pass. `docs/desktop-packaging.md` §5 gains a
log-reading note. Evidence archived to `C:\rg012-host\out-2026-08-15-v0.5.1-run4-PASS-devstack-down\`
(local-only, like the whole harness).

**What was run.** RG-012 Tier-2 against the **published** 0.5.1 installer — the staged copy hashes
`985331FF…`, matching both the bundle and the SHA-256 that was verified against the GitHub asset, so
this tested the bytes users download, not a look-alike rebuild. Windows Sandbox, `python on PATH?
False`: silent install **235 s** → `/api/health` **~40 s** → 3 PDFs → **322 chunks** → one turn,
**16 s**, 10 sources, **5 resolved citations (5 canonical `[n]`, 0 unresolvable)** → **PASS**.
`release_preflight` was green on all nine checks beforehand.

**Why it mattered — RG-028 asked for exactly one number.** The 0.5.1 release runs measured 178/194/78 s
against 14 s (0.4.1) and 17 s (0.4.2) on the same question, corpus, model and box. The entry named
the deciding experiment: re-run **with the dev stack down and Ollama freshly restarted**; near
14–17 s means contention, over 100 s means a real regression in the shipped artifact. It came back
**16 s**. The cause was host contention — my own dev stack sharing the GPU with a 97-document
corpus's embedder and reranker — and the answer path is exonerated without bisecting `v0.4.2..v0.5.1`.
**322 chunks also re-confirms KI-34**: extraction genuinely works in the frozen binary.

**Rejected alternatives.** *Bisect the answer path first* — the entry itself ranked contention as the
likelier cause and the experiment was far cheaper; bisecting would have burned a session to reach the
same place. *Download the GitHub asset to test "what users get"* — the hash already proves the local
staged copy is byte-identical, so a 1.5 GB download would have added a network round-trip, not
evidence. *Close RG-012's flakiness finding on the strength of a green run* — see below.

**What it opens.** **RG-012's citation verdict stays open and that is deliberate.** Run 4 passed, which
moves the failure rate from 1-in-3 to 1-in-4 on a byte-identical healthy artifact — it is one more
sample, not a fix, and the cause (KI-36's bimodal citing on `llama3.1:8b`) is untouched. The green run
is precisely when a flaky blocks-ship gate gets quietly retired, so the entry now says so in terms.
Its two cheap fixes (drive the gate's one turn with a paid model, or assert across N turns) remain the
work. Also opened: **dev-stack-up is now a known confounder for any RG-012 timing** — check 8001/1420
are clear before reading a turn time as signal. The 16 s is **not** promoted to `docs/performance.md`
or RG-010/011: one sandbox run over a 3-document corpus is a gate observation, not a benchmark.

---
## 2026-08-13 — ADR-045 (taxonomy display rule), and the auto-propose run that says the scope is the bug

**What changed.** New `docs/decisions/ADR-045-taxonomy-display-rule-and-document-identity.md` + its
index row; RG-015 in `.claude/RIGOR_TODO.md` gains a third evidence section. No code.

**ADR-045 fills a gap ADR-028 names in its own "Must revisit": *"the display rule is unspecified."***
A document shows its attached nodes **minus any that is an ancestor of another** (so a concept placed
under both `optics` and `neuroscience` shows both — neither subsumes the other); search and filter
match the **full ancestor closure**; attachment happens only at the most specific node and ancestors
are derived, never stored. Both rules degrade to identity on an unplaced vocabulary, which is not a
future edge case — it is the path 344 of 357 concepts take today.

**Why it matters beyond tidiness:** this is the actual fix for the partitioning failure measured
twice on 2026-08-12. `rag` is `df=1` and always will be; `machine learning` is not. Under closure a
97%-singleton vocabulary still filters cleanly, with **no change to extraction and no re-ingest**.
Two further decisions, both taken to keep something out of the vocabulary: a document identity key
(`rag_lewis_2020`) is a **computed field, not a `Concept` row** — it is an identifier, `df=1`
forever, and admitting it would manufacture 97 permanent singletons in the facet being repaired; and
bibliographic type/origin are **metadata columns**, because "is a journal article" is not "belongs to
a research field" and modelling it as `in_field` gives that edge two meanings.

**Then the measurement, because ADR-045 makes placement load-bearing.** Under closure a wrong
placement silently *widens a filter*, so `propose_taxonomy --apply --all-concepts --limit 25` ran on
`qwen3.5:9b` (49 calls, ~2 min, $0) before anything bulk. Judged against each term's own source
document: **~4 right · ~8 coarse · ~12 wrong**, and the wrong ones are not near-misses —
`acdc`→**Music** (the ACDC cardiac-MRI benchmark, from a Mamba-UNet segmentation paper),
`alpha`→**Analytical chemistry** (an EEG band), `actor`→**Performing arts** (actor–critic RL),
`accessory`→**built environment** (from a Cajal neuroanatomy paper).

**The finding is not "the model is bad".** On the **13 curated `graph_include` concepts** a *weaker*
model (llama3.1:8b, 2026-07-25) was **13/13 plausible**. On raw keyword rows a *stronger* model is
half wrong. **The variable is scope, not capability** — `--all-concepts` crosses the ADR-018
boundary on purpose and hands the classifier 344 rows nobody curated, ~40% of which are not concepts
at all (venue artifacts `aclweb`/`aclanthology`, orphans with no document links, and bare fragments
like `alpha`/`actor` that only mean something inside a phrase). So `graph_include` is a
**precondition for auto-propose working**, not bookkeeping.

**Confidence is still not a signal, third independent confirmation:** 0.80–0.95 across the sample,
`acdc`→Music at 0.80 and a correct placement at 0.80.

**A correction worth more than the run.** The taxonomy has **never had a human-accepted concept
placement**: all 37 concept→field links were `origin='proposed'`; the 213 `curated` edges are the
ANZSRC trunk itself (domain→domain). Every earlier statement of the form "13 concepts are placed"
should have said *proposed*. State which, always — they have never meant the same thing here.

**Disposition.** The 24 rows were deleted via `taxonomy.remove_hierarchy_edge` (back to 213 curated
+ 13 older proposals); backup `data/library.db.bak-20260813-pretaxonomy`. The remaining 333 were not
run. ⚠ **`--limit N` takes the alphabetically-first N**, so this was an `a*` sample — enough to
answer "usable in bulk?" (no), not enough to quote as a precision figure.

**Rejected.** Running the remaining 333 (would have written ~330 rows of this quality into a layer
ADR-045 just made load-bearing). Keeping the 24 as a labelled sample (the taxonomy view would show
`acdc → Music`, and a wrong placement that is *visible* is how a feature teaches something false —
the exact failure the 2026-08-12 relabels were about).

**What it opens.** The sequence is now clear and is upstream of the classifier: curate the
vocabulary, clean the D4/D5 residue out of it, then place a small set — by hand or with a stronger
model — and only then trust coverage. RG-015's original debt (per-kind detector precision) is
untouched and still needs the detectors to exist.

---
## 2026-08-12 (4) — keyword quality D4 + D5: the bibliography is where surnames come from, and the tokeniser was renaming genes

**What changed.** `knowledge/keywords.py` gains `strip_reference_section`, `is_citation_artifact`,
a rewritten `_TOKEN_RE`, and a head-check in `candidate_terms`; 15 more unit tests (44 in that file
now). Re-applied to the live 97-document library.

**D5 — the tokeniser was silently renaming things.** `_TOKEN_RE` split on `.` and `/`, so
`16p11.2` became `16p11` (a different locus), `C57BL/6` became `c57bl`, and `gpt-3.5`/`gpt-4.5`
became `gpt-3`/`gpt-4`. The fix keeps a separator **only when a digit follows it** — which is the
whole of what distinguishes a designator from prose. `e.g`, `i.e` and `arxiv.org` still split
exactly as before, because a *letter* follows their separator. The corpus now carries `16p11.2`,
`dlight1.1`, `gpt-3.5` and `gpt-4.5` whole, where before it carried truncations of them.

**D4 — surnames come from the bibliography, so remove the bibliography, not the surnames.**
⚠ The obvious fix here is a name filter and it is the wrong one: `cajal`, `cre`, `dbs`, `16p11`
and `c57bl` are real vocabulary in this corpus and have been mistaken for noise twice. So D4 is
structural — cut from a whole-line `References` heading to the end of the document — plus three
pure *shape* rules (`2014a` year-suffixes, `e04250` article ids, `10.xxxx` DOI prefixes). A
surname that survives that appears in the document's own prose, which is exactly when it is a real
term for that document. `shadmehr` and `wolpert` are still keywords of Shadmehr's own review, and
should be.

**The measurement that saved the fix from looking finished.** The first regex matched only a plain
`References` line and fired on **25 of 97** documents. Sampling the actual cache showed why:
PyMuPDF4LLM's dominant rendering is `## **References**` (32 of the sampled headings), with
`_REFERENCES_` behind it. Allowing markdown emphasis took it to **79 of 97** — a fix that would
otherwise have shipped working on a quarter of the corpus while reporting success.

**Effect on the two probes** (dry run, live corpus): year-suffix citations **2 → 0**;
`rag_lewis_2020.pdf` lost `10 18653 v1` and `aclweb org anthology` (DOI + URL fragments) and gained
`dpr` and `generator`; `transformer_vaswani_2017.pdf` lost `eos` / `my opinion` / `pad br` and
gained `attention function` and `convs2s`.

**A nuance worth recording, because it looks like a regression and is not.** Keywords on ≥2
documents went **48 → 42** and on ≥5 went **2 → 0**. Removing bibliographies removes terms that
were "shared" across documents *because they were citation artifacts* — spurious cross-document
links, not corpus structure. Partitioning is not what D4/D5 were for, and it remains the open
question from DEVLOG 2026-08-12 (3): per-document TF-IDF cannot produce a facet.

**Rejected.** A surname/author stopword list (the trap above). Rejecting any n-gram containing a
bare decimal — it would kill `gpt 3.5`, a real term, to remove `0.0 true`. Cutting at the *last*
References heading rather than the first past the halfway mark (an appendix after the bibliography
is smaller than the risk of matching a heading in the body).

**What it opens.** Residual noise this did not touch and did not create: `br` is an HTML
line-break artifact leaking out of the markdown (`1020 br`, `br 32`, `0.01 br`), and bare decimals
still ride inside bigrams (`0.0 true`). Both are small, both are their own fix, neither is D4/D5.

---
## 2026-08-12 (3) — the four release-readiness relabels: a rank instead of a score, keywords that are about the paper, an experimental label, and a hidden tab

**What changed.** The user's decision on `docs/REVIEW_2026-08-12_release-readiness.md` §2b was
**"fix all four, hide the graph tab"**. Four independent changes, one theme: *a built feature that
teaches a first-time user something false is worse than one that is missing.*

**R1 · Connections shows a rank, never a score.** New `lib/library/connections.ts` (+5 tests);
`DocConnections.svelte` prints `1st / 2nd / 3rd` and a one-line caveat instead of `score.toFixed(2)`.
Measured on the live corpus while verifying: one document's top four neighbours score **0.982,
0.965, 0.962, 0.956** — the old UI rendered that as "0.98 / 0.97 / 0.96 / 0.96", which reads as a
precise claim about how alike two papers are. It is not one: `doc_vectors` mean-pools every chunk,
so same-field papers collapse (750 edges, **median 0.918**, against a 0.5 threshold that therefore
excludes almost nothing). The ordering survives that; the distance does not. The caveat is a
sentence, not a tooltip, because the user who never hovers is exactly the one who would over-read it.

**R2 · Keywords — D1 (page furniture) + D2 (shingle suppression), and the honest half of the result.**
New pure functions `split_pages` / `strip_page_furniture` / `suppress_nested` in
`knowledge/keywords.py` (+14 tests), wired into the shipped `per_doc` path, then **applied to the
live 97-document library** (`extract_keywords --apply --force`, $0, deterministic; `library.db`
backed up first).

| | before | after |
|---|---|---|
| documents with ≥1 keyword | 82 / 97 | **96 / 97** |
| nested/shingle slots | **334 / 1230 (27%)** | **0 / 1440 (0%)** |
| distinct keywords | 1192 | 1375 |
| on ≥2 documents | 30 | 48 |
| on ≥5 documents | 1 | 2 |
| on ≤1 document | 1162 (**97%**) | 1327 (**97%**) |

The eyeball test is decisive. `nihms-66884.pdf` went from **11 of 15 slots** being shingles of the
PMC running header (`author manuscript` · `exp brain res` · `2008 september 26` …) to
`motor commands` · `optimal control` · `proprioceptive` · `state estimation` · `saccades` — the PMC
stamp is *gone*. `transformer_vaswani_2017.pdf` went from **9 of 15** slots on one figure's
`<eos> <pad>` artifact to `multi-head attention` · `self-attention` · `scaled dot-product` ·
`sequence transduction`. `rag_lewis_2020.pdf` went from **zero** keywords to `rag` · `rag-sequence` ·
`rag-token` · `retriever` · `non-parametric memory`.

**But the number the plan said to report did not move, and that is the finding.** Singletons are
**97% before and after**. D1/D2 fixed *what the keywords say*; they did not make the layer
**partition** the corpus — and they cannot, because per-document TF-IDF selects `df≈1` distinctive
terms *by construction* (this module's own docstring says so). **So keywords are a good per-document
descriptor and are still not a facet.** Partitioning would need the `corpus_band` or `contrastive`
mode, which is a different decision, not a bug fix. The Library overlay already tells the truth here
(PR-2.7 F4 collapses the 1-document tail behind *"on 1 document — search still finds them"*), so no
UI change was needed — but the open question survives this session intact.
*(The coverage jump 82 → 96 is **D3**, not D1/D2: it is simply what re-running a stale enrichment
does. One document still yields nothing.)*

**R3 · The epistemics chips say `experimental`, in both places they are met.** The chip carries an
inline `experimental` tag (inside the pill, so a line wrap cannot separate the qualifier from the
claim) and a tooltip that says it is not a measurement and can be wrong; the Settings section gains
a `(experimental)` heading and a plain-words **Known limitation** paragraph. The default was already
`false` (flipped 2026-08-03, KI-33) — what was missing was *saying so*: a silent default tells a user
who turns it on nothing, and a user who leaves it off nothing either. Rebuild is ADR-041.

**R4 · The Graph tab is hidden.** New `lib/core/features.ts` with a single documented
`GRAPH_TAB_ENABLED = false`; the tab is `{#if}`-gated in `Topbar.svelte` and `selectMode` coerces
`graph → chat` defensively, because that function is also the nav-history restore path and landing
in a mode with no exit is the one failure a hidden tab can still cause. **Nothing was deleted** —
`/api/concepts/*` stays mounted, the gap list keeps its triage writes, and `GraphIndex` /
`ConceptGraph` / `GapList` are untouched and still tested. Checked before hiding: the Taxonomy view
is a global overlay opened from the **Library** rail, so it is not orphaned by this.

**Rejected.** Deleting the graph code (throws away working, tested work to solve a *placement*
problem). Keeping the similarity number in a tooltip (if the number is not meaningful, hiding it one
hover away is not honesty). Growing `VENUE_STOPWORDS` for D1 — the signal is position and
repetition, not vocabulary; every publisher has a different stamp and the words in them
(`brain`, `september`) are not junk anywhere else. Applying D2 to `contrastive` (it already
discounts nesting via C-value) or `corpus_band` (its exposure is via furniture, fixed at source).

**What it opens.** The facet question above. Whether corpus growth should re-trigger keyword
enrichment automatically (**D3**, which is KI-44's question in a different costume — answer them
together). And the one document that still extracts zero keywords.

---
## 2026-08-12 (2) — the graph rail's three small fixes, and two checklist rows that were already done

**What changed.** `gaps.ts` gained `filterGapRows` (+5 tests); `GapList.svelte` gained a filter box
and turned its two checkboxes into pressed-state `.lens` buttons; `GraphIndex.svelte` did the same
to `Include under-connected` and lost the now-unused `.toggle` rule. Both `Gaps only` and
`Include under-connected` got tooltips that say what the control *does* rather than restating its
own label. `docs/ui-checklist.md` updated.

**Why.** Three rows the checklist marked "no dependencies", and they were: `Gaps only` was a button
while `Include under-connected` next to it was a checkbox — two different kinds of thing doing one
job — and the Gaps tab had no filter while the concept rail beside it did. The filter matches the
**gap kind** as well as the concept label, because "single" is the list's own vocabulary for a class
of problem; matching labels only would return nothing and read as *"no such gaps"*.

**The other half is a finding, not a change.** The two "review findings" in §3 —
`reviewer_kind="llm_haiku"` hardcoded, and `set_llm_selection` accepting an empty model — are
**both already fixed in code**, with a test pinning the first
(`test_chat_controller.py:1052`). The rows were stale. §3 now says so, because the cost here was
only ten minutes of reading, and the cost of *believing* it would have been a redundant "fix" to
working code.

**Verified live** (dev server, $0): 15 gap rows → 3 on `single`, all Single-source; a no-match query
shows the honest empty line; both rails show two matching lens buttons and zero leftover checkboxes.

---
## 2026-08-12 (1) — the app can now tell you a new version exists, and that is deliberately all it can do (ADR-044)

**What changed.** New `docs/decisions/ADR-044-update-notification-not-delivery.md` + its index row.
New `src/doc_assistant/update_check.py`, `apps/api/{models,routers}/updates.py` (three routes),
`apps/desktop/src/lib/core/{types,api}/updates.ts`, `apps/desktop/src/lib/settings/updates.ts`
(pure display logic) and an **Updates** section in `Settings.svelte`. `app_settings.py` gained
four accessors. **`src/doc_assistant/__init__.py` gained `__version__`** — it was empty, and the
app had no runtime knowledge of its own version at all. Tests: 26 unit + 12 integration + 9
frontend, all passing; `release_preflight` now checks **six** version strings, not five.

**Why.** The app ships as an NSIS installer with no store and no package manager behind it, so
every install is frozen at the version it shipped with — including one carrying a bug we have
already fixed. The user's framing was explicit: do it like calibre — signal, link, and let the
user install — and **an integrated updater is too ambitious for now** because several features
are not stable enough to push at people automatically. That rejection is the load-bearing half:
an in-app updater must replace a running binary, verify a signature and roll back a bad write,
and none of that is earned while the release process has not yet produced a verified artifact for
the tag it cut.

**The design decision that is not implementation detail: three states, never two.** A failed check
reports `unknown`, never `current`. Saying "you are up to date" because the network was down is
the one failure mode that would make this feature worse than not having it, so every network,
parse and decode error becomes `unknown` with a plain-words reason the UI shows verbatim. Verified
against the real endpoint: the repo has **no published releases**, and the app says *"no published
release to compare against"* rather than going quiet or claiming currency.

**Two defects found while building it, both worth recording.**
1. **My first cut cached only the check timestamp, not the observed version** — reasoning that a
   stored verdict would go stale. It does, but the cure was worse: a GET inside the 24 h window
   then *forgot* the update the last check found, so the banner vanished on the next page load.
   Fixed by storing the observed **version** and recomputing the verdict against the running one
   on every read — which self-corrects in both directions (after the user updates, the same cached
   observation reads `current` with no further request). Two regression tests pin it.
2. **mypy caught a real portability bug**: `datetime.UTC` is 3.11+, and this package declares
   `requires-python = ">=3.10"`. Now `timezone.utc`. CI runs the same interpreter as the dev box,
   so nothing else would have caught it.

**Rejected.** In-app download/install (user's call, and it needs a code-signing decision first).
Default-on checking (this app makes no outbound calls the user did not ask for; the toggle is off
by default, and the **manual** "Check now" runs regardless, because gating an explicit press would
leave a user who declined background traffic no way to answer the question at all). A configurable
repository URL — pointing an update banner at an arbitrary host is a way to get someone to install
something they did not choose. Notifying on `unknown` outside Settings — an offline machine would
nag forever about a check it cannot run.

**What it opens.** This project now has a **network code path in `src/`, permanently**, and it
inherits the frozen-build OS-trust obligation (KI-10) for a second caller. It also creates a
release-process coupling that is now real and currently unmet: **the banner is only truthful if
GitHub *release objects* are cut for tags**, and none exist — `v0.5.0` is not even pushed. Cutting
a release is now a step in `docs/RELEASE.md`, not a courtesy. Not built, and deliberately: any
surface outside Settings (a topbar dot, a startup banner) — `shouldNotify()` exists and is tested,
but nothing calls it yet, so today the only place an update appears is the Settings panel.
