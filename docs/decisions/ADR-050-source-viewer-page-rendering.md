<!-- status: active · updated: 2026-09-01 · class: append-only -->

# ADR-050 — The source viewer renders pages on the server, and finds its page in the cache

**Status:** accepted (2026-09-01) · **Supersedes:** nothing · **Amends:** `docs/ROADMAP.md` row 18
(its stated mechanism was wrong — see D2) · **Relates to:** ADR-002 (the FastAPI/SSE boundary),
ADR-046/ADR-3b (`root_available`), ADR-047 (identity), ROADMAP 19 (the extracted-text half) ·
**Roadmap:** 18

## Context

Row 18 has been on the roadmap since 2026-08-25, phrased by the user: a right-split pane on the
Library screen showing *the file itself*, "working only if the file is available". Row 19 shipped
the other half of the same idea — **In context**, a cited passage highlighted inside a window of the
*extracted text* (`ChunkContextView.svelte`). Row 18 is the page image that row 19's DEVLOG entry
explicitly deferred to it.

Three things had to be decided before any of it could be built, and none of them was in the row.

**1. How the file reaches the pane.** The app has never served a document's bytes. The nearest thing
is `POST /api/library/documents/{id}/reveal`, which hands the file to the OS file manager and leaves
the app entirely (`apps/api/routers/library/documents.py:148`).

**2. Which page to open on.** The row asserts its own answer — *"Page-level jump costs no ingest
change — chunks already carry `page`"*. **That is false for the path the app actually retrieves
on**, and the correction is the substance of this ADR.

**3. What a non-PDF document does.** A page image is a PDF concept. The corpus is 98/98 PDF today,
but the ingester accepts EPUB/HTML/DOCX/MD, and the robustness contract forbids designing to the
corpus in front of us.

### What was measured first (2026-09-01, read-only, $0)

- **`page` on the live retrieval path: 615 of 39,705 chunks — 1.5%.** `USE_PARENT_CHILD` defaults
  to `true` (`config.py:203`), so the parent-child store *is* the retrieval path, and every one of
  those 615 is a figure chunk (`chunk_type='figure'`, whose page comes from figure detection). No
  text parent carries a page. The flat baseline store carries `page` on 15,789 of 15,789 — 100% —
  which is where the row's claim came from, and it is not the path in use.
- **The cache carries page markers on every document: 98/98**, `<!-- page:N -->` written by
  `extractors.py:99`, count equal to `Document.page_count` exactly, sequential from 1, no exceptions.
- **Chunks carry `parent_char_start`/`parent_char_end` into that cache at 100%** — row 19's
  re-chunk moved span coverage 63.3% → 100% (DEVLOG 2026-08-30 (7)(8)).
- **A page render costs 19–31 ms and 140–261 KB** (PyMuPDF `get_pixmap`, median over 18 pages of the
  6 longest documents): 19.0 ms / 139 KB at 96 dpi, 31.2 ms / 261 KB at 150 dpi. The corpus is 2,973
  pages, so rendering it all up front would be ~760 MB and ~90 s.

## Decision

### D1 — The server renders a page to PNG; the client asks for one page at a time.

`GET /api/library/documents/{id}/page/{n}` returns `image/png`, rendered on demand with the PyMuPDF
call the figure pass already uses (`ingest/figures.py:394` `render_region` is the same
`get_pixmap`, with a clip). The client is an `<img src>`, exactly like `figureUrl`
(`core/api/chat.ts:80`) — the established precedent for "a binary the backend owns".

This follows from the non-negotiable that `apps/` are thin shells with all logic in
`src/doc_assistant/`, and from ADR-002's FastAPI boundary: a renderer in the frontend, or a Tauri
direct-disk read, would put document handling on the client and would not work in browser dev mode.
It also declines to ship a 20 MB PDF over the wire to show page 7.

**The cost this pays, stated rather than discovered later: a page image has no selectable text, no
in-page find, and nothing for a screen reader.** That is a real loss against a PDF.js viewer and it
is accepted because the *searchable, selectable* surface already exists and is better at that job —
row 19's extracted-text view, which is the same document as text. The two are complementary: the
image is for fidelity and provenance ("this is the page it came from"), the text is for reading and
search. Neither is asked to be the other.

**On-demand, never pre-rendered, nothing cached to disk.** At 19–31 ms a page the latency is
invisible, and pre-rendering trades 760 MB and a 90-second pass for it. This also keeps the viewer
free of a second derived artifact to invalidate — a re-extract or a replaced file changes nothing
it owns.

### D2 — The page is derived at read time from the cache's markers, not read off the chunk.

A chunk's `parent_char_start` is an offset into the cached markdown; the cache carries
`<!-- page:N -->`; the page is the last marker at or before that offset. That is a pure scan, it is
already written (`ingest/chunking.py:64` `extract_chunk_metadata` does exactly this — it is how the
flat store got its 100%), and it is total on this corpus: 98/98 documents, marker count equal to
`page_count`.

**So the row's conclusion survives and its reason does not.** Page-level jump genuinely costs no
ingest change — but because the cache is page-annotated and chunks are offset-annotated, *not*
because chunks carry a page. Had the row been built on its stated premise it would have worked on
figures and on nothing else, and the 1.5% would have looked like a corpus problem rather than a
design one.

Deriving beats backfilling `page` onto the parent-child store: a backfill is a re-chunk of 39,705
chunks to persist something reconstructible in microseconds from data already loaded, and it would
have to be re-run after every extraction change. The derivation is correct by construction
wherever the offsets are, and the offsets are the thing row 19 already made total.

**A document with no markers, or a chunk with no span, has no page.** The viewer opens at page 1 and
says nothing about a position it does not know — the same rule row 19 set when it declined to guess
at an unplaceable chunk (3 in 39,090).

### D3 — PDF renders pages. Every other format degrades to its text, and says so.

Rendering is gated on `format == 'pdf'`, not attempted-and-caught. For EPUB/HTML/DOCX/MD the pane
shows the extracted text — the same content, presented the way that format actually has — with one
line saying why there is no page image. This is not "unsupported": it is the honest rendering of a
document that has no pages. It also means the pane is useful on the whole library the moment a
non-PDF is ingested, rather than being a PDF feature wearing a general name.

### D4 — Availability is the existing signal, and unavailable is a sentence, not a broken pane.

`SourceView.root_available` / `RootView.available` already answer "is this file reachable right now"
(`ingest/registry.py:160`, `apps/api/models/sources.py:56`), which is what the row asked for and
what AD3b built. An unreachable root reports *the drive is not connected*; a file that moved reports
*the file is not where the library expects it*. Both name the path. The route returns 404 with a
reason and the pane renders the reason — it does not render a broken image.

### D5 — Row 18 opens the right page. It does not highlight the passage on it.

Highlighting the passage *on the image* needs page coordinates, and offsets into extracted markdown
are not coordinates. Getting a rect means read-time matching (`page.search_for`), which is
approximate exactly where extraction normalised the text — ligatures, hyphenation, column order —
and its accuracy is **unmeasured**. That measurement is the follow-on's first question, not a thing
to assume inside this row.

Row 18 therefore delivers: open the document beside its entry, land on the page the citation came
from. The passage-level highlight already exists on the text side, where it is exact.

## Consequences

- **The roadmap row's premise is corrected in place** — a row that states a wrong mechanism will be
  believed by the next session that reads it.
- **A new public route serves file-derived bytes.** It is id-addressed, never path-addressed: the
  client cannot ask for an arbitrary file, only for page `n` of a document the registry knows. Page
  numbers are clamped to the document's real page count.
- **`page` stays absent from the parent-child store**, and every consumer that wants one derives it.
  If a second consumer appears, the derivation is one shared helper, not two.
- **A figure chunk already carries a real `page`** and should keep using it — it is detection output,
  not a reconstruction, and it is the one case where the stored value is the better answer.
- **The viewer is read-only and writes nothing.** It is also most of the substrate an annotation
  layer would need (memory: user-annotatable figures/chunks), which is a reason to keep it clean of
  anything speculative.

## Rejected

- **PDF.js in the frontend.** Gives selectable text and in-page find — genuinely better on that
  axis — but puts document parsing in the thin shell, adds a worker and a CSP fight inside Tauri,
  ships whole files to show one page, and duplicates a search surface the app already has in the
  extracted text. Revisit if in-page find on the *image* becomes the thing users ask for.
- **Tauri asset protocol / direct disk read.** Fewer moving parts in the packaged app, but it
  bypasses the ADR-002 boundary, does not work in browser dev mode, and would mean two code paths
  for one feature.
- **Backfilling `page` onto the parent-child store.** A 39,705-chunk re-chunk to persist something
  derivable for free, invalidated by the next extraction change. See D2.
- **Pre-rendering page images at ingest.** ~760 MB and ~90 s for the current corpus, plus a derived
  artifact to invalidate, to save 19–31 ms.
- **Rendering non-PDFs by converting them to PDF first.** Invents pages a document never had, and
  the conversion is a second extraction with its own fidelity story. D3 shows what the format
  actually is instead.

---

## Addendum — 2026-09-01: D5 measured. The highlight is viable, and it is not the small job the row implies.

D5 scoped the on-image highlight out and named its accuracy *"unmeasured"*, calling that the
follow-on's first question. It has now been measured — read-only, $0, on the live corpus — and the
answer changes what the follow-on should be, so it is recorded here rather than left for a session
that would have to redo it.

**Method.** 150-200 sampled parent chunks per run; the page derived exactly as the shipped viewer
derives it (D2); the passage cleaned of the cache's markdown, split into sentences of >= 8 words,
and each sentence's opening *n* words handed to `pymupdf.Page.search_for`. Table-shaped passages
(>= 3 pipe-rows) were excluded throughout — a spliced table is not page text and never will be.

### Recall is a function of anchor length, and the relationship is inverted from the obvious guess

| Anchor | Sentences placed | Exactly one rect | Wrapped across lines | **Genuinely scattered** |
|---|---:|---:|---:|---:|
| 3 words | 91% | 68% | 16% | **7%** |
| 4 words | 90% | 64% | 20% | **5%** |
| 6 words | 84% | 50% | 29% | **5%** |
| 12 words | 69% | — | — | — |

A **longer** anchor finds **less**: it crosses line breaks, where hyphenation and the extractor's
reflow stop matching. 730 single-page prose sentences.

**Two measurement traps sat in the way of that table, and both produced a wrong answer first.**
A needle carrying the cache's list markers and table pipes scored 68% where a cleaned one scores
94% — the first run measured the probe, not the technique. Then "returned more than one rect" was
read as ambiguity, which made longer anchors look *less* precise; `search_for` emits one rect **per
line a match spans**, so a wrapped phrase looks identical to a repeated one until you separate them
by geometry. The rects of a wrapped phrase are horizontally **disjoint** (tail of one line, head of
the next), so the natural test — do they overlap in x? — classifies every wrapped match as
ambiguous. Only a vertical test works.

### The design the numbers point to is an envelope, not per-sentence rects

Highlighting each sentence separately leaves ~10% of them unlit and scattered through the passage,
and a reader has no way to read a gap as anything but *"this part was not the evidence"*. A parent
chunk is contiguous text, so its footprint is a contiguous block of lines: take the first and last
**unambiguously** placed anchors and highlight the band between them.

| | Median | Mean | >= 90% |
|---|---:|---:|---:|
| **Purity** — highlighted words that really are the passage | **97%** | 95% | 88% of passages |
| **Coverage** — passage words falling inside the envelope | 45% | 47% | — |

**Purity is the honesty-critical number and it is high**: the band covers the passage and almost
never its neighbours. **Coverage is a floor, not a verdict** — the probe grouped anchors into
columns by `int(x0 // 60)`, which splits one indented paragraph across two bands and drops the
lines between. A column-aware implementation is the difference between that 45% and the real
figure, and it was not built.

### What this means for the follow-on

**Viable, and worth doing — but it is an increment, not a detail.** Three things have to be solved
that the row does not mention: real column detection (the probe's proxy is not good enough to
ship), **page-straddling passages — 43% of parents cross a page break**, so the opening page can
only ever show part and the pane has to say so, and an explicit policy for the 5% of anchors that
are genuinely ambiguous (decline, never guess).

**Unchanged from D5: nothing is highlighted on evidence this thin without saying what it is.** A
band that covers 45-97% of a passage is an *aid to finding it*, not a claim about the passage's
extent, and the UI must not imply otherwise.
