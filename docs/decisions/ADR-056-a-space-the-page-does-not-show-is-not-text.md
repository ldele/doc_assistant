<!-- status: active · updated: 2026-11-01 (landed: read again as new, four faults fixed, gate 2 run again; 2026-10-24: points 7, 9, 10 accepted and built with the control, point 8 agreed with session 5c (built after ADR-055); each run's span in the record; evening amendment: the control's sketch, points 7-10 to decide; points 1-4 and 6 accepted and built, gate 2: 102 identical, 2 changed by spaces only; amendment: the review's 33 verdicts; context rules for scans, layout boxes, maths and URLs proposed; 2026-10-23: accepted by the user on two conditions: testing before the build, an ignore list; Amendment 2026-10-23) · class: append-only -->

# ADR-056 — A space the page does not show is not text: letter-spacing is closed at extraction, from the PDF's own glyphs

- **Status:** accepted on two conditions (user, 2026-10-23: "Okay I accept but we need a lot of
  testings and maybe even some kind of "ignore list". We cannot build this blindly."). The
  conditions are the *Amendment 2026-10-23* at the end: a test plan whose last gate is your review
  of every join before the library is touched, and an ignore list whose shape is still yours to
  choose. Built since, and landed 2026-11-01 (the amendments below, the last one first); the
  library itself is not re-extracted yet. Proposed by Claude Code on 2026-10-21; asked for by the user on 2026-10-21
  after labelling Cajal's keywords ("there seemed to be an issue with the extraction - meaning
  that we should correct something in the pipeline"). Draws the line
  [ADR-043](ADR-043-received-content-is-preserved-verbatim.md) left open for extraction; builds on
  [ADR-047](ADR-047-document-identity-survives-re-extraction.md) (a re-extraction keeps the
  document) and [ADR-039](ADR-039-ocr-sidecar-for-scanned-pdfs.md)'s one-extraction-path rule;
  follows KI-64's precedent (lost ligature letters, fixed in the extractor, library re-extracted).
- **Date:** 2026-10-21
- **Deciders:** user + Claude Code
- **In one sentence:** PyMuPDF writes a space wherever two glyphs are drawn far apart, so a title
  set in wide letter-spacing comes out as `N E U R O N S`; this proposes closing those spaces at
  extraction, using only what the PDF itself records (the glyph boxes, and PyMuPDF's own mark on
  every space it added), and leaving every space the page shows, which makes it a correction of how
  the page is read, not a normalisation of what it says.

**To decide** (each with a recommendation; the measurement is
[`letter_spacing_2026-10-21.md`](../../tests/eval/baselines/letter_spacing_2026-10-21.md)):

1. **Where the rejoin lives.** *Recommended:* in `extract_pdf_pymupdf`, one pass per page beside
   `_recover_lost_page`, so the cache, the chunks, search, keywords and quotes all read the same
   text. The alternatives are a derived reading layer and the tokeniser (Options).
2. **Which spaces close.** *Recommended:* (a) the spaces PyMuPDF added between letters drawn with
   letter-spacing, and (b) space glyphs between letters that touch. **Not** (c) a space glyph the
   reader can see, even between single letters (`a r t i c l e  i n f o`), and **not** (d) the
   in-word gaps of an OCR layer (`In tro d u c tio n`), which is another defect.
3. **What record is kept.** *Recommended:* the count of runs closed per document and kind, written
   to that document's `ingestion_events.notes` and the ingest log, plus the baseline. No sidecar of
   every join: the rule only deletes spaces, and the source viewer shows the page itself
   (ADR-050). *Alternative:* a `.md.joins.json` beside each cache listing page, spaced form, joined
   form and kind, 33 lines for this library.
4. **When it lands.** *Recommended:* after session 5c records its "after" round. Cajal is one of the
   15 documents 5c's labels measure, and a re-extraction inside that before/after would mix two
   causes in one number.
5. **What it leaves to their own rows.** *Recommended:* three, none in this ADR. The ASCII-only
   tokeniser (an eval-gated experiment, 90 of 104 documents); the page-furniture filter's half-page
   floor against running heads that alternate with page numbers (a keyword-layer fix); and the OCR
   book's in-word gaps (a known issue).

## In practice — five passages from the library, as extracted today

Measured 2026-10-21, read-only, on the 104 PDFs. "Rejoined" is the projected cache: the rule run
on each page's glyphs, applied to that page's cached markdown, never written to `data/`.

1. **Cajal's title line — closed (letter-spacing).** The PDF draws it letter by letter, each pushed
   0.36 em from the last (`[(A)-364(N)-364(T)-364(I)-364(A)-364(G)-364(O)-364( )250]TJ`), and
   holds no space between the letters. PyMuPDF adds one after each letter, and the cache reads:

   | | Text | What search indexes |
   |---|---|---|
   | cached | `S A N T I A G O R A M Ó N Y C A J A L` | `s` `a` `n` `t` `i` `a` `g` `o` `r` `a` `m` `n` `y` `c` `a` `j` `a` `l` |
   | rejoined | `SANTIAGO RAMÓN Y CAJAL` | `santiago` `ram` `n` `y` `cajal` |

   The word breaks come back too: in the title they are 0.76–1.10 em, more than twice the letter
   spacing, so they stay. `ram` `n` is the tokeniser's ASCII-only split, untouched here (point 5).
2. **Cajal's running head — closed, on 16 pages.** `N E U R O N S :  S T R U C T U R E  A N D  C O N N E X I O N S`
   becomes `NEURONS: STRUCTURE AND CONNEXIONS`. Here the word breaks are spaces the PDF does
   hold; only PyMuPDF's added spaces close. Cajal's letter-spaced runs go from 33 to 0, by deleting
   479 spaces and nothing else.
3. **A URL in `s41598-025-33686-0` — closed (invisible spaces).** The PDF draws a space glyph after
   every character and gives it no width, so the letters touch (gap −0.07 to 0.01 em):
   `accessed at  h t t p s : / / w w w . c r e a t i s . i n` becomes
   `accessed at  https://www.creatis.in sa-lyon.fr/Challenge/acdc/`. The space left before `sa` is
   where the URL wraps onto the next PDF line; the rule reads one line at a time. 25 runs to 0.
4. **`a r t i c l e  i n f o` in `03-Zuo2014` — left as it is.** The text layer holds a space glyph
   between each letter, and the page shows a gap of 0.44 em: the reader sees spaced letters. This
   is what the document says, so ADR-043 keeps it. Session 5c's keyword filter already keeps it out
   of the keywords.
5. **`In tro d u c tio n` in `Computational_neuroanatomy` — left as it is.** An OCR engine wrote
   this book's text layer, and its glyph positions are estimates: gaps inside a word are as wide as
   the gaps the rule would close. The rule closes a run only if the run's neighbours sit further
   away than its own spacing, and `tro` sits as close to `d u c` as `d` does to `u`, so nothing
   closes. Rejoining it would need a dictionary or a new OCR pass. 39 of its 42 runs stay.

**And what it does not fix.** Cajal's keywords under 5c's extractor are `neurofibrillar`, `s ram`,
`connexions`, `fibres`, `gehuchten`, `ram n`, `corpuscles`, `ramifications`, `reduced silver`,
`perroncito`, `plexiform`, `bethe`, `lenhoss`, `lugaro`, `retzius`. 5c's filter already removed the
13 letter-spaced fragments. With the rejoin, `neurons structure` (the running head, read across its
colon) comes in and `retzius` goes out; `s ram`, `ram n` and `lenhoss` stay, because they come from
the tokeniser. The rejoin is for the text, the quotes and search, not for this list.

## Context

ADR-043 rules that content a document brings is never silently rewritten: normalisation is a
derived, inspectable layer, and a judgement call becomes an option. It named de-hyphenation as the
first candidate. Letter-spacing looks like the same kind of problem and is not, and the difference
is checkable in the file.

A hyphen at a line end is **on the page**: a reader sees it, and deciding that `neu-` `rons` is one
word is an interpretation of something the document shows. A space between Cajal's letters is
**not in the file**. PyMuPDF infers word breaks from gaps, marks each space it infers
(`char["synthetic"]`), and a letter-spaced line has a gap after every letter. With the inferred
spaces turned off, the running head reads `NEURONS: STRUCTURE AND CONNEXIONS`. The cache is this
project's *reading* of the page; reading a space where the page has none is the extractor's error,
exactly as KI-64's lost ligature letters were. Keeping it in a "received" layer would preserve our
own mistake as if it were the source.

The measurement says how much of this there is (15 documents with a run of four or more single
letters, 133 runs):

| What the PDF holds | Runs | Documents | Damage |
|---|---:|---:|---|
| letters drawn apart, no space between them | 33 | 1 | yes |
| space glyphs between touching letters | 25 | 1 | yes |
| space glyphs with a visible gap | 1 | 1 | no |
| an OCR layer's estimated positions | 42 | 1 | another defect |
| separate letters: panel labels, variables, axis ticks | 32 | 11 | no |

So the damage at 104 documents is small: 2 documents, 58 runs. The reason to fix it in the
pipeline rather than leave it is the robustness contract. Letter-spaced titles, author lines and
running heads are an ordinary typographic style in books and older journals. A publisher's
pipeline that draws URLs with invisible spaces is a house practice, so it is likely in every paper
from it; this library holds one, and that is unmeasured beyond it.

## Options

1. **Close the spaces at extraction, from the PDF's glyphs** (recommended). Per page, read the
   text lines (`rawdict`, the layout path's flags); close a run's gaps only under the rule in the
   baseline: three or more one-letter tokens separated by spaces PyMuPDF added, all letters, one
   baseline, and neighbours further away than the run's spacing plus half a quarter-em word space;
   or, in any run of one-glyph tokens, a space between glyphs closer than 1/20 em. Then replace each
   run's spaced form in that page's markdown. Measured: 48 runs closed in 5 documents, 3 caches
   change, every change a deleted space; no maths, initials or panel labels touched. Trade-off: a
   change to `extract_pdf_pymupdf` re-extracts every PDF once, and a rule with structural
   constants to maintain, guarded by tests on real pages.
2. **Turn PyMuPDF's added spaces off** (`TEXT_INHIBIT_SPACES`). One flag. Rejected on measurement:
   814,286 of the library's 1,432,158 spaces (56.9%) are PyMuPDF's, and in 68 of 104 documents
   more than 95% are. Those PDFs carry no space glyphs, so every word in them would run together.
   It also leaves Cajal's title as `S ANTIAGO R AMÓN Y  C AJAL`: the PDF holds a stray space glyph
   after each large capital.
3. **Join single letters in the markdown** (a regex over the cache). No PDF needed. Rejected: the
   text alone cannot tell letter-spacing from the 32 runs of separate letters in 11 documents
   (`A B C D` panel labels, `x y z`), from `a r t i c l e`, or from an OCR fragment; and where the
   cache has lost the word breaks (Cajal's title has one space everywhere), it would write
   `SANTIAGORAMÓNYCAJAL`.
4. **A derived reading layer:** the cache keeps the spaced text, and search, keywords and presence
   read a rejoined copy. This is ADR-043's pattern for normalisation. Rejected here: it treats our
   own reading error as received content; every reader must pick a layer; quotes and the evidence
   handed to the model still show `N E U R O N S`; and a second text store is the cost ADR-043
   flagged as not obviously affordable at 10,000 documents.
5. **Leave it to the keyword filter.** Session 5c already removes letter-spaced fragments from
   keywords, so the symptom the user saw is gone. Trade-off: the text stays wrong in 2 documents,
   for search, quotes and the model's evidence, and a new library with more letter-spaced books
   carries it everywhere but keywords. A fair choice at this library's size if a re-extraction is
   not wanted now.

## Decision (proposed)

**Option 1, scoped by point 2 of To decide.** The deciding reason is the test ADR-043 asks first:
*did we author it?* The PDF is what was received, and for classes (a) and (b) it holds no space the
reader can see. The spaced text was authored here, by the extractor, so correcting it is fixing a
reading, as KI-64 was. Where the PDF does hold a visible space (class c), the same test keeps it,
however odd it looks.

**The rule reads only evidence in the file.** It uses glyph boxes, baselines and PyMuPDF's mark on
the spaces it added, and three constants, each structural: a word space is about a quarter em in
text faces (half of it separates tracking alone from tracking plus a space), glyphs closer than a
twentieth of an em touch, and a sub- or superscript sits well over a tenth of an em off its line.
None is fitted to this library; each is a property of type.

**Guard tests** come from real pages, as KI-64's did: Cajal's title and running head close as
above; `s41598`'s URL closes; `a r t i c l e  i n f o`, an OCR fragment, a maths line
(`d = m + n + 1`) and a panel-label row stay; the pass on a page with no letter-spaced text is the
identity (the 99 unchanged caches must stay byte-identical); and the only characters the pass ever
removes are spaces.

## Consequences

**Easier.** Cajal's quotes and search read `NEURONS: STRUCTURE AND CONNEXIONS`, the same as the page
the source viewer shows. The URLs in `s41598` are findable and copyable. And ADR-043 gets a stated
line for extraction: a reading error is fixed in the extractor; what the document shows is
preserved, and normalising it is a derived layer.

**Harder.** The build re-extracts every PDF once: about 24 minutes here (97 documents,
`docs/performance.md`), plus about 1 s per document for the pass itself (106 s over 104 PDFs in
the probe). Three documents change hash and keep their identity (ADR-047); 1,379 chunks are
re-embedded; the keyword index, the keywords and the graph are rebuilt, as on 2026-10-14. Before:
every store backed up. After: the private eval set (35 cases, `llama3.1:8b`, `--repeat`, as the
harness requires) against its control. Three of its cases cite a changed document
(`cajal_neuron_doctrine`, `cajal_method`, `friston_computational_neuroanatomy`). No locked setting
moves. The graph's 13 concepts contain none of the rejoined words, so the graph is predicted not to
move; that is read back, not assumed.

**Must revisit.** *Any* extractor fix re-extracts the whole corpus, because the fingerprint is per
format: about 41 hours at the 10,000-document contract, for a change that here touches 3
documents. A pass that needs only the PDF and the cached page could upgrade a cache in place (about
1 s per document, under 3 hours at 10,000) if the fingerprint recorded it as its own step. That is
a fingerprint design question for every future extraction fix, not for this one. Also: the
rejoin brings Cajal's running head into the keywords as `neurons structure`, because the furniture
filter needs a line on half the pages and an alternating head sits on 17 of 35.

## Confidence

- ✓ **The cause is read in the file, not inferred.** Cajal's content stream draws the letters with
  offsets and no space; with PyMuPDF's added spaces off, the running head reads as words; the
  added spaces carry PyMuPDF's own mark. `stream.py`, `inhibit.py`, `author.py` in
  `.claude/probes/2026-10-21-letter-spacing/`.
- ✓ **Every run in the cache is classified** against its page's glyphs: 133 runs, 15 documents
  (`classify.py`).
- ✓ **The rule's whole effect on this library is enumerated:** 48 runs in 5 documents, listed in the
  baseline; 3 caches change, only by deleted spaces, checked by comparing each text with its
  spaces removed (`project.py`); 99 byte-identical.
- ⚠ **Projected, not re-extracted.** The rule was applied to the cached markdown page by page, which
  equals the extractor's output only because every cache is fresh against today's extractor
  (fingerprint `dbfcf8f76d819638`, 104 of 104). The build must run the real extraction on these
  PDFs and compare.
- ⚠ **Precision is measured on one library.** The rule's errors in its earlier drafts (maths,
  initials, a superscript) set its conditions, so it has been fitted to the errors this library
  could show. A library with letter-spaced maths, or tracked lower-case text in an OCR layer, could
  show others. The guard tests and the next library's re-extraction diff are where that shows.
- ⚠ **Retrieval effect unmeasured.** No eval run yet; three private cases touch changed documents.

## Amendment 2026-10-23 — accepted on two conditions: test it hard, and give it an ignore list

The user accepted the decision with two conditions: "we need a lot of testings and maybe even some
kind of "ignore list". We cannot build this blindly." This section turns them into gates, and
leaves the ignore list's shape as a decision.

### What the first tests already found (2026-10-23)

Synthetic pages, each drawn with one known mechanism (base-14 Helvetica, 10 pt; `synthetic.py` in
`.claude/probes/2026-10-21-letter-spacing/`), then the corpus sweep re-run:

| Case | Before the refinement | After |
|---|---|---|
| tracked title, 0.3 em | closes ✓ | closes ✓ |
| a spaced word inside a sentence (emphasis by spacing), the PDF holding the sentence's word spaces | **missed**: its spacing equals the word spacing, so it looked like a fragment | closes: a space the PDF holds counts as a word break when the run's own gaps are PyMuPDF's |
| the same word in a PDF with no space glyphs at all | missed | **still missed**: geometry cannot tell it from one-letter words |
| OCR-like fragments (`In tro d u c tio n`) | stays ✓ | stays ✓ |
| variables at word spacing (`for x y z we`) | stays ✓ | stays ✓ |
| a row of single-letter labels on one line, 0.4–0.7 em apart | **closes to `ABCD`** | **still closes** |
| the same row, 0.8 em apart or more | PyMuPDF splits it into separate lines; stays ✓ | stays ✓ |

The refined rule over the 104 PDFs closes the same 48 runs as before; one OCR-noise join in
`Computational_neuroanatomy` swaps for another. Cajal, `s41598`, `pnas` and `elife` are identical.

**The label row is the case that needs the ignore list.** Real tracking here runs 0.21–0.39 em
(Cajal), and PyMuPDF keeps letters on one line up to about 0.7 em. Between the two, a row of
labels and a letter-spaced word look the same in the file, and no property of type separates
them. A threshold placed just above Cajal's 0.39 would be fitted to this library, which the
robustness contract rules out. None of the 104 documents has such a row; the next library may.

### The test plan: gates, in order

Nothing is written to `data/` before gate 5.

1. **Unit tests on PDFs built inside the test** (no private files, so CI runs them). One case per
   row of the table above, plus: maths with operators, a sub- and superscript, vertical text, a
   page with nothing to close (output identical to input), and one property over every case: the
   pass removes spaces and nothing else. The label row at 0.6 em is pinned as *closes*, with the
   ignore list shown to stop it, so the gray zone stays visible in the suite rather than hidden.
2. **The real extractor, not a projection.** The built extractor runs on all 104 PDFs into a copy
   of the data home (an ASCII path, never the working library), and the result is diffed against
   the live cache. Expected: 3 caches change, by spaces only; 101 byte-identical. Any other
   difference stops the build.
3. **A library test, local-only** (skipped where the PDFs are absent, like the private eval set):
   the list of joins on the user's library equals the reviewed list from gate 5, so a later
   change to the rule or to PyMuPDF that moves a join fails a test instead of passing silently.
4. **More PDFs than this library has.** One letter-spaced document is a thin test. Candidates:
   other early Nobel lectures (the same typesetting as Cajal's), older journal scans, a book with
   tracked headings, and the public eval corpus as a no-change control. Downloading them needs
   the user's yes, file by file, or the user adds their own.
5. **Your review of every join** before the library is touched: document, page, spaced form,
   joined form, kind. Every join you reject goes into the ignore list. Then, and only then: back
   up every store, re-extract, rebuild the keyword index, keywords and graph.
6. **After:** keywords and the graph read back against their predictions, and the private eval
   set (35 cases, `llama3.1:8b`, `--repeat 5`) against its control.

### The ignore list — decided: both, in one list (user, 2026-10-23)

The user chose option 3 the same day: "Ignore list: both, in one list. Per-join entries plus a
whole-document entry, keyed by the PDF's content hash, kept as a file in the data home, filled from
the review." The options as they were put:

Its job: stop a join the rule gets wrong, for one document, without turning the rule off for
everyone. The rejoin stays on by default (ADR-043: defaults are opinionated; the option exists
for the reader who can see the default was wrong for their document).

1. **Per join** — entries such as `A B C D` for one document; the extractor leaves those runs
   spaced. Precise; filled by the review at gate 5, and at any later time from the joins logged
   for a new document.
2. **Per document** — one switch: "do not rejoin this document". Coarse, one entry; for a
   document whose layout fools the rule throughout.
3. **Both, in one list** (recommended) — per-join entries, and `*` for the whole document. Keyed
   by the PDF's content hash, so an entry follows the file through a move or a rename and
   survives a re-extraction (ADR-042: a document's identity is its source bytes). Kept as a file
   in the data home, read by the extraction cache layer; a document's own entries go into its
   cache fingerprint, so changing them re-extracts that one document, not the library. No app
   control in this build: the review page exports the list. A control comes later if the list
   gets used.

### Built 2026-10-23: gates 1 and 2 passed; nothing in the library changed

Built in `extractors.py` (the rule and the per-page pass), `ingest/cache.py` (the ignore list and
the per-document fingerprint), `config.REJOIN_IGNORE_PATH`, and the read-only
`scripts/list_letter_spacing.py`, which writes the review list. Account: DEVLOG 2026-10-23.

- **Gate 1:** 38 new tests on PDFs drawn in the test (the label row pinned as closing, with the
  ignore list shown to stop it; the known miss pinned as staying); the full suite, 2,795 passed.
- **Gate 2:** the built extractor on all 104 PDFs into a scratch folder. 101 caches byte-identical
  with the live ones, 3 changed by deleted spaces only, exactly the three predicted. The review list
  holds 48 joins in 5 documents.
- **Two departures from the text above.** Point 3's count in `ingestion_events.notes` is not
  built: the store writes the event from the text alone, so the ingest log carries the count. And
  the ignore entries go through `extract_to_markdown`, so every format's fingerprint moves once:
  the next ingest re-extracts every document, not only the PDFs.
- **Still to come:** gate 3 needs the reviewed list, gate 4 needs the user's yes to each download,
  gate 5 is the user's review, gate 6 follows the re-extraction, which waits for 5c's "after" round.

## Amendment 2026-10-24 — the review, and context in place of hand labels

### Gate 5: the user's review

All 48 joins, grouped into 33 places (a run repeated on 16 pages is one place), on the review page
("Letter-spacing review", private; its store kept every answer). The user added a third choice
after the first two places: **Not text**, for runs that are no text at all. Verdicts: **6 Close
it** (Cajal's title and running head, the PNAS margin label, `The` and `WOl` in the OCR'd book),
**27 Not text** (the 16 URL runs, the eLife maths comma, 10 runs of OCR noise), **0 Keep spaced**.
The user's reason: "url links and math are not texts". Then: "Close the 16 URL runs in s41598 (keep
them on the not-text list), leave the math and OCR noise as extracted". The record keeps the
verdict and what extraction does beside it (`review_verdicts_2026-10-24.json` in the probe
folder).

### The user's direction (2026-10-24)

"We can have an ignore made list to save time on re-scans. But it also means that we need to let
the user delete everything in this list. We can properly detect both urls and math on context. But
apparently the page-scan + some tables + figures might cause issue depending on the instance."

### What the PDF already says about the reviewed runs

Each reviewed run at its first page, against signals in the file (`context_signals.py`):

- **Scanned pages.** All 12 runs in the OCR'd book sit on pages that are one image under the whole
  page. Over the library the split is clean: 606 pages are at least 0.9 covered by one image, every
  other page at most 0.7, and **none between 0.7 and 0.9** (`scan_pages.py`). The 606 are five
  whole documents: `hebb_1949`, `Computational_neuroanatomy`, `hodgkin_huxley_1952`,
  `hubel_wiesel_1959`, `middleton-2001`. "The page is an image" is a property of the file, not a
  tuned number.
- **Layout boxes** (the analysis pymupdf4llm itself runs). Two runs sit in formula boxes; the OCR'd
  book's figure noise is labelled "text", so the boxes alone would not have caught it. Cajal's
  runs sit in page-header boxes: skipping headers would undo the fix.
- **URLs.** 15 of the 16 URL runs close into a URL shape. The 16th, `n = D o w n l o a d s . S N`,
  is a URL's query string, directly after `https://www.itksnap.org/pmwiki/pmwiki.php?` on its line.
- **Maths.** The eLife run holds operator glyphs (`=`, `·`) and no URL.

Rules built from these four signals reproduce **31 of the user's 33 decisions**. The two they
miss are `The` and `WOl` on the scanned book, which they leave spaced: the cautious side.

### To decide (each with a recommendation)

1. **The rule skips scanned pages.** A page whose largest image covers at least 0.9 of it is an OCR
   layer over a picture, so its glyph positions are estimates (the "another defect" of class D).
   *Recommended.* Effect here: the OCR'd book's 12 runs stay as extracted.
2. **The rule skips picture, table and formula boxes.** *Recommended*, as the precaution the user
   asked for: none of the reviewed runs sat in a picture or a table, but a library with
   letter-spaced labels in its figures would.
3. **Maths stays as extracted.** A run with an operator glyph (`=`, `+`, `−`, `×`, `·`, `<`, `>`,
   `≤`, `≥`, and the like) and no URL around it is left alone and listed as not text (maths).
   *Recommended.*
4. **URLs close.** A run that closes into a URL, or continues one on its line, closes and is listed
   as not text (URL), so the later fix that keeps URLs out of keywords and search can find them.
   *Recommended.*
5. **The ignore list stays, and the user can empty it.** It remains the user's override, kept
   across re-scans so the same question is never asked twice. The user can delete any entry, or
   every entry at once, from the app; deleting re-extracts only the documents it named, through
   their fingerprints. *Recommended.* Where the control lives (the re-ingest dialog, the Library's
   document view, Settings) is a screen change, and is sketched before it is built.
6. **This library's list starts empty.** With 1–4, none of the 11 entries the review produced are
   needed: the OCR noise is on a scanned page and the eLife run is maths. *Recommended*; the review
   record stays as gate 3's reference, which pins the joins: Cajal 17 and `s41598` 16 closed, the
   OCR'd book and eLife untouched, PNAS 2 (never reaching the text).

### Decided and built 2026-10-24: points 1–4 and 6

The user: "Accept points 1-4 and 6: build the context rules (scanned pages, layout boxes, maths,
URLs) with tests and rerun gate 2. Then sketch where the ignore-list delete control goes before
building it." Point 5's control is sketched (`docs/wireframes/letter-spacing-ignore-list.excalidraw`)
and waits for the user's markup; it is not built.

- **Built in `extractors.py`.** `_SCANNED_PAGE_MIN = 0.8`, not the 0.9 point 1 names: it is the
  middle of the measured gap, as `_TEXT_LAYER_KEPT_MIN` sits in its own, and it selects the same
  606 pages. `_NOT_TEXT_BOXES` (formula, picture, table) are read from the layout analysis only on
  a page with a candidate join. A run of touching glyphs holding a maths glyph and no URL is kind
  `maths` and stays as extracted; one in a URL is kind `url` and closes.
- **Tests:** 12 more, on PDFs drawn in the test (a scanned page and a page with a large figure,
  each layout box, maths, a URL's query string); the full suite, 2,807 passed.
- **Gate 2 again**, the built extractor on all 104 PDFs: **102 byte-identical, 2 changed by
  deleted spaces only**, `cajal-lecture` (479 spaces, letter-spaced runs 33 → 0) and
  `s41598-025-33686-0` (306, 25 → 0). The OCR'd book is identical now. The review list holds 36
  joins in 4 documents, one of them maths left alone.
- **`data/rejoin_ignore.json` is not written** (point 6). The reviewed verdicts stay as gate 3's
  reference.

## Amendment 2026-10-24 (evening) — the control, a record of closed runs, and keywords

### The sketch, and what the user asked of it

The first sketch of point 5's control was accepted as drawn ("Okay as drawn"), with a question:
"should we add more options than remove ?" Four were proposed: show the runs extraction closed,
each with Keep spaced; Show on page; Re-run text now after a removal; Undo in place of the
confirmation on Remove all. The user: "Yes and maybe edit ?", read as **Edit spacing**: on any run,
choose where the words break, the letters fixed. Then two more questions: "how do we link with
keyword selection ? -> I would guess that the keywords are not always interesting", and "do you
think we could have a better way to show Spaced vs Closed ?"

The sketch now has three screens: the document page, Settings > Documents, and the Edit spacing
dialog (`docs/wireframes/letter-spacing-ignore-list.excalidraw`; its reading, at canvas revision
527, is `docs/wireframes/letter-spacing-ignore-list.md`). The options on it are the user's,
decided with the sketch: Show on page, Edit spacing, Keep spaced or Remove per run; Keep every run
spaced, Re-run text now and All entries in Settings per document; Show runs, Remove, and Remove all
with Undo in Settings.

### What closed runs do to keywords (measured)

Session 5c's keyword extractor, dry run over the whole library, on the text the built extractor
writes:

| Keywords computed from | Documents whose list changes |
|---|---|
| the new text, as built | 1: Cajal gains `neurons structure` (from the running head, read across its colon) and loses `retzius` |
| the new text, the 18 header lines left out | none |

The URLs closing in `s41598-025-33686-0` move none of its keywords. Both of Cajal's runs sit in
"page-header" boxes of the layout analysis. So the user's guess holds here: the one keyword the
rejoin adds is page furniture.

### To decide (each with a recommendation)

7. **A record of closed runs, per document.** Extraction writes it beside the cache in the same
   step as the text: for each run, its page, the run as extracted, its closed form, its kind, its
   box on the page, and the layout label of the box it sits in. A cache entry without its record
   reads as stale, so the two never disagree. *Recommended*: the closed list, the crops and the
   keyword link all read it, and it is ADR-043's "what was done to this passage" in one place.
   Here it is 17 runs for Cajal, 16 for `s41598-025-33686-0`, and nothing for the other 102
   documents.
8. **Keyword selection skips header, footer and not-text runs.** A closed run that the record
   places in a page-header or page-footer box, or whose kind is `url` or `maths`, does not feed
   keyword candidates. *Recommended*: measured above, the rejoin then moves no keyword list here.
   It lives in `knowledge/keywords.py`, session 5c's module, so it lands after 5c's "after" round
   and in agreement with that session. Ordinary running heads stay the furniture filter's job; it
   misses Cajal's (17 of 35 pages, against a floor of 17.5), which is 5c's to weigh.
9. **Each run shows as a crop of its page beside its closed text.** The closed text is the line,
   with a tick where each space was removed; hovering shows the run as extracted. The crop comes
   from the page renderer the source viewer already uses (ADR-050), clipped to the run's box from
   the record; Show on page opens the whole page. The Edit spacing dialog is the editable form of
   the same line: click a tick to break there, a gap to join. *Recommended.*
10. **An ignore-list entry can hold the user's spacing.** Besides `keep` (a run left as
    extracted), an entry can map a run to the spacing the user chose. The extractor applies it
    only when it differs from the run in spaces alone, so the rule that only spaces change holds
    for the user's edits too; an entry that changes a letter is reported and skipped, like a
    malformed list (inform, don't block). *Recommended.*

### Decided and built 2026-10-24: points 7, 9 and 10; point 8 agreed with session 5c, built later

The user: "Accept points 7-10 as recommended. Build 7, 9 and 10 with the control from the sketch;
agree point 8 with session 5c before touching keywords." `knowledge/keywords.py` is untouched.

**Point 8 as agreed with session 5c (2026-10-25).** The record showed one thing point 8 had not
seen: it labels 15 of Cajal's 16 running heads page-header and one (page 22) caption, so a skip
keyed on the label alone misses that one. 5c's rule, in its words reduced: a recorded run is
left out of keyword candidates when **(a)** its kind is `url` or `maths`, **(b)** its layout
label is page-header or page-footer, or **(c)** its closed text recurs as a recorded run on two
or more pages (a running head, whatever its label). A closed run that occurs once in body text
stays: it is the document's own text and may carry a real keyword. Conditions:

1. **Built here, after ADR-055 lands**, rebased onto that branch (it rewrites the same module and
   is not committed), not racing it.
2. **Each run carries its character span in the cached markdown**, and selection drops a run only
   where its text is. The stored span is the fast path when the text there is still the run's;
   otherwise selection re-finds the runs in the current cache with `locate_spacing_runs` and
   drops only what it locates (5c, 2026-10-25: dropping nothing on a moved span would switch
   point 8 off, unseen, for every document a table splice has rewritten). A run it cannot locate
   stays in the text and is logged, with a count per document. A document without a record
   (every cache written before this extractor) changes nothing. The dry run (condition 4)
   reports runs dropped, re-located and left unlocated, per document and in total.
3. **Order:** the recorded runs go first, on the raw text, then furniture, the reference section,
   URLs and tokenizing; one pure function with a unit test of what it drops and keeps. The tests
   5c names: Cajal's caption-labelled head is dropped (by c); the page-1 title is dropped by (b)
   and not by (c); a single letter-spaced word in prose is kept; a document without a record
   changes nothing.
4. **Before any write to the library**, a dry run listing every document whose keyword list
   moves, and 5c's invariants check (`.claude/probes/2026-10-20-keyword-cleanup/invariants.py`,
   `post.py`): the graph version, the 22 marked forms and presence must not move.

The span (condition 2) is built now, because it is part of the record and the record must be
right before the library is re-extracted: `ingest.cache.locate_spacing_runs` gives each run
`start`/`end` in the text `write_cache` writes, looking only in the run's own page and passing
over an occurrence another run of the page holds; None when the run is not there. It sits outside
the extraction fingerprint, so changing it re-extracts nothing. The table-splice scripts rewrite
the cache in place and can move later spans; condition 2's check catches it, and the same
function re-finds the runs in the current text.
5c also recorded the furniture filter's miss (alternating recto and verso heads fall under the
half-of-pages floor) as a known gap in its baseline: a structural fix of its own, not point 8.

- **7, the record.** `extract_pdf_pymupdf` fills a list per run: the run as extracted, the rule's
  closed form, its kind, its box, the layout label of the box it sits in, its page, what the text
  now holds for it (`written`), its span in the cached text (`start`, `end`, for point 8) and its
  state (closed, kept, yours, maths). `write_cache` writes it
  as `<cache>.md.spacing.json` after the markdown and before the fingerprint, so a record that
  failed to write leaves no fingerprint and the entry reads as stale. One difference from point 7
  as written: a document with no runs has **no** record file, not an empty one, so the 102
  documents without runs gain no file; a stale record from an earlier extraction is removed.
- **9, the rows.** The document page has a *Letter-spacing on this document* block, hidden for a
  document without runs. Each row is a crop of the page (the source viewer's renderer, clipped to
  the run's box with a margin: `render_page(..., clip=)`, `?clip=x0,y0,x1,y1` on the page route)
  beside the run as the text holds it, with a tick on each letter whose space was removed; the
  caption counts the removed spaces and names the pages, and a URL run says *web address*.
- **10, the user's spacing.** An entry's `spacing` maps a run to the spacing chosen in *Edit
  spacing* (the letters as a strip, a button in every gap). The API refuses a spacing that changes
  a letter (422); one written into the file by hand is skipped and logged at extraction.
- **The control, as sketched**, with the sketch's options: per run Show on page, Edit spacing,
  Keep spaced or Remove; per document Keep every run spaced, Re-run text now (the picker opens
  with *Text & chunks* ticked, still the user's to start) and All entries in Settings (opens
  Settings > Documents at the list); in Settings, Show runs, Remove, and Remove all with Undo.
  Every change writes the ignore list only, and a row says what the next re-run will do until it
  runs. Where the app differs from the sketch is in the sketch's reading
  (`docs/wireframes/letter-spacing-ignore-list.md`).
- **Checked.** The full suite, 2,831 passed; ruff, `mypy src`, bandit (CI form) and detect-secrets
  clean; svelte-check 0 errors and 0 warnings, the 308 frontend tests pass. Gate 2 with the record
  on, all 104 PDFs: 102 byte-identical, 2 changed by deleted spaces only (`cajal-lecture` 479,
  `s41598-025-33686-0` 306), as before the record; 36 runs recorded in 4 documents (3 never reach
  the markdown), every run marked in the text found there, and all 33 in the two changed texts
  located at distinct spans that hold their text. The control was driven in the app over a scratch
  copy of the data home (Cajal and `s41598-025-33686-0` extracted into it with their records): Keep
  spaced, Edit spacing and Save, Show on page, All entries in Settings, Remove all and Undo, and the
  re-run picker opened and cancelled; dark, light and 375 px; no console error from the code as
  built. `data/` was not written: the working library has no record files and no
  `rejoin_ignore.json` until this lands and the library is re-extracted.

## Amendment 2026-11-01 — landed, and what reading it as new found

The work waited out the 0.7 release as a patch against `4655aae`, and now applies to `main` after
0.7.1; its one conflict was the DEVLOG, rotated in the meantime. Points 7, 9 and 10 land; point 8
stays behind ADR-055. Read again as if new, and driven down the path the first build only opened and
cancelled (a real re-run of a document's text, on a scratch copy of the library), it showed four
faults and three loose ends. Each is fixed, with a test.

- **Bold and italic runs did not close.** The rule found them, but the text pattern wanted
  whitespace before a run, and markdown writes `**` or `_` directly against it. So a bold
  letter-spaced title stayed spaced, and its record said it never reached the text. A run may now
  follow emphasis (`_RUN_START`), and the emphasis stays. This moves what closes, so gate 2 is run
  again below.
- **A kept or maths run recorded a form the text does not hold.** `written` gave one space between
  glyphs where the markdown holds up to three at a word break, so the run's span came back empty: 16
  of the 17 runs of a document whose running head is kept. Point 8 reads those spans. `written` is
  now what the text holds.
- **The block did not follow a re-run.** It read its document once, so after *Re-run text now…* it
  went on saying what the next re-run would do. It now reads again when the library list is
  refetched, which a finished re-run does, and keeps its rows on screen meanwhile.
- **A list of the wrong shape was not read as empty.** `{"documents": null}` raised in the reader
  every freshness check calls, so a hand-edited file could stop a scan, and an entry of the wrong
  shape reached the routes as a 500. Extraction now skips a bad entry and logs it, and the routes
  refuse with a sentence and leave the file alone.
- **Loose ends.** Removing a document, or cleaning an orphan, left its record behind; `list_entries`
  read every document's row and missed a file the library had moved, as `list_letter_spacing` did;
  Settings could say "the 0 documents' text".

**Gate 2 again**, on the final rule and all 104 documents, against a copy of the live cache:
102 byte-identical, 2 changed by deleted spaces only (`cajal-lecture` 479, `s41598-025-33686-0` 306), no errors, as at every earlier run (those two compared with the live cache itself, the drive having re-extracted their copies). 36 runs recorded in 4 documents, 33 of them in the text, each at a distinct span that holds its text. The bold and italic fix moves nothing here: the same 3 runs still never reach the text.

**Every format re-reads, measured:** with this change the fingerprints of `.pdf`, `.epub`, `.docx`,
`.html`, `.md` and `.txt` all move, so the first full ingest after it reads every document once, as
the 2026-10-23 departure says. An unchanged text is skipped by its hash and not embedded again.

**Still to come** is ROADMAP row 108: the library's re-extraction with its backups, rebuilds and the
private eval (gates 3 and 6), more letter-spaced PDFs (gate 4), and point 8. Re-extracting before
point 8 lands brings the page-furniture keyword this ADR measured, on the next keyword rebuild.
