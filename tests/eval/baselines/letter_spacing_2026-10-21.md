<!-- status: active · updated: 2026-10-24 (the review: 33 verdicts; context rules: 102 identical, 2 changed; 2026-10-23: gate 2: the built extractor, 101 caches identical and 3 changed by spaces only; synthetic pages: one false join and one miss found, the miss fixed; ADR-056 accepted on conditions; 2026-10-21: measured read-only on the 104-document library; nothing re-extracted; the fix is ADR-056, proposed, not built) · class: baseline -->

# Letter-spaced text in extracted PDFs: where it comes from, and what a rejoin would change

**15 of 104 documents have a run of four or more single letters in their extracted text, 133 runs
in all, and only 2 documents are damaged by it.** Cajal's 1906 lecture (33 runs) and
`s41598-025-33686-0` (25 runs) carry spaces their pages do not show. The other 13 documents hold
letters that really are separate (figure panel labels, variables, axis ticks), one publisher's
header whose spaces are in the text layer, and one OCR'd book whose damage is of another kind. A
rejoin rule that reads only the PDF's own glyphs changes **3 caches of 104, by deleting spaces and
nothing else**: Cajal's letter-spaced runs go from 33 to 0, `s41598`'s from 25 to 0.

These are counts on one library's PDFs, measured 2026-10-21. They say where this library's
letter-spacing comes from and what the proposed rule does to it, not how common it is elsewhere.
Probe scripts and outputs: `.claude/probes/2026-10-21-letter-spacing/` (local-only).

## Cause

PyMuPDF writes a space into its text wherever the gap between two glyphs passes a share of the
font size, and marks each one it writes (`char["synthetic"]` in `rawdict`). A title set with wide
letter-spacing has such a gap after every letter, so each gets a space. Cajal's PDF draws its title
with per-letter offsets (from the page's content stream):

```
[(A)-364(N)-364(T)-364(I)-364(A)-364(G)-364(O)-364( )250]TJ
```

The page shows **SANTIAGO RAMÓN Y CAJAL**. PyMuPDF reads `S A N T I A G O  R A M Ó N  Y   C A J A L`,
and the cached markdown has `S A N T I A G O R A M Ó N Y C A J A L` (one space everywhere, so even the
word breaks are gone). With PyMuPDF's added spaces turned off (`TEXT_INHIBIT_SPACES`), the running
head reads `NEURONS: STRUCTURE AND CONNEXIONS`: the spaces between its letters are not in the file.

**Those added spaces cannot simply be turned off.** They are 814,286 of the library's 1,432,158
spaces (56.9%), and in 68 of 104 documents more than 95% of the spaces are PyMuPDF's: those PDFs
carry no space glyphs at all, and every word break in them is an added space.

`s41598-025-33686-0` is the other mechanism. Its URLs are drawn as letter, space glyph, letter,
with each space given no width: the letters touch (gap −0.07 to 0.01 em) and the page shows
`https://www.creatis.insa-lyon.fr`, while the text layer holds `h t t p s : / / w w w`.

## Every run in the cache, by what the PDF says about it

A run is four or more letters separated by one to three spaces, found in the cached markdown and
then located among that page's glyphs, whose gaps and space marks decide the class.

| Class | What the PDF holds | Runs | Documents | Pages | Damage? |
|---|---|---:|---:|---:|---|
| A. Letter-spacing | letters drawn apart, no space between them (PyMuPDF added each) | 33 | 1 (`cajal-lecture`) | 17 | yes: the title line and the running head |
| B. Invisible spaces | a space glyph between touching letters | 25 | 1 (`s41598-025-33686-0`) | 4 | yes: URLs, a licence link, "Downloads" |
| C. Visible spaces | a space glyph with a visible gap, 0.44 em | 1 | 1 (`03-Zuo2014`) | 1 | no: the text layer says `a r t i c l e  i n f o` |
| D. OCR fragments | an OCR layer whose glyph positions are estimates | 42 | 1 (`Computational_neuroanatomy`) | 16 | yes, but another defect: `In tro d u c tio n` |
| E. Separate letters | panel labels, variables, axis ticks | 32 | 11 | 16 | no: `A B C D`, `x y z` are separate |
| **Total** | | **133** | **15** | | |

Class A's letter gaps are 0.21 em (running head) and 0.36–0.39 em (title); a word break in the
title is 0.76–1.10 em. Class E's runs sit mostly in the figure text PyMuPDF4LLM emits (`<br>`
separated); their letters are positioned one by one, far apart.

## The rule, and every run it would change

Prototype: `rule.py` in the probe folder. It reads one text line of the PDF at a time.

- **Letter-spacing:** three or more one-letter tokens, separated by spaces PyMuPDF marked as its
  own, all letters (a closing `:` `,` `.` allowed after one), on one baseline, and with the
  neighbours on the line further away than the run's own spacing plus half a quarter-em word
  space. Gaps up to that close; wider gaps stay one space. The neighbour condition is what leaves
  an OCR fragment alone (`tro`, `d u c`, `tio` sit equally close); the baseline condition is what
  leaves a sub- or superscript alone (`W^V_i`).
- **Invisible spaces:** in a run of one-glyph tokens, a space between glyphs closer than 1/20 em
  closes. No typeface sets a word space that narrow: of the 1.38 million spaces that sit between
  two glyphs in the library, 1.33 million leave a gap of 0.15–0.55 em, and 3,425 one within
  0.05 em of zero (`simulate.py`'s histogram).

Over all 226,227 text lines of the 104 PDFs it closes 48 runs in 5 documents:

| Document | Runs closed | In the cache | Spaces removed | What changes |
|---|---:|---:|---:|---|
| `cajal-lecture` | 17 | 17 | 479 | the title line, 16 running heads |
| `s41598-025-33686-0` | 16 | 16 | 306 | its URLs (some in two pieces across a line), the licence link, a "Downloads" link |
| `Computational_neuroanatomy` | 12 | 9 | 67 | OCR noise in figures and formulas (`o o o o` → `oooo`), `T h e` → `The` |
| `pnas.202116950` | 2 | 0 | 0 | a vertical margin label PyMuPDF4LLM does not emit |
| `elife-78635-v2` | 1 | 0 | 0 | a space before a comma |

The other 99 caches are byte-identical. For the three that change, the text with every space
deleted is identical before and after: the rule only ever removes spaces. A URL that wraps onto a
second PDF line keeps one space at the wrap (`creatis.in sa-lyon.fr`), since the rule reads one line
at a time.

Two earlier drafts of the rule are in the probe folder because their errors set its conditions:
geometry alone closed 215 runs in 20 documents, including maths (`d = m + n + 1` → `d=m+n+1`) and
initials in an OCR layer (`W J H`); adding PyMuPDF's own space marks and the neighbour condition
brought it to 66 in 6, with one maths error left (`i , V W V` → `i ,VWV`), which the letters-only
and baseline conditions removed.

## What it does downstream (projected on the rejoined caches)

- **Keywords** (session 5c's extractor, dry run over the whole corpus): only Cajal's list moves.
  `neurons structure` comes in, read across the colon of the running head, and `retzius` goes
  out. The 13 letter-spaced fragments the user found are already gone under 5c's filter, so the
  rejoin does not improve this list. The running head survives the page-furniture filter because
  it is on 17 of 35 pages and the filter needs half (17.5).
- **The rest of Cajal's list is not letter-spacing:** `s ram`, `ram n` and `lenhoss` come from the
  keyword tokeniser, which is ASCII-only and splits `Ramón` into `ram` + `n`. That split is
  corpus-wide: 2,259 word occurrences (0.14%) in 90 of 104 documents, 823 distinct words
  (`hofstätter` → `hofst` + `tter`, `garcía`, `kölliker`, and `μm` → `m`). The search index uses
  the same tokeniser.
- **Metadata:** Cajal's stored author is `Nobel Lecture, December 12, 1906` with or without the
  rejoin. The picker reads the line after the title, and the author line comes before it.
- **The private eval set** has three cases citing a changed document: `cajal_neuron_doctrine`,
  `cajal_method`, `friston_computational_neuroanatomy`.

## What a re-extraction would cost here

All 104 caches are fresh against today's extractor (fingerprint `dbfcf8f76d819638`). A change in
`extract_pdf_pymupdf` moves the PDF fingerprint, so the next "index the folder" re-extracts every
PDF, about 24 minutes on this machine (`docs/performance.md`, 97 documents), as on 2026-10-14
(KI-64). The rule's own pass took 106 s over the 104 PDFs in the probe, about 1 s per document.
Three documents then change hash and keep their identity (ADR-047): 1,379 chunks re-embedded
(87 + 119 + 1,173), then the keyword index, the keywords and the graph rebuilt.

## 2026-10-23 — synthetic pages, and the rule refined

Pages drawn with one known mechanism each (`synthetic.py`) found one false join and one miss.
**Miss:** a letter-spaced word inside a sentence, spaced like the sentence's words, looked like an
OCR fragment and stayed spaced. Fixed for PDFs that hold their word spaces: a space the PDF holds
now counts as a word break when the run's own gaps are PyMuPDF's (`Das ist gesperrt gesetzt`). In
a PDF with no space glyphs at all it is still missed. **False join:** a row of single-letter labels
on one line, 0.4–0.7 em apart, closes to `ABCD`; at 0.8 em and more PyMuPDF splits the row into
separate lines. Real tracking here is 0.21–0.39 em, so the two overlap, and that case goes to
ADR-056's ignore list rather than a threshold. The refined rule (`rule.py`; the earlier one is
`rule_v3.py`) closes the same 48 runs over the 104 PDFs; one OCR-noise join in
`Computational_neuroanatomy` swaps for another (`i è f` out, `X a A` in).

## 2026-10-23 — gate 2: the built extractor, not a projection

The extractor as built (ADR-056, DEVLOG 2026-10-23) on all 104 PDFs, written to a scratch folder
and compared with the live cache (`gate2_extract.py` in the probe folder; nothing in `data/`
written). **101 byte-identical, 3 changed by deleted spaces only**, the three this file predicted:

| Document | Spaces removed | Letter-spaced runs |
|---|---:|---:|
| `cajal-lecture` | 479 | 33 → 0 |
| `s41598-025-33686-0` | 306 | 25 → 0 |
| `Computational_neuroanatomy` | 68 | 42 → 37 |

The OCR'd book's figures differ slightly from the projection (67 spaces, 39 runs left). The
projection used the rule before its 2026-10-23 refinement, which changed one join of OCR noise in
that book; the rest of the gap is not traced, and all of it is inside that book's OCR noise. The
pages where PyMuPDF4LLM ran OCR came out identical to the cache, so the environment has not drifted
since it was written. One document needed a second pass: its name is long enough that the scratch
path passed Windows' 260-character limit, and it is identical too.

## 2026-10-24 — the review, and the context rules

**The review** (ADR-056 gate 5): the 48 joins as 33 places, each judged by the user against a crop
of its page. 6 Close it, 27 Not text (16 URL runs, 1 maths, 10 OCR noise), 0 Keep spaced; the
user then chose to close the URLs anyway. Record: `review_verdicts_2026-10-24.json` in the probe
folder.

**Context signals against those verdicts** (`context_signals.py`, `scan_pages.py`):

| Signal | What it found |
|---|---|
| A scanned page (one image covers it) | all 12 OCR-book runs; over the library 606 pages at ≥ 0.9, every other at ≤ 0.7, none between |
| A formula box (layout analysis) | 2 runs; the book's figure noise is labelled "text", so the boxes alone miss it |
| A URL shape, or a URL before the run | 15 of 16 URL runs by shape, the 16th (a query string) by the URL before it |
| A maths glyph, no URL | the eLife run |

Rules on those signals reproduce 31 of the user's 33 decisions; the two they miss (`The`, `WOl` on
the scanned book) they leave spaced.

**Gate 2 with the rules built**, all 104 PDFs: **102 byte-identical, 2 changed by deleted spaces
only**:

| Document | Spaces removed | Letter-spaced runs |
|---|---:|---:|
| `cajal-lecture` | 479 | 33 → 0 |
| `s41598-025-33686-0` | 306 | 25 → 0 |

The OCR'd book is identical now. The run took 67 minutes on 4 processes while the full test suite
ran beside it, so it says nothing about the rules' cost; the layout analysis they add runs only on
pages with a candidate join (23 of 3,341 here, once scanned pages are skipped).
