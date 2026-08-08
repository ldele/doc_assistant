# RG-025 — OCR output quality on the one true scan (2026-08-08)

Measures whether OCR of `middleton-2001.pdf` produces text **worth retrieving**, before ADR-039's
recovery path is built. The asymmetry RG-025 names is the whole point: a document with no text
layer is *honestly absent*, while one full of OCR garbage is retrievable, rerankable and
**citable** — so a bad recovery degrades answers in a way the current failure does not.

**Status: all 4 items done.** The Marker comparison was blocked mid-session by KI-42 (Marker
unrunnable), then unblocked by pinning `marker-pdf==1.10.2` — see §5.

**Setup**
- Target: `data/sources/middleton-2001.pdf` — **15 pages, 0 characters** of text layer across the
  whole document, `extraction_health='broken'`, `chunk_count=0`. It is now the **only** non-healthy
  document in the 97-doc library; the 2026-08-07 text-layer fallback healed the other three.
- Engine: Ghostscript 10.07.1 rasterises at 300 dpi → Tesseract 5.4.0 (`--psm 1`, `eng`). That is
  the same engine `ocrmypdf` wraps (ADR-039 option 1). **`ocrmypdf` was deliberately not installed**
  — RG-025 asks whether the *text* is worth retrieving; the searchable-PDF wrapper is the code step
  that comes after, and a dependency added before the measurement is a dependency added on faith.
- Whole document: **21 seconds**.

## 1. Character-level sanity — per page

| page | chars | word-like | | page | chars | word-like |
|---|---:|---:|---|---|---:|---:|
| p001 | 2054 | 91.1% | | p009 | 1773 | 83.8% |
| p002 | 2911 | 92.5% | | p010 | 1990 | 83.2% |
| p003 | 2908 | 87.1% | | p011 | 1541 | 87.0% |
| p004 | 1189 | 90.7% | | p012 | 2954 | 94.6% |
| p005 | 3024 | 92.0% | | p013 | 2892 | 82.7% |
| p006 | 1840 | 90.6% | | p014 | 3175 | 79.4% |
| p007 | 1729 | 80.9% | | p015 | 2042 | 79.5% |
| p008 | 2746 | 87.5% | | **TOTAL** | **34,768** | **87.0%** |

**Zero pages came back empty.** Controls on the same metric, from documents with *real* text
layers: `hodgkin_huxley_1952.pdf` **92.1%**, `rag_lewis_2020.pdf` **88.5%**.

**The low-scoring tail is a content artifact, not an OCR failure.** Pages 13–15 are the
**references section**: initials, years, volume numbers and page ranges (`242, 535-560.`,
`8683-8687.`) score as "not word-like" while being read *correctly*. Read the metric as a garbage
detector, not an accuracy score.

## 2. Hand-read against the page images

**p006 body text (~850 chars, verified word-by-word against the rendered page): one error.**
`VAmc` → `VAme`. Headers, citations (`Lynch et al., 1994`), and all anatomical abbreviations
(`GPi`, `SNr`, `FEF`, `MDpl`) correct. The figure **caption** is likewise near-perfect.

**The figure *interior* is where quality drops** — Figure 2.2's circuit diagram:

| on the page | OCR |
|---|---|
| `FEFsac` / `FEFsem` | `FEF` + `Bac` / `FEF"` + `sam` |
| `cdm-GPi` | `edn GP` |
| `MDpl` | `wippt` |
| `VLcr` / `VApc` | `Vier,` / `VApe` |
| arrows between boxes | `} i i 4` , `j j | j` |

`wippt` and `Vier,` are exactly the plausible-but-wrong tokens RG-025 warns about. **p014 also
misreads `P. S.` as `P. §.`** and terminal `l` as `!` or `]` (`mediodorsa!`, `inferotempora]`) — a
font-level confusion, not a structural failure.

> **Byte-check note (non-negotiable #9).** The `§` first appeared in-terminal as `�` and looked like
> an encoding bug. The file is **valid UTF-8 with zero U+FFFD**; the bytes are `\xc2\xa7`, and the
> cp1252 console could not render them. A genuine OCR error was one step from being filed as a
> corruption bug. Byte-check before believing the console.

## 3. How much garbage actually entered

| | count | share |
|---|---:|---:|
| non-empty lines | 542 | — |
| lines <34% word-like | 23 | 4.2% of lines |
| **their characters** | **334** | **1.0% of the document** |

And roughly half of those 23 are **correctly-read reference numerals** the heuristic misjudges. The
true diagram noise is **well under 1%**, it is confined to figure regions, and it does not dominate
any chunk — the retrieved chunks sampled are clean prose.

## 4. Retrieval check — does the recovered document earn its place?

Ingested the OCR text into an **isolated data home** (verified: a different Chroma store — see the
KI-11 relocation trap) alongside **8 topically adjacent real papers** from the library
(cortical/thalamic/basal-ganglia neuroanatomy), so retrieval is genuinely contested.

| query | expected | result |
|---|---|---|
| input nuclei of the basal ganglia | caudate/putamen/ventral striatum, p1 | **rank 1** |
| which cortical area forms the major input to the oculomotor circuit | FEF in area 8, p6 | **rank 1** |
| virus tracing of the SNr output channel to the frontal eye field | Lynch et al. 1994, p6 | **rank 1** |
| revised neuroanatomy of frontal-subcortical circuits | the chapter title | **rank 1** |
| *(negative)* how do hypothalamic dopamine neurons motivate mating | should NOT appear | **clean** — returns `nihms-1776873.pdf` |

**4/4 rank-1 on questions it genuinely answers; no false positive on one it does not.**

## 5. Marker comparison (ADR-039 option 2) — done, after unblocking it

**It was blocked first, and that mattered more than the comparison.** `uvx --from marker-pdf` was
**unpinned**, so it resolved to **marker-pdf 2.0.0**, which routes surya's inference through a
spawned backend — `vllm` (auto-picked on an NVIDIA GPU) wants a running Docker daemon, `llamacpp`
wants an uninstalled `llama-server`. Both die at the **layout** stage, before any OCR. Since
`scripts/extract_tables_marker.py` resolves Marker through the same command, the shipped table path
was broken too. Filed and fixed as **KI-42** (pinned to `1.10.2`, the last 1.x); the table runner
then found **7 tables** in `rag_lewis_2020.pdf` in 40 s.

**Head-to-head on the same 3 pages** (0-based 0, 5, 13 — title, figure-heavy body, references):

| | Tesseract + Ghostscript | Marker 1.10.2 |
|---|---:|---:|
| word-like | **85.8%** | 85.5% |
| tokens | 1083 | 990 |
| hyphen-split line breaks | **17** | **3** |
| time, these 3 pages | ~4 s | **122 s** |
| whole 15-page document | **21 s** | not run (~10 min projected) |
| extra dependency | 2 small system binaries | a multi-GB model stack |

**Reading.**
- **Raw character accuracy is a tie** — 85.8% vs 85.5% is inside the noise of the metric, and
  neither engine produced garbage.
- **Marker is meaningfully better at *reflow*.** It rejoins words split across a line break;
  Tesseract does not, leaving **88 hyphen-split words across the full document**
  (`re-\nviewed`, `cortico-\nspinal`). Those are a **real retrieval defect** — a query for
  "corticospinal" will not match `cortico-` + `spinal`. Marker also emits structure (`##` headings)
  and extracts the page's figure as a separate image.
- **Tesseract is ~30x faster** (≈1.4 s/page vs ≈40 s/page) on two small binaries rather than a
  model stack that, on this box, needed a *pin* to run at all.

**Conclusion: ADR-039's option 1 stands, with one amendment.** The dependency and cost argument that
decided it has only strengthened — Marker's own runnability turned out to be version-fragile, which
is precisely the risk "one extraction path, absent-tolerant" was chosen to avoid. But Marker's
reflow advantage is real and cheap to close: **the sidecar should de-hyphenate line-broken words**,
which is deterministic text normalisation, not a second engine. That is now the one concrete
quality task the OCR runner inherits.

## Verdict

**OCR recovery of this document clears the bar RG-025 set, and the engine choice is confirmed.**
The text is ~87% word-like (inside the healthy band once the references confound is accounted for),
no page is empty, sub-1% of characters are noise confined to figure interiors, and the recovered
document wins 4/4 contested retrievals without polluting an unrelated one. Against Marker it ties
on accuracy at ~1/30th the cost, losing only on hyphen reflow — which is a post-process, not an
engine choice.

**RG-025 can close.** Two conditions carry forward into the build: de-hyphenate, and keep recovery
opt-in.

**Scope honesty:** one document, 15 pages; two pages hand-read closely. This licenses building
ADR-039's sidecar behind its runner — it does **not** license enabling recovery by default, which
needs the broken-rate measurement (RG-023/RG-024) on a heterogeneous corpus.

## Carried forward

1. **De-hyphenate line-broken words in the sidecar** — 88 of them in this document, each one a word
   that will not match a keyword query. The single highest-value fix, and deterministic.
2. **Figure interiors should not reach retrieval as prose.** The cheapest guard is to keep ADR-039's
   "visibly marked as OCR-derived" rule and let the existing figures layer own figure regions.
3. **KI-26's title extractor will start reading OCR'd front matter** — p001 OCRs cleanly here, but
   that is one sample.
4. ✅ **`marker-pdf` is pinned** (KI-42, `config.MARKER_VERSION`), so a future re-comparison is
   reproducible. Bump it only with an end-to-end runner check — the failure mode is a hard stop.
