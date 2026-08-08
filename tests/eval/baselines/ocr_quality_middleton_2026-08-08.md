# RG-025 — OCR output quality on the one true scan (2026-08-08)

Measures whether OCR of `middleton-2001.pdf` produces text **worth retrieving**, before ADR-039's
recovery path is built. The asymmetry RG-025 names is the whole point: a document with no text
layer is *honestly absent*, while one full of OCR garbage is retrievable, rerankable and
**citable** — so a bad recovery degrades answers in a way the current failure does not.

**Status: 3 of 4 required items done. The Marker comparison is BLOCKED — RG-025 stays open.**

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

## 5. ❌ Marker comparison — BLOCKED, and it broke something else

ADR-039 option 2 could **not** be evaluated. `uvx --from marker-pdf marker_single` now fails at the
**layout** stage (not merely OCR) because current `surya` routes inference through a backend:

- `vllm` (auto-selected when an NVIDIA GPU is present) → `docker run` against a daemon that is not
  running, plus a container image pull;
- `llamacpp` → needs a `llama-server` binary that is not installed, and whose documented install
  path covers only macOS/Linux.

**This is not only an RG-025 gap — it breaks a shipped runner.**
`scripts/extract_tables_marker.py` resolves Marker through the *same* unpinned
`uvx --from marker-pdf` command, so the table-extraction enrichment path is broken on this machine
too. `eval_marker_tables.py:85` still says it runs "against the pinned marker-pdf version at build
time"; **there is no pin anywhere in the repo.** Filed as **KI-42**.

**What it does and does not tell us.** It says nothing about Marker's *accuracy*. It does say
ADR-039's dependency-surface argument has moved further in option 1's favour: Tesseract + Ghostscript
read all 15 pages in 21 s from two small system binaries, while option 2 now needs Docker or
llama.cpp plus a multi-GB model before it will emit a character.

## Verdict

**On the evidence gathered, OCR recovery of this document clears the bar RG-025 set.** The text is
~87% word-like (inside the healthy band once the references confound is accounted for), no page is
empty, sub-1% of characters are noise confined to figure interiors, and the recovered document wins
4/4 contested retrievals without polluting an unrelated one.

**RG-025 stays OPEN** on the Marker comparison, which its contract requires before "concluding the
engine choice was right".

**Scope honesty:** one document, 15 pages; two pages hand-read closely. This licenses building
ADR-039's sidecar behind its runner — it does **not** license enabling recovery by default, which
needs the broken-rate measurement (RG-023/RG-024) on a heterogeneous corpus.

## Carried forward

1. **Figure interiors should not reach retrieval as prose.** The cheapest guard is to keep ADR-039's
   "visibly marked as OCR-derived" rule and let the existing figures layer own figure regions.
2. **KI-26's title extractor will start reading OCR'd front matter** — p001 OCRs cleanly here, but
   that is one sample.
3. **Pin `marker-pdf`** (KI-42) before trusting any future Marker comparison to be reproducible.
