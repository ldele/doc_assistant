<!-- status: active · updated: 2026-08-07 · class: append-only -->

# ADR-039 — Recover scanned PDFs with an opt-in OCR sidecar that restores a text layer, not markdown

- **Status:** proposed
- **Date:** 2026-08-01
- **Deciders:** user + Claude Code

## Context

A PDF with no text layer extracts to nothing, and the document silently becomes unreachable.
Measured on the live 97-document corpus (2026-08-01): **93 healthy · 2 marginal · 2 broken**, and
`middleton-2001.pdf` holds **0 chunks** — it is in the library, in the UI, and in no retrieval result
that will ever be returned. It surfaced through a side effect: it was the one case in the private
35-case A/B whose ranking was not reproducible, because with no true match the cross-encoder was
breaking ties among irrelevant candidates
([`sparse_arm_private35_2026-08-01.md`](../../tests/eval/baselines/sparse_arm_private35_2026-08-01.md) §4).

**Diagnosis is already deterministic and does not need a model.** `health.py` scores every document
on chunk count, average chunk length, page count, section-detection rate and reference ratio, and
classifies healthy/marginal/broken **with reasons**; `ingest` records `extraction_health` and logs
`no_indexable_text`. KI-26 characterised this exact file — a scan whose 290 extracted characters are
15 page markers and nothing else. The open question is **remediation**, not detection.

Three existing constraints shape the answer. The primary path is locked (extract → markdown → chunk
→ embed → store) and derived data ships as an idempotent sidecar that never mutates the chunk store
(`.claude/CONTEXT.md`, non-negotiable #4). The app must keep installing and running without new
system binaries. And **first ingest is already the binding scale constraint** — ~41 h of
single-threaded extraction projected at 10,000 documents
([`performance.md`](../performance.md)) — so nothing may make the default ingest slower.

## Options

1. **OCR to a searchable PDF, then re-run the existing extractor.** `ocrmypdf` (Apache-2.0) wraps
   Tesseract and writes a new PDF with an invisible text layer, leaving the page images intact;
   Tesseract and Ghostscript are system binaries, not pip-installable
   ([ocrmypdf docs](https://ocrmypdf.readthedocs.io/en/latest/installation.html)). Trade-off: needs
   two external binaries, but the recovered document then flows through **one unchanged extraction
   path**.
2. **OCR straight to markdown via Marker.** The project already runs Marker out-of-process for
   tables (`uvx --from marker-pdf marker_single`, `scripts/extract_tables_marker.py`), it does its
   own OCR, and it needs no system binary. Trade-off: it emits *markdown*, so it becomes a **second
   extraction path** producing different structure from PyMuPDF4LLM — and Marker was deliberately
   removed from the production extraction path already (`extractors.py` raises for any
   `PDF_EXTRACTOR` other than `pymupdf`).
3. **PyMuPDF's built-in OCR.** `pymupdf>=1.24` is already a direct dependency and exposes
   `Page.get_textpage_ocr()`, which shells out to Tesseract via `TESSDATA_PREFIX`
   ([PyMuPDF docs](https://pymupdf.readthedocs.io/en/latest/page.html#Page.get_textpage_ocr)).
   Trade-off: no new Python dependency, but it still requires the Tesseract binary, and it returns
   text per page rather than a durable artifact — every re-extraction pays the OCR cost again.
4. **A VLM reads the pages.** The 4c figure path already sends images to a vision model. Trade-off:
   it is the only option that handles pages which contain no text at all, and it is
   **non-deterministic, metered, and unbounded** in cost — the opposite of the stated preference.

## Decision

**Option 1: an opt-in sidecar that OCRs `broken` documents into a persisted searchable PDF, which
the extractor then reads instead of the original.**

The deciding reason is that it is **the only option that does not create a second extraction path**.
Everything downstream — page markers, the parent-child chunker, table splicing, the health scorer —
keeps seeing output from `extract_pdf_pymupdf`, because that is still what produces it. A recovered
document becomes an ordinary document. Option 2 would fork extraction into two shapes that must stay
compatible forever, and this codebase has just spent a session deleting exactly that kind of
second path (ADR-038), after KI-29 showed what a path nobody exercises does to the answer.

OCR emits a **text layer, not text**: the artifact is a derived PDF at `<data>/ocr/<doc_hash>.pdf`,
keyed on the original's hash, and the source file is never modified. Persisting it (rather than
OCRing to a string, option 3) makes every later re-extraction free — including `ingest --rebuild`,
which would otherwise silently drop the recovery.

**It is opt-in and absent-tolerant.** It runs only from its CLI runner, never during ingest: OCR is
seconds-to-minutes per page against an ingest path that is already the scale bottleneck. With
Tesseract absent, the runner exits with the missing-binary message and everything else works
unchanged — the precedent `extract_tables_marker` already sets.

**What would reverse it:** OCR quality on real scans being bad enough that recovered text pollutes
retrieval more than absence did (RG-025 below is the gate), or the broken rate on a heterogeneous
corpus turning out high enough that recovery must run automatically at ingest — at which point the
cost question reopens and option 2's GPU OCR becomes competitive.

## Consequences

**Easier.** A scanned document becomes retrievable with one command and no re-architecture. The
health layer gains an action: `broken` stops being a label and becomes a queue. The trigger is a
signal the system already computes, so no new classification is needed.

**Harder.** Two system binaries (Tesseract, Ghostscript) enter the *optional* dependency surface, and
they are the first ones this project has needed at all — packaging (`docs/desktop-packaging.md`) must
either bundle them or state plainly that OCR is unavailable in the frozen build. A third on-disk
artifact class appears under the data home and needs a line in the Settings → Corpus disk table
(ADR-037) and in `.gitignore`.

**Must revisit.** Whether recovery belongs in the ingest path at all once its rate is known on a
larger, more heterogeneous corpus; and whether option 4 is warranted for the residue this cannot fix
— a page that is genuinely only a figure has no text to recover, and OCR will correctly return
nothing.

## Confidence

- ✓ **The problem is real and sized on this corpus** — 93 healthy / 2 marginal / 2 broken of 97;
  `middleton-2001.pdf` at `chunk_count=0`, measured 2026-08-01 against `data/library.db`.
- ✓ **Diagnosis needs no model** — `health.py` already classifies with reasons, and KI-26 records the
  specific cause for this file.
- ⚠ **The ~2% broken rate is one corpus at n=97**, heavily weighted to modern born-digital papers,
  so it bounds nothing about a scan-heavy or historical library. Tracked as **RG-024**.
- ⚠ **OCR quality on these specific scans is unmeasured.** Recovered text that is wrong is worse than
  text that is absent: absence is honest, garbage is retrievable and citable. Tracked as **RG-025**,
  which gates enabling this by default.
- ⚠ **The engine choice is architectural, not empirical.** Option 1 wins on "one extraction path" and
  absent-tolerance, not on measured accuracy against options 2 and 3; RG-025 is where that comparison
  would happen if the first implementation disappoints.

## Amendment — 2026-08-07: the premise held, the population did not

**The decision stands. Its scope shrinks from four documents to one.**

Before installing Tesseract, the four documents this ADR was written for were examined directly.
**Three of them already carry a good text layer** — `page.get_text()` returns 45,995 / 88,754 /
776,162 characters at 88% / 72% / 94% word-like across sampled pages. They were never OCR cases.
`extract_pdf_pymupdf` was *discarding* text that was already present: PyMuPDF4LLM renders a
full-page-image page as a picture placeholder and never reaches the invisible text behind it, then
`strip_image_placeholders` (KI-14) removes the placeholder too. Retention measured **0.0%–3.2%** on
those pages against **97.3%–108.9%** on 28 healthy pages — two populations with a ~94-point gap.

Fixed by a text-layer fallback **inside the existing extraction path**, which is this ADR's own
deciding principle applied one level earlier. Result: those three went 1 / 7 / 16 chunks →
**61 / 125 / 1,019**, all `healthy`, and retrieval recall on the private 35-case set went
**28/35 → 34/35** (DEVLOG 2026-08-07 (2)).

**What remains for OCR is `middleton-2001.pdf`** — 15 pages, **zero** extractable characters, every
page an image. That is exactly the document this ADR describes, and it is now **1 of 97 (~1%)**, not
4. Everything above still applies to it: option 1, opt-in, absent-tolerant, gated on RG-025.

**What this changes about the reasoning, and it is worth carrying:**

- **The `broken`/`marginal` health labels were correct but did not imply "needs OCR".** They measure
  *what ingest produced*, not *what the file contains*. This ADR read a symptom as a diagnosis, and
  the two happened to coincide for one document out of four.
- **The sizing in Confidence — "93 healthy / 2 marginal / 2 broken" — was accurate and still
  mis-scoped the work by 4×.** The corpus statistic was never the question; the per-document cause
  was.
- **RG-024 (the ~2% broken rate bounds nothing) gets sharper, not weaker.** The true
  no-text-layer rate on this corpus is **~1%**, and the rate of *extractor-lost* text was **~3%** —
  three times larger, and previously invisible because it wore the same label.
