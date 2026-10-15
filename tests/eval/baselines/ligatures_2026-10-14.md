<!-- status: active · updated: 2026-10-14 (the library re-extracted: 38 documents changed, 519 letters; KI-64: table cells keep a ligature's second letter; re-extraction of the 19 affected PDFs and 3 others, the committed extractor against the fix) · class: baseline -->

# Table cells keep a ligature's second letter (KI-64)

**Re-extracted with the fix, the 22 PDFs measured change in 416 places, every one an insertion of
what the old extraction dropped from a table cell:** the second letter of a ligature
("fne-tuning" is "fine-tuning" again, "Ofine" "Offline", "ShufeNetV2" "ShuffleNetV2",
"sufcient" "sufficient") and, in one paper, the slash that makes "=" a "≠". No other character
of any document changes. The words of the KI-64 count go from 316 to 0. Prose was never
affected, and extraction takes as long as before.

These are counts on one library's PDFs. They say what the fix restores here, not how many
ligatures another library's tables hold.

## Cause

Importing `pymupdf4llm` calls `pymupdf.TOOLS.unset_quad_corrections(True)` for the whole process.
With quad corrections off, PyMuPDF splits a ligature glyph back into its letters but gives the
second one a box of zero width (the "i" of "fine" in the SuperAnimal paper: x0 = x1 = 151.70,
against 151.70–153.38 with corrections on). Both table-cell functions,
`pymupdf.table.extract_cells` and `pymupdf4llm.helpers.utils.extract_cells` (the one this app's
layout path runs), keep a character only if more than half of its box lies inside the cell, and
zero is never more than half of zero. The PDFs' own text layers were right throughout.

## The fix

`extractors.widen_zero_width_chars` gives a zero-width character a box 0.01 pt wide at its own
origin; `extractors._keep_ligature_letters_in_table_cells` hands both cell functions characters
measured that way, without copying or editing either library's code, and is called from
`extract_pdf_pymupdf`. That puts it in the PDF extraction fingerprint: the PDF fingerprint moves
(`c4e9e72fd9f7` → `dbfcf8f76d81`) and the EPUB, HTML, DOCX, Markdown and text fingerprints do not,
so every cached PDF re-extracts once at the next ingest and only the documents below change.

## Environment / method

- **PDFs:** the 19 whose stored table text held a damaged word (2026-10-14 count), and 3 that held
  none, as a control. PyMuPDF and pymupdf4llm 1.27.2.3, the layout path on.
- **The comparison:** the ingest's own `extract_pdf_pymupdf`, run once with the committed code
  (`87fbf34`, extracted with `git archive`) and once with the fix, each writing the full text of
  every PDF to a scratch folder; then a line diff and, inside each changed line, a character diff.
  Probe: `.claude/probes/2026-10-14-ligatures/` (`extract_dump.py`, `compare_dumps.py`,
  `compare_2026-10-14.txt`; local-only). Nothing was written to the extraction cache or the library.
- **Cost:** $0. No model.

## Results

| PDF | Insertions | Damaged words before → after |
|---|---:|---:|
| SuperAnimal (2203.07436) | 255 | 240 → 0 |
| GraphRAG (2404.16130) | 38 | 26 → 0 |
| fnana-18-1419108 | 31 | 6 → 0 |
| Pretrained Transformers for Text Ranking | 15 | 11 → 0 |
| Fakhar et al., eLife | 14 (8 letters, 6 "≠" slashes) | 2 → 0 |
| 2503.23674 | 14 | 4 → 0 |
| LLM-as-a-judge (Zheng 2023) | 10 | 7 → 0 |
| Knowledge-distillation survey (Gou 2021) | 6 | 3 → 0 |
| 2607.09666 | 6 | 3 → 0 |
| 10 more of the 19 | 1–4 each, 25 in all | 14 → 0 |
| Control: Neuron 1-s2.0-S0896627320307170 | 2 ("fly", "Optiflex") | 0 → 0 |
| Control: two others | 0 | 0 → 0 |
| **All 22** | **416** | **316 → 0** |

What was inserted: "i" 361 times, "l" 36, "fl" 6, "fi" 3, "f" 2, the "≠" slash 6, and "l" twice
with strike-through marks of its own. No character was deleted or replaced anywhere.

- **One control was damaged too:** its table held "fy" and "Optifex", words the KI-64 count's list
  did not know. The count undercounted, as KI-64 said it would.
- **The "≠" slashes:** "i ≠ j" is set as "=" with a zero-width combining overlay (U+0338); its
  box was zero too, so table cells read "i = j". They read "i ̸= j" now.
- **One cosmetic difference:** in the Neuron paper's struck-through cells the restored letter sits
  outside the strike-through marks (`~~f~~l~~y~~`). The text reads "fly" once the marks are
  removed.
- **Not touched:** the 3 damaged words in prose ("Consder frst", the scanned 1949 Hebb book) are
  OCR errors, not ligatures.

## What reaches the app

**Re-extracted the same evening, at the user's request.** "Index the folder" re-extracted all 104
PDFs (the fingerprint had moved): **38 documents changed and 66 came out identical**, no errors.
Every one of the 38 changed only by restored ligature letters, **519** in all — the 22 measured above
undercounted, as the control already showed, and one of the 38 had lost a "tt" ligature
("paterns"). The 38 kept their ids and sidecars (ADR-047); their chunks were re-embedded and the
superseded ones removed. The keyword index was rebuilt (45,581 chunks) and the graph after it
(`hard negatives` and `re-ranking` count 2 more mentions in one document each, one link moves
49 → 50 shared chunks; `hyphen_matcher_2026-10-14.md` §1). The stored text holds no damaged word
now except the 3 OCR errors of the 1949 Hebb scan. Backups of every store written:
`data/*.bak-20261014-214357-preki64`. Probe: `cache_diff.py`, `cache_diff_2026-10-14.txt`.
