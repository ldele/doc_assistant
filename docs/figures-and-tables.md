# Figures & Tables — detection and extraction (Phase 6 / Feature 4)

How doc_assistant promotes tables (and, later, figures) from lossy text into
structured, retrievable content — and, just as important, how it *avoids
mistaking figures for tables*. This is the reference for `regions.py`,
`tables.py`, and the three `scripts/*table*` tools. Locked decisions live in
[`decisions.md`](decisions.md) (Feature 4a); per-change history in
[`DEVLOG.md`](DEVLOG.md).

> **Status (2026-06-02):** detection foundation built and validated. The
> extraction *engine* (pdfplumber vs Marker) is not final — see
> [Open questions](#open-questions).

---

## The problem

- **Tables were already in the pipeline, but lossy.** The primary extractor
  (`pymupdf4llm`) emits markdown pipe-tables, but mangled: `<br>`-jammed
  cells, collapsed columns, encoding artifacts. So a table is *present* yet
  ambiguous.
- **Figures are dropped entirely.** `write_images=False` — only figure
  *caption text* survives; the image is gone.
- **Geometric detectors confuse figures, charts, and tables.** pdfplumber
  and PyMuPDF `find_tables()` match *shape* (gridded, bounded regions), not
  meaning. A bar-chart grid reads as 13 "tables"; shaded prose boxes read as
  tables; EM-image grids read as tables.

The last point is the trap: a naive "extract all tables" pass pollutes the
cache with figure-panel noise. A visual check ([`debug_tables.py`](#tooling))
made this concrete and reshaped the design — see the atlas lessons
`2026-06-02-geometric-pdf-table-detectors-*` and
`2026-06-02-extraction-success-counts-*`.

---

## The architecture: classify first, then extract

One **page content classifier** ([`regions.py`](../src/doc_assistant/regions.py))
is the shared detection layer under both tables (4a) and figures (4b).
Instead of each feature running its own geometric detector, classify what a
page contains once, then route.

```
PDF ─► regions.analyze_pages() ─► per-page PageClassification
                                     │
        ┌────────────────────────────┼─────────────────────────────┐
        ▼                            ▼                              ▼
   kind = "table"            kind = "chart"/"photo"/"figure"    kind = "text"
   tables.extract_…()        (feeds Feature 4b — figures)       (ignored)
        │
        ▼
   splice into the cached .md (idempotent, marked)
        │
        ▼
   next `ingest` run → table becomes retrievable chunk content
```

### The signals (measured, not guessed)

Across ~50 pages of the public eLife corpus the three classes separate by
orders of magnitude, so cheap heuristics suffice — **no ML, no Marker**:

| Signal | Table | Chart | Photo / raster figure |
|---|---|---|---|
| `Table N` caption | ✓ | — | — |
| `Figure N` caption | — | ✓ | ✓ |
| vector `curve_count` | 8–187 | **1,000–78,000** | varies |
| raster `image_area_fraction` | ~0 | ~0 | **0.09–0.60** |

Thresholds (in `regions.py`, tunable): `CHART_CURVE_MIN = 1000`,
`IMAGE_AREA_MIN = 0.05`.

### The routing rule (`classify_page`)

- **chart** if `curve_count ≥ CHART_CURVE_MIN`
- **photo** if `image_area_fraction ≥ IMAGE_AREA_MIN`
- **table candidate** iff `has "Table N" caption` **and** *not* chart **and**
  *not* image-dominated — so a chart page with a stray "Table" mention is
  never mis-routed
- **figure** if it has a `Figure N` caption but no dominant chart/image signal
- **text** otherwise

`classify_page(signals)` is pure (exhaustively unit-tested);
`analyze_pages(pdf_path)` / `table_candidate_pages(pdf_path)` do the PyMuPDF
signal extraction.

### Scope (v1): page-level

Signals are aggregated per page. A page mixing a table *and* a chart is
labelled by the dominant signal, not split into regions. True per-region
bbox splitting is the deeper **Feature 4b** step; this classifier is its
foundation (its chart/photo/figure verdict + image-area signal are exactly
what figure handling will consume).

---

## Table extraction ([`tables.py`](../src/doc_assistant/tables.py))

A post-ingest **enrichment layer** (see "Enrichment-Layer Pattern" in
`decisions.md`): separate module + CLI, idempotent, never mutates the chunk
store. Tables are the one sanctioned exception to "sidecar by default" —
they're text-shaped, so they splice back into the markdown cache.

- `extract_tables(pdf_path)` = `regions.table_candidate_pages` ∘
  `extract_tables_from_pages` (pdfplumber on just those pages).
- **Content guards** (`_is_meaningful`) are the second line of defence:
  reject cells > `MAX_CELL_CHARS` (prose), require ≥ `MIN_COLS` non-empty
  columns.
- **Splice** (`splice_tables`): all tables go in one demarcated block
  appended to the `.md`:
  ```
  <!-- tables:pdfplumber:begin -->
  ## Tables extracted by pdfplumber
  <!-- table-extracted-by: pdfplumber page=7 table=1 -->
  | ... |
  <!-- tables:pdfplumber:end -->
  ```
  Idempotent: re-splicing **replaces** the block (`splice == splice∘splice`),
  so the CLI's `--force` is safe.
- **Retrieval:** the splice writes the cache only; tables enter retrieval on
  the next `ingest` run that re-reads the cache.

Validated end-to-end: a paper with no `Table N` caption → 0 tables; a paper
with Table 1 → the real Table 1 data (the prior figure-as-table noise is
gone).

---

## Tooling

| Script | Purpose |
|---|---|
| [`scripts/extract_tables.py`](../scripts/extract_tables.py) | The enrichment CLI: `--apply` / `--force` / `--doc`. PDF-only; resolves source PDF + cached `.md`, splices, writes. Reminds you to re-`ingest`. |
| [`scripts/debug_tables.py`](../scripts/debug_tables.py) | **Visual inspector.** Renders per-page PNGs with pdfplumber's detection overlay + a pdfplumber-vs-PyMuPDF count comparison, to `data/tables_debug/{stem}/` (gitignored). Reach for this when detection quality is uncertain — *before* trusting counts. |
| [`scripts/eval_marker_tables.py`](../scripts/eval_marker_tables.py) | Engine eval: emits candidate pages + pdfplumber tables + (if installed) Marker's markdown, for a side-by-side fidelity comparison. Self-contained Marker call; meant for the GPU/RTX machine. |

---

## Library roles

| Library | Role | Kept? |
|---|---|---|
| `pymupdf` / `pymupdf4llm` | default PDF→markdown extractor **and** the `regions.py` classifier signals | core — yes |
| `pdfplumber` | table *cell* extraction | yes, pending the engine eval |
| Marker | removed from the production path (PDF-only, heavy ML, was uninstalled/dead); lives only behind `eval_marker_tables.py` | re-add only if the eval wins |

Non-PDF figure extraction (EPUB/DOCX/HTML) will use **native parsers**
(`ebooklib`, `python-docx`, `BeautifulSoup`) — not Marker, not pdfplumber.

---

## Open questions

1. **Extraction engine.** pdfplumber fragments multi-part tables (split
   Table 1 into two pieces) and misses tables the geometric detector can't
   see. Run `eval_marker_tables.py` on the RTX machine to decide
   Marker-vs-pdfplumber; if Marker wins, reconsider keeping pdfplumber.
2. **Inline de-dup.** `pymupdf4llm`'s lossy inline tables still sit in the
   cache alongside the clean spliced copy. Once the engine is chosen, strip
   the inline copy (regex over the `<!-- page:N -->`-marked region) so there's
   one clean table, not two.
3. **Verification.** Chosen: a table-retrieval eval case (ask a table
   question → assert the table chunk is retrieved post-ingest). Roadmap
   future: a hand-verified gold table set + a deterministic fidelity scorer.
4. **Thresholds** were measured on 2 eLife docs — validate on a wider corpus
   before trusting as universal.
5. **Region-level splitting** (multiple regions on one page) + figure image
   extraction = the proper **Feature 4b** build on top of this classifier.
