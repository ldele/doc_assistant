# Figures & Tables — detection and extraction (Phase 6 / Feature 4)

How doc_assistant promotes tables (and, later, figures) from lossy text into
structured, retrievable content — and, just as important, how it *avoids
mistaking figures for tables*. This is the reference for `regions.py`,
`tables.py`, `tables_marker.py`, and the four `scripts/*table*` tools. Locked decisions live in
[`decisions.md`](decisions.md) (Feature 4a); per-change history in
[`DEVLOG.md`](DEVLOG.md).

> **Status (2026-06-14):** detection + extraction shipped for both tables and
> figures. Marker is the primary table engine (RTX engine eval won 2026-06-02;
> ingest path designed/locked 2026-06-04; splice fidelity measured 2026-06-06);
> pdfplumber is the frozen no-dep fallback. **Feature 4b (figure region detection
> + caption pairing) shipped 2026-06-14** — `regions.py`'s page-level figure
> verdict promoted to region-level bboxes, cropped to PNGs, persisted as a
> `figures` sidecar (no LLM, no GPU). See "Figure extraction (4b)" below.

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
   tables_marker (primary)   (feeds Feature 4b — figures)       (ignored)
   tables.py (fallback)
        │
        ▼
   caption-anchored inline splice into the .md (idempotent, marked, de-duped)
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

### Scope: classifier is page-level; 4b adds region bboxes

The *classifier* aggregates signals per page — a page mixing a table *and* a
chart is labelled by the dominant signal. Per-region bbox extraction is
**Feature 4b** (shipped 2026-06-14), which consumes this classifier's
`is_figure` verdict and image-area signal to locate the figure bbox(es) on each
figure page. See "Figure extraction (4b)" below.

---

## Table extraction — pdfplumber fallback ([`tables.py`](../src/doc_assistant/tables.py))

> This is the **frozen no-dep fallback**; the primary engine is **Marker** (see
> [`tables_marker.py`](../src/doc_assistant/tables_marker.py)), which splices
> page-anchored inline blocks `<!-- table:marker:page=N:begin/end -->` at the caption
> and strips pymupdf4llm's lossy inline twin. The rest of this section describes the
> pdfplumber path verbatim.

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
  the next **`ingest`** that re-reads the cache. Incremental ingest is fine — the orphan
  sweep re-hashes each source and drops the pre-splice (old-hash) chunks
  (`ingest._find_orphan_hashes`). A content change drops that doc's sidecar enrichment,
  so re-run the citation + doc-vector enrichment afterwards.

Validated end-to-end: a paper with no `Table N` caption → 0 tables; a paper
with Table 1 → the real Table 1 data (the prior figure-as-table noise is
gone).

### Tables are indexed whole, with their caption

A spliced table only helps if its values come back at query time. During
ingestion, a detected table and its caption (e.g. *"Table 2: …"*) are kept as
**one retrievable unit** rather than being broken up by the generic text chunker.
Before this, a wide table could be split so its column labels and its numbers
landed in different pieces — a question like *"what is the top-100 accuracy?"*
would match the caption but miss the piece holding the value, so the answer came
back incomplete. Keeping the caption and the table body together means a table's
numbers are retrieved alongside the words people actually search for, so
table-grounded answers are complete. The chunk **sizes** are unchanged (the locked
`2000/200 · 400/50`); only the grouping is table-aware. Guarded by the opt-in
[`cases.tables.yaml`](../tests/eval/cases.tables.yaml).

---

## Figure extraction — region detection (4b)

A post-ingest **enrichment layer**, but — unlike tables — a **sidecar, never
spliced** (ADR-2). Figures are binary: embedding base64 in the markdown destroys
the human-readable cache, and a placeholder without the image is noise. So the
caption text stays in the markdown untouched, and each figure region persists as
a `Figure` row plus a cropped PNG under `data/figures/{doc_hash}/`.
[`figures.py`](../src/doc_assistant/figures.py) is the pure core + thin PyMuPDF
boundary; [`scripts/extract_figures.py`](../scripts/extract_figures.py) is the CLI.

- **Gated by the classifier.** Only `regions.analyze_pages` `is_figure` pages are
  scanned — no second detector.
- **Region geometry is PyMuPDF-native (ADR-1).** Raster image blocks
  (`block.type == 1`) each become a region (`image_block`); a vector-chart page's
  drawing rects merge into one region (`drawing_union`); otherwise the largest
  non-text block (`largest_block`), else a `caption_only` row with `bbox = None`.
  **OpenCV refinement is deferred** — the classifier already does the chart/
  photo/figure discrimination OpenCV was slated for. Degenerate/inverted/sliver
  bboxes are filtered (signed width/height, not `abs()` area) so the PNG writer
  can't fault on a zero-pixel crop; tiny decorative images fall below
  `FIGURE_MIN_AREA_FRACTION`.
- **Caption pairing** (`pair_caption`, pure) matches each region to its nearest
  `Figure N:` text block by vertical gap, below-preferred, no double-assignment.
- **Idempotent, `--force`-gated, per-doc isolated.** A doc with rows is skipped
  unless `--force` (which clears its rows + PNG dir, then re-renders); one bad PDF
  errors its own row and the run continues. The `figures` table is created by the
  additive `init_db()` `create_all` (no Alembic).
- **No re-ingest needed.** Figures are a sidecar — re-ingest only matters once
  Feature 4c turns them into retrievable `chunk_type='figure'` chunks. The
  `vlm_description` / `vlm_call_skipped_reason` columns ship present-but-null for 4c.

Validated on the public corpus (10 papers): 45 regions, 44 PNGs + 1 caption-only,
every region caption-paired; all four extraction methods exercised.

## Figure description — VLM (4c)

The quality lever on top of 4b: a **gated, paid, Anthropic-only** pass that turns
a figure crop into retrievable content. [`scripts/describe_figures.py`](../scripts/describe_figures.py)
calls Claude vision with **forced tool-use** against a Pydantic schema
(`FigureDescription`: `figure_type` / `summary` / `key_quantities` / `axes` /
`trend` — deliberately **no confidence field**), writes the rendered description
to `Figure.vlm_description`, and the next `ingest` materialises
`caption + description` as a `chunk_type='figure'` chunk in both stores.

- **How figure chunks reach retrieval (ADR).** The 4c enrichment writes only the
  `Figure` sidecar; **ingest** — the one component allowed to write the chunk
  store — reads described figures and emits the chunks (`ingest.figure_units`).
  This is the same split as tables (4a writes the markdown, ingest reads it) and
  keeps the Enrichment-Layer "never mutate the chunk store from enrichment" rule
  intact. So the flow is: `extract_figures --apply` → `describe_figures --apply`
  → re-`ingest`.
- **Three cost gates.** Caption-only figures (no PNG) are skipped; a figure whose
  caption already clears `FIGURE_CAPTION_DESC_MIN_CHARS` is skipped as
  self-describing; and `MAX_VLM_CALLS_PER_DOC` caps spend per paper. Every skip
  records an auditable `vlm_call_skipped_reason`
  (`no_image` / `caption_sufficient` / `budget_exhausted` / `image_missing`).
- **Model** is the `FIGURE_VLM_MODEL` knob (default `claude-haiku-4-5` — vision-
  capable and cheapest, matching the reviewer/judge cost convention; bump to
  Sonnet/Opus for higher fidelity). Anthropic-only by decision — no local path.
- **Eval hook.** `eval/scorers.py:FigureRetrievalScorer` answers "given a query
  about a figure, did its figure chunk come back?" (reads the per-chunk
  descriptors the eval adapter now exposes in `EvalOutput.raw["retrieved"]`).

Dry-run on the public corpus (zero cost): 38 of 45 figures eligible to describe,
7 skipped as already well-captioned.

## Tooling

| Script | Purpose |
|---|---|
| [`scripts/extract_tables.py`](../scripts/extract_tables.py) | The enrichment CLI: `--apply` / `--force` / `--doc`. PDF-only; resolves source PDF + cached `.md`, splices, writes. Reminds you to re-`ingest`. |
| [`scripts/extract_figures.py`](../scripts/extract_figures.py) | **The figure (4b) CLI.** `--apply` / `--force` / `--doc` / `--dpi`. PDF-only; detects regions on figure pages, renders PNGs to `data/figures/{doc_hash}/`, writes `Figure` rows. Sidecar — no re-`ingest` needed. |
| [`scripts/describe_figures.py`](../scripts/describe_figures.py) | **The figure-description (4c) CLI.** `--apply` / `--force` / `--doc` / `--max-calls` / `--model`. Gated, **paid** Claude-vision pass; writes `Figure.vlm_description`. Re-`ingest` after to pull figure chunks into retrieval. Dry-run by default; `--apply` needs `ANTHROPIC_API_KEY`. |
| [`scripts/debug_tables.py`](../scripts/debug_tables.py) | **Visual inspector.** Renders per-page PNGs with pdfplumber's detection overlay + a pdfplumber-vs-PyMuPDF count comparison, to `data/tables_debug/{stem}/` (gitignored). Reach for this when detection quality is uncertain — *before* trusting counts. |
| [`scripts/eval_marker_tables.py`](../scripts/eval_marker_tables.py) | Engine eval: emits candidate pages + pdfplumber tables + (if installed) Marker's markdown, for a side-by-side fidelity comparison. Self-contained Marker call; meant for the GPU/RTX machine. |
| [`scripts/extract_tables_marker.py`](../scripts/extract_tables_marker.py) | **The primary table CLI.** Runs isolated Marker (`uvx --from marker-pdf marker_single`) on the caption-gated candidate pages in a bounded pool, parses the paginated markdown, splices inline + de-dups pymupdf4llm's twin, supersedes any pdfplumber block. `--apply` / `--force` / `--doc` / `--workers`. Re-`ingest` after (then re-run enrichment). GPU/RTX box. |

---

## Library roles

| Library | Role | Kept? |
|---|---|---|
| `pymupdf` / `pymupdf4llm` | default PDF→markdown extractor **and** the `regions.py` classifier signals | core — yes |
| **Marker** (isolated, `uvx --from marker-pdf marker_single`) | **primary table engine** (engine eval won 2026-06-02). Run out-of-process by `scripts/extract_tables_marker.py` (parse + page-anchored inline splice in `tables_marker.py`); never imported in-process | yes — primary |
| `pdfplumber` | table *cell* extraction — **frozen no-dep fallback** (lossy on borderless/booktabs; ruled tables only). `scripts/extract_tables.py`; Marker supersedes it | yes — fallback |

Non-PDF figure extraction (EPUB/DOCX/HTML) will use **native parsers**
(`ebooklib`, `python-docx`, `BeautifulSoup`) — not Marker, not pdfplumber.

---

## Open questions

1. **Extraction engine — ✅ RESOLVED (2026-06-02).** Marker won the RTX engine
   eval; the isolated Marker ingest path shipped 2026-06-04 (`tables_marker.py` +
   `scripts/extract_tables_marker.py`). pdfplumber is retained frozen as the no-dep
   fallback (`tables_marker.strip_pdfplumber_block` supersedes its block when Marker runs).
2. **Inline de-dup — ✅ RESOLVED.** `tables_marker.splice_tables_inline` strips
   `pymupdf4llm`'s lossy GFM table(s) within each page span (`_strip_gfm_tables_text`)
   before inserting Marker's block, so there is one clean table, not two.
3. **Verification — ✅ Built.** A table-retrieval eval case
   ([`cases.tables.yaml`](../tests/eval/cases.tables.yaml), `dpr_topk_accuracy_table`)
   plus the CI mechanism gate `tests/integration/test_marker_table_retrieval.py`;
   measured 2026-06-06 (DPR Table 2: Top-20 78.4, Top-100 85.4). Roadmap future: a
   hand-verified gold table set + a deterministic cell-exact fidelity scorer (still deferred).
4. **Thresholds** were measured on 2 eLife docs — validate on a wider corpus
   before trusting as universal.
5. **Region-level splitting + figure extraction — ✅ RESOLVED (2026-06-14).**
   Feature 4b (`figures.py` + `scripts/extract_figures.py`) builds region bboxes
   on this classifier — one region per image block, one merged region per chart.
   True multi-region splitting of a mixed table+figure page beyond that is
   best-effort in v1; OpenCV contour refinement stays the deferred lever (ADR-1).
