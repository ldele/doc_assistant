# Spec — Feature 4b: figure region detection + caption pairing (sidecar manifest)

**Status:** ✅ BUILT (Claude Code, 2026-06-14) — `src/doc_assistant/figures.py` + `scripts/extract_figures.py` + `Figure` table; gate green (389 tests), validated on the public corpus (45 regions / 44 PNGs, all caption-paired). Designed by Cowork 2026-06-13. Roadmap PR 8.
**Owner of execution:** Claude Code (code + tests).
**Pattern reference:** Enrichment-Layer Pattern (`docs/decisions.md`) — post-ingest, idempotent, **sidecar** (figures are binary → never spliced into the markdown). Mirrors `scripts/extract_tables.py` + `src/doc_assistant/tables.py` / `doc_vectors.py`.
**Foundation (shipped):** the page content classifier `src/doc_assistant/regions.py` already discriminates `chart` / `photo` / `figure` / `table` / `text` per page and exposes `is_figure` + the raster `image_area_fraction` signal. 4b promotes that **page-level** verdict to **region-level**: find the figure bbox(es) on each figure page, pair each with its caption, render a PNG, and persist a `Figure` sidecar row. This is the deeper step the `regions.py` and 4a docstrings call "the proper Feature 4b build."

**Requirement.** A large share of a scientific paper's signal lives in figures and charts, and the primary extractor drops them entirely (`write_images=False`). 4b makes each figure a **first-class, addressable sidecar record** — page, bbox, caption, on-disk PNG — without touching primary ingest or the chunk store. Caption-only retrieval is unchanged (captions stay in the markdown); the `Figure` table is the substrate Feature 4c (PR 9) turns into VLM-described, retrievable figure chunks. **4b is independently shippable**: caption-paired figure records are useful on their own, and the VLM columns ship present-but-null.

**Cost & placement.** No LLM, no Marker, no torch — pure PyMuPDF geometry. Unlike 4a (GPU/Marker, RTX box only), **4b runs on either machine**, including the CPU box.

---

## ADR-1 — Region geometry source: PyMuPDF-native (v1), OpenCV deferred

**Context.** The roadmap names OpenCV for "region boundary refinement + chart-like region detection." But the chart/photo/figure *discrimination* OpenCV was slated for is **already solved** by `regions.py`'s measured signals (curve density, raster area fraction). The remaining job for 4b is *bbox extraction*, not classification.

**Decision.** v1 derives figure region bboxes from **PyMuPDF geometry only**:
- **Raster figures** (`kind ∈ {photo}`, or any page with `block.type == 1` image blocks): each raster image block's `bbox` is a figure-region candidate (`regions.page_signals` already iterates these to sum `image_area`).
- **Vector charts** (`kind == "chart"`): no image block exists; the chart is drawn paths. Region bbox = the union (bounding rect) of the page's vector drawing items (`page.get_drawings()` item rects), clipped to the page and filtered to the dominant cluster.
- **`figure`-captioned pages with neither dominant signal**: fall back to the page's largest non-text block, else record a caption-only figure with `bbox = None` and `extraction_method = "caption_only"`.

OpenCV contour refinement is an **optional, deferred lever** (ADR-1 reconsidered only if a measured precision gap appears) — it would not add a new dependency to the hot path in v1.

**Options considered.**
1. *PyMuPDF-native geometry (chosen).* No new dependency; reuses the exact signals already measured and unit-tested in `regions.py`. Bbox precision is "good enough to crop a readable PNG," which is all 4c needs.
2. *OpenCV contour detection as the primary region finder (rejected for v1).* Adds `opencv-python` (a heavy wheel) to the core env for a refinement the classifier already largely covers. Defer until the fixture/corpus shows PyMuPDF bboxes crop figures poorly.
3. *Marker / layout-model region detection (rejected).* Out-of-process, GPU-bound, slow — the 4a isolation tax for a job cheap geometry handles. Keeps 4b off the GPU box.

**Consequences.** v1 ships with zero new heavy deps; `opencv-python` stays out of `pyproject.toml`. Bbox quality is bounded by PyMuPDF block/drawing geometry — acceptable for cropping; flagged as the place to revisit if 4c's VLM struggles on crops. A page mixing a table and a figure is handled by the classifier's dominant-signal verdict (page-level); true multi-region splitting on one page is best-effort in v1 (one region per image block; one merged region per chart).

## ADR-2 — Sidecar manifest, never spliced (locked, restated)

Figures are binary. Embedding base64 in the markdown destroys the human-readable cache (the "markdown as universal intermediate" decision); placeholder strings without the image are noise. So figures persist as a **`Figure` SQLite sidecar row + a PNG on disk under `data/figures/{doc_hash}/`**, and the **caption text stays in the markdown untouched** (figures are additive, not substituting). This is the Enrichment-Layer "sidecar by default" rule — tables were the one sanctioned splice exception because they are text-shaped; figures are not.

---

## Decisions

| # | Decision |
|---|---|
| 1 | **Separate post-ingest CLI** (`scripts/extract_figures.py`), not inline in `ingest`. Same two-step UX as citations/tables: `ingest` → `extract_figures --apply`. (4b writes a sidecar + PNGs; **no re-ingest needed** for the records themselves — re-ingest only matters once 4c emits figure *chunks*.) |
| 2 | **Gate to figure pages via `regions.analyze_pages`** — only pages with `is_figure` are scanned. Reuses the shipped, measured classifier; no second detector. |
| 3 | **Region bbox from PyMuPDF geometry** (ADR-1): raster image blocks (`type == 1`) for photos; drawing-item bbox union for charts; largest-block / caption-only fallback otherwise. |
| 4 | **Caption pairing = nearest-caption heuristic.** For each region, find the text block matching `regions.FIGURE_CAPTION_RE` whose bbox is vertically nearest the region (caption typically *below* the figure; allow above). The whole caption block's text is stored. Region with no matching caption → `caption = None`. Multiple figures/page → each region paired to its nearest unused caption. |
| 5 | **Render a PNG per region** at `FIGURE_RENDER_DPI` (default 150) via `page.get_pixmap(clip=bbox, dpi=…)` → `data/figures/{doc_hash}/page{N}_fig{M}.png`. `M` is the 0-based region index in reading order on the page (stable filename → idempotent). Caption-only figures (`bbox = None`) write no PNG (`image_path = None`). |
| 6 | **`Figure` sidecar table** carries the 4c columns present-but-null: `{id, document_id, doc_hash, page, bbox_x0/y0/x1/y1, kind, caption, image_path, extraction_method, vlm_description=None, vlm_call_skipped_reason=None, extracted_at}`. Created by the additive `init_db()` `create_all` (see "Migration" below). |
| 7 | **Idempotent, `--force`-gated.** A doc with existing `Figure` rows is skipped unless `--force`; `--force` deletes that doc's rows **and** its `data/figures/{doc_hash}/` PNGs, then re-extracts. Re-running without `--force` is a no-op. |
| 8 | **No chunk-store mutation, ever** (Enrichment-Layer rule; guard-tested). 4b does not touch Chroma, the markdown cache, or `Document` columns. The caption stays where ingest already put it. |

**Edge cases (spec explicitly):**
- *Page with a figure caption but no detectable region* → one caption-only `Figure` row (`bbox=None`, `image_path=None`, `extraction_method="caption_only"`). Still useful to 4c (caption-only chunk baseline).
- *Multiple regions on one page* → one row + one PNG each, indexed `fig0..figN` in reading order; captions matched nearest-first.
- *Region with no nearby caption* (decorative/inline image) → `caption=None`; gate tiny decorative images out via `FIGURE_MIN_AREA_FRACTION` (don't emit a row for a logo/icon).
- *Content-hash drift* → like citations/tables: a markdown/content change drops the doc's sidecar enrichment; re-run `extract_figures` after re-ingest. The `doc_hash` column makes a stale row detectable.
- *Non-PDF source* (EPUB/DOCX/HTML) → **out of scope v1**; skip with a recorded reason. Native-parser figure extraction is a later step.

**Build-time confirmations (verify against the PyMuPDF on the box):**
- The drawings API shape (`page.get_drawings()` item rects) and `get_pixmap(clip=…, dpi=…)` signature — used but not yet exercised for cropping in-repo. Confirm `clip` accepts a `pymupdf.Rect` and `dpi` is honored (older PyMuPDF used a `matrix=` zoom instead).
- The exact data-root var to hang `FIGURE_DIR` off (`config.DATA_PATH`; `CHROMA_PATH`/`SQLITE_PATH` are built from it).

---

## Contract — `src/doc_assistant/figures.py` (new)

Pure core + a thin impure PyMuPDF boundary, mirroring the `regions.py` split (pure `classify_page` vs impure `analyze_pages`).

- `@dataclass FigureRegion` — `{page: int, bbox: tuple[float,float,float,float] | None, kind: RegionKind, caption: str | None, caption_bbox: tuple | None, extraction_method: str}`. Pure data.
- `pair_caption(region_bbox, caption_blocks) -> tuple[str, tuple] | None` — **pure**, exhaustively unit-tested. `caption_blocks` is the list of `(text, bbox)` for blocks matching `FIGURE_CAPTION_RE`; returns the nearest (vertical-gap metric, below-preferred) caption + its bbox, or `None`. The heart of the feature; no I/O.
- `figure_image_path(doc_hash, page, index) -> Path` — **pure**; `FIGURE_DIR / doc_hash / f"page{page}_fig{index}.png"`. Stable → idempotent.
- `detect_figure_regions(pdf_path) -> list[FigureRegion]` — **impure** (opens the PDF). Pipeline: `regions.analyze_pages(pdf_path)` → for each `is_figure` page, collect region bboxes (image blocks ∪ drawing-bbox union, ADR-1) filtered by `FIGURE_MIN_AREA_FRACTION`, collect caption blocks, call `pair_caption` per region. Returns regions in (page, reading-order) order.
- `render_region(page, bbox, out_path, *, dpi) -> None` — **impure**; `page.get_pixmap(clip=Rect(bbox), dpi=dpi).save(out_path)`; `out_path.parent.mkdir(parents=True, exist_ok=True)`.

**NOT responsible for:** the VLM call (4c), the chunk store, retrieval, OpenCV refinement (ADR-1 deferred), non-PDF sources.

## Contract — `scripts/extract_figures.py` (new)

Mirrors `scripts/extract_tables.py` exactly: reuse `_resolve_cache_path` / `_resolve_pdf_path` (import or lift), `_run_one`, `_format_report`, `main() -> int`. Flags: `--apply` (default = dry-run report), `--force`, `--doc <hash|id-prefix>`, plus `--dpi` (overrides `FIGURE_RENDER_DPI`). Per doc: resolve source PDF (skip non-PDF with a reason) → if existing rows and not `--force`, skip → `detect_figure_regions(pdf)` → on `--apply`: `render_region` each non-null bbox, upsert `Figure` rows in one transaction via **`db.session.session_scope()`** (the context manager `doc_vectors.py` / `compute_doc_vectors.py` use for sidecar writes — mirror it, not the lower-level `get_session`); on dry-run: print region/caption counts. Same report shape and "Dry run. Pass --apply to write." footer as `extract_tables`.

## Contract — `src/doc_assistant/db/models.py` (Figure table, additive)

New `class Figure(Base)`, sidecar (mirrors `DocSimilarity` / `Citation` conventions):
- `id` (uuid PK), `document_id` FK → `documents.id` `ondelete="CASCADE"` (indexed), `doc_hash` (indexed, for drift detection),
- `page: int`, `bbox_x0/y0/x1/y1: float | None`, `kind: str` (the `regions` verdict), `caption: Text | None`, `image_path: str | None`, `extraction_method: str | None`,
- `vlm_description: Text | None = None`, `vlm_call_skipped_reason: str | None = None` (4c populates),
- `extracted_at: datetime = _utcnow`.
- Add `figures: Mapped[list["Figure"]] = relationship(..., back_populates="document", cascade="all, delete-orphan")` on `Document`.
- `__table_args__`: index on `(document_id)` and `(doc_hash)`.

**Migration.** There is **no Alembic** — `db/migrations.py:init_db()` calls `Base.metadata.create_all(engine)`, which is additive and idempotent. Adding the `Figure` class **is** the migration; a one-line `python -m doc_assistant.db.migrations` (or `init_db()` on next app start) creates the table. No hand-written migration needed. (Do **not** add a destructive `reset`.)

## config.py additions

- `FIGURE_DIR = DATA_PATH / "figures"` — sidecar PNG root (alongside `chroma/`, `library.db`).
- `FIGURE_RENDER_DPI = int(os.getenv("FIGURE_RENDER_DPI", "150"))` — crop resolution; "raise for VLM-quality crops, lower to save disk."
- `FIGURE_MIN_AREA_FRACTION = float(os.getenv("FIGURE_MIN_AREA_FRACTION", "0.02"))` — skip decorative images/icons below this page-area fraction (distinct from `regions.IMAGE_AREA_MIN=0.05`, the page-dominance threshold; this is the per-region floor).

## .gitignore

Add `data/figures/` (binary artifacts — never committed, same as the already-ignored `data/tables_debug/`).

---

## Build node

**Depends on:** PR 7 / Feature 4a (`regions.py` classifier — **shipped**). Independent of torch, Marker, Chunk 2a/2b. No GPU.
**Files owned:**
- `src/doc_assistant/figures.py` (new)
- `scripts/extract_figures.py` (new)
- `src/doc_assistant/db/models.py` (`Figure` class + `Document.figures` relationship)
- `src/doc_assistant/config.py` (`FIGURE_DIR`, `FIGURE_RENDER_DPI`, `FIGURE_MIN_AREA_FRACTION`)
- `.gitignore` (`data/figures/`)
- `tests/unit/test_figures.py` (new)
- `tests/integration/test_figures_extract.py` (new)
- `tests/fixtures/` (one tiny captioned-image PDF fixture — generate with PyMuPDF in a conftest helper, don't commit a large binary)
- `docs/figures-and-tables.md` (4b: page-level → region-level shipped), `docs/decisions.md` (Feature 4b bullet → ✅), `docs/DEVLOG.md` (entry per logical change)

### Unit test — `tests/unit/test_figures.py`
Pure functions, no PDF:
- `pair_caption`: caption below region (chosen), caption above (chosen when nearer), two regions + two captions (nearest-first, no double-assignment), region with no caption → `None`. Vertical-gap metric exercised at the boundary.
- `figure_image_path`: stable path for `(doc_hash, page, index)`; distinct indices/pages differ; same inputs identical (idempotency substrate).
- Region filtering: `FIGURE_MIN_AREA_FRACTION` drops a sub-threshold block, keeps an over-threshold one (feed synthetic block dicts).

### Integration test (CI gate) — `tests/integration/test_figures_extract.py`
Build a tiny fixture PDF in-test (PyMuPDF: one page, an inserted raster image + a `Figure 1: …` caption line). Assert:
- `detect_figure_regions` finds exactly 1 region with the caption paired.
- `extract_figures --apply` (invoked via its `main`/`_run_one`) writes one PNG under a temp `FIGURE_DIR` and one `Figure` row with the right `page`/`caption`/`image_path`.
- **Idempotency:** a second run without `--force` is a no-op (no new rows, PNG unchanged); `--force` re-renders.
- **Enrichment guard:** Chroma / markdown cache / `Document` rows are untouched (assert no chunk-store writes — the Enrichment-Layer invariant).
Deterministic, no corpus or network dependency.

## Definition of done
- `extract_figures --apply` writes `Figure` rows + PNGs under `data/figures/{doc_hash}/`, caption-paired, gated to figure pages by `regions.py`; idempotent; `--force` re-renders; per-doc isolation (one bad PDF → that doc errors, run continues).
- No chunk-store / markdown / `Document` mutation; captions stay in the markdown (sidecar invariant, guard-tested).
- `Figure` table created by additive `create_all`; `data/figures/` gitignored.
- Unit + integration tests green; `ruff` / `mypy --strict` / `bandit` clean; coverage floor held.
- `docs/figures-and-tables.md` 4b section updated (region-level shipped); `decisions.md` Feature 4b → ✅; one `DEVLOG.md` entry per logical change.

## Out of scope (4c / PR 9 / deferred)
- **VLM figure description** (populates `vlm_description` / `vlm_call_skipped_reason`), figure **chunks** (`chunk_type='figure'`, `caption + VLM description` embedded), `MAX_VLM_CALLS_PER_DOC` budget — all Feature 4c.
- **Figure-retrieval eval scorer** (held-out caption → retrieves the right figure) — add to the eval harness with 4c, when there's a retrievable figure chunk to score.
- **OpenCV contour refinement** (ADR-1) — deferred lever, no new dep until a measured bbox-quality gap.
- **Non-PDF figure extraction** (EPUB/DOCX/HTML via `ebooklib` / `python-docx` / `BeautifulSoup`) — later.
- **Per-region splitting of mixed table+figure pages** beyond one-region-per-block / one-merged-chart — the classifier's page-level dominant-signal verdict stands for v1.
