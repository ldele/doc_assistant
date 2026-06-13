# Spec — Feature 4a: isolated-Marker table ingest path

**Status:** ✅ SHIPPED 2026-06-04..06 (Claude Code, commit `2933881`). `tables_marker.py` + `extract_tables_marker.py` + tests landed; Marker = primary table engine; caption-anchored inline splice; `cases.tables.yaml` measured green on the RTX/GPU box 2026-06-06. Designed 2026-06-04 (grilled with user, verified against code); retained as the design record.
**Owner of execution:** Claude Code (code + tests).
**Pattern reference:** Enrichment-Layer Pattern (`docs/decisions.md`); mirrors `scripts/extract_tables.py` + `src/doc_assistant/tables.py`.
**Engine decision:** locked 2026-06-02 (RTX eval) — **Marker wins, run isolated.** See `docs/DEVLOG.md` 2026-06-02 and `docs/figures-and-tables.md`. Step 1 (the `eval_marker_tables.py` shell-out) shipped 2026-06-04.

**Requirement:** academic tables (borderless/booktabs) are unreadable to pdfplumber (DPR 0/6, SPECTER2 0/6; SBERT extracts but scrambles rows). Marker reproduces them faithfully but **cannot co-resolve with our torch/langchain stack** → it must run **out-of-process** (`uvx --from marker-pdf marker_single`), never imported. This node wires Marker as a **parallel, post-ingest, idempotent enrichment CLI** that splices high-fidelity tables into the markdown cache, gated to detected table pages.

---

## ADR — Marker as a parallel post-ingest enrichment CLI

**Context.** `tables.py`/`extract_tables.py` already implement the splice mechanics for pdfplumber (an appended `<!-- tables:pdfplumber:begin/end -->` block). pdfplumber is near-unusable on the academic corpus. Marker is the chosen engine but is heavy (multi-GB surya models), slow (~min/doc), and isolation-bound.

**Decision.** Add `src/doc_assistant/tables_marker.py` (parse + page-anchored inline splice) and `scripts/extract_tables_marker.py` (the CLI driver), reusing `regions.table_candidate_pages` to gate Marker to table pages and the `eval_marker_tables.py` helpers (`_marker_command`, `_to_marker_page_range`) to shell out. Marker runs **per document in a bounded process pool** (`MARKER_MAX_WORKERS`). Tables are **spliced inline at their page region** (de-dup + placement in one move), not appended as a block. pdfplumber is **frozen as an explicit fallback**; the Marker pass supersedes it.

**Options considered.**
1. *Inline in `ingest`* — rejected: a `uvx` subprocess per PDF would destroy streaming-ingest ergonomics and couple a slow ML step into the hot path.
2. *Append-block (pdfplumber's mechanism), reused for Marker* — rejected: loses positional placement and forces a separate de-dup pass; inline replacement does both at once.
3. *Auto engine-selection (Marker-if-available, else pdfplumber)* — rejected: silent fallback makes table quality depend on whether `uvx`/marker happens to be installed; explicit two-CLI selection keeps the choice (and provenance) the user's.
4. *Migrate pdfplumber onto the inline splice too* — rejected: pdfplumber is demoted, not co-promoted; refactoring a near-dead path is wasted effort. Freeze it.

**Consequences.** Two-step UX (`ingest` → `extract_tables_marker --apply` → `ingest`), already the documented citations/metadata/tables flow. One new splice mechanism (inline) for Marker; pdfplumber stays on its append-block, untouched. Tables enter retrieval on the next `ingest` that re-reads the cache.

---

## Decisions (from the 2026-06-04 grilling)

| # | Decision |
|---|---|
| 1 | **Separate post-ingest CLI**, not inline in `ingest`. |
| 2 | **Bounded process pool over documents**; `MARKER_MAX_WORKERS` (default 2). One `marker_single` per doc (model load amortized over the doc), `--page_range` = that doc's `table_candidate_pages`. Per-doc isolation (one crash/timeout → that doc `error`, pool continues), per-doc `--output_dir`, reuse `_MARKER_TIMEOUT_S`. |
| 3 | **Per-doc paginated Marker run** (`--paginate_output`) → split by page delimiter → extract contiguous GFM table blocks per page → light meaningfulness filter (reuse the spirit of `tables._is_meaningful`) → **preserve Marker's rich rendering** (`<br>` multi-row cells, bold). |
| 4 | **Page-level locatability ("A")** via the `page=N` splice marker + chunk `page` metadata. The ordered objects-manifest ("B") is a **4b** design note, not built here. |
| 5 | **Caption-anchored inline replacement** (de-dup + placement). For each Marker table on page N: locate the `<!-- page:N -->`…`<!-- page:N+1 -->` span, strip pymupdf4llm's lossy GFM table(s) **within that span only**, and insert Marker's block **right after the `Table N` caption** (the caption is the query magnet — co-locating it with the grid keeps both in one parent), wrapped in idempotent `<!-- table:marker:page=N:begin/end -->` markers. End-of-span is the **no-caption fallback**. This caption-anchoring is the 2026-06-06 splice fix (`tables_marker._place_block_in_span`; `TABLE_BLOCK_RE` consumed by `ingest.build_parent_child_chunks` + `ingest._table_aware_parents`): pre-fix end-of-span placement split caption from grid, stalling `contains_all` at 0.750; post-fix it reaches 1.000. |
| 6 | **Marker primary; pdfplumber frozen as labeled fallback**; explicit two-CLI selection; the Marker pass **strips any pdfplumber block** when it runs (supersede). |
| 7 | Verification = **CI integration test** (mechanism) + **opt-in `cases.tables.yaml`** (post-Marker quality). Deferred: cell-exact gold-table fidelity scorer. |

**Edge cases (spec explicitly):** multiple tables per page (replace each; append leftovers within the page span); Marker-found-but-pymupdf-rendered-nothing (insert at end of page span); pages Marker did *not* process (leave pymupdf4llm's inline table alone — lossy beats absent); idempotent re-run (the `<!-- table:marker:* -->` wrapper regex strips + replaces, like `tables._BLOCK_RE`).

**Build-time confirmations (shell-out, so verify against the marker-pdf version on the box):** the `--paginate_output` flag name + the page-delimiter regex. Note: the marker-pdf version is **intentionally not pinned** — `uvx` fetches the latest on demand; only the interpreter is pinned (`MARKER_PYTHON=3.12`). The CLI docstring records this as a per-machine "confirm the delimiter/version" caveat rather than a recorded pin (still outstanding from build-time confirmations).

---

## Contract — `src/doc_assistant/tables_marker.py` (new)

Pure-ish; the only impure boundary is the subprocess call (delegated to the `eval_marker_tables` helpers).

- `parse_marker_tables(marker_markdown: str, page_numbers: list[int]) -> list[MarkerTable]` — split paginated Marker markdown by page delimiter, extract GFM table blocks per page, light-filter. **Shipped with a second positional `page_numbers` arg** (the ordered 1-based candidate pages passed to Marker via `--page_range`); page attribution comes from this requested page order, not from parsing the delimiter's page id. Returns `MarkerTable`, which carries the raw rich-markdown string (preserves `<br>`/bold verbatim — no round-trip through `rows`).
- `splice_tables_inline(markdown: str, tables: list[...], *, engine: str = "marker") -> str` — caption-anchored inline replacement (block placed right after the `Table N` caption; end-of-span fallback when no caption) using the `PAGE_MARKER` regex (import from `ingest`, or lift the pattern). Idempotent via the `<!-- table:{engine}:page=N:begin/end -->` wrapper.
- `strip_marker_tables(markdown) -> str` / `has_marker_tables(markdown) -> bool` — round-trip + idempotency helpers (mirror `tables.py`).
- `strip_pdfplumber_block(markdown) -> str` — reuse `tables.strip_spliced_tables` so the Marker pass supersedes pdfplumber.

**NOT responsible for:** running Marker (that's the CLI via `_marker_to_markdown`), the chunk store, retrieval.

## Contract — `scripts/extract_tables_marker.py` (new)

Mirrors `extract_tables.py`: `--apply` / `--force` / `--doc`, plus `--workers` (overrides `MARKER_MAX_WORKERS`). Reuses `_resolve_cache_path` / `_resolve_pdf_path` from `extract_tables`. Per doc: `regions.table_candidate_pages(pdf)` → if empty, skip; else `_marker_to_markdown(pdf, pages, out_dir)` → `parse_marker_tables` → `strip_pdfplumber_block` → `splice_tables_inline` → write cache (only on `--apply`). Pool via `concurrent.futures` bounded by workers; per-doc try/except; same report shape as `extract_tables`.

## config.py additions

- `MARKER_MAX_WORKERS = int(os.getenv("MARKER_MAX_WORKERS", "2"))` — documented "raise on big-VRAM/RAM, drop to 1 on OOM."
- `MARKER_PYTHON = os.getenv("MARKER_PYTHON", "3.12")` — interpreter `uvx` resolves the isolated Marker env against; 3.12 has wheels for the whole marker-pdf stack (newer defaults force a from-source `pillow` build that fails). Only the `uvx` path uses it (an on-PATH `marker_single` is used as-is).

---

## Build node

**Depends on:** Feature 4a step 1 (`eval_marker_tables.py` helpers — done). Independent of Chunk 2a.
**Files owned:** `src/doc_assistant/tables_marker.py` (new), `scripts/extract_tables_marker.py` (new), `src/doc_assistant/config.py` (added `MARKER_MAX_WORKERS` **and** `MARKER_PYTHON`), `tests/unit/test_tables_marker.py` (new), `tests/integration/test_marker_table_retrieval.py` (new), `tests/eval/cases.tables.yaml` (new), `docs/figures-and-tables.md` (engine table updated). **Note:** neither Marker var was added to `.env.example` (its only `marker` line is the unrelated `PDF_EXTRACTOR=marker`) — add them or drop `.env.example` from this list.
**Status:** ✅ done (2026-06-06). All owned files shipped; unit + integration tests green; `cases.tables.yaml` measured green on the RTX box.

### Unit test — `tests/unit/test_tables_marker.py`
Parse a fixed paginated-Marker markdown fixture → assert per-page table extraction + rich-formatting preservation (`<br>`/bold kept). Splice idempotency (`splice == splice∘splice`), page-anchored replacement strips the lossy twin within the page span only, leaves other pages untouched, and `strip_pdfplumber_block` removes a pdfplumber block. No subprocess (fixture string in).

### Integration test (CI gate) — `tests/integration/test_marker_table_retrieval.py`
Fixture markdown with a `<!-- table:marker:… -->` block containing a known value → ingest into a temp Chroma → query whose answer is that value → assert the **table chunk is retrieved** and the value surfaces. Deterministic, **no Marker/corpus dependency**. This is the regression gate for the splice→re-ingest→retrieval path.

### Opt-in eval — `tests/eval/cases.tables.yaml`
1–2 cases anchored on a known public-corpus table (e.g. DPR Table 2 Top-20/100 accuracy). Documented: run `extract_tables_marker --apply` + re-ingest first. Runs via the existing harness (`contains_all` on the value). **Not** part of the one-command-reproducible headline public eval.

## Definition of done
- `extract_tables_marker --apply` splices inline, idempotent, page-anchored, de-duped against pymupdf4llm and pdfplumber; parallel with per-doc isolation.
- Unit + integration tests green; ruff / mypy --strict / bandit clean.
- `cases.tables.yaml` runs green after a real Marker pass on the RTX box (manual gate).
- `docs/figures-and-tables.md` engine table updated (Marker = primary; pdfplumber = frozen fallback).

## Out of scope (4b / deferred)
Ordered objects-manifest ("B": `DocumentPart`/`DocumentObject` with `kind`, `page`, `order_index` for tables **and** figures) — design in **4b**, building on this node's page-level locatability. Cell-exact gold-table fidelity scorer — deferred.
