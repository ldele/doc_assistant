# Spec — PR-M1: live 7d epistemics-marker surfacing (contested / superseded-trend)

**Status:** 📋 PLANNED — designed by Cowork 2026-06-21 (Tauri migration). Second PR of the migration (M1); the **pre-migration demo win**. **Depends on PR-M0** (`ChatController`, `SourceView`, `RetrievedChunk.chunk_key`).
**Owner of execution:** Claude Code (code + tests).
**Pattern reference:** the read-side seam the 7d engine deliberately left open. `epistemics.markers_for_chunk_keys` / `load_epistemics_index` (epistemics.py:178, 248) exist and are documented as *"the read-side seam the live evidence layer will call once a stable chunk key is plumbed through retrieval (deferred)."* This PR closes that seam. The 7d engine, `chunk_epistemics` sidecar, and `reviewer.contested_evidence` tag already shipped (SESSION baton 2026-06-17, `4add13a` "Knowledge Currency").
**Migration context:** `docs/decisions/ADR-002-tauri-fastapi-desktop-shell.md` (Execution § — why M1 lands before the migration proper), PR-M1.

**Requirement (the why).** The 7d engine computes, per chunk, whether its claims sit on **contested** or **superseded-trending** concept nodes — and persists that to `chunk_epistemics`. But nothing surfaces it at answer time: the markers are computed and stored, then never shown. This PR joins the retrieved chunks against that sidecar in `ChatController` and attaches the markers to `SourceView`, so an answer can show a quiet "⚠ contested in your corpus" chip on the relevant source. It is the highest-value unbuilt feature that is **unique to this project's integrity angle**, and (per the sequencing analysis) it shares the exact `chunk_key` plumbing the migration needs — so it lands here, once, on the new `ChatController`, never retrofitted into Chainlit.

**Cost & placement.** Read-only, **free**, no LLM call (the markers are precomputed; this is a dict join). No torch/GPU. Honors the credit-leak guard trivially (no provider touched). Runs on either machine. **Synthesis output is byte-identical when markers are absent** → the eval harness is unaffected (a hard requirement, guard-tested).

---

## ADR-1 — The PC→baseline chunk-key mapping (the real decision M0 deferred)

**Context (the coupling M0 flagged).** Live retrieval runs in **parent-child mode by default** (`USE_PARENT_CHILD`). The two Chroma collections are **independent segmentations of the same document**:
- **baseline/flat** (`db`): `splitter.split_text` → `chunk_index = i` (ingest.py:499). **Epistemics indexed only this** (`load_doc_chunks`, epistemics.py:211–217) and keyed rows `{document_id}:{chunk_index}` (epistemics.py:263).
- **parent-child** (`pc_db`): `build_parent_child_chunks` → children carry `parent_index` + `child_index`, **no `chunk_index`** (ingest.py:443–448); the reranked result is the **parent** doc (pipeline.py:185–189).

So a retrieved PC parent has **no `chunk_index`**, and there is **no 1:1 id** between a parent and a baseline chunk — they share only `document_id` and overlapping text spans. M0 therefore sets `chunk_key=None` for PC chunks and hands this mapping to M1. This ADR picks the mapping.

**Decision.** Map a retrieved **PC parent → the baseline chunk_keys whose text it covers**, via **text containment at marker-join time** (no re-ingest, no new sidecar):
1. `load_epistemics_index()` already returns `{document_id:chunk_index → markers}` — only the chunks that *carry* a marker (small set; quiet-on-clean).
2. Build a **per-document marked-chunk lookup** once per turn: for each marked `{document_id}:{chunk_index}`, fetch that baseline chunk's `text` (from the baseline collection, or carry it in the index — see Decision 4).
3. For each retrieved PC parent (which carries `document_id` + `parent_text`/`page_content`), a baseline marked-chunk **belongs to it** if the marked chunk's text is contained in (or overlaps a threshold with) the parent's text. Attach that chunk's markers to the parent's `SourceView`.
4. Flat-mode retrieval (`USE_PARENT_CHILD=False`) is the trivial case: the retrieved chunk *is* a baseline chunk with `chunk_index` → direct `markers_for_chunk_keys([chunk_key])`.

**Options considered.**
1. *Text-containment mapping at join time (chosen).* No re-ingest, no schema change; uses data already in hand (parent text + the small marked-chunk set). Cost: a substring/overlap check per (retrieved parent × marked chunk in that doc) — tiny, because marked chunks are few and scoped per `document_id`. Imprecise at parent boundaries, but markers are an **advisory chip**, not a gate — over-attribution within a parent is acceptable and fail-safe (it points the user at a real contested concept in that passage).
2. *Re-project epistemics against the PC store (rejected for M1).* Re-key `chunk_epistemics` to `(document_id, parent_index)` by re-running attribution over PC parents. Cleaner joins, but it's a **substantive epistemics change** (a second projection + a migration + its own validation of attribution quality on parents) — too heavy for the pre-migration win, and it duplicates the sidecar. Park as a future refinement if containment proves too coarse.
3. *Emit a baseline `chunk_index` onto PC children at ingest (rejected).* Would require threading the flat index through `build_parent_child_chunks` and a re-ingest of the whole corpus; couples ingest to the epistemics key. Out of proportion.
4. *Disable PC for marked answers (rejected).* Changes retrieval behavior to fit a display feature — backwards.

**Consequences.** M1 ships a working chip in the **default PC configuration** without touching ingest, the chunk store, or the eval path. The mapping is admittedly coarse at parent edges; that is **named as a known limitation** (`KNOWN_ISSUES` candidate if it misfires) and the precise re-projection (option 2) is the documented upgrade. The containment check is pure and unit-testable.

## ADR-2 — Quiet-on-clean, advisory, byte-identical-when-absent

**Context.** The project's rules: *inform, never block* (no gating on markers); *quiet on clean answers* (don't clutter); *don't change synthesis* (eval comparability). The 7d engine already enforces quiet-on-clean (a clean chunk gets no marker; unique-source is neutral — epistemics.py:95–107).

**Decision.** Markers attach to `SourceView.markers` (a `list[str]`, empty when none). The renderer shows a chip **only** when non-empty. The **answer text / synthesis is untouched** — markers are presentation metadata on the source, not injected into the prompt or the generated answer. When the epistemics sidecar is absent/empty (no `build_concept_graph` + `compute_epistemics` run), `markers` is uniformly empty and the turn is **byte-identical** to today.

**Consequences.** Eval harness output is unchanged (guard-tested: a turn with markers-off vs the current code produce the same `TurnResult.answer` + same `sources_md`). The feature is purely additive and degrades to a no-op without the sidecar.

---

## Decisions

| # | Decision |
|---|---|
| 1 | **Join lives in `ChatController.handle_message`** (the RAG path, after `_build_retrieved_chunks`), not in Chainlit/FastAPI — so every frontend gets it for free. Read-only; no LLM; honors the credit guard by not touching a provider. |
| 2 | **`SourceView.markers: list[str]`** added (M0 left the comment placeholder). Values from `epistemics` constants: `"contested"` / `"superseded_trend"`. Empty = clean (no chip). |
| 3 | **Flat-mode join is direct:** `chunk_key` (set by M0 for baseline chunks) → `markers_for_chunk_keys([keys], load_epistemics_index())`. |
| 4 | **PC-mode join via text containment (ADR-1):** load the marked baseline chunks for the retrieved `document_id`s, attach a marked chunk's markers to a retrieved parent when the parent text contains the marked chunk's text (or overlaps ≥ a threshold). Implement the containment test as a **pure function** `markers_for_parent(parent_text, marked_chunks) -> list[str]`. The marked-chunk **text** is needed for the test — extend the read side to return text (Decision 5). |
| 5 | **Read side extended to carry text:** add `load_marked_chunks(document_ids) -> dict[str, list[MarkedChunk]]` to `epistemics.py` (impure; reads the `chunk_epistemics` rows joined to the baseline chunk text from Chroma, scoped to the given docs). `MarkedChunk = {chunk_index, text, markers}`. Reuses `load_doc_chunks`'s Chroma read pattern (epistemics.py:211). Keep `load_epistemics_index` for the flat path. |
| 6 | **Caching within a turn:** call the read side **once** per turn (not per source). The marked set is small (quiet-on-clean) and scoped to the retrieved docs. No cross-turn cache in v1 (correctness over speed; the sidecar can change between turns after a re-`compute_epistemics`). |
| 7 | **Renderer chip (both frontends for now):** Chainlit shows markers as a quiet inline tag on the source line (parity with how it shows the citation-check block — present only when non-clean). The chip text is a short human label (e.g. `⚠ contested in corpus`, `⚠ trend superseded`). The rich Tauri rendering is PR-M3. |
| 8 | **Synthesis untouched (ADR-2):** no change to `prompts.py`, `synthesis.py`, or the answer stream. Markers ride on `SourceView` only. |

**Edge cases (spec explicitly):**
- *Sidecar absent / empty* (no graph or no `compute_epistemics` run) → `load_epistemics_index` / `load_marked_chunks` return empty → every `markers` empty → byte-identical turn. **The default state on a fresh checkout** (per KI-7, `data/graph/` is gitignored and may be stale/absent) — so the feature must be a clean no-op, not an error.
- *KI-7 caveat* — the concept graph the markers derive from is the **superseded open-vocabulary build**; 8B-extracted polarity is noisy (the baton notes inflated `contested`). **This PR surfaces whatever the sidecar holds; it does not fix marker *quality*.** State plainly in the chip's docs that markers reflect the current (imperfect) graph; quality improves when the graph redesign (KI-7) lands. Do **not** block M1 on the redesign.
- *Figure chunks* (`chunk_type='figure'`) have no epistemics row (`load_doc_chunks` skips them) → no marker. Correct.
- *A parent containing multiple marked chunks* → union their markers (dedup). *A marked chunk spanning two parents* → both get it (over-attribution, fail-safe per ADR-1).
- *`document_id` missing on a retrieved doc* → no key, no marker (matches M0's `chunk_key=None`).

**Build-time confirmations:**
- `markers_for_chunk_keys(chunk_keys, index)` signature (epistemics.py:178) — it takes the index as an arg (pure); the caller loads the index. Confirm.
- `load_epistemics_index()` reads `chunk_epistemics` and returns `{document_id:chunk_index → markers}` (epistemics.py:248–264). Confirm the key string matches what M0 emits for flat chunks.
- The baseline collection name accessor (`embeddings.get_collection_name`, used by `load_doc_chunks`) for the text fetch in `load_marked_chunks`.

---

## Contract — `src/doc_assistant/epistemics.py` (edit, additive)

- `@dataclass MarkedChunk` — `{chunk_index: int, text: str, markers: list[str]}`. Pure data.
- `markers_for_parent(parent_text: str, marked: list[MarkedChunk], *, min_overlap: int = …) -> list[str]` — **pure**, the ADR-1 containment test. Returns the deduped union of markers whose `MarkedChunk.text` is contained in `parent_text` (start with substring containment; the `min_overlap` lever is for partial-overlap tuning, deferred unless needed). Exhaustively unit-tested.
- `load_marked_chunks(document_ids: list[str]) -> dict[str, list[MarkedChunk]]` — **impure**; for the given docs, read `chunk_epistemics` rows with markers (reuse the `load_epistemics_index` query, filtered to `document_ids`) and fetch each row's baseline chunk text from Chroma (reuse `load_doc_chunks`'s client/collection pattern, filtered by `document_id` + `chunk_index`). Returns `{document_id → [MarkedChunk]}`. Empty when the sidecar/graph is absent.
- Keep `load_epistemics_index` + `markers_for_chunk_keys` as-is (the flat path uses them).

**NOT responsible for:** marker *quality* (graph redesign, KI-7), re-projecting epistemics onto PC parents (deferred option 2), any synthesis/prompt change.

## Contract — `src/doc_assistant/chat_controller.py` (edit)

- Add `markers: list[str] = field(default_factory=list)` to `SourceView` (M0 reserved the slot).
- In `handle_message` RAG path, after building the `SourceView`s: load the epistemics data **once** for the retrieved `document_id`s, then for each source —
  - flat chunk (has `chunk_key`): `markers = index.get(chunk_key, [])`.
  - PC parent (no `chunk_key`): `markers = markers_for_parent(parent_text, marked_by_doc.get(document_id, []))`.
- Markers ride on `SourceView`; **no change** to `answer`, `sources_md` content rules beyond optionally appending the chip text, or any other `TurnResult` field. (If the chip is rendered in `sources_md`, gate it so a clean turn's `sources_md` is unchanged — ADR-2 byte-identity.)

## Contract — `apps/chainlit_app.py` (edit, renderer only)

- When a `SourceView.markers` is non-empty, render a quiet inline chip on that source's line (mirror the existing "Citation check" conditional block — present only when non-clean). No business logic; reads the field M1 populated.

---

## Build node
**Depends on:** PR-M0 (`ChatController`, `SourceView`, `RetrievedChunk.chunk_key`). The 7d engine + `chunk_epistemics` + `epistemics.py` seam are **shipped**. Independent of torch/GPU; free.
**Files owned:**
- `src/doc_assistant/epistemics.py` (`MarkedChunk`, `markers_for_parent`, `load_marked_chunks` — additive)
- `src/doc_assistant/chat_controller.py` (`SourceView.markers` + the join in `handle_message`)
- `apps/chainlit_app.py` (chip render — renderer only)
- `tests/unit/test_epistemics.py` (extend — `markers_for_parent`)
- `tests/unit/test_chat_controller.py` (extend — markers attached flat + PC; empty when sidecar absent)
- `tests/integration/test_turn_parity.py` (extend — **byte-identical when markers absent**)
- `docs/specs/feature-7d-knowledge-currency.md` (mark the "live surfacing" deferral ✅ done), `docs/decisions.md` (7d live-surfacing bullet → shipped + the containment-mapping ADR + its known coarseness), `docs/architecture.md` (note the epistemics read-seam is wired), one `docs/DEVLOG.md` entry per logical change. **Also retire the standing "7d deferred follow-up" line in `.claude/SESSION.md`'s next-actions when handing off.**

### Unit test — `tests/unit/test_epistemics.py` (extend)
Pure, no Chroma/DB:
- `markers_for_parent`: parent text containing a marked chunk → its markers; not containing → `[]`; parent containing two marked chunks → deduped union; identical markers de-duplicated.
- `derive_markers` behavior already covered; assert `markers_for_parent` composes with it correctly (contested + superseded both surface).

### Unit test — `tests/unit/test_chat_controller.py` (extend)
Fake pipeline + a stubbed epistemics index/marked-set (no real DB):
- flat retrieval: a source whose `chunk_key` is in the index gets its markers; others empty.
- PC retrieval: a parent whose text contains a marked chunk gets the markers (via the stubbed `load_marked_chunks`); a parent that doesn't stays empty.
- **sidecar empty** → all `markers` empty, and the `TurnResult.answer` + `sources_md` equal the markers-absent baseline.

### Integration test (CI gate) — `tests/integration/test_turn_parity.py` (extend)
- **Byte-identity guard (ADR-2):** with the epistemics read side stubbed to return empty (the no-sidecar default), a turn's `TurnResult` is **identical** to the M0 baseline (same answer, same `sources_md`). This is the eval-comparability gate — markers must be a no-op when absent.
- Deterministic; no network; no paid call (cpc §13).

## Definition of done
- Retrieved chunks carry `contested` / `superseded_trend` markers on `SourceView` in **both** flat and PC retrieval modes (PC via the ADR-1 containment mapping); Chainlit shows a quiet chip only when non-clean.
- **Byte-identical** turn when the epistemics sidecar is absent/empty (guard-tested) — eval path unaffected; clean no-op on a fresh checkout.
- Read-only, **free**, no provider touched; no change to `prompts.py` / `synthesis.py` / the answer stream.
- Unit + integration tests green; `ruff` / `mypy --strict src` / `bandit` clean; coverage floor held.
- `feature-7d-knowledge-currency.md` live-surfacing deferral marked done; `decisions.md` records the containment-mapping ADR **and its known coarseness** (a `KNOWN_ISSUES` entry if it misfires in practice); one `DEVLOG.md` entry per logical change; retires the 7d follow-up in the baton. **Stage + summarize the diff; do not commit/push without review** (cpc §13).

## Out of scope (later / deferred)
- **Marker quality** — the noisy 8B polarity + the open-vocabulary graph are KI-7; the **concept-graph redesign** is the real fix. M1 surfaces what exists; it does not improve extraction.
- **Precise PC re-projection** (option 2: re-key `chunk_epistemics` to PC parents) — the documented upgrade if containment proves too coarse; its own PR with attribution-quality validation.
- **`query_router.py` local/global retrieval-mode seam** (7d Decision 8) — still deferred; not needed for marker display.
- **Rich Tauri marker UI** (hover for the contested concept, link to the corroborating/contradicting docs) — PR-M3.
- **De-noising polarity** (tighter extraction prompt / stronger model) — tracked with the graph redesign.
