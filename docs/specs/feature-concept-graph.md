# Spec — Concept graph (a corroboration/coverage instrument that is also a navigation surface)

**Status:** **DESIGN-LOCKED 2026-07-17** (grilled `grill-me`: 12 branches, 11 resolved / 1 parked; ledger at
foot). Boundaries in **ADR-017**; the gap layer it surfaces is **ADR-004**; the vocabulary it reads is
**ADR-015**, which reserved this track by name. **PR-G1 ✅ BUILT 2026-07-17 (staged, not committed).**

**Owner:** Claude Code. **Never bundle:** ~~PR-G1 serve~~ **BUILT** → ~~PR-G2a index+ego+chunks~~ **BUILT** →
PR-G2b gaps as a destination → PR-G2c Library entry → ~~PR-G4 Node B~~ **RUN** (see the PR-G4 section).

**⚠ Read `## The verdict` before designing any screen.** The gap payload is **~50% precise** and the strong
half is **list-shaped, not graph-shaped**. A spec that leads with the pretty part ships the noise.

> **⚠ TWO BOXES, TWO CORPORA — this spec's live numbers are not universal (added 2026-07-18).** PR-G1/G2a,
> ADR-017 and `## The verdict` were measured on the **RTX box: 76 docs, 26 concepts, 14 gaps**. The **CPU
> box carries 47 docs, 688 keywords, 357 concepts** and *none* of the labels this spec cites (`Res2Net`,
> `PHATE`, `SBERT`, `Embeddings`) exist there. **Do not treat a live-verified number here as reproducible
> on the box in front of you — re-measure.** The CPU box's vocabulary has since been rescoped to **13**
> graph concepts / 19 edges / 15 gaps by **ADR-018** (`graph_include`), which also means
> **`## The verdict`'s "single_source is the strong signal" ranking is un-retested at that size** — see
> the PR-G2b note.

---

## The job (B1 — user, 2026-07-17)

Three questions, one surface:

1. **Corroboration** — *"is this concept backed by more than one source?"* The user's framing:
   **"Technically, having a single source is not good."** A method known only from the paper that invented it
   has no independent evaluation, replication, or critique.
2. **Coverage** — *"have I read the field?"* A PhD student must cover a literature; a professor covers a field
   plus contingent ones.
3. **Navigation** — *"explore the sources through the graph"*, down to the chunks.

**This is not an Obsidian-style corpus browser.** ADR-004: *"Phase 7's stated purpose is **gap detection** —
and the project's north-star reason for it is to surface what the user (and the LLM) cannot see: concepts the
corpus under-supports, claims it cannot source, and directions for exploration the user did not think to look."*
The graph is the substrate; **the gaps are the payload**.

## The verdict (RG-014, run 2026-07-17 — this constrains the design)

`build_gaps --apply` was run on a **fresh** skeleton (`b59a4aa6afa77978`). 14 rows persisted; re-run →
14, no duplication (idempotent). **8 of 14 defensible:**

| kind | n | verdict | design consequence |
|---|---|---|---|
| `single_source` | 3 | ✅ **TRUE POSITIVE — the product thesis.** `Res2Net` only in the Res2Net paper; `SBERT` only in the Sentence-BERT paper; `PHATE` only in one paper. | **Lead with this.** It is the corroboration job, and it is the natural trigger for the parked acquire loop. |
| `unsourced_claim` | 4 | ⚠️ **Real, ~33% contaminated input.** `RAG` 15 claims, `BM25` 5; sampled prose genuinely uncited. But **12 of the 61 underlying `unsupported` claims are markdown HEADINGS** (structurally uncitable) + 8 fragments. | **Ship, but do not present the count as precise.** Blocked on the segmenter bug (ui-checklist row) before any number is shown as authoritative. |
| `under_connected` | 5 | ❌ **Mostly noise at n=26 — and the largest kind.** It measures **graph degree**, dominated at a 26-concept vocabulary by **vocabulary sparsity, not corpus coverage**: `Tractography` (10 docs) and `Motor control` (13 docs) — among the **best-sourced** concepts — are flagged. 2 of 5 duplicate `single_source`. | **Do NOT lead with it. Do not show it by default.** Reopens as the vocabulary grows (`--promote-all` → ~86). |
| `thin_bridge` | 2 | ⚠️ **Redundant + half-misleading.** Both from **one edge**; it flags both endpoints, so **`Embeddings` (degree 20/25, 32 docs — the most-connected node) is reported as a "thin bridge".** | Suppress the hub endpoint, or defer the kind. |

**⇒ The strong kinds are LIST-shaped; the weak kinds are GRAPH-shaped.** B1 stands (corroboration rests on the
two strong kinds; navigation is independent and verified) — **but the graph is not the primary renderer for the
payload. The list is.** The graph earns its place as the *navigation and context* surface, not the dashboard.

## Grounding (measured live 2026-07-17 — not assumed; do not re-derive)

- **Data is ready.** `concepts` 26 · `concept_aliases` 17 · `concept_edges` **70** · `concept_presence` **222**
  (concept→**document**, with `chunk_keys_json` denormalized) · `doc_similarities` 760 · 3 Louvain communities
  (11/9/6) · 0 isolated nodes · degree 1–20.
- **`skeleton.json`** = `{meta, nodes[26], edges[70], communities[3]}`; **node** `{id(uuid), label, doc_ids[],
  degree, community}`; **edge** `{source, target, provenance[], provenance_strength{}, weight,
  n_cooccurrence_chunks, stance[], relation}`. **A complete render model with layout signal precomputed.**
- **The read model does not exist.** There is **no `load_skeleton()`** (`skeleton_from_dict:597` exists but
  nothing reads the file) and **zero API routes** serve graph/gap data.
- **Navigation is real: 1781/1781 (100%)** of `concept_presence.chunk_keys` resolve against the live index.
  Key = **ADR-4 composite `{document_id}:p{parent_index}`**.
- **Rebuild = 7.1 s**, zero-LLM, deterministic, idempotent.
- **The hairball is present at n=26.** 22% density, mean degree 5.4 — but **`Embeddings` has degree 20/25 and
  touches 80% of the graph** (depth-1 ego = 81% of all nodes) while the **median ego is 6 nodes**.
- **The epistemic dimension is empty.** All 26 nodes are `unique`/`stable`; `contested_edges()` → `[]`; all 70
  edges carry `relation=None`, `stance_json=None`. **Node B is BUILT but never run** (`build_concept_skeleton
  --enrich`, $0 on local Ollama, needs the RTX box).
- **All 70 edges are provenance-identical** (`{cooccurrence, similarity}`); **0 citations resolve**, so the
  `citation` token can never fire. Weights span **2.377–2.949** (nearly flat).

## Locked decisions (grill, 2026-07-17 — full ledger in the session baton)

| # | Decision | Reason / reopens if |
|---|---|---|
| B1 | **Job = corroboration + coverage + navigation**, not browsing | ADR-004's north star + a real payload. **Reopens if** the payload proves noise — *partially realized; see the verdict* |
| B3 | **Ego-first, depth-1, expand on click** | Median ego = 6 nodes; scales to any vocabulary; hub ego = 21 (bounded, and honest). **Reopens if** the vocabulary stays <40 forever |
| B4 | **Entry sequence:** concept index + gap lens → gaps as a destination → Library doc entry | User's ordering |
| B5/B6 | **Read-only for the vocabulary + deep-link to Manage-keywords** | **ADR-017 A1** |
| B7 | **node → `doc_ids[]` → `concept_presence.chunk_keys` → chunks**; reuse `openDocument()` | 100% resolve |
| B8 | **Staleness indicator + in-app Rebuild (202 + poll)** | **ADR-017 B1**; mirrors `/api/ingest` |
| B9 | **Gaps encode on NODES; stance on EDGES** | Every `Gap` is anchored to `concept_id` (even `thin_bridge` — endpoints live in `evidence.fact_ids`); stance is an edge property ⇒ **no collision** |
| B14 | **Gap triage (dismiss/promote) via a user-override sidecar** | **ADR-017 C1** — the grill's reasoning was wrong; deterministic rows are delete-and-replace |

## Carve

### PR-G1 — serve the read model (backend, thin shell) — ✅ BUILT 2026-07-17 (staged)

**Built as:** `concept_skeleton.load_skeleton()` (the read half of `write_skeleton`; `None` when absent,
**raises on a corrupt artifact** — "never built" and "unreadable" are different states) · `gaps.load_gaps()`
(the read half of the row writers; lives in `gaps.py` because that module owns the gap domain) · a new
**`src/doc_assistant/concept_graph_view.py`** assembling skeleton + gaps + staleness (`GraphView`,
`GraphStaleness`, `load_graph_view`, `load_concept_presence`) · payloads in `apps/api/models.py` · four thin
routes in `apps/api/main.py`: `GET /api/concepts/graph` (200/**404 empty state**), `GET
/api/concepts/{id}/presence`, `POST /api/concepts/graph/rebuild` (**202**/409), `GET
/api/concepts/graph/rebuild/status`. Rebuild rides a **`rebuild_graph_fn` test seam** mirroring `ingest_fn`.

**Deviation from the contract, deliberate:** presence is served **per-concept**, not bulk in the graph
payload as this spec first said. Ego-first (B3) renders one neighbourhood at a time, and bulk-shipping 1781
chunk keys for a 26-node graph is waste that scales badly (357 concepts). The graph payload keeps `doc_ids`
on each node, so doc-level navigation needs no second call; only chunk-level does.

**Verified live on the real corpus ($0/offline):** `GET /api/concepts/graph` → **26 nodes / 70 edges / 3
communities / 14 gaps**, `graph_version b59a4aa6afa77978`, `stale:false`; **the one-id-space contract holds —
70/70 edge endpoints and 14/14 gap anchors resolve to node ids**; every `relation` is `null` (Node B never
run). Presence: `Embeddings` → **32 documents, 283 chunk keys** (ADR-4 `{document_id}:p{parent_index}`);
unknown concept → `[]`. Rebuild: `202` → poll → `done`, and it returned the **identical** `graph_version` —
determinism proven end-to-end through the API. Empty state: skeleton moved aside → **404** with the rebuild
hint → restored → 200. **Gates:** ruff + ruff format + `mypy --strict src` + bandit clean; **full suite 994
passed** (was 977; +16 new, 0 regressions).

- **`src/doc_assistant/concept_skeleton.py`** — add **`load_skeleton() -> ConceptSkeleton | None`**: read
  `data/skeleton/skeleton.json`, `skeleton_from_dict` (`:597`), `None` if absent. **The loader belongs in
  `src/`** (thin-shell rule).
- **Staleness is part of the payload, not an afterthought.** Return `{graph_version, built_at,
  n_concepts_in_db, n_concepts_in_skeleton, stale: bool}`. Cheapest honest signal: compare the skeleton's node
  ids against `load_concepts()`. **Never hard-code a `graph_version`** — it changes on every rebuild
  (`055312c8c15a7e69` → `b59a4aa6afa77978` on 2026-07-17).
- **`GET /api/concepts/graph`** → skeleton + gaps + presence. **`GET /api/concepts/{id}/ego?depth=1`** may be
  folded in later; at 26 nodes one payload is fine (44 KB; ~600 KB at 357 — **state whether it can ever
  paginate**).
- **Pick ONE wire id space and document it.** Node ids are **UUIDs**; labels live only on nodes. **This exact
  UUID-vs-label mismatch caused KI-15.** Do not mix.
- **`POST /api/concepts/graph/rebuild` → `202`** + a status poll, mirroring `ingest_start` (`main.py:613`) /
  `ingest_status` (`:683`). Thin shell over `build_concept_skeleton(apply=True)`.
- **DoD:** `load_skeleton` round-trips (`skeleton_to_dict`/`from_dict` are exact inverses); absent file →
  `None`; stale fires when a `Concept` exists that the skeleton lacks; route 200 shape + the **absent-skeleton
  path (a fresh clone has none — the NORMAL first run, not an error)**; `curl … | jq '.nodes|length'` → **26**,
  `.edges|length` → **70**, `.gaps|length` → **14**.

### PR-G2a — concept index + gap lens + ego + chunk navigation (frontend) — ✅ BUILT 2026-07-17 (staged)

**Built as:** `lib/ConceptGraph.svelte` (index + gap lens + ego SVG + doc→chunk nav + staleness/empty rebuild) ·
`lib/forceLayout.ts` (pure, seeded mulberry32 + phyllotaxis init + Fruchterman–Reingold → fit-to-viewBox;
deterministic, epsilon-guarded so no coordinate is NaN) · `types.ts`/`api.ts` mirror the 7 PR-G1 payloads +
4 client fns (404 → `null`) · `app.css` 12-hue categorical community palette (both themes, cycled `community %
12`) + `--graph-edge`/`--graph-node-stroke` derived once from `--fg` via `color-mix` · `Icon.svelte` `waypoints`
glyph · `App.svelte`/`Sidebar.svelte` widened `mode` union + third rail tab.

**Deliberate choices beyond the spec:** the community palette is a **fixed per-theme categorical ramp**, not a
`color-mix` hue wheel (a rotated hue can land on low-contrast yellow in light theme; a ramp controls contrast,
and cycling past 12 is harmless because colour is a positional grouping hint). **Only zoom persists**, not pan
(pan is position-specific and resets when the ego re-centres). Gap badges use `--danger` (single_source) /
`--warn-fg` (softer kinds); stance stays reserved for **edges** (B9), so no collision when PR-G4 lands.

**Verified live ($0/offline, real corpus, `read_page` + `javascript_tool`):** index **26 concepts**, the 3
`single_source` true positives lead in danger tone, gap lens **8** → **10** with under-connected opted in;
Res2Net ego 3/3, Embeddings (deg 20) ego **21 nodes** — **no NaN**, all in-viewBox, no collapse; **determinism
holds across a re-render**; 3 distinct community fills, theme flip changes fill + the `color-mix` edge; zoom
clamps **0.4↔3.0**; Res2Net → 1 doc → "25 chunks" → **Open in Library** + **Edit** → Manage-keywords; 375px no
overflow. **Gate:** `svelte-check` 0/0, `vite build` clean (157 modules), still **one runtime dep (`marked`)**.

- **A full view, not a modal.** `mode` is a plain `$state<'chat'|'library'>` with `{#if}` branching — **no
  router**. The union appears in exactly **4 places**: `App.svelte:118`, `:562`, `:800`, `Sidebar.svelte:40,50`.
  Widen to `'chat'|'library'|'graph'`; add a third rail tab (`Sidebar.svelte:230-251`); use **`main.wide`**
  (`:1160`). *(There is no reusable modal shell — it is hand-rolled in **6** components, all `min(84vh,620px)`
  transient tasks. A graph is a destination.)*
- **The index is the home**, not the graph: a searchable list of concepts (label · doc count · gap badge),
  with a **"show only gaps"** lens. **Order by the strong signal** — `single_source` first. **`under_connected`
  is off by default** (see the verdict).
- **Ego view on select:** depth-1 neighbours + the concept's documents. **Hand-rolled SVG + a seeded force
  layout, no dependency** (ADR-017's rationale + the 1-dep/101 KB frontend + CSP `default-src 'self'` with no
  `unsafe-eval`). Emit **real `<line>`/`<circle>`/`<text>` children** per `Icon.svelte`'s pattern — its "no
  `{@html}`" is **SVG-namespace safety, not CSP/XSS** (`Markdown.svelte:107` uses `{@html}` in prod). Seed to
  match the skeleton's `seed: 42`; **run to convergence off the render path, then draw statically** — do not
  animate a simulation.
- **Pan/zoom:** one SVG `transform`. **Copy `App.svelte:301-321 startResize()`** (pointerdown → window
  pointermove/pointerup → clamp → cleanup → persist). Wheel-zoom needs **`passive:false`** — a deliberate
  divergence from `LibraryGrid.svelte:87-97`; comment it. Persist zoom/pan like `libraryView`/`librarySort`.
- **Chunk navigation:** doc → the chunks where the concept appears (`chunk_keys`). This is the "navigate the
  library through the chunks" job.
- **Encoding — only what has signal.** `community` → colour · `degree` → radius · gap kind → node badge
  (`--danger`/`--warn-fg`). **Do NOT** use `weight` for thickness (2.377–2.949, flat). **Do NOT** ship a
  provenance legend (one state). **Do NOT** ship contested/superseded colouring (renders nothing until PR-G4).
- **Palette:** only **3 non-semantic hues** exist (`--accent`/`--lavender`/`--ok-fg`) for exactly 3
  communities — **luck, not headroom.** Add a categorical palette to `app.css` (both themes) or derive via
  `color-mix`. **Zero hardcoded hex.** `IconName` is a closed 26-name union with **no graph glyph** — add a
  Lucide path (`waypoints`/`share-2`/`git-fork`, ISC).
- **Staleness + empty state share one affordance:** "Graph is N concepts behind" / "No graph yet" → **Rebuild**
  (7.1 s, 202+poll). **Inform, don't block.**
- **Deep-link "Edit this concept"** → the Manage-keywords view (ADR-017 A1). The graph never writes the
  vocabulary.

### PR-G2b — gaps as a first-class destination + triage

> **⚠ RE-MEASURE THE ORDERING FIRST (2026-07-18).** "Strong kinds first" inherits `## The verdict`'s
> ranking, measured at **26 concepts / 14 gaps** on the RTX box. After ADR-018 the CPU box sits at **13
> concepts / 15 gaps**, and the kinds are nearly **flat** — `thin_bridge` 4 · `isolated` 3 ·
> `single_source` 3 · `under_connected` 3 · `unsourced_claim` 2. `single_source` is no longer
> self-evidently the headline, and `under_connected`'s "graph-degree noise at n=26" argument needs
> restating at n=13. **Re-derive the verdict on the box you build on; do not rubber-stamp it.**
> Also blocked on **KI-17** — 10 orphaned stochastic gap rows are served today, and a gap you cannot
> resolve to a concept is a gap you cannot triage.

- Promote the gap lens to its own surface: the findings, grouped by kind, **strong kinds first**.
- **Triage (dismiss/promote) via the ADR-017 C1 override sidecar** — new table keyed on `(concept_id, kind)`,
  landing via `create_all` (the `figures` precedent), **not** `_ADDITIVE_COLUMNS` (that handles columns).
  `GapRow.status` renders as **effective = override ?? "surfaced"**.
- **Guard test (load-bearing):** dismiss a **deterministic** gap → `build_gaps --apply` → **the dismissal
  survives.** This is the exact behaviour that is broken today (`gaps.py:257` deletes deterministic rows;
  verified live: `dismissed` → `surfaced`).
- `promoted` is the **action slot** the parked acquire loop (B13) attaches to. Do not repurpose it.

### PR-G2c — Library entry (doc → its concepts)

Reverse `concept_presence` from the Library document view; reuse PR-G2a's ego view unchanged.

### PR-G4 — run Node B (separate; unblocks the epistemic encoding) — ✅ RUN 2026-07-18 (CPU box)

**Node B is BUILT** — `concept_skeleton_enrich.py` (pure core, idempotent, **never creates a node or edge**) —
and runnable via **`build_concept_skeleton --enrich`** (`:150`). **$0**:
`CONCEPT_SKELETON_LLM_PROVIDER` defaults to **local Ollama** (`llama3.1:8b`), not `LLM_PROVIDER`, guarded by
`assert_provider_intent` (KI-4); one call per document. Running it → stance → contested nodes → the reserved
edge encoding + the L1b Library row unblock.

> **CORRECTION (2026-07-18): "Never run" and "Blocker: Ollama is on the RTX box" were both false.** Ollama
> is on the **CPU** box, and Node B had already run there on **2026-07-08** (the G6 session's
> `--apply --enrich`) — that skeleton carried `node_b_calls: 46` / **1254 annotated edges**. Re-run
> 2026-07-18 after the ADR-018 rescope: **9 calls, 19/19 edges annotated, 63 stance assertions, 7
> contested edges**, `$0`, ~2 min. Node-level directions: **7 contested / 6 stable / 0 superseded_trend**
> (a 13-concept vocabulary rarely clears G6's ≥2-dated-docs-per-side floor). **The reserved edge-stance
> encoding is therefore unblocked by data now** — it can be built whenever PR-G2b/G2c allow.

*(Rebuild note: always `--apply --enrich` **together** — `--apply` alone rebuilds the edges with no
`relation`/`stance_by_doc` and silently wipes Node B's annotations; see
`tests/eval/baselines/superseded_year_rule_2026-07.md`.)*

## Verification (this is the safety net — there is no other)

**Zero frontend tests exist; screenshots time out on this box.** The SVG DOM is the only assertable surface —
which is *why* ADR-017 locks SVG over canvas. All via `read_page` + `javascript_tool`:

- `document.querySelectorAll('.graph circle').length` matches the ego's node count; `line.length` its edges.
- **No `NaN`/`undefined` in any `cx`/`cy`/`x1`/`y1`** — *the* classic force-layout bug, caught programmatically.
- Node centres within the `viewBox`; pairwise distances > 0 (**no collapse to a point**).
- `getComputedStyle(node).fill` resolves to the palette value under `[data-theme='dark']` vs `light`.
- Zoom: dispatch a synthetic `wheel` → the `transform` scale changed **and is clamped**.
- Click: dispatch a **real `MouseEvent`** (Svelte 5 `onclick` ignores a synthetic `.click()`) → `libraryDocId`
  changed.
- **Determinism: render twice → identical positions.** Seeded layout makes the assertions non-flaky.
- Live, $0: index shows **26** concepts and **14** gaps; `single_source` on `Res2Net` → its one document →
  its chunks. Both themes; 375px no-overflow.

## Out of scope / parked

- **B13 — the gap → acquisition loop** (*"download and find more information to complete the graph… we will
  need a provider list, and a quality list"*). Closes ADR-004's loop; merges with the **External literature
  discovery** row; **needs its own ADR**. Transport already spiked (stdlib urllib → Crossref, 25/25).
  **Constraint honoured here:** a gap is an object with an action slot (`GapStatus.promoted`).
- **In-graph curation** — ADR-017 A1 forbids it; reopening means reopening the ADR.
- **Retrieval-rank use of the graph** — read-only; any rank use is a separate eval-gated experiment
  (`concept-graph-redesign.md`'s own rule).
- **Contested/superseded encoding** — dead until PR-G4.
- **`under_connected` / `thin_bridge` as headline signals** — see the verdict; revisit at a larger vocabulary.

## Traps (each cost real time to find)

- **⚠ An unfamiliar short label is NOT evidence of junk — trace it to its document before judging. This trap
  has fired TWICE (2026-07-17, 2026-07-18).** The corpus is **multi-domain** (IR/RAG · systems neuroscience ·
  viral tracing/mouse genetics · AI planning), so a vocabulary reviewed from inside the retrieval domain
  reads most of it as noise. Real terms mistaken for artifacts: `cre` (Cre recombinase, **203 mentions —
  more than `BM25`**), `dbs` (deep brain stimulation, 134), `ntsr1`, `pddl`, plus the 2026-07-17 batch
  (`16p11` = 16p11.2 truncated at the dot, `c57bl` = C57BL/6 across **7 docs**, `va1v`/`dl5`/`osns` =
  Drosophila glomeruli). Genuinely broken strings are a small **clustered** minority — `mathrm`/`professium`/
  `outflux` all came from one 1952 scanned paper. **The rule, set 2026-07-17 and restated here: demote, not
  delete — deleting real vocabulary is not reversible-by-search.** Under ADR-018 the demote verb is
  `set_graph_include(cid, False)`, which keeps the row and its keyword family. Cheap check before any such
  call: the concept's aliases, `n_mentions`, and the **titles** of its presence documents.

- **`data/graph/graph.json` is a stale EMPTY DECOY** (0 nodes, Jun 15) — residue of the retired
  `concept_graph.py` (KI-7). **Reading it renders an empty graph that looks like a layout bug.** The live
  artifact is `data/skeleton/skeleton.json`.
- **`skeleton.json` is gitignored** — a fresh clone has none. That is the normal first run.
- **Community ids are positional, not identity** — they renumber when a concept is added. **Never persist a
  user preference against one.**
- **Node ids are UUIDs, labels only on nodes** — the KI-15 id-space mismatch.
- **`list_keyword_families()` returns every `Concept`** — families and graph nodes are the same rows
  (ADR-015's "boundary risk"), so a user's family edit changes the graph **and the graph lags until rebuilt**.

## Grill ledger

Grilled 2026-07-17 (`grill-me`) — **12 branches: 11 resolved, 1 parked, 0 open**; full ledger in
`.claude/SESSION.md`. Walked job → carve → render → entry → writes → click-through → staleness → palette →
empty → mobile → verification in dependency order. **The root question was overturned by the repo, not the
user:** the grill opened intending to argue against a decorative Obsidian clone; ADR-004 already named gap
detection as the north star, the layer was already built and Ollama-validated, and a free dry-run produced 14
findings. **The user's own framing independently matched the detector** (*"having a single source is not
good"* ≡ `single_source`). Two grill premises were later disproved by RG-014 and are corrected in ADR-017 and
in `## The verdict` above.
