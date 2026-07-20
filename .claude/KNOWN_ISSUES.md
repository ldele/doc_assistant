<!-- status: active · updated: 2026-07-20 (KI-25 added + resolved: the graph emptied when KI-23 landed) · class: living -->

# KNOWN ISSUES

Open weaknesses, recurring failures, workarounds. Log a bug the second time it appears.
Migrated from the old `CLAUDE.md` / `README` runtime-quirk notes on 2026-06-20 (cpc adoption).

**Shape (since 2026-07-20):** open issues in full below; closed ones as one-line rows in
**Resolved — index** at the foot, pointing at their verbatim account in
`docs/archive/KNOWN_ISSUES-resolved-001.md`. When you close an issue, move its body to the
archive and add a row — keep the trap and the fix that must not be undone, drop the diagnosis
narrative. Numbering is global and never reused (see the KI-23 note in the archive).

## KI-2 — Python-3.12 runtime pin — STILL OPEN (cause renamed after PR-M5: native deps, not Chainlit)
- **Original cause (gone):** Chainlit's anyio stack broke on 3.14. **Chainlit was removed in PR-M5
  (2026-06-25)** — so that cause no longer exists, but the runtime pin does **not** lift.
- **M5 ADR-2 verification (2026-06-25):** with Chainlit gone, `uv sync --python 3.14 --extra cpu --extra dev`
  resolves + installs cleanly (torch `2.12.0+cpu` has a cp314 wheel; chainlit absent), and ruff /
  `mypy --strict src` / bandit all pass on 3.14 — **but the full pytest suite hard-crashes the interpreter**
  (no Python traceback; the process dies at ~47–54%, first surfacing in `tests/unit/test_llm.py` under
  full-suite load). It does **not** reproduce unit-only or for that test in isolation (load/order-dependent).
  327+ tests pass before the crash; Python 3.12 runs all **602** clean.
- **Cause:** a native/compiled dependency in the LLM-client import path (anthropic / langchain /
  `pydantic-core` / `tokenizers` — Python 3.14 is new and several C/Rust wheels aren't yet cp314-stable).
- **Workaround:** **run + test on Python 3.12** (the pinned runtime). CLI / FastAPI / Tauri all work on 3.12.
- **Real fix:** revisit when the native deps ship stable cp314 wheels — re-run the M5 ADR-2 check
  (`uv sync --python 3.14 …` + full gate) and lift the pin only when 602 pass on 3.14. Do **not** add
  3.13/3.14 trove classifiers until then.
- **Note:** the literal `--python 3.12` pin (the old `just chat`/`chainlit` recipe) is **deleted** (M5); the
  only thing now holding the runtime at 3.12 is this native-dependency stability gate.

## KI-4 — Anthropic-default credit leak on "local" enrichment runs — OPEN (workaround known)
- **Symptom:** Enrichment / self-eval runs intended to be local (Ollama) silently bill the Anthropic API.
- **Cause:** `.env` defaults are all-Anthropic; every generator/reviewer/judge inherits the default
  provider unless overridden.
- **Workaround:** **Force `--provider ollama`** (and reviewer/judge provider flags) on every
  enrichment / self-eval run. Cost-guards exist but the default is the trap.
- **Real fix:** A local-first default profile, or a hard guard that refuses paid calls outside an
  explicit `--allow-paid`. Contract: `docs/specs/llm-provider-isolation.md`.

## KI-5 — Sandbox cannot write runtime data; enrichment runners must run on the host — OPEN
- **Symptom:** Metadata backfill / citation extraction / other enrichment CLI passes appear to no-op
  or fail to persist when invoked from a sandboxed context.
- **Cause:** `data/` (SQLite + Chroma) is host-local and gitignored; the sandbox/host filesystem
  isn't synced for writes.
- **Workaround:** Run the idempotent enrichment runners (`scripts/extract_*`, `compute_doc_vectors`,
  etc.) **on the host**, not the sandbox. They're idempotent, so re-running is safe.
- **Real fix:** N/A — environmental; document and run on host.

## KI-6 — SSL crash on a `uv`-managed Python (Windows) — OPEN (per-machine; documented workaround)
- **Symptom:** App dies instantly with no traceback (`OPENSSL_Uplink(...): no OPENSSL_Applink`) on the
  first HTTPS call (Claude API, Ollama, any networked test).
- **Cause:** OpenSSL in uv's bundled (python-build-standalone) interpreter; an official CPython is unaffected.
- **Status:** a persistent per-machine environmental quirk; the fix is **deliberately not pinned in-repo**
  (`docs/decisions.md`) — kept as a documented remedy, revisited only if more boxes hit it.
- **Workaround:** Rebuild the venv on an official python.org 3.12 (`py install 3.12` →
  `uv venv --clear --python …` → `uv sync --all-extras`). Behind a TLS-inspecting proxy, prefix uv
  commands with `UV_NATIVE_TLS=1`. Offline work (ingest/embeddings/retrieval) is unaffected either way.

## KI-8 — PC→baseline marker mapping (PR-M1) is coarse at parent boundaries — OPEN (advisory, fail-safe)
- **Symptom:** In the default parent-child retrieval mode, the live 7d marker chip maps a marked baseline
  chunk onto a retrieved parent by **text containment** (`epistemics.markers_for_parent`): a parent gets a
  marker if it *contains* a marked chunk's text. The two collections are independent segmentations, so a
  parent spanning a marked chunk plus three clean ones is marked as a whole — over-attribution within the
  parent. A marked chunk straddling two parents marks both.
- **Why it's acceptable (for now):** markers are an **advisory chip, not a gate** (inform-don't-block), and
  over-attribution is fail-safe — it points the user at a real contested concept *in that passage*. The
  marker never changes synthesis, ranking, or the answer (byte-identical when absent).
- **Status:** chosen in PR-M1 ADR-1 over the heavier alternative (re-project `chunk_epistemics` onto PC
  parents — a second projection + migration + its own attribution-quality validation). That precise
  re-projection is the documented upgrade **if** containment proves too coarse on real data.
- **Compounding caveat:** marker *quality* upstream still comes from the superseded open-vocabulary graph
  (KI-7) — `contested` is local-model-noisy. M1 surfaces what the sidecar holds; it does not fix extraction.
- **Mostly moot in practice since 2026-07-02 (PR-R7):** the live chip is now default-OFF
  (`EPISTEMICS_MARKERS_ENABLED=false`, ADR-005), so this containment coarseness only bites when a user opts
  the markers back on. The precise re-projection upgrade rides with Node B, alongside KI-7 retirement.
- **Update (2026-07-16, docs review):** two bullets above are outdated. (a) The "compounding caveat" —
  the open-vocabulary graph is gone (KI-7 RESOLVED 2026-07-07, G1: `concept_graph.py` deleted); marker
  data now sources from the curated Node-A/B `concept_skeleton`, and attribution actually reaches chunks
  since the KI-15 label fix (G7). (b) The "mostly moot / default-OFF" bullet — G1 flipped
  `EPISTEMICS_MARKERS_ENABLED` back to **default-ON** (superseding ADR-005), so the containment
  coarseness is live again by default. The issue this entry tracks (PC-parent containment mapping is
  coarse) is unchanged and still OPEN; the re-projection upgrade remains the documented fix.
- **Update (2026-07-19, scale review — the direction claim above is arithmetically wrong):**
  "a marked chunk straddling two parents marks both" is unreachable — containment is a strict
  full-substring test (`knowledge/epistemics.py:234`) and a `BASELINE_CHUNK_SIZE=1000` chunk can
  never fit inside a `PARENT_CHUNK_OVERLAP=200` overlap, so a straddling chunk is contained in
  **neither** parent and its markers silently vanish. The real failure mode is systematic false
  *negatives* (order ~40% of marked chunks at these sizes), not fail-safe over-attribution — in
  the default-ON, default-PC configuration. The documented upgrade (re-projection, option 2)
  or overlap-based matching fixes it. See `docs/REVIEW_2026-07-19_scale-robustness.md` WE-7.
- **Pointer:** `docs/archive/pr-m1-epistemics-markers.md` ADR-1 (option 2 = the re-projection upgrade).

## KI-17 — stochastic gap rows outlive their concept → orphaned gaps served to the graph UI (2026-07-18, OPEN)
- **Symptom:** `load_graph_view()` serves **27** gaps against a **13**-node skeleton; **10** of them
  (all `kind="suggested_concept"`, all `determinism="stochastic"`) carry a `concept_id` that resolves
  to no node. The view's own report disagrees with the sidecar: `build_gaps --apply` printed
  "Total gaps: 15 · Rows written: 15", but the route returns 27. Surfaced by the ADR-018 rescope
  (357 → 13 graph concepts, 2026-07-18); the 10 orphans were generated on 2026-07-08 against the
  old 357-concept vocabulary.
- **Cause:** the two gap classes have different write disciplines (this is ADR-017's own finding,
  read from the other end). `gaps.py:257` **delete-and-replaces** deterministic rows, so those
  self-heal on every rebuild; `_write_stochastic_gap_rows` (`:273`) is a **status-preserving
  upsert**, which is correct for not losing a user's triage — but it has **no delete pass for rows
  whose concept left the vocabulary**. Nothing reconciles a stochastic row against the current
  vocabulary, so it is immortal. Deleting a `Concept` (or, now, excluding it) strands its gaps.
- **Impact:** PR-G2a's index badges gaps by looking the concept up, so an orphan renders no row —
  it inflates the gap *count* without being reachable. Worse for **PR-G2b**, which promotes gaps to
  a first-class destination with a per-row triage action: a row you cannot resolve to a concept is a
  row you cannot dismiss or promote.
- **Workaround:** none needed for correctness today (the orphans are invisible in the index, not
  wrong answers); read the `build_gaps` report — not `len(view.gaps)` — as the true gap count until
  fixed.
- **Candidate fix (PR-G2b territory, ADR-017 C1):** in `_write_stochastic_gap_rows`, delete
  stochastic rows whose `concept_id` is not in the current vocabulary before the upsert — a
  reconcile pass, not a blanket delete, so triage on a *live* concept still survives. Guard test:
  a stochastic gap on a concept that is then excluded (`set_graph_include(cid, False)`) →
  `build_gaps --apply` → the row is gone, while a stochastic gap on an included concept keeps its
  status. Decide alongside the C1 override sidecar, since both concern what a rebuild may destroy.
- **Placement correction (2026-07-19 review):** the reconcile as sketched sits inside
  `_write_stochastic_gap_rows`, which only executes under `suggest and apply` and early-returns on
  zero suggestions — a deterministic-only `build_gaps --apply` (this KI's own repro) would never
  reach it. Hoist it to run unconditionally on every `--apply`, keyed against the
  `graph_include`-filtered `load_concepts()` (excluded = removed; the unfiltered table would fail
  this KI's own guard test). See `docs/REVIEW_2026-07-19_scale-robustness.md` (GP/KI-17 check).

## KI-18 — knowledge layer: corpus-linear/quadratic hot paths fall over well before 10k docs — OPEN (2026-07-19)
- **Symptom:** every knowledge/ cluster has at least one path whose cost scales with the corpus
  (not the vocabulary), invisible at n≈50: presence loads **every** child chunk incl. denormalized
  parent_text in one unpaginated `coll.get` then rescans the corpus once per concept
  (`knowledge/concept_skeleton.py:1100`/`:214`); edge provenance is a per-edge doc×doc Cartesian
  product (`:332`); the keyword extractor holds all corpus text + per-occurrence term streams in
  RAM and pays the full-corpus load even for a single-doc re-extract (`knowledge/keywords.py:733`);
  family Tier-2 is O(n²) pairwise cosine (`knowledge/keyword_families.py:149`) and
  `list_keyword_families` is an N+1 COUNT (`library.py:486`); the epistemics projection is a
  full-recompute O(chunks × concepts) regex scan, whole corpus in RAM, `re.compile` in the
  per-chunk loop saved only by Python's 512-pattern cache (`knowledge/epistemics.py:140/258/445`),
  and flat-mode chat loads the entire marker index per turn (`chat_controller.py:722`); the
  unsourced-claims sweep loads every claim ever persisted (`knowledge/gaps.py:242`); wiki
  synthesis re-summarizes unchanged topics with unbounded material (`knowledge/wiki.py:547/440`).
- **Cause:** built and validated on 47/76-doc corpora; the 0–10k contract was never a review lens
  until now.
- **Impact:** first failure is memory (presence + keyword loads), then rebuild wall-clock
  (provenance product, projection scan ~34s@47docs → hours@10k), then LLM-call volume (KI-19's
  budget half). Nothing is wrong at current size.
- **Workaround:** none needed at n≤~100; do not bulk-ingest thousands of docs before the P1 fixes.
- **Fix:** the mechanical P1 list in `docs/REVIEW_2026-07-19_scale-robustness.md` (page/stream the
  loads, invert the provenance loop, hoist/alternate the regex pass, blocked similarity, grouped
  counts, scope the flat index, skip-unchanged topics, bound the claims sweep). No behavior change.
- **Pointer:** REVIEW findings CS-1/2/6/9, KW-1/2/3, GP-3, WE-3/4/10.

## KI-19 — knowledge layer: corpus-tuned constants + unbounded LLM budgets encode n≈50 — OPEN (2026-07-19)
- **Symptom:** thresholds that mis-tune (or already mis-tune) off the current corpora:
  `_DEFAULT_MIN_DEGREE=3` is a frozen Q1 snapshot from a 26-concept graph while the gaps.py
  docstring claims "corpus-derived" (`scripts/build_gaps.py:46`); family Tier-2's
  `DEFAULT_EMBEDDING_THRESHOLD=0.86` sits above bge's own measured same-domain ceiling (~0.82) so
  the tier under-fires structurally (`knowledge/keyword_families.py:28`); `contested` fires on
  `nc>=1` (one disputing doc) — 53.6% of chunks already marked at 47 docs, saturating with growth,
  `agreement_ratio` computed but never consulted (`knowledge/concept_skeleton.py:699`); the wiki
  ships the absolute-cosine 0.90 clustering the monolith recorded as the wrong primitive, fix
  inert behind `WIKI_USE_CONCEPT_COMMUNITIES=false` (`config.py:387/404`);
  `CONCEPT_SKELETON_MIN_COOCCURRENCE=2` is validated at 76 docs only; `KEYWORD_MIN_CHARS=3`
  deletes (not demotes) sub-3-char specialist tokens; `KEYWORD_CORPUS_TOP_K=60` ≈ the current
  vocabulary size. Plus three **unbounded LLM loops** that scale with corpus/vocab: Node B one
  call per doc (`knowledge/concept_skeleton_enrich.py:151`), gap_suggest one per thin concept
  (`knowledge/gap_suggest.py:129`), wiki one per topic incl. singletons (`knowledge/wiki.py:547`).
- **Cause:** the exact over-optimize-on-current-corpus failure the 2026-07-19 review was ordered
  to find; contrast with the honest structural constants (`MIN_DATED_DOCS_PER_SIDE=2`,
  `_MIN_CONCEPT_LEN=3`), which show the discipline exists — it just wasn't applied everywhere.
- **Impact:** silent mis-ranking/saturation as the corpus grows; surprise hour-long (or, if a paid
  provider is forced, costly) enrichment runs.
- **Workaround:** none needed at current size; treat every constant in the REVIEW inventory table
  as suspect before citing it in a design argument.
- **Fix:** measurement-gated only — RG-016 (graph floors + kind ranking), RG-017 (family
  threshold), RG-018 (wiki communities flip), RG-019 (contested min-N); one ADR for the shared
  LLM-budget policy (Node B / gap_suggest / wiki caps). **Never hand-tune these without the
  experiment.**
- **Pointer:** REVIEW findings CS-3/4/7/8, KW-4/5/6/9, GP-1/2/5, WE-5/6; the inventory table in
  `docs/REVIEW_2026-07-19_scale-robustness.md`.

## KI-20 — concept curation hard-deletes vocabulary where ADR-018 mandates demote — OPEN (2026-07-19)
- **Symptom:** `concept_curation.remove_concepts` (`knowledge/concept_curation.py:400`) deletes
  `Concept` + `ConceptAlias` rows outright; stages 1–3 (artifact filter, `classify_noise` LLM,
  near-dup merge) route into it. `classify_noise` is precisely the path that mislabels real
  specialist vocabulary (`cre`/`dbs`/`ntsr1`/`pddl` — the trap hit twice, 2026-07-17/18).
  Deleting a Concept also deletes its keyword family (ADR-015 shared table) and cascades into
  presence/edges/gaps.
- **Cause:** the module predates ADR-018's demote verb; stage-0 ranking was correctly migrated to
  read-only but the destructive stages were not revisited.
- **Impact:** contained today — dry-run default, `--apply`-gated, and stages 1–3 have never been
  applied on the real corpus; the contract violation is the risk, not a live loss.
- **Workaround:** never run `scripts/curate_concepts.py --apply` stages 1–3 until fixed; curate
  with `set_graph_include(cid, False)`.
- **Fix:** route noise/artifact verdicts through `set_graph_include(id, False)` (keep row +
  family); reserve deletion for an explicit, separately-confirmed path. Guard test: a
  `classify_noise`-flagged concept keeps its family after `--apply`.
- **Pointer:** REVIEW finding CS-5 (verified); ADR-018; `docs/specs/feature-concept-graph.md`
  Traps; KW-9 is the same verb error at the tokenizer (`KEYWORD_MIN_CHARS` deletes unmined).

## KI-21 — in-app graph rebuild refreshes the skeleton but not the gaps the view serves — OPEN (2026-07-19)
- **Symptom:** the ADR-017 B1 rebuild route (`apps/api/main.py:232` `_default_rebuild_graph`)
  calls `build_concept_skeleton(apply=True)` only — `build_gaps` has no API caller — and
  `load_graph_view` serves all `GapRow`s with no `graph_version` cross-check
  (`knowledge/concept_graph_view.py:96`, `knowledge/gaps.py:355`). After an in-app rebuild the
  UI shows gaps computed from the previous skeleton (including the gap the user just closed)
  until the CLI runs. Distinct from KI-17 (rows outliving `build_gaps` itself): here `build_gaps`
  never runs at all on the app's only rebuild affordance.
- **Cause:** B1 shipped the skeleton half of the acquire loop ("gap → ingest → rebuild → gap
  closes"); the gaps half was left to the CLI.
- **Impact:** the loop the button exists to close does not close in-app; stale-gap confusion
  compounds KI-17's orphans.
- **Workaround:** run `python -m scripts.build_gaps --apply` after any in-app rebuild.
- **Fix:** chain `build_gaps(apply=True, min_degree=<runtime-derived>)` after the route's
  skeleton build (needs KI-19/GP-1's runtime Q1 so the route needs no hardcoded default), or
  stamp `graph_version` onto gap rows and filter mismatches in the view; land together with the
  KI-17 reconcile (both concern what a rebuild must refresh).
- **Pointer:** REVIEW finding GP-4 (verified); ADR-017 B1.

## KI-25 — the concept graph emptied itself the moment KI-23 was fixed (`graph_include` landed NULL) — RESOLVED (2026-07-20)
- **Symptom (user-reported):** the Graph view showed **nothing**. `GET /api/concepts/graph` returned
  **0 nodes / 0 edges / 0 communities**, and its own staleness block said `n_concepts_in_db: 0`
  while the `concepts` table held **26** rows.
- **Cause — the fix for one issue triggered another.** ADR-018 made the graph vocabulary
  **opt-in** via `concepts.graph_include`, and `load_concepts()` documents that "NULL (every row
  predating the migration) reads as excluded". That column had never reached this box (**KI-23**).
  Running `python -m doc_assistant.db.migrations` by hand on 2026-07-20 — while diagnosing KI-23 —
  finally added it, **NULL on all 26 rows**, so every concept became excluded at once and the
  vocabulary the graph builds from went to zero. The migration was correct; what was missing is
  that an additive column with an opt-in default needs its **backfill** run in the same breath.
- **Not detected by anything.** The graph route degrades honestly to an empty graph (it is the
  documented "empty vocabulary → empty graph" path), the suite stayed green, and no gate compares
  "concepts in the DB" against "concepts the graph can see".
- **Fix (2026-07-20):** `python -m scripts.backfill_graph_include --apply` — the runner that exists
  for exactly this, applying ADR-018's rule retroactively (`source == "manual"` opts in). All 26 are
  `source="manual"` (they were hand-inserted during the 2026-07-01 baseline run, KI-13's
  workaround), so all 26 opted back in. Then a skeleton rebuild:
  `build_concept_skeleton(apply=True)` — **Node A only, deterministic, zero-LLM, $0**.
  Result: **26 nodes / 70 edges / 3 communities / 14 gaps**, `stale: false`; the app's concept index
  and the ego view both render again (verified live, 9 circles + 11 edges for `Connectome`).
- **What is NOT restored:** `concept_edges` was already **empty** before the fix, so no Node-B
  stance annotations were lost *by this* — but none exist now either. Re-running Node B
  (`build_concept_skeleton --apply --enrich`) is an **LLM pass** and was deliberately not run;
  KI-4's rule applies (force `--provider ollama`, which lives on the other box).
- **The general trap, worth more than this instance:** *an additive column whose NULL default
  changes behaviour is not a safe additive migration.* `_ADDITIVE_COLUMNS` already carries the note
  for `graph_include` ("Lands NULL on every existing row, which reads as excluded;
  `scripts/backfill_graph_include.py` sets the policy") — the note was right and simply nobody was
  in a position to act on it, because the column had never landed. Any future opt-in column should
  pair its `_ADDITIVE_COLUMNS` entry with its backfill runner in the same change.
- **Pointer:** `src/doc_assistant/knowledge/concept_skeleton.py` (`load_concepts` — the filter and
  its NULL semantics) · `scripts/backfill_graph_include.py` · ADR-018 · KI-23 (the migration that
  triggered it) · KI-21 (the in-app rebuild's own gap).

---

## Resolved — index

Full accounts (symptom, cause, workaround, fix, guard test, pointers) live verbatim in
[`docs/archive/KNOWN_ISSUES-resolved-001.md`](../docs/archive/KNOWN_ISSUES-resolved-001.md) — the same living-index / frozen-archive
split ADR-022 applied to the decisions monolith. Each row keeps the two things that still bite:
**what the trap was**, and **what now keeps it fixed** (undo that, and the issue returns).

| KI | What it was | What keeps it fixed — do not undo | Resolved |
|----|-------------|-----------------------------------|----------|
| [KI-1](../docs/archive/KNOWN_ISSUES-resolved-001.md) | `print()` + stdlib `logging` in `src/` instead of structured logs (32 calls / 4 modules). | The single `configure_logging` seam (`logging_config.py`) called once per **entrypoint**; library code never configures logging (ADR-003). | 2026-06-23 |
| [KI-3](../docs/archive/KNOWN_ISSUES-resolved-001.md) | The win32 `cu130` torch wheel instantly segfaults on a box with no usable GPU. | Per-machine torch via mutually-exclusive uv extras. **Never `cu130` on a GPU-less box** — `--extra cpu` there. | 2026-06 |
| [KI-7](../docs/archive/KNOWN_ISSUES-resolved-001.md) | The open-vocabulary per-document LLM concept graph fragmented concepts and dominated cost (36–40 calls/doc). | `concept_graph.py` is deleted; epistemics + wiki read the Node-A/B curated skeleton. Don't reintroduce per-doc open-vocabulary extraction. | 2026-07-07 |
| [KI-9](../docs/archive/KNOWN_ISSUES-resolved-001.md) | The frozen build downloaded model weights on first run — ≈218 s of cold start, and offline launch never went green. | `doc_assistant_api.spec` stages a symlink-free, blob-less HF hub cache into the freeze; the entrypoint points `HF_HOME` at it and forces offline. Don't drop that staging block (it is why the installer is ~1.5 GB). | 2026-06-24 |
| [KI-10](../docs/archive/KNOWN_ISSUES-resolved-001.md) | The frozen build's bundled `certifi` rejected corporate-MITM'd HTTPS — startup crash on-proxy. | `truststore.inject_into_ssl()` at the API entrypoint, so outbound TLS uses the **OS** trust store. Removing it re-breaks every proxied box. | 2026-07-09 |
| [KI-11](../docs/archive/KNOWN_ISSUES-resolved-001.md) | chromadb didn't persist its hnsw `.bin` segments under a **non-ASCII** path → a corpus that wouldn't reload (any accented username). | `config._chroma_base` relocates **only** the Chroma dirs to an ASCII `%PROGRAMDATA%` path, namespaced by a hash of the data home. Never point Chroma straight at the user data home. | 2026-06-24 |
| [KI-12](../docs/archive/KNOWN_ISSUES-resolved-001.md) | Inverse orphan: chunks in both Chroma stores with no `Document` row (the F1 write reorder's narrow window). | `main()` subtracts `inverse_orphans` from the dedup set so the row is recommitted next ingest. Keep the dedup gate an **intersection** of both stores. | 2026-06-26 |
| [KI-13](../docs/archive/KNOWN_ISSUES-resolved-001.md) | `seed_concepts` mined `Keyword` rows — but nothing in the codebase ever wrote one, so the concept-skeleton seam was dead on real data. | `knowledge/keywords.py`: a deterministic, zero-LLM corpus TF-IDF extractor writing `Keyword(source="extracted")` + `document_keywords`. | 2026-07-01 |
| [KI-14](../docs/archive/KNOWN_ISSUES-resolved-001.md) | PyMuPDF4LLM `==> picture … intentionally omitted <==` placeholders polluted chunks and keywords (1027 hits / 24 papers). | `extractors.strip_image_placeholders` in the extract→markdown step, before chunking and keywording. | 2026-07-02 |
| [KI-15](../docs/archive/KNOWN_ISSUES-resolved-001.md) | `epistemics.concepts_in_text` searched chunk text for concept **UUIDs** (Node A's ids) instead of labels → 0 claims on the real corpus, silently. | Match on labels/aliases via `concept_skeleton.compile_boundary_pattern`. The trap generalises: **an id-space change upstream can silently no-op a downstream matcher** — it survived three sprints. | 2026-07-08 |
| [KI-16](../docs/archive/KNOWN_ISSUES-resolved-001.md) | Vendored `docs_check` scanned `.claude/worktrees/` (+ their `.venv`) → ~70 phantom header errors. | cpc 1.2.2 re-vendor; rule 1 now shares rules 3–4's parts-exclusion. Re-vendoring an older cpc brings it back. | 2026-07-16 |
| [KI-22](../docs/archive/KNOWN_ISSUES-resolved-001.md) | Declared base dep `send2trash` was absent from the venv → `DELETE /api/library/documents` 500s at call time (lazy import, so startup stayed green). | The dep is installed, and `tests/unit/test_declared_dependencies.py` guards it. **The real lesson:** the failures were carried in the baton as "pre-existing venv drift" for several sessions — a declared dep missing from the venv is a defect, never drift. | 2026-07-19 |
| [KI-23](../docs/archive/KNOWN_ISSUES-resolved-001.md) | Additive schema columns never reached a running install: `init_db()` ran on ingest only, so a chat-only user (and the packaged build) silently drifted for weeks. | The API lifespan calls `init_db()` and logs `schema_migrated_at_startup` / `schema_current`. Don't remove that call — it is the only migration trigger the app has. (Filed as KI-20; renumbered — the append-only DEVLOG/baton of 2026-07-20 still say KI-20.) | 2026-07-20 |
| [KI-24](../docs/archive/KNOWN_ISSUES-resolved-001.md) | `ingest --rebuild` deleted every `Document` row, so folder membership, tags, metadata overrides and **all figure chunks** silently vanished while the folders still listed. | The rebuild **keeps** its rows; `_sweep_rebuild_rows` removes only what the run did not reproduce, and protects a source that is present but produced nothing. Never reintroduce `delete(DBDocument)` there. | 2026-07-20 |
