<!-- status: active · updated: 2026-07-19 · class: disposable -->

# REVIEW 2026-07-19 — knowledge layer vs specs/ADRs: 0-doc robustness + 0→10k-doc scaling

**Trigger (user directive):** sessions keep over-optimizing the algorithms to the *current*
corpora (47 docs on this box, 76 on the work box). The product contract is the opposite: every
feature must degrade honestly at **0 documents** and scale to **~10,000** — no corpus-tuned magic
numbers. This review audits the whole `src/doc_assistant/knowledge/` layer (post-ADR-023) against
its own specs/ADRs under four lenses: **zero-doc**, **scale**, **corpus-tuned**, **conformance**.

**Method.** Four independent read-only review passes (one per cluster: concept skeleton ·
keywords/families · gaps · wiki/epistemics), each against its contract docs
(`feature-concept-graph.md`, ADR-008/015/017/018, ADR-006, `feature-tag-families.md`, ADR-004,
`feature-gap-detection.md`, `feature-7d-knowledge-currency.md`, the frozen monolith's deferred
items). Every finding must quote the code line it stands on. The seven highest-stakes claims
(CS-1, CS-5, GP-1, GP-4, KW-4, WE-5, WE-7) were **independently re-verified** against the source
before this document was written — all seven hold. Nothing was measured live this session (no
LLM/eval runs); complexity/memory claims are traced from code + the recorded baselines they cite.

**Routing.** Grouped issues → `.claude/KNOWN_ISSUES.md` **KI-18..KI-21** + a KI-8 correction;
measurement debt → `.claude/RIGOR_TODO.md` **RG-016..RG-019**; ROADMAP row C4.

---

## Executive verdict

1. **Zero-doc behavior is largely honest — the discipline held.** Across all four clusters,
   public entry points return clean empty states (`([], {})`, `None`→404-with-rebuild, `[]`,
   honest empty reports); empty-aggregate crashes are guarded. Exceptions: the wiki builder
   crashes (`OperationalError`) on a never-ingested DB (WE-1), `build_epistemics` raises on a
   missing skeleton for any non-CLI caller (WE-9), and nothing in the test suite *pins* the
   0-doc contract (GP-7) — so it survives by habit, not by gate.
2. **Nothing survives the road to 10k docs unchanged.** Every cluster has at least one
   corpus-linear-or-worse hot path that was invisible at n≈50: the presence stage loads every
   chunk into RAM unpaginated and the provenance step is a per-edge doc×doc Cartesian product
   (CS-1/CS-2); the keyword extractor holds the whole corpus + per-occurrence term streams in
   memory (KW-1); family Tier-2 is O(n²) pairwise cosine (KW-2); the epistemics projection is a
   full-recompute O(chunks × concepts) regex scan with a 512-pattern regex-cache cliff (WE-3);
   Node B and gap_suggest and the wiki synthesizer all make **unbounded per-doc/per-concept LLM
   calls** (CS-4, GP-2, WE-10). First to fall over: memory (CS-2, KW-1), then rebuild time
   (CS-1, WE-3), then LLM-call volume.
3. **The corpus-tuned constants are real and localized — the complaint is confirmed.** The
   worst offenders: `min_degree=3` is a frozen *snapshot* of a Q1 computed once at 26 concepts
   while its docstring claims it is "corpus-derived" (GP-1, verified); family Tier-2's `0.86`
   cosine sits **above bge's own measured same-domain ceiling (~0.82)**, so the semantic tier
   structurally under-fires on the exact case it was built for (KW-4); `contested` fires on
   `nc >= 1` — one disputing doc — already marking 53.6% of chunks at 47 docs and saturating
   monotonically with corpus growth (WE-5, verified); the wiki still ships the absolute-cosine
   0.90 clustering the monolith recorded as "the wrong primitive", with the fix inert behind a
   default-false flag (WE-6). By contrast, `MIN_DATED_DOCS_PER_SIDE=2` and `_MIN_CONCEPT_LEN=3`
   are honest structural constants — the discipline exists; it just wasn't applied everywhere.
4. **Conformance breaks where the layer meets its lifecycle, not its math.** The in-app graph
   rebuild (ADR-017 B1) rebuilds only the skeleton and never `build_gaps`, so the view serves
   stale gaps after the exact loop the button exists to close (GP-4, verified). Curation
   stages 1–3 **hard-delete** Concept rows where ADR-018 and the spec's own Trap mandate
   demote-via-`graph_include` (CS-5, verified) — with `classify_noise` being precisely the path
   that mislabels specialist vocabulary (`cre`/`dbs`/`ntsr1`). KI-8's recorded rationale is
   arithmetically wrong in direction: a 1000-char marked chunk can never sit inside a 200-char
   parent overlap, so straddling chunks silently *lose* their markers (systematic false
   negatives), they don't double-mark (WE-7, verified).

---

## Findings — concept skeleton cluster (CS)

| ID | Lens | Sev | Where | Claim (evidence verified in code where marked ✓) |
|----|------|-----|-------|--------------------------------------------------|
| CS-1 ✓ | scale | blocker | `knowledge/concept_skeleton.py:332` `_add_provenance` | Provenance is a full Cartesian product over each edge's endpoint doc sets (`for da in src_docs: for db in tgt_docs`), run twice (citation + similarity). A 40%-present hub at 10k docs → 16M iterations per hub–hub edge; `linked` sparsity buys nothing because the loop never iterates `linked`. Fix: iterate the small `linked` set; compute `candidate` arithmetically. |
| CS-2 | scale | blocker | `concept_skeleton.py:1100` `load_presence_inputs` + `:214` `match_presence` | Presence loads **every child chunk's metadata incl. denormalized parent_text** in one unpaginated `coll.get` (multi-GB at 10k docs), then rescans the whole corpus **once per concept**. Fix: page the get; one combined-alternation scan per chunk. |
| CS-3 | corpus-tuned | major | `config.py:427` `CONCEPT_SKELETON_MIN_COOCCURRENCE=2` | Absolute chunk-co-occurrence floor validated only at 76 docs (ADR-008, explicitly scope-bound); at ~500k chunks nearly any two included concepts co-occur ≥2× → density → hairball. Fix: derive from the pair-count distribution; re-run the ADR-008 sweep ≥1k docs (RG-016). |
| CS-4 | scale | major | `concept_skeleton_enrich.py:151` `annotate_relations` | Node B = one LLM call per document, no ceiling (call count == docs with an edge-forming pair; 10k docs → up to 10k sequential completions). Fix: cap/batch; a `MAX_*` budget mirroring `MAX_VLM_CALLS_PER_DOC`. |
| CS-5 ✓ | conformance | major | `concept_curation.py:400` `remove_concepts` (+ `classify_noise` feeding it) | Curation **hard-deletes** Concept+alias rows on LLM/regex noise verdicts — against ADR-018's "no row is deleted" and the spec Trap's *demote, not delete*; cascades into families. Mitigated by dry-run default; still the wrong verb. Fix: route verdicts through `set_graph_include(id, False)`. |
| CS-6 | scale | minor | `concept_semantics.py:111` `nearest_pairs` | O(V²) pure-Python cosine over the **unfiltered** concept table (families included). Fix: vectorized/blocked similarity. |
| CS-7 | corpus-tuned | minor | `config.py:494` `ABSTRACT_CONCEPTS_TOP_K=12` · `CONCEPT_MERGE_COSINE=0.85` | Self-described "general defaults", advisory-only blast radius. Leave unless auto-applied. |
| CS-8 | scale | minor | `concept_skeleton_enrich.py:47` `DEFAULT_MAX_TOKENS=2048` | Fixed output budget regardless of pair count; truncation → `parse_annotations` returns `[]` → a concept-dense doc silently contributes zero stance. Fix: size from `len(candidates)`; warn on shortfall. |
| CS-9 | scale | minor | `concept_skeleton.py:1063` `load_glossary` | N+1 alias loads per concept. Fix: `selectinload`, as `load_concepts` already does. |

## Findings — keywords / families cluster (KW)

| ID | Lens | Sev | Where | Claim |
|----|------|-----|-------|-------|
| KW-1 | scale | major | `knowledge/keywords.py:733` (+ `load_document_texts:603`) | Extractor materializes the entire corpus text **and** per-occurrence candidate-term streams in memory at once; a single-doc re-extract still pays the full-corpus load (`corpus = load_document_texts()` unconditional). OOM before 10k docs. Fix: stream doc frequencies / persist corpus term-stats. |
| KW-2 | scale | major | `keyword_families.py:149` `_tier2_embedding` | All-pairs cosine over un-familied names: ~50M pairs × 768 dims at 10k names. Fix: blocked/ANN similarity. |
| KW-3 | scale | major | `library.py:486` `list_keyword_families` | One `COUNT(DISTINCT …)` query **per family** (N+1); `detect_family_candidates` pays it all just to read names. Fix: one grouped count; a counts-free name helper. |
| KW-4 ✓ | corpus-tuned | major | `keyword_families.py:28` `DEFAULT_EMBEDDING_THRESHOLD=0.86` | Unvalidated magic number **above bge's measured same-domain ceiling** (~0.77–0.82 band; DEVLOG's own concept-merge measurement) — and both call paths embed with bge, not specter2. Tier 2 structurally under-fires (`connectome`≈`connectomics` never fires; only substring-close pairs pass). Fix: RG-017 — re-derive on bge, or reuse the specter2 embedder the merge path already switched to. |
| KW-5 | corpus-tuned | minor | `keywords.py:747` + `config.py:476` (`corpus_band`) | df-band `[2, 0.7N]` is only meaningful at tiny n; score is df-monotone → at 10k docs returns the 60 most-generic terms (failure already recorded once). Non-default mode. Fix: percentile band or retire the mode. |
| KW-6 | corpus-tuned | minor | `config.py:478` `KEYWORD_CORPUS_TOP_K=60` | Fixed corpus-wide cap, coincidentally ≈ the current vocabulary size; caps a 10k-doc mined vocabulary at 60. Fix: scale with N or make it a display limit. |
| KW-7 | conformance | minor | `keyword_families.py:43` `_stem` | Sibilant `-es` rule over-strips (`databases`→`databas` vs `database`) → Tier-1 misses common plurals (spec defect D4, planned PR-2.5). |
| KW-8 | zero-doc | positive | (trace) | All entry points no-op honestly at 0 docs — IDF smoothing, `_cosine`/`_edit_similarity` guards, empty-group short-circuits all verified. |
| KW-9 | corpus-tuned | minor | `config.py:470` `KEYWORD_MIN_CHARS=3` → `keywords.py:368` | Sub-3-char terms are **deleted at the tokenizer** (never mined), not demoted — `v1`/`t1`/2-letter gene symbols can never be curated in. Fix: min 2 or exempt digit-bearing tokens. |

## Findings — gaps cluster (GP)

| ID | Lens | Sev | Where | Claim |
|----|------|-----|-------|-------|
| GP-1 ✓ | corpus-tuned | major | `scripts/build_gaps.py:46` `_DEFAULT_MIN_DEGREE=3` vs `knowledge/gaps.py:153` docstring | The constant is a frozen **snapshot** of a Q1 computed once at 26 concepts; the docstring claims "corpus-derived, never a guessed absolute" — the code encodes the *result*, not the *rule*, and it has already mis-tuned at 357 and at 13 concepts. Fix: compute Q1 from the loaded skeleton at runtime (fallback structural constant at tiny graphs); fix the docstring. |
| GP-2 | scale | major | `gap_suggest.py:129` | One sequential LLM call per under-connected concept, **no cap in code or config**; `assert_provider_intent` gates the run, not the volume. Plus O(thin × E) neighbour rescans. Fix: `GAP_SUGGEST_MAX_CONCEPTS` worst-N cap + one adjacency precompute. |
| GP-3 | scale | major | `gaps.py:242` `load_unsupported_claims` | Loads **every** `unsupported` claim ever persisted (unbounded in usage-time, not doc count) into an O(concepts × claims) regex sweep per rebuild; a hub's `evidence_json` accumulates every claim id forever. Fix: retention window / per-concept aggregate + sample. |
| GP-4 ✓ | conformance | major | `apps/api/main.py:232` `_default_rebuild_graph` + `concept_graph_view.py:96` | The in-app rebuild (ADR-017 B1 — built to "close the acquire loop") rebuilds **only the skeleton, never `build_gaps`**, and `load_gaps` carries no `graph_version` cross-check — the UI serves gaps from the previous skeleton (including the one the user just closed) until someone runs the CLI. Fix: chain `build_gaps` after the route's skeleton build (needs GP-1's runtime default), or version-stamp and filter rows. |
| GP-5 | corpus-tuned | minor | `apps/desktop/src/lib/ConceptGraph.svelte:55` `GAP_META` ranks | The "single_source leads" ordering is frozen in a UI rank table on the authority of RG-014 — measured at 26 concepts on the *other* corpus, unreadable from this box, and already non-transferring (RG-015). Fix: comment → RG-015; or rank by live per-kind volume. |
| GP-6 | conformance | minor | `gap_suggest.py:154` | Stochastic evidence records concept/neighbours/target but not provider/model/prompt-version — a promoted suggestion can't be audited to what produced it (ADR-004 observability). Fix: append `provider=`/`model=` to `fact_ids`. |
| GP-7 | zero-doc | minor | `tests/unit/test_gaps.py` (absence) | Zero-doc honesty is real but **unpinned** — no empty-skeleton guard test exists, so the contract can regress silently. Fix: one parametrized empty-input test over all detectors. |
| KI-17 check | — | — | `gaps.py:261` / `:277` / `:435` | KI-17's mechanics confirmed at post-move lines (+4 drift). **Its candidate fix is mis-placed as worded:** `_write_stochastic_gap_rows` only runs under `suggest and apply` and early-returns on zero suggestions — the reconcile must run **unconditionally on every `--apply`**, keyed against `load_concepts()` (graph_include-filtered), or the orphans persist on exactly the deterministic-only rebuild the KI's own repro used. |

## Findings — wiki / epistemics cluster (WE)

| ID | Lens | Sev | Where | Claim |
|----|------|-----|-------|-------|
| WE-1 | zero-doc | minor | `knowledge/wiki.py:329` `load_doc_graph` | Never-ingested DB (no tables) → uncaught `OperationalError` instead of an honest empty library (epistemics guards exactly this; wiki doesn't). Fix: catch → `([], [])` + hint. |
| WE-2 | zero-doc | major | `wiki.py:357` `load_communities` + `concept_skeleton.py:779` | A stale skeleton (dead doc PKs after re-ingest/reset) silently degrades to one-singleton-per-doc while still labeling the result `"concept-graph"` — the retired path's freshness check was dropped, not replaced; downstream = one LLM call per singleton. Fix: live-doc overlap check → `None` → cosine fallback + the CLI's "fell back" message. |
| WE-3 | scale | major | `epistemics.py:140/183/258/445` | Projection = full-recompute O(chunks × concepts) regex scan, whole corpus in RAM via one unpaginated `coll.get`, `re.compile` **inside the per-chunk loop** (currently saved by Python's 512-pattern regex cache — which dies FIFO the moment the vocabulary exceeds 512), delete-all + re-insert writes. Measured 34s at 47 docs → ~hours + GBs at 10k. Fix: hoist per-label patterns / one alternation pass; batch the get; incremental recompute keyed on `graph_version`. |
| WE-4 | scale | major | `chat_controller.py:722` → `epistemics.py:309` `load_epistemics_index` | **Flat-mode** answer path loads the entire corpus-wide marker index per turn (no WHERE); the PC path scopes correctly and PC is default, so latent — but it's the exact query shape that melts first mid-chat at scale. Fix: scope by the turn's document_ids like `load_marked_chunks`. |
| WE-5 ✓ | corpus-tuned | major | `concept_skeleton.py:699` | `contested` = `nc >= 1` — one disputing doc anywhere marks a concept regardless of supporting mass; 53.6% of chunks already marked at 47 docs (recorded, uninvestigated), monotone in corpus size → chip saturates into noise; `agreement_ratio` is computed and **never consulted**. Fix: RG-019 — gate on a named min-N / agreement floor (sibling of `MIN_DATED_DOCS_PER_SIDE`). |
| WE-6 | corpus-tuned | major | `config.py:387` `WIKI_MIN_SIMILARITY=0.90` · `:404` flag default false | The monolith's recorded "absolute-cosine is the wrong primitive" deferred item is **still the shipped default**; the threshold-free community fix exists, inert behind `WIKI_USE_CONCEPT_COMMUNITIES=false`, validated only on the 10-paper corpus; `[[links]]` stay cosine-derived on both paths. Fix: RG-018 — run the deferred validation on the current corpora, flip the default, derive links from skeleton edges. |
| WE-7 ✓ | conformance | major | `epistemics.py:234` + `config.py:258/266` | PC containment (`text not in parent_text → skip`) can never match a chunk straddling parents: `BASELINE_CHUNK_SIZE=1000` vs `PARENT_CHUNK_OVERLAP=200` makes KI-8's "straddling marks both" arithmetically unreachable — the real failure is **systematic silent marker loss** on straddlers in the default-ON, default-PC config. Fix: land pr-m1 ADR-1 option 2 (re-projection) or overlap-based matching; correct KI-8 (done this session). |
| WE-8 | conformance | minor | `epistemics.py:191` vs `concept_skeleton.py:218` | Attribution matches canonical labels only; the presence pass that generated the stances matches label+aliases — alias-only chunks carry no marker. Fix: pass the alias map through. |
| WE-9 | zero-doc | minor | `epistemics.py:418/445` | Build path raises on missing skeleton (CLI-only guard) and `--apply` on a never-migrated DB is an uncaught `OperationalError`; read paths are exemplary. Fix: empty-result + hint; guard `_write_rows`. |
| WE-10 | scale | major | `wiki.py:547/440/419` | Wiki synthesis: unbounded per-topic LLM calls (singletons are topics), re-summarizes byte-identical unchanged topics on every `--apply` (manifest diff computed *after* summarization), `_format_material` has no docs-per-cluster cap (multi-MB prompt → swallowed into an empty note). Fix: skip unchanged `topic_id`s; cap material with "and N more". |
| WE-11 | corpus-tuned | minor | (inventory) | `_MIN_CONCEPT_LEN=3`, `MIN_DATED_DOCS_PER_SIDE=2` — honest structural constants (keep); `WIKI_MIN_CITATIONS=3`, `WIKI_CHUNK_SAMPLE=3` — harmless budget knobs. |

---

## The corpus-tuned-constants inventory (the user's core complaint, answered)

| Constant | Value | Class | Verdict |
|----------|-------|-------|---------|
| `MIN_DATED_DOCS_PER_SIDE` | 2 | structural (named, definitional) | **Keep** — the model to copy. |
| `_MIN_CONCEPT_LEN` | 3 | structural (named, in-line rationale) | Keep. |
| `CONCEPT_SKELETON_MIN_COOCCURRENCE` | 2 | validated **at 76 docs only** | Re-derive relative to corpus (RG-016). |
| `_DEFAULT_MIN_DEGREE` | 3 | frozen snapshot of a 26-concept Q1, docstring claims otherwise | Make the *rule* executable (runtime Q1) (RG-016). |
| Gap-kind rank table (UI) | single_source first | inherited from unreadable RG-014; failed transfer once | Re-rank from live volume (RG-016). |
| `DEFAULT_EMBEDDING_THRESHOLD` (families) | 0.86 | magic; **above bge's measured ceiling** | Re-derive on bge or switch embedder (RG-017). |
| `KEYWORD_MIN_CHARS` | 3 | magic; deletes instead of demoting | Lower/exempt digits. |
| `KEYWORD_CORPUS_TOP_K` | 60 | magic; ≈ current vocab size | Scale with N. |
| `corpus_band` df-band | [2, 0.7N] | structurally scale-fragile mode | Percentile or retire. |
| `contested` trigger | nc ≥ 1 (implicit) | unnamed, unvalidated; saturating | Min-N/agreement gate (RG-019). |
| `WIKI_MIN_SIMILARITY` | 0.90 | recorded-wrong primitive, still default | Validate + flip communities default (RG-018). |
| `CONCEPT_MERGE_COSINE` / `ABSTRACT_CONCEPTS_TOP_K` | 0.85 / 12 | advisory-only | Leave. |

## Prioritized fix plan

- **P0 — correctness/conformance at any size (small, no eval needed):** CS-5 demote-don't-delete
  in curation; GP-4 chain `build_gaps` into the rebuild route; KI-17 reconcile hoisted to every
  `--apply` (per the placement correction above); WE-1/WE-9 zero-state guards; GP-7 empty-input
  guard tests (pin the 0-doc contract in CI).
- **P1 — the scale cliffs (mechanical rewrites, no behavior change):** CS-1 invert the provenance
  loop; CS-2 page + single-pass presence; WE-3 hoist patterns + batch + incremental; KW-1 stream
  the extractor; KW-2/CS-6 blocked similarity; KW-3 grouped counts; WE-4 scope the flat index;
  WE-10 skip-unchanged topics + material cap; GP-3 bounded claims sweep.
- **P2 — the tuned constants (each needs its experiment, rigor-gated):** RG-016 (co-occurrence
  floor + min_degree rule + kind ranking at ≥1k synthetic docs), RG-017 (family threshold on
  bge), RG-018 (wiki communities default flip), RG-019 (contested min-N + marker-density
  recheck). **Do not hand-tune any of these without the measurement** — that is the exact
  failure mode this review exists to stop.
- **LLM-volume budgets (design decision, then small code):** CS-4 / GP-2 / WE-10 caps — one ADR
  can set the budget policy for all three (they are the same decision).

**Bottom line:** the layer is honestly built at its current size — zero-doc discipline held,
Enrichment-Layer boundaries held, provider gating held. What it is *not yet* is scale-shaped:
every cluster carries at least one O(corpus)-or-worse hot path and at least one constant that
encodes n≈50. All of it is fixable mechanically or by experiment; none of it requires redesign.
