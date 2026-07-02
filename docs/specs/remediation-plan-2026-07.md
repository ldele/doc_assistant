# Spec — Remediation plan R1–R7: review findings → evaluate + fix (2026-07)

**Status:** 🔒 PLANNED, NOT BUILT (2026-07-02, Claude Code review session). Seven one-session
increments from the 2026-07-02 direction + algorithmic review. Each increment is one PR
(build protocol: one PR per session, never bundle). Execution owner: Claude Code.
**Sources:** the review conversation (2026-07-02); `tests/eval/baselines/rg001_concept_skeleton_2026-07-01.md`
(runs a–c); `tests/eval/baselines/rg001_concept_skeleton_multidomain_2026-07-01.md`; KI-7 / KI-8 / KI-14
(`.claude/KNOWN_ISSUES.md`); RG-001/008/009 (`.claude/RIGOR_TODO.md`).
**Pattern references:** Enrichment-Layer Pattern; locked-settings rule (changes to retrieval go through the
eval harness with `--repeat`, beat the control, record a baseline — `.claude/CONTEXT.md`); cost discipline
(every run below is $0 — deterministic or local embedder only; no paid LLM anywhere in R1–R7).

**Goal (the why).** The 2026-07-01 validation runs produced honest negatives, but three of the blockers are
**method-level and confounded the measurements**: extraction placeholder noise (KI-14) polluted both the RAG
chunk store and the keyword candidates; substring presence matching fabricates co-occurrence edges (BERT→SBERT
inflation); DF-only keyword scoring cannot separate domain concepts from academic boilerplate (proven on two
corpora). R1–R4 remove those confounds deterministically; R5 is the clean re-measurement and the go/no-go
decision on the edge model + gap layer. R6–R7 are independent: R6 fixes a handicapped BM25 arm and small
pipeline correctness nits in the core answer path; R7 stops the UI surfacing integrity markers derived from
the superseded graph (KI-7) — the one finding that works *against* the product's core promise.

**Dependency order.** R1, R2, R3 are independent of each other; all three block R5. R4 before R5
(recommended — it is what makes provenance discriminating on a partial doc graph). R6 and R7 are independent
of the chain and can run any time. Suggested sequence: R1 (also improves the core product immediately) →
R7 (small, product trust) → R2 → R3 → R4 → R5 → R6 when an eval-capable box is free.

---

## R1 — Ingest hygiene: strip PyMuPDF4LLM image placeholders (closes KI-14)

**Why first:** the placeholders are retrievable noise in the **RAG chunk store** (the answer path), not just
keyword junk, and they invalidated part of the multi-domain run (11/13 "communities" were placeholder noise).

**Evaluate before (record the numbers in the PR's DEVLOG entry):**
- Count occurrences in each corpus's cache: grep `intentionally omitted` under `<data-home>/cache/`
  (multi-domain measured 1,027 across 24 papers on 2026-07-01; re-count on the box you run on).
- First **grep the actual caches for all frame variants** (`==> … <==`) before writing the regex — do not
  assume the exact `picture [W x H]` wording is the only form PyMuPDF4LLM emits.

**Fix (two seams — extraction time AND existing caches):**
1. New pure function `strip_image_placeholders(md: str) -> str` in `src/doc_assistant/extractors.py`,
   applied inside `extract_to_markdown` ([extractors.py:130](../../src/doc_assistant/extractors.py)) so every
   future extraction is clean. Anchor the regex to the `==> … <==` frame (whole line, tolerant of `**`
   emphasis and surrounding blank lines), not to the word "picture".
2. **The cache is the source of truth and `--rebuild` does NOT re-extract** — `load_or_extract`
   ([ingest/cache.py:34](../../src/doc_assistant/ingest/cache.py)) trusts mtime freshness, and `--rebuild`
   ([ingest/__init__.py:350](../../src/doc_assistant/ingest/__init__.py)) wipes the stores but re-reads the
   fresh cache. So ship an idempotent normalization runner `scripts/normalize_cache.py` (dry-run default,
   `--apply`) that rewrites each cached `.md` through `fsutil.atomic_write_text` **only when stripping changes
   the content**. Second `--apply` run must report 0 changes (idempotence guard).
3. Then run a plain `python -m doc_assistant.ingest` (no `--rebuild`): `doc_hash` is computed over the cached
   markdown ([ingest/cache.py:48](../../src/doc_assistant/ingest/cache.py)), so normalized docs get new
   hashes → the existing content-changed machinery sweeps the old-hash chunks and re-chunks/re-embeds only
   the affected documents. Document ids are reused (F1 `_existing_document_id`), so citations/keywords/concept
   links survive. Figure dirs keyed by old hashes are swept by `cleanup_orphan_figures` — re-run the figure /
   Marker-table sidecars afterwards **only** on corpora that used them; then re-run `compute_doc_vectors`,
   `extract_citations`, `extract_keywords --force` (all $0, idempotent, host per KI-5).

**Guard tests:** unit — strip removes single/multiple placeholder lines with varying dimensions and `**`
frames, preserves adjacent prose and real markdown structures (tables, headings); integration — normalize
runner dry-run writes nothing; `--apply` twice → second run reports 0 rewrites.

**Verify after:** cache grep = 0; keyword candidates on the multi-domain corpus no longer contain
`intentionally omitted` / `x 12` / `br 1`; KI-14 → RESOLVED with the numbers.

**DoD:** gate green (`ruff` / `ruff format` / `mypy --strict src` / `bandit` / `pytest tests/unit
tests/integration`); both corpora normalized + re-ingested on their host boxes; DEVLOG entry; KI-14 closed.

---

## R2 — Word-boundary presence matching (the RG-009 lever)

**Why:** substring matching sits at the **top of the edge funnel** — BERT firing 550× via
SBERT/ColBERT/RoBERTa doesn't just inflate `n_mentions`, it fabricates co-occurrence edges from BERT to
everything in those papers, inflating the exact density metric RG-008 gates on. Running R5 without this fix
produces a third confounded negative.

**Fix:** `match_presence` ([concept_skeleton.py:154](../../src/doc_assistant/concept_skeleton.py)) — replace
`low.count(form)` (line ~182) with a per-form compiled regex count. **Do not use `\b`**: it misbehaves on
surface forms with edge `-`/`+` characters (`gpt-4`, `c++`). Use alnum lookarounds consistent with the
keyword tokenizer's notion of a word: `(?<![a-z0-9])<re.escape(form)>(?![a-z0-9])` over the casefolded text
(reference implementation shape: `epistemics.concepts_in_text`,
[epistemics.py:122](../../src/doc_assistant/epistemics.py)). Precompile once per form.
- Add `CONCEPT_SKELETON_PRESENCE_MODE` config (`"boundary"` default, `"substring"` kept as the A/B lever for
  the R5 comparison). This changes the spec-locked Decision-2 primitive — update
  `docs/specs/concept-graph-redesign.md` (word-boundary was already its named upgrade lever) + DEVLOG.
- Known accepted looseness (document, don't fix now): overlapping alias spans double-count `n_mentions`
  (e.g. both `passage retrieval` and `dense passage retrieval` as forms of one concept firing on one span).
  `n_mentions` is reporting-only today; longest-match span consumption is the upgrade if it ever gates.

**Guard tests:** `sbert` / `colbert` / `roberta` text does **not** match form `bert`; `bert.` / `(bert)` /
start-and-end-of-text do match; `gpt-4` matches exactly and not inside `gpt-4o`; substring mode still
reproduces old behaviour; outputs stay sorted/deterministic.

**Verify after ($0, host):** dry-run `build_concept_skeleton` on the public corpus in both modes; append a
before/after presence table to `tests/eval/baselines/rg001_concept_skeleton_2026-07-01.md` (BERT mentions,
per-K edge counts/density). Expect BERT presence to drop from 10-doc/550-mention to its true count —
indicative, re-derived on the box.

---

## R3 — Keyword termhood: contrastive scoring + nested-term fix (keywords.py)

**Why:** both existing modes fail by construction, measured on two corpora: per-doc TF-IDF selects df≈1 terms
(per-paper cliques); `corpus_band`'s `df·(1+ln tf)` ([keywords.py:359](../../src/doc_assistant/keywords.py))
is monotone in df, so it always grabs the most-shared = most-generic terms. The terminology-extraction
literature's answer is **contrast against a reference corpus** ("weirdness" ratio) + **C-value** for nested
multi-word terms. Both deterministic, zero-LLM, and the reference is external — so this does **not** violate
the no-tuning-against-the-corpus constraint.

**DECIDED (user, 2026-07-02): Option A — `wordfreq` dependency** as the reference-frequency source
(maintained, compact, `zipf_frequency(token, "en")`; one `uv add`, data ships with the package so it works
offline after install). The executing session records the mini-ADR in `docs/decisions/` documenting this
against the rejected alternatives:
- Option B (rejected): repo-frozen frequency table — no new dep, but must be built once and committed; same math.
- Option C (rejected as sole gate): Academic-Word-List stoplist only — targets academic boilerplate but gives
  no general contrast; may still be layered on top of A later (published AWL, not tuned on our corpora).

**Fix 1 — nested n-gram discount (fixes the unigram-dominance bias everywhere):**
`candidate_terms` ([keywords.py:267](../../src/doc_assistant/keywords.py)) counts every occurrence of
"dense passage retrieval" also as its contained bigrams/unigrams, so a contained term's count is always ≥ its
container's — frequency-ranked pools and rankings systematically favor single words. Add a C-value scoring
step: `C(a) = log2(|a|+1) · (f(a) − mean f(b) over observed candidates b ⊋ a)` (the `+1` variant keeps
unigrams eligible; guard-test the *property* — a phrase seen mostly inside a longer phrase is discounted —
not the constant).

**Fix 2 — `mode="contrastive"`:** score = `(1 + ln tf_corpus) · weirdness(term)` where `weirdness` =
observed corpus frequency ÷ reference frequency; for n-grams take the min (or geometric mean) over tokens;
smooth reference OOV so unseen technical tokens rank maximally weird (that is the desired behaviour). New
CLI `--mode contrastive` on `scripts/extract_keywords.py`; config knobs following the existing `KEYWORD_*`
style. Defaults chosen a priori and frozen **before** looking at output (record them in the ADR).

**Fix 3 — orphan sweep:** `_persist_keywords(force=True)`
([keywords.py:466](../../src/doc_assistant/keywords.py)) clears links but orphaned `Keyword` rows persist and
pollute `seed_concepts` candidates forever. After clearing, delete `Keyword` rows with zero remaining
document links and no matching promoted `Concept` label.

**Optional ride-alongs (small, same module cluster — drop if the PR grows):**
`abstract_candidates` ranking ([concept_semantics.py:81](../../src/doc_assistant/concept_semantics.py)) is
frequency-first so the "multi-word phrases surface first" docstring only holds on ties — make phrase length
primary (or score `count·len`); `anchor_ranked_candidates` pool
([concept_semantics.py:209](../../src/doc_assistant/concept_semantics.py)) pools by raw frequency and
inherits the unigram bias — pool by per-doc TF-IDF or C-value instead. Plus a 20-minute measurement: count
`extract_abstract` recall over both corpora (the regex requires a standalone "Abstract" heading; inline
"Abstract—…" styles silently degrade the anchor to title-only).

**Guard tests:** C-value toy case (nested phrase outranks its unigrams when it carries the occurrences);
weirdness (OOV tech token outranks a common English word at equal tf); orphan sweep (force re-extract leaves
no zero-link unpromoted rows); determinism/tie-breaks.

**Verify after ($0, both corpora):** run `per_doc` / `corpus_band` / `contrastive` side by side; record df
histograms + a qualitative top-60 table in a new baseline `tests/eval/baselines/rg001_keyword_termhood_<date>.md`.
Indicative acceptance (pre-registered): the multi-domain top-60 contains recognizable per-domain terms
(cosmology / statistical-mechanics vocabulary) and drops `consider / introduce / respectively`-class words.
**No threshold tuning against either corpus.**

---

## R4 — Graded provenance strength (ratio, not boolean)

**Why:** `_add_provenance` ([concept_skeleton.py:244](../../src/doc_assistant/concept_skeleton.py), any-pair
test at ~line 269) asks "does *any* doc containing A link to *any* doc containing B?" — measured
non-discriminating (run (a): similarity 100%, citation ~88%). A **ratio** stays deterministic and becomes a
relative signal on a partial doc graph.

**Fix:** per edge and provenance token compute
`strength = |linked ∩ candidate pairs| / |candidate pairs|`, `candidate pairs = {(da, db) : da ∈ docs(A),
db ∈ docs(B), da ≠ db}`. Keep the token in the `provenance` set when strength > 0 (the no-edge-creation
invariant is untouched — this only annotates existing co-occurrence edges). Store strengths (new
`provenance_strength` mapping on `SkeletonEdge`; co-occurrence keeps its count field).
- `edge_weight` ([concept_skeleton.py:318](../../src/doc_assistant/concept_skeleton.py)): **preserve the
  locked invariant** (an edge with more provenance tokens always outranks one with fewer). Keep the integer
  part = token count; split the fractional tiebreak, e.g. `0.5·mean(strengths) + 0.5·(1 − 1/(1+cooc))`,
  still bounded < 1. Exact split is an in-PR decision; the invariant gets a guard test.
- Touch **both** serialization directions (`skeleton_to_dict` / `skeleton_from_dict`,
  [concept_skeleton.py:464](../../src/doc_assistant/concept_skeleton.py)) + `_write_skeleton_rows`
  (additive `strength_json` column or fold into `provenance_json`) + the `_graph_version` payload.
- **Honest expectation:** on a saturated doc graph (public corpus, 100% similarity pairs) every strength is
  1.0 — no discrimination there *by construction*. The payoff is on partial graphs (multi-domain measured
  43%). Expected, not a failure.

**Guard tests:** ratio math on small cases; no-edge-creation preserved; multi-token-beats-single invariant;
serialization round-trip; byte-identical rebuild on identical inputs.

---

## R5 — The decision run: multi-domain revalidation (RG-001/008/009) + gap wizard-of-oz

**Not a code PR** — a measurement session. Precondition: R1–R3 landed (R4 recommended). Host box with the
multi-domain data home (`DOC_DATA_DIR` pointed at it; the 2026-07-01 runs used a gitignored
`data_multidomain/`, 24 arXiv papers across 6 domains — recreate from its `manifest.json` if absent). $0.

**Protocol (pre-register the acceptance bands before running):**
1. R1 normalize + re-ingest; re-run metadata / citations / doc-vectors sidecars.
2. R3 `contrastive` extraction → candidates → **user curates** (this is the redesign's human-in-the-loop and
   run (a) explicitly flagged the vocabulary as un-signed-off; budget 30–60 min with `suggest_concepts
   --from-abstracts` / `--anchor-ranked` as aids and a SPECTER2 `--near` dedupe pass, then `seed_concepts
   --add` / `--promote`).
3. Skeleton dry-run sweep `--min-cooccurrence {1..5}`, presence mode boundary (substring once, for the A/B).
4. Record per K: edges / density / communities / isolated; provenance **strength distributions** (should
   spread, not sit at 1.0); presence recall against a ~20-term hand checklist (RG-009).
5. Indicative acceptance bands (from run (a)'s precise-vocabulary regime, chosen a priori): density roughly
   15–35% at the chosen K; communities that map to domains/topics rather than to single papers or noise.
6. **Gap wizard-of-oz (validates the destination before more road-building):** compute ADR-004's Tier-1 gap
   signals on the resulting skeleton (scripted or by hand) and answer in prose: *do ≥3 signals point at
   something a researcher would actually act on?*

**Outcomes (either way, record an ADR + baseline `rg001_concept_skeleton_r5_<date>.md`):**
- PASS → lock `CONCEPT_SKELETON_MIN_COOCCURRENCE` + presence mode (locked-settings table + spec), close
  RG-008/009, unblock ADR-004 Tier-1, proceed to Node B (PR-B).
- FAIL → descope consciously: the glossary + presence layer has standalone user value (curated definitions,
  per-concept document lookup, SPECTER2 dedupe) — keep it as the product; park the edge model + gap layer.

---

## R6 — Core retrieval: BM25 preprocessing + pipeline hygiene (eval-gated)

**Why:** the BM25 arm is handicapped — `BM25Retriever.from_documents(...)`
([pipeline.py:115](../../src/doc_assistant/pipeline.py)) uses LangChain's default `preprocess_func`, verified
to be a bare `text.split()`: case-sensitive, punctuation attached (`BM25?` never matches `bm25`). The 0.4
ensemble weight is buying less than intended. This is the most likely free win in the answer path — and it
must land **before** any 0.4/0.6 weight sweep, because fixing the tokenizer moves the weights' optimum.

**Evaluate before:** control run on the public corpus per `tests/eval/TESTING.md` with `--repeat`
(locked-settings rule).

**Fixes:**
1. Pass `preprocess_func` (casefolded, tech-token-aware — reuse `keywords.tokenize`) to
   `BM25Retriever.from_documents`; LangChain applies the same function to queries at retrieval time.
2. Candidate dedup key ([pipeline.py:156](../../src/doc_assistant/pipeline.py)) — `doc_hash + first 50 chars`
   silently collapses distinct chunks sharing a 50-char prefix (repeated headers; KI-14 placeholder-prefixed
   chunks pre-R1). Use a full-content hash.
3. `expand_query` ([pipeline.py:253](../../src/doc_assistant/pipeline.py)) — when the LLM returns valid JSON
   that isn't a list, `variations = [query]` then line 262 prepends `query` again → the same query runs the
   ensemble twice. Set `[]` (matching the `except` branch) + regression test.
4. Probe (may be a no-op): in parent-child mode only candidates carrying `parent_text` are returnable
   ([pipeline.py:186](../../src/doc_assistant/pipeline.py)). Assert every PC-collection row carries
   `parent_text` (temp-ingest test), or document the intended retrieval path for any legitimate exceptions
   (figure/table chunks).

**Verify after:** eval harness with `--repeat` vs the control; ship the preprocess change only if it beats or
matches the control beyond its variance (fixes 2–3 are correctness nits and ride regardless; present results
as reproducible/indicative with the re-run command). Record `tests/eval/baselines/bm25_preprocess_<date>.md`.
**Then, separate follow-up experiment (own session):** add the `--bm25-weight` flag (CONTEXT open question)
and sweep the vibes-locked 0.4/0.6.

---

## R7 — KI-7 containment: stop surfacing superseded-graph markers by default

**Why:** the app's differentiator is the integrity layer, yet the live `contested` / `superseded` chips are
computed from the **superseded** open-vocabulary graph (KI-7) through a coarse containment join (KI-8) —
noise wearing the trust layer's uniform. Full KI-7 retirement stays gated on Node B; this is the cheap
containment move.

**DECIDED (user, 2026-07-02): option (a) — config kill-switch, default off.** New
`EPISTEMICS_MARKERS_ENABLED` (default `false`) gating `_attach_markers` in
`src/doc_assistant/chat_controller.py`. Chip is quiet-on-clean already, so the UI needs no change; the M0/M1
parity guarantee (byte-identical when markers absent) becomes the default path. Node B flips the default back
on with trustworthy data. No further input needed — R7 is ready to build. Rejected: (b) keep on + label the
chip "experimental" in the renderers — still surfaces KI-7 noise under the trust layer's banner; (c) full
retirement now — the known connected change across `epistemics.py` → `chat_controller.py` /
`compute_epistemics.py` / `wiki.py` + tests (KI-7 "Cleanup when built"), only worth it bundled with Node B.

**Guard tests (option a):** markers absent by default (parity test passes on the default path); present when
the flag is on; flag read once per turn. Update KI-7/KI-8 notes + `feature-7d-knowledge-currency.md`.

---

## Cross-cutting execution rules (restated, not new)

- One PR per session; TDD guard tests; stage + summarize + **stop — never commit/push** (cpc §13).
- Gate per PR: `uv run ruff check src tests` · `uv run ruff format --check src tests` ·
  `uv run mypy --strict src` · `uv run bandit -r src apps` · `uv run --no-sync pytest tests/unit
  tests/integration`.
- $0 constraint holds across R1–R7: no paid LLM call anywhere; embedders are local; enrichment runners on
  the host, not a sandbox (KI-5).
- Every measurement lands as a baseline under `tests/eval/baselines/` with the re-run command; present
  results as reproducible/indicative, never as definitive verdicts.
- DEVLOG entry per PR (what / why / rejected / opens); RIGOR_TODO updates via the rigor-gate protocol;
  decisions recorded in `docs/decisions/` — R3 reference source (`wordfreq`) and R7 option (a) are
  **user-decided 2026-07-02** (executing sessions write the ADRs); the R5 outcome ADR is written when R5 runs.
