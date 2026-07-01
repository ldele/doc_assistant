# RG-001 / 008 / 009 — concept-skeleton edge-precision + presence-recall validation

**Run:** 2026-07-01, RTX box (CPU torch venv, `uv run --no-sync`), host DB.
**Subject:** `concept_skeleton.py` Node A (deterministic, zero-LLM) over the public 10-paper corpus.
**Cost:** $0 — no LLM calls (deterministic skeleton; regex citation/metadata extraction).
**Verdict:** the graph is **NOT yet certifiable as *usable*** — thresholds are **NOT locked**. RG-008 + RG-009
stay `open`. The gap-detection layer (ADR-004) stays blocked (its Tier-1 signals are defined relative to the
edge set; on a near-complete graph they are meaningless).

## Protocol / environment

- Corpus: 10 papers (DPR, ColBERT, HyDE, RAG-Lewis, SBERT, BERT-rerank-Nogueira, BGE, SPECTER2, LLM-judge,
  AI-usage-cards). `documents`=10, parent-child Chroma present. `doc_similarities`=90 (45 undirected pairs =
  fully connected — same-domain saturation). `citations`=422 parsed / **7 internal library links**.
- **Prerequisites had to be built first (were empty on this box):**
  - `keywords`=0 and **nothing in the codebase writes `Keyword` rows** → the intended vocabulary seam
    (`scripts/seed_concepts.py`, which mines `Keyword` candidates) is **non-functional on real data**. Seeded
    a **provisional** 30-concept vocabulary directly instead (grounded in the corpus; 82 surface forms;
    aliases chosen to avoid case-folded-substring false positives). This vocabulary is un-signed-off.
  - `citations`=0 and doc metadata (`title`/`authors`/`year`) all NULL → ran `extract_doc_metadata --apply`
    (regex, free) then `extract_citations --apply --force` (regex, free) → 7 internal citation links.
- Tool: `scripts/build_concept_skeleton.py --min-cooccurrence K` (dry-run sweep) + `--apply` at K=2.
- Reproduce: seed vocab → `build_concept_skeleton --min-cooccurrence {1..5}`; inspect `concept_presence` /
  `concept_edges`.

## RG-008 — edge density / precision (blocks-ship)

**Max possible edges (undirected):** C(n,2).

Full provisional vocabulary (n=30 concepts, C=435):

| min_cooccurrence | edges | density | communities | isolated |
|---|---|---|---|---|
| 1 | 260 | 60% | 4 | 1 |
| 2 (config default) | 201 | 46% | 4 | 1 |
| 3 | 164 | 38% | 7 | 3 |
| 4 | 137 | 31% | 9 | 5 |
| 5 | 118 | 27% | 10 | 6 |

Precise vocabulary — the 9 broad hubs (present in ≥7/10 docs) removed (n=21, C=210):

| min_cooccurrence | edges | density | communities | isolated |
|---|---|---|---|---|
| 1 | 92 | 44% | 3 | 0 |
| 2 | 57 | **27%** | 4 | 1 |
| 3 | 43 | **20%** | 6 | 2 |

**Observations (numbers, not judgement):**
1. **Vocabulary breadth dominates density, not `min_cooccurrence`.** Dropping 9 broad hubs cut edges
   201 → 57 at K=2 (a 72% reduction; 46% → 27% density). Raising K from 2 → 5 on the full vocab only went
   46% → 27%.
2. **The broad hubs are the driver.** 6 concepts sit in ≥8/10 docs — BERT (10), language model (9),
   fine-tuning (9), relevance (9), transformer (9), open-domain QA (8) — and hold the top degrees (text
   embedding 25, fine-tuning 25, BERT 24, transformer 23, language model 22; max degree 29). Their edges are
   generic chunk co-presence, not precise concept relations.
3. **Provenance is non-discriminating.** similarity-provenance annotates **100%** of edges (57/57 even on the
   precise vocab at K=2) and citation ~88% — because `doc_similarities` is a fully-connected same-domain graph
   and the broad concepts span nearly every cited doc pair. This is the KI-7 / Feature-6 doc-vector saturation,
   re-confirmed on the new skeleton. It is a **corpus property** (same-domain, 10 docs), not fixable by the
   vocabulary or threshold levers; it should differ on a larger cross-domain corpus.

## RG-009 — presence recall (degrades)

- 29/30 concepts surfaced in ≥1 doc; **1 miss** — "query expansion" (0 docs): its surface forms don't occur
  in these papers / alias gap (the curation/alias-coverage bound this item measures).
- **Substring inflation confirmed:** "BERT" = 550 mentions across all 10 docs because case-folded substring
  match also fires on SBERT / ColBERT / RoBERTa / DistilBERT. This over-attributes presence → the documented
  **word-boundary-matching upgrade lever** (as in `epistemics.concepts_in_text`) is needed before precision
  can be trusted.

## Recommendations (separate from the observations above — NOT yet applied/locked)

1. **Do not lock `CONCEPT_SKELETON_MIN_COOCCURRENCE` and do not mark the graph usable.** With a *precise*
   vocabulary, K=2–3 gives the working 20–27% density range — but that is conditional on (2)+(3) below.
2. **Curate the vocabulary (user sign-off).** Drop or folder-scope the broad hubs (≥7-doc concepts); the
   provisional set was seeded by Claude, not the user, and it drives the result.
3. **Add word-boundary presence matching** (kills the BERT-family substring inflation) before trusting
   precision/recall.
4. **Re-run on a larger, multi-domain corpus** to make similarity/citation provenance discriminating (the
   private multi-paper-per-topic set) — on 10 same-domain papers, provenance carries no precision signal.
5. **Gap-detection (ADR-004) stays blocked** on a usable graph — i.e. on (1)–(4) closing.

## Re-run (b), 2026-07-01 — keyword-grounded vocabulary (the KI-13 extractor closed)

After building the TF-IDF keyword extractor (KI-13 fix), re-ran with a vocabulary promoted from the
**extracted** candidates instead of the hand-seeded 30: `extract_keywords --apply` → 148 candidates → dropped
9 noise fragments → `seed_concepts --promote-all` → **139 keyword-grounded concepts** (each with one seed
alias = the surface form).

**Key property of the extracted vocabulary:** 146/148 candidates have **document-frequency = 1** (only 2 have
df=2) — TF-IDF top-N-per-doc selects each paper's *most distinctive* terms, which by construction appear in one
document. The shared cross-cutting IR concepts (BM25, dense retrieval, cross-encoder) are **absent from all 148**
— they are never any single paper's top-15, so TF-IDF drops them.

Keyword-grounded vocabulary (n=139, C=9591):

| min_cooccurrence | edges | density | communities | isolated |
|---|---|---|---|---|
| 1 | 1717 | 18% | 13 | 7 |
| 2 (applied) | 1250 | 13% | 17 | 10 |
| 3 | 1012 | 10.5% | 19 | 12 |

**Observation:** the graph is a **federation of per-paper cliques** — each paper's ~15 df=1 terms co-occur
within that paper and form a dense internal clique. The k=2 communities map cleanly to documents: c0 "passage"
(DPR), c1 "ance" (HyDE), c2 "late" (ColBERT), c3 "rag" (RAG), c4 "sts" (SBERT), c5 "text embedding" (BGE),
c6 "assistant" (LLM-judge) + 10 singleton fragments. Provenance now *partially* discriminates (citation 678/1250
= 54%, similarity 989/1250 = 79%, vs ~100% in run (a)) because within-paper clique edges span no cross-doc pair.
So the keyword-grounded graph clusters **by document, not by cross-cutting concept** — it is effectively a
per-document keyword clustering, not a cross-document concept graph.

## Re-run (c), 2026-07-01 — corpus mid-DF band vocabulary (tests the run-(b) hypothesis)

Run (b) predicted the usable vocabulary is the shared *mid-document-frequency* band. Added a general
`corpus_band` mode to the extractor (keep terms with `min_df ≤ df ≤ floor(max_df_frac·N)`, ranked by
`df·(1+ln tf)`) and ran it **once with general defaults (`min_df=2`, `max_df_frac=0.7`, `top_k=60`) — no
per-corpus tuning.** 60 terms selected, 405 doc-links (avg df≈6.8).

**The hypothesis FAILS on this corpus.** The df=6–7 band is dominated by **generic academic vocabulary that
every IR paper uses**, not domain concepts: `consider, create, including, introduce, multiple, requires,
respectively, report, release, strong, benchmark, document, average, volume, web, format` … + an author name
(`sebastian`). Only a minority are real concepts (bm25, dense, passage, ranking, encoder, transformer, marco,
trec). These generic terms co-occur everywhere → the **most saturated graph of all**: n=60, **1466 edges /
83% density @K=2**, provenance 100%/100%/100%, 3 communities, 0 isolated.

Root cause: on a small same-domain corpus, **domain concepts and common academic boilerplate occupy the same
document-frequency range**, so *no* DF band separates them. A larger stopword list would help a little but
tuning it against this output is corpus-overfitting (out of scope, per the run brief).

## Synthesis — four vocabulary regimes, none usable on this corpus

| vocabulary | n | density @K=2 | communities | failure mode |
|---|---|---|---|---|
| manual, broad hubs | 30 | 46% | 4 | over-connected (hubs link everything) |
| manual, hubs dropped | 21 | 27% | 4 | most balanced, but hand-curated |
| keyword TF-IDF (df≈1) | 139 | 13% | 17 | per-paper cliques (clusters by document) |
| keyword corpus-band (df 6–7) | 60 | **83%** | 3 | generic academic vocab saturates worse than hubs |

**Definitive verdict: the gate does NOT pass, and the blocker is the CORPUS, not the vocabulary method or the
code.** Four different vocabulary strategies span 13%→83% density and every one fails, because 10 same-domain
papers cannot support a cross-document concept graph: doc-similarity is fully saturated (provenance can't
discriminate), co-occurrence is unstable at N=10, and — proven here — statistical keyword selection cannot
separate "domain concept" from "common academic word" without a distinctive DF signal that only a larger,
**multi-domain** corpus provides.

**Design guidance (RG-001 deliverable):**
1. **The corpus is the gate.** Re-run all of this on the larger multi-domain private corpus (`cases.yaml`)
   before any threshold is locked or the graph is marked usable. On a proper corpus, `corpus_band` is likely
   the right default (mid-DF terms there *are* the shared domain concepts); it cannot be validated here.
2. **Vocabulary needs a semantic gate, not just statistics** — an embedding/LLM filter or a curated
   domain-stopword list to reject academic boilerplate. Deferred (and NOT tuned against this corpus).
3. Word-boundary presence matching (RG-009) still pending.
4. Gap-detection (ADR-004) stays blocked on a usable graph.

Applied on-disk state after this run: 60 corpus-band concepts + skeleton at K=2 (`data/skeleton/`, gitignored).
Reset: `DELETE FROM concept_aliases; DELETE FROM concepts;`.

## Follow-up filed

- **Vocabulary seam gap → RESOLVED (KI-13):** the keyword extractor (`src/doc_assistant/keywords.py`,
  `scripts/extract_keywords.py`) produces `Keyword` candidates for the `--promote` seam in two modes —
  `per_doc` (TF-IDF distinctive) and `corpus_band` (shared mid-DF). Both were exercised here; neither yields a
  usable graph *on this corpus* (the corpus is the blocker), but the machinery + config knobs are in place for
  the multi-domain re-run.
