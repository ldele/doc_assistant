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

## Follow-up filed (not part of this run)

- **Vocabulary seam gap:** `scripts/seed_concepts.py` depends on `Keyword` rows that no ingest/enrichment step
  produces — the promote-from-Keyword flow is dead on real data. Either wire a keyword extractor or add a
  direct concept-seeding CLI. (Logged in `.claude/KNOWN_ISSUES.md`.)
