# RG-001 multi-domain re-check — concept skeleton on a SEPARATE 24-paper corpus

**Separate from the public-corpus baseline** (`rg001_concept_skeleton_2026-07-01.md`). Different corpus,
different data home (`DOC_DATA_DIR=…/data_multidomain`, gitignored) — the public `data/` corpus is untouched.
**Run:** 2026-07-01, RTX box, CPU torch, `uv run --no-sync`. **Cost:** $0 (deterministic + regex + local embed).

**Purpose:** the public-corpus RG-001 concluded "the corpus is the blocker" (10 same-domain papers → saturated
doc-similarity). This tests that on a genuinely **multi-domain** corpus, re-running the concept skeleton on the
**current parameters** (corpus-band keywords, `min_df=2`/`max_df_frac=0.7`/`top_k=60`, skeleton K=2). Deliberately
**not tuned** to this corpus.

## The corpus (open-access arXiv; Sci-Hub deliberately NOT used)

24 papers, 4 each from 6 deliberately distinct domains (so document-similarity is not same-domain saturated):
`cs.LG` (machine learning) · `astro-ph.CO` (cosmology) · `cond-mat.stat-mech` (statistical mechanics) ·
`q-bio.NC` (neuroscience) · `econ.EM` (econometrics) · `math.PR` (probability). Fetched via the arXiv API
(most-recent per category); ids in `data_multidomain/manifest.json`. Ingested (24/24, 0 errors) → doc-vectors →
metadata → citations → corpus-band keywords → promote — all free.

## Findings

**1. Multi-domain DID de-saturate document-similarity (the corpus hypothesis's valid part).**
`doc_similarities` = 120 undirected / 276 possible = **43% of doc pairs**, vs the public corpus's **100%**
(45/45). So a multi-domain corpus genuinely breaks the mean-pooled doc-vector saturation that made
similarity-provenance meaningless before. (43% is still moderately high — academic PDFs share a lot of generic
structure — but it is no longer complete.)

**2. Citations: 714 parsed / 0 internal library matches** — the six domains don't cite each other, so
citation-provenance is genuinely 0 here (a *different* reason than the public corpus's metadata gap).

**3. The vocabulary is still unusable — and this exposes a corpus-independent SCORING FLAW.** The 60
corpus-band terms are **all df 13–16** (43 of 60 at the max df=16); every one is generic academic vocabulary
(`dynamics, effects, linear, finite, scaling, statistics, measure, parameters, limit, local, behavior`) — plus
extraction noise (finding 4). The `corpus_band` score `df·(1+ln tf)` maximises document-frequency, so it always
grabs the **most-shared = most-generic** terms. On a multi-domain corpus this is fatal in a new way: **domain
concepts are LOW-df** (a cosmology term appears only in the 4 cosmology papers), so they sit far below the top-60
and are never selected. df histogram of the 60: `{16: 43, 15: 9, 14: 6, 13: 2}` — nothing below 13.

  → **Generalizable conclusion (now proven on TWO corpora): document-frequency statistics alone cannot identify
  domain concepts.** Same-domain → concepts and academic boilerplate share a DF range; multi-domain → domain
  concepts are low-DF (indistinguishable from paper-specific singletons). No DF band isolates them either way.
  A **semantic gate** (embedding/LLM filter or a curated ontology) is required — this confirms and strengthens
  the public-corpus guidance.

**4. PDF extraction noise pollutes the corpus (KI-14).** PyMuPDF4LLM emits `**==> picture [W x H] intentionally
omitted <==**` placeholders for images it doesn't render inline — **1027 occurrences** across the cache, heaviest
in figure-dense physics/math papers (statmech 214, cosmo 182, econ 173). The keyword tokenizer then surfaces
`intentionally omitted`, `x 12`, `br 1`, `0 br` as candidate terms → 11 of the 13 skeleton "communities" are
these noise isolates. This pollutes the RAG chunk store too, not only keywords. Logged as `.claude/KNOWN_ISSUES.md`
KI-14. (Not filtered here — that would be a code change beyond "re-check on current params".)

**5. The concept graph is still not usable.** n=60, **1069 edges / 60% density @K2** (K3 58%, K4 55%),
similarity-provenance 100%, 13 communities = 2 generic blobs (c0 "limit" 27, c1 "local" 22) + 11 extraction-noise
singletons. 0 isolated among the real terms, but the two blobs carry no domain meaning.

## Public vs multi-domain (same method + parameters)

| metric | public (10, same-domain) | multi-domain (24, 6 domains) |
|---|---|---|
| doc-similarity saturation | 100% of pairs | **43%** of pairs (de-saturated) |
| internal citations | 7 | 0 (disconnected domains) |
| corpus-band vocab | IR-generic + academic | pure academic + extraction noise |
| skeleton density @K2 | 83% | 60% |
| structure | 3 blobs | 2 generic blobs + 11 noise isolates |
| usable? | no | **no** |

## Verdict — the blocker was NEVER only the corpus

Multi-domain fixed the *doc-similarity* blocker (43% vs 100%), but the graph is still unusable. The dominant
remaining blockers are **corpus-independent and method-level**:
1. **Vocabulary selection by DF cannot identify domain concepts** (proven on both corpora) → needs a semantic
   gate (embeddings / LLM / curated ontology), not a threshold tweak.
2. **Extraction placeholder noise** (KI-14) → strip `==> … intentionally omitted <==` markers before chunking +
   keywording.
3. (Still true) similarity-provenance stays weakly discriminating even at 43% saturation because the vocab is
   generic (generic terms span most doc pairs).

**RG-001 stays open.** The corpus is a *necessary* fix (multi-domain confirmed to help) but not *sufficient*; the
vocabulary-identification method is the deeper gate. **No parameter tuning was done on this corpus** (per the run
brief). On-disk: `data_multidomain/` (24 papers + DB + 60 corpus-band concepts + skeleton @K2), gitignored.
